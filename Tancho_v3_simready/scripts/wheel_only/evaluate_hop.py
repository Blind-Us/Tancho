#!/usr/bin/env python3
"""Trigger check on flat ground for the Climb policies (``TanchoV3-ClimbHop-Play-v0``).

The robot drives at ``--vx``; the triggers are pressed on a fixed schedule
(LT, RT, LT+RT, each held 0.4 s, 3 s apart, repeated ``--repeats`` times) instead
of randomly.  Per press: peak tire clearance of each wheel above the flat ground
during the 0.6 s after the rising edge.  ``--guidance`` sets the reference-lift
injection (training used 1 in stage A; deployment is 0).

Pass criteria: no termination, and the pressed wheel(s) clear >= ``--min-clear``
on every press.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-ClimbHop-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--guidance", type=float, default=0.0)
parser.add_argument("--vx", type=float, default=0.3)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--min-clear", type=float, default=0.02)
parser.add_argument("--min-rise", type=float, default=0.01, help="LT+RT: root must rise this far above its pre-press height while both tires are off the ground (active jump)")
parser.add_argument("--schedule", default="LT,RT,LT+RT", help="comma-separated presses, repeated --repeats times")
parser.add_argument("--gap", type=float, default=3.0, help="s between rising edges (double jump: < 1)")
parser.add_argument("--press-s", type=float, default=0.4, help="s each trigger is held")
parser.add_argument("--yaw", type=float, default=0.0, help="yaw-rate command (rad/s)")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--terrain", choices=["flat", "rough"], default="flat", help="rough: one tile of the HOP rough terrain at --difficulty (1.0 = 2 cm peak-to-peak); clearance is then relative to the start ground")
parser.add_argument("--difficulty", type=float, default=1.0)
parser.add_argument("--video", type=Path, default=None, help="record an mp4 into this folder (needs --enable_cameras)")
parser.add_argument("--ref-style", choices=["v1", "v2", "v3", "v7"], default=None, help="reference lift: v1, or v2 (stance push, crouch/push/tuck hop)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401
from tancho_v3_lab.tasks.staged.scene import WHEEL_RADIUS_M  # noqa: E402

PRESS_S, GAP_S = args.press_s, args.gap
WINDOW_S = min(0.6, GAP_S)
MODES = {"LT": (1.0, 0.0), "RT": (0.0, 1.0), "LT+RT": (1.0, 1.0)}


@torch.inference_mode()
def main() -> int:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    schedule = [m for _ in range(args.repeats) for m in args.schedule.split(",")]
    cfg.episode_length_s = 2.0 + GAP_S * len(schedule) + 5.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    r = cfg.commands.base_velocity.ranges
    r.lin_vel_x = r.lin_vel_y = r.ang_vel_z = (0.0, 0.0)
    cfg.commands.climb.random_press_prob = 0.0
    cfg.commands.climb.auto_prob = 0.0
    cfg.actions.leg_pos.guidance_scale = args.guidance
    if args.ref_style is not None:
        cfg.actions.leg_pos.ref_style = args.ref_style
    if args.terrain == "rough":
        from tancho_v3_lab.tasks.staged.terrain import HOP_GENERATOR, make_terrain, play_generator

        cfg.scene.terrain = make_terrain(play_generator(HOP_GENERATOR, "rough", args.difficulty, size=16.0), max_init_level=None)
    else:
        from tancho_v3_lab.tasks.staged.terrain import HOP_GENERATOR, make_terrain, play_generator
        cfg.scene.terrain = make_terrain(play_generator(HOP_GENERATOR, "flat", 0.0, size=16.0), max_init_level=None)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        steps = round((2.0 + GAP_S * (len(schedule) - 1) + 3.0) / cfg.sim.dt / cfg.decimation) - 2
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(args.video), step_trigger=lambda step: step == 0, video_length=steps,
            name_prefix=args.output.stem, disable_logger=True,
        )
    core = env.unwrapped
    robot = core.scene["robot"]
    vel = core.command_manager.get_term("base_velocity")
    climb = core.command_manager.get_term("climb")
    policy = torch.jit.load(str(args.checkpoint.resolve()), map_location=core.device).eval()
    wheel_ids, _ = robot.find_bodies(["wheel_L", "wheel_R"], preserve_order=True)
    leg_ids, _ = robot.find_joints(["joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R"], preserve_order=True)
    leg_stats = [{"mode": mode, "peak_joint_excursion_rad": [0.0]*4,
                  "peak_joint_speed_rad_s": [0.0]*4, "peak_joint_torque_Nm": [0.0]*4,
                  "calf_return_time_s": None, "peak_abs_pitch_after_release_deg": 0.0}
                 for mode in schedule]
    return_dwell = [0.0 for _ in schedule]
    dt = core.step_dt

    obs, _ = env.reset()
    ground = float(robot.data.body_pos_w[0, wheel_ids, 2].mean()) - WHEEL_RADIUS_M
    root_before = [None for _ in schedule]
    presses, terminated_at, t = [], None, 0.0
    starts = [2.0 + GAP_S * i for i in range(len(schedule))]
    total = starts[-1] + 3.0
    while t < total:
        vel.vel_command_b[0] = torch.tensor([args.vx, 0.0, args.yaw], device=core.device)
        # Press through the command term's own random-press slot, so the trigger,
        # its phase and the observation follow exactly the training code path.
        k = round(t / dt)
        for i, s0 in enumerate(starts):
            if k == round(s0 / dt):
                climb.random_mode[0] = torch.tensor(MODES[schedule[i]], device=core.device)
                climb.random_timer[0] = PRESS_S
        obs, _, term, _, _ = env.step(policy(obs["policy"]))
        t += dt
        clear = (robot.data.body_pos_w[0, wheel_ids, 2] - WHEEL_RADIUS_M - ground).tolist()
        qerr = (robot.data.joint_pos[0, leg_ids] - robot.data.default_joint_pos[0, leg_ids]).tolist()
        qvel = robot.data.joint_vel[0, leg_ids].tolist()
        tau = robot.data.applied_torque[0, leg_ids].tolist()
        w, x, y, z = robot.data.root_quat_w[0].tolist()
        import math
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2*(w*y-z*x)))))
        for i, s0 in enumerate(starts):
            if s0 <= t < s0 + GAP_S:
                stats = leg_stats[i]
                if t < s0 + 0.6:
                    for key, values in [("peak_joint_excursion_rad", qerr), ("peak_joint_speed_rad_s", qvel), ("peak_joint_torque_Nm", tau)]:
                        stats[key] = [max(old, abs(v)) for old, v in zip(stats[key], values)]
                if t >= s0 + 0.4:
                    stats["peak_abs_pitch_after_release_deg"] = max(stats["peak_abs_pitch_after_release_deg"], abs(pitch))
                    selected = [j for j, wanted in zip((1, 3), MODES[schedule[i]]) if wanted]
                    return_dwell[i] = return_dwell[i] + dt if all(abs(qerr[j]) <= 0.1 for j in selected) else 0.0
                    if return_dwell[i] >= 0.1 and stats["calf_return_time_s"] is None:
                        stats["calf_return_time_s"] = round(t - s0 - 0.1, 3)
        root_z = float(robot.data.root_pos_w[0, 2])
        for i, s0 in enumerate(starts):
            if s0 - dt <= t < s0:
                root_before[i] = root_z
            if s0 <= t < s0 + WINDOW_S:
                while len(presses) <= i:
                    presses.append({"mode": schedule[len(presses)], "clear_L_m": 0.0, "clear_R_m": 0.0,
                                    "both_air_s": 0.0, "air_root_rise_m": 0.0, "peak_root_vz_m_s": 0.0})
                presses[i]["clear_L_m"] = max(presses[i]["clear_L_m"], clear[0])
                presses[i]["clear_R_m"] = max(presses[i]["clear_R_m"], clear[1])
                presses[i]["peak_root_vz_m_s"] = max(presses[i]["peak_root_vz_m_s"], float(robot.data.root_lin_vel_w[0, 2]))
                if min(clear) > 0.005:  # both tires off the ground
                    presses[i]["both_air_s"] = round(presses[i]["both_air_s"] + dt, 3)
                    if root_before[i] is not None:
                        presses[i]["air_root_rise_m"] = max(presses[i]["air_root_rise_m"], root_z - root_before[i])
        if bool(term[0]):
            terminated_at = round(t, 2)
            break

    ok = []
    for p in presses:
        want = MODES[p["mode"]]
        cl = [p["clear_L_m"], p["clear_R_m"]]
        p["pass"] = all(c >= args.min_clear for c, w in zip(cl, want) if w > 0)
        if p["mode"] == "LT+RT":
            p["pass_active_jump"] = p["air_root_rise_m"] >= args.min_rise
            p["pass"] = p["pass"] and p["pass_active_jump"]
        ok.append(p["pass"])
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "guidance": args.guidance,
        "vx": args.vx,
        "yaw": args.yaw,
        "gap_s": GAP_S,
        "press_s": PRESS_S,
        "terminated_at_s": terminated_at,
        "presses": presses,
        "joint_order": ["thigh_L", "calf_L", "thigh_R", "calf_R"],
        "joint_diagnostics": leg_stats,
        "calf_return_definition": "pressed calf within 0.1 rad of +0.87 for 0.1 s after reference ends",
        "pass_no_failure": terminated_at is None,
        "pass_lift": len(presses) == len(schedule) and all(ok),
    }
    summary["pass_all"] = summary["pass_no_failure"] and summary["pass_lift"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    env.close()  # Flush RecordVideo even when a fall ends the sequence early.
    return 0 if summary["pass_all"] else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
