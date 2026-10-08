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
parser.add_argument("--schedule", default="LT,RT,LT+RT", help="comma-separated presses, repeated --repeats times")
parser.add_argument("--gap", type=float, default=3.0, help="s between rising edges (double jump: < 1)")
parser.add_argument("--press-s", type=float, default=0.4, help="s each trigger is held")
parser.add_argument("--yaw", type=float, default=0.0, help="yaw-rate command (rad/s)")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--terrain", choices=["flat", "rough"], default="flat", help="rough: one tile of the HOP rough terrain at --difficulty (1.0 = 2 cm peak-to-peak); clearance is then relative to the start ground")
parser.add_argument("--difficulty", type=float, default=1.0)
parser.add_argument("--video", type=Path, default=None, help="record an mp4 into this folder (needs --enable_cameras)")
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
    if args.terrain == "rough":
        from tancho_v3_lab.tasks.staged.terrain import HOP_GENERATOR, make_terrain, play_generator

        cfg.scene.terrain = make_terrain(play_generator(HOP_GENERATOR, "rough", args.difficulty, size=16.0), max_init_level=None)
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
    dt = core.step_dt

    obs, _ = env.reset()
    ground = float(robot.data.body_pos_w[0, wheel_ids, 2].mean()) - WHEEL_RADIUS_M
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
        for i, s0 in enumerate(starts):
            if s0 <= t < s0 + WINDOW_S:
                while len(presses) <= i:
                    presses.append({"mode": schedule[len(presses)], "clear_L_m": 0.0, "clear_R_m": 0.0})
                presses[i]["clear_L_m"] = max(presses[i]["clear_L_m"], clear[0])
                presses[i]["clear_R_m"] = max(presses[i]["clear_R_m"], clear[1])
        if bool(term[0]):
            terminated_at = round(t, 2)
            break

    ok = []
    for p in presses:
        want = MODES[p["mode"]]
        cl = [p["clear_L_m"], p["clear_R_m"]]
        p["pass"] = all(c >= args.min_clear for c, w in zip(cl, want) if w > 0)
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
        "pass_no_failure": terminated_at is None,
        "pass_lift": len(presses) == len(schedule) and all(ok),
    }
    summary["pass_all"] = summary["pass_no_failure"] and summary["pass_lift"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
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
