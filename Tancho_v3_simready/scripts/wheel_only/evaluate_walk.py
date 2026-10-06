#!/usr/bin/env python3
"""Command-tracking check for ``TanchoV3-Walk-Flat-Play-v0``.

Nominal physics, no noise, no randomization, no pushes.  The command is held
on a fixed schedule (forward / backward speed, yaw rate, combined, stop) instead
of being resampled.  Each segment's first ``--settle`` seconds are excluded from
the tracking error, so it measures steady-state tracking, not the step response.

Pass criteria:
  * no termination (tilt > 15 deg or body contact)
  * every segment: mean |vx - cmd_vx| < 0.10 m/s and mean |wz - cmd_wz| < 0.20 rad/s
  * final stop segment: chassis speed < 0.05 m/s at the end
  * left/right leg mismatch < 0.12 rad at the end (same as evaluate_stand.py)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Walk-Flat-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True, help="exported TorchScript policy.pt")
parser.add_argument("--settle", type=float, default=1.5, help="seconds excluded at the start of each segment")
parser.add_argument("--output-dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401
from tancho_v3_lab.tasks.staged.scene import LEG_JOINTS  # noqa: E402

# (name, duration s, vx m/s, wz rad/s).  Within the training ranges vx +/-0.6, wz +/-1.0.
SCHEDULE = [
    ("stand", 3.0, 0.0, 0.0),
    ("fwd_0.3", 5.0, 0.3, 0.0),
    ("fwd_0.6", 5.0, 0.6, 0.0),
    ("back_0.3", 5.0, -0.3, 0.0),
    ("yaw_+1.0", 5.0, 0.0, 1.0),
    ("yaw_-0.5", 5.0, 0.0, -0.5),
    ("arc_0.4_+0.5", 5.0, 0.4, 0.5),
    ("stop", 4.0, 0.0, 0.0),
]
VX_TOL = 0.10
WZ_TOL = 0.20
STOP_SPEED = 0.05
MIRROR_RAD = 0.12


def pitch_deg(quat_wxyz: torch.Tensor) -> float:
    w, x, y, z = quat_wxyz.tolist()
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))


def main() -> int:
    total = sum(seg[1] for seg in SCHEDULE)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.episode_length_s = total + 1.0
    # Hold the command: no resampling, no randomly "standing" env.
    cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    # The reset still samples once; make that sample zero so the first observation
    # does not carry a random command the schedule never asked for.
    ranges = cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = ranges.lin_vel_y = ranges.ang_vel_z = (0.0, 0.0)
    env = gym.make(args.task, cfg=cfg)
    core = env.unwrapped
    robot = core.scene["robot"]
    command = core.command_manager.get_term("base_velocity")
    policy = torch.jit.load(str(args.checkpoint.resolve()), map_location=core.device).eval()
    leg_ids, _ = robot.find_joints(LEG_JOINTS, preserve_order=True)

    obs, _ = env.reset()
    rows = []
    terminated_at = None
    t = 0.0
    for name, duration, vx, wz in SCHEDULE:
        for _ in range(round(duration / core.step_dt)):
            command.vel_command_b[0] = torch.tensor([vx, 0.0, wz], device=core.device)
            with torch.inference_mode():
                obs, _, term, _, _ = env.step(policy(obs["policy"]))
            t += core.step_dt
            q = robot.data.joint_pos[0, leg_ids].tolist()
            rows.append({
                "time_s": round(t, 4),
                "segment": name,
                "cmd_vx": vx,
                "cmd_wz": wz,
                "vx_m_s": float(robot.data.root_lin_vel_b[0, 0]),
                "vy_m_s": float(robot.data.root_lin_vel_b[0, 1]),
                "wz_rad_s": float(robot.data.root_ang_vel_b[0, 2]),
                "pitch_deg": pitch_deg(robot.data.root_quat_w[0]),
                "base_z_m": float(robot.data.root_pos_w[0, 2]),
                **{f"{n}_rad": v for n, v in zip(LEG_JOINTS, q)},
            })
            if bool(term[0]):
                terminated_at = t
                break
        if terminated_at is not None:
            break

    segments = {}
    for name, duration, vx, wz in SCHEDULE:
        seg = [r for r in rows if r["segment"] == name]
        if not seg:
            continue
        start = seg[0]["time_s"] - core.step_dt
        steady = [r for r in seg if r["time_s"] - start > args.settle] or seg
        segments[name] = {
            "cmd_vx": vx,
            "cmd_wz": wz,
            "mean_vx": sum(r["vx_m_s"] for r in steady) / len(steady),
            "mean_wz": sum(r["wz_rad_s"] for r in steady) / len(steady),
            "mean_abs_vx_err": sum(abs(r["vx_m_s"] - vx) for r in steady) / len(steady),
            "mean_abs_wz_err": sum(abs(r["wz_rad_s"] - wz) for r in steady) / len(steady),
            "max_abs_pitch_deg": max(abs(r["pitch_deg"]) for r in seg),
        }
        segments[name]["pass"] = (segments[name]["mean_abs_vx_err"] < VX_TOL
                                  and segments[name]["mean_abs_wz_err"] < WZ_TOL)

    last = rows[-1]
    mirror = max(abs(last["joint_thigh_L_rad"] - last["joint_thigh_R_rad"]),
                 abs(last["joint_calf_L_rad"] - last["joint_calf_R_rad"]))
    end_speed = math.hypot(last["vx_m_s"], last["vy_m_s"])
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "task": args.task,
        "terminated_at_s": terminated_at,
        "segments": segments,
        "end_speed_m_s": end_speed,
        "end_mirror_rad": mirror,
        "pass_no_failure": terminated_at is None,
        "pass_tracking": len(segments) == len(SCHEDULE) and all(s["pass"] for s in segments.values()),
        "pass_stop": terminated_at is None and end_speed < STOP_SPEED,
        "pass_leg_pose": mirror < MIRROR_RAD,
    }
    summary["pass_all"] = all(summary[k] for k in ("pass_no_failure", "pass_tracking", "pass_stop", "pass_leg_pose"))

    out = args.output_dir or args.checkpoint.resolve().parent.parent / "walk_eval"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "walk_timeseries.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "walk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass_all"] else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    # SimulationApp.close() swallows the output and can hang; exit hard (same as evaluate_standstill.py).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
