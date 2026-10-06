#!/usr/bin/env python3
"""Standstill + push-recovery check for the 6-DOF stages (``TanchoV3-Stand-Flat-Play-v0``).

Nominal physics, no noise, no randomization.  The run lasts ``--duration``
seconds with zero command; at ``--push-time`` the chassis gets a horizontal
velocity kick of ``--push-vel`` m/s (0 disables it).

Pass criteria:
  * no termination (tilt > 15 deg or body contact)
  * before the push: |pitch - pitch(t=0)| <= 2 deg and planar drift < 5 cm
  * base height never more than 3 cm below its start (no crouch / collapse)
  * every leg joint within 0.15 rad of the nominal pose at the end
  * left/right leg mismatch (thigh and calf) < 0.05 rad at the end
  * after the push: pitch back within 2 deg and chassis speed < 0.05 m/s
    by the end of the run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Stand-Flat-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True, help="exported TorchScript policy.pt")
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--push-time", type=float, default=10.0)
parser.add_argument("--push-vel", type=float, default=0.5)
parser.add_argument("--output-dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401
from tancho_v3_lab.tasks.staged.scene import LEG_JOINTS, NOMINAL_JOINT_POS, WHEEL_JOINTS  # noqa: E402

PITCH_BAND_DEG = 2.0
DRIFT_LIMIT_M = 0.05
HEIGHT_DROP_M = 0.03
LEG_DEV_RAD = 0.15
MIRROR_RAD = 0.05
SETTLED_SPEED_M_S = 0.05


def pitch_deg(quat_wxyz: torch.Tensor) -> float:
    w, x, y, z = quat_wxyz.tolist()
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))


def main() -> int:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.episode_length_s = max(cfg.episode_length_s, args.duration + 1.0)
    env = gym.make(args.task, cfg=cfg)
    core = env.unwrapped
    robot = core.scene["robot"]
    policy = torch.jit.load(str(args.checkpoint.resolve()), map_location=core.device).eval()
    leg_ids, _ = robot.find_joints(LEG_JOINTS, preserve_order=True)
    wheel_ids, _ = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    nominal = torch.tensor([NOMINAL_JOINT_POS[n] for n in LEG_JOINTS], device=core.device)

    obs, _ = env.reset()
    p0 = pitch_deg(robot.data.root_quat_w[0])
    xy0 = robot.data.root_pos_w[0, :2].clone()
    z0 = float(robot.data.root_pos_w[0, 2])
    rows = []
    terminated_at = None
    steps = math.ceil(args.duration / core.step_dt)
    push_step = round(args.push_time / core.step_dt) if args.push_vel > 0 else None
    for k in range(steps + 1):
        q = robot.data.joint_pos[0, leg_ids]
        row = {
            "time_s": round(k * core.step_dt, 4),
            "pitch_deg": pitch_deg(robot.data.root_quat_w[0]),
            "displacement_m": float(torch.linalg.norm(robot.data.root_pos_w[0, :2] - xy0)),
            "base_z_m": float(robot.data.root_pos_w[0, 2]),
            "speed_m_s": float(torch.linalg.norm(robot.data.root_lin_vel_w[0, :2])),
        }
        for name, val in zip(LEG_JOINTS, q.tolist()):
            row[f"{name}_rad"] = val
        for name, i in zip(WHEEL_JOINTS, wheel_ids):
            row[f"{name}_qd"] = float(robot.data.joint_vel[0, i])
        rows.append(row)
        if k == steps:
            break
        if push_step is not None and k == push_step:
            vel = robot.data.root_vel_w.clone()
            vel[0, 0] += args.push_vel
            robot.write_root_velocity_to_sim(vel)
        with torch.inference_mode():
            obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
        if bool(term[0]) and terminated_at is None:
            terminated_at = (k + 1) * core.step_dt
            break

    pre = [r for r in rows if push_step is None or r["time_s"] < args.push_time]
    last = rows[-1]
    q_end = torch.tensor([last[f"{n}_rad"] for n in LEG_JOINTS])
    leg_dev = (q_end - nominal.cpu()).abs().max().item()
    mirror = max(abs(q_end[0] - q_end[2]).item(), abs(q_end[1] - q_end[3]).item())
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "task": args.task,
        "push_vel_m_s": args.push_vel,
        "terminated_at_s": terminated_at,
        "pre_push_max_pitch_dev_deg": max(abs(r["pitch_deg"] - p0) for r in pre),
        "pre_push_max_drift_m": max(r["displacement_m"] for r in pre),
        "max_height_drop_m": z0 - min(r["base_z_m"] for r in rows),
        "end_leg_dev_rad": leg_dev,
        "end_mirror_rad": mirror,
        "end_pitch_dev_deg": abs(last["pitch_deg"] - p0),
        "end_speed_m_s": last["speed_m_s"],
        "max_displacement_m": max(r["displacement_m"] for r in rows),
    }
    summary["pass_no_failure"] = terminated_at is None
    summary["pass_pre_push"] = (summary["pre_push_max_pitch_dev_deg"] <= PITCH_BAND_DEG
                                and summary["pre_push_max_drift_m"] < DRIFT_LIMIT_M)
    summary["pass_height"] = summary["max_height_drop_m"] < HEIGHT_DROP_M
    summary["pass_leg_pose"] = leg_dev < LEG_DEV_RAD and mirror < MIRROR_RAD
    summary["pass_recovered"] = summary["end_pitch_dev_deg"] <= PITCH_BAND_DEG and last["speed_m_s"] < SETTLED_SPEED_M_S
    summary["pass_all"] = all(summary[k] for k in
                              ("pass_no_failure", "pass_pre_push", "pass_height", "pass_leg_pose", "pass_recovered"))

    out = args.output_dir or args.checkpoint.resolve().parent.parent / f"stand_eval_push{args.push_vel:g}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "stand_timeseries.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "stand_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    env.close()
    return 0 if summary["pass_all"] else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
