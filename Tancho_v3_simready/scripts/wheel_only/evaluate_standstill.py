#!/usr/bin/env python3
"""20 s zero-command standstill check for ``TanchoV3-WheelOnly-Flat-Play-v0``.

Pass criteria (all on nominal physics, no pushes, no noise, no randomization):
  * |pitch - pitch(t=0)| <= 1 deg for the whole run
  * planar displacement of the chassis from its start < 5 cm (max over the run)
  * no termination

Also re-derives the frozen leg angles from the *simulated* USD collision prims
(merged under ``base_link_root``) and checks them against thigh=-0.50 / calf=+0.87.
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
parser.add_argument("--task", default="TanchoV3-WheelOnly-Flat-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True, help="exported TorchScript policy.pt")
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--output-dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401

PITCH_BAND_DEG = 1.0
DRIFT_LIMIT_M = 0.05
LEG_TOL_RAD = 0.01
# Leg shapes and the link-frame pitch they carry in the merged body:
# thigh = q_thigh, calf = q_thigh + q_calf (all hinge axes are parallel to y).
LEG_SHAPES = {"thigh_L_collision_box": -0.50, "thigh_R_collision_box": -0.50,
              "calf_L_collision_mesh": -0.50 + 0.87, "calf_R_collision_mesh": -0.50 + 0.87}


def pitch_deg(quat_wxyz: torch.Tensor) -> float:
    w, x, y, z = quat_wxyz.tolist()
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))


def leg_angles_from_usd(stage, root_path: str) -> dict[str, float]:
    """Pitch of each leg collision prim relative to the root body, from the simulated stage."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache()
    root = stage.GetPrimAtPath(root_path)
    t_root = np.array(cache.GetLocalToWorldTransform(root)).T
    found = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetName() in LEG_SHAPES and prim.IsA(UsdGeom.Xformable):
            t = np.linalg.inv(t_root) @ np.array(cache.GetLocalToWorldTransform(prim)).T
            r = t[:3, :3] / np.linalg.norm(t[:3, :3], axis=0)
            # Shape frames are Ry(p) @ Rx(pi/2); the x column gives p = atan2(-r20, r00).
            # The hinge axes make the joint sum q = -p (thigh: p=+0.50 <-> q=-0.50).
            found[prim.GetName()] = math.atan2(r[2, 0], r[0, 0])
    return found


def main() -> int:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.episode_length_s = max(cfg.episode_length_s, args.duration + 1.0)
    env = gym.make(args.task, cfg=cfg)
    core = env.unwrapped
    robot = core.scene["robot"]
    policy = torch.jit.load(str(args.checkpoint.resolve()), map_location=core.device).eval()

    leg = leg_angles_from_usd(core.scene.stage, "/World/envs/env_0/Robot/base_link_root")
    leg_err = {k: leg[k] - LEG_SHAPES[k] for k in leg}

    obs, _ = env.reset()
    p0 = pitch_deg(robot.data.root_quat_w[0])
    xy0 = robot.data.root_pos_w[0, :2].clone()
    rows = []
    terminated_at = None
    steps = math.ceil(args.duration / core.step_dt)
    for k in range(steps + 1):
        pitch = pitch_deg(robot.data.root_quat_w[0])
        disp = float(torch.linalg.norm(robot.data.root_pos_w[0, :2] - xy0))
        rows.append({
            "time_s": round(k * core.step_dt, 4),
            "pitch_deg": pitch,
            "pitch_dev_deg": pitch - p0,
            "displacement_m": disp,
            "base_x_m": float(robot.data.root_pos_w[0, 0] - xy0[0]),
            "wheel_L_qd_rad_s": float(robot.data.joint_vel[0, 0]),
            "wheel_R_qd_rad_s": float(robot.data.joint_vel[0, 1]),
            "wheel_L_tau_Nm": float(robot.data.applied_torque[0, 0]),
            "wheel_R_tau_Nm": float(robot.data.applied_torque[0, 1]),
        })
        if k == steps:
            break
        with torch.inference_mode():
            obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
        if bool(term[0]) and terminated_at is None:
            terminated_at = (k + 1) * core.step_dt
            break

    max_dev = max(abs(r["pitch_dev_deg"]) for r in rows)
    max_disp = max(r["displacement_m"] for r in rows)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "duration_s": args.duration,
        "initial_pitch_deg": p0,
        "final_pitch_deg": rows[-1]["pitch_deg"],
        "max_abs_pitch_dev_deg": max_dev,
        "final_displacement_m": rows[-1]["displacement_m"],
        "max_displacement_m": max_disp,
        "max_abs_wheel_torque_Nm": max(max(abs(r["wheel_L_tau_Nm"]), abs(r["wheel_R_tau_Nm"])) for r in rows[1:]),
        "terminated_at_s": terminated_at,
        "leg_angle_rad_from_usd": leg,
        "leg_angle_error_rad": leg_err,
        "pass_pitch": max_dev <= PITCH_BAND_DEG,
        "pass_drift": max_disp < DRIFT_LIMIT_M,
        "pass_no_failure": terminated_at is None,
        "pass_leg_pose": len(leg) == 4 and max(abs(e) for e in leg_err.values()) < LEG_TOL_RAD,
    }
    summary["pass_all"] = all(summary[k] for k in ("pass_pitch", "pass_drift", "pass_no_failure", "pass_leg_pose"))

    out = args.output_dir or args.checkpoint.resolve().parent.parent / "standstill_eval"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "standstill_timeseries.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "standstill_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"STANDSTILL_DIR={out}")
    return 0 if summary["pass_all"] else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    # SimulationApp.close() can spin forever after an exception; exit hard instead.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
