#!/usr/bin/env python3
"""Step-crossing check for ``TanchoV3-Walk-Step-Play-v0``.

One 8 m tile of pyramid stairs at the hardest level (3 cm risers, 0.6 m
treads), nominal physics, no noise.  One direction per process (Isaac Sim
cannot rebuild the stage in-process), chosen with ``--direction``:

* ``up``: inverted pyramid, spawn in the 1.5 m pit, every edge ahead is a step up.
* ``down``: pyramid, spawn on the top platform, every edge ahead is a step down.

The command is held at ``--vx`` straight ahead for ``--duration`` seconds.  The
number of edges crossed is read from the change in wheel-axle height.

Pass criteria (per direction):
  * no termination (tilt > 15 deg or body contact)
  * at least ``--min-steps`` edges crossed
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
parser.add_argument("--task", default="TanchoV3-Walk-Step-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True, help="exported TorchScript policy.pt")
parser.add_argument("--direction", choices=("up", "down"), required=True)
parser.add_argument("--vx", type=float, default=0.4)
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--min-steps", type=int, default=3)
parser.add_argument("--step-height", type=float, default=0.03)
parser.add_argument("--guidance", type=float, default=None, help="Climb tasks: reference-lift guidance scale (deployment: 0)")
parser.add_argument("--output-dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401
from tancho_v3_lab.tasks.staged.terrain import STEP_GENERATOR, make_terrain, play_generator  # noqa: E402


def pitch_deg(quat_wxyz: torch.Tensor) -> float:
    w, x, y, z = quat_wxyz.tolist()
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))


def _climb_fields(core, robot) -> dict:
    """Trigger state and leg angles (Climb tasks only)."""
    if "climb" not in core.command_manager.active_terms:
        return {}
    term = core.command_manager.get_term("climb")
    q = (robot.data.joint_pos[0] - robot.data.default_joint_pos[0]).tolist()
    names = robot.joint_names
    return {
        "trig_L": float(term.trigger[0, 0]),
        "trig_R": float(term.trigger[0, 1]),
        **{f"d{n.replace('joint_', '')}_rad": round(v, 3) for n, v in zip(names, q) if "wheel" not in n},
    }


def run(direction: str) -> tuple[dict, list[dict]]:
    sub = "step_up" if direction == "up" else "step_down"
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.episode_length_s = args.duration + 2.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    ranges = cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = ranges.lin_vel_y = ranges.ang_vel_z = (0.0, 0.0)
    gen = play_generator(STEP_GENERATOR, sub, 1.0, size=8.0)
    gen.sub_terrains[sub].step_height_range = (args.step_height, args.step_height)
    cfg.scene.terrain = make_terrain(gen, max_init_level=None)
    if args.guidance is not None:
        cfg.actions.leg_pos.guidance_scale = args.guidance
    env = gym.make(args.task, cfg=cfg)
    core = env.unwrapped
    robot = core.scene["robot"]
    command = core.command_manager.get_term("base_velocity")
    policy = torch.jit.load(str(args.checkpoint.resolve()), map_location=core.device).eval()
    wheel_ids, _ = robot.find_bodies(["wheel_L", "wheel_R"], preserve_order=True)

    obs, _ = env.reset()
    axle_z0 = float(robot.data.body_pos_w[0, wheel_ids, 2].mean())
    rows, terminated_at, t = [], None, 0.0
    for _ in range(round(args.duration / core.step_dt)):
        command.vel_command_b[0] = torch.tensor([args.vx, 0.0, 0.0], device=core.device)
        with torch.inference_mode():
            obs, _, term, _, _ = env.step(policy(obs["policy"]))
        t += core.step_dt
        axle = robot.data.body_pos_w[0, wheel_ids, 2]
        rows.append({
            "time_s": round(t, 4),
            "x_m": float(robot.data.root_pos_w[0, 0] - core.scene.env_origins[0, 0]),
            "vx_m_s": float(robot.data.root_lin_vel_b[0, 0]),
            "pitch_deg": pitch_deg(robot.data.root_quat_w[0]),
            "axle_dz_L_m": float(axle[0]) - axle_z0,
            "axle_dz_R_m": float(axle[1]) - axle_z0,
            **_climb_fields(core, robot),
        })
        if bool(term[0]):
            terminated_at = t
            break

    dz = [0.5 * (r["axle_dz_L_m"] + r["axle_dz_R_m"]) for r in rows]
    # Count from where the robot ends (median of the last 0.5 s), not the peak: a hop
    # or a contact glitch can spike the axle height for a sample without a step gained.
    # After a termination the env has auto-reset, so the last sample is dropped.
    tail = sorted(dz[-26:-1] if terminated_at is not None else dz[-25:])
    final = tail[len(tail) // 2]
    extreme = final if direction == "up" else -final
    steps = int(math.floor(extreme / args.step_height + 0.5))
    summary = {
        "direction": direction,
        "terminated_at_s": terminated_at,
        "axle_height_change_m": extreme,
        "peak_axle_dz_m": max(dz),
        "steps_crossed": steps,
        "distance_m": rows[-1]["x_m"],
        "max_abs_pitch_deg": max(abs(r["pitch_deg"]) for r in rows),
        "pass": terminated_at is None and steps >= args.min_steps,
    }
    return summary, rows


def main() -> int:
    out = args.output_dir or args.checkpoint.resolve().parent.parent / "step_eval"
    out.mkdir(parents=True, exist_ok=True)
    summary, rows = run(args.direction)
    result = {"checkpoint": str(args.checkpoint.resolve()), "vx": args.vx, "step_height_m": args.step_height, **summary}
    with (out / f"step_{args.direction}_timeseries.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / f"step_{args.direction}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    # SimulationApp.close() can hang; exit hard (same as evaluate_walk.py).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
