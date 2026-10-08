#!/usr/bin/env python3
"""Step-up timing sweep for the Climb policies (``TanchoV3-Climb-Play-v0``).

One 8 m tile of inverted-pyramid stairs (every edge ahead is a step up, height
``--step-height``), one env per (approach speed x press timing), all in one
process (envs do not collide).  The simulated operator presses LT+RT when the
tire will reach the edge within the env's time-to-contact (``lookahead``); a
lookahead of 0.05 s presses almost at the edge, 0.35 s well before it.

Per env: steps climbed (axle height at the end / step height, median of the last
0.5 s alive), time of the first fall.  Summary: success map speed x timing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Climb-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--guidance", type=float, default=1.0)
parser.add_argument("--step-height", type=float, default=0.03)
parser.add_argument("--duration", type=float, default=6.0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--only", type=str, default=None, help="'vx,lookahead': a single env (for --video)")
parser.add_argument("--video", type=Path, default=None, help="record an mp4 of env 0 into this folder")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401
from tancho_v3_lab.tasks.staged.terrain import CLIMB_GENERATOR, make_terrain, play_generator  # noqa: E402

SPEEDS = [0.3, 0.4, 0.5, 0.6]
LOOKAHEAD = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]


@torch.inference_mode()
def main() -> int:
    grid = list(itertools.product(SPEEDS, LOOKAHEAD))
    if args.only:
        v, la = (float(x) for x in args.only.split(","))
        grid = [(v, la)]
    n = len(grid)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=n, use_fabric=True)
    cfg.episode_length_s = args.duration + 5.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    r = cfg.commands.base_velocity.ranges
    r.lin_vel_x = r.lin_vel_y = r.ang_vel_z = (0.0, 0.0)
    cfg.commands.climb.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.climb.auto_prob = 1.0
    cfg.commands.climb.random_press_prob = 0.0
    if hasattr(cfg.commands.climb, "burst_prob"):
        cfg.commands.climb.burst_prob = 0.0
    gen = play_generator(CLIMB_GENERATOR, "step_up", 1.0, size=8.0)
    gen.sub_terrains["step_up"].step_height_range = (args.step_height, args.step_height)
    cfg.scene.terrain = make_terrain(gen, max_init_level=None)
    cfg.actions.leg_pos.guidance_scale = args.guidance
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(args.video), step_trigger=lambda step: step == 0,
            video_length=round(args.duration / (cfg.sim.dt * cfg.decimation)) - 2,
            name_prefix=args.output.stem, disable_logger=True,
        )
    core = env.unwrapped
    dev = core.device
    robot = core.scene["robot"]
    vel = core.command_manager.get_term("base_velocity")
    climb = core.command_manager.get_term("climb")
    policy = torch.jit.load(str(args.checkpoint.resolve()), map_location=dev).eval()
    wheel_ids, _ = robot.find_bodies(["wheel_L", "wheel_R"], preserve_order=True)
    dt = core.step_dt

    cmd = torch.tensor([[v, 0.0, 0.0] for v, _ in grid], device=dev)
    look = torch.tensor([la for _, la in grid], device=dev)
    obs, _ = env.reset()
    axle0 = robot.data.body_pos_w[:, wheel_ids, 2].mean(dim=1)
    fell_at = torch.full((n,), -1.0, device=dev)
    hist = []
    presses = torch.zeros(n, device=dev)
    prev_trig = torch.zeros(n, device=dev)
    steps = round(args.duration / dt)
    for k in range(steps):
        vel.vel_command_b[:] = cmd
        climb.lookahead[:] = look
        climb.attentive[:] = True
        obs, _, term, _, _ = env.step(policy(obs["policy"]))
        alive = fell_at < 0
        dz = robot.data.body_pos_w[:, wheel_ids, 2].mean(dim=1) - axle0
        hist.append(torch.where(alive, dz, torch.full_like(dz, float("nan"))))
        trig = climb.trigger.max(dim=1).values
        presses += ((trig > 0) & (prev_trig <= 0) & alive).float()
        prev_trig = trig
        fell_at = torch.where(term.bool() & alive, torch.full_like(fell_at, (k + 1) * dt), fell_at)

    h = torch.stack(hist, dim=1)  # (n, T)
    rows = []
    for i, (v, la) in enumerate(grid):
        valid = h[i][~torch.isnan(h[i])]
        tail = valid[-25:] if len(valid) else torch.zeros(1, device=dev)
        final = float(tail.median())
        rows.append({
            "vx": v, "lookahead_s": la,
            "steps_climbed": int(math.floor(final / args.step_height + 0.5)),
            "max_axle_dz_m": round(float(valid.max()) if len(valid) else 0.0, 4),
            "presses": int(presses[i]),
            "fell_at_s": round(float(fell_at[i]), 2) if fell_at[i] >= 0 else None,
        })
    summary = {
        "checkpoint": str(args.checkpoint.resolve()), "guidance": args.guidance, "step_height_m": args.step_height,
        "duration_s": args.duration,
        "any_step_climbed": sum(r["steps_climbed"] >= 1 for r in rows),
        "climbed_without_fall": sum(r["steps_climbed"] >= 1 and r["fell_at_s"] is None for r in rows),
        "fall_rate": round(sum(r["fell_at_s"] is not None for r in rows) / n, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print("vx \\ lookahead " + " ".join(f"{la:5.2f}" for la in LOOKAHEAD))
    for v in SPEEDS:
        cells = []
        for la in LOOKAHEAD:
            r = next((x for x in rows if x["vx"] == v and x["lookahead_s"] == la), None)
            if r is None:
                continue
            cells.append(f"{r['steps_climbed']}{'F' if r['fell_at_s'] else ' '}".rjust(5))
        print(f"{v:4.1f}           " + " ".join(cells))
    return 0


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
