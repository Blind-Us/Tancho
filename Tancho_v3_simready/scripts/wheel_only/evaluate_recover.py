#!/usr/bin/env python3
"""Fall-recovery check (``TanchoV3-Recover-Play-v0``): start pitched by a grid of
angles (forward and backward, 0-90 deg), dropped 3 cm, standing-still command.
Per start angle: upright (tilt < 15 deg and root > 85% standing height) at
2 / 5 / 10 s, and whether it stayed up for the last 2 s."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Recover-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--zero-action", action="store_true", help="no policy: legs nominal, wheels 0 (physics baseline)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401
from tancho_v3_lab.tasks.staged.recover import is_up  # noqa: E402

ANGLES = [s * a for a in range(0, 95, 5) for s in (1, -1) if not (a == 0 and s == -1)]


@torch.inference_mode()
def main() -> int:
    n = len(ANGLES)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=n, use_fabric=True)
    cfg.episode_length_s = 12.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    env = gym.make(args.task, cfg=cfg)
    core = env.unwrapped
    dev = core.device
    robot = core.scene["robot"]
    obs, _ = env.reset()
    pitch = torch.tensor([math.radians(a) for a in ANGLES], device=dev)
    state = robot.data.root_state_w.clone()
    state[:, 3:7] = quat_mul(state[:, 3:7], quat_from_euler_xyz(torch.zeros_like(pitch), pitch, torch.zeros_like(pitch)))
    state[:, 2] += 0.03
    state[:, 7:] = 0.0
    robot.write_root_state_to_sim(state)
    if not args.zero_action:
        policy = torch.jit.load(str(args.checkpoint.resolve()), map_location=dev).eval()
    obs = core.observation_manager.compute()
    dt = core.step_dt
    up_at = {}
    up_tail = torch.ones(n, dtype=torch.bool, device=dev)
    first_up = torch.full((n,), -1.0, device=dev)
    steps = round(10.0 / dt)
    for k in range(1, steps + 1):
        act = torch.zeros(n, 6, device=dev) if args.zero_action else policy(obs["policy"])
        obs, _, _, _, _ = env.step(act)
        t = k * dt
        up = is_up(core) > 0
        first_up = torch.where((first_up < 0) & up & (t > 0.3), torch.full_like(first_up, t), first_up)
        for mark in (2.0, 5.0, 10.0):
            if k == round(mark / dt):
                up_at[mark] = up.clone()
        if t >= 8.0:
            up_tail &= up
    rows = [
        {
            "pitch_deg": a,
            "up_2s": bool(up_at[2.0][i]),
            "up_5s": bool(up_at[5.0][i]),
            "up_10s": bool(up_at[10.0][i]),
            "stayed_up_8_10s": bool(up_tail[i]),
            "first_up_s": round(float(first_up[i]), 2) if first_up[i] >= 0 else None,
        }
        for i, a in enumerate(ANGLES)
    ]
    fwd = [r["pitch_deg"] for r in rows if r["pitch_deg"] >= 0 and r["stayed_up_8_10s"]]
    bwd = [-r["pitch_deg"] for r in rows if r["pitch_deg"] <= 0 and r["stayed_up_8_10s"]]

    def contiguous(ok):
        m = 0
        for a in range(0, 95, 5):
            if a in ok:
                m = a
            else:
                break
        return m

    summary = {
        "checkpoint": str(args.checkpoint.resolve()) if not args.zero_action else "zero-action",
        "max_recovered_forward_deg": contiguous(fwd),
        "max_recovered_backward_deg": contiguous(bwd),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    for r in rows:
        print(r)
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
