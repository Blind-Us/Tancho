#!/usr/bin/env python3
"""Trigger stress sweep for the Climb policies, one scenario per env, all in one run.

``evaluate_hop.py`` checks one schedule per process.  This one builds a grid of
press patterns (single presses, double / triple hops, L-R shuffles, a hop while
one trigger is held) x rising-edge spacing x press length x drive command and
runs it in parallel on the flat ``TanchoV3-ClimbHop-Play-v0`` tile (envs do not
collide with each other).  Presses go through the command term's own random-press
slot, so trigger edges and phases follow the training code path.

Per scenario: did the robot fall (first termination), and the smallest peak
clearance of a pressed wheel over all presses (0.1-0.6 s after each reference
lift starts; a press during a running lift is queued, see ``climb.py``).  Summary: failure rate per pattern / spacing / speed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-ClimbHop-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--guidance", type=float, default=1.0)
parser.add_argument("--min-clear", type=float, default=0.02)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401
from tancho_v3_lab.tasks.staged.climb import REF_T_END  # noqa: E402
from tancho_v3_lab.tasks.staged.scene import WHEEL_RADIUS_M  # noqa: E402

L, R, B = (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)
# Each pattern: list of (mode, hold multiplier).  "hold_L+hop": LT held through, then a hop.
PATTERNS = {
    "single_L": [L],
    "single_R": [R],
    "hop": [B],
    "hop2": [B, B],
    "hop3": [B, B, B],
    "L_R": [L, R],
    "L_R_L_R": [L, R, L, R],
    "L_hop": [L, B],
}
GAPS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
PRESS = [0.1, 0.3]
DRIVE = [(0.0, 0.0), (0.3, 0.0), (0.6, 0.0), (-0.3, 0.0), (0.4, 0.8), (0.6, 1.0)]
T0, TAIL = 1.5, 2.5


def scenarios():
    out = []
    for (name, pat), gap, press, (vx, wz) in itertools.product(PATTERNS.items(), GAPS, PRESS, DRIVE):
        if len(pat) == 1 and gap != GAPS[0]:
            continue  # spacing is meaningless for a single press
        if press >= gap:
            continue  # the trigger would never be released between presses
        out.append({"pattern": name, "modes": pat, "gap": gap, "press": press, "vx": vx, "wz": wz})
    return out


def lift_starts(s):
    """Reference-lift start time per press under the command's queue rule (one
    queued edge per side while that side's 0.4 s lift runs); None if merged."""
    phase_start = {0: -10.0, 1: -10.0}
    queued = {0: None, 1: None}
    out = []
    for j, m in enumerate(s["modes"]):
        edge = T0 + j * s["gap"]
        t_side = []
        for side in (0, 1):
            if m[side] <= 0:
                continue
            # Release any queued lift that started before this edge.
            if queued[side] is not None and phase_start[side] + REF_T_END <= edge:
                phase_start[side] = phase_start[side] + REF_T_END
                queued[side] = None
            if edge >= phase_start[side] + REF_T_END:
                phase_start[side] = edge
                t_side.append(edge)
            elif queued[side] is None:
                queued[side] = j
                t_side.append(phase_start[side] + REF_T_END)
            else:
                t_side.append(None)
        out.append(None if any(t is None for t in t_side) else max(t_side))
    return out


@torch.inference_mode()
def main() -> int:
    sc = scenarios()
    n = len(sc)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=n, use_fabric=True)
    t_end = T0 + max(s["gap"] * (len(s["modes"]) - 1) for s in sc) + TAIL
    cfg.episode_length_s = t_end + 5.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    r = cfg.commands.base_velocity.ranges
    r.lin_vel_x = r.lin_vel_y = r.ang_vel_z = (0.0, 0.0)
    cfg.commands.climb.random_press_prob = 0.0
    cfg.commands.climb.auto_prob = 0.0
    cfg.commands.climb.resampling_time_range = (1.0e6, 1.0e6)
    if hasattr(cfg.commands.climb, "burst_prob"):
        cfg.commands.climb.burst_prob = 0.0
        cfg.commands.climb.both_skew_s = 0.0
    cfg.actions.leg_pos.guidance_scale = args.guidance
    env = gym.make(args.task, cfg=cfg)
    core = env.unwrapped
    dev = core.device
    robot = core.scene["robot"]
    vel = core.command_manager.get_term("base_velocity")
    climb = core.command_manager.get_term("climb")
    policy = torch.jit.load(str(args.checkpoint.resolve()), map_location=dev).eval()
    wheel_ids, _ = robot.find_bodies(["wheel_L", "wheel_R"], preserve_order=True)
    dt = core.step_dt

    cmd = torch.tensor([[s["vx"], 0.0, s["wz"]] for s in sc], device=dev)
    # Press table: (env, step, mode, hold steps)
    events = {}
    for i, s in enumerate(sc):
        for j, m in enumerate(s["modes"]):
            k = round((T0 + j * s["gap"]) / dt)
            events.setdefault(k, []).append((i, m, s["press"], j))

    obs, _ = env.reset()
    ground = robot.data.body_pos_w[:, wheel_ids, 2].mean(dim=1) - WHEEL_RADIUS_M  # (n,)
    fell_at = torch.full((n,), -1.0, device=dev)
    # Per env, per press: peak clearance of the pressed wheels.
    max_presses = max(len(s["modes"]) for s in sc)
    peak = torch.zeros(n, max_presses, 2, device=dev)
    want = torch.zeros(n, max_presses, 2, device=dev)
    win_start = torch.full((n, max_presses), 1.0e6, device=dev)
    win_end = torch.full((n, max_presses), -1.0, device=dev)
    for i, s in enumerate(sc):
        for j, (m, t_lift) in enumerate(zip(s["modes"], lift_starts(s))):
            want[i, j] = torch.tensor(m, device=dev)
            if t_lift is None:
                continue  # merged into an already queued lift: nothing extra expected
            # Peak tuck is 0.1-0.25 s into the lift; skip the previous lift's tail.
            win_start[i, j] = t_lift + 0.1
            win_end[i, j] = t_lift + 0.6

    steps = round(t_end / dt)
    for k in range(steps):
        t = k * dt
        vel.vel_command_b[:] = cmd
        for i, m, press, _ in events.get(k, []):
            climb.random_mode[i] = torch.tensor(m, device=dev)
            climb.random_timer[i] = press
            if hasattr(climb, "random_elapsed"):
                climb.random_elapsed[i] = 0.0
                climb.side_delay[i] = 0.0
        obs, _, term, _, _ = env.step(policy(obs["policy"]))
        t += dt
        alive = fell_at < 0
        clear = robot.data.body_pos_w[:, wheel_ids, 2] - WHEEL_RADIUS_M - ground.unsqueeze(1)  # (n, 2)
        in_win = ((t >= win_start) & (t < win_end) & alive.unsqueeze(1)).unsqueeze(-1)  # (n, P, 1)
        peak = torch.where(in_win, torch.maximum(peak, clear.unsqueeze(1)), peak)
        fell_at = torch.where(term.bool() & alive, torch.full_like(fell_at, t), fell_at)

    counted = (win_end > 0).unsqueeze(-1)
    min_clear = torch.where((want > 0) & counted, peak, torch.full_like(peak, 1.0)).amin(dim=(1, 2))
    rows = []
    for i, s in enumerate(sc):
        rows.append(
            {
                "pattern": s["pattern"], "gap": s["gap"], "press": s["press"], "vx": s["vx"], "wz": s["wz"],
                "fell_at_s": round(float(fell_at[i]), 2) if fell_at[i] >= 0 else None,
                "min_clear_mm": round(float(min_clear[i]) * 1000, 1),
            }
        )

    def rate(key):
        groups = {}
        for row in rows:
            g = groups.setdefault(row[key] if not isinstance(key, tuple) else tuple(row[x] for x in key), [0, 0, 0])
            g[0] += 1
            g[1] += row["fell_at_s"] is not None
            g[2] += row["fell_at_s"] is None and row["min_clear_mm"] < args.min_clear * 1000
        return {str(k): {"n": v[0], "fall": round(v[1] / v[0], 3), "weak_lift": round(v[2] / v[0], 3)} for k, v in groups.items()}

    falls = sum(r["fell_at_s"] is not None for r in rows)
    weak = sum(r["fell_at_s"] is None and r["min_clear_mm"] < args.min_clear * 1000 for r in rows)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "guidance": args.guidance,
        "scenarios": n,
        "fall_rate": round(falls / n, 4),
        "weak_lift_rate": round(weak / n, 4),
        "by_pattern": rate("pattern"),
        "by_gap": rate("gap"),
        "by_drive": rate(("vx", "wz")),
        "by_press": rate("press"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
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
