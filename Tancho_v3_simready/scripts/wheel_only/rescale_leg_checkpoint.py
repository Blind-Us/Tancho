#!/usr/bin/env python3
"""Change a Climb checkpoint's leg action scale without changing its behavior.

The leg rows of the actor output layer, the leg action std and the normalizer
statistics of the leg ``last_action`` inputs (actor 11-14, critic 14-17) are
multiplied by old/new, so the same weights command the same joint targets.

Usage: python scripts/wheel_only/rescale_leg_checkpoint.py <model.pt> <out.pt> --old 0.6 --new 1.0
"""

from __future__ import annotations

import argparse

import torch

LEG_ACTION_DIMS = [0, 1, 2, 3]
NORM_EPS = 1.0e-2  # rsl_rl EmpiricalNormalization: (x - mean) / (std + eps)


def _scale_obs(sd: dict, idx: list[int], k: float) -> None:
    mean = sd["obs_normalizer._mean"].clone()
    std = sd["obs_normalizer._var"].sqrt()
    mean[:, idx] *= k
    std[:, idx] = (k * (std[:, idx] + NORM_EPS) - NORM_EPS).clamp(min=1.0e-6)
    sd["obs_normalizer._mean"] = mean
    sd["obs_normalizer._var"] = std.square()
    sd["obs_normalizer._std"] = std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--old", type=float, required=True)
    parser.add_argument("--new", type=float, required=True)
    parser.add_argument("--leg-std", type=float, default=0.03, help="minimum leg action std after rescaling (new units)")
    args = parser.parse_args()
    k = args.old / args.new
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    actor, critic = ckpt["actor_state_dict"], ckpt["critic_state_dict"]
    last = max(int(key.split(".")[1]) for key in actor if key.startswith("mlp.") and key.endswith(".weight"))
    actor[f"mlp.{last}.weight"][LEG_ACTION_DIMS] *= k
    actor[f"mlp.{last}.bias"][LEG_ACTION_DIMS] *= k
    std = actor["distribution.std_param"]
    std[LEG_ACTION_DIMS] = torch.clamp(std[LEG_ACTION_DIMS] * k, min=args.leg_std)
    _scale_obs(actor, [11, 12, 13, 14], k)
    _scale_obs(critic, [14, 15, 16, 17], k)
    # optimizer state kept only so export_policy.py can load it (train.py --init_checkpoint ignores it)
    ckpt["iter"] = 0
    torch.save(ckpt, args.dst)
    print(f"leg scale {args.old} -> {args.new} (x{k:.3f}), std {std.tolist()} -> {args.dst}")


if __name__ == "__main__":
    main()
