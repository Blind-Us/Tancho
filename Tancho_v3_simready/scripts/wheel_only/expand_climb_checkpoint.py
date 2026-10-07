#!/usr/bin/env python3
"""Turn a walk/rough checkpoint into a ``TanchoV3-Climb-v0`` starting point.

* Actor input 25 -> 27 (LT, RT trigger appended): new first-layer columns are
  zero, so before training the policy ignores the triggers.
* Critic input 29 -> 71 (trigger 2 + height scan 40 appended): zero columns.
* Leg action scale 0.25 -> 0.6 rad: the leg rows of the output layer, the
  leg action std and the normalizer statistics of the leg ``last_action``
  inputs are multiplied by 0.25/0.6, so the same weights command the same
  joint targets as before.
* Normalizer count is lowered to ``--norm-count`` so the statistics of the new
  inputs adapt within a few iterations (train with ``--keep_obs_norm``).

Usage:
    python scripts/wheel_only/expand_climb_checkpoint.py <model.pt> <out.pt>
"""

from __future__ import annotations

import argparse

import torch

OLD_LEG_SCALE = 0.25
NEW_LEG_SCALE = 0.6
LEG_ACTION_DIMS = [0, 1, 2, 3]
# Actor observation layout (observations.py): gravity 3, gyro 3, command 3,
# wheel vel 2, last action 6 (legs 4 + wheels 2), leg pos 4, leg vel 4.
LAST_ACTION_LEG_OBS = [11, 12, 13, 14]

ACTOR_NEW = {"dims": 2, "mean": 0.0, "var": 0.05}
CRITIC_NEW = [
    {"dims": 2, "mean": 0.0, "var": 0.05},  # trigger
    {"dims": 40, "mean": 0.0, "var": 1.0e-4},  # height scan (m), ~1 cm std
]


def _expand(sd: dict, blocks: list[dict], leg_obs: list[int], k: float, count: float) -> None:
    w = sd["mlp.0.weight"]
    extra = sum(b["dims"] for b in blocks)
    sd["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra, dtype=w.dtype)], dim=1)
    means = [torch.full((1, b["dims"]), b["mean"]) for b in blocks]
    vars_ = [torch.full((1, b["dims"]), b["var"]) for b in blocks]
    mean = sd["obs_normalizer._mean"].clone()
    var = sd["obs_normalizer._var"].clone()
    # Leg last-action inputs shrink by k at the new scale: keep normalized values unchanged.
    mean[:, leg_obs] *= k
    var[:, leg_obs] *= k * k
    sd["obs_normalizer._mean"] = torch.cat([mean, *means], dim=1)
    sd["obs_normalizer._var"] = torch.cat([var, *vars_], dim=1)
    sd["obs_normalizer._std"] = sd["obs_normalizer._var"].sqrt()
    sd["obs_normalizer.count"] = torch.tensor(count, dtype=sd["obs_normalizer.count"].dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--norm-count", type=float, default=1.0e6)
    parser.add_argument("--leg-std", type=float, default=0.15, help="leg action std after expansion (new units)")
    parser.add_argument("--wheel-std", type=float, default=0.05)
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    k = OLD_LEG_SCALE / NEW_LEG_SCALE
    actor, critic = ckpt["actor_state_dict"], ckpt["critic_state_dict"]
    assert actor["mlp.0.weight"].shape[1] == 25, actor["mlp.0.weight"].shape
    assert critic["mlp.0.weight"].shape[1] == 29, critic["mlp.0.weight"].shape

    # Output layer: legs command 0.25*a_old = 0.6*a_new -> a_new = k*a_old.
    actor["mlp.6.weight"][LEG_ACTION_DIMS] *= k
    actor["mlp.6.bias"][LEG_ACTION_DIMS] *= k
    std = actor["distribution.std_param"]
    print(f"old std {std.tolist()}")
    std[LEG_ACTION_DIMS] = torch.clamp(std[LEG_ACTION_DIMS] * k, min=args.leg_std)
    std[4:] = torch.clamp(std[4:], min=args.wheel_std)
    print(f"new std {std.tolist()}")

    _expand(actor, [ACTOR_NEW], LAST_ACTION_LEG_OBS, k, args.norm_count)
    # Critic layout: lin vel 3, ang vel 3, gravity 3, command 3, wheel vel 2,
    # last action 6 (legs at 14..17), base height 1, leg pos 4, leg vel 4.
    _expand(critic, CRITIC_NEW, [14, 15, 16, 17], k, args.norm_count)

    ckpt.pop("optimizer_state_dict", None)
    ckpt["iter"] = 0
    torch.save(ckpt, args.dst)
    print(f"actor in {actor['mlp.0.weight'].shape[1]}, critic in {critic['mlp.0.weight'].shape[1]} -> {args.dst}")


if __name__ == "__main__":
    main()
