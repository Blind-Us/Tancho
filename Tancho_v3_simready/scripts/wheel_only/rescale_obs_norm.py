#!/usr/bin/env python3
"""Shrink the observation-normalizer std of chosen inputs and compensate in the first layer.

y = W (x - m) / s: setting s -> s' and W[:, i] -> W[:, i] * s' / s leaves the policy
unchanged, but those inputs then reach the network s / s' times larger, so the same
optimizer step moves their effect s / s' times faster (used for the wheel odometry,
whose running std had settled at ~0.4 m while standing errors are a few cm).

    python scripts/wheel_only/rescale_obs_norm.py --input model.pt --output out.pt --actor 29:31 --critic 73:75 --std 0.1
"""
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--actor", required=True, help="start:end input slice")
parser.add_argument("--critic", default=None)
parser.add_argument("--std", type=float, required=True, help="new normalizer std (in scaled observation units)")
parser.add_argument("--action-std", type=float, default=None, help="also reset the action noise to this value")
args = parser.parse_args()
ck = torch.load(args.input, map_location="cpu", weights_only=False)


def rescale(sd, span):
    a, b = (int(x) for x in span.split(":"))
    old = sd["obs_normalizer._std"][0, a:b].clone()
    sd["obs_normalizer._std"][0, a:b] = args.std
    sd["obs_normalizer._var"][0, a:b] = args.std**2
    sd["mlp.0.weight"][:, a:b] *= args.std / old
    print(span, "std", old.tolist(), "->", args.std)


rescale(ck["actor_state_dict"], args.actor)
if args.critic:
    rescale(ck["critic_state_dict"], args.critic)
if args.action_std is not None:
    ck["actor_state_dict"]["distribution.std_param"][:] = args.action_std
ck.pop("optimizer_state_dict", None)
torch.save(ck, args.output)
