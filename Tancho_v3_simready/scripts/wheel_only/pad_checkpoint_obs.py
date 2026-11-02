"""Append observation inputs to an RSL-RL checkpoint (zero weights, identity normalization).

New observation terms appended at the end of the policy / critic groups get zero
first-layer weights, so the padded network computes exactly what the old one did
until training uses the new inputs.  Their normalizer entries are mean 0, var 1
(raw values; the inputs are already in [0, 1]).  Optionally resets the action
noise of selected action dims (``--std``), e.g. leg std collapsed to ~3e-4 after
hop v2, which leaves nothing to explore.

    python scripts/wheel_only/pad_checkpoint_obs.py --input model_final.pt --output padded.pt \
        --actor-extra 3 --critic-extra 3 --std 0:4=0.1
"""

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--actor-extra", type=int, required=True)
parser.add_argument("--critic-extra", type=int, required=True)
parser.add_argument("--std", default=None, help="start:end=value, sets distribution.std_param[start:end]")
args = parser.parse_args()

ckpt = torch.load(args.input, map_location="cpu", weights_only=False)


def pad(state: dict, extra: int):
    if extra == 0:
        return
    w = state["mlp.0.weight"]
    state["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra, dtype=w.dtype)], dim=1)
    for key, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
        v = state[key]
        state[key] = torch.cat([v, torch.full((v.shape[0], extra), fill, dtype=v.dtype)], dim=1)


pad(ckpt["actor_state_dict"], args.actor_extra)
pad(ckpt["critic_state_dict"], args.critic_extra)
if args.std:
    span, value = args.std.split("=")
    start, end = (int(x) for x in span.split(":"))
    ckpt["actor_state_dict"]["distribution.std_param"][start:end] = float(value)
ckpt.pop("optimizer_state_dict", None)
torch.save(ckpt, args.output)
print("actor in", ckpt["actor_state_dict"]["mlp.0.weight"].shape[1], "critic in", ckpt["critic_state_dict"]["mlp.0.weight"].shape[1],
      "std", ckpt["actor_state_dict"]["distribution.std_param"].tolist())
