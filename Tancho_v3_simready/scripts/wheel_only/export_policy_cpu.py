#!/usr/bin/env python3
"""Export an RSL-RL actor checkpoint to TorchScript + ONNX without starting Isaac Sim.

Same network as ``export_policy.py`` produces (obs normalizer (x - mean) / (std + eps) ->
ELU MLP, deterministic mean action), for screening checkpoints while a training run holds
the GPU.  Checked against export_policy.py output (max abs diff printed with --compare).

    python scripts/wheel_only/export_policy_cpu.py --checkpoint <run>/model_300.pt [--compare <policy.pt>]
"""

import argparse
import os

import torch
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--compare", default=None, help="an export_policy.py policy.pt to compare against")
parser.add_argument("--eps", type=float, default=1e-2, help="EmpiricalNormalization eps")
args = parser.parse_args()


class Actor(nn.Module):
    def __init__(self, sd: dict, eps: float):
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"].clone())
        self.register_buffer("std", sd["obs_normalizer._std"].clone())
        self.eps = eps
        ids = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
        layers = []
        for n, i in enumerate(ids):
            w, b = sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w); lin.bias.data.copy_(b)
            layers.append(lin)
            if n < len(ids) - 1:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp((x - self.mean) / (self.std + self.eps))


sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
actor = Actor(sd, args.eps).eval()
out = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "exported")
os.makedirs(out, exist_ok=True)
ts = torch.jit.script(actor)
ts.save(os.path.join(out, "policy.pt"))
n_in = sd["mlp.0.weight"].shape[1]
torch.onnx.export(actor, torch.zeros(1, n_in), os.path.join(out, "policy.onnx"), input_names=["obs"], output_names=["actions"], opset_version=11)
if args.compare:
    ref = torch.jit.load(args.compare)
    x = torch.randn(256, n_in)
    print("max abs diff vs", args.compare, float((ref(x) - ts(x)).abs().max()))
print("exported to", out)
