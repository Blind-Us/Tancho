#!/usr/bin/env python3
"""Export an RSL-RL MLP actor checkpoint to a TorchScript ``policy.pt`` without launching Isaac Sim.

Usage:
    python scripts/rsl_rl/export_policy_jit.py INPUT.pt OUTPUT.pt [--reference OLD_policy.pt]

The actor and observation normalizer are rebuilt exactly as in
``convert_pt_to_onnx.py``.  With ``--reference`` the exported graph is compared
against another TorchScript policy on random observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from convert_pt_to_onnx import RslRlMlpActor, _activation_from_run, _actor_state, _numbered_layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="RSL-RL .pt checkpoint")
    parser.add_argument("output", type=Path, help="Destination TorchScript .pt file")
    parser.add_argument("--activation", help="Override activation if agent.yaml is absent")
    parser.add_argument("--reference", type=Path, help="TorchScript policy to compare against")
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = _actor_state(checkpoint)
    weights, biases = _numbered_layers(state)
    mean = state.get("obs_normalizer._mean")
    std = state.get("obs_normalizer._std")
    if (mean is None) != (std is None):
        raise ValueError("Checkpoint contains an incomplete observation normalizer")
    if mean is not None:
        mean, std = mean.detach().cpu().float(), std.detach().cpu().float()
    actor = RslRlMlpActor(weights, biases, mean, std, _activation_from_run(source, args.activation)).eval()
    scripted = torch.jit.trace(actor, torch.zeros(1, weights[0].shape[1]))

    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(destination))
    reloaded = torch.jit.load(str(destination)).eval()
    obs = torch.randn(256, weights[0].shape[1])
    with torch.inference_mode():
        error = float((reloaded(obs) - actor(obs)).abs().max())
        print(f"PASS: {destination} obs={weights[0].shape[1]} actions={weights[-1].shape[0]} reload_error={error:.2e}")
        if args.reference is not None:
            reference = torch.jit.load(str(args.reference), map_location="cpu").eval()
            print(f"max_abs_diff_vs_reference={float((reference(obs) - reloaded(obs)).abs().max()):.2e}")


if __name__ == "__main__":
    main()
