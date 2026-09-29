#!/usr/bin/env python3
"""Convert an RSL-RL MLP actor checkpoint to a standalone ONNX policy.

Usage:
    python scripts/rsl_rl/convert_pt_to_onnx.py INPUT.pt OUTPUT.onnx

The converter reads the actor weights and empirical observation normalizer
directly from the checkpoint.  It uses ``params/agent.yaml`` beside the run to
recover the activation function, so Isaac Sim does not need to be launched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn


ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "elu": torch.nn.functional.elu,
    "relu": torch.nn.functional.relu,
    "selu": torch.nn.functional.selu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "identity": lambda value: value,
}


class RslRlMlpActor(nn.Module):
    """Deterministic actor reconstructed from an RSL-RL state dictionary."""

    def __init__(
        self,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor],
        mean: torch.Tensor | None,
        std: torch.Tensor | None,
        activation_name: str,
    ) -> None:
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(value, requires_grad=False) for value in weights])
        self.biases = nn.ParameterList([nn.Parameter(value, requires_grad=False) for value in biases])
        self.activation_name = activation_name
        self.register_buffer("obs_mean", mean if mean is not None else torch.empty(0))
        self.register_buffer("obs_std", std if std is not None else torch.empty(0))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_mean.numel() != 0:
            obs = (obs - self.obs_mean) / (self.obs_std + 1.0e-2)
        value = obs
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = torch.nn.functional.linear(value, weight, bias)
            if index + 1 < len(self.weights):
                value = ACTIVATIONS[self.activation_name](value)
        return value


def _activation_from_run(checkpoint: Path, override: str | None) -> str:
    if override:
        return override.lower()
    config_path = checkpoint.parent / "params" / "agent.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Cannot infer activation: {config_path} is missing. Pass --activation explicitly."
        )
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    activation = str(config.get("policy", {}).get("activation", "")).lower()
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported or missing activation {activation!r} in {config_path}")
    return activation


def _actor_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "actor_state_dict" in checkpoint:
        return checkpoint["actor_state_dict"]
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        prefixes = ("actor.", "actor_critic.actor.")
        for prefix in prefixes:
            selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if selected:
                return selected
    raise KeyError("Checkpoint has no supported RSL-RL actor_state_dict/model_state_dict")


def _numbered_layers(state: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    layers: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for key, weight in state.items():
        if not key.startswith("mlp.") or not key.endswith(".weight"):
            continue
        number = int(key.split(".")[1])
        bias_key = f"mlp.{number}.bias"
        if bias_key not in state:
            raise KeyError(f"Missing {bias_key}")
        layers.append((number, weight.detach().cpu().float(), state[bias_key].detach().cpu().float()))
    layers.sort(key=lambda item: item[0])
    if not layers:
        raise ValueError("No actor MLP layers named mlp.<index>.weight were found")
    return [item[1] for item in layers], [item[2] for item in layers]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="RSL-RL .pt checkpoint")
    parser.add_argument("output", type=Path, help="Destination .onnx file")
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), help="Override activation if agent.yaml is absent")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if source.suffix.lower() != ".pt" or not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found or not a .pt file: {source}")
    if destination.suffix.lower() != ".onnx":
        raise ValueError(f"Output must end in .onnx: {destination}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = _actor_state(checkpoint)
    weights, biases = _numbered_layers(state)
    mean = state.get("obs_normalizer._mean")
    std = state.get("obs_normalizer._std")
    if (mean is None) != (std is None):
        raise ValueError("Checkpoint contains an incomplete observation normalizer")
    if mean is not None:
        mean, std = mean.detach().cpu().float(), std.detach().cpu().float()

    activation = _activation_from_run(source, args.activation)
    actor = RslRlMlpActor(weights, biases, mean, std, activation).eval()
    observation_size = weights[0].shape[1]
    action_size = weights[-1].shape[0]
    example = torch.zeros(1, observation_size, dtype=torch.float32)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = actor(example).numpy()
        torch.onnx.export(
            actor,
            example,
            destination,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=args.opset,
            dynamo=False,
        )

    import onnx
    from onnx.reference import ReferenceEvaluator

    model = onnx.load(destination)
    onnx.checker.check_model(model)
    actual = ReferenceEvaluator(model).run(None, {"obs": example.numpy()})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError(f"ONNX numerical validation failed: max_abs_error={max_error:.3e}")

    print(f"PASS: {destination}")
    print(f"input=obs[batch,{observation_size}] output=actions[batch,{action_size}]")
    print(f"activation={activation} observation_normalizer={mean is not None}")
    print(f"max_abs_error={max_error:.3e}")


if __name__ == "__main__":
    main()
