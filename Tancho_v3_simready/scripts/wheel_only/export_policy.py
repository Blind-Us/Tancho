#!/usr/bin/env python3
"""Export a staged-task checkpoint to ``<run>/exported/policy.pt`` and ``policy.onnx``.

Builds the runner exactly as ``train.py`` does (same agent config, same rsl-rl
version handling), so the checkpoint layout always matches.
"""

import argparse
import importlib.metadata as metadata
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, help="training task id, e.g. TanchoV3-WheelOnly-Flat-v0")
parser.add_argument("--checkpoint", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = 1
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint))
    export_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), "exported")
    runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
    runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
    print(f"[INFO] Exported policy to {export_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
