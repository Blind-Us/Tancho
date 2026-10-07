#!/usr/bin/env python3
"""Train the staged Tancho V3 tasks (``tasks/staged``) with RSL-RL.

Based on Isaac Lab's ``scripts/reinforcement_learning/rsl_rl/train.py``.  The
project's ``scripts/rsl_rl/train.py`` is not used because its legacy-config
translation drops the rsl-rl>=4 ``actor``/``critic`` model configs and forces
the critic onto the policy observation group.

Additions over the stock script:
  * registers the Tancho tasks (``import tancho_v3_lab.tasks``);
  * refuses to train when files that define this task are uncommitted, and writes
    the Tancho commit hash to ``<log_dir>/git_commit.txt`` (and the run name);
  * ``--init_checkpoint``: start from another run's weights (stage 2 -> 3);
  * exports ``exported/policy.pt`` (TorchScript) and ``policy.onnx`` at the end.

    python scripts/wheel_only/train.py --headless --task TanchoV3-WheelOnly-Flat-v0
    python scripts/wheel_only/train.py --headless --task TanchoV3-Stand-Flat-v0
    python scripts/wheel_only/train.py --headless --task TanchoV3-Walk-Flat-v0 \
        --init_checkpoint logs/rsl_rl/tancho_v3_stand/<run>/model_final.pt
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

REPO_DIR = Path(__file__).resolve().parents[2]
# Everything that defines the task, its asset and this training entry point.
TRACKED_PATHS = [
    "source/tancho_v3_lab/tancho_v3_lab/tasks/staged",
    "source/tancho_v3_lab/tancho_v3_lab/tasks/direct/tancho_v3/custom_events.py",
    "source/tancho_v3_lab/tancho_v3_lab/tasks/direct/tancho_v3/custom_rewards.py",
    "source/tancho_v3_lab/tancho_v3_lab/tasks/__init__.py",
    "source/tancho_v3_lab/tancho_v3_lab/assets/robots/Tancho_v3/urdf/Tancho_v3_wheel_only.urdf",
    "source/tancho_v3_lab/tancho_v3_lab/assets/robots/Tancho_v3/urdf/Tancho_v3_wheel_only.json",
    "source/tancho_v3_lab/tancho_v3_lab/assets/robots/Tancho_v3/urdf/Tancho_v3.urdf",
    "scripts/wheel_only",
]

parser = argparse.ArgumentParser(description="Train the Tancho V3 wheel-only balance policy.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="TanchoV3-WheelOnly-Flat-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--init_checkpoint", type=str, default=None, help="Model weights to start from; iteration count restarts at 0.")
parser.add_argument(
    "--keep_obs_norm",
    action="store_true",
    help="With --init_checkpoint: keep the checkpoint's observation statistics (same observation distribution, e.g. walk -> rough).",
)
parser.add_argument("--allow-dirty", action="store_true", help="Train even if task files are uncommitted.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _git(*cmd: str) -> str:
    return subprocess.run(["git", *cmd], cwd=REPO_DIR, check=True, capture_output=True, text=True).stdout.strip()


COMMIT = _git("rev-parse", "HEAD")
DIRTY = _git("status", "--porcelain", "--", *TRACKED_PATHS)
if DIRTY and not args_cli.allow_dirty:
    sys.exit(f"Refusing to train: task files are not committed.\n{DIRTY}\nCommit them or pass --allow-dirty.")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations or agent_cfg.max_iterations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    agent_cfg.device = args_cli.device or agent_cfg.device

    short = COMMIT[:7] + ("-dirty" if DIRTY else "")
    agent_cfg.run_name = f"{agent_cfg.run_name}_{short}" if agent_cfg.run_name else short
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{agent_cfg.run_name}")
    env_cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "git_commit.txt"), "w") as stream:
        stream.write(f"commit {COMMIT}\ndirty_task_files {'yes' if DIRTY else 'no'}\n{DIRTY}\n")
    print(f"[INFO] Tancho commit {COMMIT} (task files dirty: {bool(DIRTY)}) -> {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    start = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if args_cli.init_checkpoint:
        runner.load(args_cli.init_checkpoint, load_cfg={"actor": True, "critic": True})
        # Restart the observation statistics.  The stand checkpoint saw a zero
        # command for ~3e8 samples, so its command std is 0 and the running
        # average would need ~1e9 walk samples to catch up: a 0.3 m/s command
        # was normalized to ~4 instead of ~0.9 and the policy overshot to 1 m/s.
        # With count 0 the first batch replaces mean/var, then they keep updating.
        # Between tasks with the same observation distribution (walk -> rough)
        # the reset is harmful: the first batch is near-identical reset states,
        # the tiny variance blows the inputs up, and a policy that walked falls
        # within 0.2 s and needs hundreds of iterations to recover.
        if not args_cli.keep_obs_norm:
            for model in (runner.alg.actor, runner.alg.critic):
                if hasattr(model.obs_normalizer, "count"):
                    model.obs_normalizer.count.zero_()
        with open(os.path.join(log_dir, "git_commit.txt"), "a") as stream:
            stream.write(f"init_checkpoint {os.path.abspath(args_cli.init_checkpoint)}\n")
            stream.write(f"keep_obs_norm {args_cli.keep_obs_norm}\n")
        print(f"[INFO] Initialized from {args_cli.init_checkpoint}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    runner.save(os.path.join(log_dir, "model_final.pt"))
    export_dir = os.path.join(log_dir, "exported")
    runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
    runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
    print(f"[INFO] Exported policy to {export_dir}")
    print(f"Training time: {round(time.time() - start, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
