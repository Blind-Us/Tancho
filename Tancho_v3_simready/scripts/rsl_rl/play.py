"""Minimal playback script for a trained Tancho V3 RSL-RL policy."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Play a trained RSL-RL policy for Tancho V3."
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments.",
)

parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Registered Isaac Lab task name.",
)

parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent configuration entry point.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Environment seed.",
)

parser.add_argument(
    "--load_run",
    type=str,
    default=".*",
    help="Run folder name or regex.",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="model_.*.pt",
    help="Checkpoint regex or direct .pt path.",
)

# 0 = unlimited
parser.add_argument(
    "--num_steps",
    type=int,
    default=0,
    help="Maximum inference steps. 0 = unlimited.",
)

# Isaac Lab launcher arguments
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Hydra should only see Hydra arguments
sys.argv = [sys.argv[0]] + hydra_args


# =============================================================================
# Launch Isaac Sim
# =============================================================================

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# Imports after Isaac Sim startup
# =============================================================================

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Register Tancho tasks
import tancho_v3_lab.tasks  # noqa: F401


# =============================================================================
# RSL-RL config conversion
# =============================================================================

def to_rsl_rl_cfg(agent_cfg):
    """
    Convert Isaac Lab RSL-RL config to the split actor/critic format
    expected by the installed RSL-RL version.
    """

    data = agent_cfg.to_dict()

    policy_cfg = dict(data["policy"])
    algorithm_cfg = dict(data["algorithm"])

    # Everything outside policy / algorithm belongs to runner config.
    runner_cfg = {
        key: value
        for key, value in data.items()
        if key not in {
            "policy",
            "algorithm",
            "class_name",
        }
    }

    # -------------------------------------------------------------------------
    # Actor
    # -------------------------------------------------------------------------

    actor_cfg = {
        "class_name": "MLPModel",

        "hidden_dims": policy_cfg.get(
            "actor_hidden_dims",
            [256, 256, 128],
        ),

        "activation": policy_cfg.get(
            "activation",
            "elu",
        ),

        "obs_normalization": policy_cfg.get(
            "actor_obs_normalization",
            False,
        ),

        "distribution_cfg": {
            "class_name": "GaussianDistribution",

            "init_std": policy_cfg.get(
                "init_noise_std",
                1.0,
            ),
        },
    }

    # -------------------------------------------------------------------------
    # Critic
    # -------------------------------------------------------------------------

    critic_cfg = {
        "class_name": "MLPModel",

        "hidden_dims": policy_cfg.get(
            "critic_hidden_dims",
            [256, 256, 128],
        ),

        "activation": policy_cfg.get(
            "activation",
            "elu",
        ),

        "obs_normalization": policy_cfg.get(
            "critic_obs_normalization",
            False,
        ),
    }

    # -------------------------------------------------------------------------
    # PPO
    # -------------------------------------------------------------------------

    algorithm_cfg["class_name"] = "PPO"

    # Legacy option not used by current installed RSL-RL.
    algorithm_cfg.pop(
        "use_spo",
        None,
    )

    # -------------------------------------------------------------------------
    # Observation groups
    # -------------------------------------------------------------------------

    runner_cfg["obs_groups"] = {
        "actor": ["policy"],
        "critic": ["policy"],
        "policy": ["policy"],
    }

    runner_cfg.setdefault(
        "multi_gpu",
        None,
    )

    # -------------------------------------------------------------------------
    # Final config
    # -------------------------------------------------------------------------

    return {
        **runner_cfg,
        "actor": actor_cfg,
        "critic": critic_cfg,
        "algorithm": algorithm_cfg,
    }


# =============================================================================
# Main
# =============================================================================

@hydra_task_config(
    args_cli.task,
    args_cli.agent,
)
def main(
    env_cfg: (
        ManagerBasedRLEnvCfg
        | DirectRLEnvCfg
        | DirectMARLEnvCfg
    ),
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Load a trained checkpoint and run inference."""

    # -------------------------------------------------------------------------
    # Environment config
    # -------------------------------------------------------------------------

    env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.seed = (
        args_cli.seed
        if args_cli.seed is not None
        else agent_cfg.seed
    )

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # -------------------------------------------------------------------------
    # Resolve checkpoint
    # -------------------------------------------------------------------------

    log_root_path = os.path.abspath(
        os.path.join(
            "logs",
            "rsl_rl",
            agent_cfg.experiment_name,
        )
    )

    if os.path.isfile(args_cli.checkpoint):

        resume_path = os.path.abspath(
            args_cli.checkpoint
        )

    else:

        resume_path = get_checkpoint_path(
            log_root_path,
            args_cli.load_run,
            args_cli.checkpoint,
        )

    print(
        f"[INFO] Loading checkpoint:\n"
        f"       {resume_path}"
    )

    # -------------------------------------------------------------------------
    # Create environment
    # -------------------------------------------------------------------------

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    if isinstance(
        env.unwrapped,
        DirectMARLEnv,
    ):
        env = multi_agent_to_single_agent(
            env
        )

    # -------------------------------------------------------------------------
    # Isaac Lab RSL-RL wrapper
    # -------------------------------------------------------------------------

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=getattr(
            agent_cfg,
            "clip_actions",
            None,
        ),
    )

    # -------------------------------------------------------------------------
    # Build compatible RSL-RL config
    # -------------------------------------------------------------------------

    runner_cfg = to_rsl_rl_cfg(
        agent_cfg
    )

    # -------------------------------------------------------------------------
    # Create runner
    # -------------------------------------------------------------------------

    runner = OnPolicyRunner(
        env,
        runner_cfg,
        log_dir=None,
        device=env.unwrapped.device,
    )

    # -------------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------------

    runner.load(
        resume_path
    )

    # -------------------------------------------------------------------------
    # Get inference policy
    # -------------------------------------------------------------------------

    policy = runner.get_inference_policy(
        device=env.unwrapped.device
    )

    # -------------------------------------------------------------------------
    # Initial observation
    # -------------------------------------------------------------------------

    obs = env.get_observations()

    # -------------------------------------------------------------------------
    # Playback loop
    # -------------------------------------------------------------------------

    print("[INFO] Starting trained-policy playback.")
    print("[INFO] Playback will continue until:")
    print("       - the Isaac Sim window is closed, or")
    print("       - --num_steps is reached if > 0.")

    steps = 0

    try:

        with torch.inference_mode():

            while simulation_app.is_running():

                # -------------------------------------------------------------
                # Trained policy inference
                # -------------------------------------------------------------

                actions = policy(obs)

                # -------------------------------------------------------------
                # Environment step
                # -------------------------------------------------------------

                step_result = env.step(
                    actions
                )

                # Different wrapper versions may return 4 or 5 values.
                if len(step_result) == 4:

                    (
                        obs,
                        rewards,
                        dones,
                        extras,
                    ) = step_result

                else:

                    (
                        obs,
                        privileged_obs,
                        rewards,
                        dones,
                        extras,
                    ) = step_result

                steps += 1

                # -------------------------------------------------------------
                # Optional inference step limit
                # -------------------------------------------------------------

                if (
                    args_cli.num_steps > 0
                    and steps >= args_cli.num_steps
                ):

                    print(
                        f"[INFO] Reached "
                        f"{steps} inference steps."
                    )

                    break

    finally:

        env.close()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    finally:

        simulation_app.close()