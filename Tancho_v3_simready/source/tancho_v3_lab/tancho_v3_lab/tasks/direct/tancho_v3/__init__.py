"""Gym registrations for Tancho v3 environments."""

import gymnasium as gym

from . import agents


def register_environment(
    task_id: str,
    env_cfg_entry_point: str,
    runner_cfg: str = "TanchoV3PPORunnerCfg",
) -> None:
    """Register an environment with the shared Tancho v3 runner config."""
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{env_cfg_entry_point}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg}",
        },
    )


register_environment("TanchoV3-Flat-v0", "flat_env_cfg:TanchoV3FlatEnvCfg")
register_environment("TanchoV3-Rough-v0", "rough_env_cfg:TanchoV3RoughEnvCfg")
register_environment(
    "TanchoV3-Fixed-Flat-v0",
    "fixed_flat_env_cfg:TanchoV3FixedFlatEnvCfg",
    "TanchoV3FixedPPORunnerCfg",
)
