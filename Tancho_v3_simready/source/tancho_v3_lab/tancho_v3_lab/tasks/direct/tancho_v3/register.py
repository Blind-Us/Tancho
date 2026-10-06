"""Gym registrations for the legacy Tancho v3 tasks (Flat, Rough, Fixed-Flat).

Not imported by default; ``tasks/__init__.py`` loads it only when
``TANCHO_LEGACY_TASKS=1``.  ``staged`` imports ``custom_rewards`` and
``custom_events`` from this package, so registration cannot live in the
package ``__init__``.
"""

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
            "env_cfg_entry_point": f"{__package__}.{env_cfg_entry_point}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg}",
        },
    )


register_environment("TanchoV3-Flat-v0", "flat_env_cfg:TanchoV3FlatEnvCfg")
register_environment("TanchoV3-Flat-Play-v0", "flat_env_cfg:TanchoV3FlatPlayEnvCfg")
register_environment("TanchoV3-Rough-v0", "rough_env_cfg:TanchoV3RoughEnvCfg")
register_environment(
    "TanchoV3-Fixed-Flat-v0",
    "fixed_flat_env_cfg:TanchoV3FixedFlatEnvCfg",
    "TanchoV3FixedPPORunnerCfg",
)
register_environment(
    "TanchoV3-Fixed-Flat-Play-v0",
    "fixed_flat_env_cfg:TanchoV3FixedFlatPlayEnvCfg",
    "TanchoV3FixedPPORunnerCfg",
)
