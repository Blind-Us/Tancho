"""Standalone wheel-only balance task; independent of ``tasks/direct``."""

import gymnasium as gym

from . import agents

_RUNNER = f"{agents.__name__}.rsl_rl_ppo_cfg:TanchoV3WheelOnlyPPORunnerCfg"

gym.register(
    id="TanchoV3-WheelOnly-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_only_env_cfg:TanchoV3WheelOnlyFlatEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)
gym.register(
    id="TanchoV3-WheelOnly-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_only_env_cfg:TanchoV3WheelOnlyFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)
