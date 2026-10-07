"""Staged Tancho V3 tasks: wheel-only stand -> 6-DOF stand -> 6-DOF walk.

Each stage is built from the same four elements (``observations.py``,
``actions.py``, ``rewards.py``, ``terminations.py``); ``env_cfg.py`` assembles
them.  See ``README.md``.
"""

import gymnasium as gym

from . import agents

_STAGES = {
    "TanchoV3-WheelOnly-Flat": ("TanchoV3WheelOnlyFlat", "TanchoV3WheelOnlyPPORunnerCfg"),
    "TanchoV3-Stand-Flat": ("TanchoV3StandFlat", "TanchoV3StandPPORunnerCfg"),
    "TanchoV3-Walk-Flat": ("TanchoV3WalkFlat", "TanchoV3WalkPPORunnerCfg"),
    "TanchoV3-Walk-Rough": ("TanchoV3WalkRough", "TanchoV3WalkRoughPPORunnerCfg"),
    "TanchoV3-Walk-Step": ("TanchoV3WalkStep", "TanchoV3WalkStepPPORunnerCfg"),
    "TanchoV3-Climb": ("TanchoV3Climb", "TanchoV3ClimbPPORunnerCfg"),
    "TanchoV3-ClimbHop": ("TanchoV3ClimbHop", "TanchoV3ClimbHopPPORunnerCfg"),
    "TanchoV3-ClimbFree": ("TanchoV3ClimbFree", "TanchoV3ClimbFreePPORunnerCfg"),
    "TanchoV3-ClimbHopFree": ("TanchoV3ClimbHopFree", "TanchoV3ClimbHopFreePPORunnerCfg"),
    "TanchoV3-Recover": ("TanchoV3Recover", "TanchoV3RecoverPPORunnerCfg"),
    "TanchoV3-RecoverWide": ("TanchoV3RecoverWide", "TanchoV3RecoverWidePPORunnerCfg"),
}

for _task, (_cfg, _runner) in _STAGES.items():
    for _suffix, _cfg_suffix in (("-v0", "EnvCfg"), ("-Play-v0", "PlayEnvCfg")):
        gym.register(
            id=_task + _suffix,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:{_cfg}{_cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner}",
            },
        )
