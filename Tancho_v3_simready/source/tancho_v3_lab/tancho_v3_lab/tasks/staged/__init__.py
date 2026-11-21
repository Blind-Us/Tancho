"""Staged Tancho V3 tasks: wheel-only stand -> 6-DOF stand -> 6-DOF walk.

Each stage is built from the same four elements (``observations.py``,
``actions.py``, ``rewards.py``, ``terminations.py``); ``env_cfg.py`` assembles
them.  See ``README.md``.
"""

import gymnasium as gym

from . import agents

_STAGES = {
    "TanchoV3-RoughSquat": ("TanchoV3RoughSquat", "TanchoV3RoughSquatPPORunnerCfg"),
    "TanchoV3-DeepSquat": ("TanchoV3DeepSquat", "TanchoV3DeepSquatPPORunnerCfg"),
    "TanchoV3-Squat": ("TanchoV3Squat", "TanchoV3SquatPPORunnerCfg"),
    "TanchoV3-HopHip": ("TanchoV3HopHip", "TanchoV3HopHipPPORunnerCfg"),
    "TanchoV3-StairHip": ("TanchoV3StairHip", "TanchoV3StairHipPPORunnerCfg"),
    "TanchoV3-WheelOnly-Flat": ("TanchoV3WheelOnlyFlat", "TanchoV3WheelOnlyPPORunnerCfg"),
    "TanchoV3-Stand-Flat": ("TanchoV3StandFlat", "TanchoV3StandPPORunnerCfg"),
    "TanchoV3-Walk-Flat": ("TanchoV3WalkFlat", "TanchoV3WalkPPORunnerCfg"),
    "TanchoV3-Walk-Rough": ("TanchoV3WalkRough", "TanchoV3WalkRoughPPORunnerCfg"),
    "TanchoV3-Walk-Step": ("TanchoV3WalkStep", "TanchoV3WalkStepPPORunnerCfg"),
    "TanchoV3-Climb": ("TanchoV3Climb", "TanchoV3ClimbPPORunnerCfg"),
    "TanchoV3-ClimbHop": ("TanchoV3ClimbHop", "TanchoV3ClimbHopPPORunnerCfg"),
    "TanchoV3-ClimbFree": ("TanchoV3ClimbFree", "TanchoV3ClimbFreePPORunnerCfg"),
    "TanchoV3-ClimbHopFree": ("TanchoV3ClimbHopFree", "TanchoV3ClimbHopFreePPORunnerCfg"),
    "TanchoV3-ClimbHopDR": ("TanchoV3ClimbHopDR", "TanchoV3ClimbHopDRPPORunnerCfg"),
    "TanchoV3-ClimbStepDR": ("TanchoV3ClimbStepDR", "TanchoV3ClimbStepDRPPORunnerCfg"),
    "TanchoV3-ClimbStepDRv2": ("TanchoV3ClimbStepDRv2", "TanchoV3ClimbStepDRv2PPORunnerCfg"),
    "TanchoV3-StairV2": ("TanchoV3StairV2", "TanchoV3StairPPORunnerCfg"),
    "TanchoV3-StairFast": ("TanchoV3StairFast", "TanchoV3StairPPORunnerCfg"),
    "TanchoV3-Stair": ("TanchoV3Stair", "TanchoV3StairPPORunnerCfg"),
    "TanchoV3-Recover": ("TanchoV3Recover", "TanchoV3RecoverPPORunnerCfg"),
    "TanchoV3-RecoverWide": ("TanchoV3RecoverWide", "TanchoV3RecoverWidePPORunnerCfg"),
}

# 4-DOF no-thigh robot (``nothigh.py``).
_NOTHIGH_STAGES = {
    "TanchoV3-NoThigh-Stand": ("TanchoV3NoThighStand", "TanchoV3NoThighStandPPORunnerCfg"),
    "TanchoV3-NoThigh-Walk": ("TanchoV3NoThighWalk", "TanchoV3NoThighWalkPPORunnerCfg"),
}

# Active hop (``hop_push.py``).
_HOP_PUSH_STAGES = {
    "TanchoV3-HopPush": ("TanchoV3HopPush", "TanchoV3HopPushPPORunnerCfg"),
    "TanchoV3-HopPushV1": ("TanchoV3HopPushV1", "TanchoV3HopPushPPORunnerCfg"),
    "TanchoV3-HopPushV1Hi": ("TanchoV3HopPushV1Hi", "TanchoV3HopPushPPORunnerCfg"),
    "TanchoV3-HopCharge": ("TanchoV3HopCharge", "TanchoV3HopPushPPORunnerCfg"),
}

for _module, _stages in (("hop_push", _HOP_PUSH_STAGES),):
    for _task, (_cfg, _runner) in _stages.items():
        for _suffix, _cfg_suffix in (("-v0", "EnvCfg"), ("-Play-v0", "PlayEnvCfg")):
            gym.register(
                id=_task + _suffix,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{__name__}.{_module}:{_cfg}{_cfg_suffix}",
                    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner}",
                },
            )

for _task, (_cfg, _runner) in _NOTHIGH_STAGES.items():
    for _suffix, _cfg_suffix in (("-v0", "EnvCfg"), ("-Play-v0", "PlayEnvCfg")):
        gym.register(
            id=_task + _suffix,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.nothigh:{_cfg}{_cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner}",
            },
        )

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

# Stability-first retraining, preserving the HopPush actor/action contract.
for _suffix, _cfg in (("-v0", "TanchoV3StabilityEnvCfg"), ("-Play-v0", "TanchoV3StabilityPlayEnvCfg")):
    gym.register(
        id="TanchoV3-Stability" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.stability:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.stability:TanchoV3StabilityPPORunnerCfg",
        },
    )

# RL v2 stage 1: station keeping (wheel odometry in obs, position-hold reward, DR).
for _suffix, _cfg in (("-v0", "TanchoV3StationEnvCfg"), ("-Play-v0", "TanchoV3StationPlayEnvCfg")):
    gym.register(
        id="TanchoV3-Station" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station:TanchoV3StationPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationV2EnvCfg"), ("-Play-v0", "TanchoV3StationV2PlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationV2" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station:TanchoV3StationPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationV3EnvCfg"), ("-Play-v0", "TanchoV3StationV3PlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationV3" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station:TanchoV3StationPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationV4EnvCfg"), ("-Play-v0", "TanchoV3StationV4PlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationV4" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station:TanchoV3StationPPORunnerCfg",
        },
    )

# Station keeping + LT/RT leg lift + hold-to-crouch (user 2026-11-19).
for _suffix, _cfg in (("-v0", "TanchoV3StationSkillsEnvCfg"), ("-Play-v0", "TanchoV3StationSkillsPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSkills" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatEnvCfg"), ("-Play-v0", "TanchoV3StationSquatPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquat" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatHoldEnvCfg"), ("-Play-v0", "TanchoV3StationSquatHoldPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatHold" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1EnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1PlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1HoldEnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1HoldPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1Hold" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1ComEnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1ComPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1Com" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1SoftEnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1SoftPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1Soft" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1QuietEnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1QuietPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1Quiet" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationHopKd1EnvCfg"), ("-Play-v0", "TanchoV3StationHopKd1PlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationHopKd1" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1TerrainEnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1TerrainPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1Terrain" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1LatEnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1LatPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1Lat" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1LatTerrainEnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1LatTerrainPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1LatTerrain" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationIntEnvCfg"), ("-Play-v0", "TanchoV3StationIntPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationInt" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationIntNoLoadEnvCfg"), ("-Play-v0", "TanchoV3StationIntPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationIntNoLoad" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1QuietTorqueEnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1LatPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1QuietTorque" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1PushEnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1LatPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1Push" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )

for _suffix, _cfg in (("-v0", "TanchoV3StationSquatKd1Push2EnvCfg"), ("-Play-v0", "TanchoV3StationSquatKd1LatPlayEnvCfg")):
    gym.register(
        id="TanchoV3-StationSquatKd1Push2" + _suffix,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.station_skills:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.station_skills:TanchoV3StationSkillsPPORunnerCfg",
        },
    )
