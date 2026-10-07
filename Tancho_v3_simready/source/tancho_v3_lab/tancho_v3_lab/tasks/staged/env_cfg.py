"""Stage assembly: scene + the four elements + commands/events.

| Stage | Train task                     | Robot   | Command          |
|-------|--------------------------------|---------|------------------|
| 1     | TanchoV3-WheelOnly-Flat-v0     | 2 wheels, legs merged | zero   |
| 2     | TanchoV3-Stand-Flat-v0         | 6-DOF   | zero             |
| 3     | TanchoV3-Walk-Flat-v0          | 6-DOF   | vx / yaw rate    |
| 4     | TanchoV3-Walk-Rough-v0         | 6-DOF   | vx / yaw rate, bumps + slopes |
| 5     | TanchoV3-Walk-Step-v0          | 6-DOF   | vx / yaw rate, 0 -> 3 cm steps |

Each has a ``-Play-v0`` variant: nominal physics, no noise, no randomization,
upright reset, one robot.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .actions import FullActionsCfg, WheelOnlyActionsCfg
from .observations import FullObservationsCfg, WheelOnlyObservationsCfg
from .rewards import FullStandRewardsCfg, FullWalkRewardsCfg, WheelOnlyRewardsCfg
from .scene import DECIMATION, EPISODE_LENGTH_S, NOMINAL_FRICTION, SIM_DT_S, FullSceneCfg, WheelOnlySceneCfg
from .terminations import FullTerminationsCfg, WheelOnlyTerminationsCfg
from .terrain import ROUGH_GENERATOR, STEP_GENERATOR, make_terrain, play_generator, terrain_levels_tracking


# -- commands -------------------------------------------------------------------
@configclass
class StandCommandsCfg:
    """Zero velocity command."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )


@configclass
class WalkCommandsCfg:
    """Forward speed and yaw rate.  Lateral speed stays 0: two coaxial wheels cannot move sideways.

    0.6 m/s is 16.6 rad/s of wheel speed (32% of rated); 1.0 rad/s of yaw on the
    0.23 m track needs only +/-0.12 m/s of wheel-speed difference.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.6), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(0.0, 0.0)
        ),
    )


# -- events (training randomization; all disabled in Play) -----------------------
@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link_root"]),
            "mass_distribution_params": (-0.15, 0.25),  # about -6%/+10% of the robot
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"pitch": (-0.05, 0.05)},
            "velocity_range": {"x": (-0.1, 0.1), "pitch": (-0.2, 0.2)},
        },
    )
    # Back to the nominal pose (the init_state joint positions), at rest.
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    # Horizontal velocity kick on the chassis.  0.8 m/s on the 2.6 kg robot is
    # 2.1 N*s, i.e. the 42 N x 50 ms pulse of the push sweep.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.8, 0.8)}},
    )


# -- stage 1: wheel-only stand ---------------------------------------------------
@configclass
class TanchoV3WheelOnlyFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: WheelOnlySceneCfg = WheelOnlySceneCfg(num_envs=4096, env_spacing=2.0)
    observations: WheelOnlyObservationsCfg = WheelOnlyObservationsCfg()
    actions: WheelOnlyActionsCfg = WheelOnlyActionsCfg()
    rewards: WheelOnlyRewardsCfg = WheelOnlyRewardsCfg()
    terminations: WheelOnlyTerminationsCfg = WheelOnlyTerminationsCfg()
    commands: StandCommandsCfg = StandCommandsCfg()
    events: EventsCfg = EventsCfg()
    curriculum = None

    def __post_init__(self):
        self.sim.dt = SIM_DT_S
        self.decimation = DECIMATION
        self.sim.render_interval = DECIMATION
        self.episode_length_s = EPISODE_LENGTH_S


def _to_play(cfg: ManagerBasedRLEnvCfg) -> None:
    """Nominal physics, no pushes, no noise, deterministic upright reset."""
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 30.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.physics_material.params["static_friction_range"] = (NOMINAL_FRICTION, NOMINAL_FRICTION)
    cfg.events.physics_material.params["dynamic_friction_range"] = (NOMINAL_FRICTION, NOMINAL_FRICTION)
    cfg.events.add_base_mass = None
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.push_robot = None
    # Camera follows the root (about 0.26 m up) from about 1.1 m away (the default 7.5 m eye leaves a dot).
    cfg.viewer = ViewerCfg(eye=(0.8, 0.8, 0.1), lookat=(0.0, 0.0, -0.1), origin_type="asset_root", asset_name="robot")


@configclass
class TanchoV3WheelOnlyFlatPlayEnvCfg(TanchoV3WheelOnlyFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)


# -- stage 2: 6-DOF stand --------------------------------------------------------
@configclass
class TanchoV3StandFlatEnvCfg(TanchoV3WheelOnlyFlatEnvCfg):
    scene: FullSceneCfg = FullSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: FullObservationsCfg = FullObservationsCfg()
    actions: FullActionsCfg = FullActionsCfg()
    rewards: FullStandRewardsCfg = FullStandRewardsCfg()
    terminations: FullTerminationsCfg = FullTerminationsCfg()


@configclass
class TanchoV3StandFlatPlayEnvCfg(TanchoV3StandFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)


# -- stage 3: 6-DOF walk on command ---------------------------------------------
@configclass
class TanchoV3WalkFlatEnvCfg(TanchoV3StandFlatEnvCfg):
    rewards: FullWalkRewardsCfg = FullWalkRewardsCfg()
    commands: WalkCommandsCfg = WalkCommandsCfg()


@configclass
class TanchoV3WalkFlatPlayEnvCfg(TanchoV3WalkFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)
        self.commands.base_velocity.debug_vis = True


# -- stages 4/5: 6-DOF walk on uneven ground ---------------------------------------
# Same observation / action as the flat walk (blind: no height scan), started
# from a flat-walk checkpoint with ``--init_checkpoint``.
@configclass
class TerrainCurriculumCfg:
    terrain_levels = CurrTerm(func=terrain_levels_tracking)


@configclass
class TanchoV3WalkRoughEnvCfg(TanchoV3WalkFlatEnvCfg):
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = make_terrain(ROUGH_GENERATOR)


@configclass
class TanchoV3WalkRoughPlayEnvCfg(TanchoV3WalkRoughEnvCfg):
    """A single 16 m tile of the hardest bumps (2 cm peak-to-peak)."""

    def __post_init__(self):
        super().__post_init__()
        _to_play(self)
        self.commands.base_velocity.debug_vis = True
        self.curriculum = None
        self.scene.terrain = make_terrain(play_generator(ROUGH_GENERATOR, "rough", 1.0), max_init_level=None)


@configclass
class TanchoV3WalkStepEnvCfg(TanchoV3WalkRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = make_terrain(STEP_GENERATOR)


@configclass
class TanchoV3WalkStepPlayEnvCfg(TanchoV3WalkStepEnvCfg):
    """A single tile of 3 cm steps up (inverted pyramid: spawn in the pit)."""

    def __post_init__(self):
        super().__post_init__()
        _to_play(self)
        self.commands.base_velocity.debug_vis = True
        self.curriculum = None
        self.scene.terrain = make_terrain(play_generator(STEP_GENERATOR, "step_up", 1.0, size=8.0), max_init_level=None)
