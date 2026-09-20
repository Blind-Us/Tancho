"""Fixed-leg Tancho V3 flat-terrain environment.

This task keeps the validated Flat task's terrain, rewards, commands,
terminations, curricula, timing, and friction.  The physical asset and action
space are the intended experimental changes: the thigh/calf assemblies are
rigidly merged in the nominal pose and only the two wheel joints remain.  Note
that merging also makes leg collisions part of ``base_link_root``, so the
shared base-contact termination has the corresponding fixed-body semantics.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from . import custom_events as ce
from .flat_env_cfg import FlatRewardsCfg, make_flat_terrain
from .tancho_v3_env_cfg import CommandsCfg, CurriculumCfg, ObservationsCfg, TerminationsCfg


ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "robots" / "Tancho_v3"
FIXED_URDF_PATH = str(ASSET_DIR / "urdf" / "Tancho_v3_fixed.urdf")


@configclass
class FixedActionsCfg:
    """Two physical wheel-torque actions; the legs are rigid bodies, not PD-held joints."""

    joint_vel = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["joint_wheel_L", "joint_wheel_R"],
        scale=0.45,
    )


@configclass
class FixedEventCfg:
    """Match Flat events while using the fixed asset's geometry-derived reset."""

    reset_tancho_on_wheels = EventTerm(
        func=ce.reset_tancho_on_wheels,
        mode="reset",
        params={
            "terrain_height": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset": True,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link_root"]),
            "mass_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.8, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(16.0, 16.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


# Define the scene explicitly to avoid inheriting the full asset's leg actuators.
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg, ImuCfg  # noqa: E402
from .tancho_v3_env_cfg import IMU_POS_ROOT, IMU_ROT_ROOT  # noqa: E402


@configclass
class TanchoV3FixedFlatSceneCfg(InteractiveSceneCfg):
    terrain = make_flat_terrain()
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85)),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=FIXED_URDF_PATH,
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=None,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, ce.FIXED_TARGET_ROOT_HEIGHT_M),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={"joint_wheel_L": 0.0, "joint_wheel_R": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["joint_wheel_L", "joint_wheel_R"],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=0.45,
                velocity_limit_sim=188.0,
            )
        },
    )
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base_link_root",
        offset=ImuCfg.OffsetCfg(pos=IMU_POS_ROOT, rot=IMU_ROT_ROOT),
        debug_vis=False,
    )


@configclass
class TanchoV3FixedFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: TanchoV3FixedFlatSceneCfg = TanchoV3FixedFlatSceneCfg(num_envs=4096, env_spacing=3.0)
    actions: FixedActionsCfg = FixedActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: FlatRewardsCfg = FlatRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: FixedEventCfg = FixedEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
