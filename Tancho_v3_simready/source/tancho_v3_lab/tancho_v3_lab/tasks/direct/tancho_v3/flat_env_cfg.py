import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewardTerm,
    SceneEntityCfg,
    TerminationTermCfg as TerminationTerm,
)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import custom_events as ce
from . import custom_rewards as cr
from .tancho_v3_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    TanchoV3SceneCfg,
    TerminationsCfg,
)


def make_flat_terrain() -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )


@configclass
class FlatRewardsCfg:
        """Tancho-specific standing objective built from physical state errors.

        Official Isaac Lab MDP terms are used wherever possible.  The only
        custom terms are whole-robot capture-point error and bilateral leg
        symmetry, neither of which has an official equivalent.
        """

        # At 50 Hz Isaac Lab multiplies this weight by step_dt=0.02, so a
        # non-timeout fall contributes -4.0 while a full 20 s survival earns
        # +20 from the alive term.  This makes falling unambiguously worse
        # without overwhelming all dense physical feedback.
        termination_penalty = RewardTerm(
            func=mdp.is_terminated,
            weight=-200.0,
        )
        is_alive = RewardTerm(
            func=mdp.is_alive,
            weight=1.0,
        )

        # Balance state: body tilt and pitch/roll rate approach zero.
        upright = RewardTerm(
            func=mdp.flat_orientation_l2,
            weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        ang_vel = RewardTerm(
            func=mdp.ang_vel_xy_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # The standing command is exactly zero longitudinal/lateral/yaw speed.
        ang_vel_z = RewardTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        lin_vel = RewardTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        vertical_velocity = RewardTerm(
            func=mdp.lin_vel_z_l2,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # One dynamic balance objective only: at rest the axle tracks COM;
        # during motion it tracks the velocity-shifted capture point.  The
        # 50-mm normalization prevents this term from overwhelming survival,
        # posture, settling, energy, and safety objectives.
        capture_point = RewardTerm(
            func=cr.wheel_capture_point_l2,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "left_wheel_body": "wheel_L",
                "right_wheel_body": "wheel_R",
                "com_body_name": None,
                "error_scale": 0.050,
                "wheel_radius": 0.03614,
                "gravity_magnitude": 9.81,
                "minimum_com_height": 0.05,
                "max_capture_offset": 0.12,
            },
        )

        # Tancho is mechanically symmetric; equal physical pose uses equal q.
        mirror = RewardTerm(func=cr.mirror_leg_l2, weight=-0.5)

        # Normalize torque cost by each motor's squared peak torque.  Therefore
        # 100% wheel and 100% leg utilization have the same per-joint cost;
        # the large numerical difference between 0.45 and 12.5 Nm cannot bias
        # the optimizer toward one actuator family merely because of units.
        wheel_torque = RewardTerm(
            func=mdp.joint_torques_l2,
            weight=-0.049382716,  # -0.01 / (0.45 Nm)^2
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_wheel_L", "joint_wheel_R"]),
            },
        )
        leg_torque = RewardTerm(
            func=mdp.joint_torques_l2,
            weight=-0.000064,  # -0.01 / (12.5 Nm)^2
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["joint_thigh_.*", "joint_calf_.*"],
                ),
            },
        )
        action_rate = RewardTerm(
            func=mdp.action_rate_l2,
            weight=-0.01,
        )
        joint_acceleration = RewardTerm(
            func=mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        leg_position_limits = RewardTerm(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_thigh_.*", "joint_calf_.*"]),
            },
        )
        wheel_velocity_limits = RewardTerm(
            func=mdp.joint_vel_limits,
            weight=-1.0,
            params={
                "soft_ratio": 0.9,
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_wheel_.*"]),
            },
        )





@configclass
class TanchoV3FlatSceneCfg(TanchoV3SceneCfg):
    terrain = make_flat_terrain()


@configclass
class FullBodyTerminationsCfg(TerminationsCfg):
    """Match the fixed asset's effective no-nonwheel-contact rule."""

    base_contact = TerminationTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_link_root", "thigh_.*", "calf_.*"],
            ),
            "threshold": 10.0,
        },
    )


@configclass
class FlatPlayEventCfg(EventCfg):
    direction_markers = EventTerm(
        func=ce.visualize_tancho_directions,
        mode="interval",
        interval_range_s=(0.05, 0.05),
    )


@configclass
class TanchoV3FlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: TanchoV3FlatSceneCfg = TanchoV3FlatSceneCfg(num_envs=4096, env_spacing=3.0)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: FlatRewardsCfg = FlatRewardsCfg()
    terminations: FullBodyTerminationsCfg = FullBodyTerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 200 Hz PhysX integration, 50 Hz policy/action update.  The 0.25-rad
        # leg action scale is defined at this hardware-aligned control period.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005


@configclass
class TanchoV3FlatPlayEnvCfg(TanchoV3FlatEnvCfg):
    """Play-only config with the trained push curriculum already enabled."""

    curriculum: None = None
    events: FlatPlayEventCfg = FlatPlayEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.push_robot.interval_range_s = (10.0, 10.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
        self.events.push_robot.params["debug_vis"] = True
        self.events.push_robot.params["push_probability"] = 1.0
