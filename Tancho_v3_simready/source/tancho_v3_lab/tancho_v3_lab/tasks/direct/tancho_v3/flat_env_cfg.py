import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewardTerm, SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import custom_rewards as cr
from .tancho_v3_env_cfg import (
    ActionsCfg,
    BASE_HEIGHT_TARGET,
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
    alive = RewardTerm(func=mdp.is_alive, weight=0.005)
    termination_penalty = RewardTerm(func=mdp.is_terminated, weight=-300.0)
    wheel_contact = RewardTerm(func=cr.wheel_ground_contact, weight=0.2, params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0})
    tracking_lin_vel = RewardTerm(func=mdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    tracking_ang_vel = RewardTerm(func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    base_height = RewardTerm(
        func=cr.base_height_l2_normalized,
        weight=-0.1,
        params={"target_height": BASE_HEIGHT_TARGET, "height_scale": 0.05},
    )

    mirror = RewardTerm(
        func=cr.mirror_leg_actions_l1,
        weight=-0.1,
        params={
            "action_name": "joint_pos",
        },
    )

    stable_standing = RewardTerm(
        func=cr.stable_standing_bonus,
        weight=1.5,
        params={
            "target_height": BASE_HEIGHT_TARGET,
            # 擴大指數核的有效梯度範圍，不增加 reward weight。
            "height_scale": 0.10,
            "max_tilt_deg": 10.0,
        },
    )

    flat_orientation = RewardTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z = RewardTerm(func=mdp.lin_vel_z_l2, weight=-0.0)
    ang_vel_xy = RewardTerm(func=mdp.ang_vel_xy_l2, weight=-0.0)
    wheel_under_com = RewardTerm(func=cr.wheel_under_com_l2, weight=-5e-3,
                                 params={"asset_cfg": SceneEntityCfg("robot"),
                                         "left_wheel_body": "wheel_L",
                                         "right_wheel_body": "wheel_R",
                                         "com_body_name": "base_link_root",
                                         "error_scale": 0.05})
    joint_deviation = RewardTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "joint_thigh_L",
                    "joint_calf_L",
                    "joint_thigh_R",
                    "joint_calf_R",
                ],
            ),
        },
    )

    collision = RewardTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 5.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*thigh.*", ".*calf.*"],
            ),
        },
    )


    # dof_vel = RewardTerm(func=mdp.joint_vel_l2, weight=-2e-5)
    # dof_acc = RewardTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    # torques = RewardTerm(func=mdp.joint_torques_l2, weight=-5e-5)
    # action_rate = RewardTerm(func=mdp.action_rate_l2, weight=-0.005)
    # dof_pos_limits = RewardTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # dof_vel_limits = RewardTerm(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.9})


@configclass
class TanchoV3FlatSceneCfg(TanchoV3SceneCfg):
    terrain = make_flat_terrain()


@configclass
class TanchoV3FlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: TanchoV3FlatSceneCfg = TanchoV3FlatSceneCfg(num_envs=4096, env_spacing=3.0)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: FlatRewardsCfg = FlatRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
