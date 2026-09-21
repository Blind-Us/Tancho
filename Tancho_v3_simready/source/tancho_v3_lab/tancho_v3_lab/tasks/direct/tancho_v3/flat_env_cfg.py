import math

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
        # 1. 活著就給正獎勵
        is_alive = RewardTerm(
            func=mdp.is_alive,
            weight=1.0,
        )

        # 2. 機身保持直立
        # theta -> 0
        upright = RewardTerm(
            func=mdp.flat_orientation_l2,
            weight=-5.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # 3. 降低 pitch / roll angular velocity
        # theta_dot -> 0
        ang_vel = RewardTerm(
            func=mdp.ang_vel_xy_l2,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # 4. 追蹤零 yaw-rate，避免左右輪輸出不一致造成原地打轉
        ang_vel_z = RewardTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.5,
            params={
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
            },
        )

        # 5. 追蹤零水平速度，讓平衡後仍持續滾動的機器人回到靜止
        lin_vel = RewardTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=0.5,
            params={
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
            },
        )

        # 6. 官方 MDP 無 whole-body COM 到輪軸線距離，保留 Tancho 自訂項目
        wheel_under_com = RewardTerm(
            func=cr.wheel_under_com_l2,
            weight=-0.02,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "left_wheel_body": "wheel_L",
                "right_wheel_body": "wheel_R",
                "com_body_name": None,
                "error_scale": 0.015,
            },
        )

        # 7. wheel torque 不要長時間打滿
        torque = RewardTerm(
            func=mdp.joint_torques_l2,
            weight=-1.0e-4,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "joint_wheel_L",
                        "joint_wheel_R",
                    ],
                ),
            },
        )

        # 8. leg torque 不要長時間打滿 12.5 Nm
        leg_torque = RewardTerm(
            func=mdp.joint_torques_l2,
            weight=-1.0e-4,
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

        # 9. 抑制相鄰 control step 的動作跳變，避免 reset 後立即跳到大角度目標。
        action_rate = RewardTerm(
            func=mdp.action_rate_l2,
            weight=-1.0e-3,
        )

        # 10. 防止機身保持水平卻整體下蹲到觸地。
        # Flat terrain 目標高度來自 Gate B 的 nominal wheel-ground reset geometry。
        base_height = RewardTerm(
            func=mdp.base_height_l2,
            weight=-100.0,
            params={
                "target_height": BASE_HEIGHT_TARGET,
                "asset_cfg": SceneEntityCfg("robot"),
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
        self.decimation = 2
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
