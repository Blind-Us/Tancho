"""Tancho v3 6-DOF wheeled-leg manager-based environments.

設計原則
--------
1. 目前以平地直立站穩為目標；保留速度追蹤與 curriculum term，以零權重或門檻停用。
2. 只有輪子可以接地；base/thigh/calf 接地會強懲罰並終止，避免趴地刷分。
3. 站姿預設 thigh=-0.60 rad、calf=+1.10 rad，base 目標高度配合 0.15/0.10 m 腿段。
4. Mesh 為 local frame (STL 原點 == 關節軸心)，URDF visual/collision origin 全為 0。
"""
from pathlib import Path
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from . import custom_rewards as cr

ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "robots" / "Tancho_v3"
URDF_PATH = str(ASSET_DIR / "urdf" / "Tancho_v3.urdf")

# -- 站姿參數 ---------------------------------------------------------------
STAND_THIGH = -0.50
STAND_CALF = 0.80
RESPAWN_ROOT_HEIGHT = 0.26
BASE_HEIGHT_TARGET = 0.22
IMU_POS_ROOT = (-0.00835741999, 0.0000000160456, -0.0294337942)
IMU_ROT_ROOT = (0.707106781, 0.707106781, 0.0, 0.0)
PPO_STEPS_PER_ITERATION = 24
CURRICULUM_ON_ITERATION = 1500
CURRICULUM_VEL_ON_STEPS = CURRICULUM_ON_ITERATION * PPO_STEPS_PER_ITERATION
CURRICULUM_PUSH_ON_STEPS = CURRICULUM_ON_ITERATION * PPO_STEPS_PER_ITERATION
# 先禁止策略把腿當第三支點，再逐步提高最低高度。高度門檻若過早啟用，
# PPO 會在尚未學會輪上平衡前被大量截斷，反而沒有足夠軌跡學到 18 cm 站姿。
POSTURE_CURRICULUM_ON_STEPS = 400 * PPO_STEPS_PER_ITERATION
POSTURE_CURRICULUM_RAMP_STEPS = 1000 * PPO_STEPS_PER_ITERATION
LEG_CONTACT_TERMINATION_ON_STEPS = 250 * PPO_STEPS_PER_ITERATION
LEG_CONTACT_TERMINATION_RAMP_STEPS = 750 * PPO_STEPS_PER_ITERATION


@configclass
class TanchoV3SceneCfg(InteractiveSceneCfg):
    """共用 6-DOF 輪腿機器人場景。"""

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=URDF_PATH,
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=None,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, RESPAWN_ROOT_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "joint_thigh_L": STAND_THIGH,
                "joint_calf_L": STAND_CALF,
                "joint_wheel_L": 0.0,
                "joint_thigh_R": STAND_THIGH,
                "joint_calf_R": STAND_CALF,
                "joint_wheel_R": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            # 腿部 4 關節：位置控制 (implicit PD)
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["joint_thigh_L", "joint_calf_L",
                                  "joint_thigh_R", "joint_calf_R"],
                stiffness=25.0,
                damping=2.0,
                effort_limit_sim=40.0,
                velocity_limit_sim=1.0,
            ),
            # 輪子 2 關節：MDP effort action；保留少量被動阻尼。
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["joint_wheel_L", "joint_wheel_R"],
                stiffness=0.0,
                damping=0.3,
                effort_limit_sim=2.0,
                velocity_limit_sim=20.0,
            ),
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
class ActionsCfg:
    """動作空間：腿部位置 + 輪子力矩（皆使用 Isaac Lab MDP action term）。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R"],
        # 0.25 rad 無法在下沉前建立足夠的預載扭矩；放寬控制範圍，
        # 讓策略可主動伸腿支撐，而非只能等誤差變大後被動追趕。
        scale=0.8,
        use_default_offset=True,
    )
    # 實測 velocity target 在倒地前幾乎無法改變輪心位置；直接力矩可恢復控制權。
    # 名稱維持 joint_vel，以保持既有 checkpoint/action ordering 與監控腳本相容。
    joint_vel = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["joint_wheel_L", "joint_wheel_R"],
        scale=2.0,
    )


@configclass
class CommandsCfg:
    """速度指令保留；站立階段由 rel_standing_envs=1.0 固定為零指令。"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,     # 100% env 初始為站立 (command = 0)
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.02, 0.02),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.57, 1.57),
            heading=(-3.14, 3.14),
        ),
    )


@configclass
class ObservationsCfg:
    """觀測空間。"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=1.0)  
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """終止條件。"""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # 只有 base 碰地才終止；thigh/calf 碰撞保留在 collision reward 處理。
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link_root"]),
            "threshold": 10.0,
        },
    )
    minimum_base_height = TerminationTermCfg(
        func=cr.base_below_minimum_height,
        params={
            "minimum_height": 0.14,
            "initial_height": 0.06,
            "minimum_below_steps": 20,
            "initial_below_steps": 100,
            "enable_after_steps": POSTURE_CURRICULUM_ON_STEPS,
            "ramp_steps": POSTURE_CURRICULUM_RAMP_STEPS,
        },
    )
    sustained_leg_contact = TerminationTermCfg(
        func=cr.sustained_illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*thigh.*", ".*calf.*"],
            ),
            "threshold": 5.0,
            "minimum_contact_steps": 10,
            "initial_contact_steps": 50,
            "enable_after_steps": LEG_CONTACT_TERMINATION_ON_STEPS,
            "ramp_steps": LEG_CONTACT_TERMINATION_RAMP_STEPS,
        },
    )
    # 超過約 46 度視為已失去可恢復站姿，避免接近躺平仍累積存活時間。
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},
    )


@configclass
class EventCfg:
    """重置與 Domain Randomization。"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 圍繞站姿小擾動 (default_joint_pos 來自 init_state)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 零增量停用固定機身總成的質量隨機化。
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

    # 速度範圍為零，因此 curriculum 啟動前不施加推力。
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(16.0, 16.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


@configclass
class CurriculumCfg:
    """在 PPO iteration 1500 啟動速度指令與推力 curriculum。"""

    enable_velocity_commands = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "command_manager.cfg.base_velocity.rel_standing_envs",
            "modify_fn": cr.curriculum_enable_velocity,
            "modify_params": {"num_steps": CURRICULUM_VEL_ON_STEPS, "target": 0.5},
        },
    )
    enable_push = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.push_robot.params.velocity_range",
            "modify_fn": cr.curriculum_enable_push,
            "modify_params": {
                "num_steps": CURRICULUM_PUSH_ON_STEPS,
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            },
        },
    )
    # strengthen_minimum_height = CurriculumTermCfg(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "reward_manager.cfg.minimum_base_height.weight",
    #         "modify_fn": cr.curriculum_set_after_steps,
    #         "modify_params": {
    #             "num_steps": POSTURE_CURRICULUM_ON_STEPS,
    #             "target": -6.0,
    #         },
    #     },
    # )
