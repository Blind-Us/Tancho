import torch
import math
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp


def lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared horizontal root velocity; compatibility equivalent of the removed MDP term."""

    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_lin_vel_b[:, :2]), dim=1)


def ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared base yaw rate in the body frame."""

    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_ang_vel_b[:, 2])


# 雙輪接地  左右輪都有接觸地面時給分
def wheel_ground_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """輪子貼地獎勵：兩輪接觸力 > threshold 時給 1，否則 0。

    輪腿機器人需要輪子保持接觸地面才能用摩擦平衡。
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # 找 wheel body 的 index
    body_ids, _ = contact_sensor.find_bodies(".*wheel.*")
    forces = contact_sensor.data.net_forces_w[:, body_ids, :]
    contact = (torch.norm(forces, dim=-1) > threshold).float()
    # 兩輪都接觸 => 1
    return torch.prod(contact, dim=1)


def mirror_leg_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """懲罰左右腿實際 joint position 不一致。"""

    robot: Articulation = env.scene["robot"]

    joint_ids, _ = robot.find_joints(
        [
            "joint_thigh_L",
            "joint_thigh_R",
            "joint_calf_L",
            "joint_calf_R",
        ],
        preserve_order=True,
    )

    joint_pos = robot.data.joint_pos[:, joint_ids]

    return 0.5 * (
        (joint_pos[:, 0] - joint_pos[:, 1]).square()
        + (joint_pos[:, 2] - joint_pos[:, 3]).square()
    )

#-----------------------------------------------------------------------------------

# 穩定站立加分  高度接近目標、姿態穩時給較高分
def stable_standing_bonus(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_scale: float,
    max_tilt_deg: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    # -------------------------
    # Height L2 score
    # -------------------------
    height = robot.data.root_pos_w[:, 2]

    height_error = (
        height - target_height
    ) / height_scale

    height_score = torch.exp(
        -torch.square(height_error)
    )

    # -------------------------
    # Orientation L2 score
    # -------------------------
    gravity_xy = torch.linalg.vector_norm(
        robot.data.projected_gravity_b[:, :2],
        dim=1,
    )

    orientation_scale = math.sin(
        math.radians(max_tilt_deg)
    )

    orientation_error = (
        gravity_xy / orientation_scale
    )

    orientation_score = torch.exp(
        -torch.square(orientation_error)
    )

    # -------------------------
    # Standing score
    # -------------------------
    return height_score * orientation_score


def base_height_l2_normalized(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """以物理容許誤差正規化高度 L2，避免靠放大 reward weight 才看得到高度差。"""
    robot: Articulation = env.scene[asset_cfg.name]
    normalized_error = (robot.data.root_pos_w[:, 2] - target_height) / height_scale
    return torch.square(normalized_error)


def base_below_minimum_height(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    minimum_below_steps: int = 15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    counter_name = "_tancho_below_minimum_height_steps"
    below_steps = getattr(env, counter_name, None)

    if below_steps is None or below_steps.shape[0] != env.num_envs:
        below_steps = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=robot.device,
        )

    below = robot.data.root_pos_w[:, 2] < minimum_height

    below_steps = torch.where(
        below,
        below_steps + 1,
        torch.zeros_like(below_steps),
    )

    setattr(env, counter_name, below_steps)

    return below_steps >= minimum_below_steps


def sustained_illegal_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    minimum_contact_steps: int,
    initial_contact_steps: int | None = None,
    enable_after_steps: int = 0,
    ramp_steps: int = 0,
) -> torch.Tensor:
    """指定部位持續接地一段時間才終止，允許短暫擦碰但禁止用腿支撐。"""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    max_force = torch.max(torch.linalg.vector_norm(forces, dim=-1), dim=1).values
    in_contact = torch.any(max_force > threshold, dim=1)

    counter_name = "_tancho_sustained_illegal_contact_steps"
    contact_steps = getattr(env, counter_name, None)
    if contact_steps is None or contact_steps.shape[0] != env.num_envs:
        contact_steps = torch.zeros(env.num_envs, dtype=torch.long, device=forces.device)

    if env.common_step_counter < enable_after_steps:
        contact_steps.zero_()
        setattr(env, counter_name, contact_steps)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=forces.device)

    allowed_contact_steps = minimum_contact_steps
    if initial_contact_steps is not None and ramp_steps > 0:
        progress = min(
            (env.common_step_counter - enable_after_steps) / ramp_steps,
            1.0,
        )
        allowed_contact_steps = round(
            initial_contact_steps
            + progress * (minimum_contact_steps - initial_contact_steps)
        )

    contact_steps = torch.where(in_contact, contact_steps + 1, torch.zeros_like(contact_steps))
    setattr(env, counter_name, contact_steps)
    return contact_steps >= allowed_contact_steps


#-----------------------------------------------------------------------------------


# 重心保持在輪軸附近  COM 離左右輪軸越遠，扣分越多
def wheel_under_com_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_wheel_body: str,
    right_wheel_body: str,
    error_scale: float,
    com_body_name: str | None = None,
) -> torch.Tensor:
    """懲罰指定 body 或整機 COM 在水平面上偏離左右輪軸線的正規化距離。"""

    robot: Articulation = env.scene[asset_cfg.name]

    # 真正的各 rigid body COM 世界座標：[num_envs, num_bodies, 3]
    body_com_pos_w = robot.data.body_com_pos_w

    if com_body_name is not None:
        com_body_ids, _ = robot.find_bodies(com_body_name)
        target_com_w = body_com_pos_w[:, com_body_ids[0], :]
    else:
        # PhysX 的 default_mass 在 CPU，body pose 在 CUDA；首次轉移後快取。
        mass_cache_name = "_tancho_default_body_mass"
        body_mass = getattr(env, mass_cache_name, None)
        if body_mass is None or body_mass.device != body_com_pos_w.device:
            body_mass = robot.data.default_mass.to(
                device=body_com_pos_w.device,
                dtype=body_com_pos_w.dtype,
            )
            setattr(env, mass_cache_name, body_mass)
        total_mass = body_mass.sum(dim=1, keepdim=True)
        target_com_w = (
            body_com_pos_w * body_mass.unsqueeze(-1)
        ).sum(dim=1) / total_mass

    left_ids, _ = robot.find_bodies(left_wheel_body)
    right_ids, _ = robot.find_bodies(right_wheel_body)

    left_pos_w = robot.data.body_pos_w[:, left_ids[0], :]
    right_pos_w = robot.data.body_pos_w[:, right_ids[0], :]

    # 左右輪中心與輪軸方向，只取水平 XY
    wheel_mid_xy = 0.5 * (left_pos_w[:, :2] + right_pos_w[:, :2])
    axle_xy = right_pos_w[:, :2] - left_pos_w[:, :2]
    axle_xy = axle_xy / torch.clamp(
        torch.linalg.vector_norm(axle_xy, dim=1, keepdim=True),
        min=1.0e-6,
    )

    # COM 相對於輪軸中心的水平偏移
    delta_xy = target_com_w[:, :2] - wheel_mid_xy

    # 移除沿輪軸方向的分量，只保留前後傾倒方向
    along_axle = torch.sum(delta_xy * axle_xy, dim=1, keepdim=True)
    perpendicular_error = delta_xy - along_axle * axle_xy

    # 用可調的物理容許誤差正規化，避免 m^2 數值過小而被其他 reward 蓋過。
    return torch.sum(torch.square(perpendicular_error), dim=1) / (error_scale**2)



def wide_stance_reward(
    env,
    target_thigh: float,
    target_calf: float,
    sigma_thigh: float,
    sigma_calf: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward robot for staying inside a soft basin around the desired wide stance.

    Expected joint ordering:
        [
            thigh_L,
            calf_L,
            thigh_R,
            calf_R,
        ]

    Reward range:
        0 ~ 1

    At exact target:
        reward = 1

    Unlike a normal L1/L2 posture penalty, this reward saturates near the
    desired pose. Once the robot is sufficiently close to the target posture,
    there is very little incentive to use additional torque just to eliminate
    a tiny position error.
    """

    robot = env.scene[asset_cfg.name]

    q = robot.data.joint_pos[:, asset_cfg.joint_ids]

    target = torch.tensor(
        [
            target_thigh,
            target_calf,
            target_thigh,
            target_calf,
        ],
        dtype=q.dtype,
        device=q.device,
    )

    sigma = torch.tensor(
        [
            sigma_thigh,
            sigma_calf,
            sigma_thigh,
            sigma_calf,
        ],
        dtype=q.dtype,
        device=q.device,
    )

    # Normalized posture error
    error = (q - target) / sigma

    # Gaussian basin.
    #
    # exact target:
    #     reward = 1
    #
    # one-sigma average error:
    #     reward ~= exp(-1) ~= 0.37
    #
    # 避免 PPO 為了最後幾百分之一 rad 一直追加 torque。
    error_sq = torch.mean(torch.square(error), dim=1)

    reward = torch.exp(-error_sq)

    return reward


def joint_pd_torque_saturation(
    env,
    asset_cfg: SceneEntityCfg,
    stiffness: float,
    damping: float,
    effort_limit: float,
    soft_ratio: float = 0.75,
) -> torch.Tensor:
    """Penalize estimated implicit-PD torque demand near actuator saturation.

    This is designed for ImplicitActuatorCfg.

    Instead of penalizing all torque, estimate the torque requested by the
    implicit PD controller:

        tau_pd =
            Kp * (q_target - q)
            +
            Kd * (qd_target - qd)

    The penalty starts only when:

        abs(tau_pd) / effort_limit > soft_ratio

    Example with:
        effort_limit = 12.5 Nm
        soft_ratio   = 0.75

    No penalty below:
        9.375 Nm

    Penalty progressively increases from:
        9.375 Nm -> 12.5 Nm

    Demand beyond the actuator limit receives an even larger penalty.

    Return value:
        >= 0

    RewardTerm weight should therefore be negative.
    """

    robot = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids

    # ------------------------------------------------------------------
    # Current state
    # ------------------------------------------------------------------

    q = robot.data.joint_pos[:, joint_ids]
    qd = robot.data.joint_vel[:, joint_ids]

    # ------------------------------------------------------------------
    # Controller targets
    # ------------------------------------------------------------------

    q_target_all = robot.data.joint_pos_target

    if q_target_all is None:
        # 理論上 position-controlled leg 不應該進這裡。
        # 如果真的沒有 target，就不要製造假的 penalty。
        return torch.zeros(
            q.shape[0],
            dtype=q.dtype,
            device=q.device,
        )

    q_target = q_target_all[:, joint_ids]

    qd_target_all = robot.data.joint_vel_target

    if qd_target_all is None:
        qd_target = torch.zeros_like(qd)
    else:
        qd_target = qd_target_all[:, joint_ids]

    # ------------------------------------------------------------------
    # Estimated raw PD demand
    # ------------------------------------------------------------------
    #
    # 我們故意使用 clipping 前的 demand。
    #
    # 假設：
    #
    #   tau_pd = 18 Nm
    #   limit  = 12.5 Nm
    #
    # 真正 actuator 最後只能給 12.5 Nm，
    # 但 reward 必須知道 policy 正在要求 18 Nm。
    # ------------------------------------------------------------------

    position_error = q_target - q
    velocity_error = qd_target - qd

    tau_pd = (
        stiffness * position_error
        + damping * velocity_error
    )

    # ------------------------------------------------------------------
    # Normalize torque demand
    # ------------------------------------------------------------------

    torque_ratio = torch.abs(tau_pd) / effort_limit

    # Example:
    #
    # soft_ratio = 0.75
    #
    # ratio 0.50 -> 0
    # ratio 0.75 -> 0
    # ratio 0.875 -> 0.5
    # ratio 1.00 -> 1
    # ratio 1.25 -> 2
    #
    normalized_excess = (
        torque_ratio - soft_ratio
    ) / max(1.0 - soft_ratio, 1.0e-6)

    normalized_excess = torch.clamp(
        normalized_excess,
        min=0.0,
        max=3.0,
    )

    # Squared penalty.
    #
    # 75% torque -> 0
    # 87.5%      -> 0.25
    # 100%       -> 1
    # 125%       -> 4
    #
    # 使用 mean 而不是 sum，
    # 避免單純因為有四顆腿 joint 就把 reward scale 放大四倍。
    penalty = torch.mean(
        torch.square(normalized_excess),
        dim=1,
    )

    return penalty

def wheel_speed_soft_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    free_speed: float = 5.0,
    soft_speed: float = 20.0,
) -> torch.Tensor:
    """輪速低於 free_speed 不扣分，超過後漸進懲罰。"""

    robot = env.scene[asset_cfg.name]

    qd = torch.abs(
        robot.data.joint_vel[:, asset_cfg.joint_ids]
    )

    excess = (
        qd - free_speed
    ) / max(soft_speed - free_speed, 1.0e-6)

    excess = torch.clamp(
        excess,
        min=0.0,
        max=3.0,
    )

    return torch.mean(excess**2, dim=1)





# ---------------------------------------------------------------------------
# Curriculum modify_fn (不能用 lambda，Isaac Lab config 需要可序列化)
# ---------------------------------------------------------------------------

# 開啟移動訓練  到指定訓練步數後，開始讓部分環境接受移動指令
def curriculum_enable_velocity(env, env_ids, old_value, num_steps: int, target: float):
    """達到 num_steps 後，將 rel_standing_envs 從 1.0 降到 target。"""
    from isaaclab.envs.mdp.curriculums import modify_env_param
    if env.common_step_counter >= num_steps:
        return target
    return modify_env_param.NO_CHANGE


# 改變訓練參數  到指定步數後，把某個 Reward 或參數改成新值
def curriculum_set_after_steps(env, env_ids, old_value, num_steps: int, target: float):
    """達到指定步數後，將任意可修改參數切換為 target。"""
    from isaaclab.envs.mdp.curriculums import modify_env_param
    if env.common_step_counter >= num_steps:
        return target
    return modify_env_param.NO_CHANGE


# 開啟外力干擾  到指定步數後，開始對機器人施加推力測試穩定性
def curriculum_enable_push(env, env_ids, old_value, num_steps: int, velocity_range: dict):
    """達到 num_steps 後，開啟推力干擾速度範圍。"""
    from isaaclab.envs.mdp.curriculums import modify_env_param
    if env.common_step_counter >= num_steps:
        return velocity_range
    return modify_env_param.NO_CHANGE
