import torch
import math
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg



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


def mirror_leg_actions_l1(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """懲罰左右腿 action 不一致。
        action order: thigh_L, thigh_R, calf_L, calf_R
        thigh_L == thigh_R
        calf_L == calf_R
        penalty = 0

    差異越大，回傳值越大。
    FlatRewardsCfg 使用負 weight 將其轉成 penalty。
    """
    actions = env.action_manager.get_term(action_name).raw_actions
    thigh_error = torch.abs(actions[:, 0] - actions[:, 1])
    calf_error = torch.abs(actions[:, 2] - actions[:, 3])

    # 取平均，避免 thigh + calf 直接相加造成數值過大
    return 0.5 * (thigh_error + calf_error)


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
    initial_height: float | None = None,
    minimum_below_steps: int = 1,
    initial_below_steps: int | None = None,
    enable_after_steps: int = 0,
    ramp_steps: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """持續低於漸進安全下限才終止，允許短暫下蹲後主動恢復。"""
    robot: Articulation = env.scene[asset_cfg.name]
    counter_name = "_tancho_below_minimum_height_steps"
    below_steps = getattr(env, counter_name, None)
    if below_steps is None or below_steps.shape[0] != env.num_envs:
        below_steps = torch.zeros(env.num_envs, dtype=torch.long, device=robot.device)

    if env.common_step_counter < enable_after_steps:
        below_steps.zero_()
        setattr(env, counter_name, below_steps)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=robot.device)

    threshold = minimum_height
    allowed_below_steps = minimum_below_steps
    if initial_height is not None and ramp_steps > 0:
        progress = min(
            (env.common_step_counter - enable_after_steps) / ramp_steps,
            1.0,
        )
        threshold = initial_height + progress * (minimum_height - initial_height)
        if initial_below_steps is not None:
            allowed_below_steps = round(
                initial_below_steps
                + progress * (minimum_below_steps - initial_below_steps)
            )

    below = robot.data.root_pos_w[:, 2] < threshold
    below_steps = torch.where(below, below_steps + 1, torch.zeros_like(below_steps))
    setattr(env, counter_name, below_steps)
    return below_steps >= allowed_below_steps


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
