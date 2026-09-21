"""Minimal Tancho V3 terms without an equivalent Isaac Lab MDP implementation."""

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def mirror_leg_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize left/right thigh and calf position asymmetry."""

    robot: Articulation = env.scene["robot"]
    joint_ids, _ = robot.find_joints(
        ["joint_thigh_L", "joint_thigh_R", "joint_calf_L", "joint_calf_R"],
        preserve_order=True,
    )
    joint_pos = robot.data.joint_pos[:, joint_ids]
    return 0.5 * (
        (joint_pos[:, 0] - joint_pos[:, 1]).square()
        + (joint_pos[:, 2] - joint_pos[:, 3]).square()
    )


def wheel_under_com_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_wheel_body: str,
    right_wheel_body: str,
    error_scale: float,
    com_body_name: str | None = None,
) -> torch.Tensor:
    """Return normalized horizontal distance from COM to the wheel-axle line."""

    robot: Articulation = env.scene[asset_cfg.name]
    body_com_pos_w = robot.data.body_com_pos_w

    if com_body_name is not None:
        com_body_ids, _ = robot.find_bodies(com_body_name)
        target_com_w = body_com_pos_w[:, com_body_ids[0], :]
    else:
        mass_cache_name = "_tancho_default_body_mass"
        body_mass = getattr(env, mass_cache_name, None)
        if body_mass is None or body_mass.device != body_com_pos_w.device:
            body_mass = robot.data.default_mass.to(
                device=body_com_pos_w.device,
                dtype=body_com_pos_w.dtype,
            )
            setattr(env, mass_cache_name, body_mass)
        target_com_w = (body_com_pos_w * body_mass.unsqueeze(-1)).sum(dim=1) / body_mass.sum(
            dim=1, keepdim=True
        )

    left_ids, _ = robot.find_bodies(left_wheel_body)
    right_ids, _ = robot.find_bodies(right_wheel_body)
    left_pos_w = robot.data.body_pos_w[:, left_ids[0], :]
    right_pos_w = robot.data.body_pos_w[:, right_ids[0], :]

    wheel_mid_xy = 0.5 * (left_pos_w[:, :2] + right_pos_w[:, :2])
    axle_xy = right_pos_w[:, :2] - left_pos_w[:, :2]
    axle_xy = axle_xy / torch.clamp(
        torch.linalg.vector_norm(axle_xy, dim=1, keepdim=True), min=1.0e-6
    )

    delta_xy = target_com_w[:, :2] - wheel_mid_xy
    along_axle = torch.sum(delta_xy * axle_xy, dim=1, keepdim=True)
    perpendicular_error = delta_xy - along_axle * axle_xy
    return torch.sum(torch.square(perpendicular_error), dim=1) / (error_scale**2)


def curriculum_enable_push(env, env_ids, old_value, num_steps: int, velocity_range: dict):
    """Enable the configured push range after the requested training step."""

    from isaaclab.envs.mdp.curriculums import modify_env_param

    if env.common_step_counter >= num_steps:
        return velocity_range
    return modify_env_param.NO_CHANGE
