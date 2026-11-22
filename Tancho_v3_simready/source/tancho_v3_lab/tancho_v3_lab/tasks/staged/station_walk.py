"""WASD driving on top of the station-keeping policy (RL v2 stage 2, user 2026-11-22).

The user drove release r16_400 in teleop: any WASD / speed-3 command tipped it past
15 deg, because every stage-1 policy saw a zero velocity command only.  Here:

* Commands: vx +/-0.6 m/s, yaw rate +/-1 rad/s, 25% of envs standing, resampled every
  3-6 s; the range grows from 30% to 100% over the first 400 iterations.
* Hold point: while the command is non-zero, the hold point follows the robot (wheel
  angle and planar position); when the command returns to zero, it stays where the robot
  was.  The wheel odometry input (same two slots as before) is measured from the hold
  point, so on the Pi: re-zero the odometry whenever the stick is off-centre.
* Rewards: velocity tracking as the walk stage (std 0.5, weights 2 / 1); position hold,
  wheel-speed L1 and the capture point only count while the command is zero.
* Everything else as StationSquatKd1Lat (wheel Kd 1, 10-20 ms latency, LT / RT lift,
  crouch, no hop).  Same 34 inputs.  From release r16_400.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .observations import WHEELS
from .scene import WHEEL_RADIUS_M
from .station import ODOM_CLIP_M
from .station_skills import (
    StationSkillsObservationsCfg,
    TanchoV3StationSquatKd1LatEnvCfg,
    TanchoV3StationSquatKd1ComPlayEnvCfg,
)

MOVE_EPS = 0.05


# -- hold point ------------------------------------------------------------------------
def _update_hold(env: ManagerBasedRLEnv) -> None:
    """Once per step: the hold point follows the robot while commanded to move / after a reset."""
    step = int(env.common_step_counter)
    robot = env.scene["robot"]
    if not hasattr(env, "_hold_q"):
        env._wheel_ids = robot.find_joints(["joint_wheel_L", "joint_wheel_R"], preserve_order=True)[0]
        env._hold_q = robot.data.joint_pos[:, env._wheel_ids].clone()
        env._hold_xy = robot.data.root_pos_w[:, :2].clone()
        env._hold_step = -1
    if env._hold_step == step:
        return
    env._hold_step = step
    cmd = env.command_manager.get_command("base_velocity")
    follow = (cmd[:, 0].abs() > MOVE_EPS) | (cmd[:, 2].abs() > MOVE_EPS) | (env.episode_length_buf == 0)
    q = robot.data.joint_pos[:, env._wheel_ids]
    env._hold_q[follow] = q[follow]
    env._hold_xy[follow] = robot.data.root_pos_w[follow, :2]


def _standing(env: ManagerBasedRLEnv) -> torch.Tensor:
    cmd = env.command_manager.get_command("base_velocity")
    return ((cmd[:, 0].abs() <= MOVE_EPS) & (cmd[:, 2].abs() <= MOVE_EPS)).float()


def wheel_odometry_hold(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = WHEELS) -> torch.Tensor:
    """(n, 2) wheel travel since the hold point, metres, clipped (0 while moving)."""
    _update_hold(env)
    q = env.scene["robot"].data.joint_pos[:, asset_cfg.joint_ids]
    return ((q - env._hold_q) * WHEEL_RADIUS_M).clamp(-ODOM_CLIP_M, ODOM_CLIP_M)


def hold_displacement_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n, 2) planar root displacement from the hold point, yaw frame (privileged)."""
    _update_hold(env)
    robot = env.scene["robot"]
    d = robot.data.root_pos_w[:, :2] - env._hold_xy
    w, x, y, z = robot.data.root_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=-1).clamp(-1.0, 1.0)


# -- rewards (standing-gated) --------------------------------------------------------------
def _hold_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    _update_hold(env)
    return (env.scene["robot"].data.root_pos_w[:, :2] - env._hold_xy).norm(dim=-1)


def hold_exp_standing(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    return _standing(env) * torch.exp(-_hold_dist(env).square() / std**2)


def hold_l1_standing(env: ManagerBasedRLEnv, max_dist: float = 1.0) -> torch.Tensor:
    return _standing(env) * _hold_dist(env).clamp(max=max_dist)


def joint_vel_l1_standing(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _standing(env) * mdp.joint_vel_l1(env, asset_cfg)


# -- command range curriculum -------------------------------------------------------------
def command_range_curriculum(env: ManagerBasedRLEnv, env_ids, start: float = 0.3, ramp_iters: int = 400,
                             vx_max: float = 0.6, wz_max: float = 1.0, steps_per_iter: int = 24) -> float:
    k = min(1.0, start + (1.0 - start) * env.common_step_counter / steps_per_iter / ramp_iters)
    r = env.command_manager.get_term("base_velocity").cfg.ranges
    r.lin_vel_x = (-vx_max * k, vx_max * k)
    r.ang_vel_z = (-wz_max * k, wz_max * k)
    return k


# -- config --------------------------------------------------------------------------------
@configclass
class StationWalkObservationsCfg(StationSkillsObservationsCfg):
    def __post_init__(self):
        self.policy.wheel_odom = ObsTerm(func=wheel_odometry_hold, params={"asset_cfg": WHEELS}, scale=2.0)
        self.critic.wheel_odom = ObsTerm(func=wheel_odometry_hold, params={"asset_cfg": WHEELS}, scale=2.0)
        self.critic.root_disp = ObsTerm(func=hold_displacement_b, scale=2.0)


@configclass
class StationWalkCurriculumCfg:
    command_range = CurrTerm(func=command_range_curriculum)


@configclass
class TanchoV3StationWalkEnvCfg(TanchoV3StationSquatKd1LatEnvCfg):
    observations: StationWalkObservationsCfg = StationWalkObservationsCfg()
    curriculum: StationWalkCurriculumCfg = StationWalkCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = StationWalkCurriculumCfg()
        c = self.commands.base_velocity
        c.rel_standing_envs = 0.25
        c.resampling_time_range = (3.0, 6.0)
        c.ranges.lin_vel_x = (-0.18, 0.18)
        c.ranges.ang_vel_z = (-0.3, 0.3)
        r = self.rewards
        r.lin_vel_xy.weight, r.lin_vel_xy.params["std"] = 2.0, 0.5
        r.ang_vel_z.weight, r.ang_vel_z.params["std"] = 1.0, 0.5
        r.pos_hold.func = hold_exp_standing
        r.pos_l1.func = hold_l1_standing
        r.wheel_vel_l1.func = joint_vel_l1_standing
        r.capture_point.weight = 0.0
        if getattr(r, "pos_hold_wide", None) is not None:
            r.pos_hold_wide.func = hold_exp_standing


@configclass
class TanchoV3StationWalkPlayEnvCfg(TanchoV3StationSquatKd1ComPlayEnvCfg):
    observations: StationWalkObservationsCfg = StationWalkObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (-0.6, 0.6)
        c.ranges.ang_vel_z = (-1.0, 1.0)
