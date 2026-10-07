"""Fall recovery (``TanchoV3-Recover``): get back up from a lean or from lying down.

Same robot, observation (29 dims) and action (legs nominal +/-0.6 rad, wheels)
as the Climb / ClimbHop policies, so a ClimbHop checkpoint is the starting point
and a recovered robot keeps the lift / hop skill.

* Reset: the usual upright reset, then the body is pitched by up to
  ``env._recover_max_tilt`` (either way, a little roll) and dropped from 3 cm, so
  a large tilt lands on the ground as a fall.
* No tilt / body-contact termination: the episode (10 s) runs whatever happens.
* Curriculum: the maximum start tilt grows by ``step`` rad once ``promote`` of the
  episodes that reach the time-out end upright, from 20 deg up to 90 deg.
* Reward: the walk / climb terms (upright, capture point ... already pull toward
  standing) plus a bonus for being up (tilt < 15 deg, root near standing height)
  and a cost on body / thigh / calf ground contact.
"""

from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from .scene import FAILURE_TILT_RAD, FULL_ROOT_HEIGHT_M

START_TILT_RAD = math.radians(20.0)
MAX_TILT_RAD = math.radians(90.0)


def _max_tilt(env) -> float:
    if not hasattr(env, "_recover_max_tilt"):
        env._recover_max_tilt = START_TILT_RAD
        env._recover_success = 0.0
    return env._recover_max_tilt


def reset_fallen(env, env_ids, roll_range: float = 0.15, drop_m: float = 0.03, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Pitch the freshly reset root by U(-max_tilt, max_tilt) and lift it ``drop_m``."""
    robot = env.scene[asset_cfg.name]
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    k = len(ids)
    tilt = _max_tilt(env)
    pitch = (torch.rand(k, device=env.device) * 2.0 - 1.0) * tilt
    roll = (torch.rand(k, device=env.device) * 2.0 - 1.0) * roll_range
    state = robot.data.root_state_w[ids].clone()
    dq = quat_from_euler_xyz(roll, pitch, torch.zeros_like(pitch))
    state[:, 3:7] = quat_mul(state[:, 3:7], dq)
    state[:, 2] += drop_m
    robot.write_root_pose_to_sim(state[:, :7], env_ids=ids)


def recover_curriculum(env, env_ids, promote: float = 0.8, step: float = math.radians(10.0), dwell_steps: int = 1200, rate_per_env: float = 0.002) -> float:
    """Raise the start tilt when most time-outs end upright.

    The upright share is an EMA weighted by the number of finished episodes (resets
    are spread over all steps), and a level must last ``dwell_steps`` (50 PPO
    iterations) before the next one - a per-reset EMA promoted 20 -> 90 deg within
    60 iterations of run 1."""
    tilt = _max_tilt(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env.common_step_counter == 0 or len(ids) == 0:
        return tilt
    if not hasattr(env, "_recover_level_step"):
        env._recover_level_step = env.common_step_counter
    timed_out = env.termination_manager.time_outs[ids]
    if timed_out.any():
        g = env.scene["robot"].data.projected_gravity_b[ids[timed_out]]
        up = torch.acos((-g[:, 2]).clamp(-1.0, 1.0)) < FAILURE_TILT_RAD
        a = min(1.0, rate_per_env * len(up))
        env._recover_success = (1.0 - a) * env._recover_success + a * up.float().mean().item()
        settled = env.common_step_counter - env._recover_level_step >= dwell_steps
        if settled and env._recover_success > promote and tilt < MAX_TILT_RAD:
            env._recover_max_tilt = min(MAX_TILT_RAD, tilt + step)
            env._recover_level_step = env.common_step_counter
    return env._recover_max_tilt


def recover_success(env, env_ids) -> float:
    """Logged: running share of time-outs that end upright."""
    _max_tilt(env)
    return env._recover_success


def upright_cos(env) -> torch.Tensor:
    """cos(tilt) in [-1, 1]: unlike sin^2 it still has a slope when lying flat."""
    return -env.scene["robot"].data.projected_gravity_b[:, 2]


def fixed_start_tilt(env, env_ids, tilt: float = MAX_TILT_RAD) -> float:
    """Curriculum stand-in: always start within +/- ``tilt``."""
    _max_tilt(env)
    env._recover_max_tilt = tilt
    return tilt


def is_up(env, height_ratio: float = 0.85) -> torch.Tensor:
    """1 when the body is within 15 deg of vertical and the root is near standing height."""
    robot = env.scene["robot"]
    g = robot.data.projected_gravity_b
    upright = torch.acos((-g[:, 2]).clamp(-1.0, 1.0)) < FAILURE_TILT_RAD
    ground = env.scene.terrain.env_origins[:, 2] if hasattr(env.scene, "terrain") else 0.0
    tall = (robot.data.root_pos_w[:, 2] - ground) > height_ratio * FULL_ROOT_HEIGHT_M
    return (upright & tall).float()


def body_ground_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 10.0) -> torch.Tensor:
    """Number of the given bodies touching something with more than ``threshold`` N."""
    sensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1).values
    return (f > threshold).float().sum(dim=1)
