"""Element 3 - Reward.

Weights are per second; the reward manager multiplies by step_dt = 0.02.
Isaac Lab built-in terms are used wherever one exists.  The only Tancho
specific terms (capture point, left/right leg mirror) come from
``direct/tancho_v3/custom_rewards.py`` and are used by the 6-DOF stages only.

Stage differences:
* wheel-only stand: survival, upright, stand still, wheel effort/smoothness.
* 6-DOF stand: + vertical bounce, capture point over the axle, leg mirror,
  leg torque/acceleration/limits.
* 6-DOF walk: same costs, tracking terms become the main objective
  (wider kernel, higher weight) and the zero-wheel-speed pull is removed.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..direct.tancho_v3 import custom_rewards as cr
from . import climb
from .observations import LEGS, WHEELS
from .scene import LEG_EFFORT_LIMIT_NM, WHEEL_EFFORT_LIMIT_NM, WHEEL_RADIUS_M

_COMMAND = "base_velocity"


@configclass
class WheelOnlyRewardsCfg:
    # Survival: +20 for a full 20 s episode; a fall costs an extra 200*0.02 = 4.
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # Upright: |g_xy|^2 = sin^2(tilt); 5 deg costs 0.076/s.
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    # Pitch/roll rate damping.
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # Track the commanded planar speed and yaw rate (zero while standing).
    # std=0.25 m/s makes 0.1 m/s cost 15% of the term.
    lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": _COMMAND, "std": 0.25})
    ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": _COMMAND, "std": 0.25})
    # L1 keeps a constant pull toward exactly zero wheel speed, which the exp
    # term above does not (its gradient vanishes at 0).  Removes slow drift.
    # At -0.02 the wheel-only policy crept at a steady 0.74 rad/s (2.7 cm/s,
    # 53 cm in 20 s) for 0.03/s; -0.2 makes that creep cost 0.3/s.
    wheel_vel_l1 = RewTerm(func=mdp.joint_vel_l1, weight=-0.2, params={"asset_cfg": WHEELS})
    # Effort normalized by the peak: both wheels saturated costs 0.02/s.
    wheel_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.01 / WHEEL_EFFORT_LIMIT_NM**2, params={"asset_cfg": WHEELS})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    wheel_vel_limit = RewTerm(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.9, "asset_cfg": WHEELS})


@configclass
class FullStandRewardsCfg(WheelOnlyRewardsCfg):
    # Legs can now pump the body up and down.
    vertical_vel = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # With movable legs body pitch no longer fixes where the COM is, so balance
    # is also stated directly: whole-body capture point over the wheel axle
    # (reduces to COM over axle at rest).  50 mm error costs 0.5/s.
    capture_point = RewTerm(
        func=cr.wheel_capture_point_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "left_wheel_body": "wheel_L",
            "right_wheel_body": "wheel_R",
            "error_scale": 0.050,
            "wheel_radius": WHEEL_RADIUS_M,
        },
    )
    # Tancho is mirror-symmetric; equal pose means equal q on both sides.
    # mirror_leg_l2 = 0.5*(dthigh^2 + dcalf^2): 0.1 rad on both pairs costs 0.2/s.
    # (-0.5 cost only 0.005/s there and left a 0.06-0.11 rad split after a push.)
    mirror = RewTerm(func=cr.mirror_leg_l2, weight=-20.0)
    # Return to the nominal leg pose after a disturbance: 0.1 rad on all four joints costs 0.4/s.
    leg_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-1.0, params={"asset_cfg": LEGS})
    # Same normalization as the wheels: 100% leg torque costs the same per joint.
    leg_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.01 / LEG_EFFORT_LIMIT_NM**2, params={"asset_cfg": LEGS})
    leg_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": LEGS})
    leg_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": LEGS})


@configclass
class FullWalkRewardsCfg(FullStandRewardsCfg):
    def __post_init__(self):
        # Tracking is the task now: std 0.5 so a fresh command still has gradient.
        self.lin_vel_xy.weight = 2.0
        self.lin_vel_xy.params["std"] = 0.5
        self.ang_vel_z.weight = 1.0
        self.ang_vel_z.params["std"] = 0.5
        # Pulling the wheels toward zero speed fights every non-zero command.
        self.wheel_vel_l1 = None
        # The capture point leads the COM by v/sqrt(g/h) (4.3 cm at 0.3 m/s), so in
        # steady rolling it sits ahead of the axle and the term charges for speed:
        # the first walk policy held only ~60% of the commanded vx.  Kept at 0 so
        # the log columns stay comparable.
        self.capture_point.weight = 0.0


@configclass
class ClimbRewardsCfg(FullWalkRewardsCfg):
    """Walk rewards + lift on trigger; posture terms pause while a trigger is (recently) pressed."""

    # Pressed side's tire clearance, 5 cm = full credit: a 0.4 s full lift earns 0.8.
    wheel_lift = RewTerm(func=climb.wheel_lift_on_trigger, weight=2.0)

    def __post_init__(self):
        super().__post_init__()
        self.mirror.func = climb.mirror_leg_l2_gated
        self.leg_deviation.func = climb.joint_deviation_l1_gated
        self.vertical_vel.func = climb.lin_vel_z_l2_gated
