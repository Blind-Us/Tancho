"""Element 2 - Action.

* Wheels (DM-H3510): velocity target, action +/-1 -> +/-52.36 rad/s (rated
  500 rpm).  The motor's velocity loop (Kd=4) turns it into torque inside the
  +/-0.45 N*m peak.  No direct torque action.
* Legs (DM-J4310, 6-DOF stages only): position target = nominal pose +
  0.25 rad * action, tracked by the Kp=20 / Kd=0.2 position loop.  With
  ``clip_actions=1`` each leg joint stays within +/-0.25 rad of the nominal
  pose, so the policy cannot fold the legs into a crouch.

Smoothness is a reward (``action_rate``), not a smaller scale.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from .scene import CLIMB_LEG_ACTION_SCALE_RAD, LEG_ACTION_SCALE_RAD, LEG_JOINTS, WHEEL_JOINTS, WHEEL_RATED_SPEED_RAD_S

_WHEEL_VEL = mdp.JointVelocityActionCfg(
    asset_name="robot",
    joint_names=WHEEL_JOINTS,
    scale=WHEEL_RATED_SPEED_RAD_S,
    use_default_offset=True,
    preserve_order=True,
)


@configclass
class WheelOnlyActionsCfg:
    """2 dims: left/right wheel speed."""

    wheel_vel = _WHEEL_VEL


@configclass
class FullActionsCfg:
    """6 dims: thigh_L, calf_L, thigh_R, calf_R position offsets, then the two wheel speeds."""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=LEG_ACTION_SCALE_RAD,
        use_default_offset=True,
        preserve_order=True,
    )
    wheel_vel = _WHEEL_VEL


@configclass
class ClimbActionsCfg(FullActionsCfg):
    """Same layout; legs reach nominal +/-0.6 rad so a wheel can be lifted onto a step."""

    def __post_init__(self):
        self.leg_pos.scale = CLIMB_LEG_ACTION_SCALE_RAD
