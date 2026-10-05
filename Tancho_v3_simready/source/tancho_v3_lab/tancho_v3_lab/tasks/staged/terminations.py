"""Element 4 - Termination (failure set).

* ``time_out``: 20 s, a normal end, not a failure.
* ``tilt``: body z-axis more than 15 deg from vertical (the LQR push
  experiment's threshold).  With two coaxial wheels this is |pitch| > 15 deg.
* ``body_contact`` (6-DOF only): base, thigh or calf touching the ground with
  more than 10 N.  The wheel-only asset needs no contact term: inside +/-15 deg
  only the tires reach the ground (``scripts/wheel_only/ground_clearance.py``).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .scene import FAILURE_TILT_RAD


@configclass
class WheelOnlyTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tilt = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": FAILURE_TILT_RAD})


@configclass
class FullTerminationsCfg(WheelOnlyTerminationsCfg):
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link_root", "thigh_.*", "calf_.*"]),
            "threshold": 10.0,
        },
    )
