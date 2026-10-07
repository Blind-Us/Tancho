"""Element 1 - Observation.

Actor (``policy``): only what the hardware measures - IMU attitude and gyro,
joint encoders, the velocity command and the last action.  No base linear
velocity, no base height and no wheel angle (it grows without bound).  Training
adds uniform sensor noise; Play turns it off.

Critic (``critic``): the same signals noise-free plus privileged state the
value function may use during training only (asymmetric actor-critic).

Every stage carries the velocity command, so the 6-DOF standing policy has the
same input layout as the walking policy and can be fine-tuned into it.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .scene import LEG_JOINTS, WHEEL_JOINTS

WHEELS = SceneEntityCfg("robot", joint_names=WHEEL_JOINTS, preserve_order=True)
LEGS = SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)
_IMU = SceneEntityCfg("imu")
_COMMAND = {"command_name": "base_velocity"}


@configclass
class WheelOnlyObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """13 dims: gravity 3, gyro 3, command 3, wheel speed 2, last action 2."""

        imu_projected_gravity = ObsTerm(func=mdp.imu_projected_gravity, params={"asset_cfg": _IMU}, noise=Unoise(n_min=-0.02, n_max=0.02))
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel, params={"asset_cfg": _IMU}, noise=Unoise(n_min=-0.05, n_max=0.05), scale=0.25)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params=_COMMAND)
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": WHEELS}, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged, noise-free state for the value function only (never deployed)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params=_COMMAND)
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": WHEELS}, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class FullObservationsCfg:
    @configclass
    class PolicyCfg(WheelOnlyObservationsCfg.PolicyCfg):
        """25 dims: wheel-only terms + leg position 4 + leg speed 4 (last action is now 6)."""

        leg_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": LEGS}, noise=Unoise(n_min=-0.01, n_max=0.01))
        leg_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": LEGS}, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.05)

    @configclass
    class CriticCfg(WheelOnlyObservationsCfg.CriticCfg):
        base_height = ObsTerm(func=mdp.base_pos_z)
        leg_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": LEGS})
        leg_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": LEGS}, scale=0.05)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


_HEIGHT_SCANNER = SceneEntityCfg("height_scanner")


@configclass
class ClimbObservationsCfg:
    @configclass
    class PolicyCfg(FullObservationsCfg.PolicyCfg):
        """27 dims: the 25 walk dims + LT, RT trigger (0/1) appended last."""

        climb_trigger = ObsTerm(func=mdp.generated_commands, params={"command_name": "climb"})

    @configclass
    class CriticCfg(FullObservationsCfg.CriticCfg):
        climb_trigger = ObsTerm(func=mdp.generated_commands, params={"command_name": "climb"})
        # Root height above each scan point minus the nominal 0.26 m (40 points).
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": _HEIGHT_SCANNER, "offset": 0.26}, clip=(-0.5, 0.5))

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
