"""Station keeping on top of the Stability task (RL v2 stage 1, user 2026-11-03).

The Stability run (2026-11-02_19-43-59) stood 60 s without falling but crept forward
1.85 m: nothing in its 29 inputs measures position, so a slow constant creep looks the
same as standing still, and ``wheel_vel_l1`` only charges ~0.2/s for it.

* Actor + 2 dims: wheel odometry, r * q_wheel per wheel (m, x2, clipped +/-0.5 m).
  The wheel encoders are reset with the episode (default angle 0), so this is the
  travel since the hold point.  Per wheel (not the mean) so no axis-sign convention is
  baked in; the network sees both travel and heading.  The DM-H3510 reports a
  multi-turn position, so this is deployable: zero it when the hold point is set.
* Critic + 2 dims: true planar root displacement from the spawn point, heading frame.
* Rewards: ``pos_hold`` exp(-(d/0.1)^2) and ``pos_l1`` on that displacement;
  ``wheel_vel_l1`` 0.2 -> 0.05 (it fought driving back after a push);
  wheel torque cost x10 and a saturation-time cost (the Kd = 4 wheel loop saturates
  at 0.11 rad/s of error, 77-98% of the time in the 11-02 suite).
* Domain randomization for MuJoCo / hardware: leg Kp/Kd and wheel Kd x U(0.8, 1.2),
  base COM +/-1 cm, 0-20 ms action latency (as ClimbHopDR), friction 0.4-1.2,
  restitution 0-0.1.

Leg action scale, motor gains/limits and 200/50 Hz timing are unchanged; the deployment
contract grows from 29 to 31 inputs (two appended at the end).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .climb import ClimbLegActionDelayedCfg, JointVelocityActionDelayedCfg
from .env_cfg import HopDREventsCfg, _to_play
from .observations import ClimbObservationsCfg, WHEELS
from .scene import CLIMB_LEG_ACTION_SCALE_RAD, LEG_JOINTS, WHEEL_EFFORT_LIMIT_NM, WHEEL_JOINTS, WHEEL_RADIUS_M, WHEEL_RATED_SPEED_RAD_S
from .stability import StabilityRewardsCfg, TanchoV3StabilityEnvCfg, TanchoV3StabilityPPORunnerCfg

ODOM_CLIP_M = 0.5


# -- observation terms -----------------------------------------------------------
def wheel_odometry(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = WHEELS) -> torch.Tensor:
    """(n, 2) wheel travel r * (q - q_default) since reset, metres, clipped."""
    q = mdp.joint_pos_rel(env, asset_cfg)
    return (q * WHEEL_RADIUS_M).clamp(-ODOM_CLIP_M, ODOM_CLIP_M)


def root_displacement_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(n, 2) planar root displacement from the spawn point, in the yaw frame (privileged)."""
    d = _displacement_w(env)
    yaw = _yaw(env)
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=-1).clamp(-1.0, 1.0)


def _yaw(env: ManagerBasedRLEnv) -> torch.Tensor:
    w, x, y, z = env.scene["robot"].data.root_quat_w.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _displacement_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


# -- reward terms ----------------------------------------------------------------
def position_hold_exp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    d2 = _displacement_w(env).square().sum(-1)
    return torch.exp(-d2 / std**2)


def position_hold_l1(env: ManagerBasedRLEnv, max_dist: float = 1.0) -> torch.Tensor:
    return _displacement_w(env).norm(dim=-1).clamp(max=max_dist)


def wheel_saturation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = WHEELS, ratio: float = 0.95) -> torch.Tensor:
    """Number of wheels (0-2) at >= ``ratio`` of the torque limit."""
    tau = env.scene["robot"].data.applied_torque[:, asset_cfg.joint_ids]
    return (tau.abs() >= ratio * WHEEL_EFFORT_LIMIT_NM).float().sum(-1)


def wheel_slip(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["wheel_L", "wheel_R"])) -> torch.Tensor:
    """Sum over wheels of the tire contact-point planar speed (m/s), for tires on the ground.

    v_contact = v_center + w x (0, 0, -r).  Round 1 kept the wheel torque at the 0.45 N m
    limit 86% of the time: 12.5 N of traction against ~8 N of friction (13 N normal x 0.64),
    so the tires skid and the wheel odometry stops measuring travel.
    """
    robot = env.scene["robot"]
    ids = asset_cfg.body_ids
    v = robot.data.body_lin_vel_w[:, ids]
    w = robot.data.body_ang_vel_w[:, ids]
    r = WHEEL_RADIUS_M
    # w x (0, 0, -r) = (-r * w_y, r * w_x, 0)
    vc_x = v[..., 0] - r * w[..., 1]
    vc_y = v[..., 1] + r * w[..., 0]
    grounded = robot.data.body_pos_w[:, ids, 2] < r + 0.005
    return (torch.sqrt(vc_x.square() + vc_y.square()) * grounded).sum(-1)


# -- config ------------------------------------------------------------------------
@configclass
class StationObservationsCfg(ClimbObservationsCfg):
    @configclass
    class PolicyCfg(ClimbObservationsCfg.PolicyCfg):
        """31 dims: the 29 HopPush dims + wheel odometry L, R (x2 -> +/-1 at 0.5 m)."""

        wheel_odom = ObsTerm(func=wheel_odometry, params={"asset_cfg": WHEELS}, scale=2.0)

    @configclass
    class CriticCfg(ClimbObservationsCfg.CriticCfg):
        wheel_odom = ObsTerm(func=wheel_odometry, params={"asset_cfg": WHEELS}, scale=2.0)
        root_disp = ObsTerm(func=root_displacement_b, scale=2.0)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class StationRewardsCfg(StabilityRewardsCfg):
    # Inside 5 cm earns ~0.8/s of the 1/s; 20 cm earns 0.02.
    pos_hold = RewTerm(func=position_hold_exp, weight=1.0, params={"std": 0.1})
    # Gradient everywhere: 0.3 m away costs 0.6/s (the old 1.85 m/60 s creep: ~0.6 m per episode).
    pos_l1 = RewTerm(func=position_hold_l1, weight=-2.0)
    # Both wheels pinned at the torque limit costs 0.4/s.
    wheel_saturation = RewTerm(func=wheel_saturation, weight=-0.2, params={"asset_cfg": WHEELS})

    def __post_init__(self):
        # Driving back after a 0.8 m/s kick at ~5 rad/s per wheel costs 0.5/s here (was 2/s).
        self.wheel_vel_l1.weight = -0.05
        # Both wheels saturated: 0.2/s (was 0.02/s).
        self.wheel_torque.weight = -0.1 / WHEEL_EFFORT_LIMIT_NM**2


@configclass
class StationEventsCfg(HopDREventsCfg):
    def __post_init__(self):
        self.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.physics_material.params["dynamic_friction_range"] = (0.4, 1.2)
        self.physics_material.params["restitution_range"] = (0.0, 0.1)


@configclass
class TanchoV3StationEnvCfg(TanchoV3StabilityEnvCfg):
    observations: StationObservationsCfg = StationObservationsCfg()
    events: StationEventsCfg = StationEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards = StationRewardsCfg()
        guidance, ref_style = self.actions.leg_pos.guidance_scale, self.actions.leg_pos.ref_style
        self.actions.leg_pos = ClimbLegActionDelayedCfg(
            asset_name="robot", joint_names=LEG_JOINTS, scale=CLIMB_LEG_ACTION_SCALE_RAD,
            use_default_offset=True, preserve_order=True, guidance_scale=guidance, ref_style=ref_style,
        )
        self.actions.wheel_vel = JointVelocityActionDelayedCfg(
            asset_name="robot", joint_names=WHEEL_JOINTS, scale=WHEEL_RATED_SPEED_RAD_S,
            use_default_offset=True, preserve_order=True,
        )


@configclass
class TanchoV3StationPlayEnvCfg(TanchoV3StationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)
        # Nominal motors, no latency, no COM offset.
        self.events.leg_gains = None
        self.events.wheel_gains = None
        self.events.base_com = None
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.actions.leg_pos.max_delay_substeps = 0
        self.actions.wheel_vel.max_delay_substeps = 0


@configclass
# -- round 2 ----------------------------------------------------------------------
# Round 1 (run 2026-11-03_00-05-26): Isaac 60 s zero command parked 12-13 cm from the
# start with a +/-1.5 deg pitch limit cycle (Stability: 1.85 m); MuJoCo drifted 2.6 m.
# The wheel odometry obs had std ~0.4 m in training although the root stayed ~0.15 m
# away: the tires skid at the saturated torque.  Here: pay for skidding, tighter hold.
@configclass
class StationV2RewardsCfg(StationRewardsCfg):
    # 0.1 m/s of skid on both wheels costs 0.4/s.
    wheel_slip = RewTerm(func=wheel_slip, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=["wheel_L", "wheel_R"])})

    def __post_init__(self):
        super().__post_init__()
        # 5 cm now earns 0.37 of the term (was 0.78); 12 cm 0.003 (was 0.24).
        self.pos_hold.params["std"] = 0.05
        self.pos_hold.weight = 2.0


@configclass
class TanchoV3StationV2EnvCfg(TanchoV3StationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = StationV2RewardsCfg()


@configclass
class TanchoV3StationV2PlayEnvCfg(TanchoV3StationPlayEnvCfg):
    pass


# -- round 3 ----------------------------------------------------------------------
# Round 2 (run 2026-11-03_01-16-47): Isaac 60 s drift 1.8 cm, push limit 45 N, +/-10 deg
# all recovered.  MuJoCo (1 ms, implicit servos, no-slip pass) still drifts 0.6 m / 20 s:
# the tires skid at 0.15-0.17 m/s on average (Isaac training: ~0.03), the wheel torque
# being pinned at +/-0.45 N m.  PhysX friction holds more than MuJoCo's at the same mu.
# Here: lower friction in training and a 5x skid cost, so traction stays under the limit.
@configclass
class StationV3RewardsCfg(StationV2RewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        # 0.1 m/s of skid on both wheels costs 2/s.
        self.wheel_slip.weight = -10.0


@configclass
class StationV3EventsCfg(StationEventsCfg):
    def __post_init__(self):
        super().__post_init__()
        # Robot material x ground 0.8 (multiply): effective 0.2-0.96 (was 0.32-0.96).
        self.physics_material.params["static_friction_range"] = (0.25, 1.2)
        self.physics_material.params["dynamic_friction_range"] = (0.25, 1.2)


@configclass
class TanchoV3StationV3EnvCfg(TanchoV3StationEnvCfg):
    events: StationV3EventsCfg = StationV3EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards = StationV3RewardsCfg()


@configclass
class TanchoV3StationV3PlayEnvCfg(TanchoV3StationV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)
        self.events.leg_gains = None
        self.events.wheel_gains = None
        self.events.base_com = None
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.actions.leg_pos.max_delay_substeps = 0
        self.actions.wheel_vel.max_delay_substeps = 0


# -- round 4: softer wheel velocity loop --------------------------------------------
# Rounds 1-3 kept the wheel torque at +/-0.45 N m ~80% of the time: with Kd = 4 the
# loop saturates at 0.11 rad/s of error, below the policy's own action resolution
# (std 0.03 x 52 rad/s), so the wheels act as a bang-bang torque switch and 12.5 N of
# traction skids the tires in MuJoCo.  r3's skid cost + low friction made MuJoCo push
# recovery worse.  Here the velocity loop Kd = 4 -> 1 (saturates at 0.45 rad/s), so
# torque follows the action proportionally over a wider band.  DEPLOYMENT CHANGE: the
# DM-H3510 MIT-mode kd must be set to 1.0.  Rewards / friction as round 2.
WHEEL_KD_V4 = 1.0


@configclass
class TanchoV3StationV4EnvCfg(TanchoV3StationV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = self.scene.robot.replace(actuators={**self.scene.robot.actuators})
        self.scene.robot.actuators["wheels"] = self.scene.robot.actuators["wheels"].replace(damping=WHEEL_KD_V4)


@configclass
class TanchoV3StationV4PlayEnvCfg(TanchoV3StationPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = StationV2RewardsCfg()
        self.scene.robot = self.scene.robot.replace(actuators={**self.scene.robot.actuators})
        self.scene.robot.actuators["wheels"] = self.scene.robot.actuators["wheels"].replace(damping=WHEEL_KD_V4)


@configclass
class TanchoV3StationPPORunnerCfg(TanchoV3StabilityPPORunnerCfg):
    experiment_name = "tancho_v3_station"
