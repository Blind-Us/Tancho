"""Station keeping + left / right leg lift + crouch in one policy (user 2026-11-19).

Station r2 (``TanchoV3-StationV2``, run 2026-11-03_01-16-47) stands with 1.8 cm of drift
in 60 s and takes a 45 N push.  The leg skills lived in other policies: LT / RT lift and
the LT+RT hop in hop r3 (``hop-push-r3-stable``), hold-to-crouch (charge) in HopCharge.
Here they are trained on top of r2:

* LT / RT: lift that leg (v1 reference, played by the Pi at deployment, guidance 1);
  LT+RT is the standard hop (trained, not deployed for now).
* charge (teleop Shift / J): hold to crouch both legs, release to hop (as HopCharge).
* The zero velocity command, position hold, skid cost and DR of Station r2 stay.
* Posture costs / tilt limit open in the press window as in ClimbHop; leg acc / torque /
  action-rate at 10% while pressed (HopPush).

Actor 34 = 29 HopPush + 2 wheel odometry (Station) + 3 charge state (HopCharge).
Start: r2 ``model_final.pt`` padded by 3 actor / 3 critic inputs, leg std reset to 0.1.
"""

from __future__ import annotations

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import charge, climb
from .climb import ClimbTriggerCommandCfg, bad_orientation_gated
from .env_cfg import HopCommandsCfg, _to_play
from .scene import FAILURE_TILT_RAD
from .station import StationObservationsCfg, StationV2RewardsCfg, TanchoV3StationV2EnvCfg, TanchoV3StationPPORunnerCfg


@configclass
class StationSkillsCommandsCfg(HopCommandsCfg):
    charge = charge.ChargeCommandCfg()


@configclass
class StationSkillsObservationsCfg(StationObservationsCfg):
    @configclass
    class PolicyCfg(StationObservationsCfg.PolicyCfg):
        """34 dims: Station's 31 + charge held, crouch level, time since release."""

        charge_state = ObsTerm(func=charge.charge_obs)

    @configclass
    class CriticCfg(StationObservationsCfg.CriticCfg):
        charge_state = ObsTerm(func=charge.charge_obs)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class StationSkillsRewardsCfg(StationV2RewardsCfg):
    charge_reference = RewTerm(func=charge.charge_tracking, weight=3.0)
    charge_rise = RewTerm(func=charge.charge_flight_rise, weight=200.0)

    def __post_init__(self):
        super().__post_init__()
        # As ClimbRewardsCfg / hop r3: imitate the lift, pay for hop height.
        self.leg_reference.weight = 3.0
        self.leg_reference.params = {"ref_style": "v1"}
        self.hop_rise.weight = 200.0
        self.hop_rise.params = {"max_rise": 0.06}
        # Posture costs pause in the press window (ClimbRewardsCfg).
        self.mirror.func = climb.mirror_leg_l2_gated
        self.upright.func = climb.flat_orientation_l2_gated
        self.ang_vel_xy.func = climb.ang_vel_xy_l2_gated
        self.leg_deviation.func = climb.joint_deviation_l1_gated
        self.vertical_vel.func = climb.lin_vel_z_l2_gated
        # Fast strong legs while pressed (HopPush).
        self.leg_acc.func = climb.joint_acc_l2_relaxed
        self.leg_torque.func = climb.joint_torques_l2_relaxed
        self.action_rate.func = climb.action_rate_l2_relaxed


@configclass
class TanchoV3StationSkillsEnvCfg(TanchoV3StationV2EnvCfg):
    commands: StationSkillsCommandsCfg = StationSkillsCommandsCfg()
    observations: StationSkillsObservationsCfg = StationSkillsObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards = StationSkillsRewardsCfg()
        # Stability zeroed the presses; back to the hop r3 press schedule.
        hop = ClimbTriggerCommandCfg(random_press_prob=0.8, auto_prob=0.0, burst_prob=0.5, both_skew_s=0.06)
        self.commands.climb.random_press_prob = hop.random_press_prob
        self.commands.climb.auto_prob = hop.auto_prob
        self.commands.climb.burst_prob = hop.burst_prob
        self.commands.climb.both_skew_s = hop.both_skew_s
        # The Pi plays the reference on a press (as hop r3).
        self.actions.leg_pos.guidance_scale = 1.0
        self.actions.leg_pos.ref_style = "v1"
        self.actions.leg_pos.charge_command_name = "charge"
        # Tilt limit 15 deg, 35 deg in the press window (ClimbEnvCfg).
        self.terminations.tilt.func = bad_orientation_gated
        self.terminations.tilt.params = {"limit_angle": FAILURE_TILT_RAD, "limit_angle_lift": math.radians(35.0)}


@configclass
class TanchoV3StationSkillsPlayEnvCfg(TanchoV3StationSkillsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)
        self.events.leg_gains = None
        self.events.wheel_gains = None
        self.events.base_com = None
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.actions.leg_pos.max_delay_substeps = 0
        self.actions.wheel_vel.max_delay_substeps = 0
        # Presses come from the script / keyboard only.
        self.commands.climb.random_press_prob = 0.0
        self.commands.climb.burst_prob = 0.0
        self.commands.climb.both_skew_s = 0.0
        self.commands.charge.press_prob = 0.0


@configclass
class TanchoV3StationSkillsPPORunnerCfg(TanchoV3StationPPORunnerCfg):
    experiment_name = "tancho_v3_station_skills"


# -- no hops (user 2026-11-19: hops are not deployed) --------------------------------
# StationSkills from HopCharge (run 2026-11-19_19-49-13) plateaued at 31-35% time-outs
# from iteration 400 to 600 (gate: 50%), mostly tilt.  The LT+RT hop and the charged
# hop on release are the hardest presses and are not deployed; here random presses
# are L or R only and a released crouch stands back up over 0.28 s.
@configclass
class TanchoV3StationSquatEnvCfg(TanchoV3StationSkillsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.climb.allow_both = False
        self.commands.charge.jump = False
        self.rewards.hop_rise.weight = 0.0
        self.rewards.charge_rise.weight = 0.0


@configclass
class TanchoV3StationSquatPlayEnvCfg(TanchoV3StationSkillsPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.charge.jump = False


# -- round 2 of StationSquat: hold position harder ------------------------------------
# Run 2026-11-19_20-17-17: no falls anywhere (Isaac stand / push / LT / RT / crouch,
# MuJoCo too), lifts 32-43 mm, full crouch and back.  But zero-command drift 14.5 cm in
# 60 s (Station r2: 1.8 cm, gate 5 cm): the press costs dilute the position terms.
@configclass
class TanchoV3StationSquatHoldEnvCfg(TanchoV3StationSquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 5 cm away now costs 0.25/s from pos_l1 (was 0.1) and halves pos_hold's 4/s.
        self.rewards.pos_hold.weight = 4.0
        self.rewards.pos_l1.weight = -5.0


@configclass
class TanchoV3StationSquatHoldPlayEnvCfg(TanchoV3StationSquatPlayEnvCfg):
    pass


# -- MuJoCo tire skid: softer wheel velocity loop ------------------------------------
# StationSquatHold (run 2026-11-19_21-25-01) passes every Isaac check (60 s drift 1.4 cm,
# push, LT / RT / crouch) and never falls in MuJoCo, but drifts 1.7 m there in 60 s:
# the wheel torque sits at its limit and the tires skid (see station.py round 4).
# Wheel Kd 4 -> 1 as StationV4.  DEPLOYMENT CHANGE: DM-H3510 MIT-mode kd = 1.0.
from .station import WHEEL_KD_V4  # noqa: E402


def _soft_wheels(cfg):
    cfg.scene.robot = cfg.scene.robot.replace(actuators={**cfg.scene.robot.actuators})
    cfg.scene.robot.actuators["wheels"] = cfg.scene.robot.actuators["wheels"].replace(damping=WHEEL_KD_V4)


@configclass
class TanchoV3StationSquatKd1EnvCfg(TanchoV3StationSquatHoldEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _soft_wheels(self)


@configclass
class TanchoV3StationSquatKd1PlayEnvCfg(TanchoV3StationSquatHoldPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _soft_wheels(self)
