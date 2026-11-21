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
from .station import WHEEL_KD_V4, position_hold_exp  # noqa: E402


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


# Kd 1, run 2026-11-19_23-30-07: MuJoCo now close to Isaac (60 s drift 11.8 cm vs
# 20 cm, wheel torque at the limit 3% of the time, push / LT / RT / crouch no fall),
# but both creep 3 mm/s while rocking at ~0.06 m/s.  pos_hold (std 5 cm) is flat at
# 15-20 cm; a second, wider kernel and a stronger L1 pull keep the gradient there.
@configclass
class TanchoV3StationSquatKd1HoldEnvCfg(TanchoV3StationSquatKd1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 20 cm away costs 2/s (was 1).
        self.rewards.pos_l1.weight = -10.0
        # std 0.2 m: 20 cm still earns 0.37 of 2/s.
        self.rewards.pos_hold_wide = RewTerm(func=position_hold_exp, weight=2.0, params={"std": 0.2})


@configclass
class TanchoV3StationSquatKd1HoldPlayEnvCfg(TanchoV3StationSquatKd1PlayEnvCfg):
    pass


# Goal (user 2026-11-20): MuJoCo 60 s zero-command drift < 5 cm.  Kd1Hold went to 8 cm
# in Isaac but swung out 42 cm and back over ~80 s in MuJoCo (pitch within 2 deg): the
# MuJoCo build of the URDF puts the COM ~8 mm further back over the axle than Isaac's
# zero-pose check, and a slow position loop drifts while re-trimming the lean.
# Body COM x randomization +/-1 -> +/-2 cm; same rewards as Kd1Hold; from the Kd 1 run.
@configclass
class TanchoV3StationSquatKd1ComEnvCfg(TanchoV3StationSquatKd1HoldEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.base_com.params["com_range"] = {"x": (-0.02, 0.02), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}


@configclass
class TanchoV3StationSquatKd1ComPlayEnvCfg(TanchoV3StationSquatKd1HoldPlayEnvCfg):
    pass


# MuJoCo with its default soft contact (solref 0.02) creeps metres and falls even where
# the stiff-contact check (solref 0.005) passes; a rubber tire is not rigid either.
# Compliant ground in training: 5000 N/m (13 N per tire -> ~2.6 mm), 120 N s/m
# (about 0.75 of critical for one tire's share of the mass).
@configclass
class TanchoV3StationSquatKd1SoftEnvCfg(TanchoV3StationSquatKd1ComEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        mat = self.scene.terrain.physics_material
        self.scene.terrain.physics_material = mat.replace(compliant_contact_stiffness=5000.0, compliant_contact_damping=120.0)


@configclass
class TanchoV3StationSquatKd1SoftPlayEnvCfg(TanchoV3StationSquatKd1ComPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        mat = self.scene.terrain.physics_material
        self.scene.terrain.physics_material = mat.replace(compliant_contact_stiffness=5000.0, compliant_contact_damping=120.0)


# Kd1Com / Kd1Soft checkpoints scatter from 2 to 41 cm of MuJoCo 60 s drift (Isaac 5-23
# cm): the position loop is weak everywhere.  With a 0.8 m/s kick every 3-6 s the robot
# is rarely left standing long enough for a slow creep to cost much.  Kicks every 8-14 s,
# so most of each episode is quiet standing / presses; from Kd1Com iteration 300.
@configclass
class TanchoV3StationSquatKd1QuietEnvCfg(TanchoV3StationSquatKd1ComEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (8.0, 14.0)


@configclass
class TanchoV3StationSquatKd1QuietPlayEnvCfg(TanchoV3StationSquatKd1ComPlayEnvCfg):
    pass


# -- the standard hop back in (user 2026-11-20: once drift is under 5 cm, consider hops) --
# Release squat_kd1_r8_300 (Kd1Com iteration 300) holds 2.4 cm in MuJoCo.  Here LT+RT
# presses come back (v1 hop reference, as hop r3) with hop_rise; the charge still stands
# back up on release.  Everything else as Kd1Com; from r8_300.
@configclass
class TanchoV3StationHopKd1EnvCfg(TanchoV3StationSquatKd1ComEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.climb.allow_both = True
        self.rewards.hop_rise.weight = 200.0


@configclass
class TanchoV3StationHopKd1PlayEnvCfg(TanchoV3StationSquatKd1ComPlayEnvCfg):
    pass


# -- terrain and steady loads (user 2026-11-21: stability + terrain first) -----------
# Release r8_300: 0-2 cm bumps hold 3-8 cm without a fall, but on an incline (gravity
# tilted 5-8 deg) it creeps 7-53 cm in Isaac and 0.4-1.8 m in MuJoCo.  Holding still on
# a slope needs a steady wheel torque the policy never had to find on flat ground.
# * Ground: HOP generator tiles (half flat, half 0-2 cm bumps), all levels, no curriculum.
# * A constant horizontal force on the body per episode, body x U(-2, 2) N (2.6 kg on a
#   4.5 deg incline), half of the episodes none: the steady load of a slope / of a COM
#   offset, without tilting the ground.  (+/-4 N, 30% none: time-outs fell from 49% at
#   iteration 50 to 20% at 300, 78% tilt; stopped.)
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from .terrain import HOP_GENERATOR, make_terrain  # noqa: E402


def constant_body_push(env: ManagerBasedEnv, env_ids: torch.Tensor, max_force: float, none_share: float,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link_root"])):
    """Per episode, a constant force along body x on ``asset_cfg``'s body (0 for ``none_share`` of the envs)."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    n = len(env_ids)
    fx = torch.empty(n, device=asset.device).uniform_(-max_force, max_force)
    fx = torch.where(torch.rand(n, device=asset.device) < none_share, torch.zeros_like(fx), fx)
    forces = torch.zeros(n, len(asset_cfg.body_ids), 3, device=asset.device)
    forces[:, :, 0] = fx.unsqueeze(1)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torch.zeros_like(forces), body_ids=asset_cfg.body_ids, env_ids=env_ids
    )


@configclass
class TanchoV3StationSquatKd1TerrainEnvCfg(TanchoV3StationSquatKd1ComEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = make_terrain(HOP_GENERATOR, max_init_level=None)
        self.events.slope_load = EventTerm(
            func=constant_body_push, mode="reset",
            params={"max_force": 2.0, "none_share": 0.5,
                    "asset_cfg": SceneEntityCfg("robot", body_names=["base_link_root"])},
        )


@configclass
class TanchoV3StationSquatKd1TerrainPlayEnvCfg(TanchoV3StationSquatKd1ComPlayEnvCfg):
    """Flat Play tile, no steady load (scripts set terrain / incline themselves)."""
