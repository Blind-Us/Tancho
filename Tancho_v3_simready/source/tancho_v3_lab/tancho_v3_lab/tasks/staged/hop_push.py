"""Active hop on flat / rough ground (``TanchoV3-HopPush``, user 2026-11-01).

Goal: LT+RT is a real upward jump (the body rises, both tires leave the ground) that
does not fall.  The hop v2 release (ClimbHop, v1 reference) never fell but its hop
only pulled the wheels up; the push hops tried since (v2, v4, v6) fell, and v4/v6 were
trained on stairs.  Here: the ClimbHop task (flat + rough, random LT / RT / LT+RT
presses), started from hop v2 with ``--keep_obs_norm``, and

* reference ``v7``: single presses keep the v1 lift; the hop is a vertical crouch /
  push / tuck (``climb.HOP7_KEYS``), morphed in from the v1 hop over iterations 50-400;
* ``hop_rise``: pays root height above standing while both tires are off the ground;
* leg acceleration / torque / action-rate costs at 10% while a trigger is pressed
  (as StairV2: the push needs fast, strong legs).

Same observation, action scale (+/-0.6 rad) and deployment as hop v2 (Pi plays the
reference table, now the v7 one).
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import climb
from .env_cfg import HopCurriculumCfg, TanchoV3ClimbHopEnvCfg, TanchoV3ClimbHopPlayEnvCfg


@configclass
class HopPushCurriculumCfg(HopCurriculumCfg):
    hop7_morph = CurrTerm(func=climb.hop7_morph)


@configclass
class TanchoV3HopPushEnvCfg(TanchoV3ClimbHopEnvCfg):
    curriculum: HopPushCurriculumCfg = HopPushCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.ref_style = "v7"
        self.rewards.leg_reference.params = {"ref_style": "v7"}
        # A 0.1 s flight 3 cm up earns about 20 * 0.75 * 0.1 = 1.5 per hop.
        self.rewards.hop_rise = RewTerm(func=climb.hop_flight_rise, weight=20.0)
        self.rewards.leg_acc.func = climb.joint_acc_l2_relaxed
        self.rewards.leg_torque.func = climb.joint_torques_l2_relaxed
        self.rewards.action_rate.func = climb.action_rate_l2_relaxed


@configclass
class TanchoV3HopPushPlayEnvCfg(TanchoV3ClimbHopPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.ref_style = "v7"
        # Deploys like hop v2: the Pi plays the reference table on a press.
        self.actions.leg_pos.guidance_scale = 1.0


# -- round 2: keep the v1 hop, pay only for height ---------------------------------
# Round 1 (v7) at iteration 1000: 8/8 hops at 0.3 m/s without a fall, but the root
# rose 8-10 mm in the air, less than hop v2's own v1 hop (12.4 mm, also at rest), and
# the first hop standing still fell.  The v1 hop already has a 0.02 s push; here the
# reference stays v1 and only ``hop_rise`` (+ the relaxed leg costs) asks for more.
@configclass
class TanchoV3HopPushV1EnvCfg(TanchoV3ClimbHopEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.hop_rise = RewTerm(func=climb.hop_flight_rise, weight=20.0)
        self.rewards.leg_acc.func = climb.joint_acc_l2_relaxed
        self.rewards.leg_torque.func = climb.joint_torques_l2_relaxed
        self.rewards.action_rate.func = climb.action_rate_l2_relaxed


@configclass
class TanchoV3HopPushV1PlayEnvCfg(TanchoV3ClimbHopPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.guidance_scale = 1.0


# -- round 3: the height signal was too small to matter ----------------------------
# Round 2 kept hop v2's behaviour exactly (root +12.7 mm vs 12.4, no falls): hop_rise
# earned ~0.035 per episode against a total of ~82, so nothing pushed for height.
# x10 weight, and the cap raised to 6 cm so the gradient does not end at 4 cm.
@configclass
class TanchoV3HopPushV1HiEnvCfg(TanchoV3HopPushV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.hop_rise.weight = 200.0
        self.rewards.hop_rise.params = {"max_rise": 0.06}


@configclass
class TanchoV3HopPushV1HiPlayEnvCfg(TanchoV3HopPushV1PlayEnvCfg):
    pass
