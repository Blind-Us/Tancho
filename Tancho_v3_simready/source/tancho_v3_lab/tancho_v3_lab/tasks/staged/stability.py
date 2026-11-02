"""Stability-first flat training, preserving the 29x6 HopPush deployment contract.

No jump reference, trigger sampling, charge, or locomotion commands are trained.
The actor still sees IMU/encoders/commands/last_action only. Motor gains, limits,
leg action scale 0.6 rad and wheel scale 52.3598776 rad/s stay unchanged.
"""
import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import climb
from .env_cfg import _to_play
from .hop_push import TanchoV3HopPushEnvCfg
from .rewards import FullStandRewardsCfg
from .terminations import FullTerminationsCfg
from .terrain import HOP_GENERATOR, make_terrain, play_generator
from .agents.rsl_rl_ppo_cfg import TanchoV3StandPPORunnerCfg


@configclass
class StabilityRewardsCfg(FullStandRewardsCfg):
    # Keep former jump log columns, with no jump incentive or imitation.
    wheel_lift = RewTerm(func=climb.wheel_lift_on_trigger, weight=0.0)
    wheel_retract = RewTerm(func=climb.wheel_retract_on_trigger, weight=0.0)
    climb_progress = RewTerm(func=climb.climb_progress, weight=0.0)
    leg_reference = RewTerm(func=climb.leg_reference_tracking, weight=0.0)
    hop_rise = RewTerm(func=climb.hop_flight_rise, weight=0.0)


@configclass
class TanchoV3StabilityEnvCfg(TanchoV3HopPushEnvCfg):
    rewards: StabilityRewardsCfg = StabilityRewardsCfg()
    terminations: FullTerminationsCfg = FullTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # HopPush's post-init installs relaxed costs and a hop-rise reward.
        # Restore the stand objective only AFTER that parent initialization.
        self.rewards = StabilityRewardsCfg()
        self.terminations = FullTerminationsCfg()
        self.curriculum = None
        self.scene.terrain = make_terrain(
            play_generator(HOP_GENERATOR, "flat", 0.0, size=16.0), max_init_level=None
        )
        command = self.commands.base_velocity
        command.rel_standing_envs = 1.0
        command.resampling_time_range = (10.0, 10.0)
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.climb.random_press_prob = 0.0
        self.commands.climb.auto_prob = 0.0
        self.commands.climb.burst_prob = 0.0
        self.commands.climb.both_skew_s = 0.0
        self.actions.leg_pos.guidance_scale = 0.0
        self.actions.leg_pos.ref_style = "v7"
        # Training keeps the original stand material/mass/reset/push randomization.
        # No changes to the actuator, sensor, action scaling or simulation timing.


@configclass
class TanchoV3StabilityPlayEnvCfg(TanchoV3StabilityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _to_play(self)


@configclass
class TanchoV3StabilityPPORunnerCfg(TanchoV3StandPPORunnerCfg):
    experiment_name = "tancho_v3_stability"
