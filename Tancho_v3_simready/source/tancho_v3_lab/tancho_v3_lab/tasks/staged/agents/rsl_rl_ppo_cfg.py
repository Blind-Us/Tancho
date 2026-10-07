from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class TanchoV3WheelOnlyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "tancho_v3_wheel_only"
    clip_actions = 1.0
    # Actor sees only the hardware observation; critic gets privileged state.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.2),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TanchoV3StandPPORunnerCfg(TanchoV3WheelOnlyPPORunnerCfg):
    max_iterations = 3000
    experiment_name = "tancho_v3_stand"


@configclass
class TanchoV3WalkPPORunnerCfg(TanchoV3WheelOnlyPPORunnerCfg):
    # Started from a stage-2 checkpoint with ``train.py --init_checkpoint``.
    max_iterations = 3000
    experiment_name = "tancho_v3_walk"


@configclass
class TanchoV3WalkRoughPPORunnerCfg(TanchoV3WalkPPORunnerCfg):
    # Started from a flat-walk checkpoint with ``--init_checkpoint``.
    experiment_name = "tancho_v3_walk_rough"


@configclass
class TanchoV3WalkStepPPORunnerCfg(TanchoV3WalkPPORunnerCfg):
    # Started from a rough-walk checkpoint with ``--init_checkpoint``.
    experiment_name = "tancho_v3_walk_step"
