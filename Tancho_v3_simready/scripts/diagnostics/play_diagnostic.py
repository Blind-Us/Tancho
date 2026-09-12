"""Script to play a trained RSL-RL policy."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--load_run", type=str, default=".*", help="Run folder name or regex to load from.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="model_.*.pt",
    help="Checkpoint filename regex under logs, or a direct path to a .pt file.",
)
parser.add_argument("--num_steps", type=int, default=500, help="Number of inference steps to simulate.")
parser.add_argument("--leg_stiffness", type=float, default=None, help="Diagnostic override for leg actuator stiffness.")
parser.add_argument("--leg_damping", type=float, default=None, help="Diagnostic override for leg actuator damping.")
parser.add_argument("--leg_effort", type=float, default=None, help="Diagnostic override for leg actuator effort limit.")
parser.add_argument("--leg_velocity",type=float,default=None,help="Diagnostic override for leg actuator velocity limit.",)
parser.add_argument("--respawn_height", type=float, default=None, help="Diagnostic override for initial root Z height.")
parser.add_argument(
    "--disable_resets",
    action="store_true",
    default=False,
    help="Disable environment reset during visual play so short or unstable policies do not instantly jump back to the start pose.",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy.onnx from the loaded checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import inspect
import os
import torch
from rsl_rl.runners import OnPolicyRunner
try:
    from rsl_rl.algorithms import PPO
except ImportError:
    PPO = None
try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = None

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import tancho_v3_lab.tasks  # noqa: F401

import os
import torch


class _TanchoV3OnnxPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        actions = self.policy(obs)
        if isinstance(actions, (tuple, list)):
            return actions[0]
        if isinstance(actions, dict):
            if "actions" in actions:
                return actions["actions"]
            if "action" in actions:
                return actions["action"]
            return next(iter(actions.values()))
        return actions


def _stackforce_policy_obs_tensor(obs):
    if isinstance(obs, dict):
        obs = obs["policy"] if "policy" in obs else next(iter(obs.values()))
    elif not isinstance(obs, torch.Tensor) and hasattr(obs, "get"):
        try:
            candidate = obs.get("policy")
        except Exception:
            candidate = None
        if candidate is not None:
            obs = candidate
    if not isinstance(obs, torch.Tensor):
        raise TypeError(f"ONNX export requires a tensor policy observation, got {type(obs)!r}")
    if hasattr(obs, "detach"):
        obs = obs.detach()
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    elif obs.shape[0] > 1:
        obs = obs[:1]
    return obs.contiguous()


def stackforce_export_policy_as_onnx(policy, obs, output_dir, file_name="policy.onnx", opset=17):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    sample_obs = _stackforce_policy_obs_tensor(obs)
    module = _TanchoV3OnnxPolicy(policy).to(sample_obs.device).eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            sample_obs,
            output_path,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=opset,
        )
    print(f"Exported ONNX policy to: {output_path}")
    return output_path



def _runner_uses_obs_groups():
    try:
        source = inspect.getsource(OnPolicyRunner)
        if PPO is not None and hasattr(PPO, "construct_algorithm"):
            source += "\n" + inspect.getsource(PPO.construct_algorithm)
    except OSError:
        return False
    return "resolve_obs_groups" in source or '"obs_groups"' in source or "'obs_groups'" in source


def _runner_uses_split_actor_critic():
    try:
        source = inspect.getsource(OnPolicyRunner)
        if PPO is not None and hasattr(PPO, "construct_algorithm"):
            source += "\n" + inspect.getsource(PPO.construct_algorithm)
    except OSError:
        return False
    return 'cfg["actor"]' in source or "cfg['actor']" in source or 'cfg["critic"]' in source or "cfg['critic']" in source


def _runner_expects_privileged_step():
    try:
        source = inspect.getsource(OnPolicyRunner.learn)
    except OSError:
        return True
    return "privileged_obs" in source or "critic_obs" in source


def _format_rsl_rl_obs(obs_dict, use_obs_groups):
    if not use_obs_groups:
        return obs_dict["policy"]
    if TensorDict is not None and not isinstance(obs_dict, TensorDict):
        first_obs = next(iter(obs_dict.values()))
        return TensorDict(dict(obs_dict), batch_size=[first_obs.shape[0]], device=first_obs.device)
    return obs_dict


class LegacyRslRlVecEnvWrapper:
    def __init__(self, env, clip_actions=None):
        self.env = env
        self.clip_actions = clip_actions
        self.use_obs_groups = _runner_uses_obs_groups()
        self.return_privileged_obs = _runner_expects_privileged_step()
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.max_episode_length = env.unwrapped.max_episode_length
        self.cfg = env.unwrapped.cfg
        self.num_actions = gym.spaces.flatdim(env.unwrapped.single_action_space)
        obs_dict, extras = self.env.reset()
        self.obs_buf = _format_rsl_rl_obs(obs_dict, self.use_obs_groups)
        self.privileged_obs_buf = obs_dict.get("critic")
        self.num_obs = obs_dict["policy"].shape[-1]
        self.num_privileged_obs = self.privileged_obs_buf.shape[-1] if self.privileged_obs_buf is not None else None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_length_buf = env.unwrapped.episode_length_buf
        self.extras = extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self, env_ids=None):
        del env_ids
        obs_dict, extras = self.env.reset()
        self.obs_buf = _format_rsl_rl_obs(obs_dict, self.use_obs_groups)
        self.privileged_obs_buf = obs_dict.get("critic")
        self.extras = extras
        if not self.return_privileged_obs:
            return self.obs_buf
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.env.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        episode_log = dict(extras.get("episode", {}))
        if torch.any(dones.bool()) and "log" in extras:
            episode_log.update(extras["log"])
        episode_log["Step_Reward/mean"] = torch.mean(rewards.detach())
        episode_log["Step_Reward/abs_mean"] = torch.mean(torch.abs(rewards.detach()))
        extras["episode"] = episode_log
        self.obs_buf = _format_rsl_rl_obs(obs_dict, self.use_obs_groups)
        self.privileged_obs_buf = obs_dict.get("critic")
        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = extras
        if not self.return_privileged_obs:
            return self.obs_buf, rewards, dones, extras
        return self.obs_buf, self.privileged_obs_buf, rewards, dones, extras

    def close(self):
        return self.env.close()


def _runner_expects_nested_runner():
    try:
        source = inspect.getsource(OnPolicyRunner)
    except OSError:
        return True
    return 'train_cfg["runner"]' in source or "train_cfg['runner']" in source


def _runner_uses_nested_class_name():
    try:
        source = inspect.getsource(OnPolicyRunner)
    except OSError:
        return False
    return (
        'algorithm"]["class_name' in source
        or "algorithm']['class_name" in source
        or 'policy_cfg.pop("class_name")' in source
        or "policy_cfg.pop('class_name')" in source
        or 'self.policy_cfg.pop("class_name")' in source
        or "self.policy_cfg.pop('class_name')" in source
        or "resolve_callable" in source
    )


def _runner_uses_split_actor_critic():
    try:
        source = inspect.getsource(OnPolicyRunner)
        if PPO is not None and hasattr(PPO, "construct_algorithm"):
            source += "\n" + inspect.getsource(PPO.construct_algorithm)
    except OSError:
        return False
    return 'cfg["actor"]' in source or "cfg['actor']" in source or 'cfg["critic"]' in source or "cfg['critic']" in source


def to_compatible_rsl_rl_cfg(agent_cfg):
    data = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    allowed_policy_keys = {
        "actor_hidden_dims",
        "critic_hidden_dims",
        "activation",
        "init_noise_std",
        "clip_actions",
        "actor_obs_normalization",
        "critic_obs_normalization",
    }
    allowed_algorithm_keys = {
        "num_learning_epochs",
        "num_mini_batches",
        "clip_param",
        "gamma",
        "lam",
        "value_loss_coef",
        "entropy_coef",
        "learning_rate",
        "max_grad_norm",
        "use_clipped_value_loss",
        "schedule",
        "desired_kl",
        "use_spo",
    }
    if "runner" in data and "policy" in data and "algorithm" in data:
        runner_cfg = dict(data["runner"])
        policy_cfg = {key: value for key, value in dict(data["policy"]).items() if key in allowed_policy_keys or key == "class_name"}
        algorithm_cfg = {key: value for key, value in dict(data["algorithm"]).items() if key in allowed_algorithm_keys or key == "class_name"}
    else:
        policy_cfg = {key: value for key, value in dict(data["policy"]).items() if key in allowed_policy_keys}
        algorithm_cfg = {key: value for key, value in dict(data["algorithm"]).items() if key in allowed_algorithm_keys}
        runner_cfg = {key: value for key, value in data.items() if key not in {"policy", "algorithm", "class_name"}}

    runner_cfg.setdefault("num_steps_per_env", getattr(agent_cfg, "num_steps_per_env", 24))
    runner_cfg.setdefault("max_iterations", getattr(agent_cfg, "max_iterations", 1500))
    runner_cfg.setdefault("save_interval", getattr(agent_cfg, "save_interval", 50))
    runner_cfg.setdefault("obs_groups", {"policy": ["policy"], "critic": ["policy"]})
    runner_cfg.setdefault("experiment_name", getattr(agent_cfg, "experiment_name", "stackforce"))
    runner_cfg.setdefault("run_name", getattr(agent_cfg, "run_name", ""))
    runner_cfg.setdefault("resume", getattr(agent_cfg, "resume", False))
    runner_cfg.setdefault("load_run", getattr(agent_cfg, "load_run", ".*"))
    runner_cfg.setdefault("checkpoint", getattr(agent_cfg, "load_checkpoint", "model_.*.pt"))

    if _runner_uses_nested_class_name():
        policy_cfg.setdefault("class_name", "ActorCritic")
        algorithm_cfg.setdefault("class_name", "PPO")
    else:
        runner_cfg.setdefault("policy_class_name", "ActorCritic")
        runner_cfg.setdefault("algorithm_class_name", "PPO")

    if _runner_expects_nested_runner():
        return {"runner": runner_cfg, "policy": policy_cfg, "algorithm": algorithm_cfg}
    if _runner_uses_split_actor_critic():
        algorithm_cfg.setdefault("class_name", "PPO")
        algorithm_cfg.pop("use_spo", None)
        actor_cfg = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("actor_hidden_dims", [256, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": policy_cfg.get("init_noise_std", 1.0),
            },
        }
        critic_cfg = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("critic_hidden_dims", [256, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
        }
        runner_cfg.pop("policy_class_name", None)
        runner_cfg.pop("algorithm_class_name", None)
        runner_cfg["obs_groups"] = {"actor": ["policy"], "critic": ["policy"], "policy": ["policy"]}
        runner_cfg.setdefault("multi_gpu", None)
        return {**runner_cfg, "actor": actor_cfg, "critic": critic_cfg, "algorithm": algorithm_cfg}
    return {**runner_cfg, "policy": policy_cfg, "algorithm": algorithm_cfg}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.leg_stiffness is not None:
        env_cfg.scene.robot.actuators["legs"].stiffness = args_cli.leg_stiffness
    if args_cli.leg_damping is not None:
        env_cfg.scene.robot.actuators["legs"].damping = args_cli.leg_damping
    if args_cli.leg_effort is not None:
        env_cfg.scene.robot.actuators["legs"].effort_limit_sim = args_cli.leg_effort
    # IMPORTANT: the CLI argument existed before, but was not actually applied.
    if args_cli.leg_velocity is not None:
        env_cfg.scene.robot.actuators["legs"].velocity_limit_sim = args_cli.leg_velocity

    if args_cli.respawn_height is not None:
        _pos = env_cfg.scene.robot.init_state.pos
        env_cfg.scene.robot.init_state.pos = (_pos[0], _pos[1], args_cli.respawn_height)

    legs_cfg = env_cfg.scene.robot.actuators["legs"]
    print(
        f"[DIAG CFG] legs: stiffness={legs_cfg.stiffness}, "
        f"damping={legs_cfg.damping}, effort_limit_sim={legs_cfg.effort_limit_sim}, "
        f"velocity_limit_sim={legs_cfg.velocity_limit_sim}"
    )

    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    elif args_cli.disable_resets and hasattr(env_cfg, "visual_disable_resets"):
        env_cfg.visual_disable_resets = True

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_arg = args_cli.checkpoint
    if os.path.isfile(checkpoint_arg):
        resume_path = os.path.abspath(checkpoint_arg)
    else:
        resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, checkpoint_arg)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)


    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    # ------------------------------------------------------------
    # Tancho V3 leg joint IDs
    # ------------------------------------------------------------
    thigh_l_id = robot.find_joints("joint_thigh_L")[0][0]
    calf_l_id = robot.find_joints("joint_calf_L")[0][0]
    thigh_r_id = robot.find_joints("joint_thigh_R")[0][0]
    calf_r_id = robot.find_joints("joint_calf_R")[0][0]

    # Body / joint IDs used for diagnostics.
    wheel_l_id = robot.find_bodies("wheel_L")[0][0]
    wheel_r_id = robot.find_bodies("wheel_R")[0][0]
    wheel_l_joint_id = robot.find_joints("joint_wheel_L")[0][0]
    wheel_r_joint_id = robot.find_joints("joint_wheel_R")[0][0]

    print(f"[DIAG] robot.body_names = {robot.body_names}")
    print(f"[DIAG] robot.joint_names = {robot.joint_names}")

    # Contact sensor body ids. Missing/merged bodies are tolerated and reported as 0 N.
    contact_sensor = unwrapped.scene.sensors.get("contact_forces", None)

    def _contact_ids(name):
        if contact_sensor is None:
            return []
        try:
            ids, _ = contact_sensor.find_bodies(name)
            return ids
        except Exception:
            return []

    contact_ids = {
        "wheelL": _contact_ids("wheel_L"),
        "wheelR": _contact_ids("wheel_R"),
        "thighL": _contact_ids("thigh_L"),
        "thighR": _contact_ids("thigh_R"),
        "calfL": _contact_ids("calf_L"),
        "calfR": _contact_ids("calf_R"),
        "baseRoot": _contact_ids("base_link_root"),
        "base": _contact_ids("base_link"),
    }
    print(f"[DIAG] contact_ids = {contact_ids}")

    # ------------------------------------------------------------
    # Load RSL-RL policy
    # ------------------------------------------------------------
    legacy_agent_cfg = to_compatible_rsl_rl_cfg(agent_cfg)

    wrapped_env = LegacyRslRlVecEnvWrapper(
        env,
        clip_actions=getattr(agent_cfg, "clip_actions", None),
    )

    runner = OnPolicyRunner(
        wrapped_env,
        legacy_agent_cfg,
        log_dir=None,
        device=env.unwrapped.device,
    )

    runner.load(resume_path)

    policy = runner.get_inference_policy(
        device=env.unwrapped.device
    )

    obs = wrapped_env.get_observations()

    # ------------------------------------------------------------
    # Optional ONNX export
    # ------------------------------------------------------------
    if args_cli.export_onnx:
        stackforce_export_policy_as_onnx(
            policy,
            obs,
            os.path.join(
                os.path.dirname(resume_path),
                "exported",
                "policies",
            ),
        )

    # ------------------------------------------------------------
    # Play - ZERO ACTION diagnostic
    # ------------------------------------------------------------
    # The goal of this block is not to judge the PPO policy.
    # It keeps the action at zero and observes what physics does to the robot.
    steps = 0

    def _contact_force_mag(ids):
        if contact_sensor is None or len(ids) == 0:
            return 0.0
        try:
            forces = contact_sensor.data.net_forces_w[0, ids, :]
            return torch.linalg.vector_norm(forces, dim=-1).sum().item()
        except Exception:
            return 0.0

    def _print_state(tag, done=False):
        q = robot.data.joint_pos[0]
        qt = robot.data.joint_pos_target[0]
        qd = robot.data.joint_vel[0]
        tau = robot.data.applied_torque[0]

        root_z = robot.data.root_pos_w[0, 2].item()
        root_vz = robot.data.root_lin_vel_w[0, 2].item()
        wheel_l_z = robot.data.body_pos_w[0, wheel_l_id, 2].item()
        wheel_r_z = robot.data.body_pos_w[0, wheel_r_id, 2].item()

        wheel_l_qd = robot.data.joint_vel[0, wheel_l_joint_id].item()
        wheel_r_qd = robot.data.joint_vel[0, wheel_r_joint_id].item()
        wheel_l_tau = robot.data.applied_torque[0, wheel_l_joint_id].item()
        wheel_r_tau = robot.data.applied_torque[0, wheel_r_joint_id].item()
        gravity_x = robot.data.projected_gravity_b[0, 0].item()
        pitch_rate = robot.data.root_ang_vel_b[0, 1].item()

        fwL = _contact_force_mag(contact_ids["wheelL"])
        fwR = _contact_force_mag(contact_ids["wheelR"])
        fthigh = _contact_force_mag(contact_ids["thighL"]) + _contact_force_mag(contact_ids["thighR"])
        fcalf = _contact_force_mag(contact_ids["calfL"]) + _contact_force_mag(contact_ids["calfR"])
        fbase = _contact_force_mag(contact_ids["baseRoot"]) + _contact_force_mag(contact_ids["base"])

        print(
            f"{tag} done={int(done)} "
            f"root_z={root_z:+.4f} root_vz={root_vz:+.4f} "
            f"wheelL_z={wheel_l_z:+.4f} wheelR_z={wheel_r_z:+.4f} "
            f"grav_x={gravity_x:+.4f} pitch_rate={pitch_rate:+.4f} | "
            f"wheel_qd=({wheel_l_qd:+.3f},{wheel_r_qd:+.3f}) "
            f"wheel_tau=({wheel_l_tau:+.2f},{wheel_r_tau:+.2f}) | "
            f"Fwheel=({fwL:.1f},{fwR:.1f})N Fthigh={fthigh:.1f}N Fcalf={fcalf:.1f}N Fbase={fbase:.1f}N | "
            f"L thigh q={q[thigh_l_id].item():+.3f} qt={qt[thigh_l_id].item():+.3f} "
            f"qd={qd[thigh_l_id].item():+.3f} tau={tau[thigh_l_id].item():+.2f} | "
            f"L calf q={q[calf_l_id].item():+.3f} qt={qt[calf_l_id].item():+.3f} "
            f"qd={qd[calf_l_id].item():+.3f} tau={tau[calf_l_id].item():+.2f} | "
            f"R thigh q={q[thigh_r_id].item():+.3f} qt={qt[thigh_r_id].item():+.3f} "
            f"qd={qd[thigh_r_id].item():+.3f} tau={tau[thigh_r_id].item():+.2f} | "
            f"R calf q={q[calf_r_id].item():+.3f} qt={qt[calf_r_id].item():+.3f} "
            f"qd={qd[calf_r_id].item():+.3f} tau={tau[calf_r_id].item():+.2f}"
        )

    # Print the state before the first physics step.
    _print_state("[INIT ]", done=False)

    with torch.inference_mode():
        while simulation_app.is_running():

            # Save the state immediately BEFORE env.step().  If the environment
            # terminates and auto-resets inside step(), these values are the last
            # observable state before that step.
            pre_root_z = robot.data.root_pos_w[0, 2].item()
            pre_root_vz = robot.data.root_lin_vel_w[0, 2].item()

            # TEST: do not use the policy; hold the default joint target.
            actions = torch.zeros(
                (wrapped_env.num_envs, wrapped_env.num_actions),
                device=wrapped_env.device,
            )

            step_result = wrapped_env.step(actions)
            obs = step_result[0]
            done = bool(wrapped_env.reset_buf[0].item())

            # Print much more frequently than before.  The current robot usually
            # terminates around ~53 steps, so every 5 steps shows the onset clearly.
            if steps % 5 == 0 or done:
                _print_state(f"[{steps:05d}]", done=done)

            if done:
                print(
                    f"[DONE ] step={steps} "
                    f"pre_step_root_z={pre_root_z:+.4f} "
                    f"pre_step_root_vz={pre_root_vz:+.4f} "
                    "(state printed above may already be post-reset)"
                )

            steps += 1

            if args_cli.num_steps > 0 and steps >= args_cli.num_steps:
                break


if __name__ == "__main__":
    main()
    simulation_app.close()