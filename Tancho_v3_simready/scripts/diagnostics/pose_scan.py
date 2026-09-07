"""Measure passive survival for a grid of thigh/calf targets in parallel."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Flat-v0")
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--leg_stiffness", type=float, default=None)
parser.add_argument("--leg_damping", type=float, default=None)
parser.add_argument("--leg_effort", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import tancho_v3_lab.tasks  # noqa: F401


def main():
    thigh_values = torch.tensor([-0.60, -0.50, -0.40, -0.30], device=args_cli.device)
    calf_values = torch.tensor([0.80, 0.95, 1.10, 1.25, 1.40], device=args_cli.device)
    thigh_targets, calf_targets = torch.meshgrid(thigh_values, calf_values, indexing="ij")
    thigh_targets = thigh_targets.flatten()
    calf_targets = calf_targets.flatten()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=len(thigh_targets))
    if args_cli.leg_stiffness is not None:
        env_cfg.scene.robot.actuators["legs"].stiffness = args_cli.leg_stiffness
    if args_cli.leg_damping is not None:
        env_cfg.scene.robot.actuators["legs"].damping = args_cli.leg_damping
    if args_cli.leg_effort is not None:
        env_cfg.scene.robot.actuators["legs"].effort_limit_sim = args_cli.leg_effort
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    unwrapped = env.unwrapped
    episode_steps = torch.zeros(len(thigh_targets), dtype=torch.long, device=unwrapped.device)
    completed = torch.zeros_like(episode_steps)
    step_sum = torch.zeros_like(episode_steps)
    max_survival = torch.zeros_like(episode_steps)
    height_sum = torch.zeros(len(thigh_targets), device=unwrapped.device)
    height_samples = torch.zeros_like(episode_steps)

    joint_pos_term = unwrapped.action_manager.get_term("joint_pos")
    print(
        f"POSE_SCAN_ACTION_MAP robot_joint_names={unwrapped.scene['robot'].joint_names} "
        f"term_joint_ids={joint_pos_term._joint_ids}",
        flush=True,
    )
    # JointPositionActionCfg resolves the requested names to articulation order:
    # L thigh, R thigh, L calf, R calf.
    thigh_action = (thigh_targets - float(env_cfg.scene.robot.init_state.joint_pos["joint_thigh_L"])) / float(
        env_cfg.actions.joint_pos.scale
    )
    calf_action = (calf_targets - float(env_cfg.scene.robot.init_state.joint_pos["joint_calf_L"])) / float(
        env_cfg.actions.joint_pos.scale
    )
    for _ in range(args_cli.max_steps):
        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
        actions[:, 0] = thigh_action
        actions[:, 1] = thigh_action
        actions[:, 2] = calf_action
        actions[:, 3] = calf_action
        height_sum += unwrapped.scene["robot"].data.root_pos_w[:, 2]
        height_samples += 1
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        episode_steps += 1
        done = terminated | truncated
        step_sum[done] += episode_steps[done]
        max_survival[done] = torch.maximum(max_survival[done], episode_steps[done])
        completed[done] += 1
        episode_steps[done] = 0

    max_survival = torch.maximum(max_survival, episode_steps)
    for i, (thigh, calf) in enumerate(zip(thigh_targets.tolist(), calf_targets.tolist())):
        mean = float(step_sum[i]) / int(completed[i]) if completed[i] else float(episode_steps[i])
        mean_height = float(height_sum[i] / height_samples[i])
        print(
            f"POSE_SCAN thigh={thigh:.3f} calf={calf:.3f} completed={int(completed[i])} "
            f"mean_steps={mean:.2f} max_steps={int(max_survival[i])} "
            f"seconds={mean * float(unwrapped.step_dt):.3f} mean_height={mean_height:.4f}",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
