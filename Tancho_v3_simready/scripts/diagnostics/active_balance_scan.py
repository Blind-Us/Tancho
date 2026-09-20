"""Test leg preload and wheel pitch feedback with unchanged motor limits."""

import argparse
import itertools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Flat-v0")
parser.add_argument("--max_steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import tancho_v3_lab.tasks  # noqa: F401


def main():
    preload_pairs = [(0.0, 0.0), (0.1, -0.1), (0.2, -0.2), (0.3, -0.3), (0.4, -0.4)]
    wheel_gains = [(0.0, 0.0, 1.0)] + list(itertools.product((2.0, 5.0, 10.0), (0.2, 0.5), (-1.0, 1.0)))
    cases = list(itertools.product(preload_pairs, wheel_gains))
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(cases))
    cfg.seed = 42
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    core = env.unwrapped
    robot = core.scene["robot"]
    device = core.device
    pos_term = core.action_manager.get_term("joint_pos")
    wheel_term = core.action_manager.get_term("joint_vel")
    print(f"ACTIVE_MAP joint_names={robot.joint_names} leg_ids={pos_term._joint_ids} wheel_ids={wheel_term._joint_ids}", flush=True)
    preload = torch.tensor([case[0] for case in cases], device=device)
    gains = torch.tensor([case[1] for case in cases], device=device)
    actions = torch.zeros(env.action_space.shape, device=device)
    for action_index, joint_id in enumerate(pos_term._joint_ids):
        name = robot.joint_names[joint_id]
        actions[:, action_index] = preload[:, 0] / cfg.actions.joint_pos.scale if "thigh" in name else (
            preload[:, 1] / cfg.actions.joint_pos.scale
        )
    first_done = torch.zeros(len(cases), dtype=torch.bool, device=device)
    steps = torch.zeros(len(cases), dtype=torch.long, device=device)
    min_height = torch.full((len(cases),), float("inf"), device=device)
    max_wheel_torque = torch.zeros(len(cases), device=device)
    max_leg_torque = torch.zeros(len(cases), device=device)
    for _ in range(args.max_steps):
        active = ~first_done
        min_height[active] = torch.minimum(min_height[active], robot.data.root_pos_w[active, 2])
        tilt = robot.data.projected_gravity_b[:, 0]
        pitch_rate = robot.data.root_ang_vel_b[:, 1]
        wheel_effort = (gains[:, 0] * tilt + gains[:, 1] * pitch_rate) * gains[:, 2]
        actions[:, 4:6] = torch.clamp(wheel_effort / cfg.actions.joint_vel.scale, -1.0, 1.0).unsqueeze(1)
        torques = robot.data.applied_torque.abs()
        max_leg_torque[active] = torch.maximum(max_leg_torque[active], torques[active, :4].max(1).values)
        max_wheel_torque[active] = torch.maximum(max_wheel_torque[active], torques[active, 4:6].max(1).values)
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        steps[active] += 1
        first_done |= active & (terminated | truncated)
        if bool(first_done.all()):
            break
    for i, (leg, wheel) in enumerate(cases):
        print(
            f"ACTIVE_RESULT thigh_preload={leg[0]:.2f} calf_preload={leg[1]:.2f} "
            f"wheel_kp={wheel[0]:.1f} wheel_kd={wheel[1]:.1f} sign={wheel[2]:.0f} "
            f"steps={int(steps[i])} min_height={float(min_height[i]):.4f} "
            f"max_leg_torque={float(max_leg_torque[i]):.3f} max_wheel_torque={float(max_wheel_torque[i]):.3f}",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
