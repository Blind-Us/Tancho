"""One-episode sweep of matched initial leg pose and wheel-ground spawn height."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Flat-v0")
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--scan_gains", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import tancho_v3_lab.tasks  # noqa: F401


def main():
    thighs = [-0.9, -0.7, -0.5, -0.3, -0.1]
    calves = [0.8, 1.1, 1.4, 1.7, 2.0, 2.3]
    gain_pairs = [(stiffness, damping) for stiffness in (20, 30, 45, 60, 90) for damping in (1, 2, 4, 6, 10)]
    poses = [(-0.6, 0.8)] * len(gain_pairs) if args.scan_gains else [
        (thigh, calf) for thigh in thighs for calf in calves
    ]
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(poses))
    cfg.seed = 42
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    core = env.unwrapped
    robot = core.scene["robot"]
    device = core.device
    if args.scan_gains:
        stiffness = robot.data.joint_stiffness.clone()
        damping = robot.data.joint_damping.clone()
        for index, (kp, kd) in enumerate(gain_pairs):
            for name in ("joint_thigh_L", "joint_thigh_R", "joint_calf_L", "joint_calf_R"):
                joint_id = robot.joint_names.index(name)
                stiffness[index, joint_id] = kp
                damping[index, joint_id] = kd
        robot.write_joint_stiffness_to_sim(stiffness)
        robot.write_joint_damping_to_sim(damping)
    leg = torch.tensor(poses, device=device)
    # URDF's fixed 90-degree root rotation maps the joint-plane y to world z.
    # The outer wheel mesh radius is 0.03614 m. Place its bottom 1 mm above ground.
    root_z = 0.03614 + 0.001 - 0.000552656168 + 0.15 * torch.cos(leg[:, 0]) + 0.0999975373 * torch.cos(
        leg[:, 0] + leg[:, 1]
    )
    root_pose = robot.data.root_pose_w.clone()
    root_pose[:, 2] = root_z
    joint_pos = robot.data.joint_pos.clone()
    for name, column in (("joint_thigh_L", 0), ("joint_thigh_R", 0), ("joint_calf_L", 1), ("joint_calf_R", 1)):
        joint_pos[:, robot.joint_names.index(name)] = leg[:, column]
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    core.sim.forward()
    core.scene.update(0.0)
    wheel_z = robot.data.body_pos_w[:, [robot.body_names.index("wheel_L"), robot.body_names.index("wheel_R")], 2]
    print("INITIAL_POSE_WHEEL_GAP", (wheel_z - 0.03614).tolist(), flush=True)
    action_term = core.action_manager.get_term("joint_pos")
    actions = torch.zeros(env.action_space.shape, device=device)
    for action_index, joint_id in enumerate(action_term._joint_ids):
        name = robot.joint_names[joint_id]
        value = leg[:, 0] if "thigh" in name else leg[:, 1]
        default = cfg.scene.robot.init_state.joint_pos[name]
        actions[:, action_index] = (value - default) / cfg.actions.joint_pos.scale
    first_done = torch.zeros(len(poses), dtype=torch.bool, device=device)
    steps = torch.zeros(len(poses), dtype=torch.long, device=device)
    minimum_z = root_z.clone()
    reason = ["running"] * len(poses)
    contact = core.scene.sensors["contact_forces"]
    contact_names = contact.body_names
    first_contact = ["none"] * len(poses)
    for step in range(args.max_steps):
        active = ~first_done
        minimum_z[active] = torch.minimum(minimum_z[active], robot.data.root_pos_w[active, 2])
        forces = torch.linalg.vector_norm(contact.data.net_forces_w, dim=-1)
        for index in torch.where(active)[0].tolist():
            if first_contact[index] == "none":
                touching = [contact_names[j] for j, force in enumerate(forces[index].tolist()) if force >= 10.0]
                if touching:
                    first_contact[index] = ",".join(touching)
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        steps[active] += 1
        newly_done = active & (terminated | truncated)
        for index in torch.where(newly_done)[0].tolist():
            reason[index] = ",".join(
                name for name in core.termination_manager.active_terms
                if bool(core.termination_manager.get_term(name)[index])
            )
        first_done |= newly_done
        if bool(first_done.all()):
            break
    for index, (thigh, calf) in enumerate(poses):
        label = f"stiffness={gain_pairs[index][0]} damping={gain_pairs[index][1]}" if args.scan_gains else (
            f"thigh={thigh:.2f} calf={calf:.2f}"
        )
        print(
            f"INITIAL_POSE {label} root_z={float(root_z[index]):.4f} "
            f"first_steps={int(steps[index])} minimum_z={float(minimum_z[index]):.4f} "
            f"reason={reason[index]} first_contact={first_contact[index]}", flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
