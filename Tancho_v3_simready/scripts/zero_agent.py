import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero agent for Tancho V3 Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--constant_action", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many control steps; 0 runs forever.")
parser.add_argument(
    "--fail_mean_episode_steps",
    type=float,
    default=0.0,
    help="Fail when completed episodes average fewer steps than this value.",
)
parser.add_argument(
    "--fail_non_timeout_ratio",
    type=float,
    default=1.01,
    help="Fail when the fraction of completed episodes ending without timeout exceeds this value.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import tancho_v3_lab.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    root_body_ids, _ = contact_sensor.find_bodies("base_link_root")
    wheel_body_ids, _ = contact_sensor.find_bodies(".*wheel.*")
    thigh_body_ids = [index for index, name in enumerate(contact_sensor.body_names) if "thigh" in name]
    calf_body_ids = [index for index, name in enumerate(contact_sensor.body_names) if "calf" in name]
    action_terms = {
        name: unwrapped.action_manager.get_term(name)
        for name in unwrapped.action_manager.active_terms
    }
    print(
        "ZERO_AGENT_ACTION_MAP "
        f"robot_joint_names={robot.joint_names} action_terms={list(action_terms)} "
        f"default_joint_pos={robot.data.default_joint_pos[0].tolist()}",
        flush=True,
    )
    episode_steps = torch.zeros(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
    completed_episodes = 0
    completed_step_sum = 0
    termination_counts = {name: 0 for name in unwrapped.termination_manager.active_terms}
    control_steps = 0
    printed_first_done = False

    while simulation_app.is_running() and (args_cli.max_steps <= 0 or control_steps < args_cli.max_steps):
        if control_steps in (2, 5, 10, 20, 39):
            root_force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0, root_body_ids], dim=-1)
            wheel_force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0, wheel_body_ids], dim=-1)
            thigh_force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0, thigh_body_ids], dim=-1)
            calf_force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0, calf_body_ids], dim=-1)
            print(
                f"ZERO_AGENT_STATE step={control_steps} "
                f"root_z={float(robot.data.root_pos_w[0, 2]):.6f} "
                f"gravity_b={robot.data.projected_gravity_b[0].tolist()} "
                f"ang_vel_b={robot.data.root_ang_vel_b[0].tolist()} "
                f"joint_pos={robot.data.joint_pos[0].tolist()} "
                f"joint_target={robot.data.joint_pos_target[0].tolist()} "
                f"applied_torque={robot.data.applied_torque[0].tolist()} "
                f"root_force={root_force.tolist()} wheel_force={wheel_force.tolist()} "
                f"thigh_force={thigh_force.tolist()} calf_force={calf_force.tolist()}",
                flush=True,
            )
        with torch.inference_mode():
            actions = torch.full(env.action_space.shape, args_cli.constant_action, device=unwrapped.device)
            _, _, terminated, truncated, _ = env.step(actions)

        if control_steps == 0:
            print(
                "ZERO_AGENT_PROCESSED_ACTIONS "
                + " ".join(
                    f"{name}={term.processed_actions[0].tolist()}"
                    for name, term in action_terms.items()
                ),
                flush=True,
            )

        control_steps += 1
        episode_steps += 1
        done = terminated | truncated
        done_count = int(done.sum().item())
        if done_count:
            if not printed_first_done:
                first = int(torch.nonzero(done, as_tuple=False)[0, 0])
                print(
                    "ZERO_AGENT_FIRST_DONE "
                    f"step={control_steps} env={first} root_z={float(robot.data.root_pos_w[first, 2]):.6f} "
                    f"gravity_b={robot.data.projected_gravity_b[first].tolist()} "
                    f"ang_vel_b={robot.data.root_ang_vel_b[first].tolist()} "
                    f"joint_pos={robot.data.joint_pos[first].tolist()} "
                    f"joint_vel={robot.data.joint_vel[first].tolist()}",
                    flush=True,
                )
                printed_first_done = True
            completed_episodes += done_count
            completed_step_sum += int(episode_steps[done].sum().item())
            for name in termination_counts:
                termination_counts[name] += int(unwrapped.termination_manager.get_term(name).sum().item())
            episode_steps[done] = 0

    mean_episode_steps = completed_step_sum / completed_episodes if completed_episodes else float(control_steps)
    mean_episode_seconds = mean_episode_steps * float(unwrapped.step_dt)
    timeout_count = termination_counts.get("time_out", 0)
    non_timeout_ratio = (
        max(0, completed_episodes - timeout_count) / completed_episodes if completed_episodes else 0.0
    )
    terms = ",".join(f"{name}:{count}" for name, count in termination_counts.items())
    print(
        "ZERO_AGENT_SUMMARY "
        f"control_steps={control_steps} completed_episodes={completed_episodes} "
        f"mean_episode_steps={mean_episode_steps:.2f} mean_episode_seconds={mean_episode_seconds:.3f} "
        f"non_timeout_ratio={non_timeout_ratio:.4f} terminations={terms}",
        flush=True,
    )

    failed = completed_episodes > 0 and (
        (args_cli.fail_mean_episode_steps > 0 and mean_episode_steps < args_cli.fail_mean_episode_steps)
        or non_timeout_ratio > args_cli.fail_non_timeout_ratio
    )
    env.close()
    if failed:
        print("ZERO_AGENT_GATE status=failed reason=reset-loop", flush=True)
        raise SystemExit(3)
    print("ZERO_AGENT_GATE status=passed", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
