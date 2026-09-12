# """Scan simple pitch-feedback wheel controllers to verify actuator controllability."""

# import argparse
# import math

# from isaaclab.app import AppLauncher

# parser = argparse.ArgumentParser()
# parser.add_argument("--task", default="TanchoV3-Flat-v0")
# parser.add_argument("--max_steps", type=int, default=600)
# parser.add_argument("--wheel_scale", type=float, default=None)
# parser.add_argument("--wheel_damping", type=float, default=None)
# parser.add_argument("--wheel_effort", type=float, default=None)
# parser.add_argument("--leg_stiffness", type=float, default=None)
# parser.add_argument("--leg_damping", type=float, default=None)
# parser.add_argument("--leg_effort", type=float, default=None)
# parser.add_argument("--disable_base_contact", action="store_true")
# parser.add_argument("--respawn_height", type=float, default=None)
# parser.add_argument("--constant_scan", action="store_true")
# parser.add_argument("--effort_action", action="store_true")
# parser.add_argument("--effort_scale", type=float, default=10.0)
# parser.add_argument("--spawn_pitch", type=float, default=None, help="Initial pitch about world/body Y in radians.")
# parser.add_argument("--stand_thigh", type=float, default=None)
# parser.add_argument("--stand_calf", type=float, default=None)
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()

# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# import gymnasium as gym
# import torch

# import isaaclab.envs.mdp as mdp
# from isaaclab_tasks.utils import parse_env_cfg
# import tancho_v3_lab.tasks  # noqa: F401


# def main():
#     # (wheel pattern, proportional sign/gain, derivative gain)
#     controllers = [
#         (1.0, 1.0, 0.2), (1.0, -1.0, -0.2),
#         (1.0, 3.0, 0.5), (1.0, -3.0, -0.5),
#         (-1.0, 1.0, 0.2), (-1.0, -1.0, -0.2),
#         (-1.0, 3.0, 0.5), (-1.0, -3.0, -0.5),
#     ]
#     env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=len(controllers))
#     env_cfg.seed = 42
#     if args_cli.effort_action:
#         env_cfg.actions.joint_vel = mdp.JointEffortActionCfg(
#             asset_name="robot",
#             joint_names=["joint_wheel_L", "joint_wheel_R"],
#             scale=args_cli.effort_scale,
#         )
#     if args_cli.wheel_scale is not None:
#         env_cfg.actions.joint_vel.scale = args_cli.wheel_scale
#     if args_cli.wheel_damping is not None:
#         env_cfg.scene.robot.actuators["wheels"].damping = args_cli.wheel_damping
#     if args_cli.wheel_effort is not None:
#         env_cfg.scene.robot.actuators["wheels"].effort_limit_sim = args_cli.wheel_effort
#     if args_cli.leg_stiffness is not None:
#         env_cfg.scene.robot.actuators["legs"].stiffness = args_cli.leg_stiffness
#     if args_cli.leg_damping is not None:
#         env_cfg.scene.robot.actuators["legs"].damping = args_cli.leg_damping
#     if args_cli.leg_effort is not None:
#         env_cfg.scene.robot.actuators["legs"].effort_limit_sim = args_cli.leg_effort
#     if args_cli.disable_base_contact:
#         env_cfg.terminations.base_contact.params["threshold"] = 1.0e9
#     if args_cli.respawn_height is not None:
#         pos = env_cfg.scene.robot.init_state.pos
#         env_cfg.scene.robot.init_state.pos = (pos[0], pos[1], args_cli.respawn_height)
#     if args_cli.spawn_pitch is not None:
#         half = 0.5 * args_cli.spawn_pitch
#         env_cfg.scene.robot.init_state.rot = (math.cos(half), 0.0, math.sin(half), 0.0)
#     if args_cli.stand_thigh is not None:
#         env_cfg.scene.robot.init_state.joint_pos["joint_thigh_L"] = args_cli.stand_thigh
#         env_cfg.scene.robot.init_state.joint_pos["joint_thigh_R"] = args_cli.stand_thigh
#     if args_cli.stand_calf is not None:
#         env_cfg.scene.robot.init_state.joint_pos["joint_calf_L"] = args_cli.stand_calf
#         env_cfg.scene.robot.init_state.joint_pos["joint_calf_R"] = args_cli.stand_calf
#     env = gym.make(args_cli.task, cfg=env_cfg)
#     env.reset()
#     unwrapped = env.unwrapped
#     robot = unwrapped.scene["robot"]
#     wheel_term = unwrapped.action_manager.get_term("joint_vel")
#     wheel_ids, wheel_names = robot.find_joints("joint_wheel_.*")
#     wheel_body_ids, _ = robot.find_bodies("wheel_.*")
#     base_ids, _ = unwrapped.scene.sensors["contact_forces"].find_bodies("base_link_root")
#     print(
#         f"WHEEL_PD_ACTION_MAP robot_joint_names={robot.joint_names} "
#         f"term_joint_ids={wheel_term._joint_ids} wheel_ids={wheel_ids} wheel_names={wheel_names}",
#         flush=True,
#     )
#     episode_steps = torch.zeros(len(controllers), dtype=torch.long, device=unwrapped.device)
#     completed = torch.zeros_like(episode_steps)
#     step_sum = torch.zeros_like(episode_steps)
#     max_survival = torch.zeros_like(episode_steps)
#     max_abs_root_x = torch.zeros(len(controllers), device=unwrapped.device)
#     max_abs_root_vel_x = torch.zeros(len(controllers), device=unwrapped.device)
#     max_abs_wheel_ang_vel = torch.zeros(len(controllers), device=unwrapped.device)
#     max_abs_wheel_joint_vel = torch.zeros(len(controllers), device=unwrapped.device)
#     wheel_center_x_at_20 = torch.full((len(controllers),), torch.nan, device=unwrapped.device)
#     root_x_at_20 = torch.full((len(controllers),), torch.nan, device=unwrapped.device)

#     pattern = torch.tensor([c[0] for c in controllers], device=unwrapped.device)
#     kp = torch.tensor([c[1] for c in controllers], device=unwrapped.device)
#     kd = torch.tensor([c[2] for c in controllers], device=unwrapped.device)
#     constant_actions = torch.tensor(
#         [[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0], [0.5, 0.5],
#          [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [0.25, 0.25]],
#         device=unwrapped.device,
#     )
#     for _ in range(args_cli.max_steps):
#         root_x = robot.data.root_pos_w[:, 0] - unwrapped.scene.env_origins[:, 0]
#         max_abs_root_x = torch.maximum(max_abs_root_x, torch.abs(root_x))
#         max_abs_root_vel_x = torch.maximum(max_abs_root_vel_x, torch.abs(robot.data.root_lin_vel_w[:, 0]))
#         wheel_ang_vel = torch.linalg.vector_norm(robot.data.body_ang_vel_w[:, wheel_body_ids, :], dim=-1).amax(dim=1)
#         max_abs_wheel_ang_vel = torch.maximum(max_abs_wheel_ang_vel, wheel_ang_vel)
#         max_abs_wheel_joint_vel = torch.maximum(
#             max_abs_wheel_joint_vel, torch.abs(robot.data.joint_vel[:, wheel_ids]).amax(dim=1)
#         )
#         first_episode_step_20 = (episode_steps == 19) & torch.isnan(root_x_at_20)
#         wheel_center_x = robot.data.body_pos_w[:, wheel_body_ids, 0].mean(dim=1) - unwrapped.scene.env_origins[:, 0]
#         wheel_center_x_at_20[first_episode_step_20] = wheel_center_x[first_episode_step_20]
#         root_x_at_20[first_episode_step_20] = root_x[first_episode_step_20]
#         gravity_x = robot.data.projected_gravity_b[:, 0]
#         pitch_rate = robot.data.root_ang_vel_b[:, 1]
#         command = torch.clamp(kp * gravity_x + kd * pitch_rate, -1.0, 1.0)
#         actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
#         if args_cli.constant_scan:
#             actions[:, 4:] = constant_actions
#         else:
#             actions[:, 4] = command
#             actions[:, 5] = pattern * command
#         with torch.inference_mode():
#             _, _, terminated, truncated, _ = env.step(actions)
#         if int(episode_steps[0]) in (0, 4, 9, 19, 31, 32, 33, 39, 49, 99):
#             base_force = unwrapped.scene.sensors["contact_forces"].data.net_forces_w[0, base_ids]
#             print(
#                 f"WHEEL_PD_STATE step={int(episode_steps[0]) + 1} "
#                 f"raw_action={actions[0, 4:].tolist()} "
#                 f"processed_action={wheel_term.processed_actions[0].tolist()} "
#                 f"wheel_vel={robot.data.joint_vel[0, wheel_ids].tolist()} "
#                 f"wheel_torque={robot.data.applied_torque[0, wheel_ids].tolist()} "
#                 f"gravity_x={float(robot.data.projected_gravity_b[0, 0]):.5f} "
#                 f"pitch_rate={float(robot.data.root_ang_vel_b[0, 1]):.5f}",
#                 f"root_z={float(robot.data.root_pos_w[0, 2]):.5f} "
#                 f"base_force={base_force.tolist()}",
#                 flush=True,
#             )
#         episode_steps += 1
#         done = terminated | truncated
#         step_sum[done] += episode_steps[done]
#         max_survival[done] = torch.maximum(max_survival[done], episode_steps[done])
#         completed[done] += 1
#         episode_steps[done] = 0

#     max_survival = torch.maximum(max_survival, episode_steps)
#     for i, (wheel_pattern, p_gain, d_gain) in enumerate(controllers):
#         mean = float(step_sum[i]) / int(completed[i]) if completed[i] else float(episode_steps[i])
#         controller_label = (
#             f"constant={constant_actions[i].tolist()}"
#             if args_cli.constant_scan
#             else f"pattern={wheel_pattern:+.0f} kp={p_gain:+.1f} kd={d_gain:+.1f}"
#         )
#         print(
#             f"WHEEL_PD_SCAN {controller_label} "
#             f"completed={int(completed[i])} mean_steps={mean:.2f} max_steps={int(max_survival[i])} "
#             f"seconds={mean * float(unwrapped.step_dt):.3f} "
#             f"max_abs_root_x={float(max_abs_root_x[i]):.5f} "
#             f"max_abs_root_vel_x={float(max_abs_root_vel_x[i]):.5f} "
#             f"max_abs_wheel_ang_vel={float(max_abs_wheel_ang_vel[i]):.5f} "
#             f"max_abs_wheel_joint_vel={float(max_abs_wheel_joint_vel[i]):.5f} "
#             f"root_x_at_20={float(root_x_at_20[i]):.5f} "
#             f"wheel_center_x_at_20={float(wheel_center_x_at_20[i]):.5f}",
#             flush=True,
#         )
#     env.close()


# if __name__ == "__main__":
#     main()
#     simulation_app.close()


"""Scan simple pitch-feedback wheel controllers to verify actuator controllability."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Flat-v0")
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--wheel_scale", type=float, default=None)
parser.add_argument("--wheel_damping", type=float, default=None)
parser.add_argument("--wheel_effort", type=float, default=None)
parser.add_argument("--leg_stiffness", type=float, default=None)
parser.add_argument("--leg_damping", type=float, default=None)
parser.add_argument("--leg_effort", type=float, default=None)
parser.add_argument("--leg_velocity", type=float, default=None)
parser.add_argument("--disable_base_contact", action="store_true")
parser.add_argument("--respawn_height", type=float, default=None)
parser.add_argument("--constant_scan", action="store_true")
parser.add_argument("--effort_action", action="store_true")
parser.add_argument("--effort_scale", type=float, default=10.0)
parser.add_argument("--spawn_pitch", type=float, default=None, help="Initial pitch about world/body Y in radians.")
parser.add_argument("--stand_thigh", type=float, default=None)
parser.add_argument("--stand_calf", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.utils import parse_env_cfg
import tancho_v3_lab.tasks  # noqa: F401


def main():
    # (wheel pattern, proportional sign/gain, derivative gain)
    controllers = [
        (1.0, 1.0, 0.2), (1.0, -1.0, -0.2),
        (1.0, 3.0, 0.5), (1.0, -3.0, -0.5),
        (-1.0, 1.0, 0.2), (-1.0, -1.0, -0.2),
        (-1.0, 3.0, 0.5), (-1.0, -3.0, -0.5),
    ]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=len(controllers))
    env_cfg.seed = 42
    if args_cli.effort_action:
        env_cfg.actions.joint_vel = mdp.JointEffortActionCfg(
            asset_name="robot",
            joint_names=["joint_wheel_L", "joint_wheel_R"],
            scale=args_cli.effort_scale,
        )
    if args_cli.wheel_scale is not None:
        env_cfg.actions.joint_vel.scale = args_cli.wheel_scale
    if args_cli.wheel_damping is not None:
        env_cfg.scene.robot.actuators["wheels"].damping = args_cli.wheel_damping
    if args_cli.wheel_effort is not None:
        env_cfg.scene.robot.actuators["wheels"].effort_limit_sim = args_cli.wheel_effort
    if args_cli.leg_stiffness is not None:
        env_cfg.scene.robot.actuators["legs"].stiffness = args_cli.leg_stiffness
    if args_cli.leg_damping is not None:
        env_cfg.scene.robot.actuators["legs"].damping = args_cli.leg_damping
    if args_cli.leg_effort is not None:
        env_cfg.scene.robot.actuators["legs"].effort_limit_sim = args_cli.leg_effort
    if args_cli.leg_velocity is not None:
        env_cfg.scene.robot.actuators["legs"].velocity_limit_sim = args_cli.leg_velocity
    if args_cli.disable_base_contact:
        env_cfg.terminations.base_contact.params["threshold"] = 1.0e9
    if args_cli.respawn_height is not None:
        pos = env_cfg.scene.robot.init_state.pos
        env_cfg.scene.robot.init_state.pos = (pos[0], pos[1], args_cli.respawn_height)
    if args_cli.spawn_pitch is not None:
        half = 0.5 * args_cli.spawn_pitch
        env_cfg.scene.robot.init_state.rot = (math.cos(half), 0.0, math.sin(half), 0.0)
    if args_cli.stand_thigh is not None:
        env_cfg.scene.robot.init_state.joint_pos["joint_thigh_L"] = args_cli.stand_thigh
        env_cfg.scene.robot.init_state.joint_pos["joint_thigh_R"] = args_cli.stand_thigh
    if args_cli.stand_calf is not None:
        env_cfg.scene.robot.init_state.joint_pos["joint_calf_L"] = args_cli.stand_calf
        env_cfg.scene.robot.init_state.joint_pos["joint_calf_R"] = args_cli.stand_calf
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    wheel_term = unwrapped.action_manager.get_term("joint_vel")
    wheel_ids, wheel_names = robot.find_joints("joint_wheel_.*")
    wheel_body_ids, wheel_body_names = robot.find_bodies("wheel_.*")
    thigh_l_id = robot.find_joints("joint_thigh_L")[0][0]
    calf_l_id = robot.find_joints("joint_calf_L")[0][0]
    thigh_r_id = robot.find_joints("joint_thigh_R")[0][0]
    calf_r_id = robot.find_joints("joint_calf_R")[0][0]
    print(f"BODY_NAMES={robot.body_names} wheel_body_names={wheel_body_names}", flush=True)
    base_ids, _ = unwrapped.scene.sensors["contact_forces"].find_bodies("base_link_root")
    print(
        f"WHEEL_PD_ACTION_MAP robot_joint_names={robot.joint_names} "
        f"term_joint_ids={wheel_term._joint_ids} wheel_ids={wheel_ids} wheel_names={wheel_names}",
        flush=True,
    )
    episode_steps = torch.zeros(len(controllers), dtype=torch.long, device=unwrapped.device)
    completed = torch.zeros_like(episode_steps)
    step_sum = torch.zeros_like(episode_steps)
    max_survival = torch.zeros_like(episode_steps)
    max_abs_root_x = torch.zeros(len(controllers), device=unwrapped.device)
    max_abs_root_vel_x = torch.zeros(len(controllers), device=unwrapped.device)
    max_abs_wheel_ang_vel = torch.zeros(len(controllers), device=unwrapped.device)
    max_abs_wheel_joint_vel = torch.zeros(len(controllers), device=unwrapped.device)
    wheel_center_x_at_20 = torch.full((len(controllers),), torch.nan, device=unwrapped.device)
    root_x_at_20 = torch.full((len(controllers),), torch.nan, device=unwrapped.device)

    pattern = torch.tensor([c[0] for c in controllers], device=unwrapped.device)
    kp = torch.tensor([c[1] for c in controllers], device=unwrapped.device)
    kd = torch.tensor([c[2] for c in controllers], device=unwrapped.device)
    constant_actions = torch.tensor(
        [[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0], [0.5, 0.5],
         [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [0.25, 0.25]],
        device=unwrapped.device,
    )
    for _ in range(args_cli.max_steps):
        root_x = robot.data.root_pos_w[:, 0] - unwrapped.scene.env_origins[:, 0]
        max_abs_root_x = torch.maximum(max_abs_root_x, torch.abs(root_x))
        max_abs_root_vel_x = torch.maximum(max_abs_root_vel_x, torch.abs(robot.data.root_lin_vel_w[:, 0]))
        wheel_ang_vel = torch.linalg.vector_norm(robot.data.body_ang_vel_w[:, wheel_body_ids, :], dim=-1).amax(dim=1)
        max_abs_wheel_ang_vel = torch.maximum(max_abs_wheel_ang_vel, wheel_ang_vel)
        max_abs_wheel_joint_vel = torch.maximum(
            max_abs_wheel_joint_vel, torch.abs(robot.data.joint_vel[:, wheel_ids]).amax(dim=1)
        )
        first_episode_step_20 = (episode_steps == 19) & torch.isnan(root_x_at_20)
        wheel_center_x = robot.data.body_pos_w[:, wheel_body_ids, 0].mean(dim=1) - unwrapped.scene.env_origins[:, 0]
        wheel_center_x_at_20[first_episode_step_20] = wheel_center_x[first_episode_step_20]
        root_x_at_20[first_episode_step_20] = root_x[first_episode_step_20]
        gravity_x = robot.data.projected_gravity_b[:, 0]
        pitch_rate = robot.data.root_ang_vel_b[:, 1]
        command = torch.clamp(kp * gravity_x + kd * pitch_rate, -1.0, 1.0)
        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
        if args_cli.constant_scan:
            actions[:, 4:] = constant_actions
        else:
            actions[:, 4] = command
            actions[:, 5] = pattern * command
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        if int(episode_steps[0]) in (0, 4, 9, 19, 31, 32, 33, 39, 49, 99):
            base_force = unwrapped.scene.sensors["contact_forces"].data.net_forces_w[0, base_ids]
            q = robot.data.joint_pos[0]
            qt = robot.data.joint_pos_target[0]
            qd = robot.data.joint_vel[0]
            tau = robot.data.applied_torque[0]
            wheel_z = robot.data.body_pos_w[0, wheel_body_ids, 2].tolist()
            print(
                f"WHEEL_PD_STATE step={int(episode_steps[0]) + 1} "
                f"root_z={float(robot.data.root_pos_w[0, 2]):.5f} "
                f"root_vz={float(robot.data.root_lin_vel_w[0, 2]):+.5f} "
                f"wheel_z={wheel_z} "
                f"L_thigh(q={q[thigh_l_id].item():+.3f},qt={qt[thigh_l_id].item():+.3f},qd={qd[thigh_l_id].item():+.3f},tau={tau[thigh_l_id].item():+.2f}) "
                f"L_calf(q={q[calf_l_id].item():+.3f},qt={qt[calf_l_id].item():+.3f},qd={qd[calf_l_id].item():+.3f},tau={tau[calf_l_id].item():+.2f}) "
                f"R_thigh(q={q[thigh_r_id].item():+.3f},qt={qt[thigh_r_id].item():+.3f},qd={qd[thigh_r_id].item():+.3f},tau={tau[thigh_r_id].item():+.2f}) "
                f"R_calf(q={q[calf_r_id].item():+.3f},qt={qt[calf_r_id].item():+.3f},qd={qd[calf_r_id].item():+.3f},tau={tau[calf_r_id].item():+.2f}) "
                f"raw_wheel_action={actions[0, 4:].tolist()} "
                f"processed_wheel_action={wheel_term.processed_actions[0].tolist()} "
                f"wheel_vel={robot.data.joint_vel[0, wheel_ids].tolist()} "
                f"wheel_torque={robot.data.applied_torque[0, wheel_ids].tolist()} "
                f"gravity_x={float(robot.data.projected_gravity_b[0, 0]):.5f} "
                f"pitch_rate={float(robot.data.root_ang_vel_b[0, 1]):.5f} "
                f"base_force={base_force.tolist()}",
                flush=True,
            )
        episode_steps += 1
        done = terminated | truncated
        if bool(done[0].item()):
            print(
                f"WHEEL_PD_DONE step={int(episode_steps[0]) + 1} "
                f"terminated={bool(terminated[0].item())} truncated={bool(truncated[0].item())} "
                f"root_z={float(robot.data.root_pos_w[0, 2]):.5f}",
                flush=True,
            )
        step_sum[done] += episode_steps[done]
        max_survival[done] = torch.maximum(max_survival[done], episode_steps[done])
        completed[done] += 1
        episode_steps[done] = 0

    max_survival = torch.maximum(max_survival, episode_steps)
    for i, (wheel_pattern, p_gain, d_gain) in enumerate(controllers):
        mean = float(step_sum[i]) / int(completed[i]) if completed[i] else float(episode_steps[i])
        controller_label = (
            f"constant={constant_actions[i].tolist()}"
            if args_cli.constant_scan
            else f"pattern={wheel_pattern:+.0f} kp={p_gain:+.1f} kd={d_gain:+.1f}"
        )
        print(
            f"WHEEL_PD_SCAN {controller_label} "
            f"completed={int(completed[i])} mean_steps={mean:.2f} max_steps={int(max_survival[i])} "
            f"seconds={mean * float(unwrapped.step_dt):.3f} "
            f"max_abs_root_x={float(max_abs_root_x[i]):.5f} "
            f"max_abs_root_vel_x={float(max_abs_root_vel_x[i]):.5f} "
            f"max_abs_wheel_ang_vel={float(max_abs_wheel_ang_vel[i]):.5f} "
            f"max_abs_wheel_joint_vel={float(max_abs_wheel_joint_vel[i]):.5f} "
            f"root_x_at_20={float(root_x_at_20[i]):.5f} "
            f"wheel_center_x_at_20={float(wheel_center_x_at_20[i]):.5f}",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
