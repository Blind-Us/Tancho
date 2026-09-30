"""Deploy Tancho from zero leg angles to its loaded nominal stance.

All deployment steps are warm-up and are excluded from formal measurement.
No policy is used and the production reset/event configuration is not edited.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tancho staged pose/load experiment.")
parser.add_argument("--task", default="TanchoV3-Flat-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--deployment_duration_s", type=float, default=0.50)
parser.add_argument("--measurement_duration_s", type=float, default=0.50)
parser.add_argument("--gravity_ramp_duration_s", type=float, default=0.20)
parser.add_argument("--start_thigh", type=float, default=0.0)
parser.add_argument("--start_calf", type=float, default=0.0)
parser.add_argument("--target_thigh", type=float, default=-0.50)
parser.add_argument("--target_calf", type=float, default=0.87)
parser.add_argument("--output", type=Path, default=Path("logs/diagnostics/staged_pose_support.csv"))
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import tancho_v3_lab.tasks  # noqa: F401,E402
from tancho_v3_lab.tasks.direct.tancho_v3 import custom_events as ce


JOINT_NAMES = ("joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R")


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _prepare_cfg(cfg: ManagerBasedRLEnvCfg) -> None:
    cfg.scene.num_envs = 1
    if hasattr(cfg.sim, "device"):
        cfg.sim.device = args_cli.device
    if getattr(cfg, "events", None) is not None:
        if hasattr(cfg.events, "reset_gravity_ramp"):
            cfg.events.reset_gravity_ramp = None
        if hasattr(cfg.events, "push_robot"):
            cfg.events.push_robot = None
    if getattr(cfg, "curriculum", None) is not None and hasattr(cfg.curriculum, "enable_push"):
        cfg.curriculum.enable_push = None
    if getattr(cfg, "terminations", None) is not None and hasattr(cfg.terminations, "base_contact"):
        cfg.terminations.base_contact = None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    del agent_cfg
    if args_cli.deployment_duration_s <= 0.0 or args_cli.measurement_duration_s <= 0.0:
        raise ValueError("durations must be positive")
    _prepare_cfg(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    sensor = base.scene["contact_forces"]
    env_ids = torch.tensor([0], device=base.device, dtype=torch.long)
    joint_ids, resolved = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
    wheel_l_ids, _ = sensor.find_bodies("wheel_L")
    wheel_r_ids, _ = sensor.find_bodies("wheel_R")
    wheel_body_ids, _ = robot.find_bodies(["wheel_L", "wheel_R"], preserve_order=True)
    if len(joint_ids) != 4 or len(wheel_body_ids) != 2:
        raise RuntimeError(f"joint/body resolution failed: joints={resolved}, wheels={wheel_body_ids}")

    env.reset()
    dt = float(base.step_dt)
    q = robot.data.joint_pos[0].clone()
    qd = torch.zeros_like(q)
    q[joint_ids[0]] = args_cli.start_thigh
    q[joint_ids[1]] = args_cli.start_calf
    q[joint_ids[2]] = args_cli.start_thigh
    q[joint_ids[3]] = args_cli.start_calf
    root = robot.data.root_link_state_w[0].clone()
    root[7:] = 0.0
    robot.write_joint_state_to_sim(q.unsqueeze(0), qd.unsqueeze(0))
    robot.write_root_link_state_to_sim(root.unsqueeze(0))
    robot.set_joint_position_target(q.unsqueeze(0))
    robot.set_joint_velocity_target(qd.unsqueeze(0))
    ce._set_gravity_compensation_fraction(base, robot, env_ids, 1.0)
    base.scene.write_data_to_sim()
    base.sim.forward()
    base.scene.update(0.0)

    # Reposition the root so the canonical wheel support distance remains on
    # the flat ground for the zero-angle start pose.
    wheel_center_z = float(robot.data.body_pos_w[0, wheel_body_ids, 2].mean().item())
    support_distances = ce.RESET_GEOMETRY.wheel_support_distance_m
    desired_center_z = float(sum(support_distances) / len(support_distances))
    root = robot.data.root_link_state_w[0].clone()
    root[2] += desired_center_z - wheel_center_z
    root[7:] = 0.0
    robot.write_root_link_state_to_sim(root.unsqueeze(0))
    robot.write_joint_state_to_sim(q.unsqueeze(0), qd.unsqueeze(0))
    base.scene.write_data_to_sim()
    base.sim.forward()
    base.scene.update(0.0)

    rows: list[dict[str, float | int | str]] = []

    def record(stage: str, step: int, phase: float) -> None:
        fz_l = float(sensor.data.net_forces_w[0, wheel_l_ids, 2].sum().item())
        fz_r = float(sensor.data.net_forces_w[0, wheel_r_ids, 2].sum().item())
        actual = robot.data.joint_pos[0]
        target = robot.data.joint_pos_target[0]
        velocity = robot.data.joint_vel[0]
        torque = robot.data.applied_torque[0]
        row: dict[str, float | int | str] = {
            "stage": stage,
            "step": step,
            "time_s": step * dt,
            "phase": phase,
            "effective_gravity_fraction": 0.0 if stage == "deployment" else phase,
            "root_z_m": float(robot.data.root_pos_w[0, 2].item()),
            "root_vz_m_s": float(robot.data.root_lin_vel_w[0, 2].item()),
            "wheel_L_fz_N": fz_l,
            "wheel_R_fz_N": fz_r,
        }
        for joint_id, name in zip(joint_ids, resolved):
            stem = name.removeprefix("joint_")
            row[f"{stem}_q_rad"] = float(actual[joint_id].item())
            row[f"{stem}_target_rad"] = float(target[joint_id].item())
            row[f"{stem}_qd_rad_s"] = float(velocity[joint_id].item())
            row[f"{stem}_tau_Nm"] = float(torque[joint_id].item())
        rows.append(row)

    deploy_steps = max(1, round(args_cli.deployment_duration_s / dt))
    for step in range(deploy_steps + 1):
        phase = _smoothstep(step / deploy_steps)
        target = robot.data.joint_pos_target[0].clone()
        thigh = args_cli.start_thigh + phase * (args_cli.target_thigh - args_cli.start_thigh)
        calf = args_cli.start_calf + phase * (args_cli.target_calf - args_cli.start_calf)
        target[joint_ids[0]] = thigh
        target[joint_ids[1]] = calf
        target[joint_ids[2]] = thigh
        target[joint_ids[3]] = calf
        # Deployment is a reset operation, not experimental dynamics.  Write
        # the symmetric pose exactly and shift root Z by the measured wheel
        # support error so the wheels never leave or penetrate the flat plane.
        root = robot.data.root_link_state_w[0].clone()
        root[7:] = 0.0
        robot.write_joint_state_to_sim(target.unsqueeze(0), torch.zeros_like(target).unsqueeze(0))
        robot.write_root_link_state_to_sim(root.unsqueeze(0))
        robot.set_joint_position_target(target.unsqueeze(0))
        robot.set_joint_velocity_target(torch.zeros_like(target).unsqueeze(0))
        ce._set_gravity_compensation_fraction(base, robot, env_ids, 1.0)
        base.scene.write_data_to_sim()
        base.sim.forward()
        base.scene.update(0.0)
        wheel_center_z = float(robot.data.body_pos_w[0, wheel_body_ids, 2].mean().item())
        root = robot.data.root_link_state_w[0].clone()
        root[2] += desired_center_z - wheel_center_z
        root[7:] = 0.0
        robot.write_root_link_state_to_sim(root.unsqueeze(0))
        base.scene.write_data_to_sim()
        base.sim.forward()
        base.scene.update(0.0)
        record("deployment", step, phase)
        if not args_cli.headless:
            base.sim.render()

    # Formal t=0: preserve the loaded joint positions/targets, but remove all
    # deployment velocity.  Full gravity remains active from this point.
    q_loaded = robot.data.joint_pos[0].clone()
    root_loaded = robot.data.root_link_state_w[0].clone()
    root_loaded[7:] = 0.0
    robot.write_joint_state_to_sim(q_loaded.unsqueeze(0), torch.zeros_like(q_loaded).unsqueeze(0))
    robot.write_root_link_state_to_sim(root_loaded.unsqueeze(0))
    ce._set_gravity_compensation_fraction(base, robot, env_ids, 1.0)
    base.scene.write_data_to_sim()
    base.sim.forward()
    base.scene.update(0.0)

    measure_steps = max(1, round(args_cli.measurement_duration_s / dt))
    zero_action = torch.zeros((1, base.action_manager.total_action_dim), device=base.device)
    for step in range(measure_steps + 1):
        elapsed = step * dt
        gravity_phase = _smoothstep(elapsed / args_cli.gravity_ramp_duration_s)
        ce._set_gravity_compensation_fraction(base, robot, env_ids, 1.0 - gravity_phase)
        if step > 0:
            env.step(zero_action)
        record("measurement", step, gravity_phase)

    env.close()
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    final = rows[-1]
    print(f"STAGED_POSE_OUTPUT={args_cli.output.resolve()}")
    print(
        "FINAL "
        f"root_z={float(final['root_z_m']):.6f} "
        f"root_vz={float(final['root_vz_m_s']):+.6f} "
        f"wheel_fz={float(final['wheel_L_fz_N']) + float(final['wheel_R_fz_N']):.6f}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
