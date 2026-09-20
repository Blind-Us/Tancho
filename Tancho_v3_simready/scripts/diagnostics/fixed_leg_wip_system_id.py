#!/usr/bin/env python3
"""Open-loop torque-pulse identification for the fixed-leg Tancho WIP asset."""

from __future__ import annotations

import argparse
from datetime import datetime
import csv
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=2.0)
parser.add_argument("--pulse-duration", type=float, default=0.25)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "source/tancho_v3_lab/tancho_v3_lab/assets/robots/Tancho_v3/urdf/Tancho_v3_fixed.urdf"
MANIFEST = ASSET.with_suffix(".json")
PULSES = (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30)


@configclass
class WipSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            )
        ),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSET),
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=None,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={"joint_wheel_L": 0.0, "joint_wheel_R": 0.0},
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["joint_wheel_L", "joint_wheel_R"],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=0.45,
                velocity_limit_sim=188.0,
            )
        },
    )


def pitch_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))


def main() -> int:
    if not ASSET.exists() or not MANIFEST.exists():
        raise FileNotFoundError("Run scripts/physics/build_fixed_leg_wip.py first")
    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    radius = float(metadata["wheel_collision_support_radius_m"])
    sim = SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device))
    scene = InteractiveScene(WipSceneCfg(num_envs=len(PULSES), env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    wheel_ids, wheel_names = robot.find_joints("joint_wheel_.*", preserve_order=True)
    wheel_body_ids, _ = robot.find_bodies("wheel_.*", preserve_order=True)
    if wheel_names != ["joint_wheel_L", "joint_wheel_R"] or robot.num_joints != 2:
        raise RuntimeError(f"Fixed WIP must have only two wheel DOFs; got {robot.joint_names}")

    # Resolve the root height from the imported FK and collision support radius.
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[:, 7:]))
    robot.write_joint_state_to_sim(torch.zeros_like(robot.data.joint_pos), torch.zeros_like(robot.data.joint_vel))
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)
    wheel_rel_z = robot.data.body_pos_w[:, wheel_body_ids, 2] - robot.data.root_pos_w[:, 2:3]
    target_root_z = radius - torch.min(wheel_rel_z, dim=1).values
    root_pose[:, 2] = scene.env_origins[:, 2] + target_root_z
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))
    robot.write_joint_state_to_sim(torch.zeros_like(robot.data.joint_pos), torch.zeros_like(robot.data.joint_vel))
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)
    initial_gap = robot.data.body_pos_w[:, wheel_body_ids, 2] - radius - scene.env_origins[:, 2:3]
    if float(torch.max(torch.abs(initial_gap))) > 1.0e-4:
        raise RuntimeError(f"Initial wheel gap exceeds 0.1 mm: {initial_gap.tolist()}")

    output = args.output or ROOT / "logs/physics" / f"wip_system_id_{datetime.now():%Y%m%d_%H%M%S}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "command_tau_Nm", "time_s", "pitch_rad", "pitch_rate_rad_s", "base_x_m", "base_vx_m_s",
        "wheel_L_q_rad", "wheel_R_q_rad", "wheel_L_qd_rad_s", "wheel_R_qd_rad_s",
        "wheel_L_qdd_rad_s2", "wheel_R_qdd_rad_s2", "wheel_L_tau_Nm", "wheel_R_tau_Nm",
    ]
    rows: list[dict[str, float | int]] = []
    num_steps = math.ceil(args.duration / args.dt)
    pulse_steps = math.ceil(args.pulse_duration / args.dt)
    pulse_tensor = torch.tensor(PULSES, device=robot.device, dtype=robot.data.joint_pos.dtype).unsqueeze(1).repeat(1, 2)
    zero = torch.zeros_like(pulse_tensor)
    for step in range(num_steps + 1):
        q = robot.data.joint_pos[:, wheel_ids]
        qd = robot.data.joint_vel[:, wheel_ids]
        qdd = robot.data.joint_acc[:, wheel_ids]
        tau = robot.data.applied_torque[:, wheel_ids]
        pitch = pitch_wxyz(robot.data.root_quat_w)
        pitch_rate = robot.data.root_ang_vel_b[:, 1]
        base_x = robot.data.root_pos_w[:, 0] - scene.env_origins[:, 0]
        base_vx = robot.data.root_lin_vel_w[:, 0]
        for index, command in enumerate(PULSES):
            rows.append(
                {
                    "case_id": index,
                    "command_tau_Nm": command,
                    "time_s": step * args.dt,
                    "pitch_rad": float(pitch[index]),
                    "pitch_rate_rad_s": float(pitch_rate[index]),
                    "base_x_m": float(base_x[index]),
                    "base_vx_m_s": float(base_vx[index]),
                    "wheel_L_q_rad": float(q[index, 0]), "wheel_R_q_rad": float(q[index, 1]),
                    "wheel_L_qd_rad_s": float(qd[index, 0]), "wheel_R_qd_rad_s": float(qd[index, 1]),
                    "wheel_L_qdd_rad_s2": float(qdd[index, 0]), "wheel_R_qdd_rad_s2": float(qdd[index, 1]),
                    "wheel_L_tau_Nm": float(tau[index, 0]), "wheel_R_tau_Nm": float(tau[index, 1]),
                }
            )
        if step == num_steps:
            break
        robot.set_joint_effort_target(pulse_tensor if step < pulse_steps else zero, joint_ids=wheel_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(args.dt)

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WIP_SYSTEM_ID_CSV={output}")
    print(f"WIP_INITIAL_WHEEL_GAP_M={initial_gap.tolist()}")
    print(f"WIP_MOVABLE_JOINTS={robot.joint_names}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
