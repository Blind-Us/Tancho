#!/usr/bin/env python3
"""Gate B runtime validation for the geometry-derived Tancho wheel reset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Flat-v0")
parser.add_argument("--output", default="logs/physics/gate_b_reset_validation.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import tancho_v3_lab.tasks  # noqa: F401,E402
from tancho_v3_lab.tasks.direct.tancho_v3 import custom_events as ce


def snapshot(robot, wheel_body_ids, leg_ids, wheel_ids, terrain_z: float, contact_sensor=None) -> dict:
    wheel_z = robot.data.body_pos_w[0, wheel_body_ids, 2]
    support = torch.as_tensor(ce.WHEEL_SUPPORT_DISTANCE_M, device=wheel_z.device, dtype=wheel_z.dtype)
    gaps = wheel_z - terrain_z - support
    result = {
        "wheel_bottom_clearance_m": [float(value) for value in gaps],
        "base_vz_m_s": float(robot.data.root_lin_vel_w[0, 2]),
        "joint_qd_rad_s": [float(value) for value in robot.data.joint_vel[0, leg_ids]],
        "wheel_qd_rad_s": [float(value) for value in robot.data.joint_vel[0, wheel_ids]],
        "joint_pos_rad": dict(zip(robot.joint_names, [float(value) for value in robot.data.joint_pos[0]])),
        "root_quat_wxyz": [float(value) for value in robot.data.root_quat_w[0]],
    }
    if contact_sensor is not None:
        force_ids = [index for index, name in enumerate(contact_sensor.body_names) if name in ce.WHEEL_LINK_NAMES]
        result["wheel_contact_force_norm_n"] = [
            float(value)
            for value in torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0, force_ids], dim=-1)
        ]
    for name in ("joint_pos_target", "computed_torque", "applied_torque"):
        value = getattr(robot.data, name, None)
        if value is not None:
            result[name] = [float(item) for item in value[0]]
    return result


def main() -> int:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = 42
    env = gym.make(args.task, cfg=cfg)
    core = env.unwrapped
    try:
        env.reset()
        robot = core.scene["robot"]
        leg_ids, _ = robot.find_joints("joint_(thigh|calf)_.*", preserve_order=True)
        wheel_ids, _ = robot.find_joints("joint_wheel_.*", preserve_order=True)
        wheel_body_ids, _ = robot.find_bodies("wheel_.*", preserve_order=True)
        contact_sensor = core.scene["contact_forces"]
        terrain_z = float(core.scene.env_origins[0, 2])

        actions = torch.zeros(env.action_space.shape, device=core.device)
        with torch.inference_mode():
            core.action_manager.process_action(actions)
            core.action_manager.apply_action()
            core.scene.write_data_to_sim()
            warmup = ce.prepare_tancho_measurement_start(core)
            # Warm-up is outside the experiment.  Official t=0 begins only
            # after both contacts exist and all velocities have been zeroed.
            initial = snapshot(robot, wheel_body_ids, leg_ids, wheel_ids, terrain_z, contact_sensor)
            core.sim.step(render=False)
            core.scene.update(float(core.cfg.sim.dt))
        first = snapshot(robot, wheel_body_ids, leg_ids, wheel_ids, terrain_z, contact_sensor)

        checks = {
            "warmup_reached_stable_state": bool(warmup["stable"]),
            "canonical_support_symmetric": ce.WHEEL_SUPPORT_DISTANCE_M[0] == ce.WHEEL_SUPPORT_DISTANCE_M[1],
            "formal_t0_gap_symmetric": abs(
                initial["wheel_bottom_clearance_m"][0] - initial["wheel_bottom_clearance_m"][1]
            ) <= 1.0e-5,
            "contact_established_before_formal_t0": bool(warmup["contact_established"]),
            "formal_t0_nominal_pose": all(
                abs(initial["joint_pos_rad"][name] - value) <= 1.0e-6
                for name, value in ce.NOMINAL_JOINT_POSITIONS.items()
            ),
            "formal_t0_root_level": max(
                abs(initial["root_quat_wxyz"][index] - expected)
                for index, expected in enumerate((1.0, 0.0, 0.0, 0.0))
            ) <= 1.0e-6,
            "initial_base_vz": abs(initial["base_vz_m_s"]) <= 1.0e-6,
            "initial_joint_qd": max(map(abs, initial["joint_qd_rad_s"])) <= 1.0e-6,
            "initial_wheel_qd": max(map(abs, initial["wheel_qd_rad_s"])) <= 1.0e-6,
        }
        report = {
            "gate": "B",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "physics_dt_s": float(core.cfg.sim.dt),
            "joint_names": list(robot.joint_names),
            "geometry": ce.get_reset_metadata(terrain_z),
            "unrecorded_contact_warmup": {
                "warmup_steps": warmup["warmup_steps"],
                "contact_established": warmup["contact_established"],
                "stable": warmup["stable"],
                "last_step": warmup["history"][-1],
            },
            "formal_t0": initial,
            "first_recorded_physics_step": first,
            "first_recorded_step_is_observational_only": True,
            "checks": checks,
        }
        serialized = json.dumps(
            report, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        print(f"GATE_B_REPORT={output.resolve()}")
        print(f"GATE_B_{report['status']}")
        return 0 if report["status"] == "PASS" else 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
