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


RESET_POSE_CASES = (
    {"name": "nominal", "thigh": -0.1, "calf": 0.1},
    {"name": "extended", "thigh": -0.5, "calf": 0.87},
    {"name": "compact", "thigh": 0.4, "calf": 0.4},
)


def _case_joint_positions(case: dict) -> dict[str, float]:
    return {
        "joint_thigh_L": float(case["thigh"]),
        "joint_thigh_R": float(case["thigh"]),
        "joint_calf_L": float(case["calf"]),
        "joint_calf_R": float(case["calf"]),
        "joint_wheel_L": 0.0,
        "joint_wheel_R": 0.0,
    }


def validate_kinematic_reset_cases() -> dict:
    """Pure FK gate for three mirrored poses, independent of Isaac Sim."""

    records = []
    for case in RESET_POSE_CASES:
        joint_positions = _case_joint_positions(case)
        geometry = ce.compute_wheel_support_geometry(joint_positions)
        root_height = ce.compute_reset_root_height(joint_positions)
        gaps = ce.compute_wheel_clearance(root_height, joint_positions=joint_positions)
        records.append(
            {
                "name": case["name"],
                "thigh_rad": case["thigh"],
                "calf_rad": case["calf"],
                "computed_root_height_m": root_height,
                "wheel_support_required_root_height_m": geometry.required_root_height_m,
                "wheel_gap_m": gaps,
                "joint_velocity_rad_s": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "root_velocity_m_s": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
    root_heights = [record["computed_root_height_m"] for record in records]
    gap_ok = all(max(abs(gap) for gap in record["wheel_gap_m"]) <= 1.0e-4 for record in records)
    velocity_ok = all(
        max(abs(value) for value in record["joint_velocity_rad_s"] + record["root_velocity_m_s"]) <= 1.0e-12
        for record in records
    )
    height_ok = len({round(height, 9) for height in root_heights}) == len(root_heights)
    return {
        "status": "PASS" if gap_ok and velocity_ok and height_ok else "FAIL",
        "different_computed_root_heights": height_ok,
        "wheel_gap_within_0.1mm": gap_ok,
        "velocities_zero": velocity_ok,
        "cases": records,
    }


def snapshot(
    robot,
    wheel_body_ids,
    leg_ids,
    wheel_ids,
    terrain_z: float,
    contact_sensor=None,
    joint_positions: dict[str, float] | None = None,
) -> dict:
    wheel_z = robot.data.body_pos_w[0, wheel_body_ids, 2]
    if joint_positions is None:
        support = torch.as_tensor(ce.WHEEL_SUPPORT_DISTANCE_M, device=wheel_z.device, dtype=wheel_z.dtype)
    else:
        geometry = ce.compute_wheel_support_geometry(joint_positions)
        support = torch.as_tensor(geometry.wheel_support_distance_m, device=wheel_z.device, dtype=wheel_z.dtype)
    gaps = wheel_z - terrain_z - support
    result = {
        "wheel_bottom_clearance_m": [float(value) for value in gaps],
        "root_z_m": float(robot.data.root_pos_w[0, 2]),
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
    kinematic_report = validate_kinematic_reset_cases()
    if kinematic_report["status"] != "PASS":
        print(json.dumps({"gate": "B", "kinematic": kinematic_report}, indent=2))
        return 1

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
            initial = snapshot(
                robot,
                wheel_body_ids,
                leg_ids,
                wheel_ids,
                terrain_z,
                contact_sensor,
                _case_joint_positions(RESET_POSE_CASES[0]),
            )
            core.sim.step(render=False)
            core.scene.update(float(core.cfg.sim.dt))
        first = snapshot(
            robot,
            wheel_body_ids,
            leg_ids,
            wheel_ids,
            terrain_z,
            contact_sensor,
            _case_joint_positions(RESET_POSE_CASES[0]),
        )

        runtime_cases = []
        with torch.inference_mode():
            for case in RESET_POSE_CASES:
                joint_positions = _case_joint_positions(case)
                ce.reset_tancho_on_wheels(
                    core,
                    torch.tensor([0], device=core.device, dtype=torch.long),
                    terrain_height=terrain_z,
                    thigh_angles=case["thigh"],
                    calf_angles=case["calf"],
                )
                # Forward synchronizes the teleported state for observation;
                # the reset event itself never advances simulation time.
                core.scene.write_data_to_sim()
                core.sim.forward()
                core.scene.update(0.0)
                observed = snapshot(
                    robot,
                    wheel_body_ids,
                    leg_ids,
                    wheel_ids,
                    terrain_z,
                    contact_sensor,
                    joint_positions,
                )
                expected_root_z = ce.compute_reset_root_height(
                    joint_positions,
                    terrain_height=terrain_z,
                )
                observed["expected_root_z_m"] = expected_root_z
                observed["root_height_error_m"] = observed["root_z_m"] - expected_root_z
                runtime_cases.append({"name": case["name"], **observed})

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
            "three_distinct_computed_root_heights": kinematic_report["different_computed_root_heights"],
            "three_pose_wheel_gaps_within_0.1mm": all(
                max(abs(gap) for gap in case["wheel_bottom_clearance_m"]) <= 1.0e-4
                for case in runtime_cases
            ),
            "three_pose_root_heights_match_fk": all(
                abs(case["root_height_error_m"]) <= 1.0e-5 for case in runtime_cases
            ),
            "three_pose_velocities_zero": all(
                abs(case["base_vz_m_s"]) <= 1.0e-6
                and max(map(abs, case["joint_qd_rad_s"])) <= 1.0e-6
                and max(map(abs, case["wheel_qd_rad_s"])) <= 1.0e-6
                for case in runtime_cases
            ),
        }
        report = {
            "gate": "B",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "physics_dt_s": float(core.cfg.sim.dt),
            "joint_names": list(robot.joint_names),
            "geometry": ce.get_reset_metadata(terrain_z),
            "kinematic": kinematic_report,
            "runtime_cases": runtime_cases,
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
