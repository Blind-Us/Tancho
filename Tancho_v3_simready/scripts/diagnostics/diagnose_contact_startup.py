#!/usr/bin/env python3
"""Trace Tancho wheel collision/contact state across reset and first steps.

This diagnostic deliberately does not judge balance or tune a controller.  It
records the four remaining Gate-B suspects: URDF joint-frame mapping, authored
collision transforms, cooked USD collision prims, and the contact manifold
timeline immediately after a reset teleport.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Flat-v0")
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--output", default="logs/physics/contact_startup_trace.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import tancho_v3_lab.tasks  # noqa: F401,E402
from tancho_v3_lab.tasks.direct.tancho_v3 import custom_events as ce


def _numbers(tensor) -> list:
    return tensor.detach().cpu().tolist()


def _urdf_audit() -> dict:
    root = ET.parse(ce.URDF_PATH).getroot()
    joints = {}
    for name in ("joint_calf_L", "joint_calf_R", "joint_wheel_L", "joint_wheel_R"):
        joint = root.find(f"joint[@name='{name}']")
        origin = joint.find("origin")
        joints[name] = {
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "axis": joint.find("axis").get("xyz"),
            "origin_xyz": origin.get("xyz", "0 0 0"),
            "origin_rpy": origin.get("rpy", "0 0 0"),
        }
    collisions = {}
    for link_name in ce.WHEEL_LINK_NAMES:
        link = root.find(f"link[@name='{link_name}']")
        collisions[link_name] = []
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            collisions[link_name].append(
                {
                    "name": collision.get("name"),
                    "mesh": collision.find("geometry/mesh").get("filename"),
                    "origin_xyz": origin.get("xyz", "0 0 0"),
                    "origin_rpy": origin.get("rpy", "0 0 0"),
                }
            )
    return {"joints": joints, "collisions": collisions}


def _usd_collision_audit() -> list[dict]:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    records = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Robot/" not in path or not any(token in path.lower() for token in ("wheel_l", "wheel_r")):
            continue
        attrs = {}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if "collision" in name.lower() or "approximation" in name.lower() or "mesh" in name.lower():
                try:
                    value = attr.Get()
                    attrs[name] = str(value)
                except Exception:
                    attrs[name] = "<unreadable>"
        if attrs or "collision" in path.lower():
            records.append({"path": path, "type": prim.GetTypeName(), "attributes": attrs})
    return records


def _snapshot(core, robot, wheel_body_ids, contact_sensor, label: str) -> dict:
    support = torch.as_tensor(ce.WHEEL_SUPPORT_DISTANCE_M, device=core.device)
    link_pos = getattr(robot.data, "body_link_pos_w", robot.data.body_pos_w)[0, wheel_body_ids]
    com_pos = getattr(robot.data, "body_com_pos_w", robot.data.body_pos_w)[0, wheel_body_ids]
    body_quat = getattr(robot.data, "body_link_quat_w", robot.data.body_quat_w)[0, wheel_body_ids]
    gaps = link_pos[:, 2] - float(core.scene.env_origins[0, 2]) - support
    force_names = list(getattr(contact_sensor, "body_names", []))
    forces = contact_sensor.data.net_forces_w[0]
    return {
        "label": label,
        "simulation_time_s": float(core.sim.current_time),
        "root_pos_w": _numbers(robot.data.root_pos_w[0]),
        "root_lin_vel_w": _numbers(robot.data.root_lin_vel_w[0]),
        "root_ang_vel_w": _numbers(robot.data.root_ang_vel_w[0]),
        "wheel_link_pos_w": _numbers(link_pos),
        "wheel_com_pos_w": _numbers(com_pos),
        "wheel_link_quat_w_wxyz": _numbers(body_quat),
        "wheel_bottom_gap_m": _numbers(gaps),
        "joint_pos": dict(zip(robot.joint_names, _numbers(robot.data.joint_pos[0]))),
        "joint_vel": dict(zip(robot.joint_names, _numbers(robot.data.joint_vel[0]))),
        "joint_pos_target": dict(zip(robot.joint_names, _numbers(robot.data.joint_pos_target[0]))),
        "applied_torque_Nm": dict(zip(robot.joint_names, _numbers(robot.data.applied_torque[0]))),
        "contact_body_names": force_names,
        "contact_net_forces_w_N": _numbers(forces),
    }


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = 42
    env = gym.make(args.task, cfg=cfg)
    core = env.unwrapped
    try:
        env.reset()
        robot = core.scene["robot"]
        sensor = core.scene["contact_forces"]
        wheel_body_ids, wheel_body_names = robot.find_bodies("wheel_.*", preserve_order=True)
        zero_action = torch.zeros(env.action_space.shape, device=core.device)
        core.action_manager.process_action(zero_action)
        core.action_manager.apply_action()
        core.scene.write_data_to_sim()

        trace = [_snapshot(core, robot, wheel_body_ids, sensor, "after_reset_state_write")]
        core.sim.forward()
        core.scene.update(0.0)
        trace.append(_snapshot(core, robot, wheel_body_ids, sensor, "after_forward_no_time_advance"))
        for index in range(1, args.steps + 1):
            core.action_manager.apply_action()
            core.scene.write_data_to_sim()
            core.sim.step(render=False)
            core.scene.update(float(core.cfg.sim.dt))
            trace.append(_snapshot(core, robot, wheel_body_ids, sensor, f"after_physics_step_{index}"))

        urdf_audit = _urdf_audit()
        tpu_l = next(item for item in urdf_audit["collisions"]["wheel_L"] if item["name"] == "tpu_canonical_L")
        tpu_r = next(item for item in urdf_audit["collisions"]["wheel_R"] if item["name"] == "tpu_canonical_R")
        frame_residuals = []
        for sample in trace:
            left, right = sample["wheel_link_pos_w"]
            frame_residuals.append(
                {
                    "label": sample["label"],
                    "mirror_position_residual_m": [left[0] - right[0], left[1] + right[1], left[2] - right[2]],
                    "quaternion_component_residual": [
                        a - b for a, b in zip(sample["wheel_link_quat_w_wxyz"][0], sample["wheel_link_quat_w_wxyz"][1])
                    ],
                }
            )
        first_contact_step = {}
        for wheel_name in wheel_body_names:
            first_contact_step[wheel_name] = next(
                (
                    sample["label"]
                    for sample in trace
                    if any(
                        name == wheel_name and sum(component * component for component in force) > 1.0e-12
                        for name, force in zip(sample["contact_body_names"], sample["contact_net_forces_w_N"])
                    )
                ),
                None,
            )

        report = {
            "urdf": str(ce.URDF_PATH),
            "canonical_tpu_mesh": ce.CANONICAL_TPU_MESH,
            "canonical_support_height_m": ce.canonical_wheel_support_height(0.0),
            "broadcast_support_height_m": list(ce.WHEEL_SUPPORT_DISTANCE_M),
            "support_heights_bitwise_equal": ce.WHEEL_SUPPORT_DISTANCE_M[0] == ce.WHEEL_SUPPORT_DISTANCE_M[1],
            "wheel_body_names": wheel_body_names,
            "physics_dt_s": float(core.cfg.sim.dt),
            "urdf_audit": urdf_audit,
            "symmetry_checks": {
                "canonical_tpu_collision_mirror_match": {
                    "pass": (
                        tpu_l["mesh"] == tpu_r["mesh"]
                        and tpu_l["origin_xyz"][0] == tpu_r["origin_xyz"][0]
                        and tpu_l["origin_xyz"][1] == tpu_r["origin_xyz"][1]
                        and tpu_l["origin_xyz"][2] == -tpu_r["origin_xyz"][2]
                        and tpu_l["origin_rpy"] == tpu_r["origin_rpy"]
                    ),
                    "left": tpu_l,
                    "right": tpu_r,
                },
                "nominal_phase_exact_match": {
                    "pass": ce.NOMINAL_JOINT_POSITIONS["joint_wheel_L"]
                    == ce.NOMINAL_JOINT_POSITIONS["joint_wheel_R"]
                },
                "first_contact_step": first_contact_step,
                "first_contact_step_exact_match": len(set(first_contact_step.values())) == 1,
                "runtime_frame_residuals": frame_residuals,
            },
            "cooked_usd_collision_prims": _usd_collision_audit(),
            "timeline": trace,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"CONTACT_STARTUP_TRACE={output.resolve()}")
        print(f"CANONICAL_SUPPORT_EQUAL={report['support_heights_bitwise_equal']}")
        for sample in trace:
            wheel_forces = []
            for name, force in zip(sample["contact_body_names"], sample["contact_net_forces_w_N"]):
                if name in wheel_body_names:
                    wheel_forces.append((name, force))
            print(
                sample["label"],
                "gap_m=", sample["wheel_bottom_gap_m"],
                "base_vz=", sample["root_lin_vel_w"][2],
                "wheel_forces_N=", wheel_forces,
            )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
