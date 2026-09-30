#!/usr/bin/env python3
"""Scan symmetric Tancho leg poses for static balance and low holding torque."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import brentq


def xyz(text: str | None) -> np.ndarray:
    return np.array([float(v) for v in (text or "0 0 0").split()], dtype=float)


def rotation_rpy(value: np.ndarray) -> np.ndarray:
    r, p, y = value
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rotation_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, d = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [[c + x*x*d, x*y*d-z*s, x*z*d+y*s], [y*x*d+z*s, c+y*y*d, y*z*d-x*s], [z*x*d-y*s, z*y*d+x*s, c+z*z*d]]
    )


def transform(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


class UrdfModel:
    def __init__(self, path: Path):
        self.path = path
        self.root = ET.parse(path).getroot()
        self.links = {item.get("name"): item for item in self.root.findall("link")}
        self.children: dict[str, list[ET.Element]] = {name: [] for name in self.links}
        child_names = set()
        for joint in self.root.findall("joint"):
            parent = joint.find("parent").get("link")
            child_names.add(joint.find("child").get("link"))
            self.children[parent].append(joint)
        self.root_name = next(iter(set(self.links) - child_names))

    def poses(self, q: dict[str, float]) -> dict[str, np.ndarray]:
        poses = {self.root_name: np.eye(4)}
        pending = [self.root_name]
        while pending:
            parent = pending.pop()
            for joint in self.children[parent]:
                origin = joint.find("origin")
                joint_tf = transform(
                    xyz(origin.get("xyz") if origin is not None else None),
                    rotation_rpy(xyz(origin.get("rpy") if origin is not None else None)),
                )
                if joint.get("type") != "fixed":
                    axis_node = joint.find("axis")
                    axis = xyz(axis_node.get("xyz") if axis_node is not None else "1 0 0")
                    joint_tf = joint_tf @ transform(np.zeros(3), rotation_axis(axis, q.get(joint.get("name"), 0.0)))
                child = joint.find("child").get("link")
                poses[child] = poses[parent] @ joint_tf
                pending.append(child)
        return poses

    def metrics(self, thigh: float, calf: float) -> dict[str, float]:
        q = {
            "joint_thigh_L": thigh, "joint_thigh_R": thigh,
            "joint_calf_L": calf, "joint_calf_R": calf,
            "joint_wheel_L": 0.0, "joint_wheel_R": 0.0,
        }
        poses = self.poses(q)
        weighted_com = np.zeros(3)
        total_mass = 0.0
        potential_mass_height = 0.0
        for name, link in self.links.items():
            inertial = link.find("inertial")
            if inertial is None:
                continue
            mass = float(inertial.find("mass").get("value"))
            origin = inertial.find("origin")
            local_com = xyz(origin.get("xyz") if origin is not None else None)
            com = (poses[name] @ np.r_[local_com, 1.0])[:3]
            weighted_com += mass * com
            potential_mass_height += mass * com[2]
            total_mass += mass
        whole_com = weighted_com / total_mass
        axle = 0.5 * (poses["wheel_L"][:3, 3] + poses["wheel_R"][:3, 3])

        wheel_link = self.links["wheel_L"]
        collision = wheel_link.find("collision")
        collision_origin = collision.find("origin")
        collision_tf = transform(
            xyz(collision_origin.get("xyz") if collision_origin is not None else None),
            rotation_rpy(xyz(collision_origin.get("rpy") if collision_origin is not None else None)),
        )
        collision_world = poses["wheel_L"] @ collision_tf
        cylinder = collision.find("geometry/cylinder")
        radius = float(cylinder.get("radius"))
        half_length = 0.5 * float(cylinder.get("length"))
        axis_z = collision_world[2, 2]
        support_z = radius * math.sqrt(max(0.0, 1.0-axis_z*axis_z)) + half_length * abs(axis_z)
        root_height = -collision_world[2, 3] + support_z

        def potential(test_q: dict[str, float]) -> float:
            test_poses = self.poses(test_q)
            return 9.81 * sum(
                float(link.find("inertial/mass").get("value"))
                * (test_poses[name] @ np.r_[xyz(link.find("inertial/origin").get("xyz")), 1.0])[2]
                for name, link in self.links.items() if link.find("inertial") is not None
            )

        eps = 1.0e-5
        torques = []
        for joint_name in ("joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R"):
            qp, qm = dict(q), dict(q)
            qp[joint_name] += eps
            qm[joint_name] -= eps
            torques.append(-(potential(qp) - potential(qm)) / (2.0 * eps))
        torque_rms = math.sqrt(sum(value*value for value in torques) / len(torques))
        return {
            "thigh_rad": thigh,
            "calf_rad": calf,
            "root_height_m": root_height,
            "com_axle_x_m": whole_com[0] - axle[0],
            "com_height_over_ground_m": whole_com[2] + root_height,
            "thigh_gravity_torque_Nm": 0.5 * (torques[0] + torques[2]),
            "calf_gravity_torque_Nm": 0.5 * (torques[1] + torques[3]),
            "leg_torque_rms_Nm": torque_rms,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=Path(__file__).resolve().parents[2] / "source/tancho_v3_lab/tancho_v3_lab/assets/robots/Tancho_v3/urdf/Tancho_v3.urdf")
    parser.add_argument("--output", type=Path, default=Path("logs/diagnostics/initial_pose_energy_scan.csv"))
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("logs/diagnostics/initial_pose_control_candidates.csv"),
    )
    parser.add_argument("--minimum-height", type=float, default=0.20)
    parser.add_argument("--maximum-height", type=float, default=0.29)
    parser.add_argument("--com-tolerance", type=float, default=0.002)
    args = parser.parse_args()
    model = UrdfModel(args.urdf)
    rows = []
    # A 0.025-rad grid is fine enough to identify the physical basin without
    # turning this deterministic diagnostic into a long simulator run.
    for thigh in np.linspace(-1.2, 0.3, 61):
        for calf in np.linspace(0.02, 2.42, 97):
            row = model.metrics(float(thigh), float(calf))
            if args.minimum_height <= row["root_height_m"] <= args.maximum_height:
                rows.append(row)
    feasible = [row for row in rows if abs(row["com_axle_x_m"]) <= args.com_tolerance]
    feasible.sort(key=lambda row: (row["leg_torque_rms_Nm"], abs(row["com_axle_x_m"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for rank, row in enumerate(feasible[:20], 1):
        print("POSE_OPT", rank, " ".join(f"{key}={value:.9g}" for key, value in row.items()))
    for thigh, calf in ((-0.1, 0.1), (-0.5, 0.87)):
        row = model.metrics(thigh, calf)
        print("POSE_REFERENCE", " ".join(f"{key}={value:.9g}" for key, value in row.items()))

    # Solve the exact COM-over-axle equality for a useful range of knee
    # flexion.  These are the candidates for the subsequent identical-push
    # dynamic experiment; no hand-selected thigh angle enters that test.
    candidates = []
    for calf in np.arange(0.10, 0.901, 0.05):
        com_error = lambda thigh: model.metrics(float(thigh), float(calf))["com_axle_x_m"]
        thigh = brentq(com_error, -1.2, 0.3)
        row = model.metrics(float(thigh), float(calf))
        row["calf_lower_limit_margin_rad"] = float(calf)
        row["thigh_nearest_limit_margin_rad"] = min(thigh + 1.57, 1.57 - thigh)
        candidates.append(row)
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    with args.candidate_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    print(f"POSE_CANDIDATE_CSV {args.candidate_output.resolve()}")
    print(f"POSE_SCAN_CSV {args.output.resolve()}")


if __name__ == "__main__":
    main()
