#!/usr/bin/env python3
"""Build the fixed-leg Tancho V3 WIP URDF from the validated 6-DOF asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "source/tancho_v3_lab/tancho_v3_lab/assets/robots/Tancho_v3/urdf/Tancho_v3.urdf"
DEFAULT_OUTPUT = DEFAULT_SOURCE.with_name("Tancho_v3_fixed.urdf")
DEFAULT_MANIFEST = DEFAULT_SOURCE.with_name("Tancho_v3_fixed.json")
NOMINAL = {
    "joint_thigh_L": -0.50,
    "joint_calf_L": 0.87,
    "joint_thigh_R": -0.50,
    "joint_calf_R": 0.87,
}


def _fmt(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _matrix_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) * math.cos(angle) + (1.0 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * skew


def _transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def _origin_transform(node: ET.Element | None) -> np.ndarray:
    if node is None:
        return np.eye(4)
    xyz = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
    return _transform(_rpy_matrix(rpy), xyz)


def _set_origin(node: ET.Element, transform: np.ndarray) -> None:
    node.set("xyz", _fmt(transform[:3, 3]))
    node.set("rpy", _fmt(_matrix_rpy(transform[:3, :3])))


def _parallel_axis(inertia: np.ndarray, mass: float, offset: np.ndarray) -> np.ndarray:
    return inertia + mass * ((offset @ offset) * np.eye(3) - np.outer(offset, offset))


def _mesh_support_radius(urdf_dir: Path, root: ET.Element) -> float:
    radii: list[float] = []
    for side in ("L", "R"):
        link = root.find(f"link[@name='wheel_{side}']")
        if link is None:
            raise RuntimeError(f"wheel_{side} link missing")
        side_radii: list[float] = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            mesh_path = urdf_dir / mesh.get("filename")
            if not mesh_path.exists():
                mesh_path = urdf_dir.parent / mesh.get("filename")
            loaded = trimesh.load(mesh_path, force="mesh", process=False)
            # Wheel spin is local Z; radial support lies in local XY.
            side_radii.append(float(np.linalg.norm(np.asarray(loaded.vertices)[:, :2], axis=1).max()))
        if side_radii:
            radii.append(max(side_radii))
    if not radii:
        raise RuntimeError("No wheel collision mesh found")
    if max(radii) - min(radii) > 1.0e-5:
        raise RuntimeError(f"Left/right wheel support radii disagree: {radii}")
    return float(sum(radii) / len(radii))


def build(source: Path, output: Path, manifest: Path) -> None:
    source_root = ET.parse(source).getroot()
    joints_by_parent: dict[str, list[ET.Element]] = {}
    for joint in source_root.findall("joint"):
        joints_by_parent.setdefault(joint.find("parent").get("link"), []).append(joint)

    link_tf: dict[str, np.ndarray] = {"base_link_root": np.eye(4)}
    joint_tf: dict[str, np.ndarray] = {}
    pending = ["base_link_root"]
    while pending:
        parent = pending.pop()
        for joint in joints_by_parent.get(parent, []):
            transform = link_tf[parent] @ _origin_transform(joint.find("origin"))
            axis_node = joint.find("axis")
            if joint.get("name") in NOMINAL:
                axis = np.fromstring(axis_node.get("xyz", "0 0 1"), sep=" ")
                transform = transform @ _transform(_axis_rotation(axis, NOMINAL[joint.get("name")]))
            child = joint.find("child").get("link")
            joint_tf[joint.get("name")] = transform
            link_tf[child] = transform
            pending.append(child)

    merged_names = ["base_link", "drawer", "pi_case", "thigh_L", "calf_L", "thigh_R", "calf_R"]
    records: list[tuple[float, np.ndarray, np.ndarray]] = []
    for name in merged_names:
        link = source_root.find(f"link[@name='{name}']")
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        inertial_tf = link_tf[name] @ _origin_transform(inertial.find("origin"))
        element = inertial.find("inertia")
        local = np.array(
            [
                [float(element.get("ixx")), float(element.get("ixy")), float(element.get("ixz"))],
                [float(element.get("ixy")), float(element.get("iyy")), float(element.get("iyz"))],
                [float(element.get("ixz")), float(element.get("iyz")), float(element.get("izz"))],
            ]
        )
        rotated = inertial_tf[:3, :3] @ local @ inertial_tf[:3, :3].T
        records.append((mass, inertial_tf[:3, 3], rotated))
    merged_mass = sum(mass for mass, _, _ in records)
    merged_com = sum((mass * com for mass, com, _ in records), np.zeros(3)) / merged_mass
    inertia_origin = sum((_parallel_axis(inertia, mass, com) for mass, com, inertia in records), np.zeros((3, 3)))
    merged_inertia = inertia_origin - _parallel_axis(np.zeros((3, 3)), merged_mass, merged_com)

    output_root = ET.Element("robot", name="Tancho_v3_fixed")
    body = ET.SubElement(output_root, "link", name="base_link_root")
    inertial = ET.SubElement(body, "inertial")
    ET.SubElement(inertial, "origin", xyz=_fmt(merged_com), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{merged_mass:.12g}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{merged_inertia[0, 0]:.12g}", iyy=f"{merged_inertia[1, 1]:.12g}", izz=f"{merged_inertia[2, 2]:.12g}",
        ixy=f"{merged_inertia[0, 1]:.12g}", ixz=f"{merged_inertia[0, 2]:.12g}", iyz=f"{merged_inertia[1, 2]:.12g}",
    )
    for name in merged_names:
        source_link = source_root.find(f"link[@name='{name}']")
        for tag in ("visual", "collision"):
            for source_element in source_link.findall(tag):
                copied = ET.fromstring(ET.tostring(source_element))
                local_origin = _origin_transform(copied.find("origin"))
                origin = copied.find("origin")
                if origin is None:
                    origin = ET.Element("origin")
                    copied.insert(0, origin)
                _set_origin(origin, link_tf[name] @ local_origin)
                body.append(copied)

    for side in ("L", "R"):
        wheel_link = source_root.find(f"link[@name='wheel_{side}']")
        output_root.append(ET.fromstring(ET.tostring(wheel_link)))
        source_joint = source_root.find(f"joint[@name='joint_wheel_{side}']")
        joint = ET.SubElement(output_root, "joint", name=f"joint_wheel_{side}", type="continuous")
        origin = ET.SubElement(joint, "origin")
        _set_origin(origin, joint_tf[f"joint_wheel_{side}"])
        ET.SubElement(joint, "parent", link="base_link_root")
        ET.SubElement(joint, "child", link=f"wheel_{side}")
        ET.SubElement(joint, "axis", xyz=source_joint.find("axis").get("xyz"))
        source_limit = source_joint.find("limit")
        ET.SubElement(joint, "limit", effort="0.45", velocity=source_limit.get("velocity"))

    tree = ET.ElementTree(output_root)
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)

    parsed = ET.parse(output).getroot()
    movable = [joint.get("name") for joint in parsed.findall("joint") if joint.get("type") != "fixed"]
    if movable != ["joint_wheel_L", "joint_wheel_R"]:
        raise RuntimeError(f"Unexpected movable joints: {movable}")
    support_radius = _mesh_support_radius(output.parent, parsed)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_urdf": str(source.resolve()),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "generated_urdf": str(output.resolve()),
                "generated_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "nominal_joint_positions_rad": NOMINAL,
                "movable_joints": movable,
                "wheel_effort_limit_nm": 0.45,
                "wheel_collision_support_radius_m": support_radius,
                "merged_body_mass_kg": merged_mass,
                "merged_body_com_root_m": merged_com.tolist(),
                "merged_body_inertia_com_root_kg_m2": merged_inertia.tolist(),
                "merge_fixed_joints_required": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"FIXED_WIP_URDF={output}")
    print(f"FIXED_WIP_MANIFEST={manifest}")
    print(f"MOVABLE_JOINTS={movable}")
    print(f"WHEEL_SUPPORT_RADIUS_M={support_radius:.9f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()
