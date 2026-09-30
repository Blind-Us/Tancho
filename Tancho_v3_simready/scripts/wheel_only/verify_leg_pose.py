#!/usr/bin/env python3
"""Recover the frozen leg angles from the rigid wheel-only URDF.

For each merged leg link the generated URDF stores ``T_gen = T_link(q) @ T_visual``.
Given the source URDF's joint origins/axes this is inverted to
``Rot(axis, q) = (T_parent @ T_joint)^-1 @ T_gen @ T_visual^-1`` and ``q`` is
read back.  The asset has no leg DOF, so the simulated leg angle equals this
value for the whole rollout (it can only differ if the importer mis-places the
collision/visual shapes, which ``--usd`` checks separately).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
URDF_DIR = ROOT / "source/tancho_v3_lab/tancho_v3_lab/assets/robots/Tancho_v3/urdf"
TARGET = {"joint_thigh_L": -0.50, "joint_calf_L": 0.87, "joint_thigh_R": -0.50, "joint_calf_R": 0.87}
TOL_RAD = 0.01

_spec = importlib.util.spec_from_file_location("build_fixed_leg_wip", ROOT / "scripts/physics/build_fixed_leg_wip.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def _axis_angle(rotation: np.ndarray, axis: np.ndarray) -> float:
    axis = axis / np.linalg.norm(axis)
    skew = rotation - rotation.T
    sin = 0.5 * float(axis @ np.array([skew[2, 1], skew[0, 2], skew[1, 0]]))
    cos = 0.5 * (float(np.trace(rotation)) - 1.0)
    return math.atan2(sin, cos)


def recover(source: Path, generated: Path) -> dict[str, float]:
    src = ET.parse(source).getroot()
    gen_body = ET.parse(generated).getroot().find("link[@name='base_link_root']")
    gen_visuals = {
        v.find("geometry/mesh").get("filename"): builder._origin_transform(v.find("origin"))
        for v in gen_body.findall("visual")
        if v.find("geometry/mesh") is not None
    }
    joint_by_child = {j.find("child").get("link"): j for j in src.findall("joint")}

    link_tf: dict[str, np.ndarray] = {}

    def link_transform(name: str) -> np.ndarray:
        if name == "base_link_root":
            return np.eye(4)
        if name in link_tf:
            return link_tf[name]
        joint = joint_by_child[name]
        parent = link_transform(joint.find("parent").get("link"))
        base = parent @ builder._origin_transform(joint.find("origin"))
        if joint.get("name") in TARGET:
            recovered[joint.get("name")] = q = _solve(name, joint, base)
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            base = base @ builder._transform(builder._axis_rotation(axis, q))
        link_tf[name] = base
        return base

    def _solve(link_name: str, joint: ET.Element, base: np.ndarray) -> float:
        visual = src.find(f"link[@name='{link_name}']/visual")
        mesh = visual.find("geometry/mesh").get("filename")
        t_gen = gen_visuals[mesh]
        t_local = builder._origin_transform(visual.find("origin"))
        rot = (np.linalg.inv(base) @ t_gen @ np.linalg.inv(t_local))[:3, :3]
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        return _axis_angle(rot, axis)

    recovered: dict[str, float] = {}
    for side in ("L", "R"):
        link_transform(f"calf_{side}")
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=URDF_DIR / "Tancho_v3.urdf")
    parser.add_argument("--generated", type=Path, default=URDF_DIR / "Tancho_v3_wheel_only.urdf")
    args = parser.parse_args()
    recovered = recover(args.source, args.generated)
    worst = 0.0
    for name, target in TARGET.items():
        error = recovered[name] - target
        worst = max(worst, abs(error))
        print(f"{name:14s} target={target:+.4f} recovered={recovered[name]:+.6f} error={error:+.2e} rad")
    print(f"MAX_LEG_ANGLE_ERROR_RAD={worst:.3e} (tolerance {TOL_RAD})")
    sys.exit(0 if worst < TOL_RAD else 1)


if __name__ == "__main__":
    main()
