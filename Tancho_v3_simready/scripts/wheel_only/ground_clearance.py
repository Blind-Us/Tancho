#!/usr/bin/env python3
"""Pitch (about the wheel axle) at which each non-wheel collision shape reaches flat ground.

Usage: python ground_clearance.py [<rigid-leg URDF>]

Reference frame for every number printed here:
  * the ground is the plane z = 0;
  * both wheels touch it, so the axle midpoint sits at z = wheel collision radius;
  * ``base_link_root`` is rotated about the axle by ``pitch`` only (roll = yaw = 0);
    pitch > 0 is nose-down / forward lean (root +x is forward).

Rotating about the axle keeps the wheels on the ground, so a shape touches the
ground when its lowest point reaches z = 0.  Boxes use their 8 corners,
cylinders their exact lowest rim point, meshes all vertices (trimesh).  The URDF
is only read.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
URDF_DIR = ROOT / "source/tancho_v3_lab/tancho_v3_lab/assets/robots/Tancho_v3/urdf"
BODY = "base_link_root"
WHEEL_JOINTS = ("joint_wheel_L", "joint_wheel_R")
TILT_LIMIT_DEG = 15.0
SCAN_MAX_DEG = 90.0
SCAN_STEP_DEG = 0.1

_spec = importlib.util.spec_from_file_location("build_fixed_leg_wip", ROOT / "scripts/physics/build_fixed_leg_wip.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def _shape(urdf_dir: Path, collision: ET.Element):
    """Body-frame vertices of a box/mesh, or (center, axis, radius, length) of a cylinder."""
    tf = builder._origin_transform(collision.find("origin"))
    geometry = collision.find("geometry")
    box, cylinder, mesh = geometry.find("box"), geometry.find("cylinder"), geometry.find("mesh")
    if cylinder is not None:
        return tf[:3, 3], tf[:3, 2], float(cylinder.get("radius")), float(cylinder.get("length"))
    if box is not None:
        half = 0.5 * np.fromstring(box.get("size"), sep=" ")
        local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    elif mesh is not None:
        path = urdf_dir / mesh.get("filename")
        if not path.exists():
            path = urdf_dir.parent / mesh.get("filename")
        local = np.asarray(trimesh.load(path, force="mesh", process=False).vertices)
        if mesh.get("scale"):
            local = local * np.fromstring(mesh.get("scale"), sep=" ")
    else:
        raise RuntimeError(f"Unsupported collision geometry: {collision.get('name')}")
    return local @ tf[:3, :3].T + tf[:3, 3]


def _lowest_z(shape, axle: np.ndarray, radius: float, pitch: float) -> float:
    rot = builder._axis_rotation(np.array([0.0, 1.0, 0.0]), pitch)
    if isinstance(shape, tuple):
        center, axis, cyl_radius, length = shape
        axis_z = (rot @ axis)[2]
        center_z = (rot @ (center - axle))[2]
        return radius + center_z - 0.5 * length * abs(axis_z) - cyl_radius * math.sqrt(max(0.0, 1.0 - axis_z**2))
    return radius + float(((shape - axle) @ rot.T)[:, 2].min())


def _contact_pitch(shape, axle: np.ndarray, radius: float, sign: float) -> float | None:
    """Smallest |pitch| (rad) in direction ``sign`` at which the shape touches z = 0."""
    if _lowest_z(shape, axle, radius, 0.0) <= 0.0:
        return 0.0
    step = math.radians(SCAN_STEP_DEG)
    for i in range(1, round(SCAN_MAX_DEG / SCAN_STEP_DEG) + 1):
        if _lowest_z(shape, axle, radius, sign * i * step) <= 0.0:
            lo, hi = (i - 1) * step, i * step
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if _lowest_z(shape, axle, radius, sign * mid) <= 0.0 else (mid, hi)
            return hi
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf", nargs="?", type=Path, default=URDF_DIR / "Tancho_v3_wheel_only.urdf")
    args = parser.parse_args()
    urdf = args.urdf.resolve()
    root = ET.parse(urdf).getroot()

    body = root.find(f"link[@name='{BODY}']")
    joints = [root.find(f"joint[@name='{name}']") for name in WHEEL_JOINTS]
    wheels = [root.find(f"link[@name='{j.find('child').get('link')}']") for j in joints]
    joint_tfs = [builder._origin_transform(j.find("origin")) for j in joints]
    axle = 0.5 * (joint_tfs[0][:3, 3] + joint_tfs[1][:3, 3])
    radii = [float(w.find("collision/geometry/cylinder").get("radius")) for w in wheels]
    if abs(radii[0] - radii[1]) > 1.0e-9:
        raise RuntimeError(f"Left/right wheel radii disagree: {radii}")
    radius = radii[0]

    body_mass = float(body.find("inertial/mass").get("value"))
    body_com = np.fromstring(body.find("inertial/origin").get("xyz"), sep=" ")
    total_mass, moment = body_mass, body_mass * body_com
    for wheel, tf in zip(wheels, joint_tfs):
        mass = float(wheel.find("inertial/mass").get("value"))
        com = tf[:3, :3] @ np.fromstring(wheel.find("inertial/origin").get("xyz"), sep=" ") + tf[:3, 3]
        total_mass, moment = total_mass + mass, moment + mass * com
    total_com = moment / total_mass

    def height_mm(point: np.ndarray) -> float:
        return 1000.0 * (radius + float(point[2] - axle[2]))

    shapes = {c.get("name"): _shape(urdf.parent, c) for c in body.findall("collision")}
    lever = body_com - axle

    print(f"URDF: {urdf}")
    print("Reference: ground plane z = 0, both wheel collision cylinders touching it,")
    print("           base_link_root at pitch = 0 (upright), roll = yaw = 0.\n")
    print("Heights above ground at pitch = 0:")
    print(f"  wheel axle (= wheel collision radius)    {radius * 1000:7.1f} mm")
    print(f"  hip axis (base_link_root origin)         {height_mm(np.zeros(3)):7.1f} mm")
    print(f"  body COM (base_link_root, no wheels)     {height_mm(body_com):7.1f} mm  ({body_mass:.4f} kg)")
    print(f"  whole-robot COM (body + both wheels)     {height_mm(total_com):7.1f} mm  ({total_mass:.4f} kg)")
    for name, shape in shapes.items():
        print(f"  bottom of {name:31s}{1000 * _lowest_z(shape, axle, radius, 0.0):7.1f} mm")
    print(f"  body COM is {lever[0] * 1000:+.1f} mm ahead of / {lever[2] * 1000:.1f} mm above the axle; "
          f"static balance pitch {math.degrees(-math.atan2(lever[0], lever[2])):+.2f} deg\n")

    print("First ground contact of each non-wheel collision shape, rotating about the axle:")
    first = {+1.0: (math.inf, ""), -1.0: (math.inf, "")}
    for name, shape in shapes.items():
        cells = []
        for sign in (+1.0, -1.0):
            pitch = _contact_pitch(shape, axle, radius, sign)
            deg = math.inf if pitch is None else math.degrees(pitch)
            cells.append(f">{SCAN_MAX_DEG:.0f}" if pitch is None else f"{deg:.1f}")
            if deg < first[sign][0]:
                first[sign] = (deg, name)
        print(f"  {name:38s} forward {cells[0]:>5s} deg   backward {cells[1]:>5s} deg")
    print(f"\nFirst contact forward : {first[+1.0][1]} at {first[+1.0][0]:.1f} deg")
    print(f"First contact backward: {first[-1.0][1]} at {first[-1.0][0]:.1f} deg")
    margin = min(first[+1.0][0], first[-1.0][0])
    ok = margin > TILT_LIMIT_DEG
    print(f"ONLY_WHEELS_TOUCH_WITHIN_{TILT_LIMIT_DEG:.0f}_DEG={ok} (closest non-wheel contact {margin:.1f} deg)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
