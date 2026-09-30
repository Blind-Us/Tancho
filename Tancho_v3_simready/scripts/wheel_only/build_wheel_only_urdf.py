#!/usr/bin/env python3
"""Build the rigid-leg asset for ``TanchoV3-WheelOnly-Flat-v0``.

The mass/inertia merge is delegated to ``scripts/physics/build_fixed_leg_wip.py``
(thigh=-0.50 rad, calf=+0.87 rad, both sides).  That builder only understands
mesh wheel collisions; the current asset uses round-tire cylinders, so the
support-radius helper is replaced by one that also reads ``<cylinder>``.

Output (new files only; existing assets are not touched):
    assets/robots/Tancho_v3/urdf/Tancho_v3_wheel_only.urdf
    assets/robots/Tancho_v3/urdf/Tancho_v3_wheel_only.json
"""

from __future__ import annotations

import argparse
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URDF_DIR = ROOT / "source/tancho_v3_lab/tancho_v3_lab/assets/robots/Tancho_v3/urdf"

_spec = importlib.util.spec_from_file_location("build_fixed_leg_wip", ROOT / "scripts/physics/build_fixed_leg_wip.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)
_mesh_support_radius = builder._mesh_support_radius


def _support_radius(urdf_dir: Path, root: ET.Element) -> float:
    radii = []
    for side in ("L", "R"):
        cylinder = root.find(f"link[@name='wheel_{side}']/collision/geometry/cylinder")
        if cylinder is None:
            return _mesh_support_radius(urdf_dir, root)
        radii.append(float(cylinder.get("radius")))
    if abs(radii[0] - radii[1]) > 1.0e-9:
        raise RuntimeError(f"Left/right wheel radii disagree: {radii}")
    return radii[0]


builder._mesh_support_radius = _support_radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=URDF_DIR / "Tancho_v3.urdf")
    parser.add_argument("--output", type=Path, default=URDF_DIR / "Tancho_v3_wheel_only.urdf")
    parser.add_argument("--manifest", type=Path, default=URDF_DIR / "Tancho_v3_wheel_only.json")
    args = parser.parse_args()
    builder.build(args.source.resolve(), args.output.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()
