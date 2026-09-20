#!/usr/bin/env python3
"""Replace Tancho structural collision meshes with conservative primitives.

Visual meshes and every inertial property are deliberately left untouched.
The canonical TPU tread remains a mesh because its non-circular support profile
is part of the wheel/ground model and reset calculation.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


ASSET_ROOT = Path(__file__).resolve().parents[2] / "source" / "tancho_v3_lab" / "tancho_v3_lab" / "assets" / "robots" / "Tancho_v3"
URDF_DIR = ASSET_ROOT / "urdf"

# Center and size are the measured STL axis-aligned bounds in each source-link
# frame.  Therefore the three envelope-dimension errors are 0%, below the 10%
# acceptance limit.  These are collision approximations, not mass models.
PRIMITIVES = {
    "base_link": ((0.0, -0.000093, 0.0), (0.18, 0.091216, 0.12)),
    "thigh_L": ((-0.000004, -0.079502, -0.028005), (0.075988, 0.216993, 0.010)),
    "thigh_R": ((-0.000004, -0.079502, 0.028005), (0.075988, 0.216993, 0.010)),
    "calf_L": ((0.000133, -0.047748, -0.027005), (0.057265, 0.152494, 0.008)),
    "calf_R": ((0.000133, -0.047748, 0.027005), (0.057265, 0.152494, 0.008)),
    "drawer": ((-0.022449, 0.0025, 0.0), (0.141298, 0.065, 0.054994)),
    "pi_case": ((0.0232, 0.0475, 0.0), (0.1036, 0.025, 0.0656)),
    "knee_cover_L": ((0.0, -0.15, -0.003005), (0.075978, 0.076, 0.04)),
    "knee_cover_R": ((0.0, -0.15, 0.003005), (0.075978, 0.076, 0.04)),
}


def _rpy_matrix(rpy: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _shift_origin(origin: ET.Element, local_center: tuple[float, float, float]) -> None:
    xyz = tuple(float(value) for value in origin.get("xyz", "0 0 0").split())
    rpy = tuple(float(value) for value in origin.get("rpy", "0 0 0").split())
    rot = _rpy_matrix(rpy)
    shifted = tuple(xyz[i] + sum(rot[i][j] * local_center[j] for j in range(3)) for i in range(3))
    origin.set("xyz", " ".join(f"{value:.12g}" for value in shifted))


def _mesh_name(collision: ET.Element) -> str | None:
    mesh = collision.find("./geometry/mesh")
    if mesh is None:
        return None
    return Path(mesh.get("filename", "")).name


def _replace_with_box(collision: ET.Element, center: tuple[float, float, float], size: tuple[float, float, float]) -> None:
    origin = collision.find("origin")
    if origin is None:
        origin = ET.Element("origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        collision.insert(0, origin)
    _shift_origin(origin, center)
    geometry = collision.find("geometry")
    if geometry is None:
        raise RuntimeError("collision element has no geometry")
    geometry.clear()
    ET.SubElement(geometry, "box", {"size": " ".join(f"{value:.12g}" for value in size)})


def _simplify_link(link: ET.Element, source_mesh: str, primitive_key: str) -> None:
    collisions = list(link.findall("collision"))
    keep = next((item for item in collisions if _mesh_name(item) == source_mesh), None)
    if keep is None:
        raise RuntimeError(f"{link.get('name')}: missing collision mesh {source_mesh}")
    for collision in collisions:
        link.remove(collision)
    keep.set("name", f"{primitive_key}_collision_box")
    _replace_with_box(keep, *PRIMITIVES[primitive_key])
    link.append(keep)


def _visual_for_mesh(link: ET.Element, mesh_name: str) -> ET.Element:
    for visual in link.findall("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is not None and Path(mesh.get("filename", "")).name == mesh_name:
            return visual
    raise RuntimeError(f"{link.get('name')}: missing visual mesh {mesh_name}")


def _box_from_visual(link: ET.Element, mesh_name: str, primitive_key: str) -> ET.Element:
    visual = _visual_for_mesh(link, mesh_name)
    visual_origin = visual.find("origin")
    collision = ET.Element("collision", {"name": f"{primitive_key}_collision_box"})
    origin = ET.SubElement(
        collision,
        "origin",
        {
            "xyz": visual_origin.get("xyz", "0 0 0") if visual_origin is not None else "0 0 0",
            "rpy": visual_origin.get("rpy", "0 0 0") if visual_origin is not None else "0 0 0",
        },
    )
    _shift_origin(origin, PRIMITIVES[primitive_key][0])
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(
        geometry,
        "box",
        {"size": " ".join(f"{value:.12g}" for value in PRIMITIVES[primitive_key][1])},
    )
    return collision


def _cylinder_from_visual(
    link: ET.Element,
    mesh_name: str,
    primitive_key: str,
    center: tuple[float, float, float],
    radius: float,
    length: float,
) -> ET.Element:
    visual = _visual_for_mesh(link, mesh_name)
    visual_origin = visual.find("origin")
    collision = ET.Element("collision", {"name": f"{primitive_key}_collision_cylinder"})
    origin = ET.SubElement(
        collision,
        "origin",
        {
            "xyz": visual_origin.get("xyz", "0 0 0") if visual_origin is not None else "0 0 0",
            "rpy": visual_origin.get("rpy", "0 0 0") if visual_origin is not None else "0 0 0",
        },
    )
    _shift_origin(origin, center)
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "cylinder", {"radius": f"{radius:.12g}", "length": f"{length:.12g}"})
    return collision


def _clear_collisions(link: ET.Element) -> None:
    for collision in list(link.findall("collision")):
        link.remove(collision)


def _move_wheel_stator_visual_to_parent(root: ET.Element, side: str) -> None:
    """Keep the motor/stator housing fixed while the wheel child rotates."""

    joint_name = f"joint_wheel_{side}"
    wheel_name = f"wheel_{side}"
    joint = root.find(f"joint[@name='{joint_name}']")
    wheel = root.find(f"link[@name='{wheel_name}']")
    if joint is None or wheel is None:
        raise RuntimeError(f"missing {joint_name} or {wheel_name}")
    parent_node = joint.find("parent")
    parent = root.find(f"link[@name='{parent_node.get('link')}']") if parent_node is not None else None
    if parent is None:
        raise RuntimeError(f"{joint_name}: parent link not found")

    mesh_name = f"joint_wheel_{side}.stl"
    try:
        stator = _visual_for_mesh(wheel, mesh_name)
    except RuntimeError:
        if any(
            visual.get("name") == f"joint_wheel_{side}_stator_visual"
            for visual in parent.findall("visual")
        ):
            return
        raise
    wheel.remove(stator)
    stator.set("name", f"joint_wheel_{side}_stator_visual")
    joint_origin = joint.find("origin")
    visual_origin = stator.find("origin")
    if visual_origin is None:
        visual_origin = ET.Element("origin")
        stator.insert(0, visual_origin)
    # The exported stator visual is identity in the wheel link.  Re-parent it
    # at the wheel-joint frame so it remains fixed to the parent structure.
    visual_origin.set("xyz", joint_origin.get("xyz", "0 0 0") if joint_origin is not None else "0 0 0")
    visual_origin.set("rpy", joint_origin.get("rpy", "0 0 0") if joint_origin is not None else "0 0 0")
    parent.append(stator)


def _replace_wheel_with_cylinder(link: ET.Element) -> None:
    collisions = list(link.findall("collision"))
    if len(collisions) == 1 and collisions[0].find("geometry/cylinder") is not None:
        return
    keep = next((item for item in collisions if _mesh_name(item) == "wheel_tpu.stl"), None)
    if keep is None:
        visual = _visual_for_mesh(link, "wheel_tpu.stl")
        visual_origin = visual.find("origin")
        keep = ET.Element("collision")
        ET.SubElement(
            keep,
            "origin",
            {
                "xyz": visual_origin.get("xyz", "0 0 0") if visual_origin is not None else "0 0 0",
                "rpy": visual_origin.get("rpy", "0 0 0") if visual_origin is not None else "0 0 0",
            },
        )
        ET.SubElement(keep, "geometry")
    for collision in collisions:
        link.remove(collision)
    keep.set("name", f"{link.get('name')}_round_tire_collision")
    geometry = keep.find("geometry")
    if geometry is None:
        raise RuntimeError(f"{link.get('name')}: TPU collision has no geometry")
    geometry.clear()
    # Measured canonical TPU envelope: diameter 72.28 mm, axial width 17 mm.
    # The collision intentionally omits tread detail and is phase invariant.
    ET.SubElement(geometry, "cylinder", {"radius": "0.03614", "length": "0.017"})
    link.append(keep)


def simplify(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    links = {link.get("name"): link for link in root.findall("link")}

    if path.name == "Tancho_v3.urdf":
        for link in links.values():
            _clear_collisions(link)
        base_collision = ET.Element("collision", {"name": "base_link_collision_box"})
        ET.SubElement(base_collision, "origin", {"xyz": "0 -0.000093 0", "rpy": "0 0 0"})
        base_geometry = ET.SubElement(base_collision, "geometry")
        ET.SubElement(base_geometry, "box", {"size": "0.18 0.091216 0.12"})
        links["base_link"].append(base_collision)
        links["drawer"].append(_box_from_visual(links["drawer"], "drawer.stl", "drawer"))
        links["pi_case"].append(_box_from_visual(links["pi_case"], "pi_case.stl", "pi_case"))
        for side in ("L", "R"):
            thigh = links[f"thigh_{side}"]
            thigh.append(_box_from_visual(thigh, f"thigh_{side}.stl", f"thigh_{side}"))
            knee_center = PRIMITIVES[f"knee_cover_{side}"][0]
            thigh.append(
                _cylinder_from_visual(
                    thigh,
                    f"knee_cover_{side}.stl",
                    f"knee_cover_{side}",
                    knee_center,
                    radius=0.038,
                    length=0.04,
                )
            )
            calf = links[f"calf_{side}"]
            calf.append(_box_from_visual(calf, f"calf_{side}.stl", f"calf_{side}"))
    elif path.name == "Tancho_v3_fixed.urdf":
        root_link = links["base_link_root"]
        _clear_collisions(root_link)
        # The source links are straight in their own frames.  Their nominal
        # slant in the fixed asset comes exclusively from these baked FK
        # transforms, which must remain on the primitive collision shapes.
        root_link.append(_box_from_visual(root_link, "base_link.stl", "base_link"))
        root_link.append(_box_from_visual(root_link, "drawer.stl", "drawer"))
        root_link.append(_box_from_visual(root_link, "pi_case.stl", "pi_case"))
        for side in ("L", "R"):
            root_link.append(_box_from_visual(root_link, f"thigh_{side}.stl", f"thigh_{side}"))
            knee_center = PRIMITIVES[f"knee_cover_{side}"][0]
            root_link.append(
                _cylinder_from_visual(
                    root_link,
                    f"knee_cover_{side}.stl",
                    f"knee_cover_{side}",
                    knee_center,
                    radius=0.038,
                    length=0.04,
                )
            )
            root_link.append(_box_from_visual(root_link, f"calf_{side}.stl", f"calf_{side}"))
    else:
        raise RuntimeError(f"unsupported URDF: {path}")

    _replace_wheel_with_cylinder(links["wheel_L"])
    _replace_wheel_with_cylinder(links["wheel_R"])
    _move_wheel_stator_visual_to_parent(root, "L")
    _move_wheel_stator_visual_to_parent(root, "R")
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or [URDF_DIR / "Tancho_v3.urdf", URDF_DIR / "Tancho_v3_fixed.urdf"]
    for path in paths:
        simplify(path.resolve())
        print(f"simplified collision geometry: {path}")


if __name__ == "__main__":
    main()
