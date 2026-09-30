#!/usr/bin/env python3
"""Build a directly viewable MuJoCo scene from the converted Tancho MJCF."""

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Tancho_v3"
SOURCE = MODEL_DIR / "Tancho_v3_model.xml"
OUTPUT = MODEL_DIR / "Tancho_v3.xml"


tree = ET.parse(SOURCE)
root = tree.getroot()
root.set("model", "Tancho_v3_scene")

compiler = root.find("compiler")
if compiler is None:
    compiler = ET.Element("compiler")
    root.insert(0, compiler)
compiler.set("angle", "radian")
compiler.set("autolimits", "true")

option = ET.Element(
    "option",
    {
        "timestep": "0.005",
        "gravity": "0 0 -9.81",
        "integrator": "implicitfast",
        "solver": "Newton",
    },
)
root.insert(1, option)

visual = ET.Element("visual")
ET.SubElement(
    visual,
    "headlight",
    {"ambient": "0.35 0.35 0.35", "diffuse": "0.8 0.8 0.8", "specular": "0.25 0.25 0.25"},
)
ET.SubElement(visual, "rgba", {"haze": "0.12 0.15 0.20 1"})
ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-20"})
root.insert(2, visual)

asset = root.find("asset")
if asset is None:
    asset = ET.SubElement(root, "asset")
ET.SubElement(asset, "texture", {"name": "sky", "type": "skybox", "builtin": "gradient", "rgb1": "0.32 0.38 0.48", "rgb2": "0.08 0.10 0.14", "width": "512", "height": "3072"})
ET.SubElement(asset, "texture", {"name": "ground_tex", "type": "2d", "builtin": "checker", "rgb1": "0.18 0.20 0.23", "rgb2": "0.28 0.31 0.35", "width": "512", "height": "512"})
ET.SubElement(asset, "material", {"name": "ground_mat", "texture": "ground_tex", "texrepeat": "8 8", "reflectance": "0.15"})

worldbody = root.find("worldbody")
if worldbody is None:
    raise RuntimeError("Converted MJCF has no worldbody")

original_children = list(worldbody)
for child in original_children:
    worldbody.remove(child)

# MuJoCo shows geom groups 0-2 by default.  URDF visual meshes are already in
# group 1; move physics-only collision geometry to group 3 so it remains active
# in dynamics without covering the rendered robot.
for node in original_children:
    for geom in node.iter("geom"):
        is_visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
        if not is_visual:
            geom.set("group", "3")

ET.SubElement(worldbody, "light", {"name": "key", "pos": "1.5 -1.5 2.5", "dir": "-0.45 0.35 -1", "diffuse": "0.9 0.9 0.9", "castshadow": "true"})
ET.SubElement(worldbody, "light", {"name": "fill", "pos": "-1 1 1.8", "dir": "0.4 -0.3 -1", "diffuse": "0.45 0.5 0.6", "castshadow": "false"})
ET.SubElement(worldbody, "geom", {"name": "ground", "type": "plane", "size": "4 4 0.1", "material": "ground_mat", "friction": "0.8 0.01 0.0001"})

base = ET.SubElement(worldbody, "body", {"name": "floating_base", "pos": "0 0 0.28553488150815215"})
ET.SubElement(base, "freejoint", {"name": "root"})
ET.SubElement(base, "camera", {"name": "follow", "pos": "1.1 -1.1 0.65", "xyaxes": "0.707 0.707 0 -0.25 0.25 0.935", "mode": "trackcom"})
for child in original_children:
    base.append(child)

actuator = ET.SubElement(root, "actuator")
for side in ("L", "R"):
    ET.SubElement(actuator, "motor", {"name": f"thigh_{side}_motor", "joint": f"joint_thigh_{side}", "ctrlrange": "-12.5 12.5"})
    ET.SubElement(actuator, "motor", {"name": f"calf_{side}_motor", "joint": f"joint_calf_{side}", "ctrlrange": "-12.5 12.5"})
    ET.SubElement(actuator, "motor", {"name": f"wheel_{side}_motor", "joint": f"joint_wheel_{side}", "ctrlrange": "-0.45 0.45"})

keyframe = ET.SubElement(root, "keyframe")
ET.SubElement(
    keyframe,
    "key",
    {
        "name": "nominal",
        "qpos": "0 0 0.28553488150815215 1 0 0 0 0 0 0 0 0 0",
        "ctrl": "0 0 0 0 0 0",
    },
)

ET.indent(tree, space="  ")
tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)
print(OUTPUT)
