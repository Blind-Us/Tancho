"""Tancho V3 PhysX physics gates.

This diagnostic is intentionally small and self-contained.  It follows the
same ``AppLauncher``/``hydra_task_config``/registered-environment path as
``scripts/rsl_rl/analyze.py`` and only reads PhysX tensors; it does not change
the task's controller, rewards, or actuator limits.

Examples::

    python -u scripts/rsl_rl/validate_physics.py --mode gate-a --headless
    python -u scripts/rsl_rl/validate_physics.py --mode gate-c --headless
    python -u scripts/rsl_rl/validate_physics.py --mode all --headless

``--mode headless`` is accepted as a convenience alias for ``--mode all
--headless``.  Every run writes a CSV plus a JSON summary.  The final
``MACHINE_SUMMARY_JSON=...`` line is also suitable for CI log collection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI.  Keep this layout in lock-step with analyze.py: AppLauncher gets the
# launcher flags first, while Hydra receives only its own remaining arguments.
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Tancho V3 PhysX inertial (gate A) and torque (gate C) validation."
)
parser.add_argument("--task", type=str, default="TanchoV3-Flat-v0", help="Registered Tancho task name.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point used only so hydra_task_config can load the task.",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=("gate-a", "gate-c", "all", "headless"),
    default="all",
    help="Run gate-a, gate-c, both, or the headless alias for all.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="CSV output path. Default: logs/validate_physics/validate_physics_<timestamp>.csv",
)
parser.add_argument(
    "--urdf",
    type=str,
    default=None,
    help="Optional URDF path. By default the path is taken from the task config.",
)
parser.add_argument(
    "--probe_root_height",
    type=float,
    default=2.0,
    help="Airborne root height used by gate C [m].",
)
parser.add_argument(
    "--gate_c_single_dt_scale",
    type=float,
    choices=(1.0, 0.5),
    default=None,
    help="Run Gate C at one dt only. Use separate processes for 1.0 and 0.5 to avoid Isaac scene-recreation hangs.",
)
parser.add_argument(
    "--mass_rtol",
    type=float,
    default=2.0e-3,
    help="Relative tolerance for URDF-vs-PhysX mass comparison.",
)
parser.add_argument(
    "--com_atol",
    type=float,
    default=2.0e-4,
    help="Absolute tolerance for URDF-vs-PhysX COM comparison [m].",
)
parser.add_argument(
    "--inertia_rtol",
    type=float,
    default=2.0e-2,
    help="Relative tolerance for URDF-vs-PhysX full inertia comparison.",
)

# Isaac Lab launcher arguments.
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

requested_mode = args_cli.mode
if args_cli.mode == "headless":
    # AppLauncher reads this field while constructing the Kit application.
    args_cli.headless = True
    args_cli.mode = "all"

# Hydra should only see Hydra-specific arguments.
sys.argv = [sys.argv[0]] + hydra_args


# -----------------------------------------------------------------------------
# Launch Isaac Sim, then import Isaac Lab modules.  This is the same ordering
# used by analyze.py and is required by Isaac Sim's extension loader.
# -----------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

# Keep the diagnostic runnable on CPU-only validation hosts.  AppLauncher may
# still print its normal CUDA capability warning, but task tensors are created
# on the selected device below rather than failing during env construction.
if str(getattr(args_cli, "device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
    print("[WARN] CUDA was requested but is unavailable; using device=cpu for validation.", flush=True)
    args_cli.device = "cpu"

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

# IMPORTANT: this registers TanchoV3-Flat-v0/TanchoV3-Rough-v0 in Gym.
import tancho_v3_lab.tasks  # noqa: F401,E402


# -----------------------------------------------------------------------------
# Small, dependency-free report helpers
# -----------------------------------------------------------------------------


class MissingAPIError(RuntimeError):
    """Raised when a required PhysX/Isaac Lab tensor API is unavailable."""


def _json_safe(value: Any) -> Any:
    """Convert report values to strict JSON-compatible Python values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float,)):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    # numpy scalar-like values and other numeric frontends.
    try:
        if hasattr(value, "item"):
            return _json_safe(value.item())
    except Exception:
        pass
    return str(value)


def _result(name: str) -> dict[str, Any]:
    return {"name": name, "status": "FAIL", "checks": {}, "errors": [], "warnings": []}


def _check(result: dict[str, Any], name: str, passed: bool, detail: Any = None) -> None:
    result["checks"][name] = {"status": "PASS" if passed else "FAIL", "detail": _json_safe(detail)}
    if not passed:
        result["status"] = "FAIL"


def _finish_result(result: dict[str, Any]) -> dict[str, Any]:
    if result["errors"]:
        result["status"] = "FAIL"
    elif all(item.get("status") == "PASS" for item in result["checks"].values()):
        result["status"] = "PASS"
    else:
        result["status"] = "FAIL"
    return result


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _as_tensor(value: Any, *, device: str | torch.device | None = None, dtype=None) -> torch.Tensor:
    """Normalize torch, NumPy, Warp and tensor-frontends to a torch tensor."""

    if isinstance(value, torch.Tensor):
        out = value
        if device is not None:
            out = out.to(device)
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out
    if hasattr(value, "to_torch"):
        out = value.to_torch()
        if device is not None:
            out = out.to(device)
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out
    return torch.as_tensor(value, device=device, dtype=dtype)


def _normalize_mass_matrix(raw: Any, *, device: str | torch.device) -> torch.Tensor:
    """Normalize all known Isaac Lab PhysX mass-matrix layouts to [env,n,n]."""

    mass_raw = _as_tensor(raw, device=device)
    if mass_raw.ndim == 3:
        return mass_raw
    if mass_raw.ndim == 2 and mass_raw.shape[0] == 1:
        flat_width = int(round(math.sqrt(int(mass_raw.shape[1]))))
        if flat_width * flat_width != int(mass_raw.shape[1]):
            raise RuntimeError(f"Cannot reshape generalized mass matrix {tuple(mass_raw.shape)}.")
        return mass_raw.reshape(1, flat_width, flat_width)
    if mass_raw.ndim == 2 and mass_raw.shape[0] == mass_raw.shape[1]:
        return mass_raw.unsqueeze(0)
    raise RuntimeError(f"Unexpected generalized mass matrix shape: {tuple(mass_raw.shape)}")


def _generalized_joint_offset(width: int, num_joints: int) -> int:
    """Return the root-coordinate prefix; floating-base PhysX uses six DOFs."""

    if width == num_joints:
        return 0
    if width >= num_joints + 6:
        return width - num_joints
    raise RuntimeError(
        f"Unexpected generalized width {width} for {num_joints} articulation joints; "
        "expected n or n+6 (six floating-base root DOFs)."
    )


def _resolve_generalized_layout(robot: Any, matrix_width: int) -> dict[str, Any]:
    """Resolve and verify PhysX's generalized-coordinate ordering.

    ``get_generalized_mass_matrices`` is indexed in the PhysX articulation
    order, not by an arbitrary task/action ordering.  For a floating-base
    articulation PhysX defines the first six coordinates as world-frame root
    linear then angular coordinates, followed by the DOFs in
    ``shared_metatype.dof_names`` order.  Verify every part of that contract
    against the live articulation before constructing ``tau`` or ``qdd``;
    never infer a prefix only from matrix width.
    """

    num_joints = int(getattr(robot, "num_joints", -1))
    joint_names = [str(name) for name in getattr(robot, "joint_names", [])]
    if num_joints <= 0 or len(joint_names) != num_joints:
        raise RuntimeError(
            f"invalid articulation DOF metadata: num_joints={num_joints}, joint_names={joint_names!r}"
        )
    view = getattr(robot, "root_physx_view", None)
    metatype = getattr(view, "shared_metatype", None)
    metadata_names = [str(name) for name in getattr(metatype, "dof_names", [])]
    if not metadata_names:
        raise MissingAPIError("PhysX shared_metatype.dof_names is unavailable; cannot verify DOF ordering")
    if metadata_names != joint_names:
        raise RuntimeError(
            "PhysX/Isaac-Lab joint ordering mismatch: "
            f"robot.joint_names={joint_names!r}, shared_metatype.dof_names={metadata_names!r}"
        )

    fixed_base = bool(getattr(robot, "is_fixed_base", False))
    if fixed_base:
        root_dof_offset = 0
        root_dof_order: list[str] = []
        expected_width = num_joints
        convention = "fixed-base: joint DOFs only"
    else:
        root_dof_offset = 6
        root_dof_order = list(ROOT_DOF_ORDER)
        expected_width = root_dof_offset + num_joints
        convention = "floating-base PhysX: [root linear xyz world, root angular xyz world, joint DOFs]"
    if int(matrix_width) != expected_width:
        raise RuntimeError(
            "PhysX generalized matrix width disagrees with verified articulation layout: "
            f"width={matrix_width}, expected={expected_width}, fixed_base={fixed_base}, "
            f"num_joints={num_joints}"
        )

    body_names = [str(name) for name in getattr(robot, "body_names", [])]
    if not body_names:
        raise RuntimeError("PhysX body_names are unavailable; cannot verify root acceleration ordering")
    return {
        "matrix_width": int(matrix_width),
        "num_joints": num_joints,
        "fixed_base": fixed_base,
        "root_dof_offset": root_dof_offset,
        "root_dof_order": root_dof_order,
        "joint_dof_order": joint_names,
        "physx_dof_names": metadata_names,
        "physx_body_names": body_names,
        "convention": convention,
        "verified": True,
    }


def _relative_error(actual: torch.Tensor, expected: torch.Tensor, floor: float = 1.0e-12) -> float:
    denom = max(float(torch.max(torch.abs(expected)).item()), floor)
    return float(torch.max(torch.abs(actual - expected)).item()) / denom


def _to_row_value(value: Any) -> Any:
    """Keep CSV cells scalar and readable."""

    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    if isinstance(value, torch.Tensor):
        return _to_row_value(value.detach().cpu().tolist())
    return value


def _write_outputs(summary: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    if args_cli.output:
        csv_path = Path(args_cli.output)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = Path("logs") / "validate_physics" / f"validate_physics_{stamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = csv_path.with_suffix(".summary.json")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["gate", "test", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_row_value(row.get(key, "")) for key in fieldnames})

    clean_summary = _json_safe(summary)
    json_path.write_text(json.dumps(clean_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


# -----------------------------------------------------------------------------
# URDF parsing and fixed-joint aggregation
# -----------------------------------------------------------------------------


def _rpy_matrix(rpy: list[float]) -> torch.Tensor:
    """URDF roll-pitch-yaw rotation as a 3x3 float64 matrix."""

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF uses Rz(yaw) * Ry(pitch) * Rx(roll).
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float64,
    )


def _transform(xyz: list[float], rpy: list[float]) -> torch.Tensor:
    out = torch.eye(4, dtype=torch.float64)
    out[:3, :3] = _rpy_matrix(rpy)
    out[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
    return out


def _parse_xyz_rpy(element: ET.Element | None) -> torch.Tensor:
    if element is None:
        return torch.eye(4, dtype=torch.float64)
    xyz = [float(x) for x in element.attrib.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in element.attrib.get("rpy", "0 0 0").split()]
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError(f"Invalid URDF origin xyz/rpy: xyz={xyz!r}, rpy={rpy!r}")
    return _transform(xyz, rpy)


def _parse_urdf(path: Path) -> dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    links: dict[str, dict[str, Any]] = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        inertial = link.find("inertial")
        mass = 0.0
        com_tf = torch.eye(4, dtype=torch.float64)
        inertia = torch.zeros((3, 3), dtype=torch.float64)
        if inertial is not None:
            mass_node = inertial.find("mass")
            inertia_node = inertial.find("inertia")
            if mass_node is not None:
                mass = float(mass_node.attrib.get("value", "0"))
            com_tf = _parse_xyz_rpy(inertial.find("origin"))
            if inertia_node is not None:
                ixx = float(inertia_node.attrib.get("ixx", "0"))
                iyy = float(inertia_node.attrib.get("iyy", "0"))
                izz = float(inertia_node.attrib.get("izz", "0"))
                ixy = float(inertia_node.attrib.get("ixy", "0"))
                ixz = float(inertia_node.attrib.get("ixz", "0"))
                iyz = float(inertia_node.attrib.get("iyz", "0"))
                inertia = torch.tensor(
                    [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=torch.float64
                )
        links[name] = {"mass": mass, "com_tf": com_tf, "inertia": inertia}

    joints: list[dict[str, Any]] = []
    parent_joint: dict[str, dict[str, Any]] = {}
    children: dict[str, list[dict[str, Any]]] = {name: [] for name in links}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"URDF joint {joint.attrib.get('name')} lacks parent/child")
        axis_node = joint.find("axis")
        axis_values = [
            float(x)
            for x in (
                axis_node.attrib.get("xyz", "1 0 0")
                if axis_node is not None
                else "1 0 0"
            ).split()
        ]
        if len(axis_values) != 3:
            raise ValueError(f"Invalid URDF joint axis for {joint.attrib.get('name')}: {axis_values!r}")
        axis_norm = math.sqrt(sum(value * value for value in axis_values))
        if axis_norm <= 1.0e-12:
            raise ValueError(f"URDF joint {joint.attrib.get('name')} has a zero-length axis")
        item = {
            "name": joint.attrib.get("name", ""),
            "type": joint.attrib.get("type", "fixed"),
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "tf": _parse_xyz_rpy(joint.find("origin")),
            # Keep the declared axis in the report.  Effort routing is still
            # resolved through PhysX's joint id; this prevents a left/right
            # sign convention from being guessed from a joint name alone.
            "axis": tuple(value / axis_norm for value in axis_values),
        }
        joints.append(item)
        parent_joint[item["child"]] = item
        children.setdefault(item["parent"], []).append(item)

    # A body survives merge_fixed_joints if it is the URDF root or if its
    # incoming joint is non-fixed.  Every fixed descendant is aggregated into
    # that surviving body.
    dynamic_roots = [
        name
        for name in links
        if name not in parent_joint or parent_joint[name]["type"] != "fixed"
    ]
    expected_full: dict[str, dict[str, Any]] = {}
    expected_ignore_fixed: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[str]] = {}

    for body_name in dynamic_roots:
        group: list[tuple[str, torch.Tensor]] = []
        stack: list[tuple[str, torch.Tensor]] = [(body_name, torch.eye(4, dtype=torch.float64))]
        while stack:
            link_name, body_to_link = stack.pop()
            group.append((link_name, body_to_link))
            for joint in children.get(link_name, []):
                if joint["type"] == "fixed":
                    stack.append((joint["child"], body_to_link @ joint["tf"]))
        groups[body_name] = [name for name, _ in group]

        def aggregate(items: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
            total_mass = sum(float(links[name]["mass"]) for name, _ in items)
            if total_mass <= 0.0:
                return {
                    "mass": 0.0,
                    "com": torch.zeros(3, dtype=torch.float64),
                    "inertia": torch.zeros((3, 3), dtype=torch.float64),
                }
            points: list[tuple[float, torch.Tensor, torch.Tensor]] = []
            weighted_com = torch.zeros(3, dtype=torch.float64)
            for link_name, body_to_link in items:
                link = links[link_name]
                mass = float(link["mass"])
                # body frame -> link frame -> inertial principal frame.
                body_to_com = body_to_link @ link["com_tf"]
                point = body_to_com[:3, 3]
                rot = body_to_com[:3, :3]
                inertia_body_at_com = rot @ link["inertia"] @ rot.T
                points.append((mass, point, inertia_body_at_com))
                weighted_com += mass * point
            aggregate_com = weighted_com / total_mass
            aggregate_inertia = torch.zeros((3, 3), dtype=torch.float64)
            eye = torch.eye(3, dtype=torch.float64)
            for mass, point, inertia_body_at_com in points:
                delta = point - aggregate_com
                aggregate_inertia += inertia_body_at_com + mass * (
                    torch.dot(delta, delta) * eye - torch.outer(delta, delta)
                )
            return {"mass": total_mass, "com": aggregate_com, "inertia": aggregate_inertia}

        expected_full[body_name] = aggregate(group)
        # Isaac Lab sets merge_fixed_ignore_inertia=True whenever
        # merge_fixed_joints=True.  Keep this second model explicit: it lets
        # the report distinguish a valid importer policy from a stale asset.
        dynamic_only = [(name, tf) for name, tf in group if name == body_name]
        expected_ignore_fixed[body_name] = aggregate(dynamic_only)

    return {
        "robot_name": root.attrib.get("name", ""),
        "links": links,
        "joints": joints,
        "groups": groups,
        "dynamic_roots": dynamic_roots,
        "expected_full": expected_full,
        "expected_ignore_fixed": expected_ignore_fixed,
    }


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
        "size_bytes": int(stat.st_size),
    }


def _resolve_urdf(env_cfg: ManagerBasedRLEnvCfg) -> Path:
    if args_cli.urdf:
        return Path(args_cli.urdf).expanduser().resolve()
    spawn = getattr(getattr(getattr(env_cfg, "scene", None), "robot", None), "spawn", None)
    asset_path = getattr(spawn, "asset_path", None)
    if asset_path:
        candidate = Path(str(asset_path)).expanduser()
        if candidate.exists():
            return candidate.resolve()
    # A fallback keeps this diagnostic useful when a config supplies an
    # unresolved package URI instead of a filesystem path.
    fallback = Path(__file__).resolve().parents[2] / "source" / "tancho_v3_lab" / "tancho_v3_lab" / "assets" / "robots" / "Tancho_v3" / "urdf" / "Tancho_v3.urdf"
    return fallback.resolve()


# -----------------------------------------------------------------------------
# Gate A: body inertial fingerprint and generalized mass matrix
# -----------------------------------------------------------------------------


def _extract_physx_body_properties(robot: Any, base_env: Any) -> dict[str, Any]:
    view = getattr(robot, "root_physx_view", None)
    if view is None:
        raise MissingAPIError("robot.root_physx_view is unavailable")
    required = ("get_masses", "get_coms", "get_inertias", "get_generalized_mass_matrices")
    missing = [name for name in required if not callable(getattr(view, name, None))]
    if missing:
        raise MissingAPIError("missing PhysX tensor API: " + ", ".join(missing))

    body_names = list(getattr(robot, "body_names", []))
    num_bodies = int(getattr(robot, "num_bodies", len(body_names)))
    if not body_names or len(body_names) != num_bodies:
        raise RuntimeError(f"invalid robot body metadata: names={body_names!r}, num_bodies={num_bodies}")

    masses = _as_tensor(view.get_masses(), device="cpu", dtype=torch.float64)
    coms = _as_tensor(view.get_coms(), device="cpu", dtype=torch.float64)
    inertias = _as_tensor(view.get_inertias(), device="cpu", dtype=torch.float64)
    if masses.ndim == 2:
        masses = masses[0]
    elif masses.ndim == 1:
        pass
    else:
        raise RuntimeError(f"unexpected PhysX mass shape {tuple(masses.shape)}")
    if coms.ndim == 3:
        coms = coms[0]
    if inertias.ndim == 3:
        inertias = inertias[0]
    if coms.ndim != 2 or coms.shape[-1] != 7:
        raise RuntimeError(f"unexpected PhysX COM shape {tuple(coms.shape)}")
    if inertias.ndim != 2 or inertias.shape[-1] != 9:
        raise RuntimeError(f"unexpected PhysX inertia shape {tuple(inertias.shape)}")
    if masses.shape[0] < num_bodies or coms.shape[0] < num_bodies or inertias.shape[0] < num_bodies:
        raise RuntimeError(
            "PhysX body tensor count is smaller than robot.num_bodies: "
            f"m={tuple(masses.shape)}, com={tuple(coms.shape)}, I={tuple(inertias.shape)}, n={num_bodies}"
        )

    mass_matrix = _normalize_mass_matrix(view.get_generalized_mass_matrices(), device=base_env.device)
    return {
        "view": view,
        "body_names": body_names,
        "num_bodies": num_bodies,
        "masses": masses[:num_bodies],
        "coms": coms[:num_bodies],
        "inertias": inertias[:num_bodies],
        "mass_matrix": mass_matrix,
    }


def _canonical_body_name(name: str) -> str:
    text = str(name).replace("\\", "/").split("/")[-1]
    return text


def _body_fingerprint_rows(
    props: dict[str, Any],
    urdf: dict[str, Any],
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    imported_by_name = {_canonical_body_name(name): i for i, name in enumerate(props["body_names"])}
    expected_names = list(urdf["dynamic_roots"])
    fixed_groups = urdf["groups"]
    rows: list[dict[str, Any]] = []
    body_details: dict[str, Any] = {}
    all_match = len(props["body_names"]) == len(expected_names)
    extra_imported = sorted(set(imported_by_name) - set(expected_names))

    for body_name in expected_names:
        idx = imported_by_name.get(_canonical_body_name(body_name))
        # A few URDF importer versions rename a root that only has a fixed
        # predecessor to the first inertial link.  This alias is explicit in
        # the report; it is not a fuzzy or order-based body match.
        matched_name = body_name
        if idx is None and body_name == "base_link_root":
            idx = imported_by_name.get("base_link")
            if idx is not None:
                matched_name = "base_link"
        if idx is None:
            # Some importers preserve a merged body's child name.  Report the
            # unresolved mapping explicitly; guessing would create a false pass.
            all_match = False
            detail = {
                "status": "FAIL",
                "reason": f"PhysX body {body_name!r} not found",
                "group": fixed_groups[body_name],
            }
            body_details[body_name] = detail
            rows.append({"gate": "gate-a", "test": "body_inertial", "body_name": body_name, "status": "FAIL", "reason": detail["reason"]})
            continue

        mass_actual = props["masses"][idx]
        com_actual = props["coms"][idx, :3]
        inertia_actual = props["inertias"][idx].reshape(3, 3)
        candidate_stats: dict[str, Any] = {}
        for policy, expected_map in (
            ("full_fixed_merge", urdf["expected_full"]),
            ("merge_fixed_ignore_inertia", urdf["expected_ignore_fixed"]),
        ):
            expected = expected_map[body_name]
            mass_expected = torch.tensor(expected["mass"], dtype=torch.float64)
            com_expected = expected["com"]
            inertia_expected = expected["inertia"]
            mass_abs = float(torch.abs(mass_actual - mass_expected).item())
            mass_rel = mass_abs / max(abs(float(mass_expected.item())), 1.0e-12)
            com_abs = float(torch.max(torch.abs(com_actual - com_expected)).item())
            inertia_abs = float(torch.max(torch.abs(inertia_actual - inertia_expected)).item())
            inertia_scale = max(float(torch.max(torch.abs(inertia_expected)).item()), 1.0e-12)
            inertia_rel = inertia_abs / inertia_scale
            policy_pass = (
                mass_rel <= args_cli.mass_rtol
                and com_abs <= args_cli.com_atol
                and inertia_rel <= args_cli.inertia_rtol
            )
            candidate_stats[policy] = {
                "pass": policy_pass,
                "mass_expected_kg": float(mass_expected.item()),
                "mass_abs_kg": mass_abs,
                "mass_rel": mass_rel,
                "com_expected_m": com_expected,
                "com_max_abs_m": com_abs,
                "inertia_expected": inertia_expected,
                "inertia_max_abs": inertia_abs,
                "inertia_rel": inertia_rel,
            }

        # The configured importer policy is ignore-inertia when fixed links
        # are merged.  Accept the physically complete model as well, because
        # older Isaac Sim builds did not expose that importer flag consistently.
        configured = candidate_stats["merge_fixed_ignore_inertia"]
        complete = candidate_stats["full_fixed_merge"]
        matched_policy = "merge_fixed_ignore_inertia" if configured["pass"] else ("full_fixed_merge" if complete["pass"] else None)
        body_pass = matched_policy is not None
        all_match &= body_pass
        selected = candidate_stats[matched_policy] if matched_policy else configured
        body_details[body_name] = {
            "status": "PASS" if body_pass else "FAIL",
            "matched_policy": matched_policy,
            "imported_name": matched_name,
            "group": fixed_groups[body_name],
            "imported_mass_kg": float(mass_actual.item()),
            "imported_com_m": com_actual,
            "imported_inertia": inertia_actual,
            "candidates": candidate_stats,
        }
        rows.append(
            {
                "gate": "gate-a",
                "test": "body_inertial",
                "body_name": body_name,
                "status": "PASS" if body_pass else "FAIL",
                "matched_policy": matched_policy or "none",
                "merged_links": ";".join(fixed_groups[body_name]),
                "physx_mass_kg": float(mass_actual.item()),
                "expected_mass_kg": selected["mass_expected_kg"],
                "mass_rel_error": selected["mass_rel"],
                "com_x_m": float(com_actual[0].item()),
                "com_y_m": float(com_actual[1].item()),
                "com_z_m": float(com_actual[2].item()),
                "com_max_abs_error_m": selected["com_max_abs_m"],
                "inertia_max_abs_error": selected["inertia_max_abs"],
                "inertia_rel_error": selected["inertia_rel"],
            }
        )

    _check(
        result,
        "fixed_merge_body_fingerprint",
        all_match,
        {
            "status": "PASS" if all_match else "FAIL",
            "body_count_imported": len(props["body_names"]),
            "body_count_expected_after_merge": len(expected_names),
            "extra_imported_bodies": extra_imported,
            "bodies": body_details,
            "merge_fixed_joints": True,
            "merge_fixed_ignore_inertia": True,
        },
    )
    return rows, body_details


def run_gate_a(env: Any, env_cfg: ManagerBasedRLEnvCfg, urdf_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = _result("gate-a")
    try:
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
        fingerprint = _fingerprint(urdf_path)
        urdf = _parse_urdf(urdf_path)
        _check(result, "urdf_parse", True, {"links": len(urdf["links"]), "joints": len(urdf["joints"])})
        _check(result, "urdf_sha256_and_mtime", True, fingerprint)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        env.reset()
        props = _extract_physx_body_properties(robot, base_env)
        body_rows, _ = _body_fingerprint_rows(props, urdf, result)
        rows.extend(body_rows)

        mass_matrix = props["mass_matrix"]
        if mass_matrix.shape[0] < 1 or mass_matrix.shape[-1] != mass_matrix.shape[-2]:
            raise RuntimeError(f"invalid generalized mass matrix shape: {tuple(mass_matrix.shape)}")
        width = int(mass_matrix.shape[-1])
        layout = _resolve_generalized_layout(robot, width)
        num_joints = int(layout["num_joints"])
        root_offset = int(layout["root_dof_offset"])
        matrix0 = mass_matrix[0].to(dtype=torch.float64)
        sym_abs = float(torch.max(torch.abs(matrix0 - matrix0.T)).item())
        matrix_scale = max(float(torch.max(torch.abs(matrix0)).item()), 1.0e-12)
        sym_rel = sym_abs / matrix_scale
        matrix_sym = 0.5 * (matrix0 + matrix0.T)
        eig = torch.linalg.eigvalsh(matrix_sym)
        eig_min = float(torch.min(eig).item())
        eig_max = float(torch.max(eig).item())
        condition = eig_max / eig_min if eig_min > 0.0 else float("inf")
        matrix_pass = bool(
            torch.isfinite(matrix0).all().item()
            and torch.isfinite(eig).all().item()
            and root_offset in (0, 6)
            and sym_rel <= 1.0e-5
            and eig_min > 0.0
            and condition < 1.0e9
        )
        mass_detail = {
            "shape": list(matrix0.shape),
            "num_joints": num_joints,
            "root_dof_offset": root_offset,
            "root_dof_description": layout["convention"],
            "joint_dof_order": layout["joint_dof_order"],
            "physx_dof_names": layout["physx_dof_names"],
            "dof_order_verified": layout["verified"],
            "symmetry_max_abs": sym_abs,
            "symmetry_relative": sym_rel,
            "eigen_min": eig_min,
            "eigen_max": eig_max,
            "condition_number": condition,
        }
        _check(result, "generalized_mass_matrix", matrix_pass, mass_detail)
        rows.append(
            {
                "gate": "gate-a",
                "test": "generalized_mass_matrix",
                "status": "PASS" if matrix_pass else "FAIL",
                "matrix_shape": "x".join(str(x) for x in matrix0.shape),
                "root_dof_offset": root_offset,
                "joint_dof_order": layout["joint_dof_order"],
                "dof_order_verified": layout["verified"],
                "symmetry_relative": sym_rel,
                "eigen_min": eig_min,
                "eigen_max": eig_max,
                "condition_number": condition,
            }
        )

        mismatched = [name for name, item in result["checks"]["fixed_merge_body_fingerprint"]["detail"]["bodies"].items() if item.get("status") != "PASS"]
        cache_suspect = bool(mismatched)
        result["cache_diagnostics"] = {
            "urdf": fingerprint,
            "imported_body_value_mismatch": cache_suspect,
            "mismatched_bodies": mismatched,
            "hint": (
                "Imported mass/COM/inertia differs from both fixed-merge models; "
                "inspect stale generated USD/cache and re-import using the URDF fingerprint above."
                if cache_suspect
                else "No URDF-vs-imported inertial mismatch detected."
            ),
        }
        _check(result, "no_stale_imported_inertial_values", not cache_suspect, result["cache_diagnostics"])
    except MissingAPIError as exc:
        result["errors"].append(f"MISSING_API: {exc}")
        result["cache_diagnostics"] = {"missing_api": str(exc)}
    except Exception as exc:  # diagnostics must report FAIL rather than pass on partial data
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return _finish_result(result)


# -----------------------------------------------------------------------------
# Gate C: contact-free known joint torque, M^-1*tau prediction, and dt/2
# -----------------------------------------------------------------------------


PROBE_JOINT_NAMES = [
    "joint_thigh_L",
    "joint_calf_L",
    "joint_wheel_L",
    "joint_thigh_R",
    "joint_calf_R",
    "joint_wheel_R",
]
# These are the formal actuator limits in the Tancho V3 task configuration.
# Gate C intentionally probes far below the limits, while still covering the
# required wheel amplitudes and both signs for every DOF.
OFFICIAL_EFFORT_LIMITS_NM = {
    "joint_thigh_L": 12.5,
    "joint_calf_L": 12.5,
    "joint_wheel_L": 0.45,
    "joint_thigh_R": 12.5,
    "joint_calf_R": 12.5,
    "joint_wheel_R": 0.45,
}
TORQUE_AMPLITUDES = (0.05, 0.10, 0.20)
ROOT_DOF_ORDER = (
    "root_lin_x_world",
    "root_lin_y_world",
    "root_lin_z_world",
    "root_ang_x_world",
    "root_ang_y_world",
    "root_ang_z_world",
)


def _configure_probe_cfg(base_cfg: ManagerBasedRLEnvCfg, dt: float) -> ManagerBasedRLEnvCfg:
    cfg = copy.deepcopy(base_cfg)
    cfg.scene.num_envs = 1
    if hasattr(cfg.sim, "device"):
        cfg.sim.device = args_cli.device
    if hasattr(cfg.sim, "use_fabric") and args_cli.device == "cpu":
        cfg.sim.use_fabric = False
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.sim.dt = float(dt)
    cfg.decimation = 1
    cfg.episode_length_s = max(float(getattr(cfg, "episode_length_s", 20.0)), 1.0)
    # Keep the task's terrain in the scene for compatibility with its manager
    # configuration, but put the robot well above it.  Contact-free status is
    # proved from the live body clearance and contact sensor after a full
    # history warm-up; no terrain collision is allowed to enter the sample.
    state = cfg.scene.robot.init_state
    old_pos = tuple(state.pos)
    state.pos = (old_pos[0], old_pos[1], float(args_cli.probe_root_height))
    return cfg


def _contact_peak(base_env: Any) -> float:
    sensors = getattr(base_env.scene, "sensors", None)
    sensor = sensors.get("contact_forces") if sensors is not None and hasattr(sensors, "get") else None
    if sensor is None:
        raise MissingAPIError("scene contact_forces sensor is unavailable; cannot prove contact-free state")
    forces = getattr(getattr(sensor, "data", None), "net_forces_w", None)
    if forces is None:
        raise MissingAPIError("contact_forces.data.net_forces_w is unavailable")
    tensor = _as_tensor(forces, device="cpu", dtype=torch.float64)
    return float(torch.linalg.vector_norm(tensor).item())


def _zero_state_and_read(robot: Any, base_env: Any) -> dict[str, Any]:
    required = ("write_joint_state_to_sim", "write_root_link_state_to_sim")
    missing = [name for name in required if not callable(getattr(robot, name, None))]
    if missing:
        raise MissingAPIError("missing state writer API: " + ", ".join(missing))
    device = base_env.device
    joint_pos = robot.data.joint_pos[0].clone().to(device=device)
    joint_vel = torch.zeros_like(joint_pos)
    root_state = robot.data.root_link_state_w[0].clone().to(device=device)
    root_state[2] = float(args_cli.probe_root_height)
    root_state[7:] = 0.0
    robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0))
    robot.write_root_link_state_to_sim(root_state.unsqueeze(0))
    # Remove actuator-buffer history between +/- cases.  Leg position targets
    # equal the current configuration, and every feed-forward effort target is
    # zero until the selected wheel probe is injected.
    robot.set_joint_position_target(joint_pos.unsqueeze(0))
    robot.set_joint_velocity_target(joint_vel.unsqueeze(0))
    robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_effort_target))
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()

    # The task's custom reset intentionally puts the robot on its wheels,
    # overriding cfg.init_state.z.  Move it airborne and flush the complete
    # contact-sensor history with zero-force, gravity-free steps.  A single
    # warm-up step is insufficient when history_length > 1 because it retains
    # the reset contact impulse in older frames.
    sensor = getattr(getattr(base_env.scene, "sensors", None), "get", lambda _name: None)("contact_forces")
    history_length = int(getattr(getattr(sensor, "cfg", None), "history_length", 1))
    warmup_steps = max(2, history_length + 1)
    for _ in range(warmup_steps):
        robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_effort_target))
        base_env.scene.write_data_to_sim()
        base_env.sim.step(render=False)
        base_env.scene.update(float(base_env.cfg.sim.dt))

    # Define the formal measurement state after warm-up, restoring exact zero
    # velocity so finite-difference acceleration is a first-step measurement.
    root_state = robot.data.root_link_state_w[0].clone().to(device=device)
    root_state[2] = float(args_cli.probe_root_height)
    root_state[7:] = 0.0
    robot.write_joint_state_to_sim(robot.data.joint_pos[0].unsqueeze(0), joint_vel.unsqueeze(0))
    robot.write_root_link_state_to_sim(root_state.unsqueeze(0))
    robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_effort_target))
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()
    base_env.scene.update(0.0)

    # Read back the written state before the force step.  This catches a
    # backend that silently ignored the zero-velocity write.
    qd = robot.data.joint_vel[0]
    root_vel = robot.data.root_link_vel_w[0]
    return {
        "joint_vel_max_abs": float(torch.max(torch.abs(qd)).item()),
        "root_vel_max_abs": float(torch.max(torch.abs(root_vel)).item()),
        "contact_peak_before": _contact_peak(base_env),
        "root_height": float(robot.data.root_pos_w[0, 2].item()),
        "body_min_z": float(torch.min(robot.data.body_pos_w[0, :, 2]).item()),
        "contact_history_length": history_length,
        "warmup_steps": warmup_steps,
    }


def _read_generalized_acc(robot: Any, *, root_offset: int) -> torch.Tensor:
    if not hasattr(robot.data, "joint_acc") or not hasattr(robot.data, "body_acc_w"):
        raise MissingAPIError("robot.data.joint_acc/body_acc_w are required for measured qdd")
    joint_acc = robot.data.joint_acc[0]
    if root_offset == 0:
        return joint_acc.clone()
    root_acc = robot.data.body_acc_w[0, 0]
    if int(root_acc.numel()) != root_offset:
        raise RuntimeError(f"expected {root_offset} root acceleration entries, got {int(root_acc.numel())}")
    return torch.cat((root_acc, joint_acc), dim=0)


def _read_generalized_velocity(robot: Any, *, root_offset: int) -> torch.Tensor:
    """Read generalized velocity in the PhysX mass-matrix coordinate order."""

    joint_vel = robot.data.joint_vel[0]
    if root_offset == 0:
        return joint_vel.clone()
    root_vel = robot.data.root_link_vel_w[0]
    if int(root_vel.numel()) != root_offset:
        raise RuntimeError(f"expected {root_offset} root velocity entries, got {int(root_vel.numel())}")
    return torch.cat((root_vel, joint_vel), dim=0)


def _probe_joint_metadata(urdf_path: Path) -> dict[str, dict[str, Any]]:
    """Return declared axes and formal effort limits for every probe joint."""

    urdf = _parse_urdf(urdf_path)
    by_name = {str(item["name"]): item for item in urdf["joints"]}
    metadata: dict[str, dict[str, Any]] = {}
    for name in PROBE_JOINT_NAMES:
        item = by_name.get(name)
        if item is None:
            raise RuntimeError(f"URDF is missing probe joint {name!r}")
        if item.get("type") not in ("revolute", "continuous"):
            raise RuntimeError(f"Probe joint {name!r} is not a movable revolute joint: {item.get('type')!r}")
        axis = tuple(float(value) for value in item["axis"])
        axis_norm = math.sqrt(sum(value * value for value in axis))
        if abs(axis_norm - 1.0) > 1.0e-6:
            raise RuntimeError(f"URDF probe axis for {name!r} is not unit length: {axis!r}")
        metadata[name] = {
            "axis": axis,
            "effort_limit_Nm": float(OFFICIAL_EFFORT_LIMITS_NM[name]),
        }
    return metadata


def run_probe_for_dt(
    base_cfg: ManagerBasedRLEnvCfg,
    dt: float,
    rows: list[dict[str, Any]],
    urdf_path: Path,
) -> dict[str, Any]:
    """Run all +/- torque cases at one physics dt and return detailed samples."""

    cfg = _configure_probe_cfg(base_cfg, dt)
    env = None
    output: dict[str, Any] = {"dt": float(dt), "samples": [], "error": None}
    try:
        probe_metadata = _probe_joint_metadata(urdf_path)
        env = gym.make(args_cli.task, cfg=cfg)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        joint_ids, joint_names = robot.find_joints(PROBE_JOINT_NAMES, preserve_order=True)
        if len(joint_ids) != len(PROBE_JOINT_NAMES) or list(joint_names) != PROBE_JOINT_NAMES:
            raise RuntimeError(f"probe joint resolution failed: {joint_names!r}")
        view = getattr(robot, "root_physx_view", None)
        if view is None or not callable(getattr(view, "get_generalized_mass_matrices", None)):
            raise MissingAPIError("PhysX get_generalized_mass_matrices is unavailable")
        if not hasattr(robot.data, "joint_acc") or not hasattr(robot.data, "body_acc_w"):
            raise MissingAPIError("robot.data.joint_acc/body_acc_w are unavailable")
        mass_probe = _normalize_mass_matrix(view.get_generalized_mass_matrices(), device=base_env.device)
        layout = _resolve_generalized_layout(robot, int(mass_probe.shape[-1]))
        actual_effort_limits = _as_tensor(robot.data.joint_effort_limits, device="cpu", dtype=torch.float64)
        if actual_effort_limits.ndim == 2:
            actual_effort_limits = actual_effort_limits[0]
        if actual_effort_limits.ndim != 1 or actual_effort_limits.numel() != robot.num_joints:
            raise RuntimeError(
                f"unexpected joint effort-limit shape {tuple(actual_effort_limits.shape)} for {robot.num_joints} DOFs"
            )
        gravity = tuple(float(value) for value in getattr(base_env.sim.cfg, "gravity", ()))
        gravity_off_pass = len(gravity) == 3 and max(abs(value) for value in gravity) <= 1.0e-12
        if not gravity_off_pass:
            raise RuntimeError(f"probe simulation gravity is not disabled: {gravity!r}")
        for joint_slot, (joint_id, joint_name) in enumerate(zip(joint_ids, joint_names)):
            for amplitude in TORQUE_AMPLITUDES:
                for sign in (1.0, -1.0):
                    torque = float(sign * amplitude)
                    env.reset()
                    state_detail = _zero_state_and_read(robot, base_env)
                    if state_detail["root_height"] <= 0.1 or state_detail["body_min_z"] <= 0.01:
                        raise RuntimeError(
                            "airborne setup failed: "
                            f"root_z={state_detail['root_height']:.6f}, body_min_z={state_detail['body_min_z']:.6f}"
                        )
                    official_limit = float(probe_metadata[joint_name]["effort_limit_Nm"])
                    actual_limit = float(actual_effort_limits[int(joint_id)].item())
                    limit_pass = abs(torque) <= official_limit + 1.0e-12 and abs(torque) <= actual_limit + 1.0e-9
                    zero_state_pass = (
                        state_detail["joint_vel_max_abs"] <= 1.0e-7
                        and state_detail["root_vel_max_abs"] <= 1.0e-7
                        and state_detail["contact_peak_before"] <= 1.0e-4
                        and state_detail["root_height"] > 0.1
                        and state_detail["body_min_z"] > 0.01
                        and gravity_off_pass
                    )

                    mass = _normalize_mass_matrix(view.get_generalized_mass_matrices(), device=base_env.device)
                    width = int(mass.shape[-1])
                    case_layout = _resolve_generalized_layout(robot, width)
                    if case_layout != layout:
                        raise RuntimeError("PhysX generalized DOF layout changed between torque cases")
                    root_offset = int(layout["root_dof_offset"])
                    matrix = mass[0].to(dtype=torch.float64)
                    tau = torch.zeros(width, device=base_env.device, dtype=torch.float64)
                    tau[root_offset + int(joint_id)] = torque
                    qdd_pred = torch.linalg.solve(matrix, tau)
                    qd_before = _read_generalized_velocity(robot, root_offset=root_offset).to(dtype=torch.float64)
                    # Gate C injects a physical effort directly into the
                    # articulation buffer.  Manager actions stage their target
                    # one update later, which would make the first measured
                    # PhysX step appear to have zero joint acceleration.
                    effort_target = torch.zeros(
                        (1, int(robot.num_joints)), device=base_env.device, dtype=torch.float32
                    )
                    effort_target[0, int(joint_id)] = torque
                    robot.set_joint_effort_target(effort_target)
                    robot.write_data_to_sim()
                    applied_vector_before = robot.data.applied_torque[0].detach().to(dtype=torch.float64)
                    actual_torque_before = float(applied_vector_before[int(joint_id)].item())
                    other_torque_values = torch.cat(
                        (applied_vector_before[: int(joint_id)], applied_vector_before[int(joint_id) + 1 :])
                    )
                    other_torque_max = float(torch.max(torch.abs(other_torque_values)).item())
                    base_env.sim.step(render=False)
                    base_env.scene.update(float(dt))
                    qd_after = _read_generalized_velocity(robot, root_offset=root_offset).to(dtype=torch.float64)
                    # Isaac's GPU acceleration tensor is one pipeline update
                    # behind on the first force step.  With the verified zero
                    # initial velocity, Δqd/dt is the direct first-step qdd
                    # measurement and avoids that stale cache.
                    qdd_meas = (qd_after - qd_before) / float(dt)
                    if qdd_meas.numel() != qdd_pred.numel():
                        raise RuntimeError(f"measured/predicted qdd width mismatch: {qdd_meas.shape} vs {qdd_pred.shape}")
                    contact_after = _contact_peak(base_env)
                    actual_torque = actual_torque_before
                    predicted_target = float(qdd_pred[root_offset + int(joint_id)].item())
                    measured_target = float(qdd_meas[root_offset + int(joint_id)].item())
                    fit_abs = float(torch.max(torch.abs(qdd_meas - qdd_pred)).item())
                    fit_rel = _relative_error(qdd_meas, qdd_pred, floor=1.0e-6)
                    torque_error = abs(actual_torque - torque)
                    fit_pass = fit_rel <= 0.25 or fit_abs <= 0.5
                    torque_pass = torque_error <= max(2.0e-3, abs(torque) * 0.03)
                    contact_pass = contact_after <= 1.0e-4
                    name_routing_pass = layout["joint_dof_order"][int(joint_id)] == joint_name
                    axis = tuple(float(value) for value in probe_metadata[joint_name]["axis"])
                    axis_norm = math.sqrt(sum(value * value for value in axis))
                    axis_pass = abs(axis_norm - 1.0) <= 1.0e-6
                    routing_pass = (
                        name_routing_pass
                        and axis_pass
                        and torque_pass
                        and other_torque_max <= 1.0e-5
                    )
                    direction_pass = (
                        abs(predicted_target) > 1.0e-9
                        and abs(measured_target) > 1.0e-9
                        and math.copysign(1.0, predicted_target) == math.copysign(1.0, measured_target)
                        and math.copysign(1.0, actual_torque) == math.copysign(1.0, torque)
                    )
                    row = {
                        "gate": "gate-c",
                        "test": "known_torque",
                        "status": "PASS" if (zero_state_pass and fit_pass and limit_pass and contact_pass and direction_pass and routing_pass) else "FAIL",
                        "dt_s": float(dt),
                        "joint": joint_name,
                        "joint_slot": joint_slot,
                        "physx_joint_id": int(joint_id),
                        "joint_dof_order": ";".join(layout["joint_dof_order"]),
                        "root_dof_order": ";".join(layout["root_dof_order"]),
                        "dof_order_verified": bool(layout["verified"]),
                        "urdf_axis": ";".join(f"{value:.9g}" for value in axis),
                        "urdf_axis_x": axis[0],
                        "urdf_axis_y": axis[1],
                        "urdf_axis_z": axis[2],
                        "torque_Nm": torque,
                        "actual_torque_Nm": actual_torque,
                        "torque_error_Nm": torque_error,
                        "official_effort_limit_Nm": official_limit,
                        "physx_effort_limit_Nm": actual_limit,
                        "effort_limit_pass": limit_pass,
                        "other_joint_torque_max_Nm": other_torque_max,
                        "joint_name_routing_pass": name_routing_pass,
                        "joint_axis_pass": axis_pass,
                        "torque_routing_pass": routing_pass,
                        "mass_matrix_shape": "x".join(str(x) for x in matrix.shape),
                        "root_dof_offset": root_offset,
                        "qdd_pred_target_rad_s2": predicted_target,
                        "qdd_meas_target_rad_s2": measured_target,
                        "qdd_fit_max_abs_rad_s2": fit_abs,
                        "qdd_fit_relative": fit_rel,
                        "joint_qd_before_max_abs": state_detail["joint_vel_max_abs"],
                        "root_vel_before_max_abs": state_detail["root_vel_max_abs"],
                        "contact_peak_before_N": state_detail["contact_peak_before"],
                        "contact_peak_after_N": contact_after,
                        "airborne_root_height_m": state_detail["root_height"],
                        "airborne_body_min_z_m": state_detail["body_min_z"],
                        "gravity_off": gravity_off_pass,
                        "zero_state_pass": zero_state_pass,
                        "fit_pass": fit_pass,
                        "limit_pass": limit_pass,
                        "torque_pass": torque_pass,
                        "direction_pass": direction_pass,
                        "routing_pass": routing_pass,
                        "contact_free_pass": contact_pass,
                    }
                    rows.append(row)
                    output["samples"].append(
                        {
                            "joint": joint_name,
                            "joint_id": int(joint_id),
                            "amplitude": amplitude,
                            "sign": int(sign),
                            "torque": torque,
                            "pred": qdd_pred.detach().cpu(),
                            "meas": qdd_meas.detach().cpu(),
                            "target_pred": predicted_target,
                            "target_meas": measured_target,
                            "fit_rel": fit_rel,
                            "fit_abs": fit_abs,
                        }
                    )
        return output
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
        return output
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                output["error"] = output.get("error") or f"{type(exc).__name__} while closing probe env: {exc}"


def run_gate_c(base_cfg: ManagerBasedRLEnvCfg, rows: list[dict[str, Any]], urdf_path: Path) -> dict[str, Any]:
    result = _result("gate-c")
    dt = float(base_cfg.sim.dt)
    if dt <= 0.0:
        result["errors"].append(f"invalid base simulation dt={dt}")
        return _finish_result(result)

    if args_cli.gate_c_single_dt_scale is not None:
        probe_dt = dt * float(args_cli.gate_c_single_dt_scale)
        probe = run_probe_for_dt(base_cfg, probe_dt, rows, urdf_path)
        if probe.get("error"):
            result["errors"].append(f"dt={probe_dt:g}: {probe['error']}")
            return _finish_result(result)
        direct_rows = [
            row for row in rows
            if row.get("gate") == "gate-c" and row.get("test") == "known_torque"
        ]
        direct_pass = bool(direct_rows) and all(
            bool(row.get("zero_state_pass"))
            and bool(row.get("fit_pass"))
            and bool(row.get("limit_pass"))
            and bool(row.get("torque_pass"))
            and bool(row.get("direction_pass"))
            and bool(row.get("routing_pass"))
            and bool(row.get("contact_free_pass"))
            for row in direct_rows
        )
        _check(
            result,
            "known_torque_M_inverse_tau_vs_first_step_qdd",
            direct_pass,
            {"dt_s": probe_dt, "sample_count": len(direct_rows), "single_dt_process": True},
        )
        _check(
            result,
            "official_effort_limits",
            bool(direct_rows) and all(bool(row.get("limit_pass")) for row in direct_rows),
            {"limits_Nm": OFFICIAL_EFFORT_LIMITS_NM, "sample_count": len(direct_rows)},
        )
        _check(
            result,
            "joint_axis_and_torque_routing",
            bool(direct_rows) and all(bool(row.get("routing_pass")) for row in direct_rows),
            {"sample_count": len(direct_rows), "dof_order_verified": all(bool(row.get("dof_order_verified")) for row in direct_rows)},
        )
        symmetry = []
        for joint_name in PROBE_JOINT_NAMES:
            for amplitude in TORQUE_AMPLITUDES:
                pos = next((s for s in probe["samples"] if s["joint"] == joint_name and s["amplitude"] == amplitude and s["sign"] == 1), None)
                neg = next((s for s in probe["samples"] if s["joint"] == joint_name and s["amplitude"] == amplitude and s["sign"] == -1), None)
                if pos is None or neg is None:
                    symmetry.append({"joint": joint_name, "amplitude": amplitude, "pass": False, "reason": "missing pair"})
                    continue
                residual = float(torch.max(torch.abs(pos["meas"] + neg["meas"])).item())
                scale = max(float(torch.max(torch.abs(pos["pred"])).item()), 1.0e-6)
                relative = residual / scale
                symmetry.append({"joint": joint_name, "amplitude": amplitude, "max_abs": residual, "relative": relative, "pass": relative <= 0.20 or residual <= 0.5})
        symmetry_pass = bool(symmetry) and all(item["pass"] for item in symmetry)
        _check(result, "positive_negative_torque_symmetry", symmetry_pass, symmetry)
        result["warnings"].append(
            "dt convergence is evaluated by comparing the two separate --gate_c_single_dt_scale runs"
        )
        return _finish_result(result)

    samples_dt = run_probe_for_dt(base_cfg, dt, rows, urdf_path)
    samples_half = run_probe_for_dt(base_cfg, dt * 0.5, rows, urdf_path)
    if samples_dt.get("error"):
        result["errors"].append(f"dt={dt:g}: {samples_dt['error']}")
    if samples_half.get("error"):
        result["errors"].append(f"dt={dt * 0.5:g}: {samples_half['error']}")
    if result["errors"]:
        return _finish_result(result)

    samples = samples_dt["samples"] + samples_half["samples"]
    base_checks = bool(samples) and all(
        bool(row.get("zero_state_pass"))
        and bool(row.get("fit_pass"))
        and bool(row.get("limit_pass"))
        and bool(row.get("torque_pass"))
        and bool(row.get("direction_pass"))
        and bool(row.get("routing_pass"))
        and bool(row.get("contact_free_pass"))
        for row in rows
        if row.get("gate") == "gate-c" and row.get("test") == "known_torque"
    )
    _check(
        result,
        "known_torque_M_inverse_tau_vs_first_step_qdd",
        base_checks,
        {
            "sample_count": len(samples),
            "fit_relative_tolerance": 0.25,
            "fit_absolute_fallback": 0.5,
            "all_rows_pass": base_checks,
        },
    )
    direct_rows = [
        row for row in rows
        if row.get("gate") == "gate-c" and row.get("test") == "known_torque"
    ]
    _check(
        result,
        "official_effort_limits",
        bool(direct_rows) and all(bool(row.get("limit_pass")) for row in direct_rows),
        {"limits_Nm": OFFICIAL_EFFORT_LIMITS_NM, "sample_count": len(direct_rows)},
    )
    _check(
        result,
        "joint_axis_and_torque_routing",
        bool(direct_rows) and all(bool(row.get("routing_pass")) for row in direct_rows),
        {"sample_count": len(direct_rows), "dof_order_verified": all(bool(row.get("dof_order_verified")) for row in direct_rows)},
    )

    # +/- symmetry at each dt, wheel, and amplitude.  Compare the full
    # generalized acceleration vector, not only the wheel coordinate.
    symmetry_checks: list[dict[str, Any]] = []
    for probe_dt, probe_samples in ((dt, samples_dt["samples"]), (dt * 0.5, samples_half["samples"])):
        for joint_name in PROBE_JOINT_NAMES:
            for amplitude in TORQUE_AMPLITUDES:
                pos = next((s for s in probe_samples if s["joint"] == joint_name and s["amplitude"] == amplitude and s["sign"] == 1), None)
                neg = next((s for s in probe_samples if s["joint"] == joint_name and s["amplitude"] == amplitude and s["sign"] == -1), None)
                if pos is None or neg is None:
                    symmetry_checks.append({"dt_s": probe_dt, "joint": joint_name, "amplitude": amplitude, "pass": False, "reason": "missing +/- pair"})
                    continue
                residual = torch.max(torch.abs(pos["meas"] + neg["meas"])).item()
                scale = max(float(torch.max(torch.abs(pos["pred"])).item()), float(torch.max(torch.abs(neg["pred"])).item()), 1.0e-6)
                relative = float(residual) / scale
                symmetry_checks.append({"dt_s": probe_dt, "joint": joint_name, "amplitude": amplitude, "max_abs": float(residual), "relative": relative, "pass": relative <= 0.20 or residual <= 0.5})
    symmetry_pass = bool(symmetry_checks) and all(item["pass"] for item in symmetry_checks)
    _check(result, "positive_negative_torque_symmetry", symmetry_pass, symmetry_checks)

    # dt/2 convergence: measured first-step generalized qdd should approach the
    # same M^-1*tau prediction as dt shrinks.  This is intentionally evaluated
    # against the target joint coordinate and the full-vector error norm.
    convergence_checks: list[dict[str, Any]] = []
    for joint_name in PROBE_JOINT_NAMES:
        for amplitude in TORQUE_AMPLITUDES:
            for sign in (1, -1):
                coarse = next((s for s in samples_dt["samples"] if s["joint"] == joint_name and s["amplitude"] == amplitude and s["sign"] == sign), None)
                fine = next((s for s in samples_half["samples"] if s["joint"] == joint_name and s["amplitude"] == amplitude and s["sign"] == sign), None)
                if coarse is None or fine is None:
                    convergence_checks.append({"joint": joint_name, "amplitude": amplitude, "sign": sign, "pass": False, "reason": "missing dt pair"})
                    continue
                change = float(torch.max(torch.abs(coarse["meas"] - fine["meas"])).item())
                scale = max(float(torch.max(torch.abs(coarse["pred"])).item()), float(torch.max(torch.abs(fine["pred"])).item()), 1.0e-6)
                relative = change / scale
                # Finite-difference first-step acceleration is expected to
                # change O(dt); 20% is conservative for this small probe.
                convergence_checks.append({"joint": joint_name, "amplitude": amplitude, "sign": sign, "max_abs": change, "relative": relative, "pass": relative <= 0.20 or change <= 0.5})
    convergence_pass = bool(convergence_checks) and all(item["pass"] for item in convergence_checks)
    _check(result, "dt_and_dt_half_convergence", convergence_pass, {"dt_s": dt, "dt_half_s": dt * 0.5, "cases": convergence_checks})

    # Add pair-level fields to the CSV while retaining one row per direct
    # measurement above.  This keeps machine processing simple and console
    # output concise.
    for check in symmetry_checks:
        rows.append({"gate": "gate-c", "test": "torque_symmetry", "status": "PASS" if check["pass"] else "FAIL", **check})
    for check in convergence_checks:
        rows.append({"gate": "gate-c", "test": "dt_convergence", "status": "PASS" if check["pass"] else "FAIL", **check})
    return _finish_result(result)


# -----------------------------------------------------------------------------
# Main orchestration and human/machine summaries
# -----------------------------------------------------------------------------


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    del agent_cfg
    summary: dict[str, Any] = {
        "task_id": "tancho-v3-gate-ac-20260919-01",
        "task": args_cli.task,
        "requested_mode": requested_mode,
        "mode": args_cli.mode,
        "launcher_headless": bool(getattr(args_cli, "headless", False)),
        "gates": {"gate_a": {"status": "SKIPPED"}, "gate_c": {"status": "SKIPPED"}},
        "outputs": {},
    }
    rows: list[dict[str, Any]] = []
    urdf_path = _resolve_urdf(env_cfg)
    summary["urdf_path"] = str(urdf_path)

    # Ensure CPU runs do not inherit a CUDA default from the launcher.  This is
    # a device selection only; no task/controller/actuator setting is changed.
    if hasattr(env_cfg.sim, "device"):
        env_cfg.sim.device = args_cli.device
    if hasattr(env_cfg.sim, "use_fabric") and args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    if args_cli.mode in ("gate-a", "all"):
        env_a = None
        try:
            cfg_a = copy.deepcopy(env_cfg)
            cfg_a.scene.num_envs = 1
            env_a = gym.make(args_cli.task, cfg=cfg_a)
            summary["gates"]["gate_a"] = run_gate_a(env_a, cfg_a, urdf_path, rows)
        except Exception as exc:
            gate = _result("gate-a")
            gate["errors"].append(f"{type(exc).__name__}: {exc}")
            summary["gates"]["gate_a"] = _finish_result(gate)
        finally:
            if env_a is not None:
                try:
                    env_a.close()
                except Exception as exc:
                    summary["gates"]["gate_a"].setdefault("errors", []).append(
                        f"{type(exc).__name__} while closing gate-a env: {exc}"
                    )
                    summary["gates"]["gate_a"]["status"] = "FAIL"

    if args_cli.mode in ("gate-c", "all"):
        try:
            summary["gates"]["gate_c"] = run_gate_c(env_cfg, rows, urdf_path)
        except Exception as exc:
            gate = _result("gate-c")
            gate["errors"].append(f"{type(exc).__name__}: {exc}")
            summary["gates"]["gate_c"] = _finish_result(gate)

    failed_gates = [name for name, gate in summary["gates"].items() if gate.get("status") == "FAIL"]
    summary["status"] = "FAIL" if failed_gates else "PASS"
    summary["failed_gates"] = failed_gates
    csv_path, json_path = _write_outputs(summary, rows)
    summary["outputs"] = {"csv": str(csv_path.resolve()), "json": str(json_path.resolve())}
    # Rewrite once with output paths included in the machine-readable file.
    json_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n" + "=" * 96)
    print("TANCHO V3 PHYSX GATE A/C VALIDATION")
    print("=" * 96)
    print(f"Task       : {args_cli.task}")
    print(f"Mode       : {requested_mode} (effective {args_cli.mode})")
    print(f"URDF       : {urdf_path}")
    print(f"CSV        : {csv_path}")
    print(f"JSON       : {json_path}")
    direct_rows = [row for row in rows if row.get("gate") == "gate-c" and row.get("test") == "known_torque"]
    if direct_rows:
        print("\nGate-C contact-free first-step samples:")
        print("  dt[s]  joint          tau[Nm]  qdd_pred  qdd_meas  fit_rel  route  status")
        for row in direct_rows:
            print(
                f"  {float(row['dt_s']):.6f}  {str(row['joint']):<14}  "
                f"{float(row['torque_Nm']):+7.3f}  {float(row['qdd_pred_target_rad_s2']):+9.3f}  "
                f"{float(row['qdd_meas_target_rad_s2']):+9.3f}  {float(row['qdd_fit_relative']):7.4f}  "
                f"{'PASS' if row.get('routing_pass') else 'FAIL':<5}  {row['status']}"
            )
    for name in ("gate_a", "gate_c"):
        gate = summary["gates"][name]
        print(f"{name.upper():<10}: {gate.get('status')}")
        if name == "gate_c":
            for check_name, check in gate.get("checks", {}).items():
                print(f"  {check_name:<42}: {check.get('status')}")
        for error in gate.get("errors", []):
            print(f"  ERROR: {error}")
    print(f"OVERALL    : {summary['status']}")
    print("=" * 96)
    print("MACHINE_SUMMARY_JSON=" + json.dumps(_json_safe(summary), sort_keys=True, separators=(",", ":")))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
