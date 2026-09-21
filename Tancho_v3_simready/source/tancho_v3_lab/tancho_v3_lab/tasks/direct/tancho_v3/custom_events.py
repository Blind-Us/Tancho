"""Tancho V3 reset events and geometry-derived reset metadata.

The reset pose is intentionally derived from the source URDF rather than from a
hand-tuned base-height constant.  This keeps the reset valid if a mesh, joint
origin, or nominal leg angle changes.  The event itself only writes state; it
does not step the simulator or touch episode counters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Any, Callable, Mapping, Sequence
import xml.etree.ElementTree as ET


_MODULE_DIR = Path(__file__).resolve().parent
_ASSET_DIR = _MODULE_DIR.parents[2] / "assets" / "robots" / "Tancho_v3"
URDF_PATH = _ASSET_DIR / "urdf" / "Tancho_v3.urdf"
FIXED_URDF_PATH = _ASSET_DIR / "urdf" / "Tancho_v3_fixed.urdf"

# These are the one nominal configuration used by the reset.  They are also
# the values used by the FK pass below, so the pose and height calculation stay
# tied to the same state that is written into Isaac Lab.
NOMINAL_JOINT_POSITIONS: Mapping[str, float] = {
    "joint_thigh_L": -0.50,
    "joint_calf_L": 0.87,
    "joint_wheel_L": 0.0,
    "joint_thigh_R": -0.50,
    "joint_calf_R": 0.87,
    "joint_wheel_R": 0.0,
}
WHEEL_LINK_NAMES = ("wheel_L", "wheel_R")
CANONICAL_TPU_MESH = "meshes/wheel_tpu.stl"
# A tangent mesh contact is still outside PhysX's active constraint manifold on
# the first solver iteration.  Place the support surface 0.05 mm inside the
# collision envelope: this is half of Gate B's 0.1 mm clearance tolerance and
# is expressed as a contact tolerance, not as a guessed root height.
RESET_CONTACT_PRELOAD_M = 5.0e-5


@dataclass(frozen=True)
class TanchoResetGeometry:
    """Geometry needed to put both wheels on a horizontal support plane."""

    wheel_center_rel_root_m: tuple[tuple[float, float, float], ...]
    wheel_support_radius_m: float
    wheel_support_distance_m: tuple[float, ...]

    @property
    def target_root_height_m(self) -> float:
        """Root height above a zero-height plane for the lower wheel support."""

        return (
            max(
                -center[2] + support
                for center, support in zip(self.wheel_center_rel_root_m, self.wheel_support_distance_m)
            )
            - RESET_CONTACT_PRELOAD_M
        )

    @property
    def wheel_gap_m(self) -> tuple[float, ...]:
        """Residual wheel-to-plane gaps at :attr:`target_root_height_m`."""

        return tuple(
            self.target_root_height_m + center[2] - support
            for center, support in zip(self.wheel_center_rel_root_m, self.wheel_support_distance_m)
        )


def _parse_xyz(value: str | None) -> tuple[float, float, float]:
    values = tuple(float(item) for item in (value or "0 0 0").split())
    if len(values) != 3:
        raise ValueError(f"Expected a 3-vector, got {value!r}")
    return values  # type: ignore[return-value]


def _rpy_matrix(rpy: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """Return the URDF fixed-axis roll/pitch/yaw rotation matrix."""

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _mat_vec_mul(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def _mat_mul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][inner] * right[inner][col] for inner in range(3)) for col in range(3))
        for row in range(3)
    )


def _axis_rotation(axis: Sequence[float], angle: float) -> tuple[tuple[float, float, float], ...]:
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 1.0e-12:
        raise ValueError(f"URDF joint axis has zero length: {axis!r}")
    x, y, z = (value / norm for value in axis)
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return (
        (c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s),
        (y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s),
        (z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c),
    )


def _read_stl_vertices(mesh_path: Path) -> list[tuple[float, float, float]]:
    """Read binary or ASCII STL vertices without adding a mesh dependency."""

    payload = mesh_path.read_bytes()
    # Binary STL: 80-byte header, uint32 triangle count, then 50 bytes/triangle.
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        expected_size = 84 + 50 * triangle_count
        if expected_size == len(payload):
            vertices: list[tuple[float, float, float]] = []
            for triangle_index in range(triangle_count):
                offset = 84 + triangle_index * 50 + 12
                for vertex_index in range(3):
                    vertices.append(struct.unpack_from("<3f", payload, offset + vertex_index * 12))
            return vertices

    # The repository assets are binary, but accepting ASCII keeps this helper
    # usable for replacement URDF assets during diagnostics.
    vertices = []
    for line in payload.decode("utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
    if not vertices:
        raise ValueError(f"Could not read vertices from collision mesh: {mesh_path}")
    return vertices


def _resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    # URDF package URIs are not used by this asset, but handling the prefix
    # makes the parser fail clearly if a replacement asset uses one.
    relative = filename.split("package://", 1)[-1]
    candidate = urdf_path.parent / relative
    if candidate.exists():
        return candidate
    candidate = urdf_path.parent.parent / relative
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Collision mesh {filename!r} referenced by {urdf_path} was not found")


def _wheel_support_radius(urdf_path: Path, urdf_root: ET.Element) -> float:
    """Get the maximum wheel collision radius from mesh or cylinder geometry."""

    side_radii: list[float] = []
    for link_name in WHEEL_LINK_NAMES:
        link = urdf_root.find(f"link[@name='{link_name}']")
        if link is None:
            raise RuntimeError(f"URDF is missing required wheel link {link_name!r}")
        link_radius = 0.0
        for collision in link.findall("collision"):
            cylinder = collision.find("geometry/cylinder")
            if cylinder is not None:
                link_radius = max(link_radius, float(cylinder.get("radius", "0")))
                continue
            mesh = collision.find("geometry/mesh")
            if mesh is None or mesh.get("filename") is None:
                continue
            vertices = _read_stl_vertices(_resolve_mesh_path(urdf_path, mesh.get("filename", "")))
            scale = (
                _parse_xyz(mesh.get("scale"))
                if mesh.get("scale") is not None
                else (1.0, 1.0, 1.0)
            )
            origin = collision.find("origin")
            collision_translation = _parse_xyz(origin.get("xyz") if origin is not None else None)
            collision_rotation = _rpy_matrix(_parse_xyz(origin.get("rpy") if origin is not None else None))
            for vertex in vertices:
                scaled_vertex = tuple(vertex[index] * scale[index] for index in range(3))
                rotated_vertex = _mat_vec_mul(collision_rotation, scaled_vertex)
                vertex_in_link = tuple(
                    collision_translation[index] + rotated_vertex[index] for index in range(3)
                )
                # The URDF wheel spin axis is local Z.  Therefore local XY is
                # the support circle; no guessed wheel radius is introduced.
                link_radius = max(link_radius, math.hypot(vertex_in_link[0], vertex_in_link[1]))
        if link_radius <= 0.0:
            raise RuntimeError(f"No supported wheel collision geometry found for {link_name!r}")
        side_radii.append(link_radius)

    if max(side_radii) - min(side_radii) > 1.0e-5:
        raise RuntimeError(f"Left/right wheel support radii disagree: {side_radii}")
    return sum(side_radii) / len(side_radii)


def _round_tire_support_distance(link: ET.Element, link_rotation: Sequence[Sequence[float]]) -> float:
    """Return vertical support distance of a cylindrical tire from its link origin."""

    collision = next(
        (item for item in link.findall("collision") if item.find("geometry/cylinder") is not None),
        None,
    )
    if collision is None:
        raise RuntimeError(f"{link.get('name')!r} has no cylindrical tire collision")
    cylinder = collision.find("geometry/cylinder")
    assert cylinder is not None
    radius = float(cylinder.get("radius", "0"))
    half_length = 0.5 * float(cylinder.get("length", "0"))
    origin = collision.find("origin")
    translation = _parse_xyz(origin.get("xyz") if origin is not None else None)
    collision_rotation = _rpy_matrix(_parse_xyz(origin.get("rpy") if origin is not None else None))
    world_collision_rotation = _mat_mul(link_rotation, collision_rotation)
    center_z = _mat_vec_mul(link_rotation, translation)[2]
    axis_z = world_collision_rotation[2][2]
    radial_z = radius * math.sqrt(max(0.0, 1.0 - axis_z * axis_z))
    axial_z = half_length * abs(axis_z)
    return float(-(center_z - radial_z - axial_z))


def _wheel_poses_relative_to_root(
    urdf_root: ET.Element, joint_positions: Mapping[str, float] = NOMINAL_JOINT_POSITIONS
):
    """Compute wheel link origins and rotations from nominal URDF FK."""

    joints_by_child: dict[str, ET.Element] = {}
    for joint in urdf_root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None or child.get("link") is None or parent.get("link") is None:
            raise RuntimeError(f"URDF joint is missing parent/child link: {joint.get('name')!r}")
        joints_by_child[child.get("link", "")] = joint

    root_links = {
        link.get("name")
        for link in urdf_root.findall("link")
        if link.get("name") is not None
    } - set(joints_by_child)
    if len(root_links) != 1:
        raise RuntimeError(f"Could not determine a unique URDF root link: {root_links}")
    root_link = next(iter(root_links))

    poses = []
    for wheel_link in WHEEL_LINK_NAMES:
        chain: list[ET.Element] = []
        link_name = wheel_link
        while link_name != root_link:
            try:
                joint = joints_by_child[link_name]
            except KeyError as exc:
                raise RuntimeError(f"No URDF joint path from {root_link!r} to {wheel_link!r}") from exc
            chain.append(joint)
            parent = joint.find("parent")
            link_name = parent.get("link", "") if parent is not None else ""
        position = (0.0, 0.0, 0.0)
        rotation: tuple[tuple[float, float, float], ...] = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        for joint in reversed(chain):
            origin = joint.find("origin")
            origin_xyz = _parse_xyz(origin.get("xyz") if origin is not None else None)
            origin_rpy = _rpy_matrix(_parse_xyz(origin.get("rpy") if origin is not None else None))
            origin_name = joint.get("name", "")
            joint_type = joint.get("type", "fixed")
            if joint_type == "fixed":
                joint_rotation = origin_rpy
            else:
                axis_node = joint.find("axis")
                axis = _parse_xyz(axis_node.get("xyz") if axis_node is not None else None)
                joint_rotation = _mat_mul(
                    origin_rpy,
                    _axis_rotation(axis, joint_positions.get(origin_name, 0.0)),
                )
            position = tuple(
                a + b for a, b in zip(position, _mat_vec_mul(rotation, origin_xyz))
            )  # type: ignore[assignment]
            rotation = _mat_mul(rotation, joint_rotation)
        poses.append((position, rotation))
    return tuple(poses)


def canonical_wheel_support_height(wheel_phase_rad: float) -> float:
    """Evaluate the phase-invariant support height of the round tire collider."""

    urdf_root = ET.parse(URDF_PATH).getroot()
    positions = dict(NOMINAL_JOINT_POSITIONS)
    positions["joint_wheel_L"] = float(wheel_phase_rad)
    _, rotation = _wheel_poses_relative_to_root(urdf_root, positions)[0]
    link = urdf_root.find("link[@name='wheel_L']")
    if link is None:
        raise RuntimeError("URDF is missing wheel_L")
    return _round_tire_support_distance(link, rotation)


def _load_reset_geometry() -> TanchoResetGeometry:
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"Tancho V3 URDF not found: {URDF_PATH}")
    urdf_root = ET.parse(URDF_PATH).getroot()
    left_phase = NOMINAL_JOINT_POSITIONS["joint_wheel_L"]
    right_phase = NOMINAL_JOINT_POSITIONS["joint_wheel_R"]
    if left_phase != right_phase:
        raise RuntimeError(
            "Tancho V3 is bilaterally symmetric: nominal left/right wheel phases must be exactly equal "
            f"(got {left_phase!r} and {right_phase!r})"
        )
    poses = _wheel_poses_relative_to_root(urdf_root)
    nominal_support = canonical_wheel_support_height(left_phase)
    return TanchoResetGeometry(
        wheel_center_rel_root_m=tuple(pose[0] for pose in poses),
        wheel_support_radius_m=_wheel_support_radius(URDF_PATH, urdf_root),
        # A round tire is phase invariant; one result is intentionally broadcast.
        wheel_support_distance_m=(nominal_support, nominal_support),
    )


# Parse once at module import so a changed asset fails early and every reset
# uses the same source-of-truth geometry.  The public values are intentionally
# diagnostic-friendly and do not contain a hand-tuned base-height constant.
RESET_GEOMETRY = _load_reset_geometry()
WHEEL_CENTER_REL_ROOT_M = RESET_GEOMETRY.wheel_center_rel_root_m
WHEEL_SUPPORT_RADIUS_M = RESET_GEOMETRY.wheel_support_radius_m
WHEEL_SUPPORT_DISTANCE_M = RESET_GEOMETRY.wheel_support_distance_m
TARGET_ROOT_HEIGHT_M = RESET_GEOMETRY.target_root_height_m
RESET_METADATA = {
    "urdf_path": str(URDF_PATH),
    "nominal_joint_positions_rad": dict(NOMINAL_JOINT_POSITIONS),
    "wheel_center_rel_root_m": WHEEL_CENTER_REL_ROOT_M,
    "wheel_support_radius_m": WHEEL_SUPPORT_RADIUS_M,
    "wheel_support_distance_m": WHEEL_SUPPORT_DISTANCE_M,
    "wheel_collision_geometry": "cylinder",
    "wheel_collision_radius_m": WHEEL_SUPPORT_RADIUS_M,
    "canonical_tpu_visual_mesh": CANONICAL_TPU_MESH,
    "contact_preload_m": RESET_CONTACT_PRELOAD_M,
    "target_root_height_m": TARGET_ROOT_HEIGHT_M,
    "wheel_gap_m": RESET_GEOMETRY.wheel_gap_m,
}


def _load_fixed_reset_geometry() -> TanchoResetGeometry:
    """Load the two-wheel-only asset geometry without a guessed root height."""

    if not FIXED_URDF_PATH.exists():
        raise FileNotFoundError(f"Tancho V3 fixed URDF not found: {FIXED_URDF_PATH}")
    root = ET.parse(FIXED_URDF_PATH).getroot()
    centers = []
    supports = []
    for side in ("L", "R"):
        joint = root.find(f"joint[@name='joint_wheel_{side}']")
        link = root.find(f"link[@name='wheel_{side}']")
        if joint is None or link is None:
            raise RuntimeError(f"Fixed asset is missing wheel_{side} or its joint")
        joint_origin = joint.find("origin")
        center = _parse_xyz(joint_origin.get("xyz") if joint_origin is not None else None)
        rotation = _rpy_matrix(_parse_xyz(joint_origin.get("rpy") if joint_origin is not None else None))
        centers.append(center)
        supports.append(_round_tire_support_distance(link, rotation))
    if abs(supports[0] - supports[1]) > 1.0e-9:
        raise RuntimeError(f"Fixed wheel support heights are not symmetric: {supports}")
    return TanchoResetGeometry(
        wheel_center_rel_root_m=tuple(centers),
        wheel_support_radius_m=_wheel_support_radius(FIXED_URDF_PATH, root),
        wheel_support_distance_m=(supports[0], supports[0]),
    )


FIXED_RESET_GEOMETRY = _load_fixed_reset_geometry()
FIXED_TARGET_ROOT_HEIGHT_M = FIXED_RESET_GEOMETRY.target_root_height_m
FIXED_NOMINAL_JOINT_POSITIONS = {"joint_wheel_L": 0.0, "joint_wheel_R": 0.0}


def push_tancho_along_wheel_tangent(
    env,
    env_ids,
    velocity_range: Mapping[str, tuple[float, float]],
    asset_name: str = "robot",
    debug_vis: bool = False,
    push_probability: float = 1.0,
):
    """Apply a body-frame planar velocity impulse without a yaw impulse.

    Isaac Lab's generic ``push_by_setting_velocity`` interprets x/y in the
    world frame.  Here x is Tancho's wheel rolling tangent and y is its lateral
    direction.  Both are rotated into world XY using the current root yaw;
    root angular velocity is intentionally left unchanged.
    """

    import torch

    if not 0.0 <= push_probability <= 1.0:
        raise ValueError(f"push_probability must be within [0, 1], got {push_probability}")
    if push_probability < 1.0:
        selected = torch.rand(len(env_ids), device=env.device) < push_probability
        env_ids = env_ids[selected]
        if len(env_ids) == 0:
            return

    unsupported = {key: limits for key, limits in velocity_range.items() if key not in {"x", "y"}}
    if unsupported:
        raise ValueError(
            "Tancho planar push only accepts body-frame x/y ranges; "
            f"unsupported components: {unsupported}"
        )

    asset = env.scene[asset_name]
    quat = asset.data.root_quat_w[env_ids]  # w, x, y, z
    w, x, y, z = quat.unbind(dim=-1)
    forward_xy = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
        ),
        dim=-1,
    )
    forward_xy = forward_xy / torch.linalg.vector_norm(forward_xy, dim=-1, keepdim=True).clamp_min(1.0e-9)

    x_min, x_max = velocity_range.get("x", (0.0, 0.0))
    y_min, y_max = velocity_range.get("y", (0.0, 0.0))
    delta_x = torch.empty((len(env_ids), 1), device=asset.device).uniform_(x_min, x_max)
    delta_y = torch.empty((len(env_ids), 1), device=asset.device).uniform_(y_min, y_max)
    lateral_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=-1)
    delta_velocity_xy = forward_xy * delta_x + lateral_xy * delta_y
    delta_magnitude = torch.linalg.vector_norm(delta_velocity_xy, dim=-1)
    velocity_w = asset.data.root_vel_w[env_ids].clone()
    velocity_w[:, :2] += delta_velocity_xy
    asset.write_root_velocity_to_sim(velocity_w, env_ids=env_ids)

    if debug_vis:
        # Keep the signed world direction so the continuously updated body
        # marker shows the impulse that was actually applied at that instant.
        env._tancho_last_push_world_xy = delta_velocity_xy / delta_magnitude[:, None].clamp_min(1.0e-9)
        env._tancho_last_push_magnitude = delta_magnitude
        env._tancho_push_reference_magnitude = math.hypot(
            max(abs(x_min), abs(x_max)), max(abs(y_min), abs(y_max))
        )
        env._tancho_push_visible_until_step = int(env.common_step_counter) + max(
            1, math.ceil(1.0 / env.step_dt)
        )

        if len(env_ids) == 1:
            print(
                "[TANCHO PUSH] "
                f"body_xy=({delta_x[0, 0].item():+.4f}, {delta_y[0, 0].item():+.4f}) m/s, "
                f"world_xy=({delta_velocity_xy[0, 0].item():+.4f}, "
                f"{delta_velocity_xy[0, 1].item():+.4f}) m/s"
            )


def visualize_tancho_directions(env, env_ids, asset_name: str = "robot"):
    """Draw body-forward, latest impulse, and policy wheel-torque directions.

    Blue is the current body +X rolling tangent, red is the last signed push
    direction in the world frame, and green is the translational component of
    the policy response inferred from the mean applied wheel torque.
    """

    import torch
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

    asset = env.scene[asset_name]
    if not hasattr(env, "_tancho_direction_visualizers"):
        env._tancho_direction_visualizers = {
            "normal": VisualizationMarkers(
                BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/TanchoNormalDirection")
            ),
            "push": VisualizationMarkers(
                RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/TanchoPushDirection")
            ),
            "policy": VisualizationMarkers(
                GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/TanchoPolicyDirection")
            ),
        }
        wheel_ids, _ = asset.find_joints(["joint_wheel_L", "joint_wheel_R"], preserve_order=True)
        env._tancho_wheel_joint_ids = wheel_ids

    quat = asset.data.root_quat_w[env_ids]
    w, x, y, z = quat.unbind(dim=-1)
    forward_xy = torch.stack(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)), dim=-1
    )
    forward_xy = forward_xy / torch.linalg.vector_norm(forward_xy, dim=-1, keepdim=True).clamp_min(1.0e-9)

    wheel_torque = asset.data.applied_torque[env_ids][:, env._tancho_wheel_joint_ids]
    mean_torque = wheel_torque.mean(dim=1)
    policy_xy = forward_xy * torch.where(mean_torque[:, None] >= 0.0, 1.0, -1.0)

    push_xy = getattr(env, "_tancho_last_push_world_xy", forward_xy)
    push_magnitude = getattr(
        env, "_tancho_last_push_magnitude", torch.zeros(len(env_ids), device=asset.device)
    )
    if int(env.common_step_counter) >= getattr(env, "_tancho_push_visible_until_step", -1):
        push_magnitude = torch.zeros_like(push_magnitude)
    if push_xy.shape[0] != len(env_ids):
        push_xy = push_xy[env_ids]
        push_magnitude = push_magnitude[env_ids]

    def arrow_quaternion(direction_xy):
        heading = torch.atan2(direction_xy[:, 1], direction_xy[:, 0])
        zeros = torch.zeros_like(heading)
        return torch.stack((torch.cos(0.5 * heading), zeros, zeros, torch.sin(0.5 * heading)), dim=-1)

    body_pos = asset.data.root_pos_w[env_ids].clone()
    arrow_origin = body_pos.clone()
    arrow_origin[:, 2] += 0.28

    # All arrows share one origin and one shaft thickness.  Only their length
    # changes, so magnitude can be compared without a size/position confound.
    normal_scale = torch.tensor((0.65, 0.14, 0.14), device=asset.device).repeat(len(env_ids), 1)
    push_scale = torch.ones((len(env_ids), 3), device=asset.device)
    push_scale[:, 0] = push_magnitude / max(
        getattr(env, "_tancho_push_reference_magnitude", 1.0), 1.0e-9
    )
    push_scale[:, 1:] = 0.14
    policy_scale = torch.ones((len(env_ids), 3), device=asset.device)
    policy_scale[:, 0] = mean_torque.abs() / 0.45
    policy_scale[:, 1:] = 0.14

    visualizers = env._tancho_direction_visualizers
    visualizers["normal"].visualize(arrow_origin, arrow_quaternion(forward_xy), normal_scale)
    visualizers["push"].visualize(arrow_origin, arrow_quaternion(push_xy), push_scale)
    visualizers["policy"].visualize(arrow_origin, arrow_quaternion(policy_xy), policy_scale)


def get_reset_metadata(terrain_height: float = 0.0) -> dict[str, Any]:
    """Return reset geometry metadata for diagnostics.

    ``terrain_height`` is the support-plane height in simulation/world Z.  The
    returned values are plain Python values so diagnostics can serialize them
    without depending on torch.
    """

    return {
        **RESET_METADATA,
        "terrain_height_m": float(terrain_height),
        "target_root_height_m": float(TARGET_ROOT_HEIGHT_M + terrain_height),
    }


def compute_wheel_clearance(root_height: float, terrain_height: float = 0.0) -> tuple[float, ...]:
    """Compute each wheel's vertical clearance above the support plane."""

    return tuple(
        float(root_height + center[2] - terrain_height - support)
        for center, support in zip(WHEEL_CENTER_REL_ROOT_M, WHEEL_SUPPORT_DISTANCE_M)
    )


TerrainHeightFn = Callable[[Any, Any, Any], Any]


def terrain_height_from_env_origins(env: Any, env_ids: Any, root_xy: Any = None) -> Any:
    """Use the terrain/env origin Z as a callback for generated terrain.

    This is a deliberately small extension point: flat terrain uses the
    numeric ``terrain_height=0.0`` parameter in :class:`EventCfg`, while rough
    or stair environments can pass this callback (or a raycast-backed one)
    without changing the reset FK or geometry logic.
    """

    del root_xy
    return env.scene.env_origins[env_ids, 2]


def _resolve_terrain_height(
    env: Any,
    env_ids: Any,
    root_xy: Any,
    terrain_height: Any,
    terrain_height_fn: TerrainHeightFn | None,
    *,
    device: Any,
    dtype: Any,
):
    import torch

    value = terrain_height_fn(env, env_ids, root_xy) if terrain_height_fn is not None else terrain_height
    if value is None:
        raise ValueError("terrain_height_fn returned None; return one height per environment")
    heights = torch.as_tensor(value, device=device, dtype=dtype)
    if heights.ndim == 0:
        heights = heights.expand(len(env_ids))
    elif heights.ndim == 2 and heights.shape[-1] == 1:
        heights = heights.squeeze(-1)
    if heights.shape != (len(env_ids),):
        raise ValueError(
            "terrain_height must be a scalar or a tensor with one value per reset environment; "
            f"received shape {tuple(heights.shape)} for {len(env_ids)} environments"
        )
    return heights


def reset_tancho_on_wheels(
    env: Any,
    env_ids: Any,
    terrain_height: Any = 0.0,
    terrain_height_fn: TerrainHeightFn | None = None,
    asset_cfg: Any = None,
    fixed_asset: bool = False,
) -> None:
    """Atomically reset Tancho to its nominal, stationary wheel-supported pose.

    Args:
        env: Isaac Lab manager-based environment.
        env_ids: Environment indices being reset.
        terrain_height: Flat/support-plane height in simulation/world Z.  A
            scalar applies to all selected environments.
        terrain_height_fn: Optional callback ``(env, env_ids, root_xy)`` that
            returns one support-plane height per selected environment.  This is
            the extension point for rough or stairs terrains.
        asset_cfg: Scene entity config identifying the articulation.  The
            default is the ``robot`` entity; callers normally pass
            ``SceneEntityCfg("robot")`` from ``EventCfg``.

    The event writes all six nominal joint positions, zero joint velocities,
    the FK/mesh-derived root pose, and zero root linear/angular velocities in
    one reset operation.  It intentionally does not modify episode counters;
    Isaac Lab resets those counters after reset events and before the next
    episode starts.
    """

    import torch

    if asset_cfg is None:
        asset_name = "robot"
    else:
        asset_name = asset_cfg.name
    asset = env.scene[asset_name]

    if not hasattr(asset, "write_joint_state_to_sim"):
        raise TypeError(f"Tancho wheel reset requires an articulation asset, got {type(asset)!r}")
    if not hasattr(asset, "joint_names"):
        raise TypeError("Tancho wheel reset requires articulation joint names")

    device = asset.device
    env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    if env_ids.ndim != 1:
        env_ids = env_ids.reshape(-1)
    count = len(env_ids)
    if count == 0:
        return

    geometry = FIXED_RESET_GEOMETRY if fixed_asset else RESET_GEOMETRY
    nominal_positions = FIXED_NOMINAL_JOINT_POSITIONS if fixed_asset else NOMINAL_JOINT_POSITIONS
    metadata_source = {
        "urdf_path": str(FIXED_URDF_PATH if fixed_asset else URDF_PATH),
        "nominal_joint_positions_rad": dict(nominal_positions),
        "wheel_center_rel_root_m": geometry.wheel_center_rel_root_m,
        "wheel_support_radius_m": geometry.wheel_support_radius_m,
        "wheel_support_distance_m": geometry.wheel_support_distance_m,
        "canonical_tpu_mesh": CANONICAL_TPU_MESH,
        "contact_preload_m": RESET_CONTACT_PRELOAD_M,
        "target_root_height_m": geometry.target_root_height_m,
        "wheel_gap_m": geometry.wheel_gap_m,
    }

    joint_names = list(asset.joint_names)
    joint_count = len(joint_names)
    joint_dtype = asset.data.joint_pos.dtype
    joint_pos = torch.zeros((count, joint_count), device=device, dtype=joint_dtype)
    joint_vel = torch.zeros_like(joint_pos)
    for joint_name, nominal_position in nominal_positions.items():
        try:
            joint_id = joint_names.index(joint_name)
        except ValueError as exc:
            raise RuntimeError(f"Articulation is missing required joint {joint_name!r}") from exc
        joint_pos[:, joint_id] = nominal_position

    origins = getattr(env.scene, "env_origins", None)
    if origins is None:
        root_positions = torch.zeros((count, 3), device=device, dtype=joint_dtype)
        root_xy = root_positions[:, :2]
    else:
        root_positions = origins[env_ids].to(device=device, dtype=joint_dtype).clone()
        root_xy = root_positions[:, :2]
    heights = _resolve_terrain_height(
        env,
        env_ids,
        root_xy,
        terrain_height,
        terrain_height_fn,
        device=device,
        dtype=joint_dtype,
    )
    target_root_height = geometry.target_root_height_m + heights
    # ``heights`` are world-Z support heights.  XY still follows the scene's
    # environment origin, while the root Z is placed directly in world space.
    root_positions[:, 2] = target_root_height

    root_pose = torch.zeros((count, 7), device=device, dtype=joint_dtype)
    root_pose[:, :3] = root_positions
    root_pose[:, 3] = 1.0
    root_velocity = torch.zeros((count, 6), device=device, dtype=joint_dtype)

    # The FK above is relative to the root link.  XY follows env_origins while
    # the callback/numeric height supplies the support plane's world Z.
    if hasattr(asset, "write_root_state_to_sim"):
        asset.write_root_state_to_sim(torch.cat((root_pose, root_velocity), dim=-1), env_ids=env_ids)
    else:
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=slice(None), env_ids=env_ids)

    wheel_gaps = torch.as_tensor(geometry.wheel_gap_m, device=device, dtype=joint_dtype)
    metadata = {
        **metadata_source,
        "terrain_height_m": heights.detach().clone(),
        "target_root_height_m": target_root_height.detach().clone(),
        "wheel_gap_m": wheel_gaps.detach().clone(),
        "env_ids": env_ids.detach().clone(),
    }
    # EventManager clears extras after applying reset events, so keep metadata
    # on the environment object for diagnostics that inspect post-reset state.
    env.tancho_reset_metadata = metadata


def prepare_tancho_measurement_start(
    env: Any,
    *,
    asset_name: str = "robot",
    contact_sensor_name: str = "contact_forces",
    max_warmup_steps: int = 8,
    contact_force_threshold_n: float = 1.0e-3,
) -> dict[str, Any]:
    """Finish contact warm-up, then define a stationary formal measurement t=0.

    This helper is for synchronized validation/evaluation resets, after
    :func:`reset_tancho_on_wheels` and after nominal actuator targets have been
    armed.  Warm-up physics steps are explicitly outside the recorded
    experiment.  Once both wheels have contact, the exact nominal symmetric
    pose and zero root/joint velocities are written as the formal t=0 state.
    It must not be called from an asynchronous per-env
    EventTerm because stepping advances the complete simulation scene.
    """

    import torch

    if max_warmup_steps < 1:
        raise ValueError("max_warmup_steps must be at least one")
    asset = env.scene[asset_name]
    sensor = env.scene[contact_sensor_name]
    wheel_sensor_ids = [index for index, name in enumerate(sensor.body_names) if name in WHEEL_LINK_NAMES]
    if len(wheel_sensor_ids) != 2:
        raise RuntimeError(f"Expected both wheel contact bodies, got {sensor.body_names!r}")

    history = []
    contacted = False
    stable = False
    env_ids = torch.arange(asset.num_instances, device=asset.device, dtype=torch.long)
    nominal_joint_pos = torch.zeros_like(asset.data.joint_pos)
    for joint_name, nominal_position in NOMINAL_JOINT_POSITIONS.items():
        nominal_joint_pos[:, list(asset.joint_names).index(joint_name)] = nominal_position
    for step_index in range(1, max_warmup_steps + 1):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(float(env.cfg.sim.dt))
        forces = sensor.data.net_forces_w[:, wheel_sensor_ids]
        norms = torch.linalg.vector_norm(forces, dim=-1)
        contacted = bool(torch.all(norms > contact_force_threshold_n).item())
        stable = contacted
        history.append({
            "warmup_step": step_index,
            "wheel_contact_force_norm_n": norms.detach().clone(),
            "root_lin_vel_w_m_s": asset.data.root_lin_vel_w.detach().clone(),
            "joint_vel_rad_s": asset.data.joint_vel.detach().clone(),
            "stable": stable,
        })
        if stable:
            break
    if not contacted:
        raise RuntimeError(f"Both wheel contacts were not established within {max_warmup_steps} warm-up steps")
    joint_pos = nominal_joint_pos
    joint_vel = torch.zeros_like(asset.data.joint_vel)
    root_pose = asset.data.root_pose_w.clone()
    root_pose[:, :2] = env.scene.env_origins[:, :2]
    root_pose[:, 3:] = 0.0
    root_pose[:, 3] = 1.0
    root_velocity = torch.zeros((asset.num_instances, 6), device=asset.device, dtype=joint_pos.dtype)
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.0)

    result = {
        "warmup_steps": len(history),
        "contact_established": contacted,
        "stable": stable,
        "history": history,
        "formal_t0_root_velocity": asset.data.root_vel_w.detach().clone(),
        "formal_t0_joint_velocity": asset.data.joint_vel.detach().clone(),
    }
    env.tancho_measurement_start_metadata = result
    return result
