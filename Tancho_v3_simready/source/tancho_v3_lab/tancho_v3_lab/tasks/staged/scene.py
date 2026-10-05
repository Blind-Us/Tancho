"""Scene shared by every stage: flat plane, light, IMU and the two robot assets.

* Wheel-only: ``Tancho_v3_wheel_only.urdf``.  Base and both legs at
  thigh=-0.50 / calf=+0.87 rad are merged into one rigid body (built by
  ``scripts/wheel_only/build_wheel_only_urdf.py``); only the wheels move.
* Full (6-DOF): ``Tancho_v3.urdf`` with the same nominal leg pose as the reset
  and as the zero action, so the two stages start from the same physical state.

Timing for all stages: 200 Hz PhysX, 50 Hz policy (decimation 4).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..direct.tancho_v3.custom_events import compute_reset_root_height

URDF_DIR = Path(__file__).resolve().parents[2] / "assets" / "robots" / "Tancho_v3" / "urdf"
WHEEL_ONLY_URDF = URDF_DIR / "Tancho_v3_wheel_only.urdf"
FULL_URDF = URDF_DIR / "Tancho_v3.urdf"

# -- timing -------------------------------------------------------------------
SIM_DT_S = 0.005  # 200 Hz PhysX
DECIMATION = 4  # 50 Hz policy / action update (0.02 s)
EPISODE_LENGTH_S = 20.0

# -- joints ---------------------------------------------------------------------
WHEEL_JOINTS = ["joint_wheel_L", "joint_wheel_R"]
LEG_JOINTS = ["joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R"]
# Same angles the wheel-only asset is frozen at.
NOMINAL_THIGH_RAD = -0.50
NOMINAL_CALF_RAD = 0.87
NOMINAL_JOINT_POS = {
    "joint_thigh_L": NOMINAL_THIGH_RAD,
    "joint_calf_L": NOMINAL_CALF_RAD,
    "joint_wheel_L": 0.0,
    "joint_thigh_R": NOMINAL_THIGH_RAD,
    "joint_calf_R": NOMINAL_CALF_RAD,
    "joint_wheel_R": 0.0,
}

# -- DM-H3510 wheel motor (velocity mode) ------------------------------------------
WHEEL_EFFORT_LIMIT_NM = 0.45
WHEEL_VELOCITY_LIMIT_RAD_S = 188.0
WHEEL_VELOCITY_DAMPING = 4.0  # velocity-loop Kd, Kp = 0
WHEEL_RATED_SPEED_RAD_S = 52.3598776  # 500 rpm; action = +/-1 -> +/-rated speed
WHEEL_RADIUS_M = 0.03614

# -- DM-J4310 leg motor (position mode, hardware team's gains) ----------------------
LEG_STIFFNESS = 20.0
LEG_DAMPING = 0.2
LEG_EFFORT_LIMIT_NM = 12.5
LEG_VELOCITY_LIMIT_RAD_S = 12.5
LEG_ACTION_SCALE_RAD = 0.25

# -- IMU mounting on base_link_root ---------------------------------------------------
IMU_POS_ROOT = (-0.00835741999, 0.0000000160456, -0.0294337942)
IMU_ROT_ROOT = (0.707106781, 0.707106781, 0.0, 0.0)

# Ground 0.8 x robot material 0.8 (multiply) -> nominal effective friction 0.64.
NOMINAL_FRICTION = 0.8
# Spawn 1 mm above contact so a +/-0.05 rad reset pitch never starts in penetration.
RESET_CLEARANCE_M = 0.001
FAILURE_TILT_RAD = math.radians(15.0)


def _wheel_only_root_height(urdf_path: Path) -> float:
    """Root Z that puts both round-tire cylinders tangent to z=0 at zero pitch."""
    root = ET.parse(urdf_path).getroot()
    heights = []
    for side in ("L", "R"):
        axle_z = float(root.find(f"joint[@name='joint_wheel_{side}']/origin").get("xyz").split()[2])
        radius = float(root.find(f"link[@name='wheel_{side}']/collision/geometry/cylinder").get("radius"))
        heights.append(radius - axle_z)
    if abs(heights[0] - heights[1]) > 1.0e-9:
        raise RuntimeError(f"Asymmetric wheel support heights: {heights}")
    return heights[0]


WHEEL_ONLY_ROOT_HEIGHT_M = _wheel_only_root_height(WHEEL_ONLY_URDF)
# URDF forward kinematics + tire collision geometry at the nominal leg pose.
FULL_ROOT_HEIGHT_M = compute_reset_root_height(NOMINAL_JOINT_POS)

_WHEEL_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=WHEEL_JOINTS,
    stiffness=0.0,
    damping=WHEEL_VELOCITY_DAMPING,
    effort_limit_sim=WHEEL_EFFORT_LIMIT_NM,
    velocity_limit_sim=WHEEL_VELOCITY_LIMIT_RAD_S,
)
_LEG_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=LEG_JOINTS,
    stiffness=LEG_STIFFNESS,
    damping=LEG_DAMPING,
    effort_limit_sim=LEG_EFFORT_LIMIT_NM,
    velocity_limit_sim=LEG_VELOCITY_LIMIT_RAD_S,
)


def _robot_cfg(urdf: Path, root_height: float, joints: list[str], actuators: dict, contact_sensors: bool) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf),
            fix_base=False,
            merge_fixed_joints=True,
            # Adjacent hip/knee parts share their mechanical envelope; self-contact
            # would only create artificial internal impulses.
            self_collision=False,
            # Force-type drives.  ``joint_drive=None`` leaves the importer's
            # acceleration-type drive, where PhysX scales Kp/Kd by the tiny link
            # inertia (~1e-4 kg*m^2): the legs then fold under their own weight
            # while the logged ``applied_torque`` still reads 12.5 N*m.  Gains
            # and limits are written at runtime by the actuators below.
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            activate_contact_sensors=contact_sensors,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, root_height + RESET_CLEARANCE_M),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={name: NOMINAL_JOINT_POS[name] for name in joints},
            joint_vel={name: 0.0 for name in joints},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators=actuators,
    )


WHEEL_ONLY_ROBOT = _robot_cfg(
    WHEEL_ONLY_URDF, WHEEL_ONLY_ROOT_HEIGHT_M, WHEEL_JOINTS, {"wheels": _WHEEL_ACTUATOR}, contact_sensors=False
)
FULL_ROBOT = _robot_cfg(
    FULL_URDF, FULL_ROOT_HEIGHT_M, LEG_JOINTS + WHEEL_JOINTS, {"legs": _LEG_ACTUATOR, "wheels": _WHEEL_ACTUATOR}, contact_sensors=True
)


@configclass
class WheelOnlySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=NOMINAL_FRICTION,
            dynamic_friction=NOMINAL_FRICTION,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85)),
    )
    robot: ArticulationCfg = WHEEL_ONLY_ROBOT
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base_link_root",
        offset=ImuCfg.OffsetCfg(pos=IMU_POS_ROOT, rot=IMU_ROT_ROOT),
        debug_vis=False,
    )


@configclass
class FullSceneCfg(WheelOnlySceneCfg):
    robot: ArticulationCfg = FULL_ROBOT
    # Only used by the non-wheel contact termination.
    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3)
