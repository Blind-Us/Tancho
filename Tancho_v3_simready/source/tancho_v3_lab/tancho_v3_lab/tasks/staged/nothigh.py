"""No-thigh Tancho (4 DOF): base_link -> calf (revolute) -> wheel (continuous) per side.

Asset: ``Tancho_v3_no_thigh.urdf`` (``Tancho_v3/make_no_thigh_urdf.py``): thigh
links and joints removed, each calf mounted at the hip motor position, calf
range +/-90 deg.  The stand / walk stages reuse the 6-DOF stage 2 / 3 elements
with the leg set reduced to the two calf joints:

* Actor 19 dims: gravity 3, gyro 3, command 3, wheel speed 2, last action 4,
  calf position 2, calf speed 2.
* Action 4 dims: calf_L, calf_R position offsets (nominal + 0.25 rad * a),
  then the two wheel speeds.
* Nominal calf 0 rad: wheel axle straight below the hip motor.
* ``body_contact`` ends an episode on the base only (no thigh; the calf may
  graze the ground when the body tilts).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import isaaclab.envs.mdp as mdp
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .env_cfg import TanchoV3StandFlatEnvCfg, WalkCommandsCfg, _to_play
from .observations import WheelOnlyObservationsCfg
from .rewards import FullStandRewardsCfg, FullWalkRewardsCfg
from .scene import (
    _LEG_ACTUATOR,
    _WHEEL_ACTUATOR,
    LEG_ACTION_SCALE_RAD,
    URDF_DIR,
    WHEEL_JOINTS,
    FullSceneCfg,
    _robot_cfg,
)
from .actions import _WHEEL_VEL
from .terminations import WheelOnlyTerminationsCfg

NOTHIGH_URDF = URDF_DIR / "Tancho_v3_no_thigh.urdf"
NT_LEG_JOINTS = ["joint_calf_L", "joint_calf_R"]
NT_NOMINAL_CALF_RAD = 0.0
NT_LEGS = SceneEntityCfg("robot", joint_names=NT_LEG_JOINTS, preserve_order=True)


def _rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _nothigh_root_height(calf_rad: float) -> float:
    """Root Z that puts both tire cylinders tangent to z=0 at zero pitch (URDF FK)."""
    root = ET.parse(NOTHIGH_URDF).getroot()

    def origin(name):
        o = root.find(f"joint[@name='{name}']/origin")
        return np.array([float(v) for v in o.get("xyz").split()]), _rpy_matrix([float(v) for v in o.get("rpy").split()])

    heights = []
    for side in ("L", "R"):
        p0, r0 = origin("base_link_rotation_joint")
        p1, r1 = origin(f"joint_calf_{side}")
        p2, _ = origin(f"joint_wheel_{side}")
        # Calf axis is local z.
        rq = _rpy_matrix((0.0, 0.0, calf_rad))
        axle = p0 + r0 @ (p1 + r1 @ rq @ p2)
        radius = float(root.find(f"link[@name='wheel_{side}']/collision/geometry/cylinder").get("radius"))
        heights.append(radius - axle[2])
    if abs(heights[0] - heights[1]) > 1.0e-9:
        raise RuntimeError(f"Asymmetric wheel support heights: {heights}")
    return heights[0]


NT_ROOT_HEIGHT_M = _nothigh_root_height(NT_NOMINAL_CALF_RAD)

NT_ROBOT = _robot_cfg(
    NOTHIGH_URDF,
    NT_ROOT_HEIGHT_M,
    [],
    {"legs": _LEG_ACTUATOR.replace(joint_names_expr=NT_LEG_JOINTS), "wheels": _WHEEL_ACTUATOR},
    contact_sensors=True,
)
NT_ROBOT.init_state.joint_pos = {**{j: NT_NOMINAL_CALF_RAD for j in NT_LEG_JOINTS}, **{j: 0.0 for j in WHEEL_JOINTS}}
NT_ROBOT.init_state.joint_vel = {j: 0.0 for j in NT_LEG_JOINTS + WHEEL_JOINTS}


# -- scene ----------------------------------------------------------------------
@configclass
class NoThighSceneCfg(FullSceneCfg):
    robot = NT_ROBOT


# -- observations ---------------------------------------------------------------
@configclass
class NoThighObservationsCfg:
    @configclass
    class PolicyCfg(WheelOnlyObservationsCfg.PolicyCfg):
        """19 dims: wheel-only terms (last action now 4) + calf position 2 + calf speed 2."""

        leg_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": NT_LEGS}, noise=Unoise(n_min=-0.01, n_max=0.01))
        leg_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": NT_LEGS}, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.05)

    @configclass
    class CriticCfg(WheelOnlyObservationsCfg.CriticCfg):
        base_height = ObsTerm(func=mdp.base_pos_z)
        leg_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": NT_LEGS})
        leg_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": NT_LEGS}, scale=0.05)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# -- actions --------------------------------------------------------------------
@configclass
class NoThighActionsCfg:
    """4 dims: calf_L, calf_R position offsets, then the two wheel speeds."""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=NT_LEG_JOINTS,
        scale=LEG_ACTION_SCALE_RAD,
        use_default_offset=True,
        preserve_order=True,
    )
    wheel_vel = _WHEEL_VEL


# -- rewards --------------------------------------------------------------------
def mirror_calf_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Left/right calf position asymmetry (same 0.5 * d^2 form as the 6-DOF mirror term)."""
    robot: Articulation = env.scene["robot"]
    joint_ids, _ = robot.find_joints(NT_LEG_JOINTS, preserve_order=True)
    q = robot.data.joint_pos[:, joint_ids]
    return 0.5 * (q[:, 0] - q[:, 1]).square()


def _retarget_legs(rewards) -> None:
    rewards.mirror.func = mirror_calf_l2
    for term in (rewards.leg_deviation, rewards.leg_torque, rewards.leg_acc, rewards.leg_pos_limits):
        term.params["asset_cfg"] = NT_LEGS


@configclass
class NoThighStandRewardsCfg(FullStandRewardsCfg):
    def __post_init__(self):
        _retarget_legs(self)


@configclass
class NoThighWalkRewardsCfg(FullWalkRewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        _retarget_legs(self)


# -- terminations ---------------------------------------------------------------
@configclass
class NoThighTerminationsCfg(WheelOnlyTerminationsCfg):
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link_root"]), "threshold": 10.0},
    )


# -- stages ---------------------------------------------------------------------
@configclass
class TanchoV3NoThighStandEnvCfg(TanchoV3StandFlatEnvCfg):
    scene: NoThighSceneCfg = NoThighSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: NoThighObservationsCfg = NoThighObservationsCfg()
    actions: NoThighActionsCfg = NoThighActionsCfg()
    rewards: NoThighStandRewardsCfg = NoThighStandRewardsCfg()
    terminations: NoThighTerminationsCfg = NoThighTerminationsCfg()


def _nothigh_play(cfg) -> None:
    _to_play(cfg)
    # Root sits ~0.14 m up instead of ~0.26 m.
    cfg.viewer.lookat = (0.0, 0.0, -0.05)


@configclass
class TanchoV3NoThighStandPlayEnvCfg(TanchoV3NoThighStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _nothigh_play(self)


@configclass
class TanchoV3NoThighWalkEnvCfg(TanchoV3NoThighStandEnvCfg):
    rewards: NoThighWalkRewardsCfg = NoThighWalkRewardsCfg()
    commands: WalkCommandsCfg = WalkCommandsCfg()


@configclass
class TanchoV3NoThighWalkPlayEnvCfg(TanchoV3NoThighWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _nothigh_play(self)
        self.commands.base_velocity.debug_vis = True
