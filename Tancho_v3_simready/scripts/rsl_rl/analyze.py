"""Tancho V3 equilibrium / actuator diagnostic.

This script mirrors the project task-loading path used by scripts/rsl_rl/play.py,
but it does NOT load a PPO checkpoint.

Default test:
    thigh = -0.50 rad
    calf  = +0.87 rad
    wheel action = 0
    duration = 2.0 s
    num_envs = 1

It logs:
- leg joint position / velocity
- leg P / D / P+D / computed / applied torque
- PhysX generalized gravity-compensation torque for every leg joint
- optional static/dynamic gravity feed-forward command
- leg actuator target/error and P/D/FF/residual torque decomposition
- wheel position / velocity / acceleration / motor torque
- wheel incoming-joint reaction wrench
- full inverse-dynamics residual generalized torque (contact/passive/unmodeled load diagnostic)
- wheel link z / base z and vertical velocity
- base pitch / pitch rate
- base x / vx
- first termination time

CSV output is saved under logs/analyze/ by default.

Example:
    python -u scripts/rsl_rl/analyze.py --task=TanchoV3-Flat-v0
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Analyze Tancho V3 equilibrium and actuator response without PPO."
)

parser.add_argument(
    "--task",
    type=str,
    default="TanchoV3-Flat-v0",
    help="Registered Tancho task name.",
)

parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point used only so hydra_task_config can load the task.",
)

parser.add_argument(
    "--thigh",
    type=float,
    default=-0.50,
    help="Initial/default thigh joint angle [rad].",
)

parser.add_argument(
    "--calf",
    type=float,
    default=0.87,
    help="Initial/default calf joint angle [rad].",
)

parser.add_argument(
    "--duration",
    type=float,
    default=2.0,
    help="Diagnostic duration in simulation seconds.",
)

parser.add_argument(
    "--sample_dt",
    type=float,
    default=0.05,
    help="Terminal print interval in simulation seconds.",
)

parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Optional CSV path. Default: logs/analyze/analyze_<timestamp>.csv",
)

parser.add_argument(
    "--no_stop_on_termination",
    action="store_true",
    default=False,
    help=(
        "Do not stop when the environment reports termination. "
        "WARNING: ManagerBasedRLEnv may auto-reset terminated environments, "
        "so post-termination data can contain reset-state samples."
    ),
)

parser.add_argument(
    "--gravity_ff_mode",
    type=str,
    choices=("off", "static", "dynamic"),
    default="off",
    help=(
        "Gravity feed-forward mode. 'off' only logs PhysX gravity compensation; "
        "'static' applies the t=0 holding torque for the whole run; "
        "'dynamic' recomputes gravity compensation before each env step."
    ),
)

parser.add_argument(
    "--gravity_ff_scale",
    type=float,
    default=1.0,
    help="Scale applied to gravity feed-forward torque (default: 1.0).",
)

parser.add_argument(
    "--diagnostic_decimation",
    type=int,
    default=1,
    help=(
        "Diagnostic environment decimation. Default 1 logs every physics step. "
        "Use 0 to keep the task's original decimation."
    ),
)


# Isaac Lab launcher arguments
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Hydra should only see Hydra-specific arguments.
sys.argv = [sys.argv[0]] + hydra_args


# -----------------------------------------------------------------------------
# Launch Isaac Sim
# -----------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports after Isaac Sim startup
# -----------------------------------------------------------------------------

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

# IMPORTANT: this registers TanchoV3-Flat-v0 in Gym.
import tancho_v3_lab.tasks  # noqa: F401


# -----------------------------------------------------------------------------
# Constants / helpers
# -----------------------------------------------------------------------------

LEG_JOINT_NAMES = [
    "joint_thigh_L",
    "joint_calf_L",
    "joint_thigh_R",
    "joint_calf_R",
]

WHEEL_JOINT_NAMES = [
    "joint_wheel_L",
    "joint_wheel_R",
]

WHEEL_BODY_NAMES = [
    "wheel_L",
    "wheel_R",
]


def rad2deg(value: float) -> float:
    return value * 180.0 / math.pi


def quat_pitch_wxyz(quat: torch.Tensor) -> float:
    """Return pitch [rad] from a quaternion in Isaac Lab's (w, x, y, z) order."""
    w, x, y, z = [float(v) for v in quat]
    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    return math.asin(sin_pitch)


def scalar_cfg_value(value):
    """Best-effort conversion of actuator cfg values to a readable scalar."""
    if isinstance(value, (int, float)):
        return float(value)
    return value


def make_default_output_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("logs") / "analyze" / f"analyze_{stamp}.csv"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@hydra_task_config(
    args_cli.task,
    args_cli.agent,
)
def main(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    # No PPO/checkpoint is used in this script.
    del agent_cfg

    # -------------------------------------------------------------------------
    # Force a single diagnostic environment and requested posture.
    # -------------------------------------------------------------------------

    env_cfg.scene.num_envs = 1

    original_decimation = int(env_cfg.decimation)
    if args_cli.diagnostic_decimation < 0:
        raise ValueError("--diagnostic_decimation must be >= 0")
    if args_cli.diagnostic_decimation > 0:
        env_cfg.decimation = int(args_cli.diagnostic_decimation)

    joint_pos_cfg = env_cfg.scene.robot.init_state.joint_pos

    joint_pos_cfg["joint_thigh_L"] = args_cli.thigh
    joint_pos_cfg["joint_thigh_R"] = args_cli.thigh
    joint_pos_cfg["joint_calf_L"] = args_cli.calf
    joint_pos_cfg["joint_calf_R"] = args_cli.calf
    joint_pos_cfg["joint_wheel_L"] = 0.0
    joint_pos_cfg["joint_wheel_R"] = 0.0

    # -------------------------------------------------------------------------
    # Print the CURRENT actuator configuration.
    # These values are simulation settings, not automatically calibrated values.
    # -------------------------------------------------------------------------

    leg_cfg = env_cfg.scene.robot.actuators.get("legs", None)
    wheel_cfg = env_cfg.scene.robot.actuators.get("wheels", None)

    if leg_cfg is None:
        raise RuntimeError("Leg actuator config 'legs' was not found.")
    if not isinstance(leg_cfg.stiffness, (int, float)) or not isinstance(leg_cfg.damping, (int, float)):
        raise TypeError(
            "analyze.py PD split currently expects scalar leg stiffness/damping. "
            f"Got stiffness={leg_cfg.stiffness!r}, damping={leg_cfg.damping!r}."
        )

    leg_kp = float(leg_cfg.stiffness)
    leg_kd = float(leg_cfg.damping)

    print(f"[INFO] Diagnostic PD: Kp={leg_kp}, Kd={leg_kd}")

    print("\n" + "=" * 92)
    print("TANCHO EQUILIBRIUM / ACTUATOR DIAGNOSTIC")
    print("=" * 92)
    print(f"Task                 : {args_cli.task}")
    print(f"Thigh target         : {args_cli.thigh:+.4f} rad ({rad2deg(args_cli.thigh):+.2f} deg)")
    print(f"Calf target          : {args_cli.calf:+.4f} rad ({rad2deg(args_cli.calf):+.2f} deg)")
    print(f"Duration             : {args_cli.duration:.3f} s")
    print("Wheel action         : 0")
    print(f"Task decimation      : {original_decimation}")
    print(f"Diagnostic decimation: {int(env_cfg.decimation)}")

    if leg_cfg is not None:
        print("\nCurrent leg actuator cfg:")
        print(f"  stiffness          : {scalar_cfg_value(leg_cfg.stiffness)}")
        print(f"  damping            : {scalar_cfg_value(leg_cfg.damping)}")
        print(f"  effort_limit_sim   : {scalar_cfg_value(leg_cfg.effort_limit_sim)}")
        print(f"  velocity_limit_sim : {scalar_cfg_value(leg_cfg.velocity_limit_sim)}")

    if wheel_cfg is not None:
        print("\nCurrent wheel actuator cfg:")
        print(f"  stiffness          : {scalar_cfg_value(wheel_cfg.stiffness)}")
        print(f"  damping            : {scalar_cfg_value(wheel_cfg.damping)}")
        print(f"  effort_limit_sim   : {scalar_cfg_value(wheel_cfg.effort_limit_sim)}")
        print(f"  velocity_limit_sim : {scalar_cfg_value(wheel_cfg.velocity_limit_sim)}")

    print("=" * 92)

    # -------------------------------------------------------------------------
    # Create environment exactly through the registered Tancho task.
    # -------------------------------------------------------------------------

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    try:
        # ---------------------------------------------------------------------
        # Resolve joints by NAME. Never assume URDF joint ordering.
        # ---------------------------------------------------------------------

        leg_joint_ids, leg_joint_names = robot.find_joints(
            LEG_JOINT_NAMES,
            preserve_order=True,
        )

        wheel_joint_ids, wheel_joint_names = robot.find_joints(
            WHEEL_JOINT_NAMES,
            preserve_order=True,
        )

        wheel_body_ids, wheel_body_names = robot.find_bodies(
            WHEEL_BODY_NAMES,
            preserve_order=True,
        )

        print("\nResolved joints:")
        for idx, name in zip(leg_joint_ids, leg_joint_names):
            print(f"  leg   [{idx:2d}] {name}")
        for idx, name in zip(wheel_joint_ids, wheel_joint_names):
            print(f"  wheel [{idx:2d}] {name}")
        for idx, name in zip(wheel_body_ids, wheel_body_names):
            print(f"  body  [{idx:2d}] {name}")

        # ---------------------------------------------------------------------
        # PhysX inverse-dynamics helpers.
        #
        # Equation of motion (floating base):
        #   tau_required = M(q) qdd + C(q,qd) + G(q)
        # PhysX inverse-dynamics terms do NOT include contact, joint friction,
        # or damping. Therefore, for each actuated joint:
        #   tau_external_residual = tau_required - tau_applied
        # is a useful diagnostic for contact/passive/unmodeled generalized load.
        # It is NOT a motor command and is NOT limited by effort_limit_sim.
        # ---------------------------------------------------------------------

        num_robot_joints = len(robot.joint_names)
        diagnostic_api_state = {
            "inverse_ok": True,
            "inverse_warned": False,
            "wrench_ok": True,
            "wrench_warned": False,
        }

        def _as_torch(value) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                return value.to(base_env.device)
            try:
                return torch.as_tensor(value, device=base_env.device)
            except Exception:
                # Warp arrays and some tensor frontends expose .to_torch().
                if hasattr(value, "to_torch"):
                    return value.to_torch().to(base_env.device)
                raise

        def _generalized_joint_offset(width: int) -> int:
            if width == num_robot_joints:
                return 0
            if width >= num_robot_joints + 6:
                return width - num_robot_joints
            raise RuntimeError(
                f"Unexpected generalized-vector width {width} for "
                f"{num_robot_joints} articulation joints."
            )

        def get_inverse_dynamics_joint_required() -> torch.Tensor:
            """Return M*qdd + C + G for all articulation joints [Nm].

            For the floating base, PhysX expects root linear/angular acceleration
            in world coordinates followed by joint accelerations.
            """
            try:
                view = robot.root_physx_view
                mass_raw = _as_torch(view.get_generalized_mass_matrices())
                coriolis = _as_torch(view.get_coriolis_and_centrifugal_compensation_forces())
                gravity = _as_torch(view.get_gravity_compensation_forces())

                if coriolis.ndim == 1:
                    coriolis = coriolis.unsqueeze(0)
                if gravity.ndim == 1:
                    gravity = gravity.unsqueeze(0)

                width = int(gravity.shape[-1])
                joint_offset = _generalized_joint_offset(width)

                # Normalize mass matrix to [env, width, width].
                if mass_raw.ndim == 3:
                    mass_matrix = mass_raw
                elif mass_raw.ndim == 2 and mass_raw.shape[0] == 1:
                    flat_width = int(round(math.sqrt(int(mass_raw.shape[1]))))
                    if flat_width * flat_width != int(mass_raw.shape[1]):
                        raise RuntimeError(
                            f"Cannot reshape generalized mass matrix {tuple(mass_raw.shape)}."
                        )
                    mass_matrix = mass_raw.reshape(1, flat_width, flat_width)
                elif mass_raw.ndim == 2 and mass_raw.shape[0] == mass_raw.shape[1]:
                    mass_matrix = mass_raw.unsqueeze(0)
                else:
                    raise RuntimeError(
                        f"Unexpected generalized mass matrix shape: {tuple(mass_raw.shape)}"
                    )

                if int(mass_matrix.shape[-1]) != width:
                    raise RuntimeError(
                        "Mass-matrix/generalized-vector width mismatch: "
                        f"M={tuple(mass_matrix.shape)}, G={tuple(gravity.shape)}"
                    )

                joint_acc_all = robot.data.joint_acc[0].to(base_env.device)

                if joint_offset > 0:
                    # Link 0 is the floating root. body_acc_w is [lin, ang] in world.
                    root_acc = robot.data.body_acc_w[0, 0].to(base_env.device)
                    if int(root_acc.numel()) != joint_offset:
                        raise RuntimeError(
                            f"Expected {joint_offset} root accelerations, got {int(root_acc.numel())}."
                        )
                    generalized_acc = torch.cat((root_acc, joint_acc_all), dim=0)
                else:
                    generalized_acc = joint_acc_all

                required_all = (
                    mass_matrix[0] @ generalized_acc
                    + coriolis[0]
                    + gravity[0]
                )
                return required_all[joint_offset: joint_offset + num_robot_joints].clone()

            except Exception as exc:
                diagnostic_api_state["inverse_ok"] = False
                if not diagnostic_api_state["inverse_warned"]:
                    print(
                        "[WARN] Full inverse-dynamics residual unavailable; "
                        f"fields will be NaN. Reason: {exc}"
                    )
                    diagnostic_api_state["inverse_warned"] = True
                return torch.full(
                    (num_robot_joints,),
                    float("nan"),
                    device=base_env.device,
                )

        def get_wheel_incoming_joint_wrench() -> torch.Tensor:
            """Return wheel incoming-joint wrench [Fx,Fy,Fz,Tx,Ty,Tz].

            PhysX reports the wrench applied from parent to child at the incoming
            joint. This is useful context, but it is not itself the external
            contact generalized torque on the free wheel DOF.
            """
            try:
                wrench = _as_torch(robot.root_physx_view.get_link_incoming_joint_force())
                if wrench.ndim == 2:
                    # Some frontends may flatten [env*links, 6]. One environment only.
                    if wrench.shape[-1] == 6 and wrench.shape[0] == robot.num_bodies:
                        wrench = wrench.unsqueeze(0)
                    else:
                        wrench = wrench.reshape(1, robot.num_bodies, 6)
                return wrench[0, wheel_body_ids].clone()
            except Exception as exc:
                diagnostic_api_state["wrench_ok"] = False
                if not diagnostic_api_state["wrench_warned"]:
                    print(
                        "[WARN] Incoming-joint wrench unavailable; fields will be NaN. "
                        f"Reason: {exc}"
                    )
                    diagnostic_api_state["wrench_warned"] = True
                return torch.full((2, 6), float("nan"), device=base_env.device)

        # ---------------------------------------------------------------------
        # PhysX gravity-compensation helper.
        #
        # PhysX returns the generalized forces required to COUNTERACT gravity
        # at the current articulation pose. Floating-base layouts may prepend
        # 6 root generalized coordinates. This helper normalizes both layouts
        # to the four Tancho leg joints.
        # ---------------------------------------------------------------------

        def get_leg_gravity_compensation() -> torch.Tensor:
            gravity_all = robot.root_physx_view.get_gravity_compensation_forces()

            if not isinstance(gravity_all, torch.Tensor):
                gravity_all = torch.as_tensor(gravity_all, device=base_env.device)
            else:
                gravity_all = gravity_all.to(base_env.device)

            if gravity_all.ndim == 1:
                gravity_all = gravity_all.unsqueeze(0)

            width = int(gravity_all.shape[-1])
            if width == num_robot_joints:
                joint_offset = 0
            elif width >= num_robot_joints + 6:
                joint_offset = width - num_robot_joints
            else:
                raise RuntimeError(
                    "Unexpected gravity-compensation tensor shape: "
                    f"{tuple(gravity_all.shape)} for {num_robot_joints} robot joints."
                )

            gravity_joint_ids = [joint_offset + int(j) for j in leg_joint_ids]
            return gravity_all[0, gravity_joint_ids].clone()

        # ---------------------------------------------------------------------
        # Reset once.
        # ---------------------------------------------------------------------

        env.reset()

        step_dt = float(base_env.step_dt)

        # Wheel physical properties as imported by PhysX. The wheel-joint axis is
        # local Z in the URDF and each wheel inertial frame has rpy=0, so Izz is
        # the wheel body's own axial inertia. This is used only for the simple
        # Izz*qdd intuition metric; the full inverse-dynamics residual below is
        # the more meaningful articulated-system diagnostic.
        # default_mass/default_inertia may live on CPU even when the articulation
        # state (joint_acc, joint_vel, ...) is on CUDA. Normalize these static
        # properties to the simulation device once so later diagnostics can mix
        # them safely with qdd_wheel.
        state_dtype = robot.data.joint_acc.dtype
        wheel_body_mass = robot.data.default_mass[0, wheel_body_ids].to(
            device=base_env.device, dtype=state_dtype
        ).clone()
        wheel_inertia_flat = robot.data.default_inertia[0, wheel_body_ids].to(
            device=base_env.device, dtype=state_dtype
        ).clone()
        wheel_inertia_mats = wheel_inertia_flat.reshape(len(wheel_body_ids), 3, 3)
        wheel_axis_inertia = wheel_inertia_mats[:, 2, 2].clone()

        print("\nWheel PhysX properties:")
        for name, mass, inertia_zz in zip(
            wheel_body_names, wheel_body_mass, wheel_axis_inertia
        ):
            print(
                f"  {name:<18}: mass={float(mass):.6f} kg, "
                f"I_axis={float(inertia_zz):.9e} kg*m^2"
            )

        # ---------------------------------------------------------
        # Verify actual default joint position
        # ---------------------------------------------------------

        default_leg_pos = robot.data.default_joint_pos[0, leg_joint_ids]

        print("\nDefault leg joint positions:")
        for name, value in zip(leg_joint_names, default_leg_pos):
            print(f"  {name:<18}: {float(value):+.4f} rad")

        # ---------------------------------------------------------
        # Zero action
        # ---------------------------------------------------------

        action_dim = base_env.action_manager.total_action_dim

        actions = torch.zeros(
            (1, action_dim),
            device=base_env.device,
            dtype=torch.float32,
        )

        # ---------------------------------------------------------
        # PRIME ACTION TARGETS WITHOUT ADVANCING PHYSICS
        #
        # Important:
        # After reset, actuator target telemetry may still correspond
        # to the reset/initial drive state. We want JointPositionActionCfg
        # to write its default offset (-0.50 / +0.87) into the actuator,
        # but we must NOT advance the simulation before t = 0 is logged.
        #
        # This mirrors the action-processing portion of ManagerBasedEnv.step():
        #   process_action -> apply_action -> write_data_to_sim
        # and intentionally omits sim.step()/scene.update().
        # ---------------------------------------------------------

        with torch.inference_mode():
            base_env.action_manager.process_action(actions)
            base_env.action_manager.apply_action()

            # Exact PhysX holding torque at the nominal pose, before physics.
            nominal_gravity_leg = get_leg_gravity_compensation()

            if args_cli.gravity_ff_mode in ("static", "dynamic"):
                last_gravity_ff_cmd_leg = (
                    args_cli.gravity_ff_scale * nominal_gravity_leg
                ).clone()
            else:
                last_gravity_ff_cmd_leg = torch.zeros_like(nominal_gravity_leg)

            # Feed-forward effort is additive to the implicit PD drive.
            robot.set_joint_effort_target(
                last_gravity_ff_cmd_leg.unsqueeze(0),
                joint_ids=leg_joint_ids,
            )
            base_env.scene.write_data_to_sim()

        print("[INFO] Action targets primed without advancing physics.")
        print(f"[INFO] Gravity FF mode      : {args_cli.gravity_ff_mode}")
        print(f"[INFO] Gravity FF scale     : {args_cli.gravity_ff_scale:.3f}")
        print("[INFO] Nominal PhysX gravity holding torque:")
        for name, tau_g in zip(leg_joint_names, nominal_gravity_leg):
            print(f"  {name:<18}: {float(tau_g):+9.4f} Nm")

        if args_cli.gravity_ff_mode != "off":
            print(
                "[WARN] Gravity FF is a diagnostic additive effort on top of the "
                "ImplicitActuator PD drive. Use G@t0 as the static holding-torque "
                "measurement; this is not a calibrated motor-current controller."
            )
        num_steps = max(1, int(math.ceil(args_cli.duration / step_dt)))
        print_every = max(1, int(round(args_cli.sample_dt / step_dt)))

        print(f"\nEnvironment step_dt  : {step_dt:.6f} s")
        print(f"Planned env steps    : {num_steps}")
        print(f"Print every          : {print_every} step(s)")

        # ---------------------------------------------------------------------
        # Zero actions:
        # - wheel effort action = 0
        # - leg policy action = 0
        #
        # NOTE:
        # If the environment's leg position action uses default offsets,
        # the leg PD actuator can still generate torque to hold its target.
        # ---------------------------------------------------------------------


        # Targets for error calculations.
        target_by_name = {
            "joint_thigh_L": args_cli.thigh,
            "joint_calf_L": args_cli.calf,
            "joint_thigh_R": args_cli.thigh,
            "joint_calf_R": args_cli.calf,
        }

        # ---------------------------------------------------------------------
        # Data containers.
        # ---------------------------------------------------------------------

        rows = []
        termination_time = None

        # Determine a numeric leg effort limit if possible.
        leg_effort_limit = None
        if leg_cfg is not None and isinstance(leg_cfg.effort_limit_sim, (int, float)):
            leg_effort_limit = float(leg_cfg.effort_limit_sim)

        wheel_effort_limit = None
        if wheel_cfg is not None and isinstance(wheel_cfg.effort_limit_sim, (int, float)):
            wheel_effort_limit = float(wheel_cfg.effort_limit_sim)

        # ---------------------------------------------------------------------
        # State reader.
        # ---------------------------------------------------------------------

        def read_state(sim_time: float):
            q_leg = robot.data.joint_pos[0, leg_joint_ids]
            qd_leg = robot.data.joint_vel[0, leg_joint_ids]
            tau_applied_leg = robot.data.applied_torque[0, leg_joint_ids]
            tau_computed_leg = robot.data.computed_torque[0, leg_joint_ids]

            # Actual actuator targets currently held by Isaac Lab
            q_target_leg = robot.data.joint_pos_target[0, leg_joint_ids]
            qd_target_leg = robot.data.joint_vel_target[0, leg_joint_ids]

            # Position / velocity error
            q_err_leg = q_target_leg - q_leg
            qd_err_leg = qd_target_leg - qd_leg

            # Split PD torque
            tau_p_leg = leg_kp * q_err_leg
            tau_d_leg = leg_kd * qd_err_leg
            tau_pd_leg = tau_p_leg + tau_d_leg

            # Exact generalized gravity-compensation torque at the CURRENT pose.
            gravity_comp_leg = get_leg_gravity_compensation()
            gravity_ff_cmd_leg = last_gravity_ff_cmd_leg
            tau_pd_ff_leg = tau_pd_leg + gravity_ff_cmd_leg

            # Telemetry residuals.
            tau_residual_leg = tau_computed_leg - tau_pd_leg
            tau_residual_with_ff_leg = tau_computed_leg - tau_pd_ff_leg

            q_wheel = robot.data.joint_pos[0, wheel_joint_ids]
            qd_wheel = robot.data.joint_vel[0, wheel_joint_ids]
            qdd_wheel = robot.data.joint_acc[0, wheel_joint_ids]
            tau_applied_wheel = robot.data.applied_torque[0, wheel_joint_ids]
            tau_computed_wheel = robot.data.computed_torque[0, wheel_joint_ids]

            # Full articulated inverse dynamics. The residual is the generalized
            # torque not explained by M*qdd+C+G versus actuator torque. PhysX's
            # inverse dynamics excludes contact, damping and joint friction, so a
            # sharp residual spike at impact is exactly what we want to inspect.
            tau_id_required_all = get_inverse_dynamics_joint_required()
            tau_id_required_wheel = tau_id_required_all[wheel_joint_ids]
            tau_external_residual_wheel = (
                tau_id_required_wheel - tau_applied_wheel
            )

            # Simple isolated-wheel intuition only. This ignores articulated
            # coupling and therefore must NOT be interpreted as exact contact torque.
            tau_izz_qdd_wheel = wheel_axis_inertia.to(
                device=qdd_wheel.device, dtype=qdd_wheel.dtype
            ) * qdd_wheel
            tau_izz_external_est_wheel = tau_izz_qdd_wheel - tau_applied_wheel

            incoming_wrench_wheel = get_wheel_incoming_joint_wrench()

            # Both wheel joints have URDF origin rpy=(pi,0,0), so local +Z joint
            # axis maps to -Z of the parent calf frame. The incoming-wrench
            # projection below is therefore -Tz in that parent frame. Keep this
            # separate from the inverse-dynamics external residual.
            incoming_axis_tau_wheel = -incoming_wrench_wheel[:, 5]

            wheel_body_pos_w = robot.data.body_pos_w[0, wheel_body_ids]

            quat_w = robot.data.root_quat_w[0]
            pitch_rad = quat_pitch_wxyz(quat_w)
            pitch_deg = rad2deg(pitch_rad)

            # Prefer body-frame angular velocity for pitch-rate interpretation.
            if hasattr(robot.data, "root_ang_vel_b"):
                pitch_rate = float(robot.data.root_ang_vel_b[0, 1].item())
            else:
                pitch_rate = float(robot.data.root_ang_vel_w[0, 1].item())

            base_x = float(robot.data.root_pos_w[0, 0].item())
            base_z = float(robot.data.root_pos_w[0, 2].item())
            base_vx = float(robot.data.root_lin_vel_w[0, 0].item())
            base_vz = float(robot.data.root_lin_vel_w[0, 2].item())

            q_leg_v = [float(x) for x in q_leg.detach().cpu()]
            qd_leg_v = [float(x) for x in qd_leg.detach().cpu()]
            tau_applied_leg_v = [float(x) for x in tau_applied_leg.detach().cpu()]
            tau_computed_leg_v = [float(x) for x in tau_computed_leg.detach().cpu()]

            q_target_leg_v = [float(x) for x in q_target_leg.detach().cpu()]
            qd_target_leg_v = [float(x) for x in qd_target_leg.detach().cpu()]
            q_err_leg_v = [float(x) for x in q_err_leg.detach().cpu()]
            qd_err_leg_v = [float(x) for x in qd_err_leg.detach().cpu()]
            tau_p_leg_v = [float(x) for x in tau_p_leg.detach().cpu()]
            tau_d_leg_v = [float(x) for x in tau_d_leg.detach().cpu()]
            tau_pd_leg_v = [float(x) for x in tau_pd_leg.detach().cpu()]
            gravity_comp_leg_v = [float(x) for x in gravity_comp_leg.detach().cpu()]
            gravity_ff_cmd_leg_v = [float(x) for x in gravity_ff_cmd_leg.detach().cpu()]
            tau_pd_ff_leg_v = [float(x) for x in tau_pd_ff_leg.detach().cpu()]
            tau_residual_leg_v = [float(x) for x in tau_residual_leg.detach().cpu()]
            tau_residual_with_ff_leg_v = [
                float(x) for x in tau_residual_with_ff_leg.detach().cpu()
            ]

            q_wheel_v = [float(x) for x in q_wheel.detach().cpu()]
            qd_wheel_v = [float(x) for x in qd_wheel.detach().cpu()]
            qdd_wheel_v = [float(x) for x in qdd_wheel.detach().cpu()]
            tau_applied_wheel_v = [float(x) for x in tau_applied_wheel.detach().cpu()]
            tau_computed_wheel_v = [float(x) for x in tau_computed_wheel.detach().cpu()]
            tau_id_required_wheel_v = [float(x) for x in tau_id_required_wheel.detach().cpu()]
            tau_external_residual_wheel_v = [
                float(x) for x in tau_external_residual_wheel.detach().cpu()
            ]
            tau_izz_qdd_wheel_v = [float(x) for x in tau_izz_qdd_wheel.detach().cpu()]
            tau_izz_external_est_wheel_v = [
                float(x) for x in tau_izz_external_est_wheel.detach().cpu()
            ]
            incoming_wrench_wheel_v = incoming_wrench_wheel.detach().cpu().tolist()
            incoming_axis_tau_wheel_v = [
                float(x) for x in incoming_axis_tau_wheel.detach().cpu()
            ]
            wheel_body_pos_w_v = wheel_body_pos_w.detach().cpu().tolist()

            if wheel_effort_limit is not None and wheel_effort_limit > 0:
                external_over_limit = [
                    abs(x) / wheel_effort_limit
                    for x in tau_external_residual_wheel_v
                ]
            else:
                external_over_limit = [float("nan"), float("nan")]

            row = {
                "time_s": sim_time,

                "thigh_L_pos_rad": q_leg_v[0],
                "calf_L_pos_rad": q_leg_v[1],
                "thigh_R_pos_rad": q_leg_v[2],
                "calf_R_pos_rad": q_leg_v[3],

                "thigh_L_vel_rad_s": qd_leg_v[0],
                "calf_L_vel_rad_s": qd_leg_v[1],
                "thigh_R_vel_rad_s": qd_leg_v[2],
                "calf_R_vel_rad_s": qd_leg_v[3],

                "thigh_L_applied_tau_Nm": tau_applied_leg_v[0],
                "calf_L_applied_tau_Nm": tau_applied_leg_v[1],
                "thigh_R_applied_tau_Nm": tau_applied_leg_v[2],
                "calf_R_applied_tau_Nm": tau_applied_leg_v[3],

                "thigh_L_computed_tau_Nm": tau_computed_leg_v[0],
                "calf_L_computed_tau_Nm": tau_computed_leg_v[1],
                "thigh_R_computed_tau_Nm": tau_computed_leg_v[2],
                "calf_R_computed_tau_Nm": tau_computed_leg_v[3],

                # Actual actuator targets / errors used for PD decomposition.
                "thigh_L_target_rad": q_target_leg_v[0],
                "calf_L_target_rad": q_target_leg_v[1],
                "thigh_R_target_rad": q_target_leg_v[2],
                "calf_R_target_rad": q_target_leg_v[3],

                "thigh_L_vel_target_rad_s": qd_target_leg_v[0],
                "calf_L_vel_target_rad_s": qd_target_leg_v[1],
                "thigh_R_vel_target_rad_s": qd_target_leg_v[2],
                "calf_R_vel_target_rad_s": qd_target_leg_v[3],

                "thigh_L_error_rad": q_err_leg_v[0],
                "calf_L_error_rad": q_err_leg_v[1],
                "thigh_R_error_rad": q_err_leg_v[2],
                "calf_R_error_rad": q_err_leg_v[3],

                "thigh_L_vel_error_rad_s": qd_err_leg_v[0],
                "calf_L_vel_error_rad_s": qd_err_leg_v[1],
                "thigh_R_vel_error_rad_s": qd_err_leg_v[2],
                "calf_R_vel_error_rad_s": qd_err_leg_v[3],

                "thigh_L_P_tau_Nm": tau_p_leg_v[0],
                "calf_L_P_tau_Nm": tau_p_leg_v[1],
                "thigh_R_P_tau_Nm": tau_p_leg_v[2],
                "calf_R_P_tau_Nm": tau_p_leg_v[3],

                "thigh_L_D_tau_Nm": tau_d_leg_v[0],
                "calf_L_D_tau_Nm": tau_d_leg_v[1],
                "thigh_R_D_tau_Nm": tau_d_leg_v[2],
                "calf_R_D_tau_Nm": tau_d_leg_v[3],

                "thigh_L_PD_tau_Nm": tau_pd_leg_v[0],
                "calf_L_PD_tau_Nm": tau_pd_leg_v[1],
                "thigh_R_PD_tau_Nm": tau_pd_leg_v[2],
                "calf_R_PD_tau_Nm": tau_pd_leg_v[3],

                # Exact PhysX generalized effort required to counteract gravity.
                "thigh_L_gravity_comp_tau_Nm": gravity_comp_leg_v[0],
                "calf_L_gravity_comp_tau_Nm": gravity_comp_leg_v[1],
                "thigh_R_gravity_comp_tau_Nm": gravity_comp_leg_v[2],
                "calf_R_gravity_comp_tau_Nm": gravity_comp_leg_v[3],

                # Feed-forward effort requested by this diagnostic.
                "thigh_L_gravity_ff_cmd_Nm": gravity_ff_cmd_leg_v[0],
                "calf_L_gravity_ff_cmd_Nm": gravity_ff_cmd_leg_v[1],
                "thigh_R_gravity_ff_cmd_Nm": gravity_ff_cmd_leg_v[2],
                "calf_R_gravity_ff_cmd_Nm": gravity_ff_cmd_leg_v[3],

                "thigh_L_PD_plus_FF_tau_Nm": tau_pd_ff_leg_v[0],
                "calf_L_PD_plus_FF_tau_Nm": tau_pd_ff_leg_v[1],
                "thigh_R_PD_plus_FF_tau_Nm": tau_pd_ff_leg_v[2],
                "calf_R_PD_plus_FF_tau_Nm": tau_pd_ff_leg_v[3],

                # computed_tau - (P + D): old decomposition.
                "thigh_L_tau_residual_Nm": tau_residual_leg_v[0],
                "calf_L_tau_residual_Nm": tau_residual_leg_v[1],
                "thigh_R_tau_residual_Nm": tau_residual_leg_v[2],
                "calf_R_tau_residual_Nm": tau_residual_leg_v[3],

                # computed_tau - (P + D + FF).
                "thigh_L_tau_residual_with_ff_Nm": tau_residual_with_ff_leg_v[0],
                "calf_L_tau_residual_with_ff_Nm": tau_residual_with_ff_leg_v[1],
                "thigh_R_tau_residual_with_ff_Nm": tau_residual_with_ff_leg_v[2],
                "calf_R_tau_residual_with_ff_Nm": tau_residual_with_ff_leg_v[3],

                "wheel_L_pos_rad": q_wheel_v[0],
                "wheel_R_pos_rad": q_wheel_v[1],
                "wheel_L_vel_rad_s": qd_wheel_v[0],
                "wheel_R_vel_rad_s": qd_wheel_v[1],
                "wheel_L_acc_rad_s2": qdd_wheel_v[0],
                "wheel_R_acc_rad_s2": qdd_wheel_v[1],
                "wheel_L_applied_tau_Nm": tau_applied_wheel_v[0],
                "wheel_R_applied_tau_Nm": tau_applied_wheel_v[1],
                "wheel_L_computed_tau_Nm": tau_computed_wheel_v[0],
                "wheel_R_computed_tau_Nm": tau_computed_wheel_v[1],

                # Full articulated inverse-dynamics diagnostic:
                # tau_ID = M*qdd + C + G
                # tau_external_residual = tau_ID - tau_applied
                # Residual contains contact plus passive/unmodeled effects.
                "wheel_L_ID_required_tau_Nm": tau_id_required_wheel_v[0],
                "wheel_R_ID_required_tau_Nm": tau_id_required_wheel_v[1],
                "wheel_L_external_residual_tau_Nm": tau_external_residual_wheel_v[0],
                "wheel_R_external_residual_tau_Nm": tau_external_residual_wheel_v[1],
                "wheel_L_external_over_0p45x": external_over_limit[0],
                "wheel_R_external_over_0p45x": external_over_limit[1],

                # Simple Izz*qdd check (only an intuition metric; ignores coupling).
                "wheel_L_Izz_qdd_tau_Nm": tau_izz_qdd_wheel_v[0],
                "wheel_R_Izz_qdd_tau_Nm": tau_izz_qdd_wheel_v[1],
                "wheel_L_Izz_external_est_tau_Nm": tau_izz_external_est_wheel_v[0],
                "wheel_R_Izz_external_est_tau_Nm": tau_izz_external_est_wheel_v[1],

                # Incoming-joint reaction wrench in the parent-body frame.
                "wheel_L_joint_Fx_N": incoming_wrench_wheel_v[0][0],
                "wheel_L_joint_Fy_N": incoming_wrench_wheel_v[0][1],
                "wheel_L_joint_Fz_N": incoming_wrench_wheel_v[0][2],
                "wheel_L_joint_Tx_Nm": incoming_wrench_wheel_v[0][3],
                "wheel_L_joint_Ty_Nm": incoming_wrench_wheel_v[0][4],
                "wheel_L_joint_Tz_Nm": incoming_wrench_wheel_v[0][5],
                "wheel_L_joint_axis_tau_Nm": incoming_axis_tau_wheel_v[0],
                "wheel_R_joint_Fx_N": incoming_wrench_wheel_v[1][0],
                "wheel_R_joint_Fy_N": incoming_wrench_wheel_v[1][1],
                "wheel_R_joint_Fz_N": incoming_wrench_wheel_v[1][2],
                "wheel_R_joint_Tx_Nm": incoming_wrench_wheel_v[1][3],
                "wheel_R_joint_Ty_Nm": incoming_wrench_wheel_v[1][4],
                "wheel_R_joint_Tz_Nm": incoming_wrench_wheel_v[1][5],
                "wheel_R_joint_axis_tau_Nm": incoming_axis_tau_wheel_v[1],

                "wheel_L_body_x_m": wheel_body_pos_w_v[0][0],
                "wheel_L_body_y_m": wheel_body_pos_w_v[0][1],
                "wheel_L_body_z_m": wheel_body_pos_w_v[0][2],
                "wheel_R_body_x_m": wheel_body_pos_w_v[1][0],
                "wheel_R_body_y_m": wheel_body_pos_w_v[1][1],
                "wheel_R_body_z_m": wheel_body_pos_w_v[1][2],

                "pitch_rad": pitch_rad,
                "pitch_deg": pitch_deg,
                "pitch_rate_rad_s": pitch_rate,

                "base_x_m": base_x,
                "base_z_m": base_z,
                "base_vx_m_s": base_vx,
                "base_vz_m_s": base_vz,
            }

            return row

        # ---------------------------------------------------------------------
        # Print header.
        # ---------------------------------------------------------------------

        print("\n")
        print(
            f"{'t[s]':>6} | "
            f"{'ThighL':>8} {'CalfL':>8} {'ThighR':>8} {'CalfR':>8} | "
            f"{'tauTL':>7} {'tauCL':>7} {'tauTR':>7} {'tauCR':>7} | "
            f"{'Pitch':>8} {'dPitch':>8}"
        )
        print("-" * 116)

        # Log t = 0 before the first simulation step.
        initial_row = read_state(0.0)
        rows.append(initial_row)

        # Sanity checks: t = 0 should truly be the reset pose at rest,
        # with the actuator targets already primed to the requested posture.
        initial_joint_vels = [
            initial_row["thigh_L_vel_rad_s"],
            initial_row["calf_L_vel_rad_s"],
            initial_row["thigh_R_vel_rad_s"],
            initial_row["calf_R_vel_rad_s"],
        ]
        initial_target_errors = [
            initial_row["thigh_L_target_rad"] - args_cli.thigh,
            initial_row["calf_L_target_rad"] - args_cli.calf,
            initial_row["thigh_R_target_rad"] - args_cli.thigh,
            initial_row["calf_R_target_rad"] - args_cli.calf,
        ]
        initial_pose_errors = [
            initial_row["thigh_L_pos_rad"] - args_cli.thigh,
            initial_row["calf_L_pos_rad"] - args_cli.calf,
            initial_row["thigh_R_pos_rad"] - args_cli.thigh,
            initial_row["calf_R_pos_rad"] - args_cli.calf,
        ]

        max_initial_vel = max(abs(x) for x in initial_joint_vels)
        max_initial_target_err = max(abs(x) for x in initial_target_errors)
        max_initial_pose_err = max(abs(x) for x in initial_pose_errors)

        print(f"[CHECK] Initial max |joint vel|    : {max_initial_vel:.8f} rad/s")
        print(f"[CHECK] Initial max |target error| : {max_initial_target_err:.8f} rad")
        print(f"[CHECK] Initial max |pose error|   : {max_initial_pose_err:.8f} rad")
        print("[CHECK] t=0 gravity holding torque:")
        for prefix, name in zip(
            ("thigh_L", "calf_L", "thigh_R", "calf_R"),
            leg_joint_names,
        ):
            print(
                f"  {name:<18}: "
                f"G={initial_row[f'{prefix}_gravity_comp_tau_Nm']:+9.4f} Nm  "
                f"FF={initial_row[f'{prefix}_gravity_ff_cmd_Nm']:+9.4f} Nm"
            )

        print("[CHECK] t=0 wheel WHY diagnostic:")
        for prefix in ("wheel_L", "wheel_R"):
            print(
                f"  {prefix:<18}: "
                f"qd={initial_row[f'{prefix}_vel_rad_s']:+9.5f} rad/s  "
                f"qdd={initial_row[f'{prefix}_acc_rad_s2']:+10.3f} rad/s^2  "
                f"motor={initial_row[f'{prefix}_applied_tau_Nm']:+9.5f} Nm  "
                f"extResidual={initial_row[f'{prefix}_external_residual_tau_Nm']:+9.5f} Nm"
            )

        if max_initial_vel > 1.0e-4:
            print(
                "[WARN] Non-zero joint velocity exists at diagnostic t=0. "
                "The reset state may not be stationary."
            )
        if max_initial_target_err > 1.0e-5:
            print(
                "[WARN] Primed actuator targets do not match the requested "
                "thigh/calf targets."
            )
        if max_initial_pose_err > 1.0e-4:
            print(
                "[WARN] Reset joint positions do not exactly match the requested "
                "diagnostic posture."
            )

        print(
            f"{0.0:6.2f} | "
            f"{initial_row['thigh_L_pos_rad']:+8.3f} "
            f"{initial_row['calf_L_pos_rad']:+8.3f} "
            f"{initial_row['thigh_R_pos_rad']:+8.3f} "
            f"{initial_row['calf_R_pos_rad']:+8.3f} | "
            f"{initial_row['thigh_L_applied_tau_Nm']:+7.2f} "
            f"{initial_row['calf_L_applied_tau_Nm']:+7.2f} "
            f"{initial_row['thigh_R_applied_tau_Nm']:+7.2f} "
            f"{initial_row['calf_R_applied_tau_Nm']:+7.2f} | "
            f"{initial_row['pitch_deg']:+7.2f} "
            f"{initial_row['pitch_rate_rad_s']:+8.3f}"
        )
        print(
            "       WHY | "
            f"wL qd={initial_row['wheel_L_vel_rad_s']:+7.3f} "
            f"qdd={initial_row['wheel_L_acc_rad_s2']:+9.1f} "
            f"ext={initial_row['wheel_L_external_residual_tau_Nm']:+8.4f} Nm | "
            f"wR qd={initial_row['wheel_R_vel_rad_s']:+7.3f} "
            f"qdd={initial_row['wheel_R_acc_rad_s2']:+9.1f} "
            f"ext={initial_row['wheel_R_external_residual_tau_Nm']:+8.4f} Nm | "
            f"z={initial_row['base_z_m']:.5f}"
        )

        # ---------------------------------------------------------------------
        # Run diagnostic.
        # ---------------------------------------------------------------------

        with torch.inference_mode():
            for step in range(1, num_steps + 1):
                sim_time = step * step_dt

                # Re-apply gravity FF before every env step. Dynamic mode uses
                # the current pose; static mode holds the exact t=0 value.
                if args_cli.gravity_ff_mode == "dynamic":
                    last_gravity_ff_cmd_leg = (
                        args_cli.gravity_ff_scale * get_leg_gravity_compensation()
                    ).clone()
                elif args_cli.gravity_ff_mode == "static":
                    last_gravity_ff_cmd_leg = (
                        args_cli.gravity_ff_scale * nominal_gravity_leg
                    ).clone()
                else:
                    last_gravity_ff_cmd_leg = torch.zeros_like(nominal_gravity_leg)

                robot.set_joint_effort_target(
                    last_gravity_ff_cmd_leg.unsqueeze(0),
                    joint_ids=leg_joint_ids,
                )

                _, _, terminated, truncated, _ = env.step(actions)
                done = bool((terminated | truncated)[0].item())

                # ManagerBasedRLEnv can auto-reset a terminated env inside step().
                # Stop BEFORE reading robot state so reset samples do not pollute
                # the diagnostic unless the user explicitly asks to continue.
                if done and not args_cli.no_stop_on_termination:
                    termination_time = sim_time
                    print(
                        f"\n[INFO] Environment terminated at t={sim_time:.4f} s. "
                        "Stopping before auto-reset data is logged."
                    )
                    break

                if done and termination_time is None:
                    termination_time = sim_time

                row = read_state(sim_time)
                rows.append(row)

                if step % print_every == 0 or step == num_steps:
                    print(
                        f"{sim_time:6.2f} | "
                        f"{row['thigh_L_pos_rad']:+8.3f} "
                        f"{row['calf_L_pos_rad']:+8.3f} "
                        f"{row['thigh_R_pos_rad']:+8.3f} "
                        f"{row['calf_R_pos_rad']:+8.3f} | "
                        f"{row['thigh_L_applied_tau_Nm']:+7.2f} "
                        f"{row['calf_L_applied_tau_Nm']:+7.2f} "
                        f"{row['thigh_R_applied_tau_Nm']:+7.2f} "
                        f"{row['calf_R_applied_tau_Nm']:+7.2f} | "
                        f"{row['pitch_deg']:+7.2f} "
                        f"{row['pitch_rate_rad_s']:+8.3f}"
                    )
                    print(
                        "       WHY | "
                        f"wL qd={row['wheel_L_vel_rad_s']:+7.3f} "
                        f"qdd={row['wheel_L_acc_rad_s2']:+9.1f} "
                        f"ext={row['wheel_L_external_residual_tau_Nm']:+8.4f} Nm | "
                        f"wR qd={row['wheel_R_vel_rad_s']:+7.3f} "
                        f"qdd={row['wheel_R_acc_rad_s2']:+9.1f} "
                        f"ext={row['wheel_R_external_residual_tau_Nm']:+8.4f} Nm | "
                        f"z={row['base_z_m']:.5f}"
                    )

        # ---------------------------------------------------------------------
        # Save CSV.
        # ---------------------------------------------------------------------

        output_path = (
            Path(args_cli.output)
            if args_cli.output
            else make_default_output_path()
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # ---------------------------------------------------------------------
        # Summary.
        # ---------------------------------------------------------------------

        summary_width = 147
        print("\n" + "=" * summary_width)
        print("SUMMARY")
        print("=" * summary_width)
        print(
            f"{'Joint':<18}"
            f"{'Target':>10}"
            f"{'Final':>10}"
            f"{'MaxErr':>11}"
            f"{'ErrDeg':>10}"
            f"{'Mean|tau|':>12}"
            f"{'PeakApp':>11}"
            f"{'PeakCalc':>11}"
            f"{'Peak|P|':>11}"
            f"{'Peak|D|':>11}"
            f"{'Peak|Res|':>12}"
            f"{'Sat%':>9}"
        )
        print("-" * summary_width)

        joint_prefixes = {
            "joint_thigh_L": "thigh_L",
            "joint_calf_L": "calf_L",
            "joint_thigh_R": "thigh_R",
            "joint_calf_R": "calf_R",
        }

        for joint_name in LEG_JOINT_NAMES:
            prefix = joint_prefixes[joint_name]
            target = target_by_name[joint_name]

            positions = [r[f"{prefix}_pos_rad"] for r in rows]
            applied = [r[f"{prefix}_applied_tau_Nm"] for r in rows]
            computed = [r[f"{prefix}_computed_tau_Nm"] for r in rows]
            p_torque = [r[f"{prefix}_P_tau_Nm"] for r in rows]
            d_torque = [r[f"{prefix}_D_tau_Nm"] for r in rows]
            residual = [r[f"{prefix}_tau_residual_Nm"] for r in rows]

            max_err = max(abs(q - target) for q in positions)
            final_pos = positions[-1]
            mean_abs_tau = sum(abs(x) for x in applied) / len(applied)
            peak_applied = max(abs(x) for x in applied)
            peak_computed = max(abs(x) for x in computed)
            peak_p = max(abs(x) for x in p_torque)
            peak_d = max(abs(x) for x in d_torque)
            peak_residual = max(abs(x) for x in residual)

            if leg_effort_limit is not None and leg_effort_limit > 0:
                threshold = 0.98 * leg_effort_limit
                sat_count = sum(abs(x) >= threshold for x in applied)
                sat_pct = 100.0 * sat_count / len(applied)
                sat_text = f"{sat_pct:8.1f}%"
            else:
                sat_text = f"{'n/a':>9}"

            print(
                f"{joint_name:<18}"
                f"{target:>+10.3f}"
                f"{final_pos:>+10.3f}"
                f"{max_err:>11.4f}"
                f"{rad2deg(max_err):>10.2f}"
                f"{mean_abs_tau:>12.3f}"
                f"{peak_applied:>11.3f}"
                f"{peak_computed:>11.3f}"
                f"{peak_p:>11.3f}"
                f"{peak_d:>11.3f}"
                f"{peak_residual:>12.3f}"
                f"{sat_text}"
            )

        peak_pitch = max(abs(r["pitch_deg"]) for r in rows)
        peak_pitch_rate = max(abs(r["pitch_rate_rad_s"]) for r in rows)
        max_base_dx = max(abs(r["base_x_m"] - rows[0]["base_x_m"]) for r in rows)
        peak_base_vx = max(abs(r["base_vx_m_s"]) for r in rows)

        wheel_l_peak_tau = max(abs(r["wheel_L_applied_tau_Nm"]) for r in rows)
        wheel_r_peak_tau = max(abs(r["wheel_R_applied_tau_Nm"]) for r in rows)

        print("-" * summary_width)
        print(f"Peak |pitch|          : {peak_pitch:.3f} deg")
        print(f"Peak |pitch rate|     : {peak_pitch_rate:.4f} rad/s")
        print(f"Max |base dx|         : {max_base_dx:.6f} m")
        print(f"Peak |base vx|        : {peak_base_vx:.6f} m/s")
        print(f"Initial base z        : {rows[0]['base_z_m']:.6f} m")
        print(f"Minimum base z        : {min(r['base_z_m'] for r in rows):.6f} m")
        print(f"Peak wheel_L |tau|    : {wheel_l_peak_tau:.6f} Nm")
        print(f"Peak wheel_R |tau|    : {wheel_r_peak_tau:.6f} Nm")

        print("\nWHEEL WHY DIAGNOSTIC")
        print(
            f"{'Wheel':<14}"
            f"{'Iaxis':>12}"
            f"{'Peak|qd|':>12}"
            f"{'Peak|qdd|':>13}"
            f"{'Peak|motor|':>13}"
            f"{'Peak|IDreq|':>13}"
            f"{'Peak|ext|':>12}"
            f"{'Ext/limit':>12}"
            f"{'Peak|Jtau|':>13}"
        )
        print("-" * 114)
        for i, prefix in enumerate(("wheel_L", "wheel_R")):
            qd_vals = [r[f"{prefix}_vel_rad_s"] for r in rows]
            qdd_vals = [r[f"{prefix}_acc_rad_s2"] for r in rows]
            motor_vals = [r[f"{prefix}_applied_tau_Nm"] for r in rows]
            id_vals = [r[f"{prefix}_ID_required_tau_Nm"] for r in rows]
            ext_vals = [r[f"{prefix}_external_residual_tau_Nm"] for r in rows]
            joint_axis_vals = [r[f"{prefix}_joint_axis_tau_Nm"] for r in rows]

            finite_ext = [abs(x) for x in ext_vals if math.isfinite(x)]
            finite_id = [abs(x) for x in id_vals if math.isfinite(x)]
            finite_jtau = [abs(x) for x in joint_axis_vals if math.isfinite(x)]
            peak_ext = max(finite_ext) if finite_ext else float("nan")
            peak_id = max(finite_id) if finite_id else float("nan")
            peak_jtau = max(finite_jtau) if finite_jtau else float("nan")
            if wheel_effort_limit is not None and wheel_effort_limit > 0 and math.isfinite(peak_ext):
                ext_ratio = peak_ext / wheel_effort_limit
            else:
                ext_ratio = float("nan")

            print(
                f"{prefix:<14}"
                f"{float(wheel_axis_inertia[i]):>12.3e}"
                f"{max(abs(x) for x in qd_vals):>12.3f}"
                f"{max(abs(x) for x in qdd_vals):>13.1f}"
                f"{max(abs(x) for x in motor_vals):>13.4f}"
                f"{peak_id:>13.4f}"
                f"{peak_ext:>12.4f}"
                f"{ext_ratio:>12.2f}x"
                f"{peak_jtau:>13.4f}"
            )

            # First clear onset of wheel motion. This is not called "contact time"
            # because the point of this diagnostic is to infer WHY the wheel moves.
            onset = next(
                (r for r in rows if abs(r[f"{prefix}_vel_rad_s"]) >= 0.10),
                None,
            )
            if onset is not None:
                print(
                    f"  -> first |qd|>=0.10 at t={onset['time_s']:.4f}s: "
                    f"qd={onset[f'{prefix}_vel_rad_s']:+.4f}, "
                    f"qdd={onset[f'{prefix}_acc_rad_s2']:+.1f}, "
                    f"motor={onset[f'{prefix}_applied_tau_Nm']:+.4f} Nm, "
                    f"IDreq={onset[f'{prefix}_ID_required_tau_Nm']:+.4f} Nm, "
                    f"extResidual={onset[f'{prefix}_external_residual_tau_Nm']:+.4f} Nm "
                    f"({onset[f'{prefix}_external_over_0p45x']:.2f}x limit), "
                    f"jointAxisReaction={onset[f'{prefix}_joint_axis_tau_Nm']:+.4f} Nm, "
                    f"base_z={onset['base_z_m']:.6f} m"
                )

        print(
            "\nInterpretation: wheel external_residual_tau = "
            "(M*qdd + C + G)_wheel - applied_motor_tau. PhysX inverse dynamics "
            "does not include contact, damping, or joint friction, so a large "
            "impact-time residual is evidence of generalized load from contact/"
            "passive effects. It can exceed the wheel actuator's 0.45 Nm limit; "
            "that limit constrains motor command, not external joint loading."
        )

        print("\nGRAVITY / HOLDING TORQUE")
        print(
            f"{'Joint':<18}"
            f"{'G@t0[Nm]':>12}"
            f"{'Peak|G|':>12}"
            f"{'Peak|FF|':>12}"
            f"{'FF mode':>12}"
        )
        print("-" * 66)
        for joint_name in LEG_JOINT_NAMES:
            prefix = joint_prefixes[joint_name]
            gravity_vals = [r[f"{prefix}_gravity_comp_tau_Nm"] for r in rows]
            ff_vals = [r[f"{prefix}_gravity_ff_cmd_Nm"] for r in rows]
            print(
                f"{joint_name:<18}"
                f"{gravity_vals[0]:>+12.4f}"
                f"{max(abs(x) for x in gravity_vals):>12.4f}"
                f"{max(abs(x) for x in ff_vals):>12.4f}"
                f"{args_cli.gravity_ff_mode:>12}"
            )

        print(
            "\nInterpretation: G@t0 is the PhysX generalized joint torque required "
            "to counteract gravity at the exact nominal pose before the first "
            "physics step. This is the key static holding-torque value."
        )

        if termination_time is None:
            print(f"Termination           : NO within {rows[-1]['time_s']:.3f} s")
        else:
            print(f"First termination     : {termination_time:.4f} s")

        print(f"CSV                    : {output_path}")
        print("=" * summary_width)

    finally:
        env.close()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
