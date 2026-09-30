#!/usr/bin/env python3
"""Run the frozen wheel-only RL policy through the LQR push-test envelope.

The fixed asset remains at the policy's training pose (thigh=-0.50,
calf=+0.87).  A body-level horizontal pulse (default 50 ms, along the heading,
at the COM) is split over policy steps by overlap with the pulse window
(100 Hz: five full steps; 50 Hz: 1, 1, 0.5), so its total impulse is exactly
F*T without changing the trained controller frequency.  ``--pulse-duration``
and ``--push-angle-deg`` give the out-of-distribution pushes (e.g. 200 ms,
30 deg off the heading).

Evaluation-only overrides (the training cfg files are not touched):
  * training randomisation off: observation noise, action delay, mass and COM
    randomisation removed, robot friction back to the nominal 0.8;
  * every termination term is removed, so the environment never auto-resets;
  * failure is judged only by |pitch - initial pitch| >= 15 deg;
  * ground contact of ``base_link_root`` (which carries the merged leg
    collisions) is recorded in ``ground_contact`` but never counts as failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Fixed-Flat-v0")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("logs/rsl_rl/tancho_v3_fixed/2026-09-21_02-35-57/exported/policy.pt"),
)
parser.add_argument("--duration", type=float, default=5.0)
parser.add_argument("--output-dir", type=Path, default=Path("logs/evaluation"))
parser.add_argument(
    "--symmetric-actions",
    action="store_true",
    help="Apply the mean of the two wheel actions to both wheels (planar, LQR-equivalent DOF; no differential/yaw).",
)
parser.add_argument("--obs-dim", type=int, default=None, help="Expected policy observation size (checked if given).")
parser.add_argument("--pulse-duration", type=float, default=0.05, help="Push duration (s).")
parser.add_argument("--push-angle-deg", type=float, default=0.0, help="Push direction from the heading about +z (deg).")
parser.add_argument("--forces", type=str, default=None, help="Comma-separated push forces (N); default: the LQR sweep.")
parser.add_argument("--label", type=str, default="", help="Suffix for the output directory.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils import parse_env_cfg

import tancho_v3_lab
import tancho_v3_lab.tasks  # noqa: F401


# 0 N is the undisturbed standing (idle) case.
FORCES_N = (0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34, 36, 38, 39, 40, 42, 45, 50, 60)
if args.forces:
    FORCES_N = tuple(float(v) for v in args.forces.split(","))
PULSE_DURATION_S = args.pulse_duration
PUSH_ANGLE_DEG = args.push_angle_deg
FAILURE_PITCH_DEG = 15.0
SETTLE_BAND_DEG = 1.0
SETTLE_DWELL_S = 0.5
TORQUE_LIMIT_NM = 0.45
SAT_TOL_NM = 0.005
# Same threshold as the training ``base_contact`` term (illegal_contact, 10 N).
GROUND_CONTACT_THRESHOLD_N = 10.0
WHEEL_RADIUS_M = 0.03614  # wheel collision cylinder radius
WHEEL_LIFTOFF_N = 0.5
SLIP_THRESHOLD_M_S = 0.1
HOME_TOLERANCE_M = 0.05
HOME_DWELL_S = 0.5
CHECK_TIME_S = 5.0
BALANCE_BAND_DEG = 1.0
IDLE_YAW_LIMIT_DEG = 5.0
PUSH_YAW_LIMIT_DEG = 10.0


def pitch_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))


def yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_rotate_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame points ``vec`` (P, 3) by per-env quaternions (N, 4) -> (N, P, 3)."""
    w = quat[:, None, 0:1]
    u = quat[:, None, 1:4]
    v = vec[None].expand(quat.shape[0], -1, -1)
    t = 2.0 * torch.cross(u.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(u.expand_as(v), t, dim=-1)


def _rpy_matrix(rpy: list[float]) -> torch.Tensor:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def base_collision_points(urdf_path: Path) -> tuple[list[str], torch.Tensor, list[int]]:
    """Sample points of every primitive collision shape on ``base_link_root`` (body frame).

    Boxes use their 8 corners; cylinders 64 rim points on each cap.  Returns the
    shape names, stacked points and the shape index of each point.
    """
    link = next(l for l in ET.parse(urdf_path).getroot().iter("link") if l.get("name") == "base_link_root")
    names, points, owner = [], [], []
    for collision in link.findall("collision"):
        origin = collision.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rot = _rpy_matrix([float(v) for v in origin.get("rpy", "0 0 0").split()])
        geometry = collision.find("geometry")
        box, cylinder = geometry.find("box"), geometry.find("cylinder")
        if box is not None:
            half = [0.5 * float(v) for v in box.get("size").split()]
            local = torch.tensor([[sx * half[0], sy * half[1], sz * half[2]]
                                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        elif cylinder is not None:
            radius, length = float(cylinder.get("radius")), float(cylinder.get("length"))
            angle = torch.linspace(0.0, 2.0 * math.pi, 65)[:-1]
            rim = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
            local = torch.cat([torch.cat((rim, torch.full((64, 1), z)), dim=-1) for z in (-length / 2, length / 2)])
        else:
            continue
        index = len(names)
        names.append(collision.get("name"))
        points.append(local @ rot.T + torch.tensor(xyz))
        owner.extend([index] * local.shape[0])
    return names, torch.cat(points), owner


def dwell_time(rows: list[dict[str, float]], inside: list[bool]) -> tuple[float | None, int]:
    """Settling time from push onset for a band condition held for SETTLE_DWELL_S.

    Returns (time, left_band).  If the condition never fails, the response never
    left the band: (0.0, 0).  Otherwise the search starts at the first exit and
    returns the first time from which the condition holds for SETTLE_DWELL_S
    (None if that never happens with enough data left).
    """
    first_out = next((i for i, ok in enumerate(inside) if not ok), None)
    if first_out is None:
        return 0.0, 0
    for index in range(first_out, len(rows)):
        end = rows[index]["time_s"] + SETTLE_DWELL_S
        if rows[-1]["time_s"] < end - 1.0e-9:
            return None, 1
        if all(ok for row, ok in zip(rows[index:], inside[index:]) if row["time_s"] <= end + 1.0e-9):
            return rows[index]["time_s"], 1
    return None, 1


def final_entry_time(rows: list[dict[str, float]], inside: list[bool]) -> float | None:
    """Time after which the condition holds until the end of the run (>= SETTLE_DWELL_S of data left)."""
    last_out = max((r["time_s"] for r, ok in zip(rows, inside) if not ok), default=None)
    if last_out is None:
        return 0.0
    entry = next((r["time_s"] for r in rows if r["time_s"] > last_out), None)
    if entry is None or rows[-1]["time_s"] < entry + SETTLE_DWELL_S - 1.0e-9:
        return None
    return entry


def legacy_settling_time(rows: list[dict[str, float]]) -> float | None:
    """Pre-2026-10-01 definition, kept for traceability: first time >= pulse end with
    |pitch - initial| <= 1 deg for 0.5 s.  A push that never leaves the band returns
    the pulse end (0.05 s), which is not a recovery time."""
    for index, row in enumerate(rows):
        if row["time_s"] < PULSE_DURATION_S - 1.0e-9:
            continue
        end = row["time_s"] + SETTLE_DWELL_S
        if rows[-1]["time_s"] < end - 1.0e-9:
            return None
        window = [item for item in rows[index:] if item["time_s"] <= end + 1.0e-9]
        if max(abs(item["pitch_dev_deg"]) for item in window) <= SETTLE_BAND_DEG:
            return row["time_s"]
    return None


def pulse_fraction(step: int, dt: float) -> float:
    """Fraction of policy step ``step`` that overlaps the [0, PULSE_DURATION_S] pulse window."""
    overlap = min((step + 1) * dt, PULSE_DURATION_S) - step * dt
    return round(max(overlap, 0.0) / dt, 9)


def main() -> None:
    print(f"TANCHO_V3_LAB_MODULE={tancho_v3_lab.__file__}", flush=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(FORCES_N), use_fabric=True)
    cfg.seed = 42
    cfg.episode_length_s = max(cfg.episode_length_s, args.duration + 1.0)
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physx.enable_external_forces_every_iteration = True
    # Keep the comparison deterministic and prevent the training curriculum's
    # independent interval event from adding a second disturbance.
    cfg.curriculum = None
    cfg.events.push_robot = None
    # Evaluation only: no termination terms -> no auto reset.  Failure is judged
    # on pitch below; base contact is only logged.
    for term_name in list(cfg.terminations.to_dict()):
        setattr(cfg.terminations, term_name, None)
    # Stand still: zero velocity / yaw-rate command in every env, whatever the task trained with.
    command = cfg.commands.base_velocity
    command.rel_standing_envs = 1.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    # Evaluation only: no training randomisation (nominal plant, clean observations).
    disabled = []
    for group in ("policy", "critic"):
        group_cfg = getattr(cfg.observations, group, None)
        if group_cfg is not None and getattr(group_cfg, "enable_corruption", False):
            group_cfg.enable_corruption = False
            disabled.append(f"observations.{group}.enable_corruption")
    for action_name, action_cfg in cfg.actions.to_dict().items():
        if isinstance(action_cfg, dict) and action_cfg.get("delay_probability"):
            getattr(cfg.actions, action_name).delay_probability = 0.0
            disabled.append(f"actions.{action_name}.delay_probability")
    for event_name in ("add_base_mass", "randomize_com"):
        if getattr(cfg.events, event_name, None) is not None:
            setattr(cfg.events, event_name, None)
            disabled.append(f"events.{event_name}")
    friction = getattr(cfg.events, "randomize_friction", None)
    if friction is not None:
        friction.params["static_friction_range"] = (0.8, 0.8)
        friction.params["dynamic_friction_range"] = (0.8, 0.8)
        disabled.append("events.randomize_friction -> 0.8")
    print(f"EVAL_RANDOMIZATION_OFF={disabled}", flush=True)
    env = gym.make(args.task, cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    core = env.unwrapped
    robot = core.scene["robot"]
    contact = core.scene["contact_forces"]
    if robot.joint_names != ["joint_wheel_L", "joint_wheel_R"]:
        raise RuntimeError(f"Expected wheel-only asset, got {robot.joint_names}")
    if core.termination_manager.active_terms:
        raise RuntimeError(f"Termination terms still active: {core.termination_manager.active_terms}")
    checkpoint = args.checkpoint.resolve()
    policy = torch.jit.load(str(checkpoint), map_location=core.device).eval()
    obs = env.get_observations()
    if args.obs_dim is not None and obs["policy"].shape[-1] != args.obs_dim:
        raise RuntimeError(f"Expected {args.obs_dim}-D policy observation, got {tuple(obs['policy'].shape)}")
    dt = float(core.step_dt)
    if dt <= 0.0:
        raise RuntimeError(f"Invalid policy step {dt}")
    print(f"POLICY_DT_S={dt} OBS_DIM={obs['policy'].shape[-1]}", flush=True)

    urdf_path = Path(cfg.scene.robot.spawn.asset_path)
    shape_names, shape_points, point_owner = base_collision_points(urdf_path)
    shape_points = shape_points.to(core.device, torch.float32)
    point_owner_t = torch.tensor(point_owner, device=core.device)

    body_ids, _ = robot.find_bodies("base_link_root", preserve_order=True)
    wheel_body_ids, _ = robot.find_bodies(["wheel_L", "wheel_R"], preserve_order=True)
    base_sensor_id = contact.find_bodies("base_link_root")[0][0]
    wheel_sensor_ids = contact.find_bodies(["wheel_L", "wheel_R"], preserve_order=True)[0]
    force_levels = torch.tensor(FORCES_N, device=core.device, dtype=torch.float32)
    columns: list[str] = []
    sample_times: list[float] = []
    sample_batches: list[torch.Tensor] = []
    failed = torch.zeros(len(FORCES_N), dtype=torch.bool, device=core.device)
    unexpected_reset = torch.zeros_like(failed)
    steps = math.ceil(args.duration / dt)

    def record(time_s: float, applied_force: torch.Tensor) -> None:
        # Keep all samples on the GPU and transfer once after the rollout.
        quat = robot.data.root_link_quat_w
        pos = robot.data.root_link_pos_w - core.scene.env_origins
        pitch = torch.rad2deg(pitch_wxyz(quat))
        base_force = torch.linalg.norm(contact.data.net_forces_w[:, base_sensor_id], dim=-1)
        wheel_fn = contact.data.net_forces_w[:, wheel_sensor_ids, 2]
        wheel_v = robot.data.body_lin_vel_w[:, wheel_body_ids, 0]
        wheel_w = robot.data.body_ang_vel_w[:, wheel_body_ids, 1]
        # Contact-patch velocity v_center - w_y * r (0 when rolling without slip).
        wheel_slip = wheel_v - wheel_w * WHEEL_RADIUS_M
        point_z = (quat_rotate_wxyz(quat, shape_points) + pos[:, None])[..., 2]
        shape_z = torch.full((len(FORCES_N), len(shape_names)), float("inf"), device=core.device)
        shape_z = shape_z.scatter_reduce(1, point_owner_t.expand_as(point_z), point_z, reduce="amin")
        lowest_z, lowest_shape = shape_z.min(dim=1)
        rel = pos[:, :2] - pos0[:, :2]
        heading0 = heading0_w[:, :2]
        yaw = yaw_wxyz(quat)
        yaw_change = torch.atan2(torch.sin(yaw - yaw0), torch.cos(yaw - yaw0))
        qd = robot.data.joint_vel
        # PhysX wraps continuous-joint angles to [-2*pi, 2*pi]: accumulate unwrapped increments.
        raw = robot.data.joint_pos
        step_delta = torch.remainder(raw - wheel_prev + 2.0 * math.pi, 4.0 * math.pi) - 2.0 * math.pi
        wheel_travel.add_(step_delta.mean(dim=1, keepdim=True))
        wheel_prev.copy_(raw)
        fields = {
            "pitch_deg": pitch[:, None],
            "yaw_deg": torch.rad2deg(yaw_wxyz(quat))[:, None],
            "pitch_rate_rad_s": robot.data.root_ang_vel_b[:, 1:2],
            "base_x_m": pos[:, 0:1],
            "base_y_m": pos[:, 1:2],
            "base_vx_m_s": robot.data.root_lin_vel_w[:, 0:1],
            "base_vy_m_s": robot.data.root_lin_vel_w[:, 1:2],
            "displacement_m": torch.linalg.vector_norm(rel, dim=1, keepdim=True),
            "forward_displacement_m": (rel * heading0).sum(dim=1, keepdim=True),
            "forward_velocity_m_s": (robot.data.root_lin_vel_w[:, :2] * heading0).sum(dim=1, keepdim=True),
            "odometry_displacement_m": WHEEL_RADIUS_M * wheel_travel,
            "yaw_change_deg": torch.rad2deg(yaw_change)[:, None],
            "yaw_rate_rad_s": robot.data.root_ang_vel_w[:, 2:3],
            "wheel_speed_diff_rad_s": qd[:, 1:2] - qd[:, 0:1],
            "wheel_L_q_rad": robot.data.joint_pos[:, 0:1],
            "wheel_R_q_rad": robot.data.joint_pos[:, 1:2],
            "wheel_L_qd_rad_s": robot.data.joint_vel[:, 0:1],
            "wheel_R_qd_rad_s": robot.data.joint_vel[:, 1:2],
            "wheel_L_qdd_rad_s2": robot.data.joint_acc[:, 0:1],
            "wheel_R_qdd_rad_s2": robot.data.joint_acc[:, 1:2],
            "wheel_L_tau_Nm": robot.data.applied_torque[:, 0:1],
            "wheel_R_tau_Nm": robot.data.applied_torque[:, 1:2],
            "wheel_L_normal_force_N": wheel_fn[:, 0:1],
            "wheel_R_normal_force_N": wheel_fn[:, 1:2],
            "wheel_L_slip_m_s": wheel_slip[:, 0:1],
            "wheel_R_slip_m_s": wheel_slip[:, 1:2],
            "base_contact_force_N": base_force[:, None],
            "ground_contact": (base_force > GROUND_CONTACT_THRESHOLD_N).float()[:, None],
            "lowest_shape_z_mm": 1000.0 * lowest_z[:, None],
            "lowest_shape_index": lowest_shape.float()[:, None],
            "applied_force_N": applied_force[:, None],
            "failed": failed.float()[:, None],
            "unexpected_reset": unexpected_reset.float()[:, None],
        }
        if not columns:
            columns.extend(fields)
        sample_times.append(time_s)
        sample_batches.append(torch.cat(list(fields.values()), dim=1).detach())

    def make_rows(packed_samples: list[list[list[float]]]) -> list[list[dict]]:
        rows_by_case: list[list[dict]] = [[] for _ in FORCES_N]
        for time_s, batch in zip(sample_times, packed_samples):
            for i, level in enumerate(FORCES_N):
                values = dict(zip(columns, batch[i]))
                rows_by_case[i].append(
                    {
                        "controller": "trained_rl",
                        "push_force_N": float(level),
                        "pulse_duration_s": PULSE_DURATION_S,
                        "time_s": round(time_s, 6),
                        "applied_force_N": values["applied_force_N"],
                        "pitch_deg": values["pitch_deg"],
                        "pitch_dev_deg": values["pitch_deg"] - pitch0_deg[i],
                        "pitch_rad": math.radians(values["pitch_deg"]),
                        **{k: values[k] for k in columns if k not in ("pitch_deg", "applied_force_N",
                                                                     "lowest_shape_index")},
                        "lowest_shape": shape_names[int(values["lowest_shape_index"])],
                    }
                )
        return rows_by_case

    zero_force = torch.zeros(len(FORCES_N), device=core.device)
    pos0 = (robot.data.root_link_pos_w - core.scene.env_origins).clone()
    yaw0 = yaw_wxyz(robot.data.root_link_quat_w).clone()
    wheel_prev = robot.data.joint_pos.clone()
    wheel_travel = torch.zeros(len(FORCES_N), 1, device=core.device)
    heading0_w = torch.stack((torch.cos(yaw0), torch.sin(yaw0), torch.zeros_like(yaw0)), dim=1)
    push_angle = math.radians(PUSH_ANGLE_DEG)
    push_dir_w = torch.stack(
        (torch.cos(yaw0 + push_angle), torch.sin(yaw0 + push_angle), torch.zeros_like(yaw0)), dim=1
    )
    pitch0_deg = torch.rad2deg(pitch_wxyz(robot.data.root_link_quat_w)).cpu().tolist()
    pitch0_t = torch.tensor(pitch0_deg, device=core.device)
    # Static balance pitch: rotate about the wheel axle until the whole-robot COM is
    # above it (COM ahead of the axle -> lean back).  Positive pitch is nose-down.
    mass = robot.data.default_mass.to(core.device)
    com_w = (robot.data.body_com_pos_w * mass[..., None]).sum(dim=1) / mass.sum(dim=1, keepdim=True)
    axle_w = robot.data.body_link_pos_w[:, wheel_body_ids].mean(dim=1)
    heading_w = quat_rotate_wxyz(robot.data.root_link_quat_w, torch.tensor([[1.0, 0.0, 0.0]], device=core.device))[:, 0]
    rel = com_w - axle_w
    balance_deg = (pitch0_t - torch.rad2deg(torch.atan2((rel * heading_w).sum(dim=1), rel[:, 2]))).cpu().tolist()
    record(0.0, zero_force)
    for step in range(steps):
        # Scale each policy step by its overlap with the pulse window so the total
        # impulse is exactly F*0.05 N*s at the policy's own rate.
        fraction = pulse_fraction(step, dt)
        applied = force_levels * fraction
        if fraction > 0.0 or pulse_fraction(step - 1, dt) > 0.0:
            # World-fixed direction, converted with the current pose and applied in the link
            # frame at the COM: the wrench composer's is_global path uses the link pose cached
            # at the last reset.
            force_w = applied[:, None] * push_dir_w
            forces = quat_apply_inverse(robot.data.root_link_quat_w, force_w)[:, None, :].contiguous()
            robot.permanent_wrench_composer.set_forces_and_torques(
                forces=forces,
                torques=torch.zeros_like(forces),
                body_ids=torch.tensor(body_ids, dtype=torch.int32, device=core.device),
                is_global=False,
            )
        with torch.inference_mode():
            # The exported TorchScript policy takes the flat policy tensor, not the wrapper's TensorDict.
            actions = policy(obs["policy"])
            if args.symmetric_actions:
                actions = actions.mean(dim=1, keepdim=True).expand_as(actions).contiguous()
            obs, _, dones, _ = env.step(actions)
        # With no termination terms a done means something reset the env anyway: flag it.
        unexpected_reset |= dones.bool()
        # Failure is judged only on deviation from the initial pose (LQR 15 deg threshold).
        pitch_dev = torch.abs(torch.rad2deg(pitch_wxyz(robot.data.root_link_quat_w)) - pitch0_t)
        failed |= pitch_dev >= FAILURE_PITCH_DEG
        record((step + 1) * dt, applied)
    if unexpected_reset.any():
        raise RuntimeError(f"Unexpected env reset for forces {force_levels[unexpected_reset].tolist()}")

    rows_by_case = make_rows(torch.stack(sample_batches).cpu().tolist())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.label}" if args.label else ""
    output_dir = args.output_dir.resolve() / f"trained_rl_lqr_axes_{stamp}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = output_dir / "trained_rl_push_timeseries.csv"
    fields = list(rows_by_case[0][0])
    with timeseries_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case_rows in rows_by_case:
            writer.writerows(case_rows)

    summary_path = output_dir / "trained_rl_push_summary.csv"
    summary_fields = [
        "controller", "push_force_N", "pulse_duration_s", "push_angle_deg", "recovered",
        "max_pitch_deviation_deg", "max_abs_pitch_deg",
        # Recovery times from push onset; 0 with *_left_band = 0 means the band was never left.
        "settling_time_s", "pitch_left_band", "final_settling_time_s",
        "pose_position_settling_time_s", "pose_position_left_band", "settling_time_legacy_s",
        "peak_wheel_torque_Nm", "wheel_torque_limit_Nm",
        "torque_saturated", "first_saturation_time_s", "saturation_duration_s",
        "saturation_before_failure_s", "failure_time_s",
        "ground_contact", "first_ground_contact_time_s", "ground_contact_shape",
        "pitch_dev_at_first_ground_contact_deg", "ground_contact_duration_s",
        "max_wheel_speed_rad_s", "max_abs_slip_m_s", "first_slip_time_s",
        "first_wheel_liftoff_time_s", "wheel_liftoff_duration_s", "initial_pitch_deg",
        "balance_pitch_deg", "max_abs_pitch_from_balance_deg",
        "max_displacement_m", "displacement_at_5s_m", "velocity_at_5s_m_s",
        "return_time_s", "return_left_band", "return_time_final_s", "final_displacement_m",
        "returned_within_5s", "max_abs_yaw_deg", "final_yaw_change_deg",
        "max_abs_yaw_rate_rad_s", "max_abs_wheel_speed_diff_rad_s",
        "idle_pass", "push_pass",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for i, (level, case_rows, did_fail) in enumerate(zip(FORCES_N, rows_by_case, failed.tolist())):
            tau = [max(abs(r["wheel_L_tau_Nm"]), abs(r["wheel_R_tau_Nm"])) for r in case_rows]
            sat = [t >= TORQUE_LIMIT_NM - SAT_TOL_NM for t in tau]
            contact_rows = [r for r in case_rows if r["ground_contact"] > 0.5]
            fail_t = next((r["time_s"] for r in case_rows if r["failed"] > 0.5), None)
            slip_t = next((r["time_s"] for r in case_rows
                           if max(abs(r["wheel_L_slip_m_s"]), abs(r["wheel_R_slip_m_s"])) > SLIP_THRESHOLD_M_S), "")
            lift = [min(r["wheel_L_normal_force_N"], r["wheel_R_normal_force_N"]) < WHEEL_LIFTOFF_N for r in case_rows]
            disp = [r["displacement_m"] for r in case_rows]
            at_check = min(case_rows, key=lambda r: abs(r["time_s"] - CHECK_TIME_S))
            # Bands: pitch within +-1 deg of the static balance pitch; position within 5 cm of t = 0.
            pitch_ok = [abs(r["pitch_deg"] - balance_deg[i]) <= SETTLE_BAND_DEG for r in case_rows]
            home_ok = [d <= HOME_TOLERANCE_M for d in disp]
            both_ok = [p and h for p, h in zip(pitch_ok, home_ok)]
            if did_fail:
                settle, left = None, 1
                final_settle = both_t = None
                both_left = 1
            else:
                settle, left = dwell_time(case_rows, pitch_ok)
                final_settle = final_entry_time(case_rows, pitch_ok)
                both_t, both_left = dwell_time(case_rows, both_ok)
            return_t, return_left = dwell_time(case_rows, home_ok)
            return_final = final_entry_time(case_rows, home_ok)
            from_balance = max(abs(r["pitch_deg"] - balance_deg[i]) for r in case_rows)
            max_yaw = max(abs(r["yaw_change_deg"]) for r in case_rows)
            max_abs_pitch = max(abs(r["pitch_deg"]) for r in case_rows)
            returned = return_t is not None and return_t <= CHECK_TIME_S
            legacy = None if did_fail else legacy_settling_time(case_rows)
            writer.writerow(
                {
                    "controller": "trained_rl",
                    "push_force_N": level,
                    "pulse_duration_s": PULSE_DURATION_S,
                    "push_angle_deg": PUSH_ANGLE_DEG,
                    "recovered": int((not did_fail) and settle is not None),
                    "max_pitch_deviation_deg": max(abs(row["pitch_dev_deg"]) for row in case_rows),
                    "max_abs_pitch_deg": max_abs_pitch,
                    "settling_time_s": "" if settle is None else settle,
                    "pitch_left_band": left,
                    "final_settling_time_s": "" if final_settle is None else final_settle,
                    "pose_position_settling_time_s": "" if both_t is None else both_t,
                    "pose_position_left_band": both_left,
                    "settling_time_legacy_s": "" if legacy is None else legacy,
                    "peak_wheel_torque_Nm": max(tau),
                    "wheel_torque_limit_Nm": TORQUE_LIMIT_NM,
                    "torque_saturated": int(any(sat)),
                    "first_saturation_time_s": next((r["time_s"] for r, s in zip(case_rows, sat) if s), ""),
                    "saturation_duration_s": round(sum(sat[1:]) * dt, 4),
                    "saturation_before_failure_s": round(dt * sum(
                        1 for r, s in zip(case_rows[1:], sat[1:]) if s and (fail_t is None or r["time_s"] <= fail_t)), 4),
                    "failure_time_s": "" if fail_t is None else fail_t,
                    "ground_contact": int(bool(contact_rows)),
                    "first_ground_contact_time_s": contact_rows[0]["time_s"] if contact_rows else "",
                    "ground_contact_shape": contact_rows[0]["lowest_shape"] if contact_rows else "",
                    "pitch_dev_at_first_ground_contact_deg": contact_rows[0]["pitch_dev_deg"] if contact_rows else "",
                    "ground_contact_duration_s": round(len(contact_rows) * dt, 4),
                    "max_wheel_speed_rad_s": max(
                        max(abs(r["wheel_L_qd_rad_s"]), abs(r["wheel_R_qd_rad_s"])) for r in case_rows),
                    "max_abs_slip_m_s": max(
                        max(abs(r["wheel_L_slip_m_s"]), abs(r["wheel_R_slip_m_s"])) for r in case_rows),
                    "first_slip_time_s": slip_t,
                    "first_wheel_liftoff_time_s": next((r["time_s"] for r, l in zip(case_rows, lift) if l), ""),
                    "wheel_liftoff_duration_s": round(dt * sum(lift[1:]), 4),
                    "initial_pitch_deg": pitch0_deg[i],
                    "balance_pitch_deg": balance_deg[i],
                    "max_abs_pitch_from_balance_deg": from_balance,
                    "max_displacement_m": max(disp),
                    "displacement_at_5s_m": at_check["displacement_m"],
                    "velocity_at_5s_m_s": math.hypot(at_check["base_vx_m_s"], at_check["base_vy_m_s"]),
                    "return_time_s": "" if return_t is None else return_t,
                    "return_left_band": return_left,
                    "return_time_final_s": "" if return_final is None else return_final,
                    "final_displacement_m": disp[-1],
                    "returned_within_5s": int(returned),
                    "max_abs_yaw_deg": max_yaw,
                    "final_yaw_change_deg": case_rows[-1]["yaw_change_deg"],
                    "max_abs_yaw_rate_rad_s": max(abs(r["yaw_rate_rad_s"]) for r in case_rows),
                    "max_abs_wheel_speed_diff_rad_s": max(abs(r["wheel_speed_diff_rad_s"]) for r in case_rows),
                    "idle_pass": "" if level != 0 else int(
                        max(disp) < HOME_TOLERANCE_M and from_balance <= BALANCE_BAND_DEG
                        and max_yaw < IDLE_YAW_LIMIT_DEG),
                    "push_pass": "" if level == 0 else int(
                        (not did_fail) and returned and max_abs_pitch <= FAILURE_PITCH_DEG
                        and max_yaw < PUSH_YAW_LIMIT_DEG),
                }
            )
    metadata = {
        "controller": "trained_rl",
        "checkpoint": str(checkpoint),
        "task": args.task,
        "tancho_v3_lab_module": tancho_v3_lab.__file__,
        "urdf": str(urdf_path),
        "training_pose_rad": {"thigh": -0.50, "calf": 0.87},
        "push_forces_N": list(FORCES_N),
        "nominal_pulse_duration_s": PULSE_DURATION_S,
        "push_angle_deg": PUSH_ANGLE_DEG,
        "push_point": "base_link_root COM",
        "policy_dt_s": dt,
        "sim_dt_s": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "pulse_discretization": {
            "per_step_force_fraction": [pulse_fraction(k, dt) for k in range(math.ceil(PULSE_DURATION_S / dt) + 1)],
            "impulse_per_newton_Ns": sum(pulse_fraction(k, dt) for k in range(math.ceil(PULSE_DURATION_S / dt) + 1)) * dt,
            "note": "fraction = overlap of each policy step with the pulse window; exact F*T impulse",
        },
        "evaluation_overrides": {
            "terminations": "all removed (no auto reset)",
            "failure": f"|pitch - initial| >= {FAILURE_PITCH_DEG} deg only",
            "ground_contact": f"base_link_root net contact force > {GROUND_CONTACT_THRESHOLD_N} N, logged only",
            "curriculum": None,
            "push_robot_event": None,
            "randomization_off": disabled,
        },
        "base_collision_shapes": shape_names,
        "duration_s": args.duration,
        "symmetric_actions": args.symmetric_actions,
        "metrics": {
            "displacement": "horizontal distance of the root link from its t=0 position",
            "settling_time_s": f"from push onset: first time after the first exit from |pitch - balance| <= "
                               f"{SETTLE_BAND_DEG} deg from which it holds for {SETTLE_DWELL_S} s; "
                               "0 with pitch_left_band=0 if the band was never left",
            "final_settling_time_s": "last entry into the pitch band, staying until the end of the run",
            "pose_position_settling_time_s": f"as settling_time_s for pitch band AND displacement <= "
                                             f"{HOME_TOLERANCE_M} m",
            "settling_time_legacy_s": "old definition (search starts at pulse end, band around the initial pitch): "
                                      "a push that never leaves the band reports the pulse end (0.05 s)",
            "return_time_s": f"as settling_time_s for displacement <= {HOME_TOLERANCE_M} m held {HOME_DWELL_S} s",
            "return_time_final_s": f"time after which displacement stays <= {HOME_TOLERANCE_M} m to the end",
            "idle_pass": f"0 N: max displacement < {HOME_TOLERANCE_M} m, |pitch - balance| <= {BALANCE_BAND_DEG} deg "
                         f"and |yaw change| < {IDLE_YAW_LIMIT_DEG} deg for the whole run",
            "push_pass": f"no failure, return_time_s <= {CHECK_TIME_S} s, max |pitch| <= {FAILURE_PITCH_DEG} deg, "
                         f"max |yaw change| < {PUSH_YAW_LIMIT_DEG} deg",
            "balance_pitch": "pitch at which the whole-robot COM is above the wheel axle",
        },
        "plot_major_tick_s": 0.5,
        "plot_intervals": 10,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"RL_TIMESERIES_CSV={timeseries_path}")
    print(f"RL_SUMMARY_CSV={summary_path}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Print first, then hard-exit so a wedged simulation_app.close() cannot hide the error.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
