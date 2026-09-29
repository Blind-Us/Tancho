#!/usr/bin/env python3
"""Run the frozen wheel-only RL policy through the LQR push-test envelope.

The fixed asset remains at the policy's training pose (thigh=-0.50,
calf=+0.87).  A 50 ms, body-level horizontal pulse is split over policy steps
by overlap with the pulse window (100 Hz: five full steps), so its total
impulse is exactly F*0.05 without changing the trained controller frequency.

Evaluation-only overrides (the training cfg files are not touched):
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
parser.add_argument("--obs-dim", type=int, default=None, help="Expected policy observation size (checked if given).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import tancho_v3_lab
import tancho_v3_lab.tasks  # noqa: F401


FORCES_N = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34, 36, 38, 39, 40, 42, 45, 50, 60)
PULSE_DURATION_S = 0.05
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


def pitch_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))


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


def settling_time(rows: list[dict[str, float]]) -> float | None:
    """First time after the pulse ends that |pitch - initial| stays within 1 deg for >= 0.5 s."""
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


def final_settling_time(rows: list[dict[str, float]]) -> float | None:
    """Last entry into the +-1 deg band after which the pitch never leaves it (>= 0.5 s of data left)."""
    last_out = max((r["time_s"] for r in rows if abs(r["pitch_dev_deg"]) > SETTLE_BAND_DEG), default=None)
    if last_out is None:
        return PULSE_DURATION_S
    entry = next((r["time_s"] for r in rows if r["time_s"] > last_out), None)
    if entry is None or rows[-1]["time_s"] < entry + SETTLE_DWELL_S - 1.0e-9:
        return None
    return max(entry, PULSE_DURATION_S)


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
        fields = {
            "pitch_deg": pitch[:, None],
            "pitch_rate_rad_s": robot.data.root_ang_vel_b[:, 1:2],
            "base_x_m": pos[:, 0:1],
            "base_vx_m_s": robot.data.root_lin_vel_w[:, 0:1],
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
    pitch0_deg = torch.rad2deg(pitch_wxyz(robot.data.root_link_quat_w)).cpu().tolist()
    pitch0_t = torch.tensor(pitch0_deg, device=core.device)
    record(0.0, zero_force)
    for step in range(steps):
        # Scale each policy step by its overlap with the pulse window so the total
        # impulse is exactly F*0.05 N*s at the policy's own rate.
        fraction = pulse_fraction(step, dt)
        applied = force_levels * fraction
        if step == 0 or fraction != pulse_fraction(step - 1, dt):
            forces = torch.zeros((len(FORCES_N), 1, 3), device=core.device)
            forces[:, 0, 0] = applied
            robot.set_external_force_and_torque(
                forces=forces,
                torques=torch.zeros_like(forces),
                body_ids=body_ids,
                is_global=True,
            )
        with torch.inference_mode():
            # The exported TorchScript policy takes the flat policy tensor, not the wrapper's TensorDict.
            actions = policy(obs["policy"])
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
    output_dir = args.output_dir.resolve() / f"trained_rl_lqr_axes_{stamp}"
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
        "controller", "push_force_N", "pulse_duration_s", "recovered", "max_pitch_deviation_deg",
        "settling_time_s", "peak_wheel_torque_Nm", "wheel_torque_limit_Nm",
        "torque_saturated", "first_saturation_time_s", "saturation_duration_s",
        "saturation_before_failure_s", "failure_time_s", "final_settling_time_s",
        "ground_contact", "first_ground_contact_time_s", "ground_contact_shape",
        "pitch_dev_at_first_ground_contact_deg", "ground_contact_duration_s",
        "max_wheel_speed_rad_s", "max_abs_slip_m_s", "first_slip_time_s",
        "first_wheel_liftoff_time_s", "wheel_liftoff_duration_s", "initial_pitch_deg",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for i, (level, case_rows, did_fail) in enumerate(zip(FORCES_N, rows_by_case, failed.tolist())):
            settle = None if did_fail else settling_time(case_rows)
            tau = [max(abs(r["wheel_L_tau_Nm"]), abs(r["wheel_R_tau_Nm"])) for r in case_rows]
            sat = [t >= TORQUE_LIMIT_NM - SAT_TOL_NM for t in tau]
            contact_rows = [r for r in case_rows if r["ground_contact"] > 0.5]
            fail_t = next((r["time_s"] for r in case_rows if r["failed"] > 0.5), None)
            final_settle = None if did_fail else final_settling_time(case_rows)
            slip_t = next((r["time_s"] for r in case_rows
                           if max(abs(r["wheel_L_slip_m_s"]), abs(r["wheel_R_slip_m_s"])) > SLIP_THRESHOLD_M_S), "")
            lift = [min(r["wheel_L_normal_force_N"], r["wheel_R_normal_force_N"]) < WHEEL_LIFTOFF_N for r in case_rows]
            writer.writerow(
                {
                    "controller": "trained_rl",
                    "push_force_N": level,
                    "pulse_duration_s": PULSE_DURATION_S,
                    "recovered": int((not did_fail) and settle is not None),
                    "max_pitch_deviation_deg": max(abs(row["pitch_dev_deg"]) for row in case_rows),
                    "settling_time_s": "" if settle is None else settle,
                    "peak_wheel_torque_Nm": max(tau),
                    "wheel_torque_limit_Nm": TORQUE_LIMIT_NM,
                    "torque_saturated": int(any(sat)),
                    "first_saturation_time_s": next((r["time_s"] for r, s in zip(case_rows, sat) if s), ""),
                    "saturation_duration_s": round(sum(sat[1:]) * dt, 4),
                    "saturation_before_failure_s": round(dt * sum(
                        1 for r, s in zip(case_rows[1:], sat[1:]) if s and (fail_t is None or r["time_s"] <= fail_t)), 4),
                    "failure_time_s": "" if fail_t is None else fail_t,
                    "final_settling_time_s": "" if final_settle is None else final_settle,
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
                }
            )
    metadata = {
        "controller": "trained_rl",
        "checkpoint": str(checkpoint),
        "tancho_v3_lab_module": tancho_v3_lab.__file__,
        "urdf": str(urdf_path),
        "training_pose_rad": {"thigh": -0.50, "calf": 0.87},
        "push_forces_N": list(FORCES_N),
        "nominal_pulse_duration_s": PULSE_DURATION_S,
        "policy_dt_s": dt,
        "pulse_discretization": {
            "per_step_force_fraction": [pulse_fraction(k, dt) for k in range(math.ceil(PULSE_DURATION_S / dt) + 1)],
            "note": "fraction = overlap of each policy step with the pulse window; exact F*0.05 impulse",
        },
        "evaluation_overrides": {
            "terminations": "all removed (no auto reset)",
            "failure": f"|pitch - initial| >= {FAILURE_PITCH_DEG} deg only",
            "ground_contact": f"base_link_root net contact force > {GROUND_CONTACT_THRESHOLD_N} N, logged only",
            "curriculum": None,
            "push_robot_event": None,
        },
        "base_collision_shapes": shape_names,
        "duration_s": args.duration,
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
