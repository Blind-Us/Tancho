"""Tancho V3 fixed-gravity, zero-action wheel-support diagnostic.

This test is deliberately separate from the reset gravity ramp.  Each case
starts from the same nominal on-wheel reset, holds one constant effective
gravity fraction, and records whether wheel normal force and leg posture can
carry that load without policy input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tancho fixed-gravity support test.")
parser.add_argument("--task", type=str, default="TanchoV3-Flat-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--duration_s", type=float, default=0.20)
parser.add_argument("--fractions", type=float, nargs="+", default=(0.10, 0.25, 0.50, 1.00))
parser.add_argument(
    "--fixed_asset",
    action="store_true",
    help="Allow the fixed-leg asset, which intentionally has no thigh/calf DOFs.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("logs/diagnostics/fixed_gravity_support.csv"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import tancho_v3_lab.tasks  # noqa: F401,E402
from tancho_v3_lab.tasks.direct.tancho_v3 import custom_events as ce


LEG_JOINTS = ("joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R")


def _disable_confounds(cfg: ManagerBasedRLEnvCfg) -> None:
    cfg.scene.num_envs = 1
    if hasattr(cfg.sim, "device"):
        cfg.sim.device = args_cli.device
    if getattr(cfg, "events", None) is not None:
        if hasattr(cfg.events, "reset_gravity_ramp"):
            cfg.events.reset_gravity_ramp = None
        if hasattr(cfg.events, "push_robot"):
            cfg.events.push_robot = None
    if getattr(cfg, "terminations", None) is not None and hasattr(cfg.terminations, "base_contact"):
        cfg.terminations.base_contact = None
    if getattr(cfg, "curriculum", None) is not None and hasattr(cfg.curriculum, "enable_push"):
        cfg.curriculum.enable_push = None
    if args_cli.fixed_asset and getattr(cfg, "rewards", None) is not None and hasattr(cfg.rewards, "leg_torque"):
        # The fixed articulation intentionally exposes no thigh/calf joints.
        # Disable this inherited full-body reward only in the diagnostic copy.
        cfg.rewards.leg_torque = None
    cfg.episode_length_s = max(float(cfg.episode_length_s), args_cli.duration_s + 1.0)


def _force_z(sensor, body_ids: list[int]) -> float:
    return float(sensor.data.net_forces_w[0, body_ids, 2].sum().item())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    del agent_cfg
    if args_cli.duration_s <= 0.0:
        raise ValueError("--duration_s must be positive")
    if any(f <= 0.0 or f > 1.0 for f in args_cli.fractions):
        raise ValueError("all gravity fractions must be in (0, 1]")
    _disable_confounds(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    sensor = base.scene["contact_forces"]
    env_ids = torch.tensor([0], device=base.device, dtype=torch.long)
    wheel_l_ids, _ = sensor.find_bodies("wheel_L")
    wheel_r_ids, _ = sensor.find_bodies("wheel_R")
    if args_cli.fixed_asset:
        leg_ids, leg_names = [], []
    else:
        leg_ids, leg_names = robot.find_joints(list(LEG_JOINTS), preserve_order=True)
    if len(wheel_l_ids) != 1 or len(wheel_r_ids) != 1:
        raise RuntimeError(f"wheel contact bodies not unique: L={wheel_l_ids}, R={wheel_r_ids}")
    if not args_cli.fixed_asset and len(leg_ids) != len(LEG_JOINTS):
        raise RuntimeError(f"leg joints unresolved: {leg_names}")
    if args_cli.fixed_asset and len(leg_ids) != 0:
        raise RuntimeError(f"fixed asset unexpectedly exposes leg joints: {leg_names}")

    total_mass = float(robot.data.default_mass[0].sum().item())
    runtime_effort_limits = {
        name: float(robot.data.joint_effort_limits[0, joint_id].item())
        for joint_id, name in enumerate(robot.joint_names)
    }
    dt = float(base.step_dt)
    steps = max(1, round(args_cli.duration_s / dt))
    rows: list[dict[str, float | int | str]] = []
    cases: list[dict[str, float | int | str | bool]] = []

    try:
        for fraction in args_cli.fractions:
            env.reset()
            # reset_tancho_on_wheels arms 100% compensation.  Replace it with
            # a constant compensation so effective gravity is exactly the
            # requested fraction for every measured step.
            ce._set_gravity_compensation_fraction(base, robot, env_ids, 1.0 - fraction)
            zero_action = torch.zeros((1, base.action_manager.total_action_dim), device=base.device)
            q0 = robot.data.joint_pos[0, leg_ids].clone() if leg_ids else None
            case_rows = []
            for step in range(steps + 1):
                q = robot.data.joint_pos[0, leg_ids] if leg_ids else None
                qt = robot.data.joint_pos_target[0, leg_ids] if leg_ids else None
                qd = robot.data.joint_vel[0, leg_ids] if leg_ids else None
                tau = robot.data.applied_torque[0, leg_ids] if leg_ids else None
                fz_l = _force_z(sensor, wheel_l_ids)
                fz_r = _force_z(sensor, wheel_r_ids)
                demand = total_mass * abs(float(base.sim.cfg.gravity[2])) * fraction
                row: dict[str, float | int | str] = {
                    "gravity_fraction": fraction,
                    "step": step,
                    "time_s": step * dt,
                    "total_mass_kg": total_mass,
                    "weight_demand_N": demand,
                    "wheel_L_fz_N": fz_l,
                    "wheel_R_fz_N": fz_r,
                    "support_ratio": (fz_l + fz_r) / demand,
                    "root_z_m": float(robot.data.root_pos_w[0, 2].item()),
                    "root_vz_m_s": float(robot.data.root_lin_vel_w[0, 2].item()),
                }
                for index, name in enumerate(leg_names):
                    stem = name.removeprefix("joint_")
                    row[f"{stem}_q_rad"] = float(q[index].item())
                    row[f"{stem}_target_q_rad"] = float(qt[index].item())
                    row[f"{stem}_qd_rad_s"] = float(qd[index].item())
                    row[f"{stem}_applied_tau_Nm"] = float(tau[index].item())
                rows.append(row)
                case_rows.append(row)
                if step < steps:
                    env.step(zero_action)

            tail = case_rows[-min(5, len(case_rows)):]
            mean_support = sum(float(r["support_ratio"]) for r in tail) / len(tail)
            final = case_rows[-1]
            final_q = robot.data.joint_pos[0, leg_ids] if leg_ids else None
            max_q_error = float(torch.max(torch.abs(final_q - q0)).item()) if leg_ids else 0.0
            max_tau = (
                max(
                    abs(float(r[f"{name.removeprefix('joint_')}_applied_tau_Nm"]))
                    for r in case_rows
                    for name in leg_names
                )
                if leg_names
                else 0.0
            )
            passed = (
                0.90 <= mean_support <= 1.10
                and abs(float(final["root_vz_m_s"])) <= 0.05
                and max_q_error <= 0.05
                and max_tau < 12.49
            )
            cases.append(
                {
                    "gravity_fraction": fraction,
                    "mean_tail_support_ratio": mean_support,
                    "final_root_vz_m_s": float(final["root_vz_m_s"]),
                    "max_leg_q_error_rad": max_q_error,
                    "max_leg_torque_Nm": max_tau,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    finally:
        env.close()

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args_cli.output.with_suffix(".summary.json")
    summary = {
        "test": "fixed_gravity_zero_action_support",
        "status": "PASS" if all(case["status"] == "PASS" for case in cases) else "FAIL",
        "dt_s": dt,
        "duration_s": args_cli.duration_s,
        "total_mass_kg": total_mass,
        "runtime_joint_effort_limits_Nm": runtime_effort_limits,
        "cases": cases,
        "output_csv": str(args_cli.output.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"FIXED_GRAVITY_SUPPORT_{summary['status']}")
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
