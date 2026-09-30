#!/usr/bin/env python3
"""Pass/fail table of push sweeps (evaluate_trained_rl_push.py outputs) for several models.

Usage:
  python compare_push_models.py --out table \
      --model "LABEL=SWEEP_DIR" [--model ...] \
      [--ood "LABEL=DIR"] [--ood ...]

SWEEP_DIR is a run with the 0-60 N, 50 ms sweep (0 N = idle case);
OOD dirs are out-of-distribution runs (e.g. 200 ms or 30 deg pushes) of the same model.
Writes <out>.csv and <out>.md.
"""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", action="append", required=True)
parser.add_argument("--ood", action="append", default=[])
parser.add_argument("--ood-forces", default="20,38")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()


def load(run: str) -> tuple[list[dict], dict]:
    path = Path(run)
    rows = list(csv.DictReader((path / "trained_rl_push_summary.csv").open()))
    meta = json.loads((path / "metadata.json").read_text())
    return rows, meta


def num(value: str) -> float | None:
    return None if value in ("", None) else float(value)


def fmt(value, digits=2, scale=1.0) -> str:
    return "—" if value is None else f"{value * scale:.{digits}f}"


def by_force(rows: list[dict], force: float) -> dict | None:
    return next((r for r in rows if abs(float(r["push_force_N"]) - force) < 1e-6), None)


table = []
for spec in args.model:
    label, run = spec.split("=", 1)
    rows, meta = load(run)
    idle = by_force(rows, 0.0)
    pushes = [r for r in rows if float(r["push_force_N"]) > 0]
    falls = [float(r["push_force_N"]) for r in pushes if r["recovered"] != "1" and r["failure_time_s"]]
    r38 = by_force(rows, 38.0)
    entry = {
        "model": label,
        "rate_Hz": round(1.0 / meta["policy_dt_s"]),
        "idle_max_disp_cm": fmt(num(idle["max_displacement_m"]), 1, 100) if idle else "—",
        "idle_max_pitch_from_balance_deg": fmt(num(idle["max_abs_pitch_from_balance_deg"])) if idle else "—",
        "idle_yaw_drift_deg": fmt(num(idle["max_abs_yaw_deg"]), 1) if idle else "—",
        "idle_pass": idle["idle_pass"] if idle else "—",
        "sweep_pass": f"{sum(r['push_pass'] == '1' for r in pushes)}/{len(pushes)}",
        "sweep_no_fall": f"{len(pushes) - len(falls)}/{len(pushes)}",
        "first_fall_N": fmt(min(falls), 0) if falls else "none",
        "sweep_returned_5s": f"{sum(r['returned_within_5s'] == '1' for r in pushes)}/{len(pushes)}",
        "sweep_max_abs_pitch_deg": fmt(max(num(r["max_abs_pitch_deg"]) for r in pushes if r["recovered"] == "1")
                                       if any(r["recovered"] == "1" for r in pushes) else None, 1),
        "sweep_max_yaw_deg": fmt(max(num(r["max_abs_yaw_deg"]) for r in pushes), 1),
    }
    if r38:
        entry.update({
            "38N_max_abs_pitch_deg": fmt(num(r38["max_abs_pitch_deg"]), 1),
            "38N_max_disp_cm": fmt(num(r38["max_displacement_m"]), 1, 100),
            "38N_disp_5s_cm": fmt(num(r38["displacement_at_5s_m"]), 1, 100),
            "38N_return_s": fmt(num(r38["return_time_s"])),
            "38N_pitch_settle_s": fmt(num(r38["settling_time_s"])),
            "38N_pose_pos_settle_s": fmt(num(r38["pose_position_settling_time_s"])),
            "38N_yaw_deg": fmt(num(r38["max_abs_yaw_deg"]), 1),
            "38N_peak_torque_Nm": fmt(num(r38["peak_wheel_torque_Nm"]), 3),
            "38N_pass": r38["push_pass"],
        })
    for ood in args.ood:
        ood_label, ood_run = ood.split("=", 1)
        if ood_label.split(":")[0] != label:
            continue
        ood_rows, ood_meta = load(ood_run)
        tag = f"{ood_meta['nominal_pulse_duration_s'] * 1000:.0f}ms/{ood_meta.get('push_angle_deg', 0):.0f}deg"
        for force in (float(v) for v in args.ood_forces.split(",")):
            r = by_force(ood_rows, force)
            if r is None:
                continue
            state = "fall" if r["recovered"] != "1" and r["failure_time_s"] else (
                "pass" if r["push_pass"] == "1" else "no-pass")
            entry[f"{tag}_{force:g}N"] = (f"{state}; |pitch| {fmt(num(r['max_abs_pitch_deg']), 1)}°, "
                                         f"disp {fmt(num(r['max_displacement_m']), 0, 100)} cm, "
                                         f"return {fmt(num(r['return_time_s']))} s, yaw {fmt(num(r['max_abs_yaw_deg']), 1)}°")
    table.append(entry)

columns = []
for entry in table:
    columns.extend(k for k in entry if k not in columns)
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(table)
# Transposed markdown: one row per metric, one column per model.
lines = ["| metric | " + " | ".join(e["model"] for e in table) + " |",
         "|---|" + "---|" * len(table)]
for key in columns[1:]:
    lines.append(f"| {key} | " + " | ".join(str(e.get(key, "—")) for e in table) + " |")
args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
