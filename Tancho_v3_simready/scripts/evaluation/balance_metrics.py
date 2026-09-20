#!/usr/bin/env python3
"""Compute controller-independent Tancho balance metrics from a common CSV schema."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED = {
    "time_s",
    "pitch_rad",
    "pitch_rate_rad_s",
    "base_x_m",
    "wheel_L_qd_rad_s",
    "wheel_R_qd_rad_s",
    "wheel_L_tau_Nm",
    "wheel_R_tau_Nm",
}


def load(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing common evaluation columns: {sorted(missing)}")
        return [{key: float(value) for key, value in row.items()} for row in reader]


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def settling_time(rows: list[dict[str, float]], tolerance_rad: float, dwell_s: float) -> float | None:
    times = [row["time_s"] for row in rows]
    pitch = [abs(row["pitch_rad"]) for row in rows]
    for index, start in enumerate(times):
        end = start + dwell_s
        selected = [pitch[j] for j in range(index, len(rows)) if times[j] <= end]
        if times[-1] >= end and selected and max(selected) <= tolerance_rad:
            return start
    return None


def evaluate(rows: list[dict[str, float]], torque_limit: float, failure_pitch_rad: float) -> dict:
    pitch = [row["pitch_rad"] for row in rows]
    tau = [max(abs(row["wheel_L_tau_Nm"]), abs(row["wheel_R_tau_Nm"])) for row in rows]
    wheel_speed = [max(abs(row["wheel_L_qd_rad_s"]), abs(row["wheel_R_qd_rad_s"])) for row in rows]
    settled = settling_time(rows, math.radians(1.0), 0.5)
    success = max(map(abs, pitch)) < failure_pitch_rad and settled is not None
    return {
        "success": success,
        "pitch_rms_rad": rms(pitch),
        "max_abs_pitch_rad": max(map(abs, pitch)),
        "settling_time_s": settled,
        "base_displacement_m": rows[-1]["base_x_m"] - rows[0]["base_x_m"],
        "wheel_torque_rms_nm": rms(tau),
        "wheel_torque_peak_nm": max(tau),
        "saturation_percent": 100.0 * sum(value >= 0.999 * torque_limit for value in tau) / len(tau),
        "max_wheel_speed_rad_s": max(wheel_speed),
        "duration_s": rows[-1]["time_s"] - rows[0]["time_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--torque-limit", type=float, default=0.45)
    parser.add_argument("--failure-pitch-deg", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(load(args.csv), args.torque_limit, math.radians(args.failure_pitch_deg))
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
