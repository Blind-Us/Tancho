#!/usr/bin/env python3
"""Combine independent Gate-C dt runs into one reproducible PASS/FAIL report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path: Path) -> dict[tuple[str, float], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["wheel"], float(row["torque_Nm"])): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--half", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-fit-relative", type=float, default=0.01)
    parser.add_argument("--max-dt-relative", type=float, default=0.01)
    args = parser.parse_args()

    base = _load(args.base)
    half = _load(args.half)
    keys_match = set(base) == set(half) and bool(base)
    cases = []
    for key in sorted(set(base) & set(half)):
        a, b = base[key], half[key]
        pred = float(a["qdd_pred_target_rad_s2"])
        measured_a = float(a["qdd_meas_target_rad_s2"])
        measured_b = float(b["qdd_meas_target_rad_s2"])
        dt_relative = abs(measured_b - measured_a) / max(abs(pred), 1.0e-12)
        direct_pass = a["status"] == "PASS" and b["status"] == "PASS"
        fit_relative = max(float(a["qdd_fit_relative"]), float(b["qdd_fit_relative"]))
        case_pass = direct_pass and fit_relative <= args.max_fit_relative and dt_relative <= args.max_dt_relative
        cases.append(
            {
                "wheel": key[0],
                "torque_Nm": key[1],
                "qdd_pred_rad_s2": pred,
                "qdd_measured_base_dt_rad_s2": measured_a,
                "qdd_measured_half_dt_rad_s2": measured_b,
                "max_fit_relative": fit_relative,
                "dt_convergence_relative": dt_relative,
                "status": "PASS" if case_pass else "FAIL",
            }
        )

    passed = keys_match and len(cases) == 12 and all(case["status"] == "PASS" for case in cases)
    report = {
        "gate": "GATE_C",
        "status": "PASS" if passed else "FAIL",
        "definition": "gravity off, airborne/no contact, zero initial velocity, direct known wheel torque",
        "inputs": {"base_dt_csv": str(args.base.resolve()), "half_dt_csv": str(args.half.resolve())},
        "thresholds": {
            "max_qdd_fit_relative": args.max_fit_relative,
            "max_dt_convergence_relative": args.max_dt_relative,
            "required_case_count": 12,
        },
        "checks": {
            "case_keys_match": keys_match,
            "case_count": len(cases),
            "all_individual_runs_pass": all(
                base[k]["status"] == "PASS" and half[k]["status"] == "PASS" for k in set(base) & set(half)
            ),
            "max_observed_fit_relative": max((c["max_fit_relative"] for c in cases), default=None),
            "max_observed_dt_convergence_relative": max(
                (c["dt_convergence_relative"] for c in cases), default=None
            ),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"GATE_C_{report['status']}")
    print(json.dumps(report["checks"], indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
