#!/usr/bin/env python3
"""Forward displacement vs time for one push force, several models on one plot.

x 0-5 s with 0.5 s ticks, +-5 cm home band.  Displacement is the signed
distance of the root link from its t = 0 position along the initial heading.

Usage: python plot_push_displacement.py --force 38 --out fig.png LABEL=RUN_DIR [LABEL=RUN_DIR ...]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", help="LABEL=RUN_DIR (directory with trained_rl_push_timeseries.csv)")
parser.add_argument("--force", type=float, default=38.0)
parser.add_argument("--t-max", type=float, default=5.0)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for spec in args.runs:
    label, run = spec.split("=", 1)
    rows = [r for r in csv.DictReader((Path(run) / "trained_rl_push_timeseries.csv").open())
            if abs(float(r["push_force_N"]) - args.force) < 1e-6 and float(r["time_s"]) <= args.t_max + 1e-9]
    if not rows:
        raise SystemExit(f"{run}: no rows for {args.force} N")
    ax.plot([float(r["time_s"]) for r in rows], [100.0 * float(r["forward_displacement_m"]) for r in rows], label=label)
ax.axhspan(-5.0, 5.0, color="tab:green", alpha=0.12, label="±5 cm (home)")
ax.axhline(0.0, c="gray", lw=0.8)
ax.set(xlim=(0, args.t_max), xlabel="time since push onset (s)", ylabel="forward displacement (cm)",
       title=f"{args.force:g} N, {rows[0]['pulse_duration_s']} s push: displacement from start position")
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=200)
print(args.out)
