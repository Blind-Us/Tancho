#!/usr/bin/env python3
"""Plot trained-RL push-sweep summary on the same three axes as the LQR report.

(a) push force vs max pitch deviation  (x 0-60 N, y 0-16 deg, 15 deg failure line)
(b) push force vs settling time        (+-1 deg of the balance pitch held 0.5 s, from push onset;
                                        0 if the band was never left; y 0-5 s, 0.5 s ticks, recovered only)
(c) push force vs peak wheel torque    (y 0-0.47 N*m, 0.45 N*m limit line)

Usage: python plot_rl_push_lqr_axes.py <run_dir containing trained_rl_push_summary.csv>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

run = Path(sys.argv[1])
rows = list(csv.DictReader((run / "trained_rl_push_summary.csv").open()))
F = [float(r["push_force_N"]) for r in rows]
ok = [r["recovered"] == "1" for r in rows]
dev = [float(r["max_pitch_deviation_deg"]) for r in rows]
tau = [float(r["peak_wheel_torque_Nm"]) for r in rows]

TITLE = "Trained RL disturbance rejection: horizontal impulse sweep (50 ms push, applied to chassis)"


def panel_a(a):
    a.plot([f for f, o in zip(F, ok) if o], [min(d, 15) for d, o in zip(dev, ok) if o], "o-", label="recovered")
    a.plot([f for f, o in zip(F, ok) if not o], [15.3] * ok.count(False), "rx", label="failed")
    a.axhline(15, ls=":", c="gray", label="failure threshold (15 deg)")
    a.set(xlim=(0, 62), ylim=(0, 16), xlabel="push force (N)", ylabel="max pitch deviation (deg)",
          title="(a) Peak deviation vs. disturbance magnitude")
    a.xaxis.set_major_locator(MultipleLocator(10)); a.yaxis.set_major_locator(MultipleLocator(2))
    a.legend(fontsize=8); a.grid(alpha=.3)


def panel_b(b):
    st = [(f, float(r["settling_time_s"])) for f, r in zip(F, rows) if r["recovered"] == "1" and r["settling_time_s"]]
    if st:
        b.plot(*zip(*st), "o-", c="tab:green")
    b.set(xlim=(0, 62), ylim=(0, 5), xlabel="push force (N)", ylabel="settling time (s)\n(±1 deg of balance held 0.5 s; 0 = never left)",
          title="(b) Recovery time vs. disturbance magnitude")
    b.xaxis.set_major_locator(MultipleLocator(10)); b.yaxis.set_major_locator(MultipleLocator(0.5)); b.grid(alpha=.3)


def panel_c(c):
    c.plot(F, tau, "o-", c="tab:orange")
    c.axhline(0.45, ls=":", c="gray", label="torque limit (0.45 Nm)")
    c.set(xlim=(0, 62), ylim=(0, 0.47), xlabel="push force (N)", ylabel="peak wheel torque (Nm)",
          title="(c) Actuator saturation vs. disturbance magnitude")
    c.xaxis.set_major_locator(MultipleLocator(10)); c.yaxis.set_major_locator(MultipleLocator(0.1))
    c.legend(fontsize=8); c.grid(alpha=.3)


PANELS = (("a_max_pitch", panel_a), ("b_settling_time", panel_b), ("c_peak_torque", panel_c))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
fig.suptitle(TITLE)
for axis, (_, draw) in zip(ax, PANELS):
    draw(axis)
fig.tight_layout()
out = run / "trained_rl_push_lqr_axes.png"
fig.savefig(out, dpi=200)
print(out)

# The same three panels as separate figures.
for name, draw in PANELS:
    single, axis = plt.subplots(figsize=(5.4, 4.2))
    draw(axis)
    single.tight_layout()
    path = run / f"trained_rl_push_{name}.png"
    single.savefig(path, dpi=200)
    plt.close(single)
    print(path)
