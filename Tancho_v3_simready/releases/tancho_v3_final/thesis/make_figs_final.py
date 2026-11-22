#!/usr/bin/env python3
"""Report figures for the final RL policy (Isaac suite, nominal and noise + latency).

    python scripts/wheel_only/thesis/make_figs_final.py <suite dir> <suite_nl dir> <out dir>
"""
import collections, csv, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.6, "savefig.dpi": 200})
B, O, G = "#2a78d6", "#eb6834", "#8a8984"
SD, NL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def load(d):
    R = collections.defaultdict(list)
    for r in csv.DictReader(open(d + "/timeseries.csv")):
        R[r["case"]].append({k: (float(v) if k != "case" else v) for k, v in r.items()})
    return R, json.load(open(d + "/summary.json"))


R, S = load(SD); Rn, Sn = load(NL)
col = lambda R, c, k: np.array([r[k] for r in R[c]])

# Fig 1: +/-10 deg initial pitch, pitch and displacement
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for c, color in (("pitch+10", B), ("pitch-10", O)):
    ax[0].plot(col(R, c, "time_s"), col(R, c, "pitch_dev_deg"), color=color, label=c.replace("pitch", "初始 ") + "°")
    ax[1].plot(col(R, c, "time_s"), col(R, c, "disp_m") * 100, color=color)
ax[0].axhline(S["ref_pitch_dev_deg"], color=G, lw=0.8, ls="--"); ax[0].set(xlabel="時間 (s)", ylabel="俯仰偏差 (°)", xlim=(0, 5)); ax[0].legend(frameon=False)
ax[1].set(xlabel="時間 (s)", ylabel="水平位移 (cm)", xlim=(0, 5))
fig.tight_layout(); fig.savefig(OUT + "/fig_final_pitch10.png")

# Fig 2: push sweep, max deviation and pass / fail, nominal vs noise + latency
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for SS, color, lab in ((S, B, "標稱"), (Sn, O, "雜訊 + 20 ms 延遲")):
    pts = sorted((int(n[4:-1]), v) for n, v in SS["summary"].items() if n.startswith("push"))
    F = [p[0] for p in pts]; dev = [min(p[1]["max_dev_from_ref"], 20) for p in pts]; fail = [p[1]["fail"] is not None for p in pts]
    ax.plot(F, dev, color=color, label=lab)
    ax.scatter([f for f, x in zip(F, fail) if x], [d for d, x in zip(dev, fail) if x], color=color, marker="x", zorder=3)
ax.axhline(15, color=G, ls="--", lw=0.8); ax.axvline(38.5, color=G, ls=":", lw=0.8)
ax.text(39, 2, "LQR 分界 38/39 N", color=G, fontsize=8)
ax.set(xlabel="50 ms 推力 (N)", ylabel="最大俯仰偏差 (°，上限 20)", xlim=(0, 100)); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "/fig_final_push.png")

# Fig 3: command tracking (nominal)
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for a, c, k, kc, lab in ((ax[0], "vx0.3", "vx_body_m_s", "cmd_vx", "前進速度 (m/s)"), (ax[1], "yaw1.0", "yaw_rate_rad_s", "cmd_wz", "偏航角速度 (rad/s)"),
                         (ax[2], "vx0.2_yaw0.5", "vx_body_m_s", "cmd_vx", "前進速度 (m/s)")):
    t = col(R, c, "time_s"); a.plot(t, col(R, c, kc), color=G, ls="--", label="命令"); a.plot(t, col(R, c, k), color=B, label="實際")
    if c == "vx0.2_yaw0.5":
        a.plot(t, col(R, c, "cmd_wz"), color=G, ls=":"); a.plot(t, col(R, c, "yaw_rate_rad_s"), color=O, label="偏航 (rad/s)")
    a.set(xlabel="時間 (s)", ylabel=lab, title=c); a.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "/fig_final_cmd.png")
print("ok")
