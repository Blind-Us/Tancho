#!/usr/bin/env python3
"""Report table: LQR (chapter 4, MuJoCo) vs final RL policy (Isaac suite + acceptance runs) vs hardware (blank).

    python scripts/wheel_only/thesis/compare_table.py --suite <suite dir> --suite-nl <suite dir, noise+latency> \
        --stand <stand_summary dir> --push <stand_summary dir> --walk <evaluate_walk dir> --mj <dir with MuJoCo summaries> --out table.md
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
for k in ("suite", "suite_nl", "stand", "push", "walk", "mj", "out"):
    ap.add_argument("--" + k.replace("_", "-"), required=k in ("suite", "out"))
a = ap.parse_args()
j = lambda p: json.load(open(p)) if p and os.path.exists(p) else None
S = j(os.path.join(a.suite, "summary.json")); s = S["summary"]
N = j(os.path.join(a.suite_nl, "summary.json")) if a.suite_nl else None


def pitch(S):
    s = S["summary"]; ok = [n for n in s if n.startswith("pitch") and n != "pitch+0"]
    fails = [n for n in ok if s[n]["fail"] is not None]
    p10, m10 = s["pitch+10"], s["pitch-10"]
    r = lambda x: f"恢復 {x} s" if x is not None else "未收斂到 ±1° 內（持續小幅晃動）"
    return (f"{len(ok) - len(fails)}/{len(ok)} 未失敗；+10° {r(p10['rec'])}、最大位移 {p10['max_disp'] * 100:.1f} cm；"
            f"−10° {r(m10['rec'])}、最大位移 {m10['max_disp'] * 100:.1f} cm")


def cmd(S, n):
    c = S["summary"][n]
    if c["fail"] is not None:
        return f"{c['fail']:.2f} s 失敗"
    tgt = c["cmd"]
    return (f"未失敗；實際 vx {c['mean_vx']:.2f} m/s（命令 {tgt[0]}）、偏航 {c['mean_wz']:.2f} rad/s（命令 {tgt[1]}）；"
            f"最大俯仰偏差 {c['max_dev_from_ref']:.2f}°")


def push(S):
    s = S["summary"]; b = S["push_ok_up_to_N"]; f = S["first_push_fail_N"]
    p38 = s.get("push38N")
    t = f"{b} N 通過／{f} N 失敗" if f else f"至 {b} N 皆通過"
    if p38 and p38["fail"] is None:
        t += f"；38 N 最大偏差 {p38['max_dev_from_ref']:.2f}°、" + (f"恢復 {p38['rec']} s" if p38["rec"] is not None else "未收斂到 ±1° 內")
    return t


def rate(S):
    s = S["summary"]; out = []
    for p in (5, 10):
        ok = [n for n in s if n.startswith("rate") and n.endswith(f"_p{p}") and s[n]["fail"] is None]
        lo = min((float(n[4:n.index("Hz")]) for n in ok), default=None)
        out.append(f"+{p}°：最低 {lo:g} Hz 未失敗" if lo else f"+{p}°：皆失敗")
    return "；".join(out)


rows = [["項目", "LQR（MuJoCo，固定腿部姿態）", "RL 最終版（Isaac Sim，標稱）", "RL 最終版（雜訊 + 20 ms 延遲）", "實機部署"]]
rows.append(["模擬與控制頻率", "物理 1 kHz，控制 1 kHz", "物理 200 Hz，策略 50 Hz", "同左，馬達延遲 20 ms", "（待填）"])
rows.append(["初始俯仰 ±1° 至 ±10°", "十案例均通過；±10° 恢復 0.176 s，最大位移 16.9 cm", pitch(S), pitch(N) if N else "—", "（待填）"])
rows.append(["前進 0.3 m/s", "最大傾角 1.52°，峰值力矩 0.207 N·m", cmd(S, "vx0.3"), cmd(N, "vx0.3") if N else "—", "（待填）"])
rows.append(["偏航 1.0 rad/s", "最大角速度 1.002 rad/s，總轉角 3.000 rad", cmd(S, "yaw1.0"), cmd(N, "yaw1.0") if N else "—", "（待填）"])
rows.append(["0.2 m/s + 0.5 rad/s", "最大俯仰偏差 1.01°", cmd(S, "vx0.2_yaw0.5"), cmd(N, "vx0.2_yaw0.5") if N else "—", "（待填）"])
rows.append(["50 ms 推力分界", "38 N 通過／39 N 失敗；38 N 最大偏差 6.82°、恢復約 4.00 s", push(S), push(N) if N else "—", "（待填）"])
rows.append(["降低控制頻率", "200 Hz 穩定；142.9 Hz 失敗", rate(S), rate(N) if N else "—", "（待填）"])
st, pu = j(glob.glob(os.path.join(a.stand or "", "*.json"))[0]) if a.stand else None, j(glob.glob(os.path.join(a.push or "", "*.json"))[0]) if a.push else None
if st:
    rows.append(["零命令站立 60 s", "未單獨測試", f"不倒；最大漂移 {st['pre_push_max_drift_m'] * 100:.1f} cm", "見部署驗收", "（待填）"])
if pu:
    rows.append(["站立中速度擾動 0.5 m/s", "未單獨測試", "不倒" + ("，恢復" if pu["pass_recovered"] else ""), "見部署驗收", "（待填）"])
md = ["| " + " | ".join(r) + " |" for r in rows]
md.insert(1, "|" + "---|" * len(rows[0]))
open(a.out, "w").write("\n".join(md) + "\n")
print("\n".join(md))
