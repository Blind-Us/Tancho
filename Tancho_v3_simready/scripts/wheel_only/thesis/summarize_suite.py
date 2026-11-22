#!/usr/bin/env python3
"""Summarise an evaluate_lqr_suite.py run (report sections 4.6.2-4.6.5) into summary.json.

Same definitions as outputs/rl_vs_lqr_isaac_20261102/make_figs.py: recovery = time after which
the pitch stays within +/-1 deg of the policy's own steady pitch (push0N, 8-10 s); plus, for the
command cases, the mean achieved forward speed / yaw rate over the command window.

    python scripts/wheel_only/thesis/summarize_suite.py <suite output dir>
"""
import collections, csv, json, statistics as st, sys
import numpy as np

D = sys.argv[1].rstrip("/") + "/"
R = collections.defaultdict(list)
for r in csv.DictReader(open(D + "timeseries.csv")):
    R[r["case"]].append({k: (float(v) if k != "case" else v) for k, v in r.items()})
meta = json.load(open(D + "metadata.json"))
arr = lambda c, k, T=None: np.array([r[k] for r in R[c] if T is None or r["time_s"] <= T + 1e-9])
REF = st.mean(r["pitch_dev_deg"] for r in R["push0N"] if 8 <= r["time_s"] <= 10)


def rec(c, t0, T):
    t = arr(c, "time_s", T); p = arr(c, "pitch_dev_deg", T); m = t >= t0; t, p = t[m], p[m]
    bad = np.where(np.abs(p - REF) > 1)[0]
    if meta["failed_at"][c] is not None:
        return None
    if not len(bad):
        return 0.0
    return None if bad[-1] == len(t) - 1 else round(float(t[bad[-1] + 1] - t0), 2)


def summ(c, t0, T):
    t = arr(c, "time_s", T); m = t >= t0; p = arr(c, "pitch_dev_deg", T)[m]
    tau = np.maximum(np.abs(arr(c, "tau_L_Nm", T)), np.abs(arr(c, "tau_R_Nm", T)))[m]
    w = t[m] <= t0 + 1.0
    return dict(case=c, fail=meta["failed_at"][c], contact=meta["contact_at"][c], max_dev=float(np.abs(p).max()),
                max_dev_from_ref=float(np.abs(p - REF).max()), rec=rec(c, t0, T), max_x=float(np.abs(arr(c, "x_m", T)).max()),
                max_disp=float(arr(c, "disp_m", T).max()), peak_tau=float(tau.max()), sat1s=float((tau[w] > 0.449).mean() * 100))


S = {}
for f in meta["cases"]:
    n = f["name"]
    if f["kind"] == "pitch":
        S[n] = summ(n, 0, 5.0)
    elif f["kind"] == "push":
        S[n] = summ(n, 2.0, 10.0)
    elif f["kind"] == "rate":
        S[n] = summ(n, 0, 10.0)
        if S[n]["fail"] is None:
            p = arr(n, "pitch_dev_deg", 10.0)[arr(n, "time_s", 10.0) >= 5]; S[n]["osc_pp_last5s"] = float(p.max() - p.min())
    elif f["kind"] == "cmd":
        S[n] = summ(n, 0, 8.0)
        t = arr(n, "time_s"); win = (t >= f["t0"] + 1.0) & (t < f["t1"])
        S[n]["mean_vx"] = float(arr(n, "vx_body_m_s")[win].mean()); S[n]["mean_wz"] = float(arr(n, "yaw_rate_rad_s")[win].mean())
        S[n]["cmd"] = [f["vx"], f["wz"]]
pushes = sorted((int(n[4:-1]), S[n]["fail"]) for n in S if n.startswith("push"))
ok_up_to = 0
for F, fail in pushes:
    if fail is not None:
        break
    ok_up_to = F
out = dict(ref_pitch_dev_deg=REF, balance_pitch_deg=meta["balance_pitch_deg"], push_ok_up_to_N=ok_up_to,
           first_push_fail_N=next((F for F, fail in pushes if fail is not None), None), summary=S)
json.dump(out, open(D + "summary.json", "w"), indent=1, ensure_ascii=False)
print("push ok up to", ok_up_to, "N; first fail", out["first_push_fail_N"])
for n in ("pitch+10", "pitch-10", "vx0.3", "yaw1.0", "vx0.2_yaw0.5"):
    s = S.get(n)
    if s:
        print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in ("fail", "max_dev", "rec", "max_disp", "mean_vx", "mean_wz")})
print("rate", {n: ("FAIL" if s["fail"] is not None else round(s["osc_pp_last5s"], 2)) for n, s in S.items() if n.startswith("rate")})
