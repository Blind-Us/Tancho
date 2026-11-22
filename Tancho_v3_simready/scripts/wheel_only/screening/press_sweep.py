#!/usr/bin/env python3
"""Press-combination stress sweep of one exported policy in MuJoCo (deployment conditions).

Generates operator sequences (single LT / RT / crouch presses, L-R alternations, crouch then
lift, double taps, long holds) x gaps 0.15-1.5 s x hold times x latency 10/20 ms x noise
seeds, runs ``evaluate_stand_mujoco.py`` on each in parallel and reports the fall rate.

    python scripts/wheel_only/screening/press_sweep.py --checkpoint <policy.pt> --out <dir> [--jobs 12]
"""
import argparse, itertools, json, os, random, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--jobs", type=int, default=12)
ap.add_argument("--n", type=int, default=240, help="number of sequences (random sample of the grid)")
ap.add_argument("--push", type=float, default=0.0, help="also kick the body by this speed (m/s, forward or back) at a random time inside the sequence")
ap.add_argument("--hop", action="store_true", help="add LT+RT (hop) patterns")
ap.add_argument("--walk", action="store_true", help="drive while pressing: a random command from {0.3, 0.6, -0.3 m/s, +/-1 rad/s, 0.4 m/s + 0.5 rad/s}, then stop")
ap.add_argument("--extra", default="--wheel-kd 1.0 --charge-jump 0", help="extra harness arguments")
args = ap.parse_args()
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
os.makedirs(args.out, exist_ok=True)

patterns = {
    "LT": ["LT"], "RT": ["RT"], "CHG": ["CHG"], "LR": ["LT", "RT"], "RL": ["RT", "LT"], "LRLR": ["LT", "RT", "LT", "RT"],
    "CHG_LT": ["CHG", "LT"], "LT_CHG": ["LT", "CHG"], "LTLT": ["LT", "LT"], "CHGCHG": ["CHG", "CHG"], "RT_CHG_LT": ["RT", "CHG", "LT"],
}
if args.hop:
    patterns.update({"HOP": ["LT+RT"], "HOPHOP": ["LT+RT", "LT+RT"], "L_HOP": ["LT", "LT+RT"], "HOP_CHG": ["LT+RT", "CHG"], "CHG_HOP": ["CHG", "LT+RT"], "HOP_R": ["LT+RT", "RT"]})
grid = list(itertools.product(patterns, [0.15, 0.3, 0.5, 0.8, 1.5], [0.1, 0.4, 1.0], [10, 20], [0, 1, 2]))
random.Random(0).shuffle(grid)
grid = grid[: args.n]


def run(i, item):
    pat, gap, hold, lat, seed = item
    t, presses = 2.0, []
    for b in patterns[pat]:
        presses.append(f"{b}:{t:.2f}")
        t += (hold if b != "CHG" else max(hold, 0.4)) + gap
    dur = t + 4.0
    walk = []
    if args.walk:
        vx, wz = random.Random(1000 + i).choice([(0.3, 0), (0.6, 0), (-0.3, 0), (0, 1.0), (0, -1.0), (0.4, 0.5)])
        walk = ["--cmd", f"0:0:0,1.0:{vx}:{wz},{t + 1.0:.2f}:0:0"]
    d = os.path.join(args.out, f"{i:04d}")
    push = []
    if args.push:
        r = random.Random(i)
        push = ["--push-vel", f"{args.push * r.choice([-1, 1]):.2f}", "--push-time", f"{r.uniform(2.0, t):.2f}"]
    cmd = [sys.executable, "-I", os.path.join(root, "scripts/wheel_only/evaluate_stand_mujoco.py"), "--checkpoint", args.checkpoint,
           "--obs-noise", "--latency-ms", str(lat), "--seed", str(seed), "--duration", f"{dur:.2f}", "--presses", ",".join(presses),
           "--press-s", str(hold), "--charge-s", str(max(hold, 0.4)), "--output-dir", d] + push + walk + args.extra.split()
    subprocess.run(cmd, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        s = json.load(open(os.path.join(d, "summary.json")))
    except Exception:
        return dict(i=i, pat=pat, gap=gap, hold=hold, lat=lat, seed=seed, error=True)
    return dict(i=i, pat=pat, gap=gap, hold=hold, lat=lat, seed=seed, fell=s["fell_at_s"], tilt=s["max_tilt_deg"], drift=s["max_displacement_m"])


with ThreadPoolExecutor(args.jobs) as ex:
    res = list(ex.map(lambda x: run(*x), enumerate(grid)))
ok = [r for r in res if not r.get("error")]
fell = [r for r in ok if r["fell"] is not None]
summary = dict(checkpoint=args.checkpoint, runs=len(ok), errors=len(res) - len(ok), falls=len(fell), fall_rate=len(fell) / max(1, len(ok)),
               max_tilt=max((r["tilt"] for r in ok), default=None), falls_detail=fell)
json.dump(dict(summary=summary, runs=res), open(os.path.join(args.out, "sweep.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "falls_detail"}))
for r in fell:
    print("FELL", r)
