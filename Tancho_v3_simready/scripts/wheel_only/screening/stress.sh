#!/bin/bash
# MuJoCo deployment stress of one exported policy: training sensor noise + 20 ms motor
# latency, 3 seeds of 60 s standing, a 0.5 m/s push, and two rounds of LT / RT / crouch.
# usage: stress.sh <policy.pt> <out dir>
cd "$(dirname "$0")/../../.."
P=$1; O=$2; mkdir -p $O
M="nice conda run -n env_isaaclab python -I scripts/wheel_only/evaluate_stand_mujoco.py --checkpoint $P --charge-jump 0 --wheel-kd 1.0 --obs-noise --latency-ms 20"
for s in 0 1 2; do $M --seed $s --duration 60 --output-dir $O/stand_s$s > /dev/null 2>&1; done
$M --duration 20 --push-vel 0.5 --output-dir $O/push > /dev/null 2>&1
for s in 0 1 2; do $M --seed $s --duration 30 --presses LT:3,RT:7,CHG:11,LT:15,RT:19,CHG:23 --output-dir $O/presses_s$s > /dev/null 2>&1; done
python3 - "$O" <<'P'
import json, sys, glob, os
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/*/summary.json")):
    s = json.load(open(f)); rows.append((os.path.basename(os.path.dirname(f)), s["fell_at_s"], s["pre_push_max_drift_m"]))
falls = [r[0] for r in rows if r[1] is not None]
stand = max(r[2] for r in rows if r[0].startswith("stand"))
print(os.path.basename(sys.argv[1]), "falls", falls, "worst stand drift", round(stand, 3), " ".join(f"{n}={'FELL' if fl else round(d, 3)}" for n, fl, d in rows))
P
