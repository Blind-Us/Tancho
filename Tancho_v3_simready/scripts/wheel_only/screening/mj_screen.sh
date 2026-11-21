#!/bin/bash
# Fast MuJoCo screen of a run's checkpoints: export, then 60 s zero command on flat and on
# a gravity-tilted incline (+/-5, +/-8 deg); prints the worst drift per checkpoint.
# usage: mj_screen.sh <run dir> <label> <train task> "<iters>"
cd "$(dirname "$0")/../../.."
RUN=$1; LAB=$2; TASK=$3; ITERS=$4
S=logs/station_20261103/ckpt_scan
for it in $ITERS; do
  d=$S/${LAB}_$it; mkdir -p $d
  [ -f $d/exported/policy.pt ] || { cp $RUN/model_$it.pt $d/; conda run -n env_isaaclab python scripts/wheel_only/export_policy_cpu.py --checkpoint $d/model_$it.pt > $d/export.log 2>&1; }
  M="nice conda run -n env_isaaclab python -I scripts/wheel_only/evaluate_stand_mujoco.py --checkpoint $d/exported/policy.pt --charge-jump 0 --wheel-kd 1.0 --duration 60"
  $M --output-dir $d/mj_stand60 > /dev/null 2>&1
  for s in 5 -5 8 -8; do $M --slope-deg $s --output-dir $d/mj_slope$s > /dev/null 2>&1; done
  python3 - $d <<'P' | tee -a $S/mj_screen_table.txt
import json, sys, os
d = sys.argv[1]; out = []
for n in ("mj_stand60", "mj_slope5", "mj_slope-5", "mj_slope8", "mj_slope-8"):
    try:
        s = json.load(open(f"{d}/{n}/summary.json")); out.append((n, s["pre_push_max_drift_m"], s["fell_at_s"]))
    except Exception:
        out.append((n, None, "missing"))
worst = max((x[1] for x in out if x[1] is not None), default=None)
fell = [x[0] for x in out if x[2] is not None]
print(os.path.basename(d), "worst", None if worst is None else round(worst, 3), "fell", fell, " ".join(f"{n[3:]}={v if v is None else round(v, 3)}" for n, v, _ in out))
P
done
