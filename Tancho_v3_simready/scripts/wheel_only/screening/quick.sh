#!/bin/bash
# Quick MuJoCo screen under deployment conditions (training sensor noise + 20 ms latency):
# 60 s standing x3 seeds, LT / RT / crouch x2 rounds at 10 and 20 ms x2 seeds.
# usage: quick.sh <run dir> <label> "<iters>"
cd "$(dirname "$0")/../../.."
RUN=$1; LAB=$2; ITERS=$3; S=logs/station_20261103/ckpt_scan
for it in $ITERS; do
  d=$S/${LAB}_$it; mkdir -p $d
  [ -f $d/exported/policy.pt ] || { cp $RUN/model_$it.pt $d/ && conda run -n env_isaaclab python scripts/wheel_only/export_policy_cpu.py --checkpoint $d/model_$it.pt > $d/export.log 2>&1; }
  [ -f $d/exported/policy.pt ] || continue
  M="nice conda run -n env_isaaclab python -I scripts/wheel_only/evaluate_stand_mujoco.py --checkpoint $d/exported/policy.pt --charge-jump 0 --wheel-kd 1.0 --obs-noise"
  for s in 0 1 2; do $M --latency-ms 20 --seed $s --duration 60 --output-dir $d/q_stand_s$s > /dev/null 2>&1; done
  for s in 0 1; do for lat in 10 20; do $M --latency-ms $lat --seed $s --duration 30 --presses LT:3,RT:7,CHG:11,LT:15,RT:19,CHG:23 --output-dir $d/q_p_l${lat}_s$s > /dev/null 2>&1; done; done
  python3 - $d <<'P' | tee -a $S/quick_table.txt
import json, glob, os, sys
d = sys.argv[1]
P = [json.load(open(f)) for f in glob.glob(d + "/q_p_*/summary.json")]; S = [json.load(open(f)) for f in glob.glob(d + "/q_stand_*/summary.json")]
print(os.path.basename(d), f"stand fell {sum(s['fell_at_s'] is not None for s in S)}/{len(S)} drift {sorted(round(s['pre_push_max_drift_m'], 3) for s in S)}",
      f"| presses fell {sum(p['fell_at_s'] is not None for p in P)}/{len(P)} tilt max {max(p['max_tilt_deg'] for p in P):.1f} drift max {max(p['pre_push_max_drift_m'] for p in P):.3f}")
P
done
