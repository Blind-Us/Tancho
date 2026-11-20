#!/bin/bash
# MuJoCo 60 s zero-command stand of one exported policy under several sim variations.
# usage: robust.sh <policy.pt> <out_dir>
cd "$(dirname "$0")/../../.."
P=$1; O=$2; mkdir -p $O
M="nice conda run -n env_isaaclab python -I scripts/wheel_only/evaluate_stand_mujoco.py --checkpoint $P --charge-jump 0 --wheel-kd 1.0 --duration 60"
for v in "nom" "fr0.5 --friction 0.5" "fr0.8 --friction 0.8" "dt2 --dt 0.002" "p+3 --init-pitch-deg 3" "p-3 --init-pitch-deg -3" "kd0.8 --wheel-kd 0.8" "kd1.2 --wheel-kd 1.2"; do
  set -- $v; name=$1; shift
  $M "$@" --output-dir $O/$name > $O/$name.log 2>&1
done
python3 - "$O" <<'P'
import json, sys, glob, os
for f in sorted(glob.glob(sys.argv[1] + "/*/summary.json")):
    s = json.load(open(f)); print(os.path.basename(os.path.dirname(f)), "fell", s["fell_at_s"], "max", round(s["pre_push_max_drift_m"], 3), "end", round(s["end_displacement_m"], 3))
P
