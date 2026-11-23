#!/bin/bash
# Terrain / incline checks of one exported policy (60 s zero command, plus a 0.5 m/s push
# on rough ground).  Isaac: 0-2 cm bumps at difficulty 0.5 / 1, gravity tilted +/-5, +/-8 deg.
# MuJoCo: gravity tilted +/-5, +/-8 deg.   usage: terrain.sh <policy.pt> <out dir> <play task>
cd "$(dirname "$0")/../../.."
P=$1; O=$2; T=$3; mkdir -p $O
E=scripts/wheel_only/evaluate_stability.py
for v in "rough05 --terrain rough --difficulty 0.5" "rough10 --terrain rough --difficulty 1.0" "slope+5 --slope-deg 5" "slope-5 --slope-deg -5" "slope+8 --slope-deg 8" "slope-8 --slope-deg -8"; do
  set -- $v; n=$1; shift
  timeout -s KILL 1200 conda run -n env_isaaclab python $E --headless --task $T --checkpoint $P --duration 60 --push-vel 0 --output-dir $O/is_$n "$@" > $O/is_$n.log 2>&1
done
timeout -s KILL 1200 conda run -n env_isaaclab python $E --headless --task $T --checkpoint $P --duration 20 --push-vel 0.5 --terrain rough --difficulty 1.0 --output-dir $O/is_rough10_push > $O/is_rough10_push.log 2>&1
M="nice conda run -n env_isaaclab python -I scripts/wheel_only/evaluate_stand_mujoco.py --checkpoint $P --charge-jump 0 --wheel-kd 1.0 --duration 60 ${MJX:-}"
for s in 5 -5 8 -8; do $M --slope-deg $s --output-dir $O/mj_slope$s > /dev/null 2>&1; done
python3 - "$O" <<'P'
import json, sys, glob, os
for f in sorted(glob.glob(sys.argv[1] + "/*/*summary.json")):
    s = json.load(open(f)); n = os.path.basename(os.path.dirname(f))
    fell = s.get("terminated_at_s", s.get("fell_at_s"))
    print(f"{n:16s} fell {fell}  drift {s['pre_push_max_drift_m']:.3f}  max {s['max_displacement_m']:.3f}")
P
