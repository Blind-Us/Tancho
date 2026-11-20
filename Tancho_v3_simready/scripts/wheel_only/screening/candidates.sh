#!/bin/bash
# Full screen of a run's checkpoints: export, MuJoCo stand 60 s (stiff + soft), MuJoCo push,
# Isaac stand 60 s + push 0.5.  Results go to logs/station_20261103/ckpt_scan/<label>_<iter>/ and candidates_table.txt.
# usage: scripts/wheel_only/screening/candidates.sh <run dir> <label> <play task> "<iters>"
cd "$(dirname "$0")/../../.."
RUN=$1; LAB=$2; TASK=$3; ITERS=$4
S=logs/station_20261103/ckpt_scan
E=scripts/wheel_only/evaluate_stability.py
for it in $ITERS; do
  d=$S/${LAB}_$it; mkdir -p $d
  [ -f $d/exported/policy.pt ] || { cp $RUN/model_$it.pt $d/; timeout -s KILL 600 conda run -n env_isaaclab python scripts/wheel_only/export_policy.py --headless --task ${TASK%-Play-v0}-v0 --checkpoint $d/model_$it.pt > $d/export.log 2>&1; }
  P=$d/exported/policy.pt
  M="nice conda run -n env_isaaclab python -I scripts/wheel_only/evaluate_stand_mujoco.py --checkpoint $P --charge-jump 0 --wheel-kd 1.0"
  $M --duration 60 --output-dir $d/mj_stand60 > /dev/null 2>&1
  $M --duration 60 --solref 0.02 --output-dir $d/mj_soft > /dev/null 2>&1
  $M --duration 20 --push-vel 0.5 --output-dir $d/mj_push0p5 > /dev/null 2>&1
  timeout -s KILL 1200 conda run -n env_isaaclab python $E --headless --task $TASK --checkpoint $P --duration 60 --push-vel 0 --output-dir $d/isaac_stand_60s > $d/isaac_stand.log 2>&1
  timeout -s KILL 1200 conda run -n env_isaaclab python $E --headless --task $TASK --checkpoint $P --duration 20 --push-vel 0.5 --output-dir $d/isaac_push_0p5 > $d/isaac_push.log 2>&1
  python3 - $d <<'P' >> $S/candidates_table.txt
import json, sys, os
d = sys.argv[1]
def j(p):
    try: return json.load(open(p))
    except Exception: return {}
a, b, c = j(d + "/mj_stand60/summary.json"), j(d + "/mj_soft/summary.json"), j(d + "/mj_push0p5/summary.json")
e, f = j(d + "/isaac_stand_60s/stand_summary.json"), j(d + "/isaac_push_0p5/stand_summary.json")
r = lambda x: None if x is None else round(x, 3)
print(os.path.basename(d), "| mj60", r(a.get("pre_push_max_drift_m")), a.get("fell_at_s"), "| mjsoft", r(b.get("pre_push_max_drift_m")), b.get("fell_at_s"),
      "| mjpush fell", c.get("fell_at_s"), "| is60", r(e.get("pre_push_max_drift_m")), e.get("terminated_at_s"), "pitch", r(e.get("pre_push_max_pitch_dev_deg")),
      "| ispush fell", f.get("terminated_at_s"), "pass", f.get("pass_all"))
P
done
echo done > $S/${LAB}_done
