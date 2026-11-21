#!/bin/bash
# Deployment gate for one exported 34-dim policy (wheel Kd 1, no hop).  All under training
# sensor noise + motor latency.  Prints one line; a candidate must have zero falls in the
# first three items.
#   MuJoCo 990 press combinations with the 0.3 s crouch cooldown,
#   Isaac 60 s standing x4 (20 ms), MuJoCo 60 s standing x3 (20 ms),
#   MuJoCo 990 press combinations with a random +/-0.5 m/s kick.
# usage: gate.sh <policy.pt> <out dir>
cd "$(dirname "$0")/../../.."
P=$1; O=$2; mkdir -p $O
X="--wheel-kd 1.0 --charge-jump 0 --charge-cooldown 0.3"
a=$(conda run -n env_isaaclab python scripts/wheel_only/screening/press_sweep.py --checkpoint $P --out $O/sweep --jobs 20 --n 990 --extra "$X" 2>&1 | head -1 | python3 -c "import json,sys;s=json.loads(sys.stdin.read());print(f\"{s['falls']}/{s['runs']} tilt {s['max_tilt']:.0f}\")")
is=""; for k in 1 2 3 4; do timeout -s KILL 1200 conda run -n env_isaaclab python scripts/wheel_only/evaluate_stability.py --headless --task TanchoV3-StationSquatKd1Com-Play-v0 --checkpoint $P --duration 60 --push-vel 0 --obs-noise --latency-substeps 4 --output-dir $O/is_$k > $O/is_$k.log 2>&1; is="$is $(python3 -c "import json;s=json.load(open('$O/is_$k/stand_summary.json'));print('FELL' if s['terminated_at_s'] else round(s['pre_push_max_drift_m'],3))")"; done
mj=""; for s in 0 1 2; do nice conda run -n env_isaaclab python -I scripts/wheel_only/evaluate_stand_mujoco.py --checkpoint $P $X --obs-noise --latency-ms 20 --seed $s --duration 60 --output-dir $O/mj_$s > /dev/null 2>&1; mj="$mj $(python3 -c "import json;s=json.load(open('$O/mj_$s/summary.json'));print('FELL' if s['fell_at_s'] else round(s['pre_push_max_drift_m'],3))")"; done
b=$(conda run -n env_isaaclab python scripts/wheel_only/screening/press_sweep.py --checkpoint $P --out $O/sweep_push --jobs 20 --n 990 --push 0.5 --extra "$X" 2>&1 | head -1 | python3 -c "import json,sys;s=json.loads(sys.stdin.read());print(f\"{s['falls']}/{s['runs']}\")")
echo "$(basename $O) | presses $a | isaac stand$is | mj stand$mj | push-in-press $b" | tee -a $(dirname $O)/gate_table.txt
