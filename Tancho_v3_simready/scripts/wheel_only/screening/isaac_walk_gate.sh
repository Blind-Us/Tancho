#!/bin/bash
# Isaac checks for a WASD policy under noise + 20 ms latency: 60 s standing x3, the walk
# command sequence, LT / RT / LT+RT / crouch while walking at 0.3 m/s and at rest.
# usage: isaac_walk_gate.sh <policy.pt> <out dir> <play task>
cd "$(dirname "$0")/../../.."
P=$1; O=$2; T=$3; mkdir -p $O; NL="--obs-noise --latency-substeps 4"
run(){ timeout -s KILL 1200 conda run -n env_isaaclab python "$@"; }
st=""; for k in 1 2 3; do run scripts/wheel_only/evaluate_stability.py --headless --task $T --checkpoint $P --duration 60 --push-vel 0 $NL --output-dir $O/stand_$k > $O/stand_$k.log 2>&1; st="$st $(python3 -c "import json;s=json.load(open('$O/stand_$k/stand_summary.json'));print('FELL' if s['terminated_at_s'] else round(s['pre_push_max_drift_m'],3))")"; done
run scripts/wheel_only/evaluate_walk.py --headless --task $T --checkpoint $P $NL --output-dir $O/walk > $O/walk.log 2>&1
wk=$(python3 -c "import json,glob;s=json.load(open(glob.glob('$O/walk/*.json')[0]));print('FELL' if s.get('terminated_at_s') else 'ok', max(round(v['mean_abs_vx_err'],2) for v in s['segments'].values()), max(round(v['mean_abs_wz_err'],2) for v in s['segments'].values()))")
hp=""; for vx in 0.3 0; do run scripts/wheel_only/evaluate_hop.py --headless --task $T --checkpoint $P --guidance 1 --vx $vx --repeats 2 --schedule LT,RT,LT+RT,CHG --gap 3 --charge-s 0.6 $NL --output $O/hop_$vx > $O/hop_$vx.log 2>&1; hp="$hp vx$vx:$(python3 -c "
import json,re;t=open('$O/hop_$vx.log').read();j=json.loads(t[t.index('{\n'):t.rindex('}')+1]);h=[p for p in j['presses'] if p['mode']=='LT+RT'];print(('FELL@'+str(j['terminated_at_s'])) if j['terminated_at_s'] else 'ok', 'hop', [round(min(p['clear_L_m'],p['clear_R_m'])*1000) for p in h])")"; done
echo "$(basename $O) | isaac stand$st | walk $wk |$hp" | tee -a $(dirname $O)/isaac_walk_table.txt
