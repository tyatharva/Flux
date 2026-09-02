#!/bin/bash
# Extension: wait for run_all, then seeds 3-4 (500 ep) and a 1000-epoch run of the winner.
cd "$(dirname "$0")/.."
PY=/home/atyagi/miniforge3/envs/LESNet/bin/python
while [ ! -f results/ml_cfm/ALL_DONE ]; do sleep 60; done
echo "- $(date -u +%H:%M:%SZ) run_ext: starts" >> results/ml_cfm/TIMELINE.md
$PY - <<'PYEOF'
import json
c = json.load(open("results/ml_cfm/final/runs.json"))["seed0"]
ext = {"seed3": dict(c, seed=3), "seed4": dict(c, seed=4),
       "long1000_seed0": dict(c, seed=0, epochs=1000, patience=20)}
json.dump(ext, open("results/ml_cfm/final/runs_ext.json", "w"), indent=1)
PYEOF
$PY -u -m ml_cfm.campaign --runs results/ml_cfm/final/runs_ext.json -K 3 --outdir results/ml_cfm/final --tag ext \
    --base '{"epochs": 500, "save_samples": 32, "steps_final": 16}' --baseline-prefix seed > results/ml_cfm/final/campaign_ext.log 2>&1
echo "- $(date -u +%H:%M:%SZ) run_ext: done; EXT_DONE" >> results/ml_cfm/TIMELINE.md
touch results/ml_cfm/EXT_DONE
