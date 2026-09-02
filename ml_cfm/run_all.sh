#!/bin/bash
# The autonomous study: Phase 1 (4 runs) -> pick on val -> 3 final seeds -> solver study.
set -u
cd "$(dirname "$0")/.."
PY=/home/atyagi/miniforge3/envs/LESNet/bin/python
P1=results/ml_cfm/phase1; FN=results/ml_cfm/final
stamp() { echo "- $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> results/ml_cfm/TIMELINE.md; }
stamp "run_all: phase1 starts"
$PY -u -m ml_cfm.campaign --runs $P1/runs.json -K 4 --outdir $P1 --tag p1 > $P1/campaign.log 2>&1
stamp "run_all: phase1 done"
$PY - <<'PYEOF'
import json, glob, os
runs = {}
for p in glob.glob("results/ml_cfm/phase1/*/run.json"):
    n = os.path.basename(os.path.dirname(p))
    if n.startswith("smoke"): continue
    runs[n] = json.load(open(p))
best = min(runs, key=lambda k: runs[k]["val_mse_ref"])
seeds = [runs[k]["val_mse_ref"] for k in runs if k.startswith("v_s0.1_seed")]
spread = abs(seeds[0] - seeds[1]) if len(seeds) == 2 else 0.0
ref = min(seeds) if seeds else float("inf")
# a difference inside the seed pair's spread is a tie: keep velocity / sigma 0.1
if not best.startswith("v_s0.1") and runs[best]["val_mse_ref"] > ref - spread:
    best = "v_s0.1_seed0"
cfg = runs[best]["config"]
final = {f"seed{s}": dict(param=cfg["param"], sigma=cfg["sigma"], seed=s) for s in range(3)}
os.makedirs("results/ml_cfm/final", exist_ok=True)
json.dump(final, open("results/ml_cfm/final/runs.json", "w"), indent=1)
json.dump(dict(winner=best, val_mse_ref={k: v["val_mse_ref"] for k, v in runs.items()},
               composite={k: v["composite"] for k, v in runs.items()}, seed_pair_spread=spread),
          open("results/ml_cfm/phase1/pick.json", "w"), indent=1)
print("winner", best, "spread", spread)
PYEOF
stamp "run_all: final starts ($(cat $P1/pick.json | tr -d '\n' | cut -c1-80))"
$PY -u -m ml_cfm.campaign --runs $FN/runs.json -K 3 --outdir $FN --tag final \
    --base '{"epochs": 500, "save_samples": 32, "steps_final": 16}' --baseline-prefix seed > $FN/campaign.log 2>&1
stamp "run_all: final done"
$PY -u -m ml_cfm.solver_study --ckpt $FN/seed0/best.pt --S 8 --out $P1/solver_study.json > $P1/solver_study.log 2>&1
stamp "run_all: solver study done; ALL DONE"
touch results/ml_cfm/ALL_DONE
