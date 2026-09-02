#!/bin/bash
# Conditional runs triggered by pass 1 (array-in-view cover90 < 0.81 after the fine-tunes):
# share-weighted CRPS fine-tune and CRPS from scratch, then a second evaluator pass.
set -u
cd "$(dirname "$0")/.."
PY=/home/atyagi/miniforge3/envs/LESNet/bin/python
C=results/ml_cfm/calib
stamp() { echo "- $(date -u +%FT%TZ) $*" >> results/ml_cfm/TIMELINE.md; }
stamp "batch C starts (crps_share_ft lam_share 5, crps_pure_scratch; K=2) -- conditional runs triggered by pass 1"
$PY -u -m ml_cfm.campaign --runs $C/runs/runs_C.json -K 2 --outdir $C/runs --tag calibC \
    --base '{"save_ckpt": true, "save_samples": 0}' > $C/runs/campaign_C.log 2>&1
touch $C/runs/C_DONE; stamp "batch C done; calibrate pass 2 starts"
$PY -u -m ml_cfm.calibrate --tag final2 --S 64 --no-temperature --models fm_seed1=results/ml_cfm/final/seed1 \
    crps_share_ft=$C/runs/crps_share_ft crps_pure_scratch=$C/runs/crps_pure_scratch \
    crps_pure_ft=$C/runs/crps_pure_ft > $C/calibrate_final2.log 2>&1
touch $C/CALIB2_DONE; stamp "CALIB2_DONE"
