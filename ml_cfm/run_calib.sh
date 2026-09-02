#!/bin/bash
# The calibration / tail follow-up, end to end: batch A (CRPS fine-tunes), batch B (thresholded
# targets), then ml_cfm.calibrate on every variant. Markers: A_DONE, B_DONE, CALIB_DONE.
set -u
cd "$(dirname "$0")/.."
PY=/home/atyagi/miniforge3/envs/LESNet/bin/python
C=results/ml_cfm/calib; T=results/ml_cfm/tail
stamp() { echo "- $(date -u +%FT%TZ) $*" >> results/ml_cfm/TIMELINE.md; }
stamp "batch A starts (crps_pure_ft, crps_blend_ft, crps_pure_ft_S4; K=3)"
$PY -u -m ml_cfm.campaign --runs $C/runs/runs_A.json -K 3 --outdir $C/runs --tag calibA \
    --base '{"save_ckpt": true, "save_samples": 0}' > $C/runs/campaign_A.log 2>&1
touch $C/runs/A_DONE; stamp "batch A done"
stamp "batch B starts (thresh_seed0, thresh_seed1; K=2)"
$PY -u -m ml_cfm.campaign --runs $T/runs/runs_B.json -K 2 --outdir $T/runs --tag tailB \
    --base '{"save_ckpt": true, "save_samples": 0}' > $T/runs/campaign_B.log 2>&1
touch $T/runs/B_DONE; stamp "batch B done"
stamp "calibrate (S=64, 11 variants) starts"
MODELS="fm_seed1=results/ml_cfm/final/seed1 fm_seed1_e2=results/ml_cfm/final/seed1,steps=2 \
 fm_seed1_sig0.2=results/ml_cfm/final/seed1,sigma=0.2 fm_seed1_sig0.3=results/ml_cfm/final/seed1,sigma=0.3 \
 fm_seed1_sig0.5=results/ml_cfm/final/seed1,sigma=0.5 fm_sig0.3_trained=results/ml_cfm/phase1/v_s0.3 \
 crps_pure_ft=$C/runs/crps_pure_ft crps_blend_ft=$C/runs/crps_blend_ft crps_pure_ft_S4=$C/runs/crps_pure_ft_S4 \
 thresh_seed0=$T/runs/thresh_seed0 thresh_seed1=$T/runs/thresh_seed1"
$PY -u -m ml_cfm.calibrate --tag final --S 64 --models $MODELS > $C/calibrate_final.log 2>&1
stamp "calibrate final done; thresholded-target scoring starts"
$PY -u -m ml_cfm.calibrate --tag final_vs_thresholded --S 64 --score-target sa99 --no-temperature \
    --models fm_seed1=results/ml_cfm/final/seed1 fm_seed2=results/ml_cfm/final/seed2 \
             thresh_seed0=$T/runs/thresh_seed0 thresh_seed1=$T/runs/thresh_seed1 > $C/calibrate_thr.log 2>&1
touch $C/CALIB_DONE; stamp "CALIB_DONE"
