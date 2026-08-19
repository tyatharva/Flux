#!/usr/bin/env bash
# Stages 2-6 on the 8.56 m near-surface vertical grid, at the surveyed tower coordinate.
# Every run is projected under 45 min; the spin-up is chained as two restart segments.
set -uo pipefail
cd /home/atyagi/Flux
L=/tmp/claude-1000
step(){ echo; echo "########## $* ##########"; }

step "Stage 2: spin-up segment 2 (1.5 h -> 3.5 h simulated) [proj 30.3 min]"
./docker/run_case.sh runs/fv_spinup2 case.in "$L/fv_seg2.log" || exit 1
cp -f runs/fv_spinup2/output/FE_FV.370440 runs/fv_spinup/output/ 2>/dev/null || true

step "Stage 2 gate: stationarity"
FE_DT=0.0340136 ./docker/pyrun.sh docker/stage2_gate.py \
  $(ls -v runs/fv_spinup/output/FE_FV.* runs/fv_spinup2/output/FE_FV.* 2>/dev/null) \
  2>&1 | tee results/fv_stage2.txt | sed -n '1,40p'

step "Stage 5 Gate 1 (revised): sub-grid fraction of sigma_w^2 at the receptor"
./docker/pyrun.sh bin/subgrid_gate.py runs/fv_spinup2/output --stride 1 \
  2>&1 | tee results/fv_subgrid.txt

step "Stage 3/5: flat long sampling window, 3600 s at 5 s [proj 15.2 min]"
./docker/run_case.sh runs/fv_wlong case.in "$L/fv_wlong.log" || exit 1
echo "  dumps: $(ls runs/fv_wlong/output | wc -l)  $(du -sh runs/fv_wlong/output | cut -f1)"

step "Stage 6 WESTERLY: preprocessing (array upwind, water downwind)"
python3 bin/prep_stage6.py --restart-in runs/fv_spinup2/output/FE_FV.370440 \
  --outdir runs/fv_t6w_adj --wind-from 270 || exit 1
cp -f runs/fv_t6w_adj/topo.bin runs/fv_t6w/ 2>/dev/null || true

step "Stage 6 WESTERLY: adjustment 1200 s [proj 5.1 min]"
./docker/run_case.sh runs/fv_t6w_adj case.in "$L/fv_t6wa.log" || exit 1
step "Stage 6 WESTERLY: sampling window 1800 s [proj 7.6 min]"
./docker/run_case.sh runs/fv_t6w case.in "$L/fv_t6w.log" || exit 1

step "Stage 6 EASTERLY: preprocessing (water upwind, 62% of the upwind half)"
python3 bin/prep_stage6.py --restart-in runs/fv_spinup2/output/FE_FV.370440 \
  --outdir runs/fv_t6e_adj --wind-from 90 || exit 1
cp -f runs/fv_t6e_adj/topo.bin runs/fv_t6e/ 2>/dev/null || true

step "Stage 6 EASTERLY: adjustment 1200 s [proj 5.1 min]"
./docker/run_case.sh runs/fv_t6e_adj case.in "$L/fv_t6ea.log" || exit 1
step "Stage 6 EASTERLY: sampling window 1800 s [proj 7.6 min]"
./docker/run_case.sh runs/fv_t6e case.in "$L/fv_t6e.log" || exit 1

step "DONE"
du -sh runs/fv_*/output
