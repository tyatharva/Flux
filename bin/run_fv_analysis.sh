#!/usr/bin/env bash
set -uo pipefail
cd /home/atyagi/Flux; mkdir -p results
DT=0.0340136
step(){ echo; echo "########## $* ##########"; }

step "Stage 4: well-mixed test + transit time"
./docker/pyrun.sh bin/stage4_wellmixed.py runs/fv_wlong/output --dt $DT --n 60000 \
  --tlimit 900 2>&1 | grep -v 'loaded ' | tee results/fv_stage4.txt | tail -30

step "Stage 5: flat footprint, error floor from sub-windows, vs Kljun"
./docker/pyrun.sh bin/stage5_footprint.py runs/fv_wlong/output --dt $DT --res 60 \
  --tag fv_stage5 2>&1 | grep -vE 'batch [0-9]+/|loaded ' | tee results/fv_stage5.txt

step "Stage 6 WESTERLY footprint (array upwind)"
./docker/pyrun.sh bin/stage5_footprint.py runs/fv_t6w/output --dt $DT --res 60 \
  --cover-dir runs/fv_t6w_adj --tag fv_t6w 2>&1 | grep -vE 'batch [0-9]+/|loaded ' \
  | tee results/fv_t6w.txt

step "Stage 6 EASTERLY footprint (water upwind)"
./docker/pyrun.sh bin/stage5_footprint.py runs/fv_t6e/output --dt $DT --res 60 \
  --cover-dir runs/fv_t6e_adj --tag fv_t6e 2>&1 | grep -vE 'batch [0-9]+/|loaded ' \
  | tee results/fv_t6e.txt

step "Stage 6 gate: westerly vs flat"
./docker/pyrun.sh bin/stage6_compare.py results/fv_stage5.npz results/fv_t6w.npz \
  --topo runs/fv_t6w_adj/topo.npy --z0 runs/fv_t6w_adj/z0m.npy --tag fv_stage6_w \
  2>&1 | tee results/fv_stage6_w.txt
step "Stage 6 gate: easterly vs flat"
./docker/pyrun.sh bin/stage6_compare.py results/fv_stage5.npz results/fv_t6e.npz \
  --topo runs/fv_t6e_adj/topo.npy --z0 runs/fv_t6e_adj/z0m.npy --tag fv_stage6_e \
  2>&1 | tee results/fv_stage6_e.txt

step "Item 4: ensemble convergence from sub-windows"
./docker/pyrun.sh bin/ensemble_convergence.py runs/fv_wlong/output --dt $DT --nsub 18 \
  --tag fv_ensemble 2>&1 | grep -v 'batch ' | tee results/fv_ensemble.txt

step "ANALYSIS DONE"
