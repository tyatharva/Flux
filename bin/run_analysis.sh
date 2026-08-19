#!/usr/bin/env bash
# Stage 4, 5 and 6 gates on the finished sampling windows. CPU only.
set -uo pipefail
cd /home/atyagi/Flux
mkdir -p results
step() { echo; echo "########## $* ##########"; }

step "Stage 4 gate: well-mixed test + backward transit time"
./docker/pyrun.sh bin/stage4_wellmixed.py runs/s30_w1/output --n 40000 --tlimit 900 \
  2>&1 | tee results/stage4.txt

step "Stage 5: flat/neutral footprint vs Kljun, plus the error floor"
./docker/pyrun.sh bin/stage5_footprint.py runs/s30_w1/output runs/s30_w2/output \
  --tag stage5 2>&1 | tee results/stage5.txt

step "Stage 6: footprint over the real surface"
./docker/pyrun.sh bin/stage5_footprint.py runs/s30_stage6_smp/output \
  --tag stage6_raw 2>&1 | tee results/stage6_raw.txt

step "Stage 6 gate: real vs flat vs Kljun"
./docker/pyrun.sh bin/stage6_compare.py results/stage5.npz results/stage6_raw.npz \
  2>&1 | tee results/stage6.txt

step "DONE"
