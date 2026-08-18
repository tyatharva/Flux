#!/usr/bin/env bash
# Stages 3 -> 6 in sequence, once the flat spin-up has reached its target step.
# Every FastEddy run goes through run_case.sh, which scores the log (CORRUPTED/NaN) and
# the newest dump (accuracy-CFL k0/k1) and refuses to start alongside another run.
set -uo pipefail
cd /home/atyagi/Flux
L=/tmp/flux-logs
step() { echo; echo "########## $* ##########"; }

step "accuracy-CFL confirmation at the 30 m grid (3 x ~30 s)"
for c in cfl160 cfl170 cfl180; do
  echo "-- $c ($(grep -oP '^dt = \K[0-9.]+' runs/s30_$c/case.in))"
  ./docker/run_case.sh "runs/s30_$c" case.in "$L/s30_$c.log"
done

step "Stage 3/5 window A (1800 s at 5 s cadence)"
./docker/run_case.sh runs/s30_w1 case.in "$L/s30_w1.log" || exit 1
grep -A3 'Total Time' "$L/s30_w1.log" | grep -E '^ +[0-9]' | tail -1

step "decorrelation gap (1200 s)"
./docker/run_case.sh runs/s30_gap case.in "$L/s30_gap.log" || exit 1

step "Stage 5 window B (second realisation)"
./docker/run_case.sh runs/s30_w2 case.in "$L/s30_w2.log" || exit 1

step "Stage 6 preprocessing (terrain + roughness into the restart)"
python3 bin/prep_stage6.py --restart-in runs/s30_spinup/output/FE_S30.345600 \
        --outdir runs/s30_stage6 --wind-from 270 || exit 1

step "Stage 6 adjustment (1200 s over the real surface)"
./docker/run_case.sh runs/s30_stage6 case.in "$L/s30_t6a.log" || exit 1

step "Stage 6 sampling window (1800 s at 5 s cadence)"
./docker/run_case.sh runs/s30_stage6_smp case.in "$L/s30_t6.log" || exit 1

step "DONE"
du -sh runs/s30_w1/output runs/s30_w2/output runs/s30_stage6_smp/output
