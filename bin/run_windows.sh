#!/usr/bin/env bash
# The 5 s-cadence sampling windows. Separate from run_pipeline.sh because these are the
# runs where NtBatch MUST equal frqOutput: FastEddy's time loop advances by NtBatch and
# only tests it%frqOutput at batch boundaries (FastEddy.c:400,423), so a frqOutput finer
# than NtBatch is silently ignored and you get two dumps instead of 360.
set -uo pipefail
cd /home/atyagi/Flux
L=/tmp/claude-1000
step() { echo; echo "########## $* ##########"; }

step "window A (1800 s at 5 s cadence)"
rm -f runs/s30_w1/output/*
./docker/run_case.sh runs/s30_w1 case.in "$L/s30_w1.log" || exit 1
echo "  dumps: $(ls runs/s30_w1/output | wc -l)   $(du -sh runs/s30_w1/output | cut -f1)"
grep -A3 'Total Time' "$L/s30_w1.log" | grep -E '^ +[0-9]' | tail -1

step "decorrelation gap (1200 s)"
rm -f runs/s30_gap/output/*
./docker/run_case.sh runs/s30_gap case.in "$L/s30_gap.log" || exit 1

step "window B (1800 s at 5 s cadence, second realisation)"
rm -f runs/s30_w2/output/*
./docker/run_case.sh runs/s30_w2 case.in "$L/s30_w2.log" || exit 1
echo "  dumps: $(ls runs/s30_w2/output | wc -l)"

step "Stage 6 sampling window (1800 s at 5 s cadence, real surface)"
rm -f runs/s30_stage6_smp/output/*
./docker/run_case.sh runs/s30_stage6_smp case.in "$L/s30_t6.log" || exit 1
echo "  dumps: $(ls runs/s30_stage6_smp/output | wc -l)"

step "DONE"
du -sh runs/s30_w1/output runs/s30_w2/output runs/s30_stage6_smp/output
