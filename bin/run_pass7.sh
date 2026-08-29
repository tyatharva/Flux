#!/usr/bin/env bash
# SEVENTH PASS, phases B and C: two seeds, two targets, and the deciding test.
#
# Runs unattended and keeps the GPU busy end to end. Every step asserts on the ARTIFACT it
# was supposed to produce, never on an exit status -- analyses are piped into grep and tee,
# so $? belongs to the last stage of the pipe (FASTEDDY_TRAPS.md 12).
#
# ORDER, and why it is this order:
#   B1  seed nbl-deep + accelerator, open-ended, 3.0 sim-h CEILING. Neutral spins up
#       slowest (h/u* ~ 1500 s against T* ~ 350 s convectively) and deeper is slower still,
#       so this is the run that MEASURES the library's budget and the one that exercises a
#       new configuration longest.
#   B2  the acceptance battery on whatever step it stopped at.
#   C1  seed cbl-deep. At 2928 m it runs at L/z_i = 3.08, where at 1952 m it LOCKED IN
#       (mode-1 share 53.9-72.0%, peak wavelength pinned at L). Its battery carries the
#       lock-in diagnostic, which is the direct test of the domain fix.
#   C2  one target off each seed, at datetimes and rotations chosen and PRE-REGISTERED
#       before either ran (results/deciding_test_preregistration.txt).
#   C3  the deciding test.
#
# The pre-target go/no-go -- the sub-grid fraction of sigma_w^2 at the receptor -- has
# already run on the bring-up control window and returned 52.5% against an expected ~52%
# (results/subgrid_fraction_30m.txt). It is re-scored here on each target's own window,
# where it is the fair number.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
L="${LOGDIR:-/tmp/flux-logs}"; mkdir -p "$L" results

export GRID=data/grid24_raised ZTARGET=28.5 EXACT_AGL=1
export TEMPLATE=runs/g24_base/base.in DX=24 ZCEILING=3000 DEFORM=0.346601 ZI_MAX_ABS=1250
export SEED_LIB=jobs24 ALLOW_INDETERMINATE=1 LPDM_WORKERS=12
# t_back = 900 s AT A 30 m RECEPTOR, not the 600 s measured at 10 m. Descent time scales
# roughly as z/sigma_w, so tripling the height roughly triples it; the fourth pass measured
# the shape converged at 450-600 s at THIS receptor height and ran production at 900. The
# window is (30 min averaging + t_back) = 2700 s, so a case is 4500 s = 1.25 sim-h =
# 0.60 GPU-h. --tback-marks re-measures the convergence curve for free, from touchdown ages
# already in hand, which is what fixes the production value.
export ADJ_S=1800 WINDOW_S="${WINDOW_S:-2700}" TBACK="${TBACK:-900}"
export TBACK_MARKS="${TBACK_MARKS:-150,300,450,600,750}"
export KEEP_TD=100000
# KEEP THE WINDOW FIELDS UNTIL THE NO-OP CONTROL HAS RUN. PROJECT_BRIEF.md's standing rule is to
# quote the no-op control beside every gate result, and here it is load-bearing: the
# deciding test asks whether the PEAK responds to meteorology, and at 86% sub-grid
# (neutral) a peak can move because the closure's own stability function moved rather than
# because the LES resolved different turbulence. Re-running each footprint with the floor
# OFF separates the two -- and it can only be done while the fields are still on disk,
# which is 15 GB per case for about twenty minutes.
export KEEP_FIELDS=1

die(){ echo "FATAL: $*" >&2; exit 1; }
say(){ echo; echo "########## $* ##########"; date '+%F %H:%M:%S'; }
have_seed(){ [ -s "jobs24/$1/return/seed_restart.nc" ]; }

bash bin/preflight.sh || die "preflight"

# ---------------------------------------------------------------- B: the neutral seed
if ! have_seed seed_nbl-deep_a000; then
  say "B1: seed nbl-deep + Steinfeld accelerator, open-ended, 3.0 sim-h ceiling"
  SEED_ACCEL_S=3000 SEED_EARLY_STOP=1 jobs/run_seed.sh jobs24/seed_nbl-deep_a000 \
    2>&1 | tee -a "$L/seedB.log"
  have_seed seed_nbl-deep_a000 || echo "  nbl-deep did not return a seed (see the gate)" >&2
fi
say "B2: acceptance battery, nbl-deep"
bash bin/seed_accept.sh jobs24/seed_nbl-deep_a000 2>&1 | tail -60

# ---------------------------------------------------------------- C: the deep CBL seed
if ! have_seed seed_cbl-deep_a000; then
  say "C1: seed cbl-deep, open-ended, 3.0 sim-h ceiling -- the lock-in re-test at L/z_i 3.08"
  SEED_EARLY_STOP=1 jobs/run_seed.sh jobs24/seed_cbl-deep_a000 2>&1 | tee -a "$L/seedC.log"
  have_seed seed_cbl-deep_a000 || echo "  cbl-deep did not return a seed (see the gate)" >&2
fi
say "C2: acceptance battery, cbl-deep (item 9 is the lock-in diagnostic)"
bash bin/seed_accept.sh jobs24/seed_cbl-deep_a000 2>&1 | tail -70

# ---------------------------------------------------------------- the two targets
# PRE-REGISTERED. A is convective off cbl-deep (Kljun x_peak 119 m), B is near-neutral off
# nbl-deep (160 m); the LES peaks must differ by more than each case's own half-vs-half
# floor and order the same way.
run_target(){   # tag  timestamp
  local TAG="$1" TS="$2"
  if [ -s "pairs/$TAG.json" ]; then echo "  $TAG already paired; skipping"; return 0; fi
  say "target $TAG at $TS"
  bash bin/run_corpus_case.sh "$TS" 2>&1 | tail -45
  [ -s "pairs/$TAG.json" ] || echo "  *** $TAG produced no pair" >&2
}
# THE NO-OP CONTROL, run while the fields are still on disk, and then the fields go.
# Same window, same releases, same everything -- with the sigma_w floor OFF. Without it a
# peak that moved between the two targets could have moved because the closure's own
# stability function moved rather than because the LES resolved different turbulence, and
# at 86% sub-grid (neutral) that is not a remote possibility.
nofloor(){
  local TAG="$1"
  ls -1 "runs/$TAG"/window/*.[0-9]* >/dev/null 2>&1 || { echo "  no fields for $TAG"; return 0; }
  local DTC
  DTC=$(python3 -c "import json;print(json.load(open('results/forcing/$TAG.json'))['params']['dt'])") \
    || { echo "  $TAG: no forcing dt"; return 0; }
  say "$TAG: NO-OP CONTROL -- identical window, sigma_w floor OFF"
  ./docker/pyrun.sh bin/stage5_footprint.py "runs/$TAG/window" --dt "$DTC" \
      --tback "$TBACK" --cover-dir "data/case_grids/$TAG" \
      --receptor-from "data/case_grids/$TAG" --fp16-cache \
      --z-target "$ZTARGET" --exact-agl --rel-seconds 1800 \
      --cover-groups 10 --outdir "${FPDIR:-results/corpus}" --tag "${TAG}_nofloor" \
      2>&1 | grep -vE 'batch [0-9]+/' | tail -25
  [ -s "${FPDIR:-results/corpus}/${TAG}_nofloor.json" ] \
    || echo "  *** $TAG no-op control produced no json" >&2
}
run_target case_2023052519 "2023-05-25 19:00"
nofloor    case_2023052519
run_target case_2023121921 "2023-12-19 21:00"
nofloor    case_2023121921
# The fields have served both the production footprint and its control; release them.
rm -f runs/case_2023052519/window/* runs/case_2023121921/window/* 2>/dev/null || true

# ---------------------------------------------------------------- the deciding test
say "THE DECIDING TEST"
./docker/pyrun.sh bin/case_compare.py case_2023052519 case_2023121921 \
  2>&1 | tee results/pass7_deciding_test.txt
[ -s results/pass7_deciding_test.txt ] || die "case_compare wrote nothing"
say "pass 7 complete"
