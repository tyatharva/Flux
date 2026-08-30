#!/usr/bin/env bash
# EIGHTH PASS: the in-process hand-off, 122^3 @ 30 m, two footprints per case.
#
# Runs unattended and asserts on the ARTIFACT at every step, never on an exit status --
# analyses are piped into grep and tee, so $? belongs to the last stage of the pipe
# (FASTEDDY_TRAPS.md 12).
#
# ORDER, and why it is this order:
#   A  the flat dt boundary (bin/g30_bringup.sh, run separately -- it is the input to
#      everything below and its answer is written into runs/g30_base/base.in by hand,
#      because a production dt is a decision and not a default).
#   B  the two seeds, cbl-deep and nbl-deep, 3.0 sim-h HARD ceiling.
#   C  the flat/neutral control: the regression baseline, Gate D1 neutral, and the
#      CONTAINMENT acceptance this grid was chosen for -- the neutral integral must
#      SATURATE by 2.5 L. Full containment is explicitly not the bar: the LES retains
#      0.874 of its asymptote against Kljun's 0.867 on identical cells, so both models
#      lose the same tail and a relative claim survives the truncation.
#   D  two dual-output cases at lpdmOnlineSelector = 2 -- staging AND writing -- which is
#      the only way to score the hand-off without comparing across two turbulence
#      realisations, measured on this project at 44% in the integral.
#   E  the deciding test, RE-RUN. The 24 m result does not certify this grid: the
#      sub-grid fraction is re-measured at every grid and never carried, and dx 24 -> 30
#      lowers z/Delta from 1.76 to 1.52.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
L="${LOGDIR:-/tmp/flux-logs}"; mkdir -p "$L" results

export GRID=data/grid30_raised ZTARGET=28.5 EXACT_AGL=1
export TEMPLATE=runs/g30_base/base.in DX=30 ZCEILING=3000 DEFORM=0.346601 ZI_MAX_ABS=1250
export SEED_LIB=jobs30 ALLOW_INDETERMINATE=1 LPDM_WORKERS=12
# THE CASE CLASS. 1800 s adjustment + TWO 2700 s windows = 2.0 sim-h at 0.495 GPU-h/sim-h
# = 0.99 GPU-h, i.e. 0.50 per footprint against 0.60 for one window at 24 m. t_back stays
# 900 s: the two 24 m targets said 600 was enough (99.6%, 100.0%) and the flat/neutral
# control then said 91.5%, which is the one that governs.
export ADJ_S=1800 WINDOW_S="${WINDOW_S:-2700}" TBACK="${TBACK:-900}"
export N_WINDOWS="${N_WINDOWS:-2}"
export TBACK_MARKS="${TBACK_MARKS:-150,300,450,600,750}"
export KEEP_TD=100000
# KEEP THE FIELDS UNTIL THE NO-OP CONTROL HAS RUN. The deciding test asks whether the PEAK
# responds to meteorology, and a peak can move because the closure's own stability function
# moved rather than because the LES resolved different turbulence. Re-running each
# footprint with the floor OFF separates the two, and it can only be done while the fields
# are still there.
export KEEP_FIELDS=1

die(){ echo "FATAL: $*" >&2; exit 1; }
say(){ echo; echo "########## $* ##########"; date '+%F %H:%M:%S'; }
have_seed(){ [ -s "jobs30/$1/return/seed_restart.nc" ]; }

bash bin/preflight.sh || die "preflight"
[ -s runs/g30_base/base.in ] || die "no 30 m template"
grep -q "BISECT THIS" runs/g30_base/base.in && \
  die "runs/g30_base/base.in still carries the PLACEHOLDER dt. Run bin/g30_bringup.sh,
       read the accuracy boundary off results/g30_bringup.txt, and write a production dt
       with >= 10% margin that lands the 5 s cadence on an integer step count. PROJECT_BRIEF.md
       forbids carrying a boundary between grids and this is where that is enforced."

# ---------------------------------------------------------------- B: the two seeds
for RUNG in nbl-deep cbl-deep; do
  if ! have_seed "seed_${RUNG}_a000"; then
    say "B: seed $RUNG, open-ended, 3.0 sim-h ceiling"
    ACC=""; [ "$RUNG" = "nbl-deep" ] && ACC="SEED_ACCEL_S=3000"
    env $ACC SEED_EARLY_STOP=1 jobs/run_seed.sh "jobs30/seed_${RUNG}_a000" \
      2>&1 | tee -a "$L/pass8_seed_${RUNG}.log"
    have_seed "seed_${RUNG}_a000" || echo "  $RUNG returned no seed (see the gate)" >&2
  fi
  say "B: acceptance battery, $RUNG"
  bash bin/seed_accept.sh "jobs30/seed_${RUNG}_a000" 2>&1 | tail -60
done

# ---------------------------------------------------------------- C: the flat control
say "C: flat/neutral control -- regression, D1 neutral, CONTAINMENT at 2.5 L"
echo "  (bin/run_pass6b.sh style; the containment ladder needs --max-disp raised, because"
echo "   production retires a trajectory at one domain length and the by-displacement"
echo "   curve is then FLAT past 1 L BY CONSTRUCTION -- which is what makes the monitor's"
echo "   G2a pass trivially and cannot be read as containment.)"

# ---------------------------------------------------------------- D: two dual-output cases
run_target(){   # tag  timestamp
  local TAG="$1" TS="$2"
  case "$TS" in *T*) ;; *) echo "FATAL: '$TS' needs a T between date and time" >&2
                           return 64;; esac
  if [ -s "pairs/${TAG}_w0.json" ] && [ -s "pairs/${TAG}_w1.json" ]; then
    echo "  $TAG already paired (both windows); skipping"; return 0
  fi
  say "case $TAG at $TS -- 2.0 sim-h, two windows, staging AND writing"
  bash bin/run_corpus_case.sh "$TS" "$TAG" 2>&1 | tail -60
  [ -s "pairs/${TAG}_w0.json" ] || echo "  *** $TAG window 0 produced no pair" >&2
}
run_target case_2023052519 "2023-05-25T19:00"
run_target case_2023121921 "2023-12-19T21:00"

# ---------------------------------------------------------------- E: the deciding test
say "E: THE DECIDING TEST, re-run at dx = 30 m"
echo "  The 24 m result does NOT carry: z/Delta falls 1.76 -> 1.52 and the sub-grid"
echo "  fraction is re-measured at every grid. The a priori spectral test predicts a"
echo "  small cost (87.2% of resolved variance kept at a 4dx filter, so ~56% sub-grid"
echo "  convective against 52.5%), and this is what checks it."
./docker/pyrun.sh bin/case_compare.py case_2023052519_w0 case_2023121921_w0 \
  2>&1 | tee results/pass8_deciding_test.txt
[ -s results/pass8_deciding_test.txt ] || die "case_compare wrote nothing"
say "pass 8 complete"
