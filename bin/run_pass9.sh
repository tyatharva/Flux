#!/usr/bin/env bash
# NINTH PASS: two seeds, two targets, one flat control, all at 122^3 @ 30 m.
#
# ORDER, AND WHY IT IS THIS ORDER.
#   CONVECTIVE FIRST. If the convective half fails, the neutral GPU-hours are never spent.
#   NOTHING CPU-SIDE BLOCKS A GPU RUN. Every analysis that can be done on data already
#     produced is launched in the BACKGROUND and runs while the next LES is on the GPU
#     (PROJECT_BRIEF.md's standing rule). The LPDM, the acceptance comparisons, the containment
#     ladder and the well-mixed battery are all CPU; only the LES needs the card.
#     THE ONE EXCEPTION IS bin/seed_accept.sh, whose Gate C2 restarts the seed and re-dumps
#     it -- that IS an LES. Backgrounding it raced run 2 into docker/run_case.sh's
#     one-FastEddy-at-a-time refusal and killed the case before a single step ran, so the
#     battery runs in the FOREGROUND and wait_for_gpu guards every launch besides.
#   ASSERT ON THE ARTIFACT. Every step checks the file it was supposed to write, never an
#     exit status -- the analyses are piped into grep and tee (docs/FASTEDDY_TRAPS.md 12).
#
# THE TWO DATETIMES, and why these:
#   CONVECTIVE 2023-11-17T18:00Z -- z_i 681 m (mid-band, and 19 m from the cbl-mid rung's
#     own 700 m target, the smallest seed gap available), SHTFL 179 W/m2, |dz_i/dt| 13.1
#     %/h, wind FROM 331 deg. Northerly matters: at a 30 m receptor Kljun's array share is
#     30.7% for a northerly footprint and 0.04% for an easterly one, so the array signal
#     this site exists to measure is only present in the northern sector. pick_seed lands
#     it on seed_cbl-mid_a015 rot 3 with a direction gap of 1.0 deg.
#   NEUTRAL 2023-11-21T20:00Z -- SHTFL 1 W/m2 (z/L = -0.005, the most neutral hour in the
#     table), dz_i/dt +0.8 %/h (the most stationary), z_i 711 m, wind FROM 326 deg, so the
#     same northern sector as the convective case and therefore comparable on the array.
#     seed_nbl-deep_a015 rot 3, gap 4.4 deg.
#   Both are inside HRRR v4 (from 2020-12-02). Selected from results/candidates.tsv, which
#   is already built, so choosing them cost no network and no GPU.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
L="${LOGDIR:-/tmp/flux-logs}"; mkdir -p "$L" results/pass9

# ---- the 30 m configuration, and every one of these has to travel together -------------
export GRID=data/grid30_raised ZTARGET=28.5 EXACT_AGL=1
export TEMPLATE=runs/g30_base/base.in DX=30 ZCEILING=3000 DEFORM=0.346601 ZI_MAX_ABS=1250
export SEED_LIB=jobs30 ALLOW_INDETERMINATE=1 LPDM_WORKERS=12
export ADJ_S=1800 WINDOW_S=2700 TBACK=900 N_WINDOWS=2
export KEEP_TD=100000 KEEP_FIELDS=1 COVER_GROUPS=10
export RING=1 RING_SELECTOR=2          # stage AND write, so both paths come from one run
export NPZ_DIR=pairs_npz
DT30=0.0308642

CONV_TS="2023-11-17T18:00"; CONV_TAG=case_2023111718; CONV_SEED=seed_cbl-mid_a015
NEUT_TS="2023-11-21T20:00"; NEUT_TAG=case_2023112120; NEUT_SEED=seed_nbl-deep_a015

die(){ echo "FATAL: $*" >&2; exit 1; }
say(){ echo; echo "########## $* ##########"; date '+%F %H:%M:%S'; }
have_seed(){ [ -s "jobs30/$1/return/seed_restart.nc" ]; }
BG_PIDS=""
bg(){ "$@" & BG_PIDS="$BG_PIDS $!"; echo "  [bg $!] $*"; }
# ONE FastEddy AT A TIME, AND WAIT FOR IT RATHER THAN COLLIDING WITH IT.
# docker/run_case.sh REFUSES a second FastEddy container, which is correct and is also a
# hard failure for whoever loses the race -- measured: bin/seed_accept.sh's Gate C2 needs
# the GPU too (it restarts the seed and re-dumps), so backgrounding the battery raced run 2
# into a refusal that killed the case before a single step ran. seed_accept is now in the
# FOREGROUND, and this is the belt to that braces: nothing launches an LES while one is up.
wait_for_gpu(){
  local t=0
  while [ -n "$(docker ps -q --filter ancestor=flux-fasteddy:cuda118)" ]; do
    [ $((t % 60)) -eq 0 ] && echo "  waiting for the GPU: $(docker ps --format '{{.Names}}' \
      --filter ancestor=flux-fasteddy:cuda118 | tr '\n' ' ')"
    sleep 10; t=$((t + 10))
    [ "$t" -gt 7200 ] && die "a FastEddy container has held the GPU for 2 h; not queueing behind it"
  done
}

bash bin/preflight.sh || die "preflight"

# ---- 1. the convective seed, 1.0 sim-h HARD ceiling ------------------------------------
if ! have_seed "$CONV_SEED"; then
  say "RUN 1: convective seed $CONV_SEED, 1.0 sim-h ceiling"
  SEED_CEILING_H=1.0 SEED_EARLY_STOP=1 bash jobs/run_seed.sh "jobs30/$CONV_SEED" \
    2>&1 | tee "$L/pass9_run1.log" | tail -30
  have_seed "$CONV_SEED" || die "run 1 returned no seed"
else
  echo "  run 1: $CONV_SEED already returned a seed; skipping"
fi
wait_for_gpu
say "RUN 1 battery: $CONV_SEED (FOREGROUND -- Gate C2 needs the GPU)"
bash bin/seed_accept.sh "jobs30/$CONV_SEED" 2>&1 | tee "$L/pass9_accept1.log" | tail -25

# ---- 2. the convective target, two windows, through the ring ---------------------------
say "RUN 2: convective target $CONV_TAG at $CONV_TS"
wait_for_gpu
bash bin/run_corpus_case.sh "$CONV_TS" "$CONV_TAG" 2>&1 | tee "$L/pass9_run2.log" | tail -40
[ -s "pairs/${CONV_TAG}_w0.json" ] || die "run 2 produced no w0 pair"
[ -s "pairs/${CONV_TAG}_w1.json" ] || echo "  *** run 2 produced no w1 pair" >&2

# The acceptance, and the well-mixed battery, on the dumps selector 2 wrote from the very
# buffers the ring consumed. Both are CPU and both run while run 3 is on the card.
bg bash bin/pass9_accept.sh "$CONV_TAG" "$DT30" convective

# ---- 3. the neutral seed, 3.0 sim-h ceiling, with the accelerator ----------------------
# Steinfeld's spin-up accelerator: 3000 s at surflayer_wth = +0.05 to trip the transition
# to resolved turbulence, then a restart with htFlux ZEROED IN THE FILE, because htFlux is
# IO-registered and the .in cannot override it (docs/FASTEDDY_TRAPS.md 17).
if ! have_seed "$NEUT_SEED"; then
  say "RUN 3: neutral seed $NEUT_SEED, 3.0 sim-h ceiling, accelerator on"
  SEED_CEILING_H=3.0 SEED_ACCEL_S=3000 SEED_EARLY_STOP=1 \
    bash jobs/run_seed.sh "jobs30/$NEUT_SEED" 2>&1 | tee "$L/pass9_run3.log" | tail -30
  have_seed "$NEUT_SEED" || die "run 3 returned no seed"
else
  echo "  run 3: $NEUT_SEED already returned a seed; skipping"
fi
wait_for_gpu
say "RUN 3 battery: $NEUT_SEED (FOREGROUND -- Gate C2 needs the GPU)"
bash bin/seed_accept.sh "jobs30/$NEUT_SEED" 2>&1 | tee "$L/pass9_accept3.log" | tail -25

# ---- 4. the flat/neutral control: regression, D1-neutral, CONTAINMENT at 2.5 L ----------
# THE ONLY PLACE KLJUN IS DIAGNOSTIC, and the case the containment gate is decided on -- at
# 2928 m the flat/neutral integral needed 1.5 domain lengths to stop growing while the
# convective target saturated by 1. It runs on the DISK path: it needs its fields kept for
# the by-displacement ladder, the well-mixed battery and the closure controls, so staging
# them as well would buy nothing.
say "RUN 4: flat/neutral control off $NEUT_SEED"
mkdir -p runs/g30_flat
UGV=$(python3 -c "
import re;s=open('jobs30/$NEUT_SEED/seed.in').read()
print(re.search(r'^U_g = ([-0-9.]+)',s,re.M).group(1),
      re.search(r'^V_g = ([-0-9.]+)',s,re.M).group(1))")
read -r FUG FVG <<< "$UGV"
echo "  flat control forcing from the seed's own .in: U_g $FUG  V_g $FVG"
wait_for_gpu
TBACK=900 DT="$DT30" SRC="jobs30/$NEUT_SEED/return/seed_restart.nc" D=runs/g30_flat \
GRID=data/grid30 TAG=g30_flat BASE=runs/g30_base/base.in ZTARGET=28.5 \
UG="$FUG" VG="$FVG" MARKS=150,300,450,600,750 KEEP_FIELDS=1 \
REFJSON=results/regression_baseline_g30.json TAGJSON=results/g30_flat.json \
KEEPTD=100000 LPDM_WORKERS=12 \
  bash bin/regression_flat.sh 2>&1 | tee "$L/pass9_run4.log" | tail -30
[ -s results/g30_flat.json ] || die "run 4 produced no flat-control footprint"
bg bash bin/pass9_flat.sh "$DT30"

# ---- 5. the neutral target, two windows, through the ring ------------------------------
# THE NEUTRAL SEED IS DRIFTING IN z_i AND pick_seed REFUSES IT BY DEFAULT.
# Measured on seed_nbl-deep_a015 at 2.917 sim-h: +5.76 %/h against a 3 %/h limit. That is
# not a surprise -- docs/PLAN.md item 0aa predicted it from a scoring-window sweep ("z_i ...
# TRENDING AWAY from band ... a longer run resolves these into a FAIL, not a pass") -- and
# it means no affordable neutral spin-up will ever satisfy the limit, because a neutral
# Ekman layer's depth keeps growing for several inertial periods.
#
# The directive this pass runs under is explicit that a threshold not met is a MEASUREMENT
# and not a failure, so the neutral target is built and the state is recorded rather than
# the data being dropped: gate_state = DRIFTING is stamped on both pairs and a warning goes
# into the training record. THE CORPUS DRIVER DOES NOT SET THIS. Whether the corpus should
# is a design decision for the user, and docs/results/NINTH_PASS_RESULTS.md puts the numbers next to it.
export ALLOW_DRIFTING=1
say "RUN 5: neutral target $NEUT_TAG at $NEUT_TS (seed is DRIFTING in z_i; see above)"
wait_for_gpu
bash bin/run_corpus_case.sh "$NEUT_TS" "$NEUT_TAG" 2>&1 | tee "$L/pass9_run5.log" | tail -40
[ -s "pairs/${NEUT_TAG}_w0.json" ] || die "run 5 produced no w0 pair"
bg bash bin/pass9_accept.sh "$NEUT_TAG" "$DT30" neutral

say "waiting for the background analyses"
for p in $BG_PIDS; do wait "$p" 2>/dev/null; done
say "pass 9 GPU complete; see results/pass9/"
