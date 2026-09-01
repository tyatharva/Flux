#!/usr/bin/env bash
# ONE corpus case, end to end: a timestamp in, an (input, target) training pair out.
#
#   usage: bin/run_corpus_case.sh 2023-01-18T18:00 [tag]
#   env:   WINDOW_S=2700  ADJ_S=1800  TBACK=900  KEEP_FIELDS=0  COVER_GROUPS=10
#          -- COVER_GROUPS is the number of independent release groups the array-share
#             standard error is estimated from. The default in stage5_footprint.py is 2,
#             which is ONE difference and therefore ~one degree of freedom; PROJECT_BRIEF.md's
#             standing rule is N >= 8 wherever a share or a shape is being tested, and
#             Phase E measured a FACTOR OF 5 in the estimated floor between a 2-group and a
#             10-group split. The split costs nothing -- the touchdowns are already
#             labelled by release time -- so the corpus takes 10.
#          -- ADJ_S + WINDOW_S = 4500 s = 1.25 sim-h is the class length at the 30 m
#             production geometry, and the footprint is the LAST 30 MINUTES of it.
#          SKIP_LES=1     stop after stage 4, for a dry run with no GPU
#          SEED_LIB=jobs30  where the spun-up seed library lives
#          SEED_ANY=1     rank seeds with no returned artifact too (planning only)
#          ALLOW_INDETERMINATE=0  require ESTABLISHED stationarity (default 1;
#                         no seed in the library can supply it -- see stage 4)
#          GRID=data/grid30_raised  ZTARGET=28.5  EXACT_AGL=1  (the DEFAULTS; see below)
#
# The eight stages, and which file owns each:
#
#   1  HRRR pseudo-sounding at the tower        bin/hrrr_sounding.py
#   2  sounding -> FastEddy .in parameters      bin/sounding_to_forcing.py
#   3  this case's surface heat-flux map        bin/case_surface.py
#   4  which seed, and which 90-degree rotation bin/pick_seed.py
#   5  rotate the seed's FLOW, inject the       bin/prep_restart.py
#      STATIC surface (terrain, z0, htFlux)
#   6  30 min adjustment + (30 min + t_back) window,
#      as ONE continuous invocation                bin/run_window.sh
#   7  backward LPDM -> 122 x 122 footprint     bin/stage5_footprint.py
#   8  assemble the training record             bin/make_pair.py
#
# THE ADJUSTMENT IS THE POINT OF THE WHOLE SEED LIBRARY. A cold start would need 3
# simulated hours here; restarting from a seed of the right regime, depth and heading
# needs 30 minutes. Across ~1825 cases that is the difference between ~2000 and ~7600 GPU-h.
#
# ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS at every stage (docs/FASTEDDY_TRAPS.md 12): the
# analyses are piped into grep, so bash reports GREP's status and a python traceback lands
# quietly in a redirected .txt. Each step below checks the file it was supposed to write.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FLUX_ROOT="$ROOT"
cd "$ROOT"

TS="${1:-}"
[ -n "$TS" ] || { echo "usage: run_corpus_case.sh <YYYY-MM-DDTHH:MM> [tag]" >&2; exit 64; }
# Pure bash, because `tr -d '-:'` parses `-:` as an OPTION BUNDLE and fails -- and the
# failure went to stderr, which stage 1's grep filter swallowed, so the tag silently came
# out as the empty string and every artifact landed on top of the last case's. Same class
# as docs/FASTEDDY_TRAPS.md 12: a filtered stream is a hidden error.
_t="${TS//[-:]/}"; _t="${_t/T/}"
TAG="${2:-case_${_t:0:10}}"
[[ "$TAG" =~ ^[A-Za-z0-9_]+$ ]] || { echo "FATAL: bad tag '$TAG' from '$TS'" >&2; exit 65; }
[ "${#TAG}" -ge 8 ] || { echo "FATAL: tag '$TAG' too short; is '$TS' a valid timestamp?" >&2
                         exit 65; }
# THE RAISED SURFACE IS PRODUCTION, settled by the sixth pass (docs/results/SIXTH_PASS_RESULTS.md):
# topoPos is raised by the displacement height over the array so the first model level
# clears panel top, z0_array goes 0.10 -> 0.25 (which is the only thing that gives the
# array ANY neutral signal -- at 0.10 it is aerodynamically identical to the cropland
# WorldCover labels it as), and the receptor is released at a FRACTIONAL level 8.500 m
# above the RAISED surface = 10.000 m above bare ground. Snapping to the nearest level
# there would put the receptor 10 m above the PANELS, an 11.5 m receptor and a 15% error
# in exactly the quantity this pass exists to get right.
# === THE DEFAULTS ARE THE 30 m PRODUCTION GEOMETRY -- changed 2026-08-31 ===============
# They were the retired 16 m ones, which meant every corpus entry point had to export the
# whole block around them and a caller that forgot would get a complete, plausible case on
# the wrong grid. Every existing caller (run_pass7/8/9, run_corpus.sh) sets the block
# explicitly and is unaffected; what changes is that FORGETTING now yields production
# rather than a retired configuration. The retired geometries are still reachable and the
# pass drivers still name them in full.
GRID="${GRID:-data/grid30_raised}"
ZTARGET="${ZTARGET:-28.5}"
EXACT_AGL="${EXACT_AGL:-1}"
# GRID GEOMETRY, so one driver serves both configurations. The seventh pass runs
#   GRID=data/grid24_raised ZTARGET=28.5 TEMPLATE=runs/g24_base/base.in DX=24
#   ZCEILING=3000 DEFORM=0.346601 ZI_MAX_ABS=1250 SEED_LIB=jobs24
# and every one of those has to travel together: a 24 m case built against the 16 m
# template would carry the wrong d_zeta and the wrong dt and would still run.
TEMPLATE="${TEMPLATE:-runs/g30_base/base.in}"
DX="${DX:-30.0}"; ZCEILING="${ZCEILING:-3000.0}"; DEFORM="${DEFORM:-0.346601}"
ZI_MAX_ABS="${ZI_MAX_ABS:-1250}"
KEEP_TD="${KEEP_TD:-100000}"
ADJ_S="${ADJ_S:-1800}"
# 2700 s = 900 s t_back + 1800 s of releases, the 30 m production window. With ADJ_S that
# makes a case 4500 s = 1.25 sim-h.
WINDOW_S="${WINDOW_S:-2700}"
# === t_back IS A PROPERTY OF THE GRID AND THE RECEPTOR, SO THE FILE IS KEYED BY dx =======
# This read `results/tback_production.txt` unconditionally. That file holds **600**, the
# 16 m measurement, and it would have silently governed every 30 m case the moment a driver
# stopped exporting TBACK -- a stale artifact quietly overriding a production default, which
# is the exact failure class this project keeps paying for. Keyed by dx it cannot: a 16 m
# measurement can only govern a 16 m run.
#
# 900 s is the 30 m production value. Note it is deliberately LONGER than the 600 s the
# capture curve measured as sufficient at this receptor (PROJECT_BRIEF.md: 99.6% and 100.0% of the
# 900 s integral is in by 600 s) -- production kept the fourth pass's 900 and the 6.7% it
# costs is recorded rather than taken.
_TBF="results/tback_production_dx${DX%.*}.txt"
TBACK="${TBACK:-$(cat "$_TBF" 2>/dev/null || echo 900)}"
# ONE WINDOW PER CASE IS THE CORPUS DEFAULT (2026-08-30). Named here rather than left to
# ${N_WINDOWS:-1} at each use site, so the corpus's case length is one readable number.
# N_WINDOWS=2 is still supported and still validated -- see the schedule comment below.
N_WINDOWS="${N_WINDOWS:-1}"
# REL_S is the RELEASE period, 1800 s = the EC averaging period by definition.
# It is an env var only so a smoke run can exercise the machinery on a short
# window; a production case never sets it.
REL_S="${REL_S:-1800}"
D="runs/$TAG"
L="${LOGDIR:-${TMPDIR:-/tmp}/flux-logs}"; mkdir -p "$L"
die(){ echo "FATAL: $*" >&2; exit 1; }
say(){ echo; echo "########## $* ##########"; date '+%F %H:%M:%S'; }

mkdir -p "$D/output" "$D/window" results/soundings results/forcing results/pick pairs

# ---- 1. the sounding -------------------------------------------------------------
say "$TAG  stage 1: HRRR sounding at $TS"
SND=results/soundings/$TAG.json
[ -s "$SND" ] || ./docker/pyrun.sh bin/hrrr_sounding.py "$TS" --out "$SND" \
    2>&1 | grep -vE 'Found ┊|BallTree|^INFO:|│|╭|╰'
[ -s "$SND" ] || die "stage 1 wrote no sounding"

# ---- 2. the forcing --------------------------------------------------------------
say "$TAG  stage 2: sounding -> .in"
FRC=results/forcing/$TAG.json
./docker/pyrun.sh bin/sounding_to_forcing.py "$SND" --out "$FRC" --grid "$GRID" \
    --template "$TEMPLATE" --dx "$DX" --zceiling "$ZCEILING" --deform "$DEFORM" \
    ${ZI_MAX_ABS:+--zi-max-abs $ZI_MAX_ABS} \
    --in-out "$D/case.in" || true
[ -s "$FRC" ] || die "stage 2 wrote no forcing json"
[ -s "$D/case.in" ] || die "stage 2 wrote no .in"
# REFUSE A CASE THE DOMAIN CANNOT HOLD, rather than running it and mis-labelling it.
# z_i outside 300-1250 m is not representable: below it the 30 m receptor leaves the
# surface layer, above it the 3660 m box cannot hold the boundary layer at L >= 2 z_i.
REPR=$(python3 -c "import json;print(json.load(open('$FRC')).get('representable'))")
if [ "$REPR" != "True" ]; then
  echo "  SKIPPED: not representable --"
  python3 -c "import json;[print('   ',w) for w in json.load(open('$FRC'))['warnings']]"
  exit 3
fi
read -r UG VG DT WTHREF < <(python3 -c "
import json; d=json.load(open('$FRC')); p=d['params']
print(p['U_g'], p['V_g'], p['dt'], d['labels']['wth_cropland_reference'])")

# ---- 3. this case's surface ------------------------------------------------------
# NOT OPTIONAL, and its absence is silent. prep_restart.py injects htFlux from the grid
# directory into the restart file, and the restart read OVERWRITES the .in's scalar
# (PROJECT_BRIEF.md, the Stage 6 lever). data/grid16 ships with htFlux ALL ZEROS, so a convective
# case pointed at it runs NEUTRAL, exits 0, and says nothing. The static geography is
# hardlinked, so this costs ~116 kB and no copy.
say "$TAG  stage 3: per-case surface flux map"
CG="data/case_grids/$TAG"
./docker/pyrun.sh bin/case_surface.py --grid "$GRID" --wth-ref "$WTHREF" --out "$CG" \
    || die "case_surface"
[ -s "$CG/htFlux.npy" ] || die "stage 3 wrote no htFlux map"
GRID="$CG"

# ---- 4. the seed -----------------------------------------------------------------
say "$TAG  stage 4: pick a seed"
PICK=results/pick/$TAG.json
# --available-only BY DEFAULT, because this driver is about to RESTART from the chosen
# seed. Ranking a seed that has no returned artifact is never right here: the pick would
# name a file that does not exist and the case would stop at the guard below, and an
# unbuilt seed's heading is an ESTIMATE (geostrophic angle minus a nominal Ekman backing)
# being compared against a spun seed's MEASURED one. With a complete library the flag is a
# no-op. SEED_ANY=1 restores the full-library ranking for planning.
./docker/pyrun.sh bin/pick_seed.py "$FRC" --json "$PICK" --zm "$ZTARGET" \
    --library "${SEED_LIB:-jobs30}" --index "${SEED_LIB:-jobs30}/index.json" \
    $([ "${SEED_ANY:-0}" = "1" ] || echo --available-only) \
    $([ "${ALLOW_INDETERMINATE:-1}" = "1" ] && echo --allow-indeterminate) \
    --allow-drifting "$(case "${ALLOW_DRIFTING:-any}" in
                          1) echo any;; 0) echo off;; *) echo "${ALLOW_DRIFTING:-any}";;
                        esac)" \
    ${SEED_EXCLUDE:+--exclude "$SEED_EXCLUDE"} || true
# === ALLOW_INDETERMINATE IS ON BY DEFAULT, BECAUSE INDETERMINATE IS THE LIBRARY'S
# === NORMAL STATE AND NOT AN EXCEPTION ============================================
# Two of the seven stationarity limits -- TKE_BL/u*^2 and z_i -- cannot be resolved
# against their own thresholds in a 3.0 h spin-up, at ANY scoring window. They decorrelate
# on the EDDY TURNOVER (h/u* = 1258-1345 s), not on the 300 s dump interval, so n_eff
# saturates at 3-5 from a 1.0 h window to a 2.5 h one. Dumping more often cannot help;
# the RUN is what is short. Every seed in this library is therefore expected to return
# INDETERMINATE on those two, and refusing them by default would refuse the whole library.
#
# What this does NOT do is call them PASS. The verdict stays INDETERMINATE, the thresholds
# are untouched, seed.gate_state = INDETERMINATE is stamped onto every pair, and
# make_pair.py writes a warning into the training record. Set ALLOW_INDETERMINATE=0 to
# require established stationarity, which today no seed in the library can supply.
#
# === ALLOW_DRIFTING DEFAULTS TO `any` -- CHANGED 2026-08-31, AND THAT IS THE WHOLE ====
# === LIBRARY. The paragraphs below are the 2026-08-30 `zi-neutral` reasoning, kept  ====
# === because the DECISION superseded them but the ARGUMENT is what generalised.     ====
#
# THE GENERALISATION: a seed is an INITIAL CONDITION, not a corpus point. The case restarts
# from it, integrates ADJ_S under its OWN sounding's forcing, and every ML input is then
# measured by window_stats over exactly the same window as the footprint. So the pair is
# self-consistent whatever the seed's drift state -- which is precisely the argument the
# z_i concession below was granted on, and it never depended on the limit being z_i.
# Refusing a seed therefore removes a RESTART POINT without removing any error.
#
# MEASURED COST OF THE NARROW FORM on the 30-seed library (docs/results/SEED_LIBRARY_RESULT.md): it
# admitted 11 of 30. It took out ALL SIX cbl-shallow seeds, leaving the weakly-convective
# rung with no restart point; eight of twelve neutral seeds, dropping the neutral half to
# four base angles and firing pick_seed's own half-spacing warning at 14.5 deg against a
# 15 deg spacing; and the Ekman-backing calibration to n = 1 and 2 on three of five rungs.
# And the seeds it kept were not the steady ones -- the accept/refuse split tracked the
# trend's STANDARD ERROR, not the trend, so cbl-shallow_a000 at +23.5 %/h was admitted
# while a030 at +22.0 %/h was refused for being measured three times more precisely.
#
# `ALLOW_DRIFTING=zi-neutral` restores the narrow 2026-08-30 form and `off` the original
# refusal; 1 and 0 still mean any and off.
#
# --- the 2026-08-30 reasoning, superseded in scope and correct in substance -----------
# ALLOW_DRIFTING was OFF, and OFF makes the NEUTRAL HALF OF THE CORPUS UNBUILDABLE. z_i in a neutral
# boundary layer with no capping inversion grows without bound -- measured on
# seed_nbl-deep_a015 at +5.76 %/h and still climbing at 3.0 sim-h -- so the z_i stationarity
# limit is UNSATISFIABLE on those rungs rather than failed, and no affordable spin-up will
# ever satisfy it. The old fallback was worse than the refusal: pick_seed would hand a
# neutral case a CONVECTIVE seed, which this driver itself warns about.
#
# `zi-neutral` is the NARROW form: a neutral rung whose ONLY drifting limit is z_i. A
# neutral seed drifting in u*, sigma_w or a Kljun geometry term is still refused, and so is
# a convective seed drifting in z_i -- there a capping inversion and subsidence are holding
# the depth, so drift is a defect. `ALLOW_DRIFTING=any` restores the wide manual opt-in and
# `ALLOW_DRIFTING=off` the old refusal; 1 and 0 still mean any and off.
#
# THE PAIR STAYS VALID and that is the whole basis for the decision: the inputs come from
# window_stats over the same 30 minutes as the footprint, not from the seed. Letting z_i
# grow to a FIXED simulated-time ceiling is deterministic and reproducible, and z_i is a
# weak input at a 30 m receptor -- Kljun's 1/(1 - z_m/h) channel spans ~5% over 400-1200 m.
# NO CAPPING INVERSION WAS ADDED, deliberately. Every such pair carries
# seed.gate_state = DRIFTING, meta.zi_accepted_drifting = true and meta.zi_achieved_m, so
# the achieved distribution can be checked before training; if it turns out too narrow, a
# per-case lid from the sounding is the fallback.
[ -s "$PICK" ] || die "stage 4 wrote no pick json"
read -r JOB ROT < <(python3 -c "
import json; c=json.load(open('$PICK'))['chosen']; print(c['job'], c['rot'])")
SEED="${SEED_LIB:-jobs30}/$JOB/return/seed_restart.nc"
[ "${SKIP_LES:-0}" = "1" ] && { echo "  SKIP_LES=1: stopping after stage 4"; exit 0; }
[ -f "$SEED" ] || die "seed $SEED has not been spun up yet; run jobs/run_seed.sh $JOB"

# ---- 5. rotate the flow, inject the static surface -------------------------------
say "$TAG  stage 5: seed $JOB rot $ROT -> restart"
./docker/pyrun.sh bin/prep_restart.py "$SEED" "$D/FE_RST.0" --rot "$ROT" --grid "$GRID" \
    || die "prep_restart"
[ -s "$D/FE_RST.0" ] || die "prep_restart wrote no restart"
cp -f "$GRID/topo.bin" "$D/topo.bin" || die "topo.bin"

# ---- 6. adjustment AND window, ONE CONTINUOUS INVOCATION --------------------------
#
# These were two FastEddy runs -- an adjustment, then a restart from its final dump into
# the window. That restart is gone. CHAINING IS RETIRED (2026-08-26) and the only restart
# left in the project is seed -> target, which happens at stage 5 above. What it buys is
# docs/FASTEDDY_TRAPS.md 17 removed structurally rather than by assertion: a restart READ
# overwrites every IO-registered field, so each restart is an opportunity to silently
# inherit state the .in does not describe.
#
# THE SCHEDULE, which is what makes the timeline correct rather than merely shorter.
#
# === ONE WINDOW PER CASE IS THE CORPUS DEFAULT -- set 2026-08-30 ======================
# A case is 1.25 SIMULATED HOURS and the footprint is the LAST 30 MINUTES of it, stamped at
# T. At the production geometry (ADJ_S 1800, WINDOW_S 2700, TBACK 900), for a footprint
# stamped 01:00 UTC covering 00:30-01:00:
#
#     T - 1.25 h   23:45  restart from the seed; adjustment begins    step 0
#     T - 0.75 h   00:15  adjustment complete (ADJ_S = 1800 s)        step A_NT
#                         -- field staging begins; nothing before this survives
#     T - 0.50 h   00:30  first release (needs t_back = 900 s back)   step A_NT + t_back
#     T            01:00  last release; window closes                 step TOT_NT
#
# so the run is ADJ_S + TBACK + REL_S = 1800 + 900 + 1800 = 4500 s = 1.25 sim-h, and the
# earliest field a backward trajectory can reach is EXACTLY the adjustment end. That is
# enforced TWICE and neither is decorative: run_window.sh SKIP_S deletes the adjustment's
# dumps and asserts the earliest survivor is step A_NT, and stage5_footprint.py --t-min
# refuses anything earlier independently.
#
# WHY THE SECOND WINDOW WAS CUT. Two windows per case were validated and kept as an option
# (N_WINDOWS=2), and they ARE independent turbulence draws -- the 20 release groups
# decorrelate in 180 s against an 1800 s separation. But MEASURED on both ninth-pass
# validation cases, the two windows are near-duplicates in SHAPE (median |w0 - w1| / the
# within-footprint half-vs-half floor = 0.19 and 0.33, where two independent draws would
# give ~sqrt2), and shape is what the FNO learns. 1.25 sim-h buying one distinct condition
# beats 2.0 sim-h buying one condition plus a near-copy. N_WINDOWS=2 stays behind the flag
# because a model estimating SPREAD at fixed conditions would want exactly those replicates.
#
# FastEddy has a single frqOutput and writes from step 0, so the adjustment's 360 dumps
# are produced and then deleted; run_window.sh does the deleting and then ASSERTS that the
# earliest surviving dump is step A_NT. Without that, lpdm/fields.py:dump_series would glob
# the whole directory and compute_footprint would release from t[0] + t_back -- putting the
# entire 30-minute averaging period inside the adjustment, on fields still settling, with
# nothing in the output to say so.
say "$TAG  stage 6: ${ADJ_S}s adjustment + ${N_WINDOWS:-1} x ${WINDOW_S}s window, one invocation"
A_NT=$(python3 -c "
frq=int(round(5.0/$DT)); print(int(round($ADJ_S/$DT/frq))*frq)")
[ "$A_NT" -gt 0 ] 2>/dev/null || die "adjustment step count did not compute"
# N_WINDOWS x WINDOW_S, not one: the sampling windows run back to back inside this one
# invocation, so the LES has to be long enough for all of them. Getting this wrong would
# not error -- the last window's --t-max would simply select fields that were never
# written, and stage 7 would refuse with "leaves no fields", which is the good case. The
# bad case is N_WINDOWS=1 arithmetic silently truncating a two-window case to one.
# FROM THE DUMP-ALIGNED ADJUSTMENT END, NOT FROM ADJ_S. A_NT is ADJ_S rounded UP to a
# whole number of output intervals, so it exceeds ADJ_S by up to one interval (101 s at
# this grid). The single-window driver absorbed that silently -- its window simply came out
# 101 s short and compute_footprint took t_last from the dumps that existed. With two
# windows it cannot be absorbed: window 1 would be asked for fields past the end of the run
# and would come up short at exactly the end, where the releases are.
# ---- THE WINDOW SCHEDULE, AND WHY CONSECUTIVE WINDOWS ARE ONE OUTPUT INTERVAL APART ----
#
# Naively, window WI spans [A_NT + WI*W_NT, A_NT + (WI+1)*W_NT] and consecutive windows
# SHARE their boundary dump. On the disk path that is harmless -- nothing is deleted, so
# both windows can read it. **Through the ring it is not possible**: the consumer deletes
# each snapshot as it reads it, which is what releases the producer's backpressure, so
# window 0 consumes the boundary and window 1 starts one interval late. Its field span is
# then W_NT - FRQ, its release period comes out one output interval short, and --strict-rel
# refuses it -- correctly, and after the GPU time is spent. MEASURED on the first two-window
# ring run: 195.0 s of releases against the 200 s asked for.
#
# So the windows are spaced W_NT + FRQ apart and the run is one interval longer per extra
# window. Every window then owns a full W_NT of fields and delivers exactly
# W_NT/FRQ + 1 snapshots, on BOTH paths -- which also keeps the CPU-from-disk vs
# GPU-from-ring acceptance a comparison of the same window rather than of two schedules.
# N_WINDOWS = 1 is arithmetically unchanged.
FRQ_NT=$(python3 -c "print(int(round(5.0/$DT)))")
W_NT=$(python3 -c "print(int(round($WINDOW_S/$DT/$FRQ_NT))*$FRQ_NT)")
NW_EXPECT=$(python3 -c "print($W_NT//$FRQ_NT + 1)")
TOT_NT=$(python3 -c "print($A_NT + (${N_WINDOWS:-1}-1)*($W_NT+$FRQ_NT) + $W_NT)")
RUN_S=$(python3 -c "print(f'{$TOT_NT*$DT:.3f}')")
win_t0(){ python3 -c "print(f'{($A_NT + $1*($W_NT+$FRQ_NT))*$DT:.3f}')"; }
win_t1(){ python3 -c "print(f'{($A_NT + $1*($W_NT+$FRQ_NT) + $W_NT)*$DT:.3f}')"; }
echo "  ${RUN_S}s total = ${ADJ_S}s adjust + ${N_WINDOWS:-1} x (${TBACK}s t_back + $(python3 -c "print($WINDOW_S-$TBACK)")s releases)"
echo "  each window is $W_NT steps = $NW_EXPECT dumps; consecutive windows are one output interval ($FRQ_NT steps) apart"

# ---- STUB_LES=1: NO LES AND NO LPDM, FOR CPU-ONLY VERIFICATION OF EVERYTHING ELSE -------
# Stages 1-5 and 8 run for REAL -- the sounding is fetched, the forcing is fitted, the
# per-case surface is built, a seed is picked and the restart is rotated and injected -- and
# only 6 and 7 are replaced. That is deliberate: a stub that reimplemented the driver would
# be a second code path, and this project's standing rule is that a check exercising a
# different path than production is not a check. The synthetic raster is built on THE CASE'S
# OWN GRID, so a stubbed run on the wrong geometry still produces the wrong record and
# bin/check_npz.py still catches it.
#
# A STUBBED RECORD IS NOT A CORPUS RECORD. stub=true is stamped into the footprint JSON and
# bin/run_month.sh refuses to record one.
if [ "${STUB_LES:-0}" = "1" ]; then
  say "$TAG  stages 6+7 STUBBED (STUB_LES=1) -- no GPU, no LES, no LPDM"
  echo "  the geometry this case is running on:"
  echo "    GRID=$GRID  ZTARGET=$ZTARGET  DX=$DX  TEMPLATE=$TEMPLATE  SEED_LIB=${SEED_LIB:-jobs30}"
  echo "    ADJ_S=$ADJ_S  WINDOW_S=$WINDOW_S  TBACK=$TBACK  N_WINDOWS=$N_WINDOWS"
  echo "    a case is $(python3 -c "print(f'{($ADJ_S+$WINDOW_S)/3600:.4f}')") sim-h; the"\
" footprint is its last $(python3 -c "print(int($WINDOW_S-$TBACK))") s"
  FPDIR="${FPDIR:-results/corpus}"; mkdir -p "$FPDIR"
  ./docker/pyrun.sh bin/stub_footprint.py --grid "$GRID" --forcing "$FRC" \
      --outdir "$FPDIR" --tag "$TAG" --z-target "$ZTARGET" || die "the stub footprint"
  [ -s "$FPDIR/$TAG.json" ] || die "the stub wrote no footprint json"
  say "$TAG  stage 8: the training record"
  ./docker/pyrun.sh bin/make_pair.py --tag "$TAG" --footprint "$FPDIR/$TAG.json" \
      --forcing "$FRC" --seed "$PICK" --grid "$GRID" --outdir pairs \
      --npz-dir "${NPZ_DIR:-pairs_npz}" || die "stage 8"
  [ -s "pairs/$TAG.json" ] || die "stage 8 wrote no pair"
  say "$TAG COMPLETE (STUBBED)"
  exit 0
fi

# ---- RING=1: THE LES AND THE LPDM RUN AT THE SAME TIME, IN TWO CONTAINERS ---------------
#
# The in-process hand-off (SRC/IO/io_lpdmonline.c) stages every output step into a tmpfs
# directory that both containers mount at an IDENTICAL path, and blocks the LES at each
# window boundary until the consumer has integrated that window. So the LES cannot be a
# blocking call any more: it is backgrounded here and bin/stage5_footprint.py --ring is the
# foreground process, once per window.
#
# THE PAUSE STEPS ARE COMPUTED FROM THE SAME ARITHMETIC AS THE WINDOW BOUNDS, not written
# down twice. A pause at a step the run never outputs at means the LES sails past it, the
# consumer waits out its 300 s stall timeout, and the case dies with a message about a dead
# producer -- so run_window.sh checks each one against frqOutput and Nt before launching.
if [ "${RING:-0}" = "1" ]; then
  RING_DIR="${RING_DIR:-${FLUX_RINGROOT:-/dev/shm/flux}/$TAG}"
  export RING_DIR RING_SELECTOR="${RING_SELECTOR:-2}" RING_QUEUE="${RING_QUEUE:-4}"
  # The pause is the LAST step of each window, i.e. the step whose snapshot completes it.
  export RING_PAUSE1=$(( A_NT + W_NT ))
  [ "${N_WINDOWS:-1}" -ge 2 ] && export RING_PAUSE2=$(( A_NT + (W_NT+FRQ_NT) + W_NT ))
  echo "  RING: staging to $RING_DIR, selector $RING_SELECTOR, pauses at "\
"${RING_PAUSE1}${RING_PAUSE2:+ and $RING_PAUSE2} of $TOT_NT"
  LESLOG="$L/${TAG}_les.log"
  ( SKIP_S="$ADJ_S" BASE="$D/case.in" bin/run_window.sh "$D" "$D/FE_RST.0" "$DT" "$RUN_S" \
      ./topo.bin "$UG" "$VG" > "$LESLOG" 2>&1; echo $? > "$D/.les_rc" ) &
  LES_PID=$!
  echo "  LES backgrounded as pid $LES_PID, log $LESLOG"
  # If the LES dies before it stages anything the consumer would sit for 300 s and then
  # blame the producer. It would be right, but the log is what says why -- so surface it.
  # STOPPING THE SHELL IS NOT STOPPING THE LES. run_window.sh launches FastEddy in a
  # container; killing the subshell leaves the container running, holding the GPU, and the
  # next run is then REFUSED by run_case.sh's concurrency guard with no hint of why.
  # Measured exactly that on the first ring smoke run. Stop the container by name.
  FE_CONTAINER_NAME="flux-fe-$(echo "$D" | tr -c 'A-Za-z0-9_.-' '-')"
  export FE_CONTAINER_NAME
  trap 'kill $LES_PID 2>/dev/null; docker rm -f "$FE_CONTAINER_NAME" >/dev/null 2>&1;
        wait $LES_PID 2>/dev/null' EXIT
else
  SKIP_S="$ADJ_S" BASE="$D/case.in" bin/run_window.sh "$D" "$D/FE_RST.0" "$DT" "$RUN_S" \
      ./topo.bin "$UG" "$VG" || die "adjustment+window"
  rm -f "$D/FE_RST.0"
fi

# ---- 6b. ASSERT ON THE STATE THE DUMPS CARRY, NOT ON THE .in ----------------------
# The surface reaches FastEddy only through the restart file, and the restart READ
# overwrites whatever the .in said (PROJECT_BRIEF.md, the Stage 6 lever). prep_restart.py now
# reads back what it injected, but that scores the file, not the RUN -- and this project
# has four separate instances of a configured value that the model never actually used.
# The window dumps carry z0m, so the question "did the run use this case's surface?" is
# answerable directly, for the price of opening one file.
#
# For a NEUTRAL case this is the whole ballgame: htFlux is zero everywhere, so the array's
# entire signal is the z0 contrast, and a run that silently fell back to a uniform z0 would
# produce a clean, complete, perfectly plausible case with no array in it.
# WITH THE RING, THIS MOVES AFTER THE RUN. The dumps do not exist yet -- the LES is still
# going, and at selector 1 it will never write any. The check is not dropped: it runs after
# the wait below, on the dumps selector 2 produced, and what covers the window in the
# meantime is that the consumer reads z0m and invOblen out of every staged snapshot and
# window_stats derives the surface flux from them, so a run on the wrong surface shows up
# in stage 7b and 7c rather than passing quietly.
surface_readback(){
EARLY=$(ls -1 "$D"/window/FE_WIN.[0-9]* 2>/dev/null | sed 's/.*\.//' | sort -n | head -1)
[ -n "$EARLY" ] || die "stage 6 left no window dumps"
./docker/pyrun.sh - "$D/window/FE_WIN.$EARLY" "$GRID" <<'PYSURF' || die "the window ran on the wrong surface"
import sys, os
import numpy as np
from netCDF4 import Dataset
dump, grid = sys.argv[1], sys.argv[2]
with Dataset(dump) as ds:
    have = {v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
            for v in ("z0m", "htFlux") if v in ds.variables}
if "z0m" not in have:
    print(f"FATAL: {dump} carries no z0m; the run's surface cannot be verified",
          file=sys.stderr)
    raise SystemExit(1)
bad = []
for nm, fn in (("z0m", "z0m.npy"), ("htFlux", "htFlux.npy")):
    if nm not in have:
        continue
    want = np.load(os.path.join(grid, fn)).astype(np.float64).reshape(have[nm].shape)
    sc = max(float(np.abs(want).max()), 1e-30)
    rel = float(np.abs(have[nm] - want).max()) / sc
    print(f"  the window's {nm} matches {grid}/{fn} to {rel:.2e} relative")
    if rel > 1e-5:
        bad.append(f"{nm} differs by {rel:.2e}")
am = os.path.join(grid, "array.npy")
if os.path.exists(am):
    a = np.load(am).astype(bool).reshape(have["z0m"].shape)
    if a.any():
        za, zo = float(np.median(have["z0m"][a])), float(np.median(have["z0m"][~a]))
        print(f"  the run saw the array at z0 = {za:.3f} m against a domain median "
              f"{zo:.3f} m ({za/max(zo,1e-12):.2f}x)")
        if za <= zo * 1.001:
            print("  WARNING: no aerodynamic array contrast in the state the RUN used")
if bad:
    print("FATAL: " + "; ".join(bad), file=sys.stderr)
    raise SystemExit(1)
PYSURF
}
if [ "${RING:-0}" != "1" ]; then surface_readback; fi

# ---- 7-8, ONCE PER SAMPLING WINDOW -------------------------------------------------
# A case runs N_WINDOWS sampling windows back to back inside ONE FastEddy invocation, and
# each yields its own footprint, its own gates and its own training pair. Wrapped in a
# function rather than duplicated, because these hundred lines are where every per-case
# gate lives and two copies of them would drift -- the same argument that keeps
# window_stats a single implementation with two sources.
#
# WHY MORE THAN ONE WINDOW. Re-running an identical case gave integral 1.463 -> 1.019 and
# array share 5.65% -> 1.07%: turbulence REALISATION variance, against which every floor
# this project quotes is a WITHIN-realisation floor and therefore too small. A second
# window costs 0.75 h of simulated time rather than a whole case, and takes the class from
# 1.25 h per footprint to 1.0.
#
# N_WINDOWS DEFAULTS TO 1 AND THAT PATH IS BYTE-IDENTICAL to the single-window driver, tag
# included -- a two-window corpus is opted into, never inherited.
#
# BOTH TIME BOUNDS ARE REQUIRED. compute_footprint releases over
# [t_last - rel_seconds, t_last], so t_last is what selects the averaging period: with only
# a lower bound, window 0's footprint would silently absorb window 1's fields and release
# over the wrong 30 minutes. --t-max is the mirror of --t-min, not the lesser of the two.
#
# The body below is NOT re-indented, deliberately: it carries python heredocs whose
# terminators must stay at column 0, and indenting them is a syntax error that only
# appears when the function is first called.
run_one_window(){        # $1 = window index, $2 = t_min (s), $3 = t_max (s)
local WI="$1" W_TMIN="$2" W_TMAX="$3" WTAG
if [ "${N_WINDOWS:-1}" -le 1 ]; then WTAG="$TAG"; else WTAG="${TAG}_w${WI}"; fi
echo
echo "  ---- window $WI: fields ${W_TMIN}-${W_TMAX} s, tag $WTAG ----"
# ---- 7. the footprint ------------------------------------------------------------
# INTO results/corpus/, WHICH IS GITIGNORED. The repo already tracks 72 footprint .npz
# files at ~300 kB each; 1825 more would be ~550 MB of binaries in git history. The
# retired passes' results stay where they are and stay tracked -- they are the record.
say "$WTAG  stage 7: backward LPDM"
FPDIR="${FPDIR:-results/corpus}"; mkdir -p "$FPDIR"
# THE SOURCE IS THE STAGING DIRECTORY, NOT THE WINDOW DIRECTORY, when the ring is driving.
# stage5 then blocks on the live LES: it consumes each snapshot as it is staged, releases
# it, integrates at the pause, and writes the resume marker. At selector 2 the netCDF dumps
# are ALSO on disk, so a failure here is recoverable by re-running this stage without
# --ring -- which is the entire reason the validation runs at 2 rather than 1.
S5SRC="$D/window"; S5RING=""
if [ "${RING:-0}" = "1" ]; then S5SRC="$RING_DIR"; S5RING="--ring"; fi
LPDM_WORKERS="${LPDM_WORKERS:-8}" \
./docker/pyrun.sh bin/stage5_footprint.py "$S5SRC" $S5RING --dt "$DT" --tback "$TBACK" \
    --sgs-most --cover-dir "$GRID" --receptor-from "$GRID" --fp16-cache \
    --z-target "$ZTARGET" ${EXACT_AGL:+--exact-agl} --rel-seconds "${REL_S:-1800}" --strict-rel \
    --cover-groups "${COVER_GROUPS:-10}" \
    --keep-touchdowns "$KEEP_TD" \
    ${TBACK_MARKS:+--tback-marks "$TBACK_MARKS"} \
    --t-min "$W_TMIN" --t-max "$W_TMAX" \
    ${S5RING:+--ring-expect $NW_EXPECT} \
    --outdir "$FPDIR" --tag "$WTAG" 2>&1 | grep -vE 'batch [0-9]+/' > "$FPDIR/$WTAG.txt"
[ -s "$FPDIR/$WTAG.json" ] || { tail -12 "$FPDIR/$WTAG.txt" >&2
  die "stage 7 produced no footprint json"; }

# ---- 7b. the per-case health gate -------------------------------------------------
# ASSERT ON THE ARTIFACT, NOT ON THE EXIT STATUS. Stage 7 is piped into grep, so bash
# reports GREP's status and a python traceback lands quietly in the redirected .txt. The
# gates are read back out of the JSON stage 7 was supposed to write.
#
# WHAT IT GATES, AND WHY NOT THE ARRAY SHARE. The share is the scientific result and a
# poor fault detector: the h-fell-through defect was a SIX-fold error in the quantity
# setting the sigma_w floor and it moved the share 0.8 points against a 3.66-point SE,
# while it moved the near-field peak a full raster cell. Peak location, floor health and
# the integral are the sharp quantities; the share is reported beside them.
#
# NON-FATAL BY DEFAULT. A failed gate marks the case and lets the campaign continue --
# across 1370 cases the useful output is a LIST of suspect cases, not a driver that stops
# on the first one at 3 a.m. Set MONITOR_FATAL=1 to make it stop.
say "$WTAG  stage 7b: per-case health gate"
# ${PIPESTATUS[0]}, NOT $?. Piping into tee makes $? TEE's status, which is 0 whatever
# the monitor decided -- and writing that mistake into the very check that exists to catch
# it would be the joke completing itself.
./docker/pyrun.sh bin/corpus_monitor.py "$FPDIR/$WTAG.json" \
    --json "$FPDIR/$TAG.monitor.json" 2>&1 | tee -a "$FPDIR/$WTAG.txt"
MON_RC=${PIPESTATUS[0]}
[ -s "$FPDIR/$TAG.monitor.json" ] || die "stage 7b wrote no monitor json"
# AND RE-READ THE VERDICT FROM THE JSON, so the gate does not rest on an exit code at all.
MON_V=$(python3 -c "import json;d=json.load(open('$FPDIR/$TAG.monitor.json'));\
print('FAIL' if d['fail'] else ('UNJUDGED' if d['unjudged'] else 'OK'))")
echo "  health gate: $MON_V (rc $MON_RC)"
if [ "$MON_V" != "OK" ]; then
  echo "  *** $TAG health gate: $MON_V -- see $FPDIR/$TAG.txt" >&2
  echo "$TAG $MON_V" >> "$FPDIR/SUSPECT_CASES.txt"
  [ "${MONITOR_FATAL:-0}" = "1" ] && die "the per-case health gate returned $MON_V"
fi

# ---- 7c. the sigma_w ACCEPTANCE gate ----------------------------------------------
# THE ONLY EXTERNAL CHECK IN THE PROJECT, AND IT IS NOW A GATE RATHER THAN A DIAGNOSTIC.
# data/raw/H_and_sigma_w.csv is a year of half-hourly eddy covariance at the real
# instrument, never used for training, tuning or forcing; bin/sigma_w_tower.py translates
# it from the 10 m sensor to the 30 m model receptor through MOST (u* is constant in the
# surface layer, H is a surface flux, so only phi_w(z/L) changes with height).
#
# WHY A GATE. At the retired 10 m receptor the LES ran 2.33-2.99x the tower median with the
# closure floor INACTIVE, i.e. the near-field variance was closure output rather than LES
# output -- and it was REPORTED, case after case, without stopping anything. A case whose
# sigma_w falls outside the measured IQR for its own surface heat flux is not a usable
# target, and refusing it is the same discipline the z_i band already uses: refused, never
# mis-labelled.
#
# THE IQR IS DELIBERATELY STRICT and the failure mode is stated in advance: the file
# carries no wind speed, so conditioning on H alone leaves a band spanning a factor of ~2,
# and if the refusal rate comes out high that number gets REPORTED rather than the band
# widened. Non-fatal by default for the same reason stage 7b is.
say "$WTAG  stage 7c: sigma_w against the tower, translated to the receptor"
CURVE="${SIGMAW_CURVE:-results/sigma_w_curve_30m.json}"
if [ -s "$CURVE" ]; then
  ./docker/pyrun.sh - "$FPDIR/$WTAG.json" "$SND" "$CURVE" "$ZTARGET" <<'PYSW' \
      2>&1 | tee -a "$FPDIR/$WTAG.txt"
import json, sys
fp, snd, curve, zt = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
d = json.load(open(fp)); c = json.load(open(curve)); s = json.load(open(snd))
sw = d["stats"].get("sigma_w")
H = float(s["surface"].get("shtfl_wm2", s["surface"].get("shtfl", 0.0)))
b = min(c["bins"], key=lambda r: abs(r["h_median"] - H))
q = b["sigma_w_30m"]
inside = q["p25"] <= sw <= q["p75"]
print(f"  LES sigma_w at the receptor {sw:.3f} m/s; tower H = {H:+.0f} W/m2 -> bin "
      f"[{b['h_lo']:.0f}, {b['h_hi']:.0f}] (n = {b['n']})")
print(f"  tower sigma_w({zt:.0f} m): median {q['p50']:.3f}, "
      f"IQR [{q['p25']:.3f}, {q['p75']:.3f}]  -> "
      f"{'INSIDE' if inside else 'OUTSIDE'}, {sw/q['p50']:.2f}x the median")
json.dump({"sigma_w_les": sw, "H_wm2": H, "bin": b, "inside_iqr": bool(inside),
           "ratio_to_median": sw / q["p50"]},
          open(fp.replace(".json", ".sigmaw.json"), "w"), indent=1)
print(f"  sigma_w gate: {'OK' if inside else 'OUTSIDE THE IQR'}")
PYSW
  SW_V=$(python3 -c "import json;print('OK' if json.load(open('$FPDIR/$TAG.sigmaw.json'))['inside_iqr'] else 'OUTSIDE')" 2>/dev/null || echo "UNJUDGED")
  echo "  sigma_w acceptance: $SW_V"
  if [ "$SW_V" != "OK" ]; then
    echo "$TAG sigma_w-$SW_V" >> "$FPDIR/SUSPECT_CASES.txt"
    [ "${SIGMAW_FATAL:-0}" = "1" ] && die "sigma_w outside the tower IQR for this H"
  fi
else
  echo "  no $CURVE -- run bin/sigma_w_tower.py. NO VERDICT (not a pass)."
fi

# ---- 8. the training record ------------------------------------------------------
say "$WTAG  stage 8: assemble the pair"
./docker/pyrun.sh bin/make_pair.py --tag "$WTAG" --footprint "$FPDIR/$WTAG.json" \
    --forcing "$FRC" --seed "$PICK" --grid "$GRID" --outdir pairs \
    --npz-dir "${NPZ_DIR:-pairs_npz}" || true
[ -s "pairs/$WTAG.json" ] || die "stage 8 wrote no pair"
}

for WI in $(seq 0 $(( ${N_WINDOWS:-1} - 1 ))); do
  T0=$(win_t0 "$WI"); T1=$(win_t1 "$WI")
  run_one_window "$WI" "$T0" "$T1"
done

# ---- THE LES IS STILL RUNNING IF THE RING DROVE IT -------------------------------------
# The last window's resume marker lets it finish its final steps and write its final
# restartable dump; nothing above waited for that. Wait now, score its exit, and only then
# run the surface read-back that stage 6b defers under RING -- it needs the dumps.
if [ "${RING:-0}" = "1" ]; then
  say "$TAG  waiting for the LES to finish after the last window"
  wait "$LES_PID" 2>/dev/null
  trap - EXIT
  LES_RC=$(cat "$D/.les_rc" 2>/dev/null || echo 1)
  rm -f "$D/.les_rc" "$D/FE_RST.0"
  echo "  LES exited $LES_RC (log $LESLOG)"
  tail -6 "$LESLOG" 2>/dev/null | sed 's/^/    /'
  [ "$LES_RC" = "0" ] || die "the LES failed after the windows were taken (see $LESLOG)"
  # WHAT THE HAND-OFF COST THE LES, from its own log rather than from the consumer's clock.
  # FASTEDDY'S OWN STDOUT, NOT run_window.sh's. run_case.sh redirects the container to
  # /tmp/flux-logs/<case>_win1.log and only the scoring lines come back up the pipe, so
  # the pause/resume/backpressure record lives there. Grepping $LESLOG finds nothing and
  # says nothing, which is the quietest possible way to lose the measurement.
  FELOG="/tmp/flux-logs/$(basename "$D")_win1.log"
  echo "  the LES's own hand-off record ($FELOG):"
  grep -E 'lpdmOnline: (PAUSED|RESUMED|BLOCKED|finished|staging)' "$FELOG" \
    | sed 's/^/    /' || echo "    (none -- the hand-off never engaged)"
  # STAGED vs WRITTEN, at selector 2. Both come from the same buffers, so a mismatch is a
  # lost snapshot on one path or the other and it would be invisible in the footprints.
  if [ "${RING_SELECTOR:-2}" != "1" ]; then
    _staged=$(grep -oP 'finished, \K[0-9]+' "$FELOG" | tail -1)
    _written=$(grep -cE '^Dumped state' "$FELOG")
    echo "    staged $_staged snapshots against $_written netCDF dumps written"
    [ -n "$_staged" ] && [ "$_staged" = "$_written" ] \
      || echo "    *** staged/written MISMATCH -- one path lost a snapshot" >&2
  fi
  if [ "${RING_SELECTOR:-2}" != "1" ]; then
    say "$TAG  stage 6b (deferred): did the run use this case's surface?"
    surface_readback
  fi
  rm -rf "$RING_DIR"
fi

# ---- 8b. are the two windows two draws, or one draw written twice? -----------------
# Only measurable when there are two, and the answer decides whether the extra 0.75 h per
# case buys anything. REPORTED, NEVER GATED: a near-duplicate verdict is a pricing result,
# not a broken case.
if [ "${N_WINDOWS:-1}" -ge 2 ] && [ -s "$FPDIR/${TAG}_w0.json" ] \
   && [ -s "$FPDIR/${TAG}_w1.json" ]; then
  say "$TAG  stage 8b: are the two windows independent?"
  ./docker/pyrun.sh bin/window_independence.py \
      "$FPDIR/${TAG}_w0.json" "$FPDIR/${TAG}_w1.json" \
      --out "$FPDIR/${TAG}_windows.txt" 2>&1 | tail -22 || true
fi


[ "${KEEP_FIELDS:-0}" = "1" ] || { rm -f $D/window/*; rm -rf "$CG"
                                  echo "  window fields and the case grid deleted"; }
say "$TAG COMPLETE"
