#!/usr/bin/env bash
# ONE corpus case, end to end: a timestamp in, an (input, target) training pair out.
#
#   usage: bin/run_corpus_case.sh 2023-01-18T18:00 [tag]
#   env:   WINDOW_S=2400  ADJ_S=1800  TBACK=600  KEEP_FIELDS=0  COVER_GROUPS=10
#          -- COVER_GROUPS is the number of independent release groups the array-share
#             standard error is estimated from. The default in stage5_footprint.py is 2,
#             which is ONE difference and therefore ~one degree of freedom; PROJECT_BRIEF.md's
#             standing rule is N >= 8 wherever a share or a shape is being tested, and
#             Phase E measured a FACTOR OF 5 in the estimated floor between a 2-group and a
#             10-group split. The split costs nothing -- the touchdowns are already
#             labelled by release time -- so the corpus takes 10.
#          -- ADJ_S + WINDOW_S = 4200 s = 1.167 sim-h is the class length, and TBACK is
#             MEASURED (results/g16_tback.txt: converged at 500 s, x1.25, rounded to 600).
#          SKIP_LES=1     stop after stage 4, for a dry run with no GPU
#          SEED_LIB=jobs  where the spun-up seed library lives
#          SEED_ANY=1     rank seeds with no returned artifact too (planning only)
#          GRID=data/grid16_raised  ZTARGET=8.5  EXACT_AGL=1   (production; see below)
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
# ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS at every stage (FASTEDDY_TRAPS.md 12): the
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
# as FASTEDDY_TRAPS.md 12: a filtered stream is a hidden error.
_t="${TS//[-:]/}"; _t="${_t/T/}"
TAG="${2:-case_${_t:0:10}}"
[[ "$TAG" =~ ^[A-Za-z0-9_]+$ ]] || { echo "FATAL: bad tag '$TAG' from '$TS'" >&2; exit 65; }
[ "${#TAG}" -ge 8 ] || { echo "FATAL: tag '$TAG' too short; is '$TS' a valid timestamp?" >&2
                         exit 65; }
# THE RAISED SURFACE IS PRODUCTION, settled by the sixth pass (SIXTH_PASS_RESULTS.md):
# topoPos is raised by the displacement height over the array so the first model level
# clears panel top, z0_array goes 0.10 -> 0.25 (which is the only thing that gives the
# array ANY neutral signal -- at 0.10 it is aerodynamically identical to the cropland
# WorldCover labels it as), and the receptor is released at a FRACTIONAL level 8.500 m
# above the RAISED surface = 10.000 m above bare ground. Snapping to the nearest level
# there would put the receptor 10 m above the PANELS, an 11.5 m receptor and a 15% error
# in exactly the quantity this pass exists to get right.
GRID="${GRID:-data/grid16_raised}"
ZTARGET="${ZTARGET:-8.5}"
EXACT_AGL="${EXACT_AGL:-1}"
ADJ_S="${ADJ_S:-1800}"
WINDOW_S="${WINDOW_S:-2400}"
TBACK="${TBACK:-$(cat results/tback_production.txt 2>/dev/null || echo 600)}"
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
    --in-out "$D/case.in" || true
[ -s "$FRC" ] || die "stage 2 wrote no forcing json"
[ -s "$D/case.in" ] || die "stage 2 wrote no .in"
# REFUSE A CASE THE DOMAIN CANNOT HOLD, rather than running it and mis-labelling it.
# z_i outside 100-976 m is not representable: below it the 10 m receptor leaves the
# surface layer, above it the 1952 m box cannot hold the boundary layer at L >= 2 z_i.
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
./docker/pyrun.sh bin/pick_seed.py "$FRC" --json "$PICK" \
    --library "${SEED_LIB:-jobs}" --index "${SEED_LIB:-jobs}/index.json" \
    $([ "${SEED_ANY:-0}" = "1" ] || echo --available-only) || true
[ -s "$PICK" ] || die "stage 4 wrote no pick json"
read -r JOB ROT < <(python3 -c "
import json; c=json.load(open('$PICK'))['chosen']; print(c['job'], c['rot'])")
SEED="${SEED_LIB:-jobs}/$JOB/return/seed_restart.nc"
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
# FASTEDDY_TRAPS.md 17 removed structurally rather than by assertion: a restart READ
# overwrites every IO-registered field, so each restart is an opportunity to silently
# inherit state the .in does not describe.
#
# THE SCHEDULE, which is what makes the timeline correct rather than merely shorter. For a
# footprint stamped 01:00 UTC, covering 00:30-01:00:
#
#     23:50  restart from the seed; adjustment begins       step 0
#     00:20  adjustment complete (ADJ_S = 1800 s)           step A_NT = 123120
#     00:30  first release (needs history back to 00:20)    step A_NT + t_back
#     01:00  last release; window closes                    step TOT = 287280
#
# so the run is ADJ_S + t_back + 1800 = 1800 + 600 + 1800 = 4200 s = 1.167 sim-h, and the
# earliest field a backward trajectory can reach is EXACTLY the adjustment end.
#
# FastEddy has a single frqOutput and writes from step 0, so the adjustment's 360 dumps
# are produced and then deleted; run_window.sh does the deleting and then ASSERTS that the
# earliest surviving dump is step A_NT. Without that, lpdm/fields.py:dump_series would glob
# the whole directory and compute_footprint would release from t[0] + t_back -- putting the
# entire 30-minute averaging period inside the adjustment, on fields still settling, with
# nothing in the output to say so.
say "$TAG  stage 6: ${ADJ_S}s adjustment + ${WINDOW_S}s window, one invocation"
A_NT=$(python3 -c "
frq=int(round(5.0/$DT)); print(int(round($ADJ_S/$DT/frq))*frq)")
[ "$A_NT" -gt 0 ] 2>/dev/null || die "adjustment step count did not compute"
RUN_S=$(python3 -c "print($ADJ_S + $WINDOW_S)")
echo "  ${RUN_S}s total = ${ADJ_S}s adjust + ${TBACK}s t_back + $(python3 -c "print($WINDOW_S-$TBACK)")s releases"
SKIP_S="$ADJ_S" BASE="$D/case.in" bin/run_window.sh "$D" "$D/FE_RST.0" "$DT" "$RUN_S" \
    ./topo.bin "$UG" "$VG" || die "adjustment+window"
rm -f "$D/FE_RST.0"

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

# ---- 7. the footprint ------------------------------------------------------------
# INTO results/corpus/, WHICH IS GITIGNORED. The repo already tracks 72 footprint .npz
# files at ~300 kB each; 1825 more would be ~550 MB of binaries in git history. The
# retired passes' results stay where they are and stay tracked -- they are the record.
say "$TAG  stage 7: backward LPDM"
FPDIR="${FPDIR:-results/corpus}"; mkdir -p "$FPDIR"
LPDM_WORKERS="${LPDM_WORKERS:-8}" \
./docker/pyrun.sh bin/stage5_footprint.py "$D/window" --dt "$DT" --tback "$TBACK" \
    --sgs-most --cover-dir "$GRID" --receptor-from "$GRID" --fp16-cache \
    --z-target "$ZTARGET" ${EXACT_AGL:+--exact-agl} --rel-seconds 1800 --strict-rel \
    --cover-groups "${COVER_GROUPS:-10}" \
    --t-min "$(python3 -c "print(f'{$A_NT*$DT:.3f}')")" \
    --outdir "$FPDIR" --tag "$TAG" 2>&1 | grep -vE 'batch [0-9]+/' > "$FPDIR/$TAG.txt"
[ -s "$FPDIR/$TAG.json" ] || { tail -12 "$FPDIR/$TAG.txt" >&2
  die "stage 7 produced no footprint json"; }

# ---- 8. the training record ------------------------------------------------------
say "$TAG  stage 8: assemble the pair"
./docker/pyrun.sh bin/make_pair.py --tag "$TAG" --footprint "$FPDIR/$TAG.json" \
    --forcing "$FRC" --seed "$PICK" --outdir pairs || true
[ -s "pairs/$TAG.json" ] || die "stage 8 wrote no pair"

[ "${KEEP_FIELDS:-0}" = "1" ] || { rm -f $D/window/*; rm -rf "$CG"
                                  echo "  window fields and the case grid deleted"; }
say "$TAG COMPLETE"
