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
#          ALLOW_INDETERMINATE=0  require ESTABLISHED stationarity (default 1;
#                         no seed in the library can supply it -- see stage 4)
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
# GRID GEOMETRY, so one driver serves both configurations. The seventh pass runs
#   GRID=data/grid24_raised ZTARGET=28.5 TEMPLATE=runs/g24_base/base.in DX=24
#   ZCEILING=3000 DEFORM=0.346601 ZI_MAX_ABS=1250 SEED_LIB=jobs24
# and every one of those has to travel together: a 24 m case built against the 16 m
# template would carry the wrong d_zeta and the wrong dt and would still run.
TEMPLATE="${TEMPLATE:-runs/g16_base/base.in}"
DX="${DX:-16.0}"; ZCEILING="${ZCEILING:-2500.0}"; DEFORM="${DEFORM:-0.194059}"
ZI_MAX_ABS="${ZI_MAX_ABS:-}"
KEEP_TD="${KEEP_TD:-100000}"
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
    --template "$TEMPLATE" --dx "$DX" --zceiling "$ZCEILING" --deform "$DEFORM" \
    ${ZI_MAX_ABS:+--zi-max-abs $ZI_MAX_ABS} \
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
./docker/pyrun.sh bin/pick_seed.py "$FRC" --json "$PICK" --zm "$ZTARGET" \
    --library "${SEED_LIB:-jobs}" --index "${SEED_LIB:-jobs}/index.json" \
    $([ "${SEED_ANY:-0}" = "1" ] || echo --available-only) \
    $([ "${ALLOW_INDETERMINATE:-1}" = "1" ] && echo --allow-indeterminate) \
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
# make_pair.py writes a warning into the training record. A seed with a DRIFTING limit is
# still refused outright -- that is a different and stronger statement, and no flag admits
# it. Set ALLOW_INDETERMINATE=0 to require established stationarity, which today no seed
# in the library can supply.
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
RUN_S=$(python3 -c "print(f'{$A_NT*$DT + ${N_WINDOWS:-1}*$WINDOW_S:.3f}')")
echo "  ${RUN_S}s total = ${ADJ_S}s adjust + ${N_WINDOWS:-1} x (${TBACK}s t_back + $(python3 -c "print($WINDOW_S-$TBACK)")s releases)"
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
LPDM_WORKERS="${LPDM_WORKERS:-8}" \
./docker/pyrun.sh bin/stage5_footprint.py "$D/window" --dt "$DT" --tback "$TBACK" \
    --sgs-most --cover-dir "$GRID" --receptor-from "$GRID" --fp16-cache \
    --z-target "$ZTARGET" ${EXACT_AGL:+--exact-agl} --rel-seconds 1800 --strict-rel \
    --cover-groups "${COVER_GROUPS:-10}" \
    --keep-touchdowns "$KEEP_TD" \
    ${TBACK_MARKS:+--tback-marks "$TBACK_MARKS"} \
    --t-min "$W_TMIN" --t-max "$W_TMAX" \
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
    --forcing "$FRC" --seed "$PICK" --outdir pairs || true
[ -s "pairs/$WTAG.json" ] || die "stage 8 wrote no pair"
}

for WI in $(seq 0 $(( ${N_WINDOWS:-1} - 1 ))); do
  T0=$(python3 -c "print(f'{$A_NT*$DT + $WI*$WINDOW_S:.3f}')")
  T1=$(python3 -c "print(f'{$A_NT*$DT + ($WI+1)*$WINDOW_S:.3f}')")
  run_one_window "$WI" "$T0" "$T1"
done

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
