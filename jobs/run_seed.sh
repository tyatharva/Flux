#!/usr/bin/env bash
# Run ONE seed spin-up, end to end, on whatever machine this checkout is sitting on.
#
#   usage: jobs/run_seed.sh jobs/seed_cbl-mid_a030
#          jobs/run_seed.sh jobs/seed_cbl-mid_a030 --dry-run
#          jobs/run_seed.sh jobs/seed_cbl-mid_a030 --restart-over   discard a partial run
#
# NO ABSOLUTE PATHS. The repo root is discovered from this script's own location, so a
# rented GPU can check the repo out anywhere; FLUX_ROOT is exported for docker/run_case.sh
# and docker/pyrun.sh, which is why those two now read it instead of a literal.
#
# NO SHARED STATE. Everything this job writes lands under its own job directory. Two seed
# jobs on two machines cannot interact; two on ONE machine are serialised by
# docker/run_case.sh, which refuses to start a second FastEddy container.
#
# NO CHAIN, AND THEREFORE NO RESUME. A seed is ONE continuous FastEddy invocation --
# 738,720 steps, 3.0 simulated hours, ~2.9 h wall. Chaining was retired 2026-08-26 and
# with it the entire failure mode of FASTEDDY_TRAPS.md 17: a restart READ overwrites every
# IO-registered field (htFlux, z0m, z0t, tskin, topoPos, zPos) with whatever the restart
# file holds, so every segment boundary was an opportunity to inherit state the .in does
# not describe. It cost a whole segment of a stable seed running at zero surface flux
# while its .in asked for -0.012. THE ONLY RESTART LEFT IN THE PROJECT IS SEED -> TARGET.
#
# The price, stated plainly: a killed job now costs the whole run rather than one segment,
# and the one-hour-per-run wall cap no longer applies to a seed. Re-invoking a COMPLETE
# job is still a no-op -- that is idempotence, not a restart.
set -uo pipefail

JOB_ARG="${1:-}"
[ -n "$JOB_ARG" ] || { echo "usage: run_seed.sh <job_dir> [--dry-run]" >&2; exit 64; }
DRY=0; OVER=0
for f in "${@:2}"; do
  case "$f" in
    --dry-run)      DRY=1;;
    --restart-over) OVER=1;;
    *) echo "unknown flag $f" >&2; exit 64;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB="$(cd "$JOB_ARG" && pwd)" || { echo "FATAL: no job dir $JOB_ARG" >&2; exit 66; }
JOB_REL="${JOB#$ROOT/}"
export FLUX_ROOT="$ROOT"
cd "$ROOT"

die(){ echo "FATAL: $*" >&2; exit 1; }
[ -f "$JOB/manifest.json" ] || die "no manifest.json in $JOB"
[ -f "$JOB/seed.in" ] || die "no seed.in in $JOB"

read -r NAME DT FRQ TOTAL WTH OUTBASE WALLMIN ZM ZK DXG < <(python3 - "$JOB/manifest.json" <<'PYMAN'
import json,sys
m=json.load(open(sys.argv[1])); r=m["run"]; g=m.get("gate", {})
print(m["job"], r["dt"], r["frqOutput"], r["steps_total"],
      m["target"]["wth_virtual"], r["outFileBase"], r["projected_wall_min"],
      g.get("zm", 10.0), g.get("k", 2), m.get("grid", {}).get("dx", 16.0))
PYMAN
) || die "manifest.json unreadable"

# ---- SEED_CEILING_H: a HARD ceiling in simulated hours, overriding the manifest --------
# The manifest's steps_total is the library's standing 3.0 sim-h ceiling. A convective rung
# decorrelates on z_i/w* ~ 540 s against a neutral layer's h/u* ~ 1700 s, so it reaches band
# far sooner and 3.0 h of it is mostly wasted GPU. The ceiling is a CEILING either way --
# jobs/seed_watch.sh normally stops the run before it -- and a seed that has not entered
# band by it stops there and THAT IS THE RESULT: no extension, no respec.
# Rounded DOWN to a whole number of dumps, because a ceiling that is not a dump boundary
# stops the run between dumps and the last dump is then not the state that was scored.
if [ -n "${SEED_CEILING_H:-}" ]; then
  _NEW=$(python3 -c "print(int($SEED_CEILING_H*3600.0/$DT/$FRQ)*$FRQ)")
  [ "$_NEW" -gt 0 ] || die "SEED_CEILING_H=$SEED_CEILING_H rounds to zero dumps"
  WALLMIN=$(python3 -c "print(f'{$WALLMIN*$_NEW/$TOTAL:.1f}')")
  echo "  SEED_CEILING_H=$SEED_CEILING_H: ceiling $TOTAL -> $_NEW steps "\
"($(python3 -c "print(f'{$_NEW*$DT/3600:.3f}')") sim-h, $(python3 -c "print($_NEW//$FRQ)") dumps)"
  TOTAL="$_NEW"
fi

echo "########## seed $NAME ##########"
date '+%F %H:%M:%S'
echo "  dt $DT, dump every $FRQ steps, $TOTAL steps in ONE invocation (proj ${WALLMIN} min wall)"

# ---- preflight: the GPU, before any GPU time is spent -----------------------------
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [ -z "$CC" ]; then
  echo "  WARNING: nvidia-smi gave no compute capability; cannot verify sm_89" >&2
elif [ "$CC" != "8.9" ]; then
  # Not fatal. The image is compiled for sm_89 and CUDA will JIT from PTX on a newer
  # architecture, which works and is slower; on an OLDER one it will not run at all.
  echo "  WARNING: compute capability $CC, image built for 8.9 (sm_89)." >&2
  echo "           Newer: JIT from PTX, slower but correct. Older: it will not run." >&2
fi
docker image inspect flux-fasteddy:cuda118 >/dev/null 2>&1 \
  || die "image flux-fasteddy:cuda118 not present; build it with docker/build_fasteddy.sh"

if [ "$DRY" = "1" ]; then echo "  --dry-run: preflight only, stopping here"; exit 0; fi

mkdir -p "$JOB/output" "$JOB/return"
LOG="$JOB/return/seed.log"

# ---- the run: ONE invocation ------------------------------------------------------
# Idempotent, not resumable. If the final dump is already on disk the LES is skipped and
# the gate is re-scored; anything short of that is re-run from the start, because there is
# no restart within a seed any more.
FINAL_DUMP="$JOB/output/$OUTBASE.$TOTAL"
if [ -f "$FINAL_DUMP" ]; then
  echo "  $OUTBASE.$TOTAL already on disk; skipping the LES and re-scoring the gate"
else
  # A PARTIAL RUN IS REFUSED, NOT SILENTLY WIPED. It cannot be resumed -- resuming is the
  # restart this design retires -- but it is also up to ~3 h of GPU that someone may want
  # to look at, and deleting it on their behalf is not this script's call to make.
  if ls -1 "$JOB/output/$OUTBASE".* >/dev/null 2>&1; then
    if [ "$OVER" = "1" ]; then
      echo "  --restart-over: discarding the partial run in $JOB/output/"
      rm -f "$JOB/output/$OUTBASE".*
    else
      P=$(ls -1 "$JOB/output/$OUTBASE".* | sort -t. -k2 -n | tail -1)
      die "a PARTIAL run is on disk (newest ${P##*/}, wanted step $TOTAL). A seed is one
      continuous invocation and cannot resume -- that restart is exactly what was retired.
      Re-run with --restart-over to discard it and start from step 0, or move it aside."
    fi
  fi
  # ---- Steinfeld spin-up accelerator (neutral rungs) -------------------------------
  # Steinfeld et al. (2008) start a neutral LES with a surface temperature flux of
  # ~0.05 K m/s for the first ~3000 s to trip the transition to resolved turbulence, then
  # remove it. A neutral boundary layer has no buoyant production to organise the initial
  # perturbation field, so it is the slowest regime to spin up -- h/u* ~ 1500 s here
  # against T* ~ 350 s convectively -- and the accelerator is aimed exactly at that.
  #
  # IT COSTS ONE RESTART, AND THE RESTART IS THE DANGEROUS PART. htFlux is IO-registered,
  # so the main invocation would inherit +0.05 from the burn-in dump whatever its .in says
  # (FASTEDDY_TRAPS.md 17). bin/zero_htflux.py writes the zero into the FILE and reads it
  # back, and the existing per-run htFlux assertion below is the second lock.
  ACC_S="${SEED_ACCEL_S:-0}"
  if [ "$ACC_S" != "0" ]; then
    ACC_NT=$(python3 -c "print(int(round($ACC_S/$DT/$FRQ))*$FRQ)")
    ACC_WTH="${SEED_ACCEL_WTH:-0.05}"
    echo "  --- accelerator burn-in: $ACC_NT steps at surflayer_wth = $ACC_WTH"
    sed -e "s|^Nt = .*|Nt = $ACC_NT|" \
        -e "s|^inPath = .*|inPath = |" -e "s|^inFile = .*|inFile = |" \
        -e "s|^surflayer_wth = .*|surflayer_wth = $ACC_WTH|" \
        -e "s|^outFileBase = .*|outFileBase = ${OUTBASE}_ACC|" \
        "$JOB/seed.in" > "$JOB/accel.in"
    ./docker/run_case.sh "$JOB_REL" "accel.in" "$JOB/return/accel.log" || die "burn-in"
    ACC_LAST="$JOB/output/${OUTBASE}_ACC.$ACC_NT"
    [ -f "$ACC_LAST" ] || die "the burn-in wrote no dump at step $ACC_NT"
    cp -f "$ACC_LAST" "$JOB/FE_ACC.0" || die "staging the burn-in restart"
    ./docker/pyrun.sh bin/zero_htflux.py "${JOB_REL}/FE_ACC.0" --value "$WTH" \
      || die "could not clear the burn-in htFlux out of the restart"
    # ONE RUN PER DIRECTORY, OR IT IS NOT A SERIES (FASTEDDY_TRAPS.md 18c): the burn-in's
    # dumps carry step numbers that overlap the main run's and would interleave into a
    # single sorted "history" with two states at the same time.
    rm -f "$JOB/output/${OUTBASE}_ACC".*
  fi
  # A COLD START, so inPath/inFile are empty and surflayer_wth is whatever the .in says.
  # With no restart to read, htFlux CANNOT be inherited -- which is the point of retiring
  # the chain. The assertion below is kept anyway: it costs seconds once per run, and
  # PROJECT_BRIEF.md's standing rule is to validate the state the model actually loaded, never
  # the config handed to it.
  if [ "$ACC_S" != "0" ]; then
    sed -e "s|^Nt = .*|Nt = $TOTAL|" \
        -e "s|^inPath = .*|inPath = ./|" -e "s|^inFile = .*|inFile = FE_ACC.0|" \
        -e "s|^surflayer_wth = .*|surflayer_wth = $WTH|" \
        "$JOB/seed.in" > "$JOB/run.in"
  else
    sed -e "s|^Nt = .*|Nt = $TOTAL|" \
        -e "s|^inPath = .*|inPath = |" -e "s|^inFile = .*|inFile = |" \
        -e "s|^surflayer_wth = .*|surflayer_wth = $WTH|" \
        "$JOB/seed.in" > "$JOB/run.in"
  fi
  echo "  --- single invocation -> step $TOTAL ($(python3 -c "print(f'{$TOTAL*$DT/60:.0f}')") min simulated)"
  date '+%F %H:%M:%S'
  rm -f "$JOB/output/.early_stop"
  WATCH_PID=""
  if [ "${SEED_EARLY_STOP:-1}" = "1" ]; then
    # OPEN-ENDED WITH A CEILING. Nt is the ceiling; the watcher usually ends the run
    # sooner and stamps the step it stopped at. See jobs/seed_watch.sh for why the
    # criterion is the oscillation-immune limits only.
    ./jobs/seed_watch.sh "$JOB" & WATCH_PID=$!
  fi
  ./docker/run_case.sh "$JOB_REL" "run.in" "$JOB/return/run.log"; RC_RUN=$?
  [ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null
  # A container the watcher stopped exits non-zero, which is not a failure -- but a run
  # that failed for any OTHER reason must still fail. Distinguish on the marker, and on
  # the artifact, never on the exit status alone (FASTEDDY_TRAPS.md 12).
  if [ "$RC_RUN" != "0" ] && [ ! -f "$JOB/output/.early_stop" ]; then
    die "the seed run"
  fi
  cat "$JOB/return/run.log" >> "$LOG"
fi
LAST=$(ls -1 "$JOB/output/$OUTBASE".* | sort -t. -k2 -n | tail -1)
[ -n "$LAST" ] || die "the run wrote no dump"
# EARLY STOP. With SEED_EARLY_STOP=1 a watcher scores the trailing window every 30
# simulated minutes and stops the run as soon as the oscillation-immune limits are in
# band, so the newest dump is legitimately short of Nt -- and Nt is then a CEILING rather
# than a target. It stamps the step it stopped at, so "short" is only accepted when
# something actually decided to stop there; a crash still fails, which is the distinction
# that matters.
if [ "${LAST##*.}" != "$TOTAL" ]; then
  if [ -f "$JOB/output/.early_stop" ] && \
     [ "$(cat "$JOB/output/.early_stop")" = "${LAST##*.}" ]; then
    echo "  EARLY STOP at step ${LAST##*.} of a $TOTAL ceiling "
    echo "    = $(python3 -c "print(f'{${LAST##*.}*$DT/3600:.2f}')") of "\
         "$(python3 -c "print(f'{$TOTAL*$DT/3600:.2f}')") simulated hours"
  else
    die "newest dump is step ${LAST##*.}, wanted $TOTAL and no .early_stop marker matches"
  fi
fi
# ASSERT ON THE ARTIFACT: the flux the run actually USED, not the one its .in asked for.
./docker/pyrun.sh - "${LAST#$ROOT/}" "$WTH" <<'PYCHK' || die "the run used the wrong surface flux"
import sys, numpy as np
from netCDF4 import Dataset
path, want = sys.argv[1], float(sys.argv[2])
with Dataset(path) as ds:
    got = float(np.asarray(ds["htFlux"][:]).mean())
if abs(got - want) > 1e-6:
    print(f"FATAL: the dump carries htFlux {got:+.6f}, the run asked for {want:+.6f}",
          file=sys.stderr)
    raise SystemExit(1)
print(f"  htFlux confirmed {got:+.6f}")
PYCHK

# ---- the gate, HERE, so the 300 s dumps never travel -------------------------------
# ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS (FASTEDDY_TRAPS.md 12): the gate's stdout is
# tee'd, so $? would be tee's. The verdict is read back out of the JSON it wrote.
# The gate scores the LAST 1.5 h, which is past the warm-up, so the flux it needs for the
# Kljun terms is the TARGET one and not the zero the first segment ran under.
./docker/pyrun.sh bin/seed_stationarity.py "$JOB_REL/output" --dt "$DT" --wth "$WTH" \
    --zm "$ZM" --k "$ZK" --dx "$DXG" ${SCORE_H:+--score-h $SCORE_H} \
    --json "$JOB_REL/return/stationarity.json" --label "$NAME" \
    2>&1 | tee "$JOB/return/stationarity.txt"
[ -s "$JOB/return/stationarity.json" ] \
  || { tail -20 "$JOB/return/stationarity.txt" >&2; die "the gate wrote no JSON"; }
VERDICT=$(python3 -c "import json;print('PASS' if json.load(open('$JOB/return/stationarity.json'))['pass'] else 'FAIL')")

# ---- what goes home ----------------------------------------------------------------
FINAL=$(ls -1 "$JOB/output/$OUTBASE".* | sort -t. -k2 -n | tail -1)
cp -f "$FINAL" "$JOB/return/seed_restart.nc" || die "could not stage the restart"
cp -f "$JOB/manifest.json" "$JOB/return/manifest.json"
python3 - "$JOB/return/manifest.json" "$JOB/return/stationarity.json" "$(basename "$FINAL")" <<'PY'
import json,sys
man=json.load(open(sys.argv[1])); st=json.load(open(sys.argv[2]))
man["achieved"]=st["final"]                 # LABEL THE SEED BY WHAT IT ACHIEVED, not by
man["achieved"]["pass"]=st["pass"]          # what it was asked for -- pick_seed.py reads
man["achieved"]["source_dump"]=sys.argv[3]  # these, and PROJECT_BRIEF.md says the same for
json.dump(man,open(sys.argv[1],"w"),indent=1) # direction.
PY
SZ=$(du -sh "$JOB/return" | cut -f1)
echo
echo "########## seed $NAME: $VERDICT ##########"
echo "  return/ is $SZ  ($(ls "$JOB/return" | tr '\n' ' '))"
date '+%F %H:%M:%S'
[ "$VERDICT" = "PASS" ] || exit 1
