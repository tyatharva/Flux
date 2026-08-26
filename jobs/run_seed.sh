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

read -r NAME DT FRQ TOTAL WTH OUTBASE WALLMIN < <(python3 - "$JOB/manifest.json" <<'PYMAN'
import json,sys
m=json.load(open(sys.argv[1])); r=m["run"]
print(m["job"], r["dt"], r["frqOutput"], r["steps_total"],
      m["target"]["wth_virtual"], r["outFileBase"], r["projected_wall_min"])
PYMAN
) || die "manifest.json unreadable"

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
  # A COLD START, so inPath/inFile are empty and surflayer_wth is whatever the .in says.
  # With no restart to read, htFlux CANNOT be inherited -- which is the point of retiring
  # the chain. The assertion below is kept anyway: it costs seconds once per run, and
  # PROJECT_BRIEF.md's standing rule is to validate the state the model actually loaded, never
  # the config handed to it.
  sed -e "s|^Nt = .*|Nt = $TOTAL|" \
      -e "s|^inPath = .*|inPath = |" -e "s|^inFile = .*|inFile = |" \
      -e "s|^surflayer_wth = .*|surflayer_wth = $WTH|" \
      "$JOB/seed.in" > "$JOB/run.in"
  echo "  --- single invocation -> step $TOTAL ($(python3 -c "print(f'{$TOTAL*$DT/60:.0f}')") min simulated)"
  date '+%F %H:%M:%S'
  ./docker/run_case.sh "$JOB_REL" "run.in" "$JOB/return/run.log" || die "the seed run"
  cat "$JOB/return/run.log" >> "$LOG"
fi
LAST=$(ls -1 "$JOB/output/$OUTBASE".* | sort -t. -k2 -n | tail -1)
[ -n "$LAST" ] || die "the run wrote no dump"
[ "${LAST##*.}" = "$TOTAL" ] || die "newest dump is step ${LAST##*.}, wanted $TOTAL"
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
