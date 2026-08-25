#!/usr/bin/env bash
# Run ONE seed spin-up, end to end, on whatever machine this checkout is sitting on.
#
#   usage: jobs/run_seed.sh jobs/seed_cbl-mid_a030
#          jobs/run_seed.sh jobs/seed_cbl-mid_a030 --dry-run
#
# NO ABSOLUTE PATHS. The repo root is discovered from this script's own location, so a
# rented GPU can check the repo out anywhere; FLUX_ROOT is exported for docker/run_case.sh
# and docker/pyrun.sh, which is why those two now read it instead of a literal.
#
# NO SHARED STATE. Everything this job writes lands under its own job directory. Two seed
# jobs on two machines cannot interact; two on ONE machine are serialised by
# docker/run_case.sh, which refuses to start a second FastEddy container.
#
# RESUMABLE. The chain restarts from the newest dump on disk, so a killed job costs at most
# one segment. The gate is re-scored from scratch each time, which is cheap and CPU-only.
set -uo pipefail

JOB_ARG="${1:-}"
[ -n "$JOB_ARG" ] || { echo "usage: run_seed.sh <job_dir> [--dry-run]" >&2; exit 64; }
DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB="$(cd "$JOB_ARG" && pwd)" || { echo "FATAL: no job dir $JOB_ARG" >&2; exit 66; }
JOB_REL="${JOB#$ROOT/}"
export FLUX_ROOT="$ROOT"
cd "$ROOT"

die(){ echo "FATAL: $*" >&2; exit 1; }
[ -f "$JOB/manifest.json" ] || die "no manifest.json in $JOB"
[ -f "$JOB/seed.in" ] || die "no seed.in in $JOB"

read -r NAME DT FRQ NSEG SEG TOTAL WTH OUTBASE WALLMIN < <(python3 - "$JOB/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); r=m["run"]
print(m["job"], r["dt"], r["frqOutput"], r["n_segments"], r["steps_per_segment"],
      r["steps_total"], m["target"]["wth_virtual"], r["outFileBase"],
      r["projected_wall_min_per_segment"])
PY
) || die "manifest.json unreadable"

echo "########## seed $NAME ##########"
date '+%F %H:%M:%S'
echo "  dt $DT, dump every $FRQ steps, $NSEG x $SEG steps (proj ${WALLMIN} min wall each)"

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

# ---- the chain --------------------------------------------------------------------
IN=""; IPATH=""
LAST=$(ls -1 "$JOB/output/$OUTBASE".* 2>/dev/null | sort -t. -k2 -n | tail -1)
S0=1
if [ -n "$LAST" ]; then
  STEP=${LAST##*.}
  S0=$((STEP / SEG + 1))
  IN=$(basename "$LAST"); IPATH="./output/"
  echo "  resuming from $(basename "$LAST") (segment $S0 of $NSEG)"
fi
for s in $(seq "$S0" "$NSEG"); do
  NT=$((s * SEG))
  [ "$NT" -gt "$TOTAL" ] && NT=$TOTAL
  sed -e "s|^Nt = .*|Nt = $NT|" \
      -e "s|^inPath = .*|inPath = $IPATH|" -e "s|^inFile = .*|inFile = $IN|" \
      "$JOB/seed.in" > "$JOB/seg$s.in"
  echo "  --- segment $s/$NSEG -> step $NT ($(python3 -c "print(f'{$NT*$DT/60:.0f}')") min simulated)"
  date '+%F %H:%M:%S'
  ./docker/run_case.sh "$JOB_REL" "seg$s.in" "$JOB/return/seg$s.log" \
      || die "segment $s"
  cat "$JOB/return/seg$s.log" >> "$LOG"
  LAST=$(ls -1 "$JOB/output/$OUTBASE".* | sort -t. -k2 -n | tail -1)
  [ -n "$LAST" ] || die "segment $s wrote no dump"
  IN=$(basename "$LAST"); IPATH="./output/"
done

# ---- the gate, HERE, so the 300 s dumps never travel -------------------------------
# ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS (FASTEDDY_TRAPS.md 12): the gate's stdout is
# tee'd, so $? would be tee's. The verdict is read back out of the JSON it wrote.
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
