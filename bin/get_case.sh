#!/usr/bin/env bash
# ONE datetime in, ONE training record out. Everything else is scratch and is deleted.
#
#   usage: bin/get_case.sh 2023-07-15T19:00
#          bin/get_case.sh 2023-07-15T19:00 --keep-scratch     (debugging only)
#   env:   STUB_LES=1     run the whole path with the LES stubbed, for CPU verification
#          NPZ_DIR=...    where the record lands (default pairs_npz/)
#
# WHAT IT LEAVES BEHIND, AND IT IS EXACTLY ONE FILE:
#
#     pairs_npz/<case>.npz     scalars(6), kljun(128,128), target(128,128), meta
#
# plus the per-machine manifest.json line beside it. Nothing else survives -- not the
# sounding, not the forcing .in, not the seed-derived restart, not the ~19 GB of window the
# LES would write on the disk path, not the case grid, not the footprint JSON. A corpus is
# generated on rented machines that are destroyed afterwards, so a pipeline whose output is
# "a directory tree" has no output.
#
# THE SCRATCH IS REMOVED ON FAILURE TOO. A trap, not a line at the end: a case that dies at
# stage 6 leaves the largest scratch of any outcome, and that is exactly the case a
# retry-until-it-works loop meets most often.
#
# NO DEPENDENCE ON THIS MACHINE'S PATHS. The repo root comes from this script's location,
# the seed library and the grids come from inside it, and the staging tmpfs is discovered.
# The only requirements on a bare rented box are: this repo, the flux-fasteddy:cuda118
# image, an NVIDIA GPU, and jobs30/*/return/seed_restart.nc for the seeds it will use.
#
# IT CARRIES THE FULL 30 m PRODUCTION CONFIGURATION ITSELF. run_corpus_case.sh now DEFAULTS
# to that geometry as well (changed 2026-08-31), so this block is agreement rather than
# override -- but it is written out in full and then VERIFIED against what each stage
# actually read, because "the default is right" is a claim about a file somewhere else.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"

TS="${1:-}"
[ -n "$TS" ] || { echo "usage: get_case.sh <YYYY-MM-DDTHH:00> [--keep-scratch]" >&2; exit 64; }
KEEP=0; for f in "${@:2}"; do case "$f" in --keep-scratch) KEEP=1;;
  *) echo "unknown flag $f" >&2; exit 64;; esac; done

# ---- the timestamp must be a ROUND HOUR -------------------------------------------------
# HRRR analyses are hourly and the pseudo-sounding needs the `nat` hybrid-level profile,
# which is hourly-only. A :30 timestamp has no analysis behind it, so it is refused here
# rather than silently snapped -- snapping would produce a case labelled with a time whose
# forcing came from another one.
[[ "$TS" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:00$ ]] || {
  echo "FATAL: '$TS' is not a round-hour UTC timestamp (YYYY-MM-DDTHH:00). HRRR analyses" >&2
  echo "       are hourly and the nat-level sounding exists only on the hour." >&2; exit 64; }

_t="${TS//[-:]/}"; _t="${_t/T/}"; TAG="case_${_t:0:10}"
NPZ_DIR="${NPZ_DIR:-pairs_npz}"
OUT="$NPZ_DIR/$TAG.npz"
die(){ echo "FATAL: $*" >&2; exit 1; }

# ---- the split is decided HERE, before any work -----------------------------------------
# A month outside the corpus is refused before a byte is downloaded, and the split travels
# into the record. Nothing downstream re-derives it.
SPLIT=$(./docker/pyrun.sh - "$TS" <<'PY' 2>&1
import sys, datetime as dt
from lpdm.corpus import split_of
try:
    print(split_of(dt.datetime.fromisoformat(sys.argv[1])))
except ValueError as e:
    print(f"REFUSED {e}"); raise SystemExit(3)
PY
) || { echo "FATAL: ${SPLIT#REFUSED }" >&2; exit 3; }
SPLIT="$(echo "$SPLIT" | tr -d '\r' | tail -1)"
case "$SPLIT" in train|val|test) ;; *) die "split resolved to '$SPLIT'";; esac

echo "########## $TAG  ($TS)  split=$SPLIT ##########"

# ---- THE 30 m PRODUCTION CONFIGURATION, in full -----------------------------------------
export GRID=data/grid30_raised ZTARGET=28.5 EXACT_AGL=1
export TEMPLATE=runs/g30_base/base.in DX=30 ZCEILING=3000 DEFORM=0.346601 ZI_MAX_ABS=1250
export SEED_LIB=jobs30
export N_WINDOWS=1 ADJ_S=1800 WINDOW_S=2700 TBACK=900 REL_S=1800
export ALLOW_INDETERMINATE=1 ALLOW_DRIFTING=zi-neutral
export RING=1 RING_SELECTOR=1          # stage only: no window netCDF, ~19 GB never written
export COVER_GROUPS=10 KEEP_TD=100000 KEEP_FIELDS=0
export LPDM_WORKERS="${LPDM_WORKERS:-12}"
export NPZ_DIR CASE_SPLIT="$SPLIT"

# ---- scratch, and the trap that removes it ----------------------------------------------
# runs/<tag>/ is the LES scratch, data/case_grids/<tag>/ the per-case surface, and
# results/{soundings,forcing,pick,corpus}/<tag>.* the intermediate JSONs. All of it goes.
# The tmpfs staging directory goes too -- it lives outside the repo and a leaked one holds
# RAM until the box is destroyed.
RINGROOT="${FLUX_RINGROOT:-/dev/shm/flux}"
cleanup(){
  local rc=$?
  [ "$KEEP" = "1" ] && { echo "  --keep-scratch: leaving runs/$TAG and the intermediates";
                         return $rc; }
  docker rm -f "flux-fe-runs-$TAG" >/dev/null 2>&1
  rm -rf "runs/$TAG" "data/case_grids/$TAG" "$RINGROOT/$TAG"
  rm -f "results/soundings/$TAG.json" "results/forcing/$TAG.json" "results/pick/$TAG.json" \
        "results/corpus/$TAG.json" "results/corpus/$TAG.npz" "results/corpus/$TAG.txt" \
        "results/corpus/$TAG.monitor.json" "results/corpus/$TAG.sigmaw.json" \
        "pairs/$TAG.json"
  return $rc
}
trap cleanup EXIT

# ---- idempotent: a record on disk is the only evidence a case is done --------------------
if [ -s "$OUT" ]; then echo "  $OUT already exists; nothing to do"; exit 0; fi
mkdir -p "$NPZ_DIR"

# ---- the case ---------------------------------------------------------------------------
# STUB_LES=1 is handled INSIDE run_corpus_case.sh, at stages 6 and 7 only, so a stubbed run
# goes through the same driver, the same stages 1-5 and the same stage 8 as a real one.
bash bin/run_corpus_case.sh "$TS" "$TAG"; rc=$?
[ "$rc" = "3" ] && { echo "  SKIPPED: the domain cannot hold this case"; exit 3; }
[ "$rc" = "0" ] || die "run_corpus_case.sh returned $rc"

# ---- ASSERT ON THE ARTIFACT ------------------------------------------------------------
[ -s "$OUT" ] || die "the case completed but wrote no $OUT"
./docker/pyrun.sh bin/check_npz.py "$OUT" --expect-split "$SPLIT" --expect-datetime "$TS" ${STUB_LES:+--allow-stub} \
  || die "the record failed its own schema check"
echo
echo "########## $TAG COMPLETE -> $OUT ($(du -h "$OUT" | cut -f1)) ##########"
