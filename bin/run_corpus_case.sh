#!/usr/bin/env bash
# ONE corpus case, end to end: a timestamp in, an (input, target) training pair out.
#
#   usage: bin/run_corpus_case.sh 2023-01-18T18:00 [tag]
#   env:   WINDOW_S=2400  ADJ_S=1800  TBACK=600  KEEP_FIELDS=0  GRID=data/grid16
#          SKIP_LES=1     stop after stage 4, for a dry run with no GPU
#
# The eight stages, and which file owns each:
#
#   1  HRRR pseudo-sounding at the tower        bin/hrrr_sounding.py
#   2  sounding -> FastEddy .in parameters      bin/sounding_to_forcing.py
#   4  which seed, and which 90-degree rotation bin/pick_seed.py
#   5  rotate the seed's FLOW, inject the       bin/prep_restart.py
#      STATIC surface (terrain, z0, htFlux)
#   6a ~30 min adjustment under THIS case's forcing
#   6b the (30 min + t_back) sampling window    bin/run_window.sh
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
GRID="${GRID:-data/grid16}"
ADJ_S="${ADJ_S:-1800}"
WINDOW_S="${WINDOW_S:-2400}"
TBACK="${TBACK:-$(cat results/tback_production.txt 2>/dev/null || echo 600)}"
D="runs/$TAG"
L=/tmp/flux-logs
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

# ---- 4. the seed -----------------------------------------------------------------
say "$TAG  stage 4: pick a seed"
PICK=results/pick/$TAG.json
./docker/pyrun.sh bin/pick_seed.py "$FRC" --json "$PICK" || true
[ -s "$PICK" ] || die "stage 4 wrote no pick json"
read -r JOB ROT < <(python3 -c "
import json; c=json.load(open('$PICK'))['chosen']; print(c['job'], c['rot'])")
SEED="jobs/$JOB/return/seed_restart.nc"
[ "${SKIP_LES:-0}" = "1" ] && { echo "  SKIP_LES=1: stopping after stage 4"; exit 0; }
[ -f "$SEED" ] || die "seed $SEED has not been spun up yet; run jobs/run_seed.sh $JOB"

# ---- 5. rotate the flow, inject the static surface -------------------------------
say "$TAG  stage 5: seed $JOB rot $ROT -> restart"
./docker/pyrun.sh bin/prep_restart.py "$SEED" "$D/FE_RST.0" --rot "$ROT" --grid "$GRID" \
    || die "prep_restart"
[ -s "$D/FE_RST.0" ] || die "prep_restart wrote no restart"
cp -f "$GRID/topo.bin" "$D/topo.bin" || die "topo.bin"

# ---- 6a. adjustment --------------------------------------------------------------
say "$TAG  stage 6a: ${ADJ_S}s adjustment under this case's forcing"
A_NT=$(python3 -c "
frq=int(round(5.0/$DT)); print(int(round($ADJ_S/$DT/frq))*frq)")
[ "$A_NT" -gt 0 ] 2>/dev/null || die "adjustment step count did not compute"
sed -e "s|^Nt = .*|Nt = $A_NT|" -e "s|^NtBatch = .*|NtBatch = $((A_NT/4))|" \
    -e "s|^frqOutput = .*|frqOutput = $((A_NT/4))|" \
    -e "s|^inPath = .*|inPath = ./|" -e "s|^inFile = .*|inFile = FE_RST.0|" \
    -e "s|^topoFile = .*|topoFile = ./topo.bin|" \
    -e "s|^outFileBase = .*|outFileBase = FE_ADJ|" \
    "$D/case.in" > "$D/adj.in"
rm -f $D/output/FE_ADJ.*
./docker/run_case.sh "$D" adj.in "$L/${TAG}_adj.log" || die "adjustment"
ADJ=$(ls -1 $D/output/FE_ADJ.* 2>/dev/null | sort -t. -k2 -n | tail -1)
[ -n "$ADJ" ] || die "the adjustment wrote no dump"
rm -f "$D/FE_RST.0"

# ---- 6b. the sampling window -----------------------------------------------------
say "$TAG  stage 6b: ${WINDOW_S}s window (30 min averaging + ${TBACK}s t_back)"
BASE="$D/case.in" bin/run_window.sh "$D" "$ADJ" "$DT" "$WINDOW_S" ./topo.bin "$UG" "$VG" \
    || die "window"

# ---- 7. the footprint ------------------------------------------------------------
say "$TAG  stage 7: backward LPDM"
LPDM_WORKERS="${LPDM_WORKERS:-8}" \
./docker/pyrun.sh bin/stage5_footprint.py "$D/window" --dt "$DT" --tback "$TBACK" \
    --sgs-most --cover-dir "$GRID" --receptor-from "$GRID" --fp16-cache \
    --z-target "${ZTARGET:-10.0}" --rel-seconds 1800 \
    --tag "$TAG" 2>&1 | grep -vE 'batch [0-9]+/' > "results/$TAG.txt"
[ -s "results/$TAG.json" ] || { tail -12 "results/$TAG.txt" >&2
  die "stage 7 produced no footprint json"; }

# ---- 8. the training record ------------------------------------------------------
say "$TAG  stage 8: assemble the pair"
./docker/pyrun.sh bin/make_pair.py --tag "$TAG" --footprint "results/$TAG.json" \
    --forcing "$FRC" --seed "$PICK" --outdir pairs || true
[ -s "pairs/$TAG.json" ] || die "stage 8 wrote no pair"

[ "${KEEP_FIELDS:-0}" = "1" ] || { rm -f $D/window/*; echo "  window fields deleted"; }
say "$TAG COMPLETE"
