#!/usr/bin/env bash
# Four wind directions from ONE spun-up state, on the static domain.
#
# The geography never moves; the flow turns. A square, doubly periodic, flat, uniform
# spin-up with dx = dy is exactly equivariant under a 90-degree rotation, so one expensive
# spin-up re-indexes into four directions -- and because terrain, roughness and the array
# are bit-identical in all four, any directional difference in the footprint is flow and
# cannot be a resampling artifact.
#
#   usage: run_directions.sh <prefix> <spin_dump> <grid_dir> <dt> <window_s> <tback>
#   env:   REUSE_ADJ=1   skip the adjustment if an FE_ADJ dump is already on disk
#          ADJ_S=1200    adjustment length in seconds
#          ONLY=wN       run a single direction
set -uo pipefail
cd /home/atyagi/Flux
PRE="$1"; SPIN="$2"; GRID="$3"; DT="$4"; WIN="$5"; TBACK="$6"
BASE="${BASE:-runs/g16_base/base.in}"
L=/tmp/flux-logs
ADJ_S="${ADJ_S:-1200}"
die(){ echo "FATAL: $*" >&2; exit 1; }
[ -f "$SPIN" ] || die "spin-up dump $SPIN not found"

# rot 0 leaves the geostrophic wind along +x -- BLOWING east, so FROM the west. Each +90
# deg turn takes (u,v) -> (-v,u), so the SOURCE direction runs W -> S -> E -> N. The
# surface wind is BACKED from the geostrophic one by the Ekman angle (12-24 deg measured
# here), so each case reports the direction it achieved rather than the one it was asked
# for; that is the label the gate uses.
CASES=("wW 0" "wS 1" "wE 2" "wN 3")
for spec in "${CASES[@]}"; do
  read -r NAME ROT <<<"$spec"
  # ONLY is a comma-separated allow-list, so a partially completed campaign can be
  # resumed from where it stopped without redoing the directions that finished.
  if [ -n "${ONLY:-}" ] && ! printf '%s' ",$ONLY," | grep -q ",$NAME,"; then continue; fi
  D=runs/${PRE}_$NAME
  mkdir -p "$D/output" "$D/window"
  echo; echo "########## $PRE / $NAME (rot ${ROT}x90) ##########"; date '+%F %H:%M:%S'
  read -r UGX VGY < <(python3 -c "
u,v=10.0,0.0
for _ in range($ROT): u,v=-v,u
print('%.6f %.6f'%(u,v))")
  cp -f "$GRID/topo.bin" "$D/topo.bin" || die "topo.bin"

  ADJ=$(ls -1 $D/output/FE_ADJ.* 2>/dev/null | sort -t. -k2 -n | tail -1)
  if [ "${REUSE_ADJ:-0}" = "1" ] && [ -n "$ADJ" ]; then
    echo "--- reusing the adjusted state $ADJ"
  else
    python3 bin/prep_restart.py "$SPIN" "$D/FE_RST.0" --rot "$ROT" --grid "$GRID" \
        || die "$NAME: prep_restart"
    A_NT=$(python3 -c "
frq=int(round(5.0/$DT)); print(int(round($ADJ_S/$DT/frq))*frq)")
    [ -n "$A_NT" ] && [ "$A_NT" -gt 0 ] 2>/dev/null \
        || die "$NAME: adjustment step count did not compute (got '$A_NT')"
    sed -e "s|^dt = .*|dt = $DT|" -e "s|^Nt = .*|Nt = $A_NT|" \
        -e "s|^NtBatch = .*|NtBatch = $((A_NT/4))|" -e "s|^frqOutput = .*|frqOutput = $((A_NT/4))|" \
        -e "s|^inPath = .*|inPath = ./|" -e "s|^inFile = .*|inFile = FE_RST.0|" \
        -e "s|^topoFile = .*|topoFile = ./topo.bin|" \
        -e "s|^U_g = .*|U_g = $UGX|" -e "s|^V_g = .*|V_g = $VGY|" \
        -e "s|^outFileBase = .*|outFileBase = FE_ADJ|" \
        "$BASE" > "$D/adj.in"
    rm -f $D/output/FE_ADJ.*
    SPS="${SPS:-0.0155}"
    echo "--- adjustment: $A_NT steps = ${ADJ_S} s  [proj $(python3 -c "print(f'{$A_NT*$SPS/60:.1f}')") min]"
    ./docker/run_case.sh "$D" adj.in "$L/${PRE}_${NAME}_adj.log" || die "$NAME: adjustment"
    ADJ=$(ls -1 $D/output/FE_ADJ.* | sort -t. -k2 -n | tail -1)
    rm -f "$D/FE_RST.0"
  fi


  bin/run_window.sh "$D" "$ADJ" "$DT" "$WIN" ./topo.bin "$UGX" "$VGY" \
      || die "$NAME: window"

  # The previous direction's analysis is CPU-only and this direction's LES was GPU-only,
  # so they overlapped. Only ONE analysis runs at a time -- each holds a ~10.5 GB fp16
  # field cache at 122^3 (28 GB at the retired 186^2 grid) -- so join it here, immediately
  # before starting the next.
  if [ -n "${APID:-}" ]; then wait "$APID" || echo "  (previous analysis exited non-zero)"; fi
  # Analysis in the background; the next direction's LES starts immediately. Peak storage
  # is at most TWO windows (~13 GB at this grid), never the sum over directions.
  (
    LPDM_WORKERS="${LPDM_WORKERS:-8}" \
    ./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt "$DT" --tback "$TBACK" \
        --sgs-most --cover-dir "$GRID" --receptor-from "$GRID" --fp16-cache \
        --z-target "${ZTARGET:-10.0}" ${EXACT_AGL:+--exact-agl} \
        --tag ${PRE}_$NAME 2>&1 | grep -vE 'batch [0-9]+/' > results/${PRE}_$NAME.txt
    tail -32 results/${PRE}_$NAME.txt
    [ "${KEEP_FIELDS:-0}" = "1" ] || { rm -f $D/window/*; echo "--- $NAME window fields deleted"; }
  ) &
  APID=$!
done
[ -n "${APID:-}" ] && wait "$APID"
echo; echo "########## $PRE DIRECTIONS COMPLETE ##########"; date '+%F %H:%M:%S'
