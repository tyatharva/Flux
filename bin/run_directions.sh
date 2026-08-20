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
BASE="${BASE:-runs/g24_base/base.in}"
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
  [ -n "${ONLY:-}" ] && [ "$ONLY" != "$NAME" ] && continue
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
frq=int(round(5.0/$DT)); print(int(round($ADJ_S/$DT/frq))*frq")
    sed -e "s|^dt = .*|dt = $DT|" -e "s|^Nt = .*|Nt = $A_NT|" \
        -e "s|^NtBatch = .*|NtBatch = $((A_NT/4))|" -e "s|^frqOutput = .*|frqOutput = $((A_NT/4))|" \
        -e "s|^inPath = .*|inPath = ./|" -e "s|^inFile = .*|inFile = FE_RST.0|" \
        -e "s|^topoFile = .*|topoFile = ./topo.bin|" \
        -e "s|^U_g = .*|U_g = $UGX|" -e "s|^V_g = .*|V_g = $VGY|" \
        -e "s|^outFileBase = .*|outFileBase = FE_ADJ|" \
        "$BASE" > "$D/adj.in"
    rm -f $D/output/FE_ADJ.*
    echo "--- adjustment: $A_NT steps = ${ADJ_S} s  [proj $(python3 -c "print(f'{$A_NT*0.0364/60:.1f}')") min]"
    ./docker/run_case.sh "$D" adj.in "$L/${PRE}_${NAME}_adj.log" || die "$NAME: adjustment"
    ADJ=$(ls -1 $D/output/FE_ADJ.* | sort -t. -k2 -n | tail -1)
    rm -f "$D/FE_RST.0"
  fi


  bin/run_window.sh "$D" "$ADJ" "$DT" "$WIN" ./topo.bin "$UGX" "$VGY" \
      || die "$NAME: window"

  ./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt "$DT" --tback "$TBACK" \
      --sgs-most --cover-dir data/grid --receptor-from data/grid --fp16-cache \
      --tag ${PRE}_$NAME 2>&1 | grep -vE 'batch [0-9]+/' | tee results/${PRE}_$NAME.txt
  # Peak storage is ONE window, never the sum (PROJECT_BRIEF.md).
  [ "${KEEP_FIELDS:-0}" = "1" ] || { rm -f $D/window/*; echo "--- window fields deleted"; }
done
echo; echo "########## $PRE DIRECTIONS COMPLETE ##########"; date '+%F %H:%M:%S'
