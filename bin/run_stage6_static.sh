#!/usr/bin/env bash
# Stage 6 on the static 24 m domain: four wind directions from ONE spun-up state.
#
# The geography never moves. Direction is changed by rotating the FLOW by 90 degrees and
# rotating (U_g, V_g) with it -- a square, doubly periodic, flat, uniform spin-up with
# dx = dy is exactly equivariant under that rotation, so four directions cost one spin-up.
#
# Per direction: rotate + inject the surface -> 20 min of adjustment -> two 20-min
# sampling segments at 5 s cadence -> footprint -> DELETE THE FIELDS. Peak storage is one
# window, never the sum (PROJECT_BRIEF.md).
#
# Every run is projected before launch and every one is under the 45-minute cap:
#   adjustment      44,160 steps @ 0.0359 s = 26.4 min
#   sampling seg    44,160 steps @ 0.0359 s = 26.4 min   (x2)
set -uo pipefail
cd /home/atyagi/Flux
L=/tmp/flux-logs
BASE=runs/g24_base/base.in
SPIN=runs/g24_spin
DT_FLAT=0.0328947          # 5/152 s,  CFL_3d 1.4946
DT_TERR=0.0271739          # 5/184 s,  CFL_3d 1.2347 -> 1.4977 at the steepest cell
ADJ_STEPS=44160            # 1200 s of simulated adjustment
SEG_STEPS=44160            # 1200 s per sampling segment, two per direction
FRQ=184                    # 5 s at the terrain dt
KEEP="${KEEP_FIELDS:-0}"   # 1 = keep the window fields after the footprint

die(){ echo "FATAL: $*" >&2; exit 1; }

SRC=$(ls -1 $SPIN/output/FE_G24.* 2>/dev/null | sort -t. -k2 -n | tail -1)
[ -n "$SRC" ] || die "no spin-up dump found in $SPIN/output"
echo "### spun-up state: $SRC"

# name, rotation (90 deg CCW turns), flag
CASES=("flat 0 --flat" "wN 1 " "wE 2 " "wS 3 " "wW 0 ")
# rot 0 puts the geostrophic wind along +x (blowing east) => surface wind FROM the west.
# Each +90 deg turn advances the source direction by 90 deg: W -> N -> E -> S.

for spec in "${CASES[@]}"; do
  read -r NAME ROT FLAG <<<"$spec"
  D=runs/g24_$NAME
  mkdir -p $D/output
  echo; echo "############################## $NAME (rot ${ROT}x90) ##############################"

  # ---- 1. build the restart: rotate the flow, inject the fixed geography ----------
  UG=$(python3 -c "
u,v=10.0,0.0
for _ in range($ROT): u,v=-v,u
print('%.6f %.6f'%(u,v))")
  read -r UGX VGY <<<"$UG"
  python3 bin/prep_restart.py "$SRC" "$D/restart.nc" --rot "$ROT" $FLAG \
      || die "$NAME: prep_restart failed"

  # ---- 2. adjustment: real surface, flow adjusts before anything is sampled --------
  if [ "$NAME" = "flat" ]; then DT=$DT_FLAT; TOPO=""; else DT=$DT_TERR; TOPO="./topo.bin"; fi
  [ -n "$TOPO" ] && cp -f data/grid/topo.bin $D/topo.bin
  A_NT=$ADJ_STEPS
  sed -e "s|^dt = .*|dt = $DT|" -e "s|^Nt = .*|Nt = $A_NT|" \
      -e "s|^NtBatch = .*|NtBatch = $((A_NT/4))|" -e "s|^frqOutput = .*|frqOutput = $((A_NT/4))|" \
      -e "s|^inPath = .*|inPath = ./|" -e "s|^inFile = .*|inFile = restart.nc|" \
      -e "s|^topoFile = .*|topoFile = $TOPO|" \
      -e "s|^U_g = .*|U_g = $UGX|" -e "s|^V_g = .*|V_g = $VGY|" \
      -e "s|^outFileBase = .*|outFileBase = FE_ADJ|" \
      $BASE > $D/adj.in
  echo "--- adjustment: $A_NT steps, dt=$DT, U_g=$UGX V_g=$VGY  [proj 26.4 min]"
  ./docker/run_case.sh $D adj.in "$L/g24_${NAME}_adj.log" || die "$NAME: adjustment failed"
  ADJ=$(ls -1 $D/output/FE_ADJ.* | sort -t. -k2 -n | tail -1)
  echo "--- adjusted state: $ADJ"

  # ---- 3. sampling, two segments, 5 s cadence, lean 16-bit output -----------------
  mkdir -p $D/window && rm -f $D/window/*
  PREV=$(basename "$ADJ"); PSTEP=${PREV##*.}
  for seg in 1 2; do
    NT=$((PSTEP + SEG_STEPS*seg))
    IN=$([ $seg -eq 1 ] && echo "$PREV" || echo "FE_WIN.$((PSTEP+SEG_STEPS))")
    IPATH=$([ $seg -eq 1 ] && echo "./output/" || echo "./window/")
    sed -e "s|^dt = .*|dt = $DT|" -e "s|^Nt = .*|Nt = $NT|" \
        -e "s|^NtBatch = .*|NtBatch = $FRQ|" -e "s|^frqOutput = .*|frqOutput = $FRQ|" \
        -e "s|^inPath = .*|inPath = $IPATH|" -e "s|^inFile = .*|inFile = $IN|" \
        -e "s|^topoFile = .*|topoFile = $TOPO|" \
        -e "s|^U_g = .*|U_g = $UGX|" -e "s|^V_g = .*|V_g = $VGY|" \
        -e "s|^outPath = .*|outPath = ./window/|" \
        -e "s|^outFileBase = .*|outFileBase = FE_WIN|" \
        -e "s|^ioOutputMode = .*|ioOutputMode = 0|" \
        $BASE > $D/win$seg.in
    # ioLPDMmode is appended rather than substituted: it is a NEW parameter on the fork
    # and the stock base.in has no line for it.
    if [ "${USE_LPDM_MODE:-1}" = "1" ]; then echo "ioLPDMmode = 1" >> $D/win$seg.in; fi
    echo "--- window seg $seg: -> step $NT, frq=$FRQ (5 s)  [proj 26.4 min]"
    ./docker/run_case.sh $D win$seg.in "$L/g24_${NAME}_win$seg.log" \
        || die "$NAME: window segment $seg failed"
  done
  echo "--- dumps: $(ls $D/window | wc -l), $(du -sh $D/window | cut -f1)"

  # ---- 4. footprint + land-cover attribution --------------------------------------
  COVER=$([ "$NAME" = "flat" ] && echo "" || echo "--cover-dir data/grid")
  ./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt $DT --res 60 \
      $COVER --tag g24_$NAME 2>&1 | grep -vE 'batch [0-9]+/|loaded ' \
      | tee results/g24_$NAME.txt
  ./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt $DT --res 24 --ml-raster \
      $COVER --tag g24_${NAME}_ml 2>&1 | tail -3

  # ---- 5. delete the fields. Peak storage is one window, never the sum. -----------
  if [ "$KEEP" != "1" ]; then rm -f $D/window/*; echo "--- window fields deleted"; fi
done
echo; echo "########## STAGE 6 STATIC COMPLETE ##########"
