#!/usr/bin/env bash
# Resume the campaign after the westerly neutral case, which completed.
#   usage: phase2b.sh <window_seconds> <t_back> [directions]
set -uo pipefail
cd /home/atyagi/Flux
WIN="$1"; TB="$2"; DIRS="${3:-wS,wE,wN}"
DT_T=0.0271739
DT_F=0.0328947
say(){ echo; echo "######## $* ########"; date '+%F %H:%M:%S'; }
wait_gpu(){
  local n=0
  while docker ps -q --filter ancestor=flux-fasteddy:cuda118 | while read c; do
          docker inspect -f '{{json .Config.Cmd}}' "$c" 2>/dev/null | grep -q FEMAIN/FastEddy && echo busy
        done | grep -q busy; do
    [ $((n % 20)) -eq 0 ] && echo "  waiting for the GPU to free up ($((n/2)) min)"
    sleep 30; n=$((n+1))
  done
}
wait_gpu
say "NEUTRAL: $DIRS"
ONLY="$DIRS" REUSE_ADJ=1 bin/run_directions.sh g24 runs/g24_spin/output/FE_G24.547200 \
    data/grid $DT_T "$WIN" "$TB" || exit 1
python3 bin/fig_static.py --prefix g24 --cases wN,wE,wS,wW,flat || true
python3 bin/fig_gate6.py --prefix g24 --title neutral || true
./docker/pyrun.sh bin/stage6_predict.py g24 2>&1 | tee results/g24_stage6_gate.txt || true

wait_gpu
say "CONVECTIVE: flat uniform control"
CSPIN=$(ls -1 runs/cbl_spin/output/FE_CBL.* | sort -t. -k2 -n | tail -1)
D=runs/cbl_flat; mkdir -p $D/window
BASE=runs/g24_base/base_cbl.in bin/run_window.sh $D "$CSPIN" $DT_F "$WIN" - 10.0 0.0 || exit 1
./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt $DT_F --tback "$TB" \
    --sgs-most --receptor-from data/grid --fp16-cache --tag cbl_flat 2>&1 \
    | grep -vE 'batch [0-9]+/' | tee results/cbl_flat.txt
./docker/pyrun.sh bin/subgrid_gate.py $D/window 2>&1 | tail -22 | tee results/cbl_subgrid.txt || true
./docker/pyrun.sh bin/subgrid_gate.py runs/g24_flat/window 2>&1 | tail -22 \
    | tee results/g24_subgrid.txt || true
rm -f $D/window/*

wait_gpu
say "CONVECTIVE: four directions"
BASE=runs/g24_base/base_cbl.in bin/run_directions.sh cbl "$CSPIN" \
    data/grid_cbl $DT_T "$WIN" "$TB" || exit 1
python3 bin/fig_static.py --prefix cbl --cases wN,wE,wS,wW || true
python3 bin/fig_gate6.py --prefix cbl --title convective --grid data/grid_cbl || true
./docker/pyrun.sh bin/stage6_predict.py cbl 2>&1 | tee results/cbl_stage6_gate.txt || true
say "PHASE 2 COMPLETE"
