#!/usr/bin/env bash
# The convective directional set, resumed after the adjustment step-count fix.
set -uo pipefail
cd /home/atyagi/Flux
WIN="${1:-2700}"; TB="${2:-900}"; DIRS="${3:-wW,wS,wE,wN}"
DT_T=0.0271739
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
CSPIN=$(ls -1 runs/cbl_spin/output/FE_CBL.* | sort -t. -k2 -n | tail -1)
say "CONVECTIVE: $DIRS   (spun-up state $CSPIN)"
ONLY="$DIRS" BASE=runs/g24_base/base_cbl.in bin/run_directions.sh cbl "$CSPIN" \
    data/grid_cbl $DT_T "$WIN" "$TB" || exit 1
python3 bin/fig_static.py --prefix cbl --cases wN,wE,wS,wW || true
python3 bin/fig_gate6.py --prefix cbl --title convective --grid data/grid_cbl || true
python3 bin/upwind_transect.py wS wN --prefix cbl --grid data/grid_cbl || true
./docker/pyrun.sh bin/stage6_predict.py cbl 2>&1 | tee results/cbl_stage6_gate.txt || true
say "CONVECTIVE SET COMPLETE"
