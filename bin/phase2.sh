#!/usr/bin/env bash
# Everything after the t_back decision, unattended.
#   usage: phase2.sh <window_seconds> <t_back>
set -uo pipefail
cd /home/atyagi/Flux
WIN="$1"; TB="$2"
DT_T=0.0271739     # terrain: CFL_3d 1.2347 -> 1.498 at the steepest cell
DT_F=0.0328947     # flat:    CFL_3d 1.4946
say(){ echo; echo "######## $* ########"; date '+%F %H:%M:%S'; }

# One GPU, one FastEddy. run_case.sh REFUSES to start a second one -- correctly, since two
# runs writing the same output/ silently interleave their dumps -- so wait here rather than
# letting the refusal abort the campaign.
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

# ---- 1. neutral, four directions. The adjusted states are already on disk from the
#         third pass and the surface has not changed, so the 25-minute adjustment is
#         reused rather than repeated.
say "NEUTRAL: four directions, ${WIN} s windows, t_back = ${TB} s"
REUSE_ADJ=1 bin/run_directions.sh g24 runs/g24_spin/output/FE_G24.547200 \
    data/grid $DT_T "$WIN" "$TB" || exit 1
python3 bin/fig_static.py --prefix g24 --cases wN,wE,wS,wW,flat || true
python3 bin/fig_gate6.py --prefix g24 --title neutral || true

# ---- 2. convective flat control. Straight off the spin-up: it is already flat and
#         uniform, so there is nothing to adjust to.
wait_gpu
say "CONVECTIVE: flat uniform control"
CSPIN=$(ls -1 runs/cbl_spin/output/FE_CBL.* | sort -t. -k2 -n | tail -1)
D=runs/cbl_flat; mkdir -p $D/window
BASE=runs/g24_base/base_cbl.in bin/run_window.sh $D "$CSPIN" $DT_F "$WIN" - 10.0 0.0 || exit 1
./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt $DT_F --tback "$TB" \
    --sgs-most --receptor-from data/grid --fp16-cache --tag cbl_flat 2>&1 \
    | grep -vE 'batch [0-9]+/' | tee results/cbl_flat.txt
# the sub-grid fraction of sigma_w^2 at the receptor, neutral vs convective -- the
# quantity Stage 5's gate is written on
./docker/pyrun.sh bin/subgrid_gate.py $D/window 2>&1 | tail -20 | tee results/cbl_subgrid.txt || true
rm -f $D/window/*

# ---- 3. convective, four directions, with the per-cell heat-flux map -----------------
say "CONVECTIVE: four directions"
BASE=runs/g24_base/base_cbl.in bin/run_directions.sh cbl "$CSPIN" \
    data/grid_cbl $DT_T "$WIN" "$TB" || exit 1
python3 bin/fig_static.py --prefix cbl --cases wN,wE,wS,wW,flat || true
python3 bin/fig_gate6.py --prefix cbl --title convective --grid data/grid_cbl || true
say "PHASE 2 COMPLETE"
