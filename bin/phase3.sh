#!/usr/bin/env bash
# How much is the sigma_w floor's similarity anchor worth?
#
# The convective flat control put the floor at 3.37x at the receptor, which is a large
# intervention, and the two standard relations it could be anchored to disagree by 25% at
# a 30 m receptor under a 900 m CBL. That is the largest remaining modelling freedom in
# the convective footprints, so it gets measured rather than argued about.
#
# Re-runs the convective flat window (the case where similarity is valid and Kljun is
# diagnostic), keeps the fields, and scores it four ways on IDENTICAL fields and an
# identical release ensemble -- so every difference is the closure and nothing else.
set -uo pipefail
cd /home/atyagi/Flux
WIN="${1:-2700}"; TB="${2:-900}"
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
CSPIN=$(ls -1 runs/cbl_spin/output/FE_CBL.* | sort -t. -k2 -n | tail -1)
D=runs/cbl_flat; mkdir -p $D/window
say "sigma_w floor sensitivity: re-running the convective flat window"
BASE=runs/g24_base/base_cbl.in bin/run_window.sh $D "$CSPIN" $DT_F "$WIN" - 10.0 0.0 || exit 1

for v in "surface cbl_flat" "blend cbl_flat_blend" "mixed cbl_flat_mixed"; do
  read -r MODE TAG <<<"$v"
  say "closure variant: --sgs-most-mode $MODE  -> $TAG"
  ./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt $DT_F --tback "$TB" \
      --sgs-most --sgs-most-mode "$MODE" --receptor-from data/grid --fp16-cache \
      --tag "$TAG" 2>&1 | grep -vE 'batch [0-9]+/' | tee results/$TAG.txt
done
say "closure variant: NO floor at all -> cbl_flat_nofloor"
./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt $DT_F --tback "$TB" \
    --receptor-from data/grid --fp16-cache --tag cbl_flat_nofloor 2>&1 \
    | grep -vE 'batch [0-9]+/' | tee results/cbl_flat_nofloor.txt
rm -f $D/window/*
say "PHASE 3 COMPLETE"
