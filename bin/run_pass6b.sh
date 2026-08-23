#!/usr/bin/env bash
# The closure pass, second half: validate the sub-grid-weighted floor in BOTH regimes and
# BOTH directions, measure what the retired closure was worth, and regenerate production.
#
# Resumable and self-freezing. Sentinels in results/.done6b/.
#   usage: bin/run_pass6b.sh      env: FORCE=1 to redo
#
# WALL RULE: every LES goes through run_window.sh / run_directions.sh, which project the
# segment before launching and refuse. Longest single run is a 2400 s window, 42.4 min.
set -uo pipefail
cd /home/atyagi/Flux
if [ "${FROZEN:-0}" != "1" ]; then
  cp -f "$0" /tmp/flux-logs/run_pass6b.frozen.sh
  FROZEN=1 exec bash /tmp/flux-logs/run_pass6b.frozen.sh "$@"
fi
L=/tmp/flux-logs; R=results; DONE=$R/.done6b
mkdir -p $DONE $R $L
DT_WIN=0.0146199; TB=600; WIN=2400; NPART=40000
say(){ echo; echo "########## $* ##########"; date '+%F %H:%M:%S'; }
die(){ echo "PASS 6B STOPPED: $*" | tee -a $R/pass6b_status.txt >&2; exit 1; }
newest(){ ls -1 "$1"/*.[0-9]* 2>/dev/null | sort -t. -k2 -n | tail -1; }
have(){ [ "${FORCE:-0}" = "1" ] && return 1; [ -f "$DONE/$1" ]; }
mark(){ date '+%F %H:%M:%S' > "$DONE/$1"; echo "  [stage $1 recorded done]"; }

# ---------------------------------------------------------------------- the battery
# Three closures on ONE window: production, the retired taper, and no floor at all. The
# no-floor row is not decoration -- it is what says whether a failure belongs to the floor
# or to the model, and it is the row that localised the last two failures.
battery(){                      # battery <windowdir> <grid> <tag>
  local W="$1" G="$2" T="$3"
  for CFG in new legacy nofloor; do
    local F="--sgs-most"
    case $CFG in legacy) F="--sgs-most --sgs-most-legacy" ;; nofloor) F="" ;; esac
    echo "--- well-mixed $T / $CFG"
    ./docker/pyrun.sh bin/stage4_wellmixed.py "$W" --dt $DT_WIN --n $NPART \
        --z-target 10.0 --dmap "$G/dmap.npy" $F --fp16-cache --tag ${T}_wm_$CFG \
        2>&1 | grep -vE '^    loaded' > $R/${T}_wm_$CFG.txt
    grep -E "MOST floor|turnovers|GATE:|BOTH DIR" $R/${T}_wm_$CFG.txt
    echo "--- footprint  $T / $CFG"
    ./docker/pyrun.sh bin/stage5_footprint.py "$W" --dt $DT_WIN --tback $TB \
        --z-target 10.0 --rel-seconds 1800 $F --receptor-from "$G" --cover-dir "$G" \
        --fp16-cache --cover-groups 10 --tag ${T}_$CFG 2>&1 \
        | grep -vE 'batch [0-9]+/|^    loaded' > $R/${T}_$CFG.txt
  done
  python3 bin/floor_bias.py new=${T}_new legacy=${T}_legacy none=${T}_nofloor \
      2>&1 | tee $R/${T}_floorbias.txt
}

# ============================================== 1-2. the two control batteries, concurrent
# Each battery is serial internally (one 10.6 GB field cache at a time); the two run
# together because the machine has 62 GB and the retired one-analysis-at-a-time rule was
# written for the 186^2 grid at 28 GB per cache.
if ! have controls; then
  say "1. flat/neutral control window"
  BASE=runs/g16_base/base.in bin/run_window.sh runs/g16_flat \
      runs/g16_flat/output/FE_ADJ.0 $DT_WIN $WIN - 10.000000 0.000000 || die "nbl window"
  say "2. flat/convective control window"
  SRC=$(newest runs/g16_cbl_shallow/output)
  BASE=runs/g16_base/base_cbl_shallow.in bin/run_window.sh runs/g16_flatcbl "$SRC" \
      $DT_WIN $WIN - 10.000000 0.000000 || die "cbl window"
  say "3. both closure batteries, concurrently"
  ( battery runs/g16_flat/window    data/grid16     g16p6b_flat    > $L/bat_nbl.out 2>&1 ) &
  P1=$!
  ( battery runs/g16_flatcbl/window data/grid16_cbl g16p6b_flatcbl > $L/bat_cbl.out 2>&1 ) &
  P2=$!
  wait $P1; R1=$?; wait $P2; R2=$?
  cat $L/bat_nbl.out; cat $L/bat_cbl.out
  [ $R1 -eq 0 ] && [ $R2 -eq 0 ] || die "a control battery failed"
  TAGJSON=$R/g16p6b_flat_new.json bin/regression_flat.sh --compare-only \
      2>&1 | tail -12 | tee $R/g16p6b_regression.txt
  mark controls
fi

# ============================================ 4-5. production on the RAISED map, both regimes
for REG in nbl cbl; do
  if have dirs_$REG; then continue; fi
  say "production, $REG, raised topography"
  case $REG in
    nbl) SRC=$(newest runs/g16_spin/output);        GRID=data/grid16r_nbl
         RBASE=runs/g16_base/base.in ;;
    cbl) SRC=$(newest runs/g16_cbl_shallow/output); GRID=data/grid16_raised
         RBASE=runs/g16_base/base_cbl_shallow.in ;;
  esac
  [ -f "$GRID/topo.bin" ] || die "$GRID not built"
  # MINIMUM PRODUCTION SPOT CHECK, not the full corpus. The closure is validated on the two
  # flat control windows, which cost no GPU because they are already on disk -- but a flat
  # uniform window never exercises terrain, the array, the raised topography or the
  # fractional receptor, so one case per regime confirms the production path end to end.
  # The northerly is the right one: largest array share (78-81% convective) and the
  # direction the displacement-height sensitivity was measured on.
  #
  # All four directions per regime is 8.4 GPU-h and is DEFERRED -- that updates corpus
  # numbers rather than confirming correctness. DIRS= takes a comma-separated list, so
  # widening it later is one word, and the sentinels make it resumable.
  BASE=$RBASE ADJ_S=1200 SPS=0.0155 ZTARGET=8.5 EXACT_AGL=1 KEEP_FIELDS=1 \
    ONLY="${DIRS:-wN}" \
    bin/run_directions.sh g16r_$REG "$SRC" "$GRID" $DT_WIN $WIN $TB || die "dirs $REG"
  mark dirs_$REG
done

# =================================== 6. the retired closure on the production fields
if ! have legacy_prod; then
  say "the retired closure on the production fields"
  for C in $(ls -d runs/g16r_*_w? 2>/dev/null | xargs -r -n1 basename); do
    D=runs/$C; G=data/grid16_raised
    case $C in g16r_nbl_*) G=data/grid16r_nbl ;; esac
    [ -d "$D/window" ] && [ "$(ls -1 $D/window/*.[0-9]* 2>/dev/null | wc -l)" -gt 10 ] || \
      { echo "  (no retained fields for $C -- skipped)"; continue; }
    ./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt $DT_WIN --tback $TB \
        --z-target 8.5 --exact-agl --sgs-most --sgs-most-legacy --receptor-from "$G" \
        --cover-dir "$G" --fp16-cache --cover-groups 10 --tag ${C}_legacy 2>&1 \
        | grep -vE 'batch [0-9]+/|^    loaded' > $R/${C}_legacy.txt
    python3 bin/floor_bias.py new=$C legacy=${C}_legacy 2>&1 | tee $R/${C}_floorbias.txt
  done
  mark legacy_prod
fi

if ! have cleanup; then
  say "releasing the retained window fields"
  for D in runs/g16r_*/window runs/g16_flatcbl/window runs/g16_flat/window; do
    rm -f $D/*.[0-9]* 2>/dev/null; done
  df -h /home/atyagi | tail -1
  mark cleanup
fi
say "PASS 6B COMPLETE"
echo "COMPLETE $(date '+%F %H:%M:%S')" >> $R/pass6b_status.txt
