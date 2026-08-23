#!/usr/bin/env bash
# THE CLOSURE PASS: fix the sigma_w floor, revalidate it in both directions and both
# regimes, measure what the retired form was worth, and regenerate production on it.
#
# Resumable and self-freezing, exactly like bin/run_campaign.sh: every stage drops a
# sentinel in results/.done6/, and the script re-execs from a snapshot so that editing it
# mid-run cannot drop bash into the middle of a line.
#
#   usage: bin/run_pass6.sh [stage ...]      (default: all, in order)
#   env:   FORCE=1   ignore sentinels and redo
#
# WALL RULE. Every LES invocation goes through bin/run_window.sh or bin/run_directions.sh,
# both of which project the segment length before launching and REFUSE rather than ask.
# Longest single run here is a 2400 s window: 164,159 steps x 0.0155 s = 42 min.
set -uo pipefail
cd /home/atyagi/Flux
if [ "${FROZEN:-0}" != "1" ]; then
  cp -f "$0" /tmp/flux-logs/run_pass6.frozen.sh
  FROZEN=1 exec bash /tmp/flux-logs/run_pass6.frozen.sh "$@"
fi

L=/tmp/flux-logs
R=results
DONE=$R/.done6
mkdir -p $DONE $R $L
DT_WIN=0.0146199          # 5/342: a 5 s cadence is an integer step count, CFL_3d 1.348
TB=600                    # measured in the fifth pass: converged at 500 s, x1.25
WIN=2400                  # 1800 s averaging period + t_back
NPART="${NPART:-40000}"

say(){ echo; echo "########## $* ##########"; date '+%F %H:%M:%S'; }
die(){ echo "PASS 6 STOPPED: $*" | tee -a $R/pass6_status.txt >&2; exit 1; }
newest(){ ls -1 "$1"/*.[0-9]* 2>/dev/null | sort -t. -k2 -n | tail -1; }
have(){ [ "${FORCE:-0}" = "1" ] && return 1; [ -f "$DONE/$1" ]; }
mark(){ date '+%F %H:%M:%S' > "$DONE/$1"; echo "  [stage $1 recorded done]"; }

# ---------------------------------------------------------------- the validation battery
# Run on ONE window, in every closure configuration production could use. The point of the
# pass: the neutral PASS previously tested the UNMODIFIED model, because the floor is
# inactive neutrally -- so "inherited" was never a valid word for the convective closure.
battery(){                      # battery <windowdir> <grid> <tag> <dmap>
  local W="$1" G="$2" T="$3"
  for CFG in new legacy nofloor; do
    local FLAGS="--sgs-most"
    case $CFG in
      legacy)  FLAGS="--sgs-most --sgs-most-legacy" ;;
      nofloor) FLAGS="" ;;
    esac
    echo "--- well-mixed, $T, closure=$CFG"
    ./docker/pyrun.sh bin/stage4_wellmixed.py "$W" --dt $DT_WIN --n $NPART \
        --z-target 10.0 --dmap "$G/dmap.npy" $FLAGS --fp16-cache \
        --tag ${T}_wm_$CFG 2>&1 | tee $R/${T}_wm_$CFG.txt
    echo "--- footprint,  $T, closure=$CFG"
    ./docker/pyrun.sh bin/stage5_footprint.py "$W" --dt $DT_WIN --tback $TB \
        --z-target 10.0 --rel-seconds 1800 $FLAGS --receptor-from "$G" \
        --cover-dir "$G" --fp16-cache --cover-groups 10 --tag ${T}_$CFG 2>&1 \
        | grep -vE 'batch [0-9]+/' > $R/${T}_$CFG.txt
    tail -6 $R/${T}_$CFG.txt
  done
  python3 bin/floor_bias.py new=${T}_new legacy=${T}_legacy none=${T}_nofloor \
      2>&1 | tee $R/${T}_floorbias.txt
}

# ==================================================== 1. flat / neutral control + battery
if ! have flat_nbl; then
  say "1. flat/neutral control window (2400 s) and the full closure battery"
  SRC=runs/g16_flat/output/FE_ADJ.0
  [ -f "$SRC" ] || die "flat neutral restart $SRC missing"
  BASE=runs/g16_base/base.in bin/run_window.sh runs/g16_flat "$SRC" $DT_WIN $WIN - \
      10.000000 0.000000 || die "flat neutral window"
  battery runs/g16_flat/window data/grid16 g16p6_flat
  # The standing regression, on the SAME fields, in the production closure.
  TAGJSON=$R/g16p6_flat_new.json bin/regression_flat.sh --compare-only \
      2>&1 | tail -20 | tee $R/g16p6_regression.txt
  rm -f runs/g16_flat/window/*
  mark flat_nbl
fi

# ================================================ 2. flat / convective control + battery
if ! have flat_cbl; then
  say "2. flat/convective control window (2400 s) and the full closure battery"
  SRC=$(newest runs/g16_cbl_shallow/output)
  [ -n "$SRC" ] || die "convective spin-up state missing"
  mkdir -p runs/g16_flatcbl
  BASE=runs/g16_base/base_cbl_shallow.in bin/run_window.sh runs/g16_flatcbl "$SRC" \
      $DT_WIN $WIN - 10.000000 0.000000 || die "flat convective window"
  battery runs/g16_flatcbl/window data/grid16_cbl g16p6_flatcbl
  rm -f runs/g16_flatcbl/window/*
  mark flat_cbl
fi

# ==================================================== 3-4. production, on the RAISED map
# --raise-topo becomes the default. At z0_array = 0.10 m the array is aerodynamically
# IDENTICAL to the WorldCover cropland it replaced, so its entire neutral signal is zero
# and the share is pure geometry; raising topoPos by d restores a 2.5x roughness contrast
# AND puts the first model level above panel top. The receptor is then released at a
# FRACTIONAL level 8.500 m above the raised surface = 10.000 m above BARE GROUND.
for REG in nbl cbl; do
  if have dirs6_$REG; then continue; fi
  say "$( [ $REG = nbl ] && echo 3 || echo 4 ). production, $REG, raised topography"
  case $REG in
    nbl) SRC=$(newest runs/g16_spin/output);        GRID=data/grid16r_nbl
         RBASE=runs/g16_base/base.in ;;
    cbl) SRC=$(newest runs/g16_cbl_shallow/output); GRID=data/grid16_raised
         RBASE=runs/g16_base/base_cbl_shallow.in ;;
  esac
  [ -f "$GRID/topo.bin" ] || die "$GRID not built"
  BASE=$RBASE ADJ_S=1200 SPS=0.0155 ZTARGET=8.5 EXACT_AGL=1 KEEP_FIELDS=1 \
    bin/run_directions.sh g16r_$REG "$SRC" "$GRID" $DT_WIN $WIN $TB \
    || die "production directions $REG"
  mark dirs6_$REG
done

# ============================ 5. what the retired closure was worth, on production fields
# The SAME LES fields, the same releases, the same seed -- only the closure differs. This
# is the only way to attribute an array-share difference to the closure rather than to a
# different turbulence realisation, and it is why the window fields are kept above.
if ! have legacy_prod; then
  say "5. the retired closure on the production fields"
  for C in g16r_cbl_wN g16r_cbl_wS g16r_cbl_wE g16r_cbl_wW g16r_nbl_wN; do
    D=runs/$C; G=data/grid16_raised
    case $C in g16r_nbl_*) G=data/grid16r_nbl ;; esac
    [ -d "$D/window" ] && [ "$(ls -1 $D/window | wc -l)" -gt 10 ] || \
      { echo "  (no retained fields for $C -- skipped)"; continue; }
    ./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt $DT_WIN --tback $TB \
        --z-target 8.5 --exact-agl --sgs-most --sgs-most-legacy --receptor-from "$G" \
        --cover-dir "$G" --fp16-cache --cover-groups 10 --tag ${C}_legacy 2>&1 \
        | grep -vE 'batch [0-9]+/' > $R/${C}_legacy.txt
    python3 bin/floor_bias.py new=$C legacy=${C}_legacy 2>&1 | tee $R/${C}_floorbias.txt
  done
  mark legacy_prod
fi

# =========================================================================== 6. clean up
if ! have cleanup6; then
  say "6. releasing the retained window fields"
  for D in runs/g16r_*/window runs/g16_flatcbl/window; do rm -f $D/* 2>/dev/null; done
  df -h /home/atyagi | tail -1
  mark cleanup6
fi

say "PASS 6 COMPLETE"
echo "COMPLETE $(date '+%F %H:%M:%S')" >> $R/pass6_status.txt
