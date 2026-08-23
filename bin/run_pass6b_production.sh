#!/usr/bin/env bash
# The production LES chain, split out so it can run on the GPU while the control batteries
# run on the CPU. Called by bin/run_pass6b.sh; safe to run alone (same sentinels).
set -uo pipefail
cd /home/atyagi/Flux
L=/tmp/flux-logs; R=results; DONE=$R/.done6b
mkdir -p $DONE
DT_WIN=0.0146199; TB=600; WIN=2400
say(){ echo; echo "########## $* ##########"; date '+%F %H:%M:%S'; }
die(){ echo "PRODUCTION STOPPED: $*" >&2; exit 1; }
newest(){ ls -1 "$1"/*.[0-9]* 2>/dev/null | sort -t. -k2 -n | tail -1; }
have(){ [ "${FORCE:-0}" = "1" ] && return 1; [ -f "$DONE/$1" ]; }
mark(){ date '+%F %H:%M:%S' > "$DONE/$1"; echo "  [stage $1 recorded done]"; }

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
  # The full corpus: four directions per regime, ~8.5 GPU-h, which is now the ONLY thing
  # setting the wall clock. The CPU analysis hides inside it -- see the concurrency note
  # at the batteries.
  BASE=$RBASE ADJ_S=1200 SPS=0.0155 ZTARGET=8.5 EXACT_AGL=1 KEEP_FIELDS=1 \
    ONLY="${DIRS:-wW,wS,wE,wN}" \
    bin/run_directions.sh g16r_$REG "$SRC" "$GRID" $DT_WIN $WIN $TB || die "dirs $REG"
  mark dirs_$REG
done

