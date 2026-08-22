#!/usr/bin/env bash
# Run ONE sampling window as an automatically chained series of sub-45-minute segments.
#
# A production window is (averaging period + t_back) long, because a backward trajectory
# needs t_back seconds of history behind it before it can be released -- the first t_back
# of any window yields no releases at all. At this grid that is 45 minutes of simulated
# time, which is about 60 minutes of wall clock: past the 45-minute ceiling this project
# runs under, and there was no way to split it, because lean ioLPDMmode output is not
# restartable (rho and pressure are absent by construction).
#
# At 122^3 @ 16 m a whole (30 min + t_back) window is ~35 min wall and fits in ONE segment,
# so the chain below usually has length 1. It is kept because it costs nothing when NSEG=1
# and it is the only thing that makes a longer window or a slower grid possible at all.
#
# ioLPDMfullFrq on the kegonsa fork fixes exactly that: it writes one FULL upstream dump at
# each segment boundary while every other dump stays lean and 16-bit packed. The chain
# below is therefore seamless -- FastEddy restart is bit-for-bit, so the joined window is
# the same trajectory a single long run would have produced.
#
# usage: run_window.sh <dir> <restart> <dt> <window_s> <topofile|-> <Ug> <Vg> [extra.in]
set -uo pipefail
cd /home/atyagi/Flux
D="$1"; RST="$2"; DT="$3"; WIN="$4"; TOPO="$5"; UG="$6"; VG="$7"; EXTRA="${8:-}"
BASE="${BASE:-runs/g16_base/base.in}"
L=/tmp/flux-logs
# ONE wall cap, and the planner and the refusal both derive from it. They used to be two
# independent constants (2350 and 2700) that happened to agree; a cap raised in one place
# and not the other is a silent way to blow the limit this project runs under.
WALLCAP="${WALLCAP:-3600}"      # s of wall clock per segment -- the hard rule
MARGIN="${MARGIN:-0.93}"        # plan to this fraction of it, so a slow segment still fits
MAXWALL=$(python3 -c "print(int($WALLCAP*$MARGIN))")
SPS="${SPS:-0.0155}"            # measured s/step at 122x122x122, block 1x2x64, with IO
CAD="${CAD:-5.0}"               # output cadence, s

die(){ echo "FATAL: $*" >&2; exit 1; }
[ -f "$RST" ] || die "restart $RST not found"
mkdir -p "$D/window" || die "cannot make $D/window"

read -r FRQ NSEG SEGS TOT < <(python3 -c "
import math
dt=$DT; win=$WIN; cad=$CAD
frq=int(round(cad/dt));            assert abs(frq*dt-cad)<2e-4, 'cadence not an integer step count'
tot=int(round(win/dt/frq))*frq     # window rounded to a whole number of dumps
maxsteps=int($MAXWALL/$SPS)
nseg=max(1, math.ceil(tot/ (maxsteps//frq*frq) ))
segs=math.ceil(tot/nseg/frq)*frq   # segment length, a whole number of dumps
nseg=math.ceil(tot/segs)
print(frq, nseg, segs, nseg*segs)")
echo "### window $D: ${TOT} steps = $(python3 -c "print(f'{$TOT*$DT:.0f}')") s"
echo "###   $NSEG segment(s) of $SEGS steps = $(python3 -c "print(f'{$SEGS*$SPS/60:.1f}')") min wall each (cap $(python3 -c "print(f'{$WALLCAP/60:.0f}')") min)"
echo "###   frqOutput = $FRQ ($CAD s), ioLPDMfullFrq = $SEGS"
[ "$(python3 -c "print(int($SEGS*$SPS>$WALLCAP))")" = "1" ] \
  && die "segment projects $(python3 -c "print(f'{$SEGS*$SPS/60:.1f}')") min, over the ${WALLCAP}s cap"

rm -f "$D"/window/* "$D"/FE_RST.*
cp -f "$RST" "$D/FE_RST.0" || die "copy restart"
IN="FE_RST.0"; IPATH="./"; PREV=0
for s in $(seq 1 "$NSEG"); do
  NT=$((s * SEGS))
  sed -e "s|^dt = .*|dt = $DT|" -e "s|^Nt = .*|Nt = $NT|" \
      -e "s|^NtBatch = .*|NtBatch = $FRQ|" -e "s|^frqOutput = .*|frqOutput = $FRQ|" \
      -e "s|^inPath = .*|inPath = $IPATH|" -e "s|^inFile = .*|inFile = $IN|" \
      -e "s|^topoFile = .*|topoFile = $([ "$TOPO" = "-" ] && echo "" || echo "$TOPO")|" \
      -e "s|^U_g = .*|U_g = $UG|" -e "s|^V_g = .*|V_g = $VG|" \
      -e "s|^outPath = .*|outPath = ./window/|" \
      -e "s|^outFileBase = .*|outFileBase = FE_WIN|" \
      "$BASE" > "$D/win$s.in"
  printf 'ioLPDMmode = 1\nioLPDMfullFrq = %d\n' "$SEGS" >> "$D/win$s.in"
  [ -n "$EXTRA" ] && cat "$EXTRA" >> "$D/win$s.in"
  echo "--- segment $s/$NSEG: $PREV -> $NT"
  ./docker/run_case.sh "$D" "win$s.in" "$L/$(basename $D)_win$s.log" \
      || die "segment $s failed (see $L/$(basename $D)_win$s.log)"
  # Preserve the boundary dump OUTSIDE window/ before the next segment overwrites it with
  # its own lean copy. Without this a failed segment s+1 would take the chain point with it.
  if [ "$s" -lt "$NSEG" ]; then
    cp -f "$D/window/FE_WIN.$NT" "$D/FE_RST.$NT" || die "checkpoint copy"
    IN="FE_RST.$NT"; IPATH="./"
  fi
  PREV=$NT
done
rm -f "$D"/FE_RST.*
echo "--- window complete: $(ls $D/window | wc -l) dumps, $(du -sh $D/window | cut -f1)"
