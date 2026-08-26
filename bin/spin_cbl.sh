#!/usr/bin/env bash
# RETIRED-PASS DRIVER. This still CHAINS segments, and that is deliberate: it belongs to
# the per-bin campaign design (bin/run_campaign.sh, bin/run_pass7.sh), not to the seed
# library or the corpus. PRODUCTION IS UNCHAINED -- a seed (jobs/run_seed.sh) and a target
# case (bin/run_corpus_case.sh -> bin/run_window.sh) are each ONE continuous FastEddy
# invocation, so that the only restart left in the project is seed -> target and
# FASTEDDY_TRAPS.md 17's failure mode is absent by construction rather than by assertion.
# Do not copy this file's chaining into anything new.
# Convective spin-up: flat, uniform, dry CBL, cold start, chained sub-45-minute segments.
#
#   usage: D=runs/g16_cbl_deep BASE=runs/g16_base/base_cbl_deep.in NSEG=8 bin/spin_cbl.sh
#
# The regime this project cares about most. CONUS404 at the tower says 27% of
# quality-controlled hours are VERY unstable (z/L < -0.5) and another 30% are unstable, so
# the convective boundary layer is the modal daytime state, not an edge case -- and the
# neutral corpus built so far does not contain it at all.
#
# Why the spin-up must be flat and uniform: a square, doubly periodic, flat, uniform state
# with dx = dy is exactly equivariant under 90-degree rotation, so ONE spin-up serves four
# wind directions. Terrain is introduced afterwards, per direction, with its own adjustment.
#
# Length. The CBL's eddy turnover time is T* = z_i/w* = 800/1.42 = 562 s, and a cold-start
# CBL is converged after roughly 8 T*. 5400 s is 9.6 T*. z_i then grows by entrainment at
# about 1.2 w'th'/(gamma z_i) = 148 m/h, which is real CBL behaviour and not drift -- a
# convective boundary layer has no stationary depth. The achieved z_i is measured and
# reported per window rather than assumed.
set -uo pipefail
cd "${FLUX_ROOT:-/home/atyagi/Flux}"
# Parameterised so the same driver serves the neutral spin-up and BOTH convective targets
# (z_i ~ 490 m at L = 4 z_i and z_i ~ 976 m at L = 2 z_i), which differ only in their
# capping inversion -- the surface heat flux is deliberately identical, so the
# domain-adequacy pair isolates the box rather than surface-layer physics.
D="${D:-runs/g16_cbl_shallow}"
BASE="${BASE:-runs/g16_base/base_cbl_shallow.in}"
DT="${DT:-0.0162686}"
SPS="${SPS:-0.0149}"          # measured s/step at 122^3, 1x2x64, spin-up IO cadence
WALLCAP="${WALLCAP:-3600}"
OUTBASE="${OUTBASE:-FE_CBL}"
FRQ="${FRQ:-18440}"           # a dump every ~300 s, for the stationarity series
# One segment is as long as the wall cap allows, rounded DOWN to a whole number of dumps.
SEG=$(python3 -c "
import math
n=int($WALLCAP*0.93/$SPS); print(max($FRQ, n//$FRQ*$FRQ))")
NSEG="${NSEG:-6}"
python3 -c "print(f'### segments: {$SEG} steps = {$SEG*$DT:.0f} s simulated, {$SEG*$SPS/60:.1f} min wall (cap {$WALLCAP/60:.0f} min)')"
[ "$(python3 -c "print(int($SEG*$SPS>$WALLCAP))")" = "1" ] && { echo "FATAL: segment over the wall cap"; exit 1; }
mkdir -p $D/output
IN=""; IPATH=""
LAST=$(ls -1 $D/output/$OUTBASE.* 2>/dev/null | sort -t. -k2 -n | tail -1)
S0=1
if [ -n "$LAST" ]; then                       # resume a partial chain
  STEP=${LAST##*.}; S0=$((STEP/SEG + 1)); IN=$(basename $LAST); IPATH="./output/"
  echo "### resuming from $LAST (segment $S0)"
fi
for s in $(seq $S0 $NSEG); do
  NT=$((s * SEG))
  sed -e "s|^dt = .*|dt = $DT|" -e "s|^Nt = .*|Nt = $NT|" \
      -e "s|^NtBatch = .*|NtBatch = $FRQ|" -e "s|^frqOutput = .*|frqOutput = $FRQ|" \
      -e "s|^inPath = .*|inPath = $IPATH|" -e "s|^inFile = .*|inFile = $IN|" \
      -e "s|^outFileBase = .*|outFileBase = $OUTBASE|" \
      "$BASE" > $D/seg$s.in
  echo "### CBL spin-up segment $s/$NSEG -> step $NT ($(python3 -c "print(f'{$NT*$DT/60:.0f}')") min simulated)"
  date '+%F %H:%M:%S'
  ./docker/run_case.sh $D seg$s.in /tmp/flux-logs/$(basename $D)_seg$s.log || { echo "FATAL: segment $s"; exit 1; }
  IN=$(basename $(ls -1 $D/output/$OUTBASE.* | sort -t. -k2 -n | tail -1)); IPATH="./output/"
done
echo "### CBL spin-up complete: $(ls $D/output | wc -l) dumps"
FE_DT=$DT ./docker/pyrun.sh bin/cbl_check.py $(ls -1 $D/output/$OUTBASE.* | sort -t. -k2 -n) 2>&1 \
  | tee "results/$(basename $D)_spinup.txt"
