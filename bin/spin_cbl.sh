#!/usr/bin/env bash
# Convective spin-up: flat, uniform, dry CBL, cold start, chained sub-45-minute segments.
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
cd /home/atyagi/Flux
D=runs/cbl_spin
BASE=runs/g24_base/base_cbl.in
DT=0.0328947
SEG=54720        # 1800 s per segment, 33.2 min wall at 0.0364 s/step
NSEG="${NSEG:-3}"
FRQ=9120         # a dump every 300 s, for the stationarity series
mkdir -p $D/output
IN=""; IPATH=""
LAST=$(ls -1 $D/output/FE_CBL.* 2>/dev/null | sort -t. -k2 -n | tail -1)
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
      "$BASE" > $D/seg$s.in
  echo "### CBL spin-up segment $s/$NSEG -> step $NT ($(python3 -c "print(f'{$NT*$DT/60:.0f}')") min simulated)"
  date '+%F %H:%M:%S'
  ./docker/run_case.sh $D seg$s.in /tmp/claude-1000/cbl_seg$s.log || { echo "FATAL: segment $s"; exit 1; }
  IN=$(basename $(ls -1 $D/output/FE_CBL.* | sort -t. -k2 -n | tail -1)); IPATH="./output/"
done
echo "### CBL spin-up complete: $(ls $D/output | wc -l) dumps"
./docker/pyrun.sh bin/cbl_check.py $(ls -1 $D/output/FE_CBL.* | sort -t. -k2 -n) 2>&1 | tee results/cbl_spinup.txt
