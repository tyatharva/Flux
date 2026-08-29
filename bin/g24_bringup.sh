#!/usr/bin/env bash
# Seventh-pass GPU bring-up on 122^3 @ 24 m, receptor 30 m.
#
# WHAT IT ESTABLISHES, and why each item cannot be inherited from the 16 m grid:
#
#  B1  the grid launches at all, and the measured s/step (the cost model is a projection)
#  B3  THE FLAT dt ACCURACY BOUNDARY, re-measured. PROJECT_BRIEF.md's standing rule: the boundary
#      is NOT a property of CFL_3d alone -- it was ~1.64 at dx/dz = 2.80 (the retired 24 m
#      grid) and ~1.51 at dx/dz = 4.007 (the 16 m grid). This grid is back at 2.80, so the
#      old number is the EXPECTATION, not the answer. Re-measure or do not claim it.
#  turb_alive everywhere k0/k1 runs. k0/k1 is a ratio of two things that die together and
#      read a clean 0.442 through a boundary layer that had collapsed.
#  a short LPDM window, which is the field set the GPU-LPDM acceptance suite replays.
#
# The ladder branches off a DEVELOPED state. A cold start at 500 steps leaves ww[1] below
# k0k1_check.py's floor, the check SKIPs, and the ladder then only detects the gross
# acoustic failure (learned in Phase B3 of the fifth pass).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
L="${LOGDIR:-/tmp/flux-logs}"; mkdir -p "$L" results
BU=runs/g24_bringup
BASE=runs/g24_base/base.in
# dt = CFL_3d / KFAC, KFAC = c sqrt(2/dx^2 + 1/dz_sfc^2), c = 347.2 m/s, dz_sfc = 8.5583 m.
# Same form that reproduced the 24 m grid's stated 1.4946 to four digits (PROJECT_BRIEF.md).
KFAC=45.43513
die(){ echo "FATAL: $*" >&2; exit 1; }
newest(){ ls -1 "$1"/"${2:-FE_}"*.[0-9]* 2>/dev/null | sort -t. -k2 -n | tail -1; }

STEP0=60840
SEED_DUMP="$BU/output/FE_BU.$STEP0"
[ -f "$SEED_DUMP" ] || die "no developed state at $SEED_DUMP -- run the 1800 s bring-up first"

{
echo "=== B1: the grid ran. measured cost ==="
python3 - <<PY
import os,glob,time
fs=sorted(glob.glob("$BU/output/FE_BU.*"), key=lambda p:int(p.rsplit('.',1)[1]))
t=[os.path.getmtime(p) for p in fs]; s=[int(p.rsplit('.',1)[1]) for p in fs]
# first interval includes start-up; use the median of the rest
d=[(t[i+1]-t[i])/(s[i+1]-s[i]) for i in range(len(t)-1)]
import statistics
sps=statistics.median(d[1:]) if len(d)>2 else statistics.median(d)
print(f"  {len(fs)} dumps, {s[-1]} steps")
print(f"  {sps*1e3:.3f} ms/step measured (spin-up IO cadence)")
print(f"  -> {sps*3600/0.0295858/3600:.3f} GPU-h per simulated hour")
print(f"  -> a 3.0 sim-h seed is {3.0*sps*3600/0.0295858/3600:.2f} GPU-h, "
      f"{3.0*sps*3600/0.0295858/60:.0f} min wall")
PY

echo
echo "=== B3: the FLAT dt accuracy boundary, branched from step $STEP0 ==="
echo "  expectation ~1.64 (the retired 24 m grid, same dx/dz = 2.80). NOT assumed."
printf "  %-9s %-12s %-9s %-9s %s\n" "CFL_3d" "dt (s)" "k0/k1" "turb" "verdict"
NT=$((STEP0 + 4000))
for cfl in 1.30 1.45 1.55 1.60 1.65 1.70 1.75; do
  DTT=$(python3 -c "print(f'{$cfl/$KFAC:.7f}')")
  sed -e "s|^dt = .*|dt = $DTT|" -e "s|^Nt = .*|Nt = $NT|" -e 's|^NtBatch = .*|NtBatch = 4000|' \
      -e 's|^frqOutput = .*|frqOutput = 4000|' -e 's|^inPath = .*|inPath = ./output/|' \
      -e "s|^inFile = .*|inFile = FE_BU.$STEP0|" \
      -e 's|^outFileBase = .*|outFileBase = FE_DT|' \
      -e 's|^surflayer_wth = .*|surflayer_wth = 0.06|' \
      -e 's|^surflayer_z0 = .*|surflayer_z0 = 0.083229|' \
      -e 's|^zStableBottom = .*|zStableBottom = 450.0|' \
      -e 's|^U_g = .*|U_g = 7.0|' -e 's|^V_g = .*|V_g = 0.0|' \
      -e 's|^lsf_w_lev1 = .*|lsf_w_lev1 = -25.0|' -e 's|^lsf_w_zlev1 = .*|lsf_w_zlev1 = 450.0|' \
      "$BASE" > "$BU/dt.in"
  rm -f "$BU"/output/FE_DT.*
  res=$(./docker/run_case.sh "$BU" dt.in "$L/g24_dt.log" 2>&1)
  k=$(echo "$res" | grep -oE 'k0/k1= *[0-9.]+|k0/k1= *(nan|inf)' | head -1 | sed 's/.*= *//')
  v="OK"; echo "$res" | grep -q "RUN REJECTED" && v="REJECT"
  echo "$res" | grep -q "k0/k1 SKIP" && v="SKIP (undeveloped)"
  ta=$(./docker/pyrun.sh docker/turb_alive.py "$BU/output/FE_DT.$NT" 2>&1 | grep -oE '\b(OK|DEAD|SKIP|MARGINAL)\b' | head -1)
  printf "  %-9s %-12s %-9s %-9s %s\n" "$cfl" "$DTT" "${k:-?}" "${ta:-?}" "$v"
done
rm -f "$BU"/output/FE_DT.*
echo
echo "  k0/k1 < 1 is the accuracy criterion (~0.27 when correct; ~9 means dt is too large)."
echo "  Production takes the boundary with ~10% margin, as PROJECT_BRIEF.md requires."
} 2>&1 | tee results/g24_bringup.txt
