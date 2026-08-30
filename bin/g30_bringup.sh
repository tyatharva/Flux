#!/usr/bin/env bash
# 30 m BRING-UP: measured cost, and the FLAT dt accuracy boundary re-measured.
#
# WHY THE BOUNDARY IS RE-MEASURED AND NOT CARRIED. PROJECT_BRIEF.md's standing rule, and it has
# been right twice: the accuracy boundary is NOT a property of CFL_3d alone. It was ~1.64
# at dx/dz_sfc = 2.80 on the retired 24 m grid, ~1.51 at 4.007 on the 16 m one, and
# 1.55-1.60 at 2.804 on the current 24 m one -- the SAME anisotropy as the retired grid and
# a different answer. This grid is dx/dz_sfc = 3.505, between the two measured extremes,
# and the honest expectation is therefore ~1.53 with no confidence at all in that guess.
# The transition is sharp (k0/k1 0.132 -> 8.511 across 0.05 of CFL at 24 m), so a ladder
# resolves it cheaply and an assumption does not.
#
# Between the accuracy boundary and the stability boundary FastEddy runs to completion,
# exits 0, prints nothing, and produces near-surface w that is grid-scale acoustic noise
# rather than turbulence. That is the failure this exists to find.
#
# THE LADDER BRANCHES OFF A DEVELOPED STATE. A cold start at a few thousand steps leaves
# ww[1] below k0k1_check.py's floor, the check SKIPs, and the ladder then detects only the
# gross acoustic failure -- which is the one case that would have been obvious anyway.
# turb_alive runs at every rung beside k0/k1, because k0/k1 is a ratio of two quantities
# that die together and reads a clean 0.442 through a boundary layer that has collapsed.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
L="${LOGDIR:-/tmp/flux-logs}"; mkdir -p "$L" results
BU="runs/g30_bringup"; BASE="runs/g30_base/base.in"
# dt = CFL_3d / KFAC, KFAC = c sqrt(2/dx^2 + 1/dz_sfc^2), c = 347.2, dx = 30, dz_sfc = 8.5583
KFAC=43.74600
SPIN_S=1800
die(){ echo "FATAL: $*" >&2; exit 1; }

mkdir -p "$BU/output" || die "cannot make $BU"
[ -f "$BASE" ] || die "no template at $BASE"
# ONE RUN PER DIRECTORY (TRAPS 18c): a directory that has held two runs holds two families
# with overlapping step numbers, and sorting their union on the step interleaves them.
rm -f "$BU"/output/*

DT0=$(python3 -c "print(f'{1.30/$KFAC:.7f}')")
NT0=$(python3 -c "print(int(round($SPIN_S/$DT0)))")

{
echo "=== 30 m BRING-UP: 122^3 @ 30 m = 3660 m, receptor k=3 at 30.000000 m ==="
echo "  KFAC $KFAC, so dt = CFL_3d/$KFAC.  dx/dz_sfc = 3.505."
echo
echo "=== step 1: ${SPIN_S} s cold start at CFL_3d 1.30 (dt $DT0, $NT0 steps) ==="
sed -e "s|^dt = .*|dt = $DT0|" -e "s|^Nt = .*|Nt = $NT0|" \
    -e 's|^NtBatch = .*|NtBatch = 2000|' -e 's|^frqOutput = .*|frqOutput = 20000|' \
    -e 's|^outFileBase = .*|outFileBase = FE_BU|' \
    -e 's|^surflayer_wth = .*|surflayer_wth = 0.06|' \
    -e 's|^zStableBottom = .*|zStableBottom = 450.0|' \
    -e 's|^U_g = .*|U_g = 7.0|' -e 's|^V_g = .*|V_g = 0.0|' \
    -e 's|^lsf_w_lev1 = .*|lsf_w_lev1 = -25.0|' -e 's|^lsf_w_zlev1 = .*|lsf_w_zlev1 = 450.0|' \
    "$BASE" > "$BU/bu.in"
# ASSERT THE SEDS LANDED. `sed s|^key = .*|...|` on a template with no such key is a silent
# no-op: the run proceeds as a DIFFERENT run than the one asked for.
for kv in "dt|$DT0" "Nt|$NT0" "outFileBase|FE_BU" "surflayer_wth|0.06" "U_g|7.0"; do
  k="${kv%%|*}"; v="${kv#*|}"
  n=$(grep -c "^$k = " "$BU/bu.in")
  [ "$n" -eq 1 ] || die "bu.in carries $n '$k' lines, wanted 1"
  got=$(grep -m1 "^$k = " "$BU/bu.in" | sed "s|^$k = ||" | sed 's| *#.*||')
  [ "$got" = "$v" ] || die "bu.in has $k = '$got', asked for '$v'"
done
./docker/run_case.sh "$BU" bu.in "$L/g30_bu.log" 2>&1 | tail -12
SEED_DUMP="$BU/output/FE_BU.$NT0"
[ -f "$SEED_DUMP" ] || die "no developed state at $SEED_DUMP"

echo
echo "=== measured cost ==="
python3 - <<PY
import os,glob,statistics
fs=sorted(glob.glob("$BU/output/FE_BU.*"), key=lambda p:int(p.rsplit('.',1)[1]))
t=[os.path.getmtime(p) for p in fs]; s=[int(p.rsplit('.',1)[1]) for p in fs]
d=[(t[i+1]-t[i])/(s[i+1]-s[i]) for i in range(len(t)-1)]
sps=statistics.median(d[1:]) if len(d)>2 else (statistics.median(d) if d else float('nan'))
print(f"  {len(fs)} dumps, {s[-1]} steps")
print(f"  {sps*1e3:.3f} ms/step measured")
gph=sps/$DT0
print(f"  -> {gph:.3f} GPU-h per simulated hour")
print(f"  -> a 2.0 sim-h two-window case is {2.0*gph:.2f} GPU-h ({2.0*gph*60:.0f} min wall)")
print(f"  -> a 3.0 sim-h seed is {3.0*gph:.2f} GPU-h ({3.0*gph*60:.0f} min wall)")
PY

echo
echo "=== the FLAT dt accuracy boundary, branched from step $NT0 ==="
echo "  NOT assumed. dx/dz_sfc 3.505 sits between 16 m's 4.007 (boundary ~1.51) and"
echo "  24 m's 2.804 (1.55-1.60), so ~1.53 is a guess and the ladder is the answer."
printf "  %-9s %-12s %-9s %-9s %s\n" "CFL_3d" "dt (s)" "k0/k1" "turb" "verdict"
NT=$((NT0 + 4000))
for cfl in 1.30 1.40 1.45 1.50 1.55 1.60 1.65 1.70; do
  DTT=$(python3 -c "print(f'{$cfl/$KFAC:.7f}')")
  sed -e "s|^dt = .*|dt = $DTT|" -e "s|^Nt = .*|Nt = $NT|" -e 's|^NtBatch = .*|NtBatch = 4000|' \
      -e 's|^frqOutput = .*|frqOutput = 4000|' -e 's|^inPath = .*|inPath = ./output/|' \
      -e "s|^inFile = .*|inFile = FE_BU.$NT0|" \
      -e 's|^outFileBase = .*|outFileBase = FE_DT|' \
      -e 's|^surflayer_wth = .*|surflayer_wth = 0.06|' \
      -e 's|^zStableBottom = .*|zStableBottom = 450.0|' \
      -e 's|^U_g = .*|U_g = 7.0|' -e 's|^V_g = .*|V_g = 0.0|' \
      -e 's|^lsf_w_lev1 = .*|lsf_w_lev1 = -25.0|' -e 's|^lsf_w_zlev1 = .*|lsf_w_zlev1 = 450.0|' \
      "$BASE" > "$BU/dt.in"
  rm -f "$BU"/output/FE_DT.*
  res=$(./docker/run_case.sh "$BU" dt.in "$L/g30_dt.log" 2>&1)
  k=$(echo "$res" | grep -oE 'k0/k1= *[0-9.]+|k0/k1= *(nan|inf)' | head -1 | sed 's/.*= *//')
  v="OK"; echo "$res" | grep -q "RUN REJECTED" && v="REJECT"
  echo "$res" | grep -q "k0/k1 SKIP" && v="SKIP (undeveloped)"
  ta=$(./docker/pyrun.sh docker/turb_alive.py "$BU/output/FE_DT.$NT" 2>&1 | grep -oE '\b(OK|DEAD|SKIP|MARGINAL)\b' | head -1)
  printf "  %-9s %-12s %-9s %-9s %s\n" "$cfl" "$DTT" "${k:-?}" "${ta:-?}" "$v"
done
rm -f "$BU"/output/FE_DT.*
echo
echo "  k0/k1 < 1 is the accuracy criterion (~0.27 when correct; ~9 means dt is too large)."
echo "  Production takes the boundary with ~10% margin and lands the 5 s cadence on an"
echo "  INTEGER step count, which is what run_window.sh asserts."
} 2>&1 | tee results/g30_bringup.txt
