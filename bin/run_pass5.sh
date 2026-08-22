#!/usr/bin/env bash
# Fifth pass, unattended: neutral stationarity gates -> terrain dt -> the flat/neutral
# control window. Each step is a chain of sub-1-hour segments and each gate stops the
# chain rather than letting a later stage inherit a bad state.
#
# It exists because the pass is ~30 GPU-h of wall clock and none of the waiting needs a
# human. Every step writes its verdict to results/ so the chain can be picked up cold.
set -uo pipefail
cd /home/atyagi/Flux
L=/tmp/flux-logs
DT_FLAT="${DT_FLAT:-0.0146417}"      # CFL_3d 1.35, measured
SPIN=runs/g16_spin
GRID="${GRID:-data/grid16}"
BASE=runs/g16_base/base.in
R=results
say(){ echo; echo "########## $* ##########"; date '+%F %H:%M:%S'; }
die(){ echo "GATE FAILED: $*" >&2; exit 1; }
newest(){ ls -1 "$1"/*.[0-9]* 2>/dev/null | sort -t. -k2 -n | tail -1; }

# ---------------------------------------------------------------- Gate C1: stationarity
say "Gate C1: neutral stationarity"
SERIES=$(ls -1 $SPIN/output/FE_G16.* | sort -t. -k2 -n)
FE_DT=$DT_FLAT ./docker/pyrun.sh docker/stage2_gate.py $SERIES 2>&1 | tee $R/g16_stationarity.txt
./docker/pyrun.sh - <<PY | tee $R/g16_c1.txt
import glob, re, numpy as np
from netCDF4 import Dataset
# u* and domain TKE against simulated time. Stationarity is a TREND test, not a snapshot:
# the second pass failed this at -8.4 sigma and the third passed it, so it is scored as a
# slope against its own scatter rather than by eye.
ps = sorted(glob.glob('$SPIN/output/FE_G16.*'), key=lambda p:int(p.rsplit('.',1)[1]))
t,us,tke = [],[],[]
for p in ps:
    with Dataset(p) as ds:
        g=lambda v: np.squeeze(np.asarray(ds[v][:],dtype=np.float64))
        u,v,w = g('u'),g('v'),g('w')
        us.append(float(g('fricVel').mean()))
    pr=lambda a:a-a.mean(axis=(-2,-1),keepdims=True)
    tke.append(float(0.5*((pr(u)**2+pr(v)**2+pr(w)**2).mean())))
    t.append(int(p.rsplit('.',1)[1])*$DT_FLAT/3600.0)
t=np.array(t); us=np.array(us); tke=np.array(tke)
print(f"  {'t (h)':>7}{'u* (m/s)':>11}{'domain TKE':>13}")
for a,b,c in zip(t,us,tke): print(f"  {a:7.3f}{b:11.4f}{c:13.6f}")
# score the LAST HALF: the early part is the cold-start transient by construction
h = len(t)//2
ok=True
for nm,y in (('u*',us),('TKE',tke)):
    x=t[h:]; yy=y[h:]
    if len(x)<3: print(f"  {nm}: too few points to trend"); ok=False; continue
    A=np.vstack([x,np.ones_like(x)]).T
    m,c0=np.linalg.lstsq(A,yy,rcond=None)[0]
    resid=yy-(m*x+c0); se=np.sqrt((resid**2).sum()/max(len(x)-2,1))
    sem=se/max(np.sqrt(((x-x.mean())**2).sum()),1e-12)
    sig=abs(m)/max(sem,1e-30)
    rel=m/max(abs(yy.mean()),1e-30)
    print(f"  {nm:4s} trend over the last half: {m:+.5g} per hour "
          f"({100*rel:+.2f}%/h), {sig:.2f} sigma")
    ok &= sig < 2.0
print(f"\n  GATE C1: {'PASS -- trend within 2 sigma' if ok else 'FAIL -- still drifting'}")
raise SystemExit(0 if ok else 1)
PY
[ "${PIPESTATUS[0]}" = "0" ] || die "C1 stationarity"

# ---------------------------------------------------------------- Gate C2: bit-for-bit
say "Gate C2: the saved restart restarts bit-for-bit"
LAST=$(newest $SPIN/output)
STEP=${LAST##*.}
mkdir -p runs/g16_c2/output; rm -f runs/g16_c2/output/* runs/g16_c2/FE_RST.*
cp -f "$LAST" runs/g16_c2/FE_RST.$STEP
sed -e "s|^dt = .*|dt = $DT_FLAT|" -e "s|^Nt = .*|Nt = $((STEP+20000))|" \
    -e 's|^NtBatch = .*|NtBatch = 20000|' -e 's|^frqOutput = .*|frqOutput = 20000|' \
    -e 's|^inPath = .*|inPath = ./|' -e "s|^inFile = .*|inFile = FE_RST.$STEP|" \
    -e 's|^outFileBase = .*|outFileBase = FE_C2|' "$BASE" > runs/g16_c2/c2.in
./docker/run_case.sh runs/g16_c2 c2.in $L/g16_c2.log || die "C2 run"
./docker/pyrun.sh - "$LAST" "$(newest runs/g16_c2/output)" <<'PY' | tee $R/g16_c2.txt
import sys, numpy as np
from netCDF4 import Dataset
# The restart is a state RESUME, so re-running the same 20000 steps from the same file
# must land on the same state the chain already produced -- and it must be bit-for-bit,
# because anything else means the restart is lossy and every chained segment is a
# different trajectory from the one it claims to continue.
a,b = sys.argv[1], sys.argv[2]
print(f"  chain dump   {a}\n  re-run dump  {b}")
bad=0
with Dataset(a) as A, Dataset(b) as B:
    for v in ('u','v','w','theta','rho','TKE_0'):
        if v not in A.variables or v not in B.variables: continue
        x=np.asarray(A[v][:],dtype=np.float64); y=np.asarray(B[v][:],dtype=np.float64)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            print(f"  {v:7s} NON-FINITE"); bad+=1; continue
        d=np.abs(x-y).max()
        print(f"  {v:7s} max|diff| = {d:.3e}   {'bit-for-bit' if d==0 else 'DIFFERS'}")
        bad += (d != 0)
print(f"\n  GATE C2: {'PASS' if not bad else 'FAIL'}")
raise SystemExit(1 if bad else 0)
PY
[ "${PIPESTATUS[0]}" = "0" ] || die "C2 bit-for-bit"

# ---------------------------------------------------------------- B4: terrain dt
# The flat boundary was measured at CFL_3d ~ 1.51 and the terrain amplification at the
# steepest cell is 1.252, so the terrain boundary should sit near 1.51/1.252 = 1.21. That
# BRACKETS the search; it does not answer it. Branch the ladder off a state that has both
# developed turbulence AND has already adjusted to the terrain, or the k0/k1 signal is the
# terrain adjustment transient rather than acoustic noise.
say "B4: terrain dt bisection"
LAST=$(newest $SPIN/output)
mkdir -p runs/g16_terr/output; rm -f runs/g16_terr/output/* runs/g16_terr/FE_RST.*
python3 bin/prep_restart.py "$LAST" runs/g16_terr/FE_RST.0 --rot 0 --grid "$GRID"   || die "B4 prep_restart"
cp -f "$GRID/topo.bin" runs/g16_terr/topo.bin
DT_ADJ=$(python3 -c "print(f'{1.05/92.20239148570417:.7f}')")
sed -e "s|^dt = .*|dt = $DT_ADJ|" -e 's|^Nt = .*|Nt = 40000|' -e 's|^NtBatch = .*|NtBatch = 20000|'     -e 's|^frqOutput = .*|frqOutput = 20000|' -e 's|^inPath = .*|inPath = ./|'     -e 's|^inFile = .*|inFile = FE_RST.0|' -e 's|^topoFile = .*|topoFile = ./topo.bin|'     -e 's|^outFileBase = .*|outFileBase = FE_TADJ|' "$BASE" > runs/g16_terr/adj.in
echo "--- terrain adjustment at CFL_3d 1.05, 40000 steps"
./docker/run_case.sh runs/g16_terr adj.in $L/g16_terr_adj.log || die "B4 terrain adjustment"
TADJ=$(newest runs/g16_terr/output)
{
echo "terrain dt ladder, branched from $TADJ (adjusted at CFL_3d 1.05)"
printf "%-9s %-12s %-9s %s\n" "CFL_3d" "dt (s)" "k0/k1" "verdict"
for cfl in 1.00 1.10 1.15 1.20 1.25 1.30 1.35 1.40; do
  DTT=$(python3 -c "print(f'{$cfl/92.20239148570417:.7f}')")
  sed -e "s|^dt = .*|dt = $DTT|" -e 's|^Nt = .*|Nt = 4000|' -e 's|^NtBatch = .*|NtBatch = 4000|' \
      -e 's|^frqOutput = .*|frqOutput = 4000|' -e 's|^inPath = .*|inPath = ./output/|' \
      -e "s|^inFile = .*|inFile = $(basename $TADJ)|" \
      -e 's|^topoFile = .*|topoFile = ./topo.bin|' \
      -e 's|^outFileBase = .*|outFileBase = FE_TDT|' "$BASE" > runs/g16_terr/tdt.in
  res=$(./docker/run_case.sh runs/g16_terr tdt.in $L/g16_tdt.log 2>&1)
  k=$(echo "$res" | grep -oE 'k0/k1= *[0-9.]+|k0/k1= *(nan|inf)' | head -1 | sed 's/.*= *//')
  v="OK"; echo "$res" | grep -q "RUN REJECTED" && v="REJECT"
  echo "$res" | grep -q "k0/k1 SKIP" && v="SKIP (undeveloped)"
  printf "%-9s %-12s %-9s %s\n" "$cfl" "$DTT" "${k:-?}" "$v"
done
} 2>&1 | tee $R/g16_terrain_dt.txt

# ---------------------------------------------------------------- Phase D: the control
say "Phase D: the flat/neutral control window"
# The 5 s cadence must be an integer step count -- run_window.sh asserts it -- so the
# window dt is 5/342, CFL_3d 1.348, a hair below the 1.35 the spin-up used. Restart is a
# state resume, so changing dt between segments is free.
DT_WIN="${DT_WIN:-0.0146199}"
mkdir -p runs/g16_flat/output
cp -f "$LAST" runs/g16_flat/output/FE_ADJ.0
# t_back is GENEROUS on purpose the first time: the capture curve this produces is what
# SIZES t_back, and a window too short to contain the answer cannot report that it was
# too short.
TBACK=600 DT="$DT_WIN" SRC=runs/g16_flat/output/FE_ADJ.0 D=runs/g16_flat   GRID="$GRID" TAG=g16_flat BASE="$BASE" bin/regression_flat.sh --baseline   2>&1 | tee $R/g16_phaseD.txt

say "PASS5 CHAIN COMPLETE: C1, C2, B4 and Phase D"
