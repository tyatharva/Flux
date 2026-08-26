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
import glob, numpy as np, math, sys
sys.path.insert(0, "/work")
from netCDF4 import Dataset
from lpdm import kljun
# ============================================================================
# STATIONARITY, GATED ON THE QUANTITIES THE FOOTPRINT ACTUALLY DEPENDS ON.
#
# This gate was wrong twice before it was right, and both errors are worth keeping.
#
# (1) It scored "the last half" of the series -- from a cold start, so it penalised the
#     run for the transient it was always going to have to spend.
# (2) It gated on u* IN ISOLATION, at a fixed %/h. Measured here: u* rises at +6.3 %/h at
#     6 simulated hours, having FALLEN for the first four. That sign flip is not drift and
#     it is not a defect -- it is the INERTIAL OSCILLATION. f = 9.94e-5 gives a period of
#     17.6 h and the u* minimum landed at ~4.4 h, one quarter period. Damping it would
#     take several periods, i.e. 35-50 simulated hours for ONE base state, and a real
#     boundary layer does not do that either: the tower measures during the oscillation.
#
# WHAT ACTUALLY MATTERS. Kljun's Pi_4 -- the only channel through which the wind enters
# the streamwise footprint shape -- is u(z_m)/u*, a RATIO. Both terms ride the inertial
# oscillation together, so the ratio does not move: measured +0.03 %/h while its numerator
# and denominator each move at +6.3 %/h. The derived x_peak spans 32.2-32.5 m over 1.5 h
# against a 16 m raster cell.
#
# So the gate is: the footprint's own controlling parameters are stationary, and the
# turbulence is in equilibrium with the instantaneous shear. The mean-flow drift is
# REPORTED, named as the inertial oscillation, and carried into the corpus as a label --
# every case is tagged with its ACHIEVED u*, U, direction and L by window_stats, so a
# slowly turning mean is a different point in the input space, not an error.
#
# These thresholds are far TIGHTER in footprint terms than the u* test they replace.
# ============================================================================
SCORE_H = float("${SCORE_H:-1.5}")
# The seven limits live in bin/seed_stationarity.py, which is the portable form of this
# gate and the one the seed jobs run. Imported, not restated: a gate carrying its own copy
# of a definition is exactly how stage4_wellmixed.py came to score a closure the
# footprints did not compute (PROJECT_BRIEF.md, Conventions).
sys.path.insert(0, "bin")
from seed_stationarity import LIMITS as LIM, zi_fixed
ps = sorted(glob.glob('$SPIN/output/FE_G16.*'), key=lambda p:int(p.rsplit('.',1)[1]))
t,us,tke,zi,sw,sv,Um,wd = [],[],[],[],[],[],[],[]
for p in ps:
    with Dataset(p) as ds:
        g=lambda v: np.squeeze(np.asarray(ds[v][:],dtype=np.float64))
        u,v,w = g('u'),g('v'),g('w'); z=g('zPos')[:,0,0]; e=np.maximum(g('TKE_0'),0.0)
        us.append(float(g('fricVel').mean()))
    pr=lambda a:a-a.mean(axis=(-2,-1),keepdims=True)
    tk=0.5*((pr(u)**2+pr(v)**2+pr(w)**2).mean(axis=(-2,-1)))
    tke.append(float(tk.mean()))
    # IMPORTED, NOT RESTATED -- and this line was the counter-example to its own comment
    # four lines up: it carried an inline 5%-of-peak copy while the gate it imports LIMITS
    # from moved to a fixed threshold. Same shape as stage4_wellmixed.py's private copy of
    # the sigma_w floor.
    zi.append(zi_fixed(tk, z))
    k=2
    sw.append(float(np.sqrt((pr(w)[k]**2).mean()+(2/3)*e[k].mean())))
    sv.append(float(np.sqrt(((pr(u)[k]**2+pr(v)[k]**2).mean())/2+(2/3)*e[k].mean())))
    Um.append(float(np.hypot(u[k].mean(),v[k].mean())))
    wd.append(float((270-np.degrees(np.arctan2(v[k].mean(),u[k].mean())))%360))
    t.append(int(p.rsplit('.',1)[1])*$DT_FLAT/3600.0)
t,us,tke,zi,sw,sv,Um,wd=[np.array(a) for a in (t,us,tke,zi,sw,sv,Um,wd)]
xp=np.array([kljun.peak_distance(10.0,zi[i],us[i],umean=Um[i],L=np.inf) for i in range(len(t))])
x90=[]
for i in range(len(t)):
    x=np.linspace(0.5,3000,4000)
    fy,_=kljun.crosswind_integrated(x,10.0,zi[i],us[i],umean=Um[i],L=np.inf)
    c=np.cumsum(fy); c/=c[-1]; x90.append(float(np.interp(0.90,c,x)))
x90=np.array(x90)
f=2*7.292e-5*math.sin(math.radians(42.957160)); P=2*math.pi/f/3600
print(f"  {len(t)} dumps to {t[-1]:.2f} simulated hours = {t[-1]/P:.2f} inertial periods "
      f"(2pi/f = {P:.1f} h)")
sel = t >= t[-1]-SCORE_H
def tr(y):
    A=np.vstack([t[sel],np.ones(sel.sum())]).T
    return 100*np.linalg.lstsq(A,y[sel],rcond=None)[0][0]/max(abs(y[sel].mean()),1e-30)
print(f"\n  per-hour windows: is the TURBULENCE settled while the mean turns?")
print(f"  {'window':>12}{'u*':>9}{'U(10)':>8}{'U/u*':>8}{'sw/u*':>8}{'TKE/u*^2':>10}"
      f"{'z_i':>7}{'dir':>7}")
for lo in np.arange(1.0, t[-1]-1.0+1e-9, 1.0):
    m=(t>=lo)&(t<lo+1.0)
    if m.sum()<4: continue
    print(f"  {lo:4.1f}-{lo+1:<7.1f}{us[m].mean():9.4f}{Um[m].mean():8.3f}"
          f"{(Um/us)[m].mean():8.3f}{(sw/us)[m].mean():8.3f}{(tke/us**2)[m].mean():10.3f}"
          f"{zi[m].mean():7.0f}{wd[m].mean():7.1f}")
print(f"\n  === GATED: the footprint's controlling parameters, last {SCORE_H:.1f} h ===")
ok=True
for nm,y in (("U/u* (Kljun Pi_4)",Um/us),("sigma_v/u*",sv/us),
             ("sigma_w/u* at the receptor",sw/us),("TKE/u*^2",tke/us**2),
             ("z_i",zi),("Kljun x_peak",xp),("Kljun x90",x90)):
    v=tr(y); g_=abs(v)<LIM[nm]; ok&=g_
    print(f"  {nm:<28}{y[sel].mean():10.4f}{v:+9.2f} %/h  (limit {LIM[nm]:.0f})   "
          f"{abs(v)*40/60:5.2f}% per 40-min window   {'ok' if g_ else 'DRIFTING'}")
print(f"\n  === REPORTED, not gated: the mean flow rides the inertial oscillation ===")
for nm,y in (("u*",us),("U(10 m)",Um),("wind direction",wd),("domain TKE",tke)):
    print(f"  {nm:<28}{y[sel].mean():10.4f}{tr(y):+9.2f} %/h")
print(f"  x_peak spans {xp[sel].min():.1f}-{xp[sel].max():.1f} m across the scored window, "
      f"against a {16.0:.0f} m raster cell.")
print(f"\n  GATE C1: {'PASS -- the turbulence is in equilibrium and the footprint parameters are stationary; the mean flow is oscillating inertially, which is physical and is carried as a per-case LABEL' if ok else 'FAIL -- a footprint-controlling parameter is still drifting'}")
raise SystemExit(0 if ok else 1)
PY
[ "${PIPESTATUS[0]}" = "0" ] || die "C1 stationarity"

# ---------------------------------------------------------------- Gate C2: bit-for-bit
say "Gate C2: the restart is a lossless state resume"
# WHAT THIS GATE CAN AND CANNOT ASSERT.
#
# The load-bearing claim is that a chained segment boundary is not a seam -- that segment
# N+1 starts from exactly the state segment N ended in, so a 7-segment spin-up is the same
# trajectory a single long run would have produced. That is a statement about the restart
# READ, and it is exactly testable: Nt = the restart step performs zero timesteps and
# writes the state straight back out (trap 6), so the echo must equal the file BIT FOR BIT.
#
# It is NOT testable by re-running a whole segment and demanding bit equality with the
# chain. FastEddy is chaotic and runs in fp32, and its reductions are not bitwise
# reproducible run to run, so ANY two executions of the same 20,000 steps diverge -- as
# they do for two identical re-runs with no restart involved at all. An earlier cut of
# this gate demanded that equality and "failed" a lossless restart. The divergence is
# still REPORTED here, next to the floor from two identical re-runs, because the ratio of
# those two numbers is what tells you which of the two things you are looking at.
SERIES2=$(ls -1 $SPIN/output/FE_G16.* | sort -t. -k2 -n | tail -2)
PREV=$(echo "$SERIES2" | head -1); LASTD=$(echo "$SERIES2" | tail -1)
PSTEP=${PREV##*.}; LSTEP=${LASTD##*.}
mkdir -p runs/g16_c2/output; rm -f runs/g16_c2/output/* runs/g16_c2/FE_RST.*
cp -f "$PREV" runs/g16_c2/FE_RST.$PSTEP
echo "  restarting from step $PSTEP, re-running to step $LSTEP ($((LSTEP-PSTEP)) steps)"
sed -e "s|^dt = .*|dt = $DT_FLAT|" -e "s|^Nt = .*|Nt = $LSTEP|" \
    -e "s|^NtBatch = .*|NtBatch = $((LSTEP-PSTEP))|" \
    -e "s|^frqOutput = .*|frqOutput = $((LSTEP-PSTEP))|" \
    -e 's|^inPath = .*|inPath = ./|' -e "s|^inFile = .*|inFile = FE_RST.$PSTEP|" \
    -e 's|^outFileBase = .*|outFileBase = FE_C2|' "$BASE" > runs/g16_c2/c2.in
./docker/run_case.sh runs/g16_c2 c2.in $L/g16_c2.log || die "C2 run"
./docker/pyrun.sh - "$PREV" "$LASTD" "$PSTEP" "$LSTEP" <<'PY' | tee $R/g16_c2.txt
import sys, os, numpy as np
from netCDF4 import Dataset
prev, lastd, pstep, lstep = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
echo = f"runs/g16_c2/output/FE_C2.{pstep}"
fwd  = f"runs/g16_c2/output/FE_C2.{lstep}"
FLDS = ('u','v','w','theta','rho','pressure','TKE_0','fricVel','htFlux','z0m')
print(f"  === GATED: the restart READ, echoed back before any timestep ===")
print(f"  {prev}\n  -> {echo}")
bad = 0
if not os.path.exists(echo):
    print("  FAIL: no step-0 echo written"); bad = 1
else:
    with Dataset(prev) as A, Dataset(echo) as B:
        for v in FLDS:
            if v not in A.variables or v not in B.variables: continue
            x=np.asarray(A[v][:],dtype=np.float64); y=np.asarray(B[v][:],dtype=np.float64)
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                print(f"  {v:<10} NON-FINITE"); bad += 1; continue
            d=float(np.abs(x-y).max()); bad += (d != 0)
            print(f"  {v:<10} max|diff| = {d:.3e}   {'BIT-FOR-BIT' if d==0 else 'LOSSY'}")
print(f"\n  === REPORTED: divergence over {int(lstep)-int(pstep)} steps, and its floor ===")
if os.path.exists(fwd):
    with Dataset(lastd) as A, Dataset(fwd) as B:
        for v in ('u','theta','TKE_0'):
            x=np.asarray(A[v][:],dtype=np.float64); y=np.asarray(B[v][:],dtype=np.float64)
            print(f"  {v:<10} chain vs re-run: max {np.abs(x-y).max():.3e}, "
                  f"rms {np.sqrt(((x-y)**2).mean()):.3e}")
    print("  Two identical re-runs of the same steps diverge by the same order -- this is")
    print("  fp32 chaos, not a lossy restart. See results/g16_c2_floor.txt.")
print(f"\n  GATE C2: {'PASS -- the restart read is lossless, so a chained boundary is not a seam' if not bad else 'FAIL -- the restart read is LOSSY'}")
raise SystemExit(1 if bad else 0)
PY
[ "${PIPESTATUS[0]}" = "0" ] || die "C2 restart read is lossy"

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
TSTEP=${TADJ##*.}
# Nt is an ABSOLUTE target step, not a step count (FASTEDDY_TRAPS.md 6): restarting from
# step 40000 with Nt = 4000 performs ZERO timesteps, writes one dump and exits 0. Every
# rung would then have scored the adjustment dump instead of its own.
TNT=$((TSTEP + 4000))
{
echo "terrain dt ladder, branched from $TADJ (adjusted at CFL_3d 1.05)"
echo "each rung: 4000 steps from step $TSTEP to $TNT"
printf "%-9s %-12s %-9s %s\n" "CFL_3d" "dt (s)" "k0/k1" "verdict"
for cfl in 1.00 1.10 1.15 1.20 1.25 1.30 1.35 1.40; do
  DTT=$(python3 -c "print(f'{$cfl/92.20239148570417:.7f}')")
  sed -e "s|^dt = .*|dt = $DTT|" -e "s|^Nt = .*|Nt = $TNT|" -e 's|^NtBatch = .*|NtBatch = 4000|' \
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
echo
echo "The domain-mean k0/k1 above is NOT sufficient over terrain: the amplification is"
echo "local, and 1.7% of cells carry the steep slopes. Condition on slope:"
./docker/pyrun.sh bin/k0k1_by_slope.py runs/g16_terr/output/FE_TDT.$TNT --grid "$GRID"
echo
echo "And k0/k1 -- by slope or domain-mean -- is a dt check, not a physics check. It read"
echo "0.442 on a boundary layer whose turbulence had entirely collapsed. Ask separately:"
./docker/pyrun.sh docker/turb_alive.py --calibrate "runs/g16_terr/output/FE_TDT.*"
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
# KEEP_FIELDS is on because GATE D1 -- the well-mixed test -- runs on these dumps and must
# run BEFORE they are deleted. Leaving it off silently skipped the one gate the plan calls
# non-negotiable: if particles accumulate near the surface, every footprint computed
# afterwards is wrong in exactly the near field where the whole signal now lives.
KEEP_FIELDS=1 TBACK=600 DT="$DT_WIN" SRC=runs/g16_flat/output/FE_ADJ.0 D=runs/g16_flat \
  GRID="$GRID" TAG=g16_flat BASE="$BASE" bin/regression_flat.sh --baseline \
  2>&1 | tee $R/g16_phaseD.txt

say "Gate D1: the well-mixed condition, IN THE PRODUCTION CLOSURE"
# --sgs-most is not optional here. The floor rescales sigma^2 by a height-dependent factor,
# and a rescaling the Thomson drift does not know about breaks well-mixedness -- which is
# exactly the bug the fourth pass found. The gate has to run in the configuration the
# footprints are actually computed in, not in the unmodified one.
./docker/pyrun.sh bin/stage4_wellmixed.py runs/g16_flat/window --dt "$DT_WIN" \
  --z-target 10.0 --tlimit 600 --sgs-most --fp16-cache 2>&1 | tee $R/g16_wellmixed.txt
WM=${PIPESTATUS[0]}
rm -f runs/g16_flat/window/*
[ "$WM" = "0" ] || die "Gate D1 well-mixed"

# ---------------------------------------------------------------- t_back for production
say "t_back, read off the control window's own capture curve"
./docker/pyrun.sh bin/pick_tback.py $R/g16_flat.json --out $R/tback_production.txt 2>&1 \
  | tee $R/g16_tback.txt
[ -s $R/tback_production.txt ] || die "t_back was not determined -- Phase F cannot be sized"

say "PASS5 CHAIN COMPLETE: C1, C2, B4, Phase D and t_back"
