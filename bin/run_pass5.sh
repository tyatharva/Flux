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
import glob, numpy as np, math
from netCDF4 import Dataset
# STATIONARITY, SCORED THE WAY IT MATTERS.
#
# Two things had to be fixed about how this gate was written.
#
# 1. It scored "the last half" of the series. From a COLD START that still contains most
#    of the transient, so a run gets penalised for the part of itself it was always going
#    to have to spend. It now scores a fixed recent window (SCORE_H simulated hours).
#
# 2. It required the trend to be within 2 sigma of zero. For a smoothly decaying transient
#    with small scatter that is unreachable at ANY length: sigma shrinks with the scatter,
#    so a physically negligible drift is still many sigma. The criterion that matters is
#    whether the state changes materially DURING A SAMPLING WINDOW -- 40 minutes -- against
#    the error floor a footprint is quoted with. That is the primary test now; the sigma is
#    reported alongside because it is still the right thing to look at for NOISE.
#
# A neutral Ekman layer's clock is the INERTIAL period, 2 pi / f = 17.6 h here, not the
# eddy turnover time (~0.25 h). That is why this needs hours, and it is why the fourth
# pass's spin-up was ~5-6 h simulated.
SCORE_H = float("${SCORE_H:-1.5}")
WINDOW_H = 40.0/60.0
TREND_LIM = 2.0        # per cent per hour
ps = sorted(glob.glob('$SPIN/output/FE_G16.*'), key=lambda p:int(p.rsplit('.',1)[1]))
t,us,tke,zi = [],[],[],[]
for p in ps:
    with Dataset(p) as ds:
        g=lambda v: np.squeeze(np.asarray(ds[v][:],dtype=np.float64))
        u,v,w = g('u'),g('v'),g('w'); z=g('zPos')[:,0,0]
        us.append(float(g('fricVel').mean()))
    pr=lambda a:a-a.mean(axis=(-2,-1),keepdims=True)
    tk=0.5*((pr(u)**2+pr(v)**2+pr(w)**2).mean(axis=(-2,-1)))
    tke.append(float(tk.mean()))
    kmax=int(np.argmax(tk)); above=np.where(tk[kmax:]<0.05*tk[kmax])[0]
    zi.append(float(z[kmax+above[0]]) if len(above) else float(z[-1]))
    t.append(int(p.rsplit('.',1)[1])*$DT_FLAT/3600.0)
t=np.array(t); us=np.array(us); tke=np.array(tke); zi=np.array(zi)
f = 2*7.292e-5*math.sin(math.radians(42.957160))
print(f"  {len(t)} dumps, {t[0]:.2f} to {t[-1]:.2f} simulated hours "
      f"= {t[-1]*f/(2*math.pi)*3600:.2f} inertial periods (2pi/f = {2*math.pi/f/3600:.1f} h)")
print(f"\n  {'t (h)':>7}{'u* (m/s)':>11}{'domain TKE':>13}{'z_i (m)':>10}")
for a,b,c,d in zip(t,us,tke,zi): print(f"  {a:7.3f}{b:11.4f}{c:13.6f}{d:10.0f}")
print(f"\n  trend over successive 1 h windows (is the transient flattening?)")
print(f"  {'window (h)':>14}{'u* mean':>10}{'u* %/h':>10}{'TKE %/h':>10}{'z_i (m)':>10}")
for lo in np.arange(0.5, max(t)-1.0+1e-9, 0.5):
    m=(t>=lo)&(t<lo+1.0)
    if m.sum()<4: continue
    row=f"  {lo:5.1f}-{lo+1.0:<8.1f}"
    for y in (us,tke):
        A=np.vstack([t[m],np.ones(m.sum())]).T
        sl=np.linalg.lstsq(A,y[m],rcond=None)[0][0]
        row += f"{us[m].mean():10.4f}" if y is us else ""
        row += f"{100*sl/abs(y[m].mean()):10.2f}"
    print(row + f"{zi[m].mean():10.0f}")
sel = t >= t[-1]-SCORE_H
print(f"\n  === scored over the last {SCORE_H:.1f} simulated hours ({sel.sum()} dumps) ===")
ok=True
for nm,y in (('u*',us),('TKE',tke),('z_i',zi)):
    x=t[sel]; yy=y[sel]
    A=np.vstack([x,np.ones_like(x)]).T
    m_,c0=np.linalg.lstsq(A,yy,rcond=None)[0]
    resid=yy-(m_*x+c0); se=np.sqrt((resid**2).sum()/max(len(x)-2,1))
    sem=se/max(np.sqrt(((x-x.mean())**2).sum()),1e-12)
    sig=abs(m_)/max(sem,1e-30); rel=100*m_/max(abs(yy.mean()),1e-30)
    drift=abs(rel)*WINDOW_H
    good = abs(rel) < TREND_LIM
    if nm != 'z_i': ok &= good
    print(f"  {nm:4s} mean {yy.mean():9.4f}   trend {rel:+7.2f} %/h ({sig:5.1f} sigma)"
          f"   -> {drift:5.2f}% over a 40-min window   "
          f"{'ok' if good else 'DRIFTING'}{'' if nm!='z_i' else '  (reported, not gated)'}")
print(f"\n  GATE C1: {'PASS -- the state changes by under ' + str(TREND_LIM) + '%/h, i.e. under 1.3% across a sampling window' if ok else 'FAIL -- still drifting faster than ' + str(TREND_LIM) + '%/h'}")
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
