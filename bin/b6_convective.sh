#!/usr/bin/env bash
# Gate B6, RE-RUN CONVECTIVELY. 90-degree equivariance on a flat uniform CBL.
#
# WHY THIS EXISTS. B6 passed in the fifth pass on a NEUTRAL state, and the seed library
# leans on the same rotation to turn 3 base angles into 12 headings -- for convective rungs
# as much as neutral ones. PROJECT_BRIEF.md's own conventions forbid carrying a gate across
# regimes: "treat a regime where a component is inert as NO EVIDENCE AT ALL about that
# component", written after the neutral well-mixed gate passed a closure carrying nine
# turnovers. A convective boundary layer has buoyancy, entrainment and a prescribed surface
# heat flux that the neutral test exercised none of, so it gets its own run.
#
# EQUIVARIANCE IS A STATEMENT ABOUT STATISTICS, NOT TRAJECTORIES. FastEddy is chaotic and
# fp32, and the [i][j][k] layout with kStride = 1 sums x and y in different orders, so the
# two runs are asymmetric at roundoff from the first step and that seed diverges. What must
# agree is the slab-mean profiles, to a tolerance set by what a FOOTPRINT can resolve.
#
# The rotation itself is separately exact (1.2e-14, fifth pass), so prep_restart.py is not
# what any residual is measuring.
#
#   usage: bin/b6_convective.sh [spun-up convective dump]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
SRC="${1:-$(ls -1 runs/g16_cbl_shallow/output/FE_CBL.* 2>/dev/null | sort -t. -k2 -n | tail -1)}"
[ -f "$SRC" ] || { echo "FATAL: no convective dump ($SRC)" >&2; exit 1; }
BASE="${BASE:-runs/g16_base/base_cbl_shallow.in}"
DT="${DT:-0.0146199}"
NT="${NT:-200}"
D=runs/g16_b6cbl
die(){ echo "FATAL: $*" >&2; exit 1; }
WTH=$(grep -oP '^surflayer_wth\s*=\s*\K[^#[:space:]]*' "$BASE")
echo "########## B6 CONVECTIVE: 90-deg equivariance, w'th_v' = $WTH ##########"
echo "  source $SRC"
[ "$(python3 -c "print(int(float('$WTH')>0.01))")" = "1" ] \
  || die "surflayer_wth = $WTH is not convective; this would be the neutral gate again"

for c in a b; do
  if [ "$c" = a ]; then ROT=0; UG=10.000000; VG=0.000000; else ROT=1; UG=0.000000; VG=10.000000; fi
  mkdir -p $D/$c/output
  ./docker/pyrun.sh bin/prep_restart.py "$SRC" "$D/$c/FE_RST.0" --rot $ROT --flat >/dev/null \
      || die "prep_restart rot$ROT"
  sed -e "s|^dt = .*|dt = $DT|" -e "s|^Nt = .*|Nt = $NT|" -e "s|^NtBatch = .*|NtBatch = $NT|" \
      -e "s|^frqOutput = .*|frqOutput = $NT|" -e 's|^inPath = .*|inPath = ./|' \
      -e 's|^inFile = .*|inFile = FE_RST.0|' -e 's|^topoFile = .*|topoFile = |' \
      -e "s|^U_g = .*|U_g = $UG|" -e "s|^V_g = .*|V_g = $VG|" \
      -e 's|^outFileBase = .*|outFileBase = FE_EQC|' \
      -e 's|^lsfSelector = .*|lsfSelector = 0|' -e 's|^lsf_horMnSubTerms = .*|lsf_horMnSubTerms = 0|' \
      "$BASE" > $D/$c/eq.in
  rm -f $D/$c/output/*
  ./docker/run_case.sh $D/$c eq.in /tmp/flux-logs/g16_b6cbl_$c.log >/dev/null || die "run $c"
  echo "  rot=$ROT complete"
done

./docker/pyrun.sh - <<'PY'
import glob, numpy as np
from netCDF4 import Dataset
def prof(f):
    with Dataset(f) as ds:
        g=lambda v: np.squeeze(np.asarray(ds[v][:],dtype=np.float64))
        u,v,w,th,e=g('u'),g('v'),g('w'),g('theta'),np.maximum(g('TKE_0'),0)
        z=g('zPos')[:,0,0]
    if not all(np.isfinite(a).all() for a in (u,v,w,th,e)):
        raise SystemExit("  FAIL: non-finite field")
    pr=lambda a: a-a.mean(axis=(-2,-1),keepdims=True)
    return dict(z=z, spd=np.hypot(u.mean(axis=(-2,-1)), v.mean(axis=(-2,-1))),
                tke=0.5*((pr(u)**2+pr(v)**2+pr(w)**2).mean(axis=(-2,-1))),
                th=th.mean(axis=(-2,-1)), e=e.mean(axis=(-2,-1)),
                # the convective additions: buoyancy flux and w variance are what a
                # neutral run has nothing meaningful of, so they are what this gate is for
                wt=(pr(w)*pr(th)).mean(axis=(-2,-1)),
                ww=(pr(w)**2).mean(axis=(-2,-1)))
a=prof(sorted(glob.glob('runs/g16_b6cbl/a/output/FE_EQC.*'))[-1])
b=prof(sorted(glob.glob('runs/g16_b6cbl/b/output/FE_EQC.*'))[-1])
m = a['tke'] > 0.01*a['tke'].max()
ka=int(np.argmin(a['wt'])); kb=int(np.argmin(b['wt']))
print(f"\n  z_i (buoyancy-flux minimum): rot0 {a['z'][ka]:.0f} m, rot1 {b['z'][kb]:.0f} m")
print(f"  {'field':<28}{'max rel diff':>14}{'tolerance':>12}   verdict")
bad=0
for nm,tol,lab in (('spd',1e-3,'mean wind speed'),
                   ('th',1e-4,'mean theta'),
                   ('tke',3e-2,'resolved TKE'),
                   ('e',3e-2,'SGS TKE'),
                   ('ww',3e-2,'sigma_w^2 (resolved)'),
                   ('wt',5e-2,"buoyancy flux w'theta'")):
    x,y=a[nm],b[nm]
    d=float(np.max(np.abs(x[m]-y[m])/np.maximum(np.abs(x[m]),1e-12)))
    ok=d<tol; bad+=(not ok)
    print(f"  {lab:<28}{d:14.3e}{tol:12.1e}   {'ok' if ok else 'DIFFERS'}")
ia=float(np.trapezoid(a['tke'],a['z'])); ib=float(np.trapezoid(b['tke'],b['z']))
d=abs(ia/ib-1); bad+=(d>=1e-2)
print(f"  {'column-integrated TKE':<28}{d:14.3e}{1e-2:12.1e}   {'ok' if d<1e-2 else 'DIFFERS'}")
print(f"\n  GATE B6 (CONVECTIVE): {'PASS' if not bad else 'FAIL'}")
raise SystemExit(1 if bad else 0)
PY
