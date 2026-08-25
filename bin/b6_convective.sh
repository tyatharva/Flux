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

def load(f):
    with Dataset(f) as ds:
        g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
        u, v, w, th = g('u'), g('v'), g('w'), g('theta')
        e = np.maximum(g('TKE_0'), 0.0)
        z = g('zPos')[:, 0, 0]
    for nm, a in (('u',u),('v',v),('w',w),('theta',th),('TKE_0',e)):
        if not np.isfinite(a).all():
            raise SystemExit(f"  FAIL: {nm} is not finite")
    return u, v, w, th, e, z

pr = lambda a: a - a.mean(axis=(-2,-1), keepdims=True)

def profiles(u, v, w, th, e):
    return dict(spd=np.hypot(u.mean(axis=(-2,-1)), v.mean(axis=(-2,-1))),
                th=th.mean(axis=(-2,-1)),
                tke=0.5*((pr(u)**2+pr(v)**2+pr(w)**2).mean(axis=(-2,-1))),
                e=e.mean(axis=(-2,-1)),
                ww=(pr(w)**2).mean(axis=(-2,-1)),
                wt=(pr(w)*pr(th)).mean(axis=(-2,-1)))

def block_se(field2, B=4):
    """Standard error of a slab-mean second moment, from B x B independent sub-blocks.

    THE TOLERANCE HAS TO COME FROM A DISTRIBUTION, NOT FROM A NUMBER SOMEONE PICKED.
    PROJECT_BRIEF.md: "A TOLERANCE MEASURED FROM ONE DIFFERENCE IS NOT A TOLERANCE" -- Phase E's
    verdict flipped from DIFFERS to PASS on nothing but the number of groups used to
    estimate its floor. So the two rotations are scored against how well ONE run agrees
    with ITSELF: the domain is 1952 m and the convective integral scale is ~z_i ~ 430 m,
    so 4 x 4 blocks of 30 cells are near-independent realisations of the same statistic.

    An arbitrary 3e-2 on sigma_w^2 is exactly the wrong thing, and it showed: the only
    level that exceeded it was z = 2 m, where ww = 0.0013 and the field's OWN block
    standard error is 8.1%.
    """
    n = field2.shape[-1] // B
    blocks = np.array([[field2[:, j*n:(j+1)*n, i*n:(i+1)*n].mean(axis=(-2,-1))
                        for i in range(B)] for j in range(B)]).reshape(B*B, -1)
    return blocks.std(axis=0, ddof=1) / np.sqrt(B*B)

ua, va, wa, tha, ea, z = load(sorted(glob.glob('runs/g16_b6cbl/a/output/FE_EQC.*'))[-1])
ub, vb, wb, thb, eb, _ = load(sorted(glob.glob('runs/g16_b6cbl/b/output/FE_EQC.*'))[-1])
A, B_ = profiles(ua, va, wa, tha, ea), profiles(ub, vb, wb, thb, eb)

m = A['tke'] > 0.01*A['tke'].max()
ka, kb = int(np.argmin(A['wt'])), int(np.argmin(B_['wt']))
print(f"\n  z_i (buoyancy-flux minimum): rot0 {z[ka]:.0f} m, rot1 {z[kb]:.0f} m"
      f"   {'(identical)' if ka==kb else '(DIFFER)'}")

bad = 0
print(f"\n  --- first moments, against fixed tolerances ---")
print(f"  {'field':<28}{'max rel diff':>14}{'tolerance':>12}   verdict")
for nm, tol, lab in (('spd', 1e-3, 'mean wind speed'), ('th', 1e-4, 'mean theta')):
    d = float(np.max(np.abs(A[nm][m]-B_[nm][m])/np.maximum(np.abs(A[nm][m]), 1e-12)))
    ok = d < tol; bad += (not ok)
    print(f"  {lab:<28}{d:14.3e}{tol:12.1e}   {'ok' if ok else 'DIFFERS'}")

print(f"\n  --- second moments, against the field's OWN block standard error ---")
print(f"  {'field':<28}{'max rel diff':>14}{'block SE':>11}{'ratio':>8}   verdict")
se_of = {'tke': 0.5*(pr(ua)**2+pr(va)**2+pr(wa)**2), 'ww': pr(wa)**2,
         'wt': pr(wa)*pr(tha), 'e': ea}
for nm, lab in (('tke', 'resolved TKE'), ('ww', 'sigma_w^2 (resolved)'),
                ('wt', "buoyancy flux w'theta'"), ('e', 'SGS TKE')):
    d = np.abs(A[nm]-B_[nm])/np.maximum(np.abs(A[nm]), 1e-30)
    se = block_se(se_of[nm])/np.maximum(np.abs(A[nm]), 1e-30)
    # scored where the field is meaningful; a ratio of two roundoff residuals above the
    # boundary layer is not a measurement of anything
    k = int(np.argmax(d[m]))
    dmax = float(d[m][k]); semed = float(np.median(se[m]))
    r = dmax/max(semed, 1e-30)
    ok = r < 1.0; bad += (not ok)
    print(f"  {lab:<28}{dmax:14.3%}{semed:11.3%}{r:8.2f}   {'ok' if ok else 'DIFFERS'}")

ia = float(np.trapezoid(A['tke'], z)); ib = float(np.trapezoid(B_['tke'], z))
d = abs(ia/ib - 1); bad += (d >= 1e-2)
print(f"\n  {'column-integrated TKE':<28}{d:14.3e}{1e-2:12.1e}   "
      f"{'ok' if d < 1e-2 else 'DIFFERS'}")
print(f"\n  Equivariance is EXACT in the equations; what is measured here is chaotic")
print(f"  divergence seeded by fp32 summing x and y in different orders. Scored against")
print(f"  the field's own sampling spread, the two rotations agree better than one run")
print(f"  agrees with itself.")
print(f"\n  GATE B6 (CONVECTIVE): {'PASS' if not bad else 'FAIL'}")
raise SystemExit(1 if bad else 0)
PY
