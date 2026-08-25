#!/usr/bin/env bash
# Seventh pass: the STABLE well-mixed gate, and the domain cap at z_i = 1200 m.
#
# TWO THINGS THE CORPUS NEEDS AND DOES NOT HAVE.
#
# 1. GATE D1 HAS NEVER RUN IN STABLE CONDITIONS. bin/run_pass6.sh ran the battery on
#    g16_flat (neutral) and g16_flatcbl (convective) and nothing else. A corpus that walks
#    the diurnal cycle at a site stable ~29% of QC'd hours therefore contains a regime the
#    closure has no evidence in -- the same inheritance mistake the fifth pass made when it
#    called the convective closure validated by a neutral PASS.
#
# 2. THE z_i CAP IS BEING RAISED 976 -> 1200 m, AND 1200 IS OUTSIDE WHAT PHASE E MEASURED.
#    Phase E compared L/z_i = 4.56 against 2.28 and found the 10 m footprint
#    indistinguishable (p ~ 0.54). 1200 m is L/z_i = 1.63. Phase E's own experiment is the
#    template: run the deeper case, diagnose lock-in directly from the w spectrum, and
#    compare the footprint observables against the window's OWN half-vs-half floor.
#
# THE GATE PARAMETERS SCALE WITH z_i, AND THE RATIO IS HELD FIXED. wellmixed.run_test
# releases uniformly over [z_touch, z_release] and scores [z_touch, z_score] in 20 bins, so
# the per-bin count is n*(z_score/z_release)/20 and the counting-noise floor is
# sqrt(2/that). The convective and neutral rows used 400/1200 = 1/3 with n = 40000, giving
# 667 per bin and the 5.48% floor every published row is quoted against. Scoring a 150 m
# stable layer over 0-400 m would put most of the scored column in the free atmosphere
# above the SBL, where nothing moves and the ratio is trivially 1. So z_score tracks the
# regime's own z_i and z_release stays 3x it: the layer is right AND the floor stays 5.48%,
# which is what makes the stable row comparable to the other two rather than merely present.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
L="${LOGDIR:-${TMPDIR:-/tmp}/flux-logs}"; mkdir -p "$L" results
R=results
DT_WIN=0.0146199; WIN=2400; NPART=40000; TB=600
SEED=jobs/seed_sbl_a030/return/seed_restart.nc
die(){ echo "FATAL: $*" >&2; exit 1; }
say(){ echo; echo "########## $* ##########"; date '+%F %H:%M:%S'; }

bash bin/preflight.sh || die "preflight"

# =====================================================================================
say "1. the STABLE flat control window"
# =====================================================================================
[ -f "$SEED" ] || die "no sbl seed at $SEED -- run jobs/run_seed.sh jobs/seed_sbl_a030"
read -r UG VG < <(python3 -c "
import re
s=open('jobs/seed_sbl_a030/seed.in').read()
g=lambda k: re.search(rf'^{k}\s*=\s*([^#\s]+)', s, re.M).group(1)
print(g('U_g'), g('V_g'))")
BASE=jobs/seed_sbl_a030/seed.in bin/run_window.sh runs/g16_flatsbl "$SEED" \
    $DT_WIN $WIN - "$UG" "$VG" || die "sbl window"
# ASSERT ON THE ARTIFACT: run_window.sh stamps .window_complete only on success, and the
# dump count is the thing the gate actually consumes.
NW=$(ls -1 runs/g16_flatsbl/window/*.[0-9]* 2>/dev/null | wc -l)
[ "$NW" -ge 100 ] || die "the sbl window holds only $NW dumps"
echo "--- sbl window: $NW dumps"

# =====================================================================================
say "2. GATE D1, STABLE -- production closure and the no-floor control"
# =====================================================================================
# The scored layer is set from the window's OWN achieved z_i, not from the rung's target.
read -r ZI ZSCORE ZREL < <(./docker/pyrun.sh - <<'PY'
import glob, numpy as np
from netCDF4 import Dataset
ps=sorted(glob.glob('runs/g16_flatsbl/window/*.[0-9]*'), key=lambda p:int(p.rsplit('.',1)[1]))
p=ps[len(ps)//2]
with Dataset(p) as ds:
    g=lambda v: np.squeeze(np.asarray(ds[v][:],dtype=np.float64))
    z=g('zPos')[:,0,0]; u,v,w=g('u'),g('v'),g('w')
pr=lambda a: a-a.mean(axis=(-2,-1),keepdims=True)
tk=0.5*((pr(u)**2+pr(v)**2+pr(w)**2).mean(axis=(-2,-1)))
km=int(np.argmax(tk)); ab=np.where(tk[km:]<0.05*tk[km])[0]
zi=float(z[km+ab[0]]) if len(ab) else float(z[-1])
# round to something readable; hold z_release = 3 * z_score so the floor stays 5.48%
zs=max(60.0, round(zi/10.0)*10.0)
print(f"{zi:.1f} {zs:.0f} {3*zs:.0f}")
PY
) || die "could not size the scored layer"
echo "--- achieved stable z_i = $ZI m  ->  score 2-$ZSCORE m, release 2-$ZREL m"
echo "---   (z_score/z_release = 1/3 as in the neutral and convective rows, so the"
echo "---    counting-noise floor is the same 5.48% and the rows are comparable)"

for CFG in new nofloor; do
  FLAGS="--sgs-most"; [ "$CFG" = nofloor ] && FLAGS=""
  echo; echo "--- well-mixed, STABLE, closure=$CFG"
  ./docker/pyrun.sh bin/stage4_wellmixed.py runs/g16_flatsbl/window --dt $DT_WIN \
      --n $NPART --z-target 10.0 --dmap data/grid16/dmap.npy --zscore "$ZSCORE" \
      --zrelease "$ZREL" $FLAGS --fp16-cache --tag sbl_wm_$CFG 2>&1 \
      | tee $R/sbl_wm_$CFG.txt
  [ -s "$R/sbl_wm_$CFG.json" ] || { tail -20 $R/sbl_wm_$CFG.txt >&2
    die "the stable gate wrote no JSON for closure=$CFG"; }
done

./docker/pyrun.sh - <<'PY' | tee $R/sbl_gate_d1.txt
import json, numpy as np
print("\n=== GATE D1, STABLE -- the row that belongs beside neutral and convective ===")
print(f"  {'regime':<9}{'closure':<12}{'backward rms / lo3':>23}{'forward rms / lo3':>22}"
      f"{'max fac':>9}{'receptor':>10}{'turn':>6}   verdict")
ok = True
for cfg, lab in (("nofloor", "no floor"), ("new", "production")):
    d = json.load(open(f"results/sbl_wm_{cfg}.json"))
    b, f = d["backward"], d["forward"]
    fl = d.get("floor")
    if fl:
        zl = np.asarray(fl["zl"], float); fac = np.asarray(fl["fac"], float)
        mx = f"{fac.max():.2f}"; rc = f"{np.interp(10.0, zl, fac):.3f}"
        tn = f"{int(fl['n_new_turnovers']):d}"
    else:
        mx = rc = tn = "--"
    print(f"  {'stable':<9}{lab:<12}{b['rms']*100:10.2f}% / {b['lo3']:<8.3f}"
          f"{f['rms']*100:10.2f}% / {f['lo3']:<8.3f}{mx:>9}{rc:>10}{tn:>6}   "
          f"{'PASS' if d['pass'] else 'FAIL'}")
    print(f"  {'':21}(noise {b['noise']*100:.2f}%, maxdev {b['maxdev']*100:.2f}% back / "
          f"{f['maxdev']*100:.2f}% fwd)")
    if cfg == "new":
        ok = bool(d["pass"])
print(f"\n  counting-noise floor as measured in-run; the neutral and convective rows were")
print(f"  quoted against 5.48%, held here by keeping z_score/z_release = 1/3 and n = 40000.")
print(f"  GATE D1 STABLE: {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
PY
GATE=${PIPESTATUS[0]}
if [ "$GATE" != "0" ]; then
  echo
  echo "########## GATE D1 STABLE: FAIL -- STOPPING ##########"
  echo "  Not tuning the floor to make it pass. The no-floor control above is the"
  echo "  diagnostic: if the base model passes and the floor does not, the fault is the"
  echo "  floor in stable stratification, where phi_w is a different MOST branch and the"
  echo "  anchor factor has never been measured."
  exit 1
fi
say "GATE D1 STABLE PASSES -- continuing to the domain cap"

# =====================================================================================
say "3. the z_i = 1200 m case -- spin-up (L/z_i = 1.63)"
# =====================================================================================
# Phase E compared L/z_i = 4.56 against 2.28 and found the 10 m footprint
# indistinguishable. Raising the cap to 1200 m means L/z_i = 1.63, which is OUTSIDE that,
# so it gets the same experiment rather than an extrapolation.
#
# The pair differs ONLY in the capping inversion: same grid, same geostrophic wind, and the
# SAME surface heat flux (0.1363), so u* and L match and the comparison isolates the BOX.
# One recorded asymmetry: cbl_shallow was spun at CFL_3d = 1.500 and this case at the
# production 1.348. Both are below the measured 1.51 accuracy boundary, both windows run at
# the same dt, and in a test looking for the ABSENCE of a difference an extra source of
# difference is conservative -- it can only make agreement harder to obtain.
D=runs/g16_cbl_1200 BASE=runs/g16_base/base_cbl_1200.in NSEG="${NSEG_1200:-6}" \
  DT=0.0146199 SPS=0.0149 OUTBASE=FE_CBL bin/spin_cbl.sh || die "cbl_1200 spin-up"
LAST1200=$(ls -1 runs/g16_cbl_1200/output/FE_CBL.* 2>/dev/null | sort -t. -k2 -n | tail -1)
[ -n "$LAST1200" ] || die "the 1200 m spin-up wrote no dump"
echo "--- deepest dump: $LAST1200"

# =====================================================================================
say "4. windows for the domain-cap pair"
# =====================================================================================
SRC490=$(ls -1 runs/g16_cbl_shallow/output/FE_CBL.* | sort -t. -k2 -n | tail -1)
BASE=runs/g16_base/base_cbl_shallow.in bin/run_window.sh runs/g16_adq490 "$SRC490" \
    $DT_WIN $WIN - 10.000000 0.000000 || die "z_i 490 window"
BASE=runs/g16_base/base_cbl_1200.in bin/run_window.sh runs/g16_adq1200 "$LAST1200" \
    $DT_WIN $WIN - 10.000000 0.000000 || die "z_i 1200 window"
for W in runs/g16_adq490 runs/g16_adq1200; do
  N=$(ls -1 $W/window/*.[0-9]* 2>/dev/null | wc -l)
  [ "$N" -ge 100 ] || die "$W holds only $N dumps"
done

# =====================================================================================
say "5. lock-in diagnostic, and the footprint observables"
# =====================================================================================
./docker/pyrun.sh bin/domain_adequacy.py spectra \
    $(ls -1 runs/g16_adq490/window/*.[0-9]* | sort -t. -k2 -n | tail -3) \
    2>&1 | tee $R/adq490_spectra.txt
./docker/pyrun.sh bin/domain_adequacy.py spectra \
    $(ls -1 runs/g16_adq1200/window/*.[0-9]* | sort -t. -k2 -n | tail -3) \
    2>&1 | tee $R/adq1200_spectra.txt

for T in 490 1200; do
  LPDM_WORKERS="${LPDM_WORKERS:-10}" ./docker/pyrun.sh bin/stage5_footprint.py \
      runs/g16_adq$T/window --dt $DT_WIN --tback $TB --rel-seconds 1800 --z-target 10.0 \
      --sgs-most --receptor-from data/grid16_cbl --cover-dir data/grid16_cbl \
      --fp16-cache --cover-groups 10 --tag adq$T 2>&1 \
      | grep -vE 'batch [0-9]+/' > $R/adq$T.txt
  [ -s "$R/adq$T.json" ] || { tail -12 $R/adq$T.txt >&2; die "adq$T produced no footprint"; }
done

./docker/pyrun.sh bin/domain_adequacy.py compare $R/adq490.json $R/adq1200.json \
    2>&1 | tee $R/adq_compare_1200.txt

say "PASS 7 COMPLETE"
