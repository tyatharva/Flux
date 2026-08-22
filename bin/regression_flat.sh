#!/usr/bin/env bash
# THE STANDING CONTROL: flat, uniform, neutral. Re-run at every configuration change.
#
# This is the only case where Kljun is diagnostic rather than descriptive -- FFP is derived
# for a horizontally homogeneous surface layer, so over flat uniform ground in neutral
# conditions a disagreement is a bug in us, not a result about the site. It is also the
# canary for silent geometric bugs: a receptor placed in the wrong column, a raster whose
# indexing has drifted from the surface masks, a rotation applied twice. Over real terrain
# all three of those produce a plausible-looking footprint. Here they do not: the answer
# has to be a symmetric plume with a known peak distance and a unit integral.
#
# The tolerances are NOT chosen. They come from the run's own half-versus-half difference,
# which is the irreducible sampling floor of a 30-minute window (PLAN.md Stage 5 Gate 2).
# Anything inside that floor is a different turbulence realisation; anything outside it is
# a change in the pipeline.
#
# usage: regression_flat.sh [--baseline]     (--baseline rewrites the reference)
set -uo pipefail
cd /home/atyagi/Flux
# t_back is generous on purpose the FIRST time this runs at a new grid: the capture curve
# it produces is what SIZES t_back, and a window too short to contain the answer cannot
# report that it was too short. Drop it to the measured value afterwards.
TBACK="${TBACK:-600}"
WIN=$(python3 -c "print(1800+$TBACK)")
DT="${DT:-0.0162686}"
SRC="${SRC:-runs/g16_flat/output/FE_ADJ.0}"
D="${D:-runs/g16_flat}"
GRID="${GRID:-data/grid16}"
TAG="${TAG:-g16_flat}"
MARKS="${MARKS:-60,100,150,200,250,300,400,500}"

if [ "${1:-}" != "--analysis-only" ]; then
  BASE="${BASE:-runs/g16_base/base.in}" bin/run_window.sh $D $SRC $DT $WIN - 10.000000 0.000000 || exit 1
fi
./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt $DT --tback "$TBACK" \
    --z-target 10.0 --tback-marks "$MARKS" \
    --sgs-most --receptor-from "$GRID" --cover-dir "$GRID" --fp16-cache --tag $TAG 2>&1 \
    | grep -vE 'batch [0-9]+/' | tee results/$TAG.txt
[ "${KEEP_FIELDS:-0}" = "1" ] || rm -f $D/window/*

python3 - "$@" <<'PY'
import json, os, sys
new = json.load(open(os.environ.get("TAGJSON", "results/g16_flat.json")))
ref_p = os.environ.get("REFJSON", "results/regression_baseline_g16.json")
if "--baseline" in sys.argv or not os.path.exists(ref_p):
    json.dump(new, open(ref_p, "w"), indent=2)
    print("\n  BASELINE WRITTEN -> " + ref_p); raise SystemExit(0)
ref = json.load(open(ref_p))
# Floor: this run's own half-vs-half difference, with a small absolute guard so a
# freakishly quiet window cannot make the test unpassable.
fl = new.get("halves", {})
tol = dict(peak_x=max(2.5 * abs(fl.get("dpeak", 0.0)), 48.0),          # 2 cells
           area80_ha=max(0.35 * ref["les"]["area80_ha"], 5.0),
           integral_les=0.08,
           overlap_kljun=0.15)
rows, bad = [], 0
for k, path in (("peak_x", ("les", "peak_x")), ("area80_ha", ("les", "area80_ha")),
                ("integral_les", ("integral_les",)), ("overlap_kljun", ("overlap_kljun",))):
    a = ref; b = new
    for p in path: a = a[p]; b = b[p]
    d = b - a; ok = abs(d) <= tol[k]
    bad += (not ok)
    rows.append(f"  {k:<15} baseline {a:9.3f}   now {b:9.3f}   diff {d:+9.3f}"
                f"   tol {tol[k]:.3f}   {'ok' if ok else 'CHANGED'}")
print("\n=== FLAT / NEUTRAL REGRESSION ===")
print(f"  tolerance from this window's own half-vs-half floor: peak "
      f"{fl.get('dpeak', float('nan')):+.0f} m, centroid {fl.get('dcentroid', float('nan')):.0f} m, "
      f"80% overlap {100*fl.get('overlap', float('nan')):.0f}%")
print("\n".join(rows))
print("  RESULT: " + ("PASS -- inside the sampling floor" if not bad
                      else f"CHANGED in {bad} metric(s) -- explain it before proceeding"))
raise SystemExit(1 if bad else 0)
PY
