#!/usr/bin/env bash
# The flat/neutral control's own analyses. CPU only, so they run while the next LES is on
# the card.
#
#   the CONTAINMENT acceptance this grid was chosen for -- the neutral integral must
#     SATURATE by 2.5 L. Full containment is explicitly NOT the bar: at 2928 m the LES
#     retained 0.874 of its asymptote against Kljun's 0.867 on identical cells, so both
#     models lose the same tail and a RELATIVE claim survives the truncation.
#   Gate D1 well-mixed, NEUTRAL, both directions, in the production closure.
#   the sub-grid fraction of sigma_w^2 at the receptor.
#
# usage: pass9_flat.sh <dt>
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
DT="$1"; O=results/pass9; mkdir -p "$O"
D=runs/g30_flat; G=data/grid30
say(){ echo; echo "===== $* ====="; date '+%F %H:%M:%S'; }
OUT="$O/flat_control.txt"; : > "$OUT"

N=$(ls -1 "$D"/window/FE_WIN.[0-9]* 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then
  echo "flat control: no dumps kept; the ladder and D1 are unavailable and this is NOT a pass." \
    | tee "$OUT"; exit 0
fi
echo "flat/neutral control: $N kept dumps" | tee -a "$OUT"

# ---- CONTAINMENT: the cap raised to 3 L, which is the only way to ask the question ------
# production retires a trajectory at ONE domain length, so the by-displacement curve is FLAT
# past 1 L by construction and corpus_monitor's G2a passes trivially -- it tests that the
# cap BINDS, not that the footprint is contained.
say "flat control: the by-displacement ladder, cap raised to 3 L = 10980 m"
LPDM_WORKERS="${LPDM_WORKERS:-12}" \
./docker/pyrun.sh bin/stage5_footprint.py "$D/window" --dt "$DT" --tback 900 \
    --sgs-most --cover-dir "$G" --receptor-from "$G" --fp16-cache \
    --z-target 28.5 --rel-seconds 1800 --cover-groups 10 --max-disp 10980 \
    --outdir results --tag g30_flat_3L 2>&1 | grep -vE 'batch [0-9]+/' \
    > results/g30_flat_3L.txt
if [ -s results/g30_flat_3L.json ]; then
  ./docker/pyrun.sh bin/containment_gate.py results/g30_flat_3L.json \
      --out "$O/flat_containment.json" 2>&1 | tee -a "$OUT"
else
  echo "  *** the 3 L ladder produced no json; see results/g30_flat_3L.txt" | tee -a "$OUT"
  tail -12 results/g30_flat_3L.txt >> "$OUT"
fi

# ---- Gate D1, NEUTRAL, both directions, production closure ------------------------------
# A NEUTRAL PASS IS NOT EVIDENCE ABOUT THE CONVECTIVE CLOSURE and never has been -- the
# sigma_w floor is nearly inert neutrally, and the neutral gate once passed a closure
# carrying NINE turnovers. This is the neutral half only; pass9_accept.sh runs the
# convective half on the convective target.
say "flat control: Gate D1 well-mixed, NEUTRAL, both directions"
./docker/pyrun.sh bin/stage4_wellmixed.py "$D/window" --dt "$DT" --sgs-most \
    --z-target 28.5 2>&1 | tail -40 | tee -a "$OUT"

# ---- the no-op control ------------------------------------------------------------------
say "flat control: the same window with the floor OFF"
LPDM_WORKERS="${LPDM_WORKERS:-12}" \
./docker/pyrun.sh bin/stage5_footprint.py "$D/window" --dt "$DT" --tback 900 \
    --cover-dir "$G" --receptor-from "$G" --fp16-cache \
    --z-target 28.5 --rel-seconds 1800 --cover-groups 10 \
    --outdir results --tag g30_flat_nofloor 2>&1 | grep -vE 'batch [0-9]+/' \
    > results/g30_flat_nofloor.txt
python3 - results/g30_flat.json results/g30_flat_nofloor.json <<'PYNF' 2>/dev/null | tee -a "$OUT"
import json, os, sys
a = json.load(open(sys.argv[1]))
if not os.path.exists(sys.argv[2]):
    print("  the no-op control produced no json"); raise SystemExit(0)
b = json.load(open(sys.argv[2]))
print("\n  floor ON vs OFF on the flat/neutral control (the floor is nearly INERT here,")
print("  which is exactly why a neutral D1 pass says nothing about the convective closure):")
for k in ("peak_x", "x80", "area80_ha", "centroid_dist"):
    print(f"    {k:<16}{a['les'].get(k, float('nan')):10.1f}{b['les'].get(k, float('nan')):10.1f}")
print(f"    {'integral':<16}{a['integral_les']:10.3f}{b['integral_les']:10.3f}")
PYNF
echo; echo "flat control done -> $OUT"
