#!/usr/bin/env bash
# The per-target analyses that need no GPU, so they run while the next LES has the card.
#
#   (a) CPU-from-disk vs from-ring, per footprint, against the run's own half-vs-half floor
#   (b) Gate D1 well-mixed, both directions, on this window
#   (c) the negative-lobe share on both paths (handoff_accept.py reports it)
#   (d) fp16 parity is inherent to (a): the disk path is CF-packed to 16 bit and the ring
#       is raw fp32, so (a) IS the parity measurement -- there is no separate one to make
#   plus the sub-grid fraction of sigma_w^2 at the receptor, and the by-displacement
#   containment ladder on the kept fields.
#
# WHAT "THROUGH THE RING" MEANS FOR D1, said plainly rather than implied. The well-mixed
# battery needs the window fields, and the ring consumes them -- so D1 runs on the netCDF
# dumps that lpdmOnlineSelector = 2 wrote from the IDENTICAL buffers the ring read. That the
# two carry the same bytes is established separately and exactly: bin/test_lpdmonline.py
# scores producer against consumer at max |diff| 0.000e+00, and this run's own driver
# asserts the staged and written counts match. It is not a claim that D1 was computed
# inside the ring; nothing is.
#
# usage: pass9_accept.sh <tag> <dt> <regime>
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
TAG="$1"; DT="$2"; REGIME="${3:-}"
O=results/pass9; mkdir -p "$O"
D="runs/$TAG"; FP=results/corpus
CG="data/case_grids/$TAG"
[ -d "$CG" ] || CG="data/grid30_raised"
say(){ echo; echo "===== $* ====="; date '+%F %H:%M:%S'; }

NDUMP=$(ls -1 "$D"/window/FE_WIN.[0-9]* 2>/dev/null | wc -l)
if [ "$NDUMP" -eq 0 ]; then
  echo "pass9_accept $TAG: no window dumps kept, so the disk path cannot be recomputed."
  echo "  KEEP_FIELDS=1 is what preserves them; without it (a), (b) and the ladder are"
  echo "  not available and this is NOT a pass." | tee "$O/${TAG}_accept.txt"
  exit 0
fi
echo "pass9_accept $TAG: $NDUMP kept dumps" | tee "$O/${TAG}_accept.txt"

# window 0's bounds, recovered from the pair rather than recomputed, so this cannot drift
# from what the run actually did.
read -r T0 T1 < <(python3 -c "
import json,glob
st=[int(p.rsplit('.',1)[1]) for p in glob.glob('$D/window/FE_WIN.[0-9]*')]
st.sort()
import sys
d=json.load(open('$FP/${TAG}_w0.json'))
n=len(d.get('step_per_dump') or [])
s=d.get('step_per_dump') or st
print(f'{min(s)*$DT:.3f}', f'{max(s)*$DT:.3f}')")
echo "  window 0 spans ${T0}-${T1} s" | tee -a "$O/${TAG}_accept.txt"

# ---- (a) the same window, recomputed from the dumps -------------------------------------
say "$TAG (a): CPU-from-disk, the same window the ring already did"
LPDM_WORKERS="${LPDM_WORKERS:-12}" \
./docker/pyrun.sh bin/stage5_footprint.py "$D/window" --dt "$DT" --tback 900 \
    --sgs-most --cover-dir "$CG" --receptor-from "$CG" --fp16-cache \
    --z-target 28.5 --exact-agl --rel-seconds 1800 --strict-rel --cover-groups 10 \
    --keep-touchdowns 100000 --t-min "$T0" --t-max "$T1" \
    --outdir "$FP" --tag "${TAG}_w0_disk" 2>&1 | grep -vE 'batch [0-9]+/' \
    > "$FP/${TAG}_w0_disk.txt"
if [ -s "$FP/${TAG}_w0_disk.json" ]; then
  ./docker/pyrun.sh bin/handoff_accept.py --ring "$FP/${TAG}_w0.json" \
      --disk "$FP/${TAG}_w0_disk.json" --json "$O/${TAG}_handoff.json" \
      2>&1 | tee -a "$O/${TAG}_accept.txt"
else
  echo "  *** the disk recompute produced no json; see $FP/${TAG}_w0_disk.txt" \
    | tee -a "$O/${TAG}_accept.txt"
  tail -12 "$FP/${TAG}_w0_disk.txt" >> "$O/${TAG}_accept.txt"
fi

# ---- (b) Gate D1, well-mixed, BOTH directions -------------------------------------------
# Non-negotiable, and it must run in the PRODUCTION closure configuration: --sgs-most with
# the displacement map, because a gate that scores a different closure than the footprints
# use is not a gate. Backward is what footprints use; forward is the control that localises
# a sign error in the reverse-time drift.
say "$TAG (b): Gate D1 well-mixed, production closure, both directions"
# --dmap, NOT --receptor-from: stage4_wellmixed.py takes the displacement MAP directly.
# Passing the wrong flag made argparse refuse and the gate never ran -- and because the
# whole thing is piped into tail|tee, the refusal scrolled past as usage text rather than
# stopping anything. docs/FASTEDDY_TRAPS.md 12, one more time.
./docker/pyrun.sh bin/stage4_wellmixed.py "$D/window" --dt "$DT" --sgs-most \
    --z-target 28.5 ${DMAP:+--dmap "$DMAP"} 2>&1 | tail -40 | tee -a "$O/${TAG}_accept.txt"

# ---- the containment ladder, uncapped to 3 L --------------------------------------------
# production retires a trajectory at ONE domain length, so the by-displacement curve is flat
# past 1 L BY CONSTRUCTION and corpus_monitor's G2a passes trivially. Raising the cap is
# what separates influence that has run out from influence still accumulating because the
# trajectory re-entered turbulence it already sampled.
say "$TAG: the containment ladder, cap raised to 3 L"
LPDM_WORKERS="${LPDM_WORKERS:-12}" \
./docker/pyrun.sh bin/stage5_footprint.py "$D/window" --dt "$DT" --tback 900 \
    --sgs-most --cover-dir "$CG" --receptor-from "$CG" --fp16-cache \
    --z-target 28.5 --exact-agl --rel-seconds 1800 --cover-groups 10 \
    --max-disp 10980 --t-min "$T0" --t-max "$T1" \
    --outdir "$FP" --tag "${TAG}_w0_3L" 2>&1 | grep -vE 'batch [0-9]+/' \
    > "$FP/${TAG}_w0_3L.txt"
if [ -s "$FP/${TAG}_w0_3L.json" ]; then
  ./docker/pyrun.sh bin/containment_gate.py "$FP/${TAG}_w0_3L.json" \
      --out "$O/${TAG}_containment.json" 2>&1 | tee -a "$O/${TAG}_accept.txt"
fi

# ---- the no-op closure control ----------------------------------------------------------
# QUOTE THE NO-OP CONTROL ALONGSIDE THE RESULT. A peak can move because the closure's own
# stability function moved rather than because the LES resolved different turbulence, and
# the only way to tell is to recompute the identical window with the floor OFF.
say "$TAG: the same window with the sigma_w floor OFF (the no-op control)"
LPDM_WORKERS="${LPDM_WORKERS:-12}" \
./docker/pyrun.sh bin/stage5_footprint.py "$D/window" --dt "$DT" --tback 900 \
    --cover-dir "$CG" --receptor-from "$CG" --fp16-cache \
    --z-target 28.5 --exact-agl --rel-seconds 1800 --cover-groups 10 \
    --t-min "$T0" --t-max "$T1" \
    --outdir "$FP" --tag "${TAG}_w0_nofloor" 2>&1 | grep -vE 'batch [0-9]+/' \
    > "$FP/${TAG}_w0_nofloor.txt"
python3 - "$FP/${TAG}_w0.json" "$FP/${TAG}_w0_nofloor.json" <<'PYNF' 2>/dev/null | tee -a "$O/${TAG}_accept.txt"
import json, sys, os
a = json.load(open(sys.argv[1]))
if not os.path.exists(sys.argv[2]):
    print("  the no-op control produced no json"); raise SystemExit(0)
b = json.load(open(sys.argv[2]))
fa, fb = a["les"], b["les"]
print(f"\n  floor ON vs OFF, identical window:")
for k in ("peak_x", "x80", "area80_ha", "centroid_dist"):
    print(f"    {k:<16}{fa.get(k, float('nan')):10.1f}{fb.get(k, float('nan')):10.1f}")
sa = (a.get('cover_share') or {}).get('solar array')
sb = (b.get('cover_share') or {}).get('solar array')
if sa is not None and sb is not None:
    print(f"    {'array share %':<16}{sa*100:10.2f}{sb*100:10.2f}"
          f"   ({(sa-sb)*100:+.2f} points)")
print(f"    {'integral':<16}{a['integral_les']:10.3f}{b['integral_les']:10.3f}")
PYNF

echo; echo "pass9_accept $TAG done -> $O/${TAG}_accept.txt"
