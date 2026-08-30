#!/usr/bin/env python3
"""THE CONTAINMENT GATE: does the footprint fit in the box, or is the cap hiding it?

WHAT IS BEING ASKED, and why the existing monitor gate cannot ask it. `corpus_monitor`'s
G2a checks `|I(2L)/I(1L) - 1| <= 1e-3` and every case passes it trivially -- because
production retires a trajectory at one domain length, so the by-displacement curve is FLAT
past 1 L BY CONSTRUCTION. G2a tests that the cap BINDS. It cannot test containment, and
reading it as containment is the mistake its own tolerance was written to prevent.

The question needs the cap RAISED (`stage5_footprint.py --max-disp`), and then it separates
two things that look identical at the cap:

  KEEPS CLIMBING past 1 L        the domain repeats, so a trajectory that travels further
                                 re-enters turbulence it already sampled. Those touchdowns
                                 are the same eddies counted twice -- WRAP DOUBLE-COUNTING,
                                 an artifact, and the cap is what removes it. The size of
                                 the climb past 1 L is the size of what the cap is hiding.
  SATURATES, possibly HIGH       influence has genuinely run out inside one domain length.
                                 If it saturates ABOVE the 1 - z_m/z_i ceiling that is not
                                 truncation -- truncation can only lose influence -- it is
                                 the advective non-closure (`w_bar` times the concentration
                                 integral at a receptor over sloping ground) or, in a CBL,
                                 an elevated concentration maximum. Both are physics and
                                 both stay.

THREE THINGS ARE SCORED, and the third is the one that matters for the grid decision:

  C1  the integral SATURATES before the cap        |I(1L) - I(0.75L)| / I(1L) small
  C2  the 80% source area fits                     x80 < L, with margin
  C3  what the cap is HIDING                       I(3L)/I(1L) - 1, only measurable when
                                                   the run was made with the cap raised

usage: containment_gate.py results/corpus/<tag>.json [more.json ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SATURATE_TOL = 0.02      # 2% of the integral over the last quarter-domain. Two raster
# cells of peak is the project's other near-field tolerance; this is the tail's equivalent
# and is set at the level below which the remaining tail cannot move an 80% source area.
X80_MARGIN = 0.80        # x80 must sit inside 80% of a domain length


def score(d):
    rows = []
    by = d.get("by_disp") or []
    L = None
    cur = {}
    for r in by:
        cur[round(float(r["frac_of_Lx"]), 3)] = float(r["integral"])
        L = float(r["max_disp"]) / float(r["frac_of_Lx"])
    I75, I100 = cur.get(0.75), cur.get(1.0)
    # ---- C1: has the curve flattened by the time it reaches one domain length? ---------
    if I75 is not None and I100:
        d75 = abs(I100 - I75) / abs(I100)
        rows.append(("C1 integral saturates by 1 L", bool(d75 <= SATURATE_TOL),
                     f"I(0.75L) {I75:.3f} -> I(1L) {I100:.3f}, {100*d75:+.1f}% over the "
                     f"last quarter-domain (<= {100*SATURATE_TOL:.0f}%)"))
    else:
        rows.append(("C1 integral saturates by 1 L", None, "no by_disp curve"))
    # ---- C2: does the 80% source area fit? ---------------------------------------------
    x80 = (d.get("les") or {}).get("x80")
    if x80 is None:
        x80 = (d.get("les") or {}).get("x80_far")
    if x80 is not None and L:
        rows.append(("C2 x80 inside the box", bool(x80 <= X80_MARGIN * L),
                     f"x80 {x80:.0f} m against {X80_MARGIN:.0%} of L = "
                     f"{X80_MARGIN*L:.0f} m (L = {L:.0f} m)"))
    else:
        rows.append(("C2 x80 inside the box", None, "no x80 in this record"))
    # ---- C3: what is the cap hiding? ----------------------------------------------------
    beyond = {k: v for k, v in cur.items() if k > 1.0}
    cap = d.get("max_disp_used")
    at_cap = (cap is None) or (L and abs(cap - L) < 0.01 * L)
    if beyond and I100:
        kmax = max(beyond)
        hid = beyond[kmax] / I100 - 1.0
        # A RUN MADE AT THE CAP REPORTS ~ZERO HERE AND IT IS NOT EVIDENCE. Detect it from
        # the recorded cap rather than from the number being small, because "small" is
        # exactly what a contained footprint also looks like -- the two are
        # indistinguishable in the value and completely distinguishable in the metadata.
        if at_cap:
            rows.append(("C3 what the cap hides", None,
                         f"I({kmax:g}L)/I(1L) - 1 = {hid:+.2e}, but this run was made AT "
                         f"the cap (max_disp = {cap if cap else L:.0f} m = 1 L), so the "
                         f"tail is flat BY CONSTRUCTION and the number is not evidence. "
                         f"NO VERDICT: re-run with --max-disp {3*L:.0f}."))
        else:
            rows.append(("C3 what the cap hides", bool(abs(hid) <= SATURATE_TOL),
                         f"I({kmax:g}L)/I(1L) - 1 = {hid:+.1%} -- the influence the cap "
                         f"removes. Large and positive = wrap double-counting; near zero = "
                         f"the cap is not doing anything and the footprint is contained."))
    else:
        rows.append(("C3 what the cap hides", None, "the ladder does not extend past 1 L"))
    # ---- reported: where the integral sits relative to the physical ceiling -------------
    I, A = d.get("integral_les"), d.get("integral_asymptote")
    if I is not None and A:
        rows.append(("   integral vs 1 - z_m/z_i", None,
                     f"{I:.3f} / {A:.4f} = {I/A:.3f}. Truncation can only push this BELOW "
                     f"1; above 1 is advective non-closure or an elevated maximum."))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    fails = 0
    for p in a.json:
        d = json.load(open(p))
        P(f"\n=== containment: {os.path.basename(p)} ===")
        st = d.get("stats") or {}
        P(f"  z_m {d.get('zm_agl', float('nan')):.1f} m, z_i {st.get('h', float('nan')):.0f} m, "
          f"u* {st.get('ustar', float('nan')):.3f}, L {st.get('L', float('nan')):+.1f} m, "
          f"W at the receptor {d.get('w_bar', float('nan')):+.4f} m/s")
        for name, ok, why in score(d):
            tag = "??  " if ok is None else ("ok  " if ok else "FAIL")
            fails += (ok is False)
            P(f"  [{tag}] {name:<32} {why}")
    P("")
    P(f"  {fails} failure(s)")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write("\n".join(lines) + "\n")
        print(f"  wrote {a.out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
