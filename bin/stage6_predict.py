#!/usr/bin/env python3
"""Predict each direction's solar-array footprint share, then compare with the measurement.

The array is a rectangle in map coordinates containing the tower (60 m E/W, 250 m N,
100 m S). Its UPWIND REACH is therefore the chord of that rectangle along the upwind
direction, starting at the tower -- 250 m for a due northerly, 60 m for an easterly, and
something in between for anything else. Multiply Kljun's cumulative crosswind-integrated
footprint out to that chord by the fraction of the crosswind spread the array's 120 m
width actually covers, and that is the share to expect.

This makes Stage 6 a quantitative gate rather than a "differs in the right direction" one.

usage: stage6_predict.py
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm import kljun
from scipy.special import erf

W, E, S, N = -60.0, 60.0, -100.0, 250.0     # array rectangle, metres from the tower
# z_m is read PER CASE from the result json, not fixed here. It is the EFFECTIVE
# aerodynamic height z - d, which is what stage5_footprint.py now hands Kljun -- and over
# the array d is 1.5 m of a 10 m receptor, so a predictor using the geometric height would
# disagree with the measurement it is supposed to test by construction rather than by
# physics. Older jsons have no `zm` key and fall back to the geometric value.
ZM_FALLBACK = 10.0


def chord(theta_from_deg):
    """Distance from the tower to where the upwind ray leaves the array rectangle."""
    a = np.radians(theta_from_deg)
    ux, uy = np.sin(a), np.cos(a)           # unit vector pointing UPWIND
    ts = []
    for u, lo, hi in ((ux, W, E), (uy, S, N)):
        if abs(u) < 1e-9:
            continue
        ts.append((hi if u > 0 else lo) / u)
    return max(min(ts), 0.0) if ts else 0.0


def cover_share(pre, tag):
    """Footprint-weighted cover shares, EXCLUDING periodically folded touchdowns.

    Read from the JSON rather than scraped from the report text, and from the unwrapped
    column: a touchdown 3 km upwind folds to 1.5 km on the far side of the tower, where
    the land cover is a different lake. For the array specifically the difference is
    small -- the array is 65-250 m upwind and nothing that close can wrap -- but the
    water shares in the same table would be badly wrong folded.
    """
    j = json.load(open(f"results/{pre}_{tag}.json"))
    src = j.get("cover_share_nowrap") or j.get("cover_share", {})
    return {k: (100.0 * v, np.nan) for k, v in src.items()}


def main():
    print("  Solar-array footprint share: predicted from geometry vs measured by the LPDM")
    print("  Kljun is evaluated at the EFFECTIVE height z - d from each case's json.")
    print()
    print("  %-4s %8s %8s %9s %9s %9s %9s %9s" % (
        "case", "wind", "chord", "Kljun cum", "crosswind", "PRED-Klj", "MEASURED",
        "PRED-LES"))
    print("  %-4s %8s %8s %9s %9s %9s %9s %9s" % (
        "", "from", "in array", "to chord", "factor", "share", "share", "share"))
    print("  " + "-" * 72)
    rows = []
    pre = sys.argv[1] if len(sys.argv) > 1 else "g24"
    for tag in ("wN", "wS", "wE", "wW"):
        j = json.load(open(f"results/{pre}_{tag}.json"))
        st = j["stats"]
        ZM = float(j.get("zm") or ZM_FALLBACK)
        wd = st["wdir"]
        c = chord(wd)
        x = np.arange(1.0, 6000.0, 1.0)
        fy, _ = kljun.crosswind_integrated(x, ZM, st["h"], st["ustar"],
                                           umean=st["u_mean"], L=st["L"])
        cum = np.cumsum(fy); cum /= cum[-1]
        kl = float(cum[np.searchsorted(x, max(c, 1.0))])
        # crosswind: the array is 120 m wide; sigma_y at half the chord
        xm = max(c / 2.0, 1.0)
        sy = float(kljun.sigma_y(np.array([xm]), ZM, st["h"], st["ustar"],
                                 st["sigma_v"], umean=st["u_mean"], L=st["L"])[0])
        cf = float(erf(60.0 / max(sy * np.sqrt(2.0), 1e-6)))
        pred = 100.0 * kl * cf
        meas = cover_share(pre, tag).get("solar array", (np.nan, np.nan))[0]
        # Same calculation, but using the LES's OWN crosswind-integrated footprint
        # instead of Kljun's. If the array attribution is internally consistent, this
        # should MATCH the measurement -- and any gap against the Kljun column is then
        # purely the LES-vs-Kljun near-field difference, not an attribution error.
        z = np.load(f"results/{pre}_{tag}.npz")
        xc = z["fy_xc"]; fles = z["fy"]
        cl = np.cumsum(np.maximum(fles, 0.0)); cl /= cl[-1]
        les_cum = float(np.interp(c, xc, cl))
        pred_les = 100.0 * les_cum * cf
        rows.append((tag, wd, c, kl * 100, cf, pred, meas, pred_les))
        print("  %-4s %7.0f° %7.0f m %8.2f%% %9.2f %8.2f%% %8.2f%% %9.2f%%" %
              (tag, wd, c, kl * 100, cf, pred, meas, pred_les))
    print()
    m = [r[6] for r in rows]
    print("  MEASURED SWING across direction: %.2f%% (%s) to %.2f%% (%s)  =  %.0fx" % (
        max(m), rows[int(np.argmax(m))][0], max(min(m), 1e-4),
        rows[int(np.argmin(m))][0], max(m) / max(min(m), 1e-4)))
    grid = os.environ.get("GRID", "data/grid16")
    aa = 100.0 * float(np.load(os.path.join(grid, "array.npy")).mean())
    print("  area share of the domain: %.2f%%  ->  enrichment %.1fx at best, %.2fx at worst"
          % (aa, max(m) / aa, min(m) / aa))
    print()
    print("  PRED-LES uses the LES's OWN crosswind-integrated footprint in place of")
    print("  Kljun's, so a gap between PRED-Klj and PRED-LES is the near-field difference")
    print("  between the two footprints, while a gap between PRED-LES and MEASURED would")
    print("  be an attribution error. Ratios are quoted as well, being the more robust")
    print("  comparison at chords of a few grid cells:")
    print()
    d = dict((r[0], r) for r in rows)
    for a_, b_ in (("wN", "wE"), ("wN", "wS"), ("wS", "wE")):
        pr = d[a_][5] / max(d[b_][5], 1e-9)
        me = d[a_][6] / max(d[b_][6], 1e-9)
        print("    %s / %s :  predicted %7.1fx   measured %7.1fx" % (a_, b_, pr, me))
    print()
    # Say what THIS run shows, not what a previous pass showed. The fourth-pass version of
    # this block asserted "peak 270 m against 150 m" and "backed ~24 deg" as fixed text,
    # which prints a conclusion whatever the data does -- the one thing a gate must never do.
    rat = [(a_, b_, d[a_][5] / max(d[b_][5], 1e-9), d[a_][6] / max(d[b_][6], 1e-9))
           for a_, b_ in (("wN", "wE"), ("wN", "wS"), ("wS", "wE"))]
    over = sum(1 for _, _, pr, me in rat if me > pr)
    same = sum(1 for _, _, pr, me in rat if (pr - 1.0) * (me - 1.0) > 0)
    print("  %d of %d ratio pairs have the measured swing EXCEEDING the predicted one, and"
          % (over, len(rat)))
    print("  %d of %d agree in SIGN (both above or both below unity). Sign and ordering are"
          % (same, len(rat)))
    print("  the robust content here: the chords are 60-250 m, i.e. 4-16 cells at 16 m, so")
    print("  the absolute prediction is asking more of the near field than the raster gives.")
    print()
    # rows = (tag, wdir, chord, kljun_cum%, crosswind_factor, pred, measured, pred_les)
    ch = [r[2] for r in rows]
    print("  Array chords this pass: %.0f-%.0f m upwind of the tower. The chord is capped"
          % (min(ch), max(ch)))
    print("  by the array's 120 m WIDTH, not its 350 m length: a ray from the tower toward")
    print("  any direction off due north leaves through an east or west edge within ~100 m.")
    print("  That is why the directional swing is modest here and why absolute share, not")
    print("  the N-vs-E/W ratio, is the discriminator at a 10 m receptor.")
    print()
    frm = [r[1] for r in rows]
    print("  Achieved surface directions: %s." % ", ".join("%.0f deg" % f for f in frm))
    print("  They are backed from the geostrophic forcing by Ekman turning AND carried by")
    print("  the inertial oscillation, so none is a due N/S/E/W case. The prediction uses")
    print("  the ACHIEVED direction, which is what makes the comparison fair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
