#!/usr/bin/env python3
"""Re-score the LES-vs-Kljun asymptote parity with the OFFICIAL FFP, on identical cells.

WHY THIS EXISTS. The decision to accept a truncated domain rests on ONE number: that Kljun,
evaluated on the same box and the same cells, loses the same fraction of its own asymptote
as the LES does. At 122^3 @ 24 m that was LES 0.874 against Kljun 0.867 -- parity to 0.7
points -- and at 122^3 @ 30 m it is LES 0.765 against Kljun 0.923, which is not parity at
all. The 24 m number was computed with `lpdm/kljun.py`, the reimplementation now known to
be **1.2500x wide in sigma_y whenever |L| > 5000** -- and a flat/neutral control is exactly
that regime. So the 24 m half of the comparison was made with a Kljun the 30 m half does
not use, and the asymmetry could be an artifact of the two halves disagreeing about Kljun
rather than a statement about either LES.

This settles it CPU-only, on footprints already on disk: it re-evaluates the Kljun channel
with `third_party/FFP` at the target raster's own cell edges -- structurally the same cells
the LES footprint was binned onto, not merely the same nominal grid -- and re-integrates.

  usage: bin/kljun_parity.py results/g24_flatnbl.json results/g30_flat.json
         bin/kljun_parity.py <json> [...] --out results/kljun_parity.json

Each <json> is a stage-5 footprint record; the matching stage-5 .npz beside it supplies
`xe`/`ye`. Cells are asserted, not assumed: the edges' midpoints must BE the centres.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm import kljun_ffp  # noqa: E402


def parity(jpath):
    d = json.load(open(jpath))
    npz = os.path.splitext(jpath)[0] + ".npz"
    if not os.path.exists(npz):
        raise SystemExit(f"{jpath}: no {npz} beside it, so there are no cell edges to "
                         f"evaluate Kljun on. This is not something to approximate.")
    with np.load(npz) as z:
        for nm in ("xc", "yc", "xe", "ye", "les", "kljun"):
            if nm not in z:
                raise SystemExit(f"{npz} carries no {nm}")
        xc, yc = np.asarray(z["xc"], float), np.asarray(z["yc"], float)
        xe, ye = np.asarray(z["xe"], float), np.asarray(z["ye"], float)
        les = np.asarray(z["les"], float)
        kl_stored = np.asarray(z["kljun"], float)
    # SAME CELLS, ASSERTED. Without this the whole comparison is a word.
    for nm, e, c in (("x", xe, xc), ("y", ye, yc)):
        if len(e) != len(c) + 1:
            raise SystemExit(f"{npz}: {nm} edges do not bound its centres")
        mid = 0.5 * (e[1:] + e[:-1])
        if not np.allclose(mid, c, rtol=0, atol=1e-9):
            raise SystemExit(f"{npz}: the {nm} edge midpoints are not the {nm} centres "
                             f"(max |diff| {np.max(np.abs(mid - c)):.3e} m)")
    da = abs(xc[1] - xc[0]) * abs(yc[1] - yc[0])

    st = d["stats"]
    zm_eff = float(d["zm"])
    ang = np.radians(float(d["wind_angle"]))
    prof = kljun_ffp.ffp_profile(zm_eff, float(st["h"]), float(st["L"]),
                                 float(st["ustar"]), float(st["sigma_v"]),
                                 umean=float(st["u_mean"]))
    kl_new = kljun_ffp.footprint_on_static(
        xe, ye, ang, zm_eff, float(st["h"]), float(st["ustar"]), float(st["sigma_v"]),
        umean=float(st["u_mean"]), L=float(st["L"]), prof=prof)

    asym = d.get("integral_asymptote")
    if asym is None:
        asym = 1.0 - zm_eff / float(st["h"])
    I_les = float(d["integral_les"])
    I_kl_stored = float(kl_stored.sum() * da)
    I_kl_new = float(kl_new.sum() * da)
    den = max(float(np.abs(kl_stored).max()), 1e-300)
    return {
        "file": jpath,
        "kljun_source_recorded": d.get("kljun_source", "lpdm/kljun.py (reimplementation)"),
        "dx_m": float(abs(xc[1] - xc[0])), "n": int(les.shape[-1]),
        "domain_m": float(len(xc) * abs(xc[1] - xc[0])),
        "zm_m": zm_eff, "h_m": float(st["h"]), "L_m": float(st["L"]),
        "ustar": float(st["ustar"]), "sigma_v": float(st["sigma_v"]),
        "asymptote": float(asym),
        "integral_les": I_les,
        "integral_kljun_stored": I_kl_stored,
        "integral_kljun_official": I_kl_new,
        "les_over_asymptote": I_les / asym,
        "kljun_stored_over_asymptote": I_kl_stored / asym,
        "kljun_official_over_asymptote": I_kl_new / asym,
        "parity_gap_stored": I_les / asym - I_kl_stored / asym,
        "parity_gap_official": I_les / asym - I_kl_new / asym,
        "kljun_raster_max_reldiff": float(np.abs(kl_new - kl_stored).max()) / den,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [parity(p) for p in a.json]
    print("=== LES vs Kljun, as a fraction of the 1 - z_m/z_i asymptote, identical cells ===")
    hdr = (f"{'case':<26}{'dx':>4}{'box m':>8}{'LES/A':>9}{'Kl/A old':>10}"
           f"{'Kl/A FFP':>10}{'gap old':>10}{'gap FFP':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{os.path.basename(r['file'])[:25]:<26}{r['dx_m']:>4.0f}{r['domain_m']:>8.0f}"
              f"{r['les_over_asymptote']:>9.4f}{r['kljun_stored_over_asymptote']:>10.4f}"
              f"{r['kljun_official_over_asymptote']:>10.4f}"
              f"{r['parity_gap_stored']:>+10.4f}{r['parity_gap_official']:>+10.4f}")
    print()
    for r in rows:
        print(f"  {os.path.basename(r['file'])}: L = {r['L_m']:.1f} m, h = {r['h_m']:.0f} m, "
              f"z_m = {r['zm_m']:.1f} m; the official FFP moves the Kljun raster by "
              f"{r['kljun_raster_max_reldiff']:.2e} of its peak and its integral by "
              f"{r['integral_kljun_official'] - r['integral_kljun_stored']:+.4f}")
        print(f"    recorded Kljun source: {r['kljun_source_recorded']}")
    if a.out:
        json.dump({"rows": rows}, open(a.out, "w"), indent=1)
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
