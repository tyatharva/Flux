#!/usr/bin/env python3
"""Is `lpdm/kljun_ffp.py` faithful to the official FFP, and how far from it was ours?

Two separate questions, and they are scored differently on purpose.

**A. The adapter against the code it wraps -- ASSERTED.** The adapter re-evaluates the
official's own two separable factors at our cell coordinates instead of resampling its
raster. That reconstruction is exact algebra, so scored at the official's OWN grid points
it must reproduce the official's own `f_2d` to floating-point roundoff. A tolerance here
is not a judgement call: there is no physics between the two and the right answer is
machine epsilon. This is the gate.

**B. Our reimplementation against the official -- MEASURED AND PRINTED, never asserted.**
`lpdm/kljun.py` stays in the tree for the gates already validated against it, so what is
useful is the size of the gap in each regime, on the record, rather than a red tick. One
divergence is known and is the reason this file exists: at `|L| > 5000` the official's
`scale_const` clips to 1.0 where ours returns 0.8, so our neutral `sigma_y` is 1.25x wide.
If that number moves, something else has changed.

usage: docker/pyrun.sh bin/test_kljun_adapter.py [--json results/kljun_adapter.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpdm import kljun as ours                      # noqa: E402
from lpdm import kljun_ffp as ffp                   # noqa: E402

ZM = 30.0          # the production receptor
Z0 = 0.0615        # the 122^3 @ 30 m box's geometric-mean z0

# CORPUS-REPRESENTATIVE, not a grid of round numbers. The two production targets are in
# here by their own measured scalars, and the rest span the z_i band [300, 1250] and the
# L range the corpus actually reaches -- including the exactly-neutral flat control, which
# is where the known divergence lives and which a convective-only sweep would miss.
CASES = [
    dict(tag="convective target (case_2023052519)",
         h=1229.5, L=-26.22, ustar=0.4614, sigmav=1.5538, umean=5.1432),
    dict(tag="near-neutral target (case_2023121921)",
         h=937.5, L=-737.55, ustar=0.5695, sigmav=0.8813, umean=8.4844),
    dict(tag="strongly convective, shallow", h=450.0, L=-15.0, ustar=0.30,
         sigmav=1.20, umean=3.2),
    dict(tag="convective, deep", h=1250.0, L=-60.0, ustar=0.45, sigmav=1.30, umean=5.0),
    dict(tag="weakly unstable", h=600.0, L=-400.0, ustar=0.40, sigmav=0.90, umean=6.0),
    dict(tag="L = -4000 (inside the official's oln)", h=500.0, L=-4000.0, ustar=0.42,
         sigmav=0.80, umean=7.0),
    dict(tag="L = -50000 (OUTSIDE oln -- the divergence)", h=500.0, L=-50000.0,
         ustar=0.42, sigmav=0.80, umean=7.0),
    dict(tag="L = +inf, the flat/neutral control", h=400.0, L=float("inf"), ustar=0.35,
         sigmav=0.75, umean=6.5),
    dict(tag="stable, +200 (corpus excludes these; scored anyway)", h=300.0, L=200.0,
         ustar=0.25, sigmav=0.50, umean=5.5),
    dict(tag="z0 branch rather than umean", h=800.0, L=-100.0, ustar=0.40, sigmav=1.0,
         z0=Z0),
]


def part_a(c):
    """The adapter's reconstruction against the official's own f_2d. ASSERTED."""
    kw = {k: c[k] for k in ("h", "L", "ustar", "sigmav") if k in c}
    kw.update({k: c[k] for k in ("umean", "z0") if k in c})
    prof = ffp.ffp_profile(ZM, nx=ffp.NX_DEFAULT, **kw)

    mod = ffp._ffp()
    out = mod.FFP(zm=ZM, z0=c.get("z0"), umean=c.get("umean"), h=c["h"],
                  ol=float(c["L"]), sigmav=c["sigmav"], ustar=c["ustar"],
                  wind_dir=None, rs=None, rslayer=1, nx=ffp.NX_DEFAULT,
                  crop=False, fig=False)
    f_2d = np.asarray(out["f_2d"], dtype=np.float64)
    x_ax = np.asarray(out["x_2d"], dtype=np.float64)[:, 0]
    y_ax = np.asarray(out["y_2d"], dtype=np.float64)[0]

    # Score on the official's own nodes: X from its x axis, Y from its y axis. Sub-sample
    # the y axis so the comparison is not dominated by the far tail, where both are ~0.
    jy = np.unique(np.linspace(0, len(y_ax) - 1, 401).astype(int))
    XX, YY = np.meshgrid(x_ax, y_ax[jy], indexing="ij")
    f0, sy = ffp._interp_factors(prof, XX.ravel())
    rec = (f0 * np.exp(-YY.ravel() ** 2 / (2.0 * sy ** 2))).reshape(XX.shape)
    ref = f_2d[:, jy]

    scale = float(np.nanmax(np.abs(ref)))
    absdiff = float(np.nanmax(np.abs(rec - ref)))
    m = np.isfinite(ref) & (np.abs(ref) > scale * 1e-12)
    reldiff = float(np.nanmax(np.abs(rec[m] - ref[m]) / np.abs(ref[m]))) if m.any() else 0.0
    return dict(peak=scale, max_abs=absdiff, max_abs_over_peak=absdiff / max(scale, 1e-300),
                max_rel_where_significant=reldiff,
                x_peak_official=float(out["x_ci_max"]), x_peak_adapter=prof["x_peak"])


def part_b(c):
    """Ours against the official, on the same inputs. MEASURED, not asserted."""
    kw = {k: c[k] for k in ("umean", "z0") if k in c}
    L = float(c["L"])
    prof = ffp.ffp_profile(ZM, c["h"], L, c["ustar"], c["sigmav"], **kw)

    # crosswind-integrated profile, on the official's own x axis
    x = prof["x"]
    um, zz = c.get("umean"), c.get("z0")
    fy_ours, _ = ours.crosswind_integrated(x, ZM, c["h"], c["ustar"], umean=um,
                                           z0=zz, L=L)
    sy_ours = ours.sigma_y(x, ZM, c["h"], c["ustar"], c["sigmav"], umean=um, z0=zz, L=L)
    pk_ours = ours.peak_distance(ZM, c["h"], c["ustar"], umean=um, z0=zz, L=L)

    fci = prof["f_ci"]
    m = fci > fci.max() * 1e-6          # where the footprint actually is
    r_fy = float(np.max(np.abs(fy_ours[m] - fci[m]) / fci[m])) if m.any() else 0.0
    r_sy = float(np.max(np.abs(sy_ours[m] - prof["sigy"][m]) / prof["sigy"][m])) \
        if m.any() else 0.0
    return dict(x_peak_ffp=prof["x_peak"], x_peak_ours=float(pk_ours),
                x_peak_ratio=float(pk_ours) / prof["x_peak"],
                max_rel_f_ci=r_fy, max_rel_sigma_y=r_sy,
                sigma_y_ratio_ours_over_ffp=float(
                    np.median(sy_ours[m] / prof["sigy"][m])) if m.any() else float("nan"))


def part_c(nsub=8):
    """The static-raster adapter against the 1-D profile, on the production grid.

    THE REFERENCE IS THE CELL AVERAGE, NOT THE CELL-CENTRE VALUE, and getting that wrong is
    a real trap rather than pedantry: `f_ci` climbs through three orders of magnitude across
    ONE 30 m cell in the near field, so a centre sample sits ~80% below the average of the
    cell it labels. The raster is a cell average by construction (that is what the nsub x
    nsub subdivision is FOR), so the only meaningful comparison is against `f_ci` averaged
    over the same sub-cell positions.

    What this then actually tests is the geometry: that the north-up rotation puts the
    footprint upwind, that the crosswind Gaussian quadrature is adequate at this cell size,
    and that nothing is lost off the sides of the box beyond the domain truncation.
    """
    n, dx = 122, 30.0
    e = (np.arange(n + 1) - n / 2.0) * dx        # edges, receptor at the origin
    xc = 0.5 * (e[1:] + e[:-1])
    c = dict(h=1229.5, L=-26.22, ustar=0.4614, sigmav=1.5538, umean=5.1432)
    prof = ffp.ffp_profile(ZM, c["h"], c["L"], c["ustar"], c["sigmav"], umean=c["umean"])
    # wind blowing due east (ang = 0) puts the footprint due west, along -x
    f = ffp.footprint_on_static(e, e, 0.0, ZM, c["h"], c["ustar"], c["sigmav"],
                                umean=c["umean"], L=c["L"], prof=prof)
    fy = f.sum(axis=0) * dx                       # crosswind-integrate the raster

    off = (np.arange(nsub) + 0.5) / nsub - 0.5
    Xsub = -(xc[:, None] + dx * off[None, :])     # upwind distance at each x sub-position
    ref = np.interp(Xsub, prof["x"], prof["f_ci"], left=0.0, right=0.0).mean(axis=1)

    m = ref > ref.max() * 1e-3
    ctr = np.interp(-xc, prof["x"], prof["f_ci"], left=0.0, right=0.0)
    return dict(integral=float(f.sum() * dx * dx),
                max_rel_f_ci_on_raster=float(np.max(np.abs(fy[m] - ref[m]) / ref[m])),
                max_rel_if_scored_against_cell_centres=float(
                    np.max(np.abs(fy[m] - ctr[m]) / ctr[m])),
                peak_cell_x=float(xc[int(np.argmax(fy))]),
                peak_cell_x_cellavg_reference=float(xc[int(np.argmax(ref))]),
                x_peak_profile=-prof["x_peak"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="Part A only. The reconstruction is exact algebra on the "
                         "official's own arrays, so this is a roundoff bound and not a "
                         "physics tolerance.")
    a = ap.parse_args()

    print(f"official FFP: {ffp.FFP_PATH}")
    print(f"  nx = {ffp.NX_DEFAULT},  z_m = {ZM} m,  z0 = {Z0} m\n")

    rows, worst_a = [], 0.0
    print("=== A. the adapter against the official code it wraps (ASSERTED) ===")
    print(f"{'case':<48} {'max|d|/peak':>12} {'max rel':>10} {'x_peak':>9}")
    for c in CASES:
        r = part_a(c)
        worst_a = max(worst_a, r["max_abs_over_peak"])
        print(f"{c['tag']:<48} {r['max_abs_over_peak']:12.2e} "
              f"{r['max_rel_where_significant']:10.2e} {r['x_peak_adapter']:9.1f}")
        rows.append({"case": c["tag"], "A": r})

    print("\n=== B. lpdm/kljun.py against the official (MEASURED, not asserted) ===")
    print(f"{'case':<48} {'sigy ours/ffp':>14} {'max rel f_ci':>13} {'x_peak ratio':>13}")
    for c, row in zip(CASES, rows):
        r = part_b(c)
        row["B"] = r
        print(f"{c['tag']:<48} {r['sigma_y_ratio_ours_over_ffp']:14.4f} "
              f"{r['max_rel_f_ci']:13.2e} {r['x_peak_ratio']:13.6f}")

    print("\n=== C. the static-raster drop-in on the production 122^2 @ 30 m grid ===")
    rc = part_c()
    print(f"  raster integral                        {rc['integral']:.6f}")
    print(f"  max rel error vs CELL-AVERAGED f_ci    "
          f"{rc['max_rel_f_ci_on_raster']:.3e}   <- the real check")
    print(f"  (same, scored against cell CENTRES)    "
          f"{rc['max_rel_if_scored_against_cell_centres']:.3e}   <- the trap")
    print(f"  raster peak cell x = {rc['peak_cell_x']:.0f} m; cell-averaged reference "
          f"peaks at {rc['peak_cell_x_cellavg_reference']:.0f} m; the continuous profile "
          f"peaks at {rc['x_peak_profile']:.0f} m (one cell is 30 m)")

    ok = worst_a <= a.tol
    print(f"\nPart A worst max|diff|/peak = {worst_a:.3e} against {a.tol:.0e}: "
          f"{'PASS' if ok else 'FAIL'}")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"zm": ZM, "z0": Z0, "nx": ffp.NX_DEFAULT, "ffp_path": ffp.FFP_PATH,
                   "tol": a.tol, "worst_A": worst_a, "pass": bool(ok),
                   "raster": rc, "cases": rows}, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
