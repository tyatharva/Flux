#!/usr/bin/env python3
"""Footprint geometry on the REAL map, at the real receptor height. Phase A gate.

PROJECT_BRIEF.md's array-share and water-share tables are Kljun evaluated against idealised
upwind distances -- straight lines from the tower at 60 / 100 / 250 m. That was the right
way to size a domain before the surface existed. It is not a measurement, and two of the
decisions resting on it (the domain length, and whether the lake is still in the science)
are expensive to get wrong.

This evaluates the SAME model on the SAME cells the LES and the LPDM use, with the actual
WorldCover array and water masks, folded exactly the way lpdm.driver folds touchdowns. The
numbers it produces are directly comparable to what stage5_footprint.py will report from
the LES, which is what makes the comparison a test rather than a coincidence.

It is run at TWO receptor heights: the geometric 10 m, and the effective z - d. Kljun's
z_m is an aerodynamic height, and at a 10 m receptor over a 2-3 m array the difference is
not a refinement -- it moves the array's crosswind share by tens of percent relative.

GATE: water share below ~10% in every direction, or the domain is too small and the lake
has to be brought back in.

usage: phaseA_geometry.py [--grid data/grid16] [--out results/phaseA_geometry.txt]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm import kljun

K = 0.4
OCT = [("N", 0.0), ("NE", 45.0), ("E", 90.0), ("SE", 135.0),
       ("S", 180.0), ("SW", 225.0), ("W", 270.0), ("NW", 315.0)]
# Stability classes, as z_m/L at the RECEPTOR, with the z_i and u* the CONUS404 sample
# gives for each. Chosen to bracket the site, not to be exhaustive.
STAB = [("very unstable", -1.00, 900.0, 0.30),
        ("unstable",      -0.20, 700.0, 0.35),
        ("neutral",        0.00, 500.0, 0.35),
        ("stable",         0.20, 250.0, 0.25),
        ("very stable",    1.00, 120.0, 0.20)]


def psi_m(zm, L):
    return kljun.psi_m(zm, L)


def sigma_v_of(ustar, zi, L):
    """Panofsky et al. (1977) unstable; the neutral surface-layer value otherwise.

    sigma_v/u* = (12 + 0.5 z_i/|L|)^(1/3) under free convection -- the crosswind
    fluctuation scales with the BOUNDARY-LAYER depth, not with z, which is why sigma_y and
    hence the footprint width care about z_i even where the streamwise shape barely does.
    """
    if L is None or not np.isfinite(L) or abs(L) > 1e5 or L > 0:
        return 1.9 * ustar
    return ustar * (12.0 + 0.5 * zi / abs(L)) ** (1.0 / 3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="data/grid16")
    ap.add_argument("--out", default="results/phaseA_geometry.txt")
    ap.add_argument("--zm", type=float, default=10.0, help="geometric receptor height")
    ap.add_argument("--nsub", type=int, default=8)
    a = ap.parse_args()

    g = lambda n: np.load(os.path.join(a.grid, n + ".npy"))
    meta = np.load(os.path.join(a.grid, "meta.npy"), allow_pickle=True).item()
    z0m, water, array, dmap = g("z0m"), g("water") > 0.5, g("array") > 0.5, g("dmap")
    lc = g("lcclass")
    nx, ny, dx = int(meta["nx"]), int(meta["ny"]), float(meta["dx"])
    it, jt, pad = int(meta["itower"]), int(meta["jtower"]), int(meta["pad"])
    d_tower = float(dmap[jt, it])
    # Geometric-mean roughness: the scalar a horizontally-homogeneous model has to be
    # given, and the same one runs/g16_base uses for surflayer_z0.
    z0g = float(np.exp(np.log(np.maximum(z0m, 1e-6)).mean()))

    # The identical raster lpdm.driver builds: cell edges relative to the tower column.
    xe = (np.arange(nx + 1) - 0.5 - it) * dx
    ye = (np.arange(ny + 1) - 0.5 - jt) * dx
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    R = np.hypot(*np.meshgrid(xc, yc))
    r_real = (min(it, nx - it, jt, ny - jt) - pad) * dx     # real terrain radius
    cell = dx * dx

    out = []
    p = out.append
    p(f"Kljun FFP on the REAL {nx} x {ny} @ {dx:.0f} m map ({a.grid})")
    p(f"  tower cell ({it},{jt});  geometric-mean z0 = {z0g:.4f} m;  "
      f"d at the tower = {d_tower:.3f} m")
    p(f"  land cover: " + ", ".join(
        f"{nm} {100*(lc==k).mean():.1f}%" for nm, k in
        (("crop", 40), ("tree", 10), ("grass", 30), ("built", 50), ("water", 80))) +
      f", array {100*array.mean():.2f}%")
    p(f"  real terrain reaches {r_real:.0f} m from the tower "
      f"({pad}-cell taper); land cover is NOT tapered and is real to the seam")
    p("")

    gate_water = 0.0
    keep, peaks = {}, {}
    for zm_lab, zm in ((f"z_m = {a.zm:.1f} m (geometric)", a.zm),
                       (f"z_m = {a.zm - d_tower:.1f} m (effective, z - d)", a.zm - d_tower)):
        p(f"=== {zm_lab} ===")
        p(f"  {'stability':<14}{'dir':>5}{'x_peak':>8}{'x90':>7}"
          f"{'ARRAY':>9}{'water':>8}{'>real':>8}{'>930m':>8}")
        for sname, zoL, zi, ustar in STAB:
            L = (zm / zoL) if zoL != 0.0 else np.inf
            sv = sigma_v_of(ustar, zi, L)
            um = (ustar / K) * (np.log(zm / z0g) - psi_m(zm, L))
            xpk = kljun.peak_distance(zm, zi, ustar, umean=um, z0=z0g, L=L)
            # x90 from the crosswind-integrated form, on a fine 1-D axis
            xx = np.linspace(0.5, 4000.0, 8000)
            fyy, _ = kljun.crosswind_integrated(xx, zm, zi, ustar, umean=um, z0=z0g, L=L)
            cum = np.cumsum(fyy); cum /= cum[-1]
            x90 = float(np.interp(0.90, cum, xx))
            for dname, bearing in OCT:
                b = np.radians(bearing)
                ang = np.arctan2(-np.cos(b), -np.sin(b))   # the wind BLOWS toward here
                f = kljun.footprint_on_static(xe, ye, ang, zm, zi, ustar, sv,
                                              umean=um, z0=z0g, L=L, nsub=a.nsub)
                w = np.maximum(f, 0.0) * cell
                tot = w.sum()
                if tot <= 0:
                    continue
                sh = lambda m: 100.0 * w[m].sum() / tot
                fa, fw = sh(array), sh(water)
                fr, f930 = sh(R > r_real), sh(R > 930.0)
                gate_water = max(gate_water, fw)
                keep[(round(zm, 3), sname, dname)] = fa
                peaks[(round(zm, 3), sname)] = xpk
                p(f"  {sname:<14}{dname:>5}{xpk:8.0f}{x90:7.0f}"
                  f"{fa:8.2f}%{fw:7.2f}%{fr:7.2f}%{f930:7.2f}%")
        p("")

    p("=== the directional discriminator is weaker than the idealised table said ===")
    p("PROJECT_BRIEF.md's array shares are CROSSWIND-INTEGRATED fractions inside the array's")
    p("upwind reach along a line from the tower. The tower is INSIDE a 2-D rectangle, so")
    p("flux arriving from crosswind angles still lands on the array and the real 2-D share")
    p("is larger -- and the N-vs-E/W RATIO, which is what Gate F leans on, is smaller.")
    p(f"  {'stability':<15}{'N':>8}{'E':>8}{'N/E':>8}   (z_m = %.1f m, real map)" % a.zm)
    zA, zB = round(a.zm, 3), round(a.zm - d_tower, 3)
    for sname, _, _, _ in STAB:
        n_, e_ = keep.get((zA, sname, "N")), keep.get((zA, sname, "E"))
        if n_ and e_:
            p(f"  {sname:<15}{n_:7.1f}%{e_:7.1f}%{n_/e_:8.2f}x")
    p("  PROJECT_BRIEF.md quotes ~3.7x for neutral from the idealised table; the real map gives")
    p("  the number above. Gate F should lean on ABSOLUTE share by direction, not the ratio.")
    p("")
    p("=== how much displacement height is worth ===")
    p(f"  d at the tower = {d_tower:.2f} m, so z_m goes {a.zm:.1f} -> {a.zm-d_tower:.1f} m.")
    p(f"  {'stability':<15}{'dir':>4}{'z_m=%.1f' % a.zm:>9}{'z_m=%.1f' % (a.zm-d_tower):>9}"
      f"{'delta':>8}{'rel':>7}")
    for sname, _, _, _ in STAB:
        for dn in ("N", "E", "S", "W"):
            lo, hi = keep.get((zA, sname, dn)), keep.get((zB, sname, dn))
            if lo and hi:
                p(f"  {sname:<15}{dn:>4}{lo:8.2f}%{hi:8.2f}%{hi-lo:+8.2f}{hi/lo:7.2f}x")
    p("  This is far above any sampling floor, so the array's share is partly a MODELLING")
    p("  CHOICE until d is handled. It is handled on the LPDM and Kljun side here; whether")
    p("  it also belongs in the LES terrain is the --raise-topo sensitivity in Phase F.")
    p("")
    p("=== near-field resolution at dx = %.0f m ===" % dx)
    p("  The peak sits a few cells from the tower, which is what a 10 m receptor means.")
    p(f"  {'stability':<15}{'x_peak':>9}{'cells':>8}")
    for sname, _, _, _ in STAB:
        if (zA, sname) in peaks:
            p(f"  {sname:<15}{peaks[(zA, sname)]:8.0f}m{peaks[(zA, sname)]/dx:8.1f}")
    p("  Recorded, not gated: the grid is set by corpus economics and the near field is")
    p("  closure-dominated at z/Delta ~ 1 regardless. It does bound how sharply the CNF")
    p("  target can represent the peak.")
    p("")
    p("=== GATE A1: water share ===")
    p(f"  worst-case water share over every direction and stability: {gate_water:.2f}%")
    ok = gate_water < 10.0
    verdict = "PASS" if ok else "FAIL -- the domain is too small; bring the lake back in"
    p(f"  threshold 10%.  {verdict}")
    txt = "\n".join(out)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
        print(f"\n  wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
