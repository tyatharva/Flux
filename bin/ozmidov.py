#!/usr/bin/env python3
"""Can this grid resolve the turbulence in a stratified layer? The Ozmidov scale, measured.

WHY. A stable seed at GABLS1's own regime ran healthy for 1.75 simulated hours at
dx = 16 m and then collapsed. Every standing check passed while it died -- finiteness, a
clean CORRUPTED grep, and k0/k1 at 0.442. The question this answers is not "did it
collapse" (that is obvious afterwards) but "was the grid ever able to carry it", which is
answerable at the HEALTHY dump and therefore BEFORE spending the GPU time.

THE OZMIDOV SCALE is the largest eddy that stratification permits to overturn:

    L_O = sqrt(eps / N^3)

Above L_O, buoyancy wins and motion is wave-like rather than turbulent. An LES can only
represent turbulence in the band Delta < l < L_O, so L_O/Delta ~ 1 means there is no band
at all: the model is running a sub-grid closure and calling it a boundary layer. There is
no equivalent constraint in a convective or neutral layer, where the energy-containing
scale is set by z_i or by the wall and grows away from the surface.

eps AND l ARE FASTEDDY'S OWN, imported from lpdm/fields.py rather than re-chosen here:

    l   = min(0.76 sqrt(e)/N, Delta)   for N^2 > 0,   Delta otherwise
    eps = C_E e^(3/2) / l,             C_E = 0.93

which is the same closure the LPDM is driven by. Picking a different dissipation constant
for the diagnostic than the transport model runs on would make the two incomparable, and
this project has already been bitten once by a gate carrying its own copy of a closure.

REPORTED ALONGSIDE: the resolved fraction of sigma_w^2, f = ww_res/(ww_res + (2/3)e_sgs).
That number is NOT a gate -- PROJECT_BRIEF.md retired the 40% sub-grid gate because no affordable
grid clears it at a 10 m receptor -- but it is what makes L_O/Delta concrete: a layer with
L_O ~ Delta has almost nothing resolved, and the two numbers say the same thing twice.

usage: ozmidov.py <dump.nc> [--dx 16.0] [--top 400] [--receptor 10.0]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.fields import C_E                                        # noqa: E402

G = 9.81
BAND_MIN = 10.0     # L_O/Delta at the receptor below which there is no resolved band


def _zpos(path):
    """Level heights, falling back to a sibling dump -- lean output omits zPos."""
    with Dataset(path) as ds:
        if "zPos" in ds.variables:
            return np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))[:, 0, 0]
    fam = os.path.basename(path).rsplit(".", 1)[0]   # the anchor's OWN run, not the dir's
    sibs = sorted((q for q in glob.glob(os.path.join(os.path.dirname(path) or ".",
                                                     "*.[0-9]*"))
                   if os.path.basename(q).rsplit(".", 1)[0] == fam),
                  key=lambda q: int(q.rsplit(".", 1)[1]))
    for q in sibs:
        with Dataset(q) as ds:
            if "zPos" in ds.variables:
                return np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))[:, 0, 0]
    raise KeyError("no zPos in this dump or any sibling")


def profile(path, dx):
    with Dataset(path) as ds:
        g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
        th, w = g("theta"), g("w")
        e = np.maximum(g("TKE_0"), 0.0)
        ust = float(g("fricVel").mean()) if "fricVel" in ds.variables else float("nan")
    z = _zpos(path)
    thb = th.mean(axis=(-2, -1))
    esgs = e.mean(axis=(-2, -1))
    wp = w - w.mean(axis=(-2, -1), keepdims=True)
    ww = (wp ** 2).mean(axis=(-2, -1))

    dz = np.gradient(z)
    delta = np.cbrt(dx * dx * dz)
    dthdz = np.gradient(thb, z)
    n2 = G / np.maximum(thb, 1.0) * dthdz
    nn = np.sqrt(np.maximum(n2, 0.0))
    # FastEddy's own mixing length and dissipation (lpdm/fields.py).
    with np.errstate(divide="ignore", invalid="ignore"):
        ell = np.where(n2 > 0.0,
                       np.minimum(0.76 * np.sqrt(np.maximum(esgs, 0.0)) / np.maximum(nn, 1e-12),
                                  delta), delta)
        eps = C_E * np.maximum(esgs, 0.0) ** 1.5 / np.maximum(ell, 1e-2)
        lo = np.where(n2 > 0.0, np.sqrt(eps / np.maximum(nn, 1e-12) ** 3), np.inf)
    f_res = ww / np.maximum(ww + (2.0 / 3.0) * esgs, 1e-30)
    return dict(z=z, delta=delta, n2=n2, eps=eps, lo=lo, f_res=f_res, ww=ww,
                esgs=esgs, ustar=ust, theta=thb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--dx", type=float, default=16.0)
    ap.add_argument("--top", type=float, default=400.0)
    ap.add_argument("--receptor", type=float, default=10.0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--sl-top", type=float, default=50.0,
                    help="top of the band the verdict is scored on. THE COLUMN MEDIAN IS "
                         "NOT THE ANSWER: L_O/Delta rises steeply with height as N falls, "
                         "so a column median reads ~9 on a layer whose surface region is "
                         "at 2.4. The receptor is at 10 m and the footprint is made in the "
                         "lowest few tens of metres; that is what has to be resolved.")
    a = ap.parse_args()

    if not os.path.exists(a.dump):
        print(f"FATAL: {a.dump} does not exist", file=sys.stderr)
        return 2
    P = profile(a.dump, a.dx)
    z, lo, delta = P["z"], P["lo"], P["delta"]
    sel = z <= a.top

    print(f"=== Ozmidov / resolution, {a.label or os.path.basename(a.dump)} ===")
    print(f"  dx = {a.dx:.1f} m, u* = {P['ustar']:.4f} m/s")
    print(f"  {'z':>8}{'Delta':>8}{'N':>10}{'eps':>11}{'L_O':>10}{'L_O/D':>8}"
          f"{'f_res':>8}{'ww':>11}")
    for k in np.where(sel)[0]:
        n = np.sqrt(max(P["n2"][k], 0.0))
        lod = lo[k] / delta[k]
        print(f"  {z[k]:8.1f}{delta[k]:8.2f}{n:10.5f}{P['eps'][k]:11.3e}"
              f"{lo[k]:10.2f}{lod:8.2f}{P['f_res'][k]:8.3f}{P['ww'][k]:11.3e}")

    # The summary the decision rests on: the STRATIFIED part of the column only. Where
    # N^2 <= 0 there is no Ozmidov constraint at all and averaging those levels in would
    # flatter the answer.
    strat = sel & (P["n2"] > 0) & (P["esgs"] > 1e-6)
    surf = (z <= a.sl_top) & (P["n2"] > 0) & (P["esgs"] > 1e-6)
    kr = int(np.argmin(np.abs(z - a.receptor)))
    print()
    if strat.sum():
        r = lo[strat] / delta[strat]
        print(f"  L_O/Delta, whole stratified column to {a.top:.0f} m (n={int(strat.sum())}):"
              f"  min {r.min():.2f}  median {np.median(r):.2f}  max {r.max():.2f}")
        print(f"    ^ REPORTED, NOT SCORED. L_O/Delta rises steeply with height because N")
        print(f"      falls, so this median flatters a layer whose surface region is starved.")
    if surf.sum():
        rs = lo[surf] / delta[surf]
        lr = lo[kr] / delta[kr]
        print(f"  L_O/Delta, SURFACE LAYER (z <= {a.sl_top:.0f} m, n={int(surf.sum())}):"
              f"  min {rs.min():.2f}  median {np.median(rs):.2f}  max {rs.max():.2f}"
              f"   |  at the {z[kr]:.0f} m receptor: {lr:.2f}")
        # SCORED AT THE RECEPTOR, not on the band median. The receptor is where the
        # footprint is made and it is the single least ambiguous point in the column.
        # BAND_MIN = 10 is the conventional requirement for an inertial subrange to exist
        # at all -- about one decade between the grid scale below and the buoyancy limit
        # above. Nothing here is near the threshold in either direction: the measured
        # contrast is 3.6 (stable) against 318 (neutral), a factor of 89, so the verdict
        # does not turn on where in 5-20 the line is drawn.
        print(f"  VERDICT: "
              + (f"NO RESOLVED BAND at the receptor (L_O/Delta {lr:.2f} < {BAND_MIN}) -- "
                 f"the model is running a sub-grid closure where the footprint is made"
                 if lr < BAND_MIN else
                 f"resolved band at the receptor: L_O/Delta {lr:.2f} >= {BAND_MIN}"))
    elif np.all(P["n2"][z <= a.sl_top] <= 0):
        print(f"  SURFACE LAYER IS UNSTRATIFIED (N^2 <= 0 below {a.sl_top:.0f} m): there is")
        print(f"  NO Ozmidov constraint at all. Buoyancy produces the largest eddies here")
        print(f"  rather than limiting them, which is why convective cases are the easy")
        print(f"  case for this grid and stable ones are the hard case.")
    elif not strat.sum():
        print("  no stratified, turbulent levels below the top -- nothing to score")
    print(f"  resolved fraction of sigma_w^2: {P['f_res'][kr]:.3f} at z = {z[kr]:.1f} m"
          f"  (receptor);  {P['f_res'][max(kr-1,0)]:.3f} at {z[max(kr-1,0)]:.1f} m,"
          f"  {P['f_res'][min(kr+1,len(z)-1)]:.3f} at {z[min(kr+1,len(z)-1)]:.1f} m")
    print(f"  GABLS1 (Beare et al. 2006) runs this regime at dx = 6.25 m: "
          f"{a.dx/6.25:.1f}x finer, {(a.dx/6.25)**3:.0f}x the cells for this domain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
