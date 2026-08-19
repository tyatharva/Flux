#!/usr/bin/env python3
"""Stage 5 Gate 1 (revised): what fraction of sigma_w^2 at the receptor is SUB-GRID?

The old gate -- "reproduce Kljun" -- was the wrong instrument. It measured resolution
through a proxy that also folds in the sub-grid closure, the surface-layer treatment and
FFP's own calibration, so a failure did not say what to change. The sub-grid fraction says
it directly: it is a property of the LES alone, computable before a single particle is
released, and it is what determines whether the near-field footprint is resolved physics or
a Langevin model's output.

  resolved(z) = variance of the LES w field over the horizontal plane
  sub-grid(z) = (2/3) e_sgs, FastEddy's own SGS TKE, isotropically partitioned

GATE: sub-grid fraction at the receptor < 40%.

Also reports the fraction against z / Delta, where Delta = (dx dy dz)^(1/3) is the filter
width, because that is the variable it actually collapses onto -- and therefore the one
that says what grid would pass.
"""
import argparse
import glob
import os
import sys

import numpy as np
from netCDF4 import Dataset

LIMIT = 0.40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--ztarget", type=float, default=30.0)
    ap.add_argument("--stride", type=int, default=20)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.outdir, "*.[0-9]*")),
                   key=lambda p: int(p.split(".")[-1]))[::a.stride]
    ww = ee = None
    ust = 0.0
    for p in paths:
        with Dataset(p) as d:
            g = lambda v: np.squeeze(np.asarray(d[v][:], dtype=np.float64))
            w = g("w"); e = np.maximum(g("TKE_0"), 0.0)
            z = g("zPos")[:, 0, 0]; zg = np.squeeze(np.asarray(d["topoPos"][:]))
            ust += float(g("fricVel").mean())
        v = ((w - w.mean(axis=(-2, -1), keepdims=True)) ** 2).mean(axis=(-2, -1))
        ww = v if ww is None else ww + v
        em = e.mean(axis=(-2, -1))
        ee = em if ee is None else ee + em
    n = len(paths); ww /= n; ee /= n; ust /= n
    sg = (2.0 / 3.0) * ee
    frac = sg / np.maximum(ww + sg, 1e-30)
    dz = np.gradient(z)
    with Dataset(paths[0]) as d:
        xp = np.squeeze(np.asarray(d["xPos"][:], dtype=np.float64))
        dx = float(xp[0, 0, 1] - xp[0, 0, 0])
    delta = (dx * dx * dz) ** (1.0 / 3.0)
    zagl = z - float(np.mean(zg))
    k = int(np.argmin(np.abs(zagl - a.ztarget)))

    print(f"  {len(paths)} dumps from {a.outdir};  dx = {dx:.1f} m,  u* = {ust:.4f} m/s")
    print(f"  {'k':>3} {'z AGL':>8} {'dz':>6} {'Delta':>7} {'z/Delta':>8} "
          f"{'resolved':>10} {'sub-grid':>10} {'sg frac':>8}")
    for kk in list(range(0, min(16, len(z)))) + [20, 26, 34, 44]:
        if kk >= len(z):
            break
        mark = "  <-- RECEPTOR" if kk == k else ""
        print(f"  {kk:3d} {zagl[kk]:8.2f} {dz[kk]:6.2f} {delta[kk]:7.2f} "
              f"{zagl[kk]/delta[kk]:8.2f} {ww[kk]:10.5f} {sg[kk]:10.5f} "
              f"{frac[kk]*100:7.1f}%{mark}")
    f = frac[k]
    print(f"\n  RECEPTOR k={k}, z = {zagl[k]:.3f} m AGL, Delta = {delta[k]:.2f} m, "
          f"z/Delta = {zagl[k]/delta[k]:.2f}")
    print(f"  sub-grid fraction of sigma_w^2 = {f*100:.1f}%   "
          f"GATE (< {LIMIT*100:.0f}%): {'PASS' if f < LIMIT else 'FAIL'}")
    # What Delta would pass? Interpolate the measured fraction against z/Delta.
    ok = np.isfinite(frac) & (zagl > 0)
    r = zagl[ok] / delta[ok]
    order = np.argsort(r)
    tgt = np.interp(LIMIT, frac[ok][order][::-1], r[order][::-1])
    print(f"  the fraction collapses onto z/Delta; it crosses {LIMIT*100:.0f}% at "
          f"z/Delta ~ {tgt:.2f}")
    print(f"  => a 30 m receptor needs Delta <~ {a.ztarget/max(tgt,1e-9):.1f} m, i.e. "
          f"isotropic spacing of about that, NOT merely a finer dz.")
    return 0 if f < LIMIT else 1


if __name__ == "__main__":
    sys.exit(main())
