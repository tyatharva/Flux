#!/usr/bin/env python3
"""Is the convective boundary layer a convective boundary layer?

The Stage 2 gate compares against NCAR's NEUTRAL validation case, which says nothing about
a CBL. A dry convective boundary layer has its own well-established similarity structure,
and it is sharp enough to be a real test:

  z_i        height of the MINIMUM buoyancy flux (the entrainment minimum) -- the
             standard definition, and independent of the TKE threshold used elsewhere
  w*         (g/theta * w'theta'_s * z_i)^(1/3), the convective velocity scale
  entrainment ratio   -w'theta'(z_i) / w'theta'(0)  ~  0.2   (Deardorff 1972; Moeng &
             Sullivan 1994). Too small means the inversion is not yet being eroded;
             too large means the capping inversion is too weak to hold the CBL down.
  sigma_w/w* 1.34 (z/z_i)^(1/3) (1 - 0.8 z/z_i)   (Lenschow et al. 1980), peaking at
             ~0.6 w* near z/z_i ~ 0.35
  buoyancy flux profile: linear from w'theta'_s at the surface to its minimum at z_i

Variances here are RESOLVED PLUS SUB-GRID, for the same reason les_stats.py gives: at the
lowest levels most of the variance is sub-grid, and comparing the resolved part alone
against a similarity relation would flag a correct simulation as broken.

usage: cbl_check.py <dump.nc> [<dump.nc> ...]
"""
import os
import sys

import numpy as np
from netCDF4 import Dataset

G = 9.81


def prof(path):
    with Dataset(path) as ds:
        g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
        z = g("zPos")[:, 0, 0]
        u, v, w, th = g("u"), g("v"), g("w"), g("theta")
        e = np.maximum(g("TKE_0"), 0.0)
        ust = float(g("fricVel").mean())
        hf = float(g("htFlux").mean()) if "htFlux" in ds.variables else np.nan
    p = lambda a: a - a.mean(axis=(-2, -1), keepdims=True)
    wt = (p(w) * p(th)).mean(axis=(-2, -1))
    ww = (p(w) ** 2).mean(axis=(-2, -1))
    tke = 0.5 * ((p(u) ** 2) + (p(v) ** 2) + (p(w) ** 2)).mean(axis=(-2, -1))
    return dict(z=z, wt=wt, ww=ww, e=e.mean(axis=(-2, -1)), tke=tke,
                th=th.mean(axis=(-2, -1)), ust=ust, hf=hf,
                U=np.hypot(u.mean(axis=(-2, -1)), v.mean(axis=(-2, -1))))


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: cbl_check.py <dump.nc> ..."); return 1
    ps = [prof(p) for p in paths]
    dt = float(os.environ.get("FE_DT", "0.0328947"))
    steps = np.array([int(p.split(".")[-1]) for p in paths], dtype=float)
    tt = steps * dt

    print(f"  {len(ps)} dumps, t = {tt[0]/60:.0f} .. {tt[-1]/60:.0f} min of simulated time")
    print(f"\n  {'t (min)':>8} {'z_i (m)':>8} {'w* (m/s)':>9} {'u* (m/s)':>9} "
          f"{'wth_s':>8} {'entr ratio':>11} {'TKE int':>9}")
    zis = []
    for t_, p in zip(tt, ps):
        k = int(np.argmin(p["wt"]))
        zi = float(p["z"][k])
        wts = float(p["wt"][0])
        th0 = float(p["th"][0])
        ws = (G / th0 * max(wts, 1e-9) * zi) ** (1.0 / 3.0)
        er = -float(p["wt"][k]) / max(wts, 1e-9)
        ti = float(np.trapz(p["tke"], p["z"]))
        zis.append(zi)
        print(f"  {t_/60:8.1f} {zi:8.0f} {ws:9.3f} {p['ust']:9.3f} {wts:8.4f} "
              f"{er:11.3f} {ti:9.1f}")

    last = ps[-1]
    k = int(np.argmin(last["wt"]))
    zi = float(last["z"][k])
    th0 = float(last["th"][0])
    wts = float(last["wt"][0])
    ws = (G / th0 * max(wts, 1e-9) * zi) ** (1.0 / 3.0)
    print(f"\n  --- final state: z_i = {zi:.0f} m, w* = {ws:.3f} m/s, "
          f"T* = z_i/w* = {zi/ws:.0f} s, u* = {last['ust']:.3f} m/s")
    L = -last["ust"] ** 3 * th0 / (0.4 * G * max(wts, 1e-9))
    print(f"      L = {L:.1f} m,  z_i/L = {zi/L:+.1f},  30/L = {30/L:+.3f}  "
          f"(CONUS404 midday p50 at this site: z_i/L = -19.8)")
    if len(zis) > 2:
        gr = np.polyfit(tt[-3:] / 3600.0, np.array(zis[-3:]), 1)[0]
        print(f"      z_i growth over the last 3 dumps: {gr:+.0f} m/h "
              f"(entrainment; a CBL has no stationary depth)")

    print(f"\n  --- sigma_w / w* against Lenschow et al. (1980) "
          f"(resolved + sub-grid) ---")
    print(f"  {'z/z_i':>7} {'z (m)':>7} {'LES':>7} {'Lenschow':>9} {'ratio':>7}")
    zz = last["z"] / zi
    sw = np.sqrt(last["ww"] + (2.0 / 3.0) * last["e"]) / ws
    for target in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0):
        kk = int(np.argmin(np.abs(zz - target)))
        ref = 1.34 * max(zz[kk], 1e-6) ** (1 / 3) * max(1 - 0.8 * zz[kk], 0.0)
        print(f"  {zz[kk]:7.2f} {last['z'][kk]:7.0f} {sw[kk]:7.3f} {ref:9.3f} "
              f"{sw[kk]/max(ref,1e-9):7.2f}")
    er = -float(last["wt"][k]) / max(wts, 1e-9)
    ok = 0.10 <= er <= 0.35
    print(f"\n  entrainment ratio {er:.3f}  (expected ~0.2): "
          f"{'OK' if ok else 'OUTSIDE 0.10-0.35 -- check the capping inversion'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
