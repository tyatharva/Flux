#!/usr/bin/env python3
"""Stage 6 gate: does the real-surface footprint differ from Kljun in an EXPLICABLE way?

Takes the flat/neutral result (Stage 5) and the real-surface result and asks what changed
and whether the change points at something we put into the model. The two things put in
are known exactly, because bin/prep_stage6.py wrote them: a terrain field and a roughness
patch. So the test is not "is it different" -- it will be -- but "does the difference sit
where the array and the terrain are, with the sign roughening and lifting would give".

usage: stage6_compare.py <flat.npz> <terrain.npz> [--outdir results] [--tag stage6]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.footprint import source_area_overlap


def contrib(f, xe, ye, x0, x1, y0, y1, res):
    xm = (xe[:-1] >= x0) & (xe[:-1] < x1)
    ym = (ye[:-1] >= y0) & (ye[:-1] < y1)
    return float(f[np.ix_(ym, xm)].sum() * res * res)


def stats(f, xc, yc, res):
    tot = f.sum()
    fy = f.sum(axis=0)
    flat = np.sort(f.ravel())[::-1]
    cum = np.cumsum(flat) / tot
    thr = flat[np.searchsorted(cum, 0.80)]
    return dict(peak_x=float(xc[int(np.argmax(fy))]),
                centroid_x=float((fy * xc).sum() / tot),
                centroid_y=float((f.sum(axis=1) * yc).sum() / tot),
                area80_ha=float((f >= thr).sum() * res * res / 1e4),
                integral=float(tot * res * res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flat"); ap.add_argument("terrain")
    ap.add_argument("--outdir", default="results"); ap.add_argument("--tag", default="stage6")
    ap.add_argument("--topo", default="runs/s30_stage6/topo.npy")
    ap.add_argument("--z0", default="runs/s30_stage6/z0m.npy")
    # ACTUAL array window relative to the receptor, not the nominal one. prep_stage6.py
    # was run with --itower 109 while the LPDM places the receptor at i = round(0.75*nx)
    # = 110, so the patch occupies cells i = 101..104, i.e. 180-270 m upwind.
    ap.add_argument("--array-centre", type=float, default=225.0)
    ap.add_argument("--array-x", type=float, default=120.0)
    ap.add_argument("--array-y", type=float, default=390.0)  # cells j=19..31, -180..180 m
    a = ap.parse_args()
    A, B = np.load(a.flat), np.load(a.terrain)
    xc, yc, xe, ye = A["xc"], A["yc"], A["xe"], A["ye"]
    res = float(xc[1] - xc[0])
    fa, fb, kl = A["les"], B["les"], A["kljun"]

    print("  quantity                     flat      terrain        Kljun")
    sa, sb, sk = (stats(x, xc, yc, res) for x in (fa, fb, kl))
    for k, lab in (("peak_x", "peak upwind distance (m)"),
                   ("centroid_x", "centroid, upwind (m)"),
                   ("centroid_y", "centroid, crosswind (m)"),
                   ("area80_ha", "80% source area (ha)"),
                   ("integral", "integral over grid")):
        print(f"  {lab:<28}{sa[k]:9.2f} {sb[k]:12.2f} {sk[k]:12.2f}")

    print(f"\n  80% source-area overlap   terrain vs flat  {source_area_overlap(np.maximum(fb,0), np.maximum(fa,0))*100:5.1f}%")
    print(f"                            terrain vs Kljun {source_area_overlap(np.maximum(fb,0), np.maximum(kl,0))*100:5.1f}%")
    print(f"                            flat    vs Kljun {source_area_overlap(np.maximum(fa,0), np.maximum(kl,0))*100:5.1f}%")

    x0, x1 = a.array_centre - a.array_x / 2, a.array_centre + a.array_x / 2
    y0, y1 = -a.array_y / 2, a.array_y / 2
    ca = contrib(fa, xe, ye, x0, x1, y0, y1, res)
    cb = contrib(fb, xe, ye, x0, x1, y0, y1, res)
    ck = contrib(kl, xe, ye, x0, x1, y0, y1, res)
    print(f"\n  --- contribution of the solar-array footprint "
          f"(x in [{x0:.0f},{x1:.0f}], y in [{y0:.0f},{y1:.0f}]) ---")
    print(f"  flat {ca*100:5.2f}%   terrain {cb*100:5.2f}%   Kljun {ck*100:5.2f}%"
          f"   terrain/flat = {cb/max(ca,1e-12):.2f}x")

    plot(a, xc, yc, fa, fb, kl)
    print(f"\n  wrote {a.outdir}/{a.tag}.png")


def plot(a, xc, yc, fa, fb, kl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.outdir, exist_ok=True)
    topo = np.load(a.topo) if os.path.exists(a.topo) else None
    z0 = np.load(a.z0) if os.path.exists(a.z0) else None
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))
    ext = [xc[0], xc[-1], yc[0], yc[-1]]
    vmax = max(fa.max(), fb.max(), kl.max())
    for A_, name, axx in ((kl, "Kljun et al. (2015)", ax[0, 0]),
                          (fa, "LES+LPDM, flat uniform", ax[0, 1]),
                          (fb, "LES+LPDM, real surface", ax[1, 0])):
        im = axx.imshow(A_, origin="lower", extent=ext, aspect="equal", vmin=0, vmax=vmax,
                        cmap="magma")
        axx.plot(0, 0, "w*", ms=12); axx.set_title(name)
        axx.set_xlabel("upwind distance (m)"); axx.set_ylabel("crosswind (m)")
        fig.colorbar(im, ax=axx, label="f (m$^{-2}$)")
        if z0 is not None:
            r = plt.Rectangle((a.array_centre - a.array_x / 2, -a.array_y / 2),
                              a.array_x, a.array_y, fill=False, ec="cyan", lw=1.2)
            axx.add_patch(r)
    d = fb - fa
    m = np.abs(d).max()
    im = ax[1, 1].imshow(d, origin="lower", extent=ext, aspect="equal", vmin=-m, vmax=m,
                         cmap="RdBu_r")
    if topo is not None:
        # terrain in the same wind-aligned frame, receptor at (i=109, j=25)
        tx = (np.arange(topo.shape[1]) - 109) * 30.0
        ty = (np.arange(topo.shape[0]) - 25) * 30.0
        ax[1, 1].contour(-tx, ty, topo, levels=8, colors="k", linewidths=0.5, alpha=0.6)
    ax[1, 1].plot(0, 0, "k*", ms=12)
    ax[1, 1].set_title("real minus flat (black: terrain contours)")
    ax[1, 1].set_xlabel("upwind distance (m)"); ax[1, 1].set_ylabel("crosswind (m)")
    ax[1, 1].set_xlim(xc[0], xc[-1]); ax[1, 1].set_ylim(yc[0], yc[-1])
    fig.colorbar(im, ax=ax[1, 1], label="$\\Delta f$ (m$^{-2}$)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, f"{a.tag}.png"), dpi=130)


if __name__ == "__main__":
    sys.exit(main())
