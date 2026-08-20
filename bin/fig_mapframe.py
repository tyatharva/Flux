#!/usr/bin/env python3
"""Footprints in the MAP frame: north-up, on the real domain, not rotated to the wind.

The wind-aligned view is what the estimator works in and it is the right frame for
comparing shapes. It is the wrong frame for looking at a SITE, because the geography moves
from panel to panel. Here the domain is fixed and north-up -- the lake, the array and the
terrain sit where they actually are -- and the footprint turns instead.

The footprint is computed on a wind-aligned raster, so each map cell is sampled from it by
the inverse of the transform in lpdm/driver.py:

    forward   X = -(dx ca + dy sa),   Y = -dx sa + dy ca      (ca,sa from the mean wind)
    inverse   dx = -(ca X + sa Y),    dy = -(sa X) + ca Y     (the map is an involution)

That resamples a 60 m raster onto a 24 m one, so the near field is interpolated rather than
recomputed; it is a picture of the footprint, not a new measurement of it.

usage: fig_mapframe.py [--outdir figures] [--grid data/grid]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from scipy.ndimage import map_coordinates

CASES = [("wN", "wind FROM the north"), ("wE", "wind FROM the east"),
         ("wS", "wind FROM the south"), ("wW", "wind FROM the west"),
         ("flat", "flat uniform control")]
RES = 60.0


def to_map(field, xc, yc, X, Y):
    """Sample a wind-frame raster at wind-frame coordinates (X, Y)."""
    ci = (X - xc[0]) / (xc[1] - xc[0])
    cj = (Y - yc[0]) / (yc[1] - yc[0])
    out = map_coordinates(field, [cj, ci], order=1, mode="constant", cval=np.nan)
    out[(ci < 0) | (ci > len(xc) - 1) | (cj < 0) | (cj > len(yc) - 1)] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--grid", default="data/grid")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    m = np.load(os.path.join(a.grid, "meta.npy"), allow_pickle=True).item()
    dxg, it_, jt_ = m["dx"], m["itower"], m["jtower"]
    water = np.load(os.path.join(a.grid, "water.npy")) > 0.5
    array = np.load(os.path.join(a.grid, "array.npy")) > 0.5
    topo = np.load(os.path.join(a.grid, "topo.npy"))
    ny, nx = topo.shape
    J, I = np.mgrid[0:ny, 0:nx]
    dxm = (I - it_) * dxg
    dym = (J - jt_) * dxg
    ext = [dxm.min() - dxg / 2, dxm.max() + dxg / 2,
           dym.min() - dxg / 2, dym.max() + dxg / 2]

    made = []
    for tag, lab in CASES:
        p = f"results/g24_{tag}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        j = json.load(open(f"results/g24_{tag}.json"))
        st = j["stats"]
        ang = np.arctan2(st["V"], st["U"])
        ca, sa = np.cos(ang), np.sin(ang)
        # inverse of the driver's map: wind-frame coords of every MAP cell
        X = -(ca * dxm + sa * dym)
        Y = -sa * dxm + ca * dym
        xc, yc = z["xc"], z["yc"]
        les = to_map(z["les"], xc, yc, X, Y)
        klj = to_map(z["kljun"], xc, yc, X, Y)
        # renormalise each over the map window actually shown
        cell = dxg * dxg
        les = les / np.nansum(les) / cell
        klj = klj / np.nansum(klj) / cell

        fig, axg = plt.subplots(2, 2, figsize=(13.4, 11.4))
        ax = [axg[0, 0], axg[0, 1], axg[1, 0]]
        ax1d = axg[1, 1]
        vmax = float(np.nanmax([np.nanmax(les), np.nanmax(klj)]))
        norm = LogNorm(vmin=vmax / 1e4, vmax=vmax)

        def decorate(axx, dark=True):
            axx.contour(dxm, dym, topo, levels=9, colors="k", linewidths=0.4, alpha=0.32)
            axx.contourf(dxm, dym, water.astype(float), levels=[0.5, 1.5],
                         colors=["#00b8ff"], alpha=0.20)
            axx.contour(dxm, dym, water.astype(float), levels=[0.5], colors="#00b8ff",
                        linewidths=1.1)
            axx.contour(dxm, dym, array.astype(float), levels=[0.5], colors="#39ff14",
                        linewidths=2.0)
            axx.plot(0, 0, marker="*", ms=15, mfc="w" if dark else "yellow",
                     mec="k", mew=0.8, ls="none", zorder=6)
            axx.set_xlim(ext[0], ext[1]); axx.set_ylim(ext[2], ext[3])
            axx.set_aspect("equal")
            axx.set_xlabel("east of the tower (m)")
            # arrow pointing the way the wind BLOWS, anchored upwind
            wd = np.radians(st["wdir"])
            ux, uy = -np.sin(wd), -np.cos(wd)
            axx.annotate("", xy=(1500 * ux, 1500 * uy), xytext=(1900 * -ux, 1900 * -uy),
                         arrowprops=dict(arrowstyle="-|>", lw=2.4,
                                         color="w" if dark else "k", alpha=0.85))

        for k, (F, name) in enumerate(((les, "LES + backward LPDM"),
                                       (klj, "Kljun et al. (2015), same inputs"))):
            im = ax[k].imshow(np.ma.masked_invalid(F), origin="lower", extent=ext,
                              cmap="magma", norm=norm)
            decorate(ax[k])
            ax[k].set_title(name, fontsize=11)
            if k == 0:
                ax[k].set_ylabel("north of the tower (m)")
            fig.colorbar(im, ax=ax[k], fraction=0.046, pad=0.02, label="$f$ (m$^{-2}$)")

        d = np.where(np.isfinite(les) & np.isfinite(klj), les - klj, np.nan)
        mm = float(np.nanmax(np.abs(d)))
        im = ax[2].imshow(np.ma.masked_invalid(d), origin="lower", extent=ext,
                          cmap="RdBu_r",
                          norm=SymLogNorm(linthresh=vmax / 1e3, vmin=-mm, vmax=mm, base=10))
        decorate(ax[2], dark=False)
        ax[2].set_title("LES $-$ Kljun  (red: the LES sees more here)", fontsize=11)
        fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02, label="$\\Delta f$ (m$^{-2}$)")

        # ---- bottom right: crosswind-integrated footprint, in the WIND frame ----------
        # Deliberately from the native wind-aligned raster, not from the map-frame
        # resample above: f_y is an integral across the wind, so it belongs in the frame
        # the estimator actually built, and resampling would blur the near field it is
        # meant to resolve.
        fyl = z["les"].sum(axis=0) * RES
        fyk = z["kljun"].sum(axis=0) * RES
        sm = np.convolve(fyl, np.ones(5) / 5.0, mode="same")
        ax1d.plot(xc, fyl, lw=1.0, color="#1f77b4", alpha=0.40)
        ax1d.plot(xc, sm, lw=2.3, color="#1f77b4", label="LES + LPDM (thin: raw)")
        ax1d.plot(xc, fyk, lw=2.3, ls="--", color="#d62728", label="Kljun et al. (2015)")
        pl = xc[int(np.argmax(fyl))]; pk_ = xc[int(np.argmax(fyk))]
        ax1d.axvline(pl, color="#1f77b4", lw=0.9, ls=":")
        ax1d.axvline(pk_, color="#d62728", lw=0.9, ls=":")
        ax1d.set_xlim(-50, 2600)
        ax1d.set_ylim(min(0.0, 1.15 * fyl.min()), 1.28 * max(fyl.max(), fyk.max()))
        ax1d.set_xlabel("upwind distance (m)")
        ax1d.set_ylabel("$f_y$ (m$^{-1}$)")
        ax1d.grid(alpha=0.25)
        ax1d.legend(frameon=False, fontsize=9, loc="upper center")
        ax1d.set_title("Crosswind-integrated footprint", fontsize=11)
        il, ik = j["integral_les"], j["integral_kljun"]
        cov = ""
        cf = f"results/g24_{tag}.txt"
        if os.path.exists(cf):
            on = False
            for line in open(cf):
                if "land-cover share" in line:
                    on = True; continue
                if on:
                    if "(domain area share" not in line:
                        break
                    nm_ = line.split()[0] if not line.strip().startswith("solar") else "array"
                    if nm_ in ("array", "water"):
                        v_ = float(line.split("%")[0].split()[-1])
                        a_ = float(line.split("area share")[1].split("%")[0])
                        cov += f"\n{nm_:<6s} {v_:6.2f}% of the footprint ({a_:5.2f}% of area)"
        ax1d.text(0.985, 0.74,
                  "INTEGRAL over the grid\n"
                  f"LES + LPDM   {il:.3f}\n"
                  f"Kljun        {ik:.3f}\n"
                  f"\npeak  LES {pl:.0f} m   Kljun {pk_:.0f} m" + cov,
                  transform=ax1d.transAxes, fontsize=8.5, family="monospace", va="top",
                  ha="right", bbox=dict(fc="w", ec="0.7", boxstyle="round,pad=0.4"))

        sub = (f"{lab}   |   surface wind FROM {st['wdir']:.0f}$\\degree$   |   "
               f"$u_*$={st['ustar']:.3f} m s$^{{-1}}$, $h$={st['h']:.0f} m")
        fig.suptitle("Flux footprint on the real domain, north-up — 186 x 186 @ 24 m, "
                     "tower at the centre\n" + sub, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.925])
        out = os.path.join(a.outdir, f"g24_map_{tag}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        made.append(out)
        print(f"  wrote {out}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
