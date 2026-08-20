#!/usr/bin/env python3
"""Footprints on the static domain, north-up, with nothing resampled.

The predecessor of this script computed the footprint on a wind-aligned raster and then
interpolated it onto the map. That interpolation fell hardest on the near field, which is
where the footprint peak lives and where the solar array sits -- so the picture was blurred
in exactly the place the result depends on. The estimator now accumulates onto the LES
columns directly, so a pixel here IS an LES column and the only thing that moves between
panels is the wind.

Kljun is evaluated at those same cells' own coordinates rather than rotated onto them
(lpdm.kljun.footprint_on_static). FFP is a closed-form function; interpolating it would
have been gratuitous.

Panels: LES+LPDM | Kljun | difference | crosswind-integrated, with the integrals.

usage: fig_static.py [--outdir figures] [--grid data/grid] [--cases wN,wE,...]
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

LABEL = {"wN": "wind FROM the north", "wE": "wind FROM the east",
         "wS": "wind FROM the south", "wW": "wind FROM the west",
         "flat": "flat uniform control", "cflat": "flat uniform control, convective"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--grid", default="data/grid")
    ap.add_argument("--prefix", default="g24")
    ap.add_argument("--cases", default="wN,wE,wS,wW,flat")
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

    made = []
    for tag in a.cases.split(","):
        p = f"results/{a.prefix}_{tag}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        j = json.load(open(f"results/{a.prefix}_{tag}.json"))
        st = j["stats"]
        les, klj = z["les"], z["kljun"]
        ext = [z["xe"][0], z["xe"][-1], z["ye"][0], z["ye"][-1]]
        flat = ("flat" in tag)

        fig, axg = plt.subplots(2, 2, figsize=(13.4, 11.6))
        ax = [axg[0, 0], axg[0, 1], axg[1, 0]]
        ax1d = axg[1, 1]
        vmax = float(max(les.max(), klj.max()))
        vmin = vmax / 1e4
        norm = LogNorm(vmin=vmin, vmax=vmax)

        def decorate(axx, dark=True):
            if not flat:
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
            wd = np.radians(st["wdir"])
            ux, uy = -np.sin(wd), -np.cos(wd)
            axx.annotate("", xy=(1500 * ux, 1500 * uy), xytext=(1900 * -ux, 1900 * -uy),
                         arrowprops=dict(arrowstyle="-|>", lw=2.4,
                                         color="w" if dark else "k", alpha=0.85))

        for k, (F, name) in enumerate(((les, "LES + backward LPDM"),
                                       (klj, "Kljun et al. (2015), same scalars"))):
            # Mask below the colour floor rather than painting it black: Kljun is
            # analytically positive over a wide wedge and tiny over most of it, so a
            # clipped log scale renders the whole wedge as a solid dark block and hides
            # the geography underneath. Masking shows where each field is actually
            # negligible, and it is the same threshold for both panels.
            im = ax[k].imshow(np.ma.masked_less(F, vmin), origin="lower", extent=ext,
                              cmap="magma", norm=norm)
            decorate(ax[k])
            ax[k].set_title(name, fontsize=11)
            if k == 0:
                ax[k].set_ylabel("north of the tower (m)")
            fig.colorbar(im, ax=ax[k], fraction=0.046, pad=0.02, label="$f$ (m$^{-2}$)")

        d = les - klj
        mm = float(np.nanmax(np.abs(d)))
        im = ax[2].imshow(d, origin="lower", extent=ext, cmap="RdBu_r",
                          norm=SymLogNorm(linthresh=vmax / 1e3, vmin=-mm, vmax=mm, base=10))
        decorate(ax[2], dark=False)
        ax[2].set_title("LES $-$ Kljun  (red: the LES sees more here)", fontsize=11)
        ax[2].set_ylabel("north of the tower (m)")
        fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02, label="$\\Delta f$ (m$^{-2}$)")

        # ---- crosswind-integrated, accumulated in the wind frame from the touchdowns ---
        xc, fyl, fyk = z["fy_xc"], z["fy"], z["fy_kljun"]
        sm = np.convolve(fyl, np.ones(5) / 5.0, mode="same")
        ax1d.plot(xc, fyl, lw=1.0, color="#1f77b4", alpha=0.35)
        ax1d.plot(xc, sm, lw=2.3, color="#1f77b4", label="LES + LPDM (thin: raw, 24 m bins)")
        ax1d.plot(xc, fyk, lw=2.3, ls="--", color="#d62728", label="Kljun et al. (2015)")
        pl, pk_ = j["les"]["peak_x"], j["kljun"]["peak_x"]
        ax1d.axvline(pl, color="#1f77b4", lw=0.9, ls=":")
        ax1d.axvline(pk_, color="#d62728", lw=0.9, ls=":")
        ax1d.set_xlim(-100, 2600)
        ax1d.set_ylim(min(0.0, 1.2 * float(fyl.min())), 1.30 * float(max(fyl.max(), fyk.max())))
        ax1d.set_xlabel("upwind distance (m)")
        ax1d.set_ylabel("$f_y$ (m$^{-1}$)")
        ax1d.grid(alpha=0.25)
        ax1d.legend(frameon=False, fontsize=9, loc="upper center")
        ax1d.set_title("Crosswind-integrated footprint", fontsize=11)
        cov = ""
        for nm in ("solar array", "water"):
            v = j.get("cover_share", {}).get(nm)
            if v is not None and np.isfinite(v):
                cov += f"\n{nm:<11s} {100*v:6.2f}% of the footprint"
        ax1d.text(0.985, 0.72,
                  "INTEGRAL over the domain\n"
                  f"LES + LPDM   {j['integral_les']:.3f}\n"
                  f"Kljun        {j['integral_kljun']:.3f}\n"
                  f"\npeak  LES {pl:.0f} m   Kljun {pk_:.0f} m\n"
                  f"80% source area {j['les']['area80_ha']:.0f} ha" + cov,
                  transform=ax1d.transAxes, fontsize=8.5, family="monospace", va="top",
                  ha="right", bbox=dict(fc="w", ec="0.7", boxstyle="round,pad=0.4"))

        sub = (f"{LABEL.get(tag, tag)}   |   surface wind FROM {st['wdir']:.0f}$\\degree$"
               f"   |   $u_*$={st['ustar']:.3f} m s$^{{-1}}$, $z_i$={st['h']:.0f} m, "
               f"$z_i/L$={st['h']*(1/st['L']):+.1f}   |   "
               f"{j['tback']:.0f} s of backward time, 30 min of releases")
        fig.suptitle("Flux footprint on the static domain, north-up — 186 x 186 @ 24 m, "
                     "accumulated on the LES columns\n" + sub, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.925])
        out = os.path.join(a.outdir, f"{a.prefix}_static_{tag}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        made.append(out)
        print(f"  wrote {out}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
