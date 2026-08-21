#!/usr/bin/env python3
"""Stage 6 on the static domain: what the tower sees as the wind turns.

One geography, four winds. Terrain, roughness, the array and (convectively) the surface
heat-flux map are bit-identical in all four panels, so every difference between them is
flow. The footprints are accumulated on the LES columns themselves, so the near field --
where the array is -- is not blurred by any rotation or resample.

The array CONTAINS the tower (60 m E/W, 250 m N, 100 m S), so its upwind reach swings from
250 m on a northerly to 60 m on an easterly, and the share of the footprint it takes should
swing by two orders of magnitude. That swing is the gate.

usage: fig_gate6.py [--prefix g24] [--outdir figures]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CASES = [("wN", "FROM the north"), ("wE", "FROM the east"),
         ("wS", "FROM the south"), ("wW", "FROM the west")]


def load(pre, tag):
    p = f"results/{pre}_{tag}.npz"
    if not os.path.exists(p):
        return None
    return np.load(p), json.load(open(f"results/{pre}_{tag}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="g24")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--grid", default="data/grid")
    ap.add_argument("--title", default="neutral")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    m = np.load(os.path.join(a.grid, "meta.npy"), allow_pickle=True).item()
    dxg, it_, jt_ = m["dx"], m["itower"], m["jtower"]
    water = np.load(os.path.join(a.grid, "water.npy")) > 0.5
    array = np.load(os.path.join(a.grid, "array.npy")) > 0.5
    topo = np.load(os.path.join(a.grid, "topo.npy"))
    ny, nx = topo.shape
    J, I = np.mgrid[0:ny, 0:nx]
    dxm, dym = (I - it_) * dxg, (J - jt_) * dxg

    have = [(t, lab, load(a.prefix, t)) for t, lab in CASES]
    have = [(t, lab, d) for t, lab, d in have if d]
    if not have:
        print("  no directional results yet"); return 1

    fig = plt.figure(figsize=(17.0, 9.4))
    gs = fig.add_gridspec(2, len(have), height_ratios=[1.3, 1.0], hspace=0.30,
                          wspace=0.18, left=0.05, right=0.975, top=0.845, bottom=0.085)
    vmax = max(float(d[0]["les"].max()) for _, _, d in have)
    norm = LogNorm(vmin=vmax / 1e4, vmax=vmax)
    shares = {}
    for c, (tag, lab, (z, j)) in enumerate(have):
        st = j["stats"]
        ext = [z["xe"][0], z["xe"][-1], z["ye"][0], z["ye"][-1]]
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(np.ma.masked_less_equal(z["les"], 0), origin="lower", extent=ext,
                       aspect="equal", cmap="magma", norm=norm)
        ax.contour(dxm, dym, topo, levels=9, colors="k", linewidths=0.35, alpha=0.30)
        ax.contourf(dxm, dym, water.astype(float), levels=[0.5, 1.5], colors=["#00b8ff"],
                    alpha=0.18)
        ax.contour(dxm, dym, water.astype(float), levels=[0.5], colors="#00b8ff", lw=1.0)
        ax.contourf(dxm, dym, array.astype(float), levels=[0.5, 1.5], colors=["#39ff14"],
                    alpha=0.55)
        ax.contour(dxm, dym, array.astype(float), levels=[0.5], colors="#39ff14", lw=1.8)
        ax.plot(0, 0, marker="*", ms=14, mfc="w", mec="k", mew=0.8, ls="none", zorder=6)
        wd = np.radians(st["wdir"]); ux, uy = -np.sin(wd), -np.cos(wd)
        ax.annotate("", xy=(1400*ux, 1400*uy), xytext=(1900*-ux, 1900*-uy),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color="w", alpha=0.85))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("east of the tower (m)")
        if c == 0:
            ax.set_ylabel("north of the tower (m)")
        # Quote the UNWRAPPED shares: a touchdown 3 km upwind folds to 1.5 km on the far
        # side of the tower, where the real land cover is a different lake and a different
        # wood. The folded shares are right for the tiled world the LES simulates and for
        # the emulator's target; they are not right for attributing flux to this site.
        sh = j.get("cover_share_nowrap") or j.get("cover_share", {})
        shares[tag] = sh
        ax.set_title(f"wind {lab}\nachieved {st['wdir']:.0f}$\\degree$   |   "
                     f"array {100*sh.get('solar array', np.nan):.2f}% of the footprint",
                     fontsize=10)
        if c == len(have) - 1:
            cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.035)
            cb.set_label("$f$ (m$^{-2}$)", fontsize=8)
        # Put the near-field inset on the DOWNWIND side, so it never covers the plume.
        # The plume runs upwind along (ux, uy), so the free corner is the opposite one.
        # (ux, uy) is the direction the wind BLOWS, so it points DOWNWIND -- which is the
        # empty half of the panel, the footprint being upwind of the tower.
        ix0 = 0.56 if ux > 0 else 0.02
        iy0 = 0.60 if uy > 0 else 0.03
        axi = ax.inset_axes([ix0, iy0, 0.42, 0.37])
        axi.imshow(np.ma.masked_less_equal(z["les"], 0), origin="lower", extent=ext,
                   aspect="equal", cmap="magma", norm=norm)
        axi.contourf(dxm, dym, array.astype(float), levels=[0.5, 1.5], colors=["#39ff14"],
                     alpha=0.55)
        axi.contour(dxm, dym, array.astype(float), levels=[0.5], colors="#39ff14", lw=1.4)
        axi.plot(0, 0, marker="*", ms=9, mfc="w", mec="k", mew=0.6, ls="none")
        axi.set_xlim(-450, 450); axi.set_ylim(-450, 450)
        axi.set_xticks([]); axi.set_yticks([])
        for s_ in axi.spines.values():
            s_.set_edgecolor("#39ff14"); s_.set_linewidth(1.4)
        axi.set_title("near field, $\\pm$450 m", fontsize=7.5, color="0.25", pad=1.5)

    # ---- the gate ---------------------------------------------------------------------
    axa = fig.add_subplot(gs[1, 0:2])
    tags = [t for t, _, _ in have]
    arr = [100 * shares[t].get("solar array", np.nan) for t in tags]
    wat = [100 * shares[t].get("water", np.nan) for t in tags]
    aa, wa = 100 * array.mean(), 100 * water.mean()
    x = np.arange(len(tags))
    axa.bar(x - 0.19, arr, 0.36, color="#2ca02c", label="solar array")
    axa.bar(x + 0.19, wat, 0.36, color="#1f77b4", label="open water")
    axa.axhline(aa, color="#2ca02c", ls="--", lw=1.2)
    axa.axhline(wa, color="#1f77b4", ls="--", lw=1.2)
    axa.text(-0.45, aa + 0.15, f"array area share {aa:.2f}%", fontsize=8,
             color="#1a6b1a", va="bottom", ha="left")
    axa.text(-0.45, wa + 0.15, f"water area share {wa:.1f}%", fontsize=8,
             color="#14507f", va="bottom", ha="left")
    # Log scale: the whole point is a swing of two to three orders of magnitude, and a
    # linear axis renders every direction but the easterly as a flat line at zero.
    axa.set_yscale("symlog", linthresh=0.01, linscale=0.5)
    axa.set_ylim(0, 40)
    for i, v in enumerate(arr):
        axa.text(i - 0.19, max(v, 0.011) * 1.35, f"{v:.2f}%\n{v/aa:.1f}x area",
                 ha="center", fontsize=8, color="#1a6b1a")
    for i, v in enumerate(wat):
        axa.text(i + 0.19, max(v, 0.011) * 1.35, f"{v:.2f}%\n{v/wa:.2f}x area",
                 ha="center", fontsize=8, color="#14507f")
    axa.set_xticks(x)
    axa.set_xticklabels([f"{t}\n{load(a.prefix,t)[1]['stats']['wdir']:.0f}$\\degree$"
                         for t in tags], fontsize=9)
    axa.set_ylabel("% of the flux footprint")
    axa.grid(alpha=0.25, axis="y")
    axa.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
    axa.set_title("THE GATE — one fixed patch, one fixed lake, only the wind turns",
                  fontsize=10.5)

    axb = fig.add_subplot(gs[1, 2:])
    for tag, lab, (z, j) in have:
        axb.plot(z["fy_xc"], np.convolve(z["fy"], np.ones(5)/5, mode="same"), lw=2.0,
                 label=f"{lab} ({j['stats']['wdir']:.0f}$\\degree$)")
    z0, j0 = have[0][2]
    axb.plot(z0["fy_xc"], z0["fy_kljun"], lw=2.0, ls="--", color="k",
             label="Kljun (from the first panel's scalars)")
    axb.axvspan(0, 250, color="#39ff14", alpha=0.18)
    axb.text(125, 0, " array reach,\n due northerly", fontsize=8, color="#1a6b1a", va="bottom")
    axb.set_xlim(-100, 2600); axb.grid(alpha=0.25)
    axb.set_xlabel("upwind distance (m)"); axb.set_ylabel("$f_y$ (m$^{-1}$)")
    axb.legend(frameon=False, fontsize=8.5)
    axb.set_title("Crosswind-integrated footprint by direction", fontsize=10.5)

    fig.suptitle(f"Stage 6 — static 186 x 186 @ 24 m domain, {a.title}: four directions "
                 "from ONE spun-up state (90$\\degree$ re-indexing)\n"
                 "Footprints accumulated on the LES columns; 30 min of releases per case; "
                 "geography bit-identical across panels. Cover shares exclude "
                 "periodically folded touchdowns.", fontsize=12)
    p = os.path.join(a.outdir, f"{a.prefix}_gate6.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
