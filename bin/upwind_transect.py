#!/usr/bin/env python3
"""Put the crosswind-integrated footprint next to the surface it came from.

The Stage 6 gate asks for a footprint that "differs from Kljun in an explicable direction"
-- and explicable means you can point at the map. This script puts f_y(x) directly above
the terrain height, the roughness and the land-cover class along the same upwind ray, so
the pointing is on one page rather than in an argument.

It exists because of the neutral southerly, whose f_y is BIMODAL -- a near lobe at 300 m
and a LARGER far lobe at 1080 m, reproduced independently by both halves of the window, so
not sampling noise. The transect shows why in one line: a 300 m band of tree cover
(z0 = 1.0 m against 0.1 for the cropland around it) sitting in the deepest part of a
hollow, then open cropland on ground that climbs 10 m over the next kilometre. Tall
roughness in a hollow lifts backward trajectories over it instead of letting them touch
down; the rising ground beyond comes up to meet them.

usage: upwind_transect.py <tag> [<tag> ...] [--prefix g24] [--outdir figures]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built",
        60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
COL = {10: "#1a6b1a", 20: "#8fbc45", 30: "#c9e35a", 40: "#e8c46a", 50: "#c04040",
       60: "#b9a37a", 70: "#ffffff", 80: "#00b8ff", 90: "#4fbfa8", 95: "#4fbfa8",
       100: "#d0d0d0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--prefix", default="g24")
    ap.add_argument("--grid", default="data/grid")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--rmax", type=float, default=2600.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    m = np.load(os.path.join(a.grid, "meta.npy"), allow_pickle=True).item()
    it, jt, dxg = m["itower"], m["jtower"], m["dx"]
    topo = np.load(os.path.join(a.grid, "topo.npy"))
    z0 = np.load(os.path.join(a.grid, "z0m.npy"))
    lc = np.load(os.path.join(a.grid, "lcclass.npy"))
    ny, nx = topo.shape

    tags = [t for t in a.tags if os.path.exists(f"results/{a.prefix}_{t}.npz")]
    if not tags:
        print("  no results for those tags"); return 1
    fig, axs = plt.subplots(3, len(tags), figsize=(5.6 * len(tags), 9.2), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.5, 1.0, 0.45], hspace=0.12))
    axs = np.atleast_2d(axs.reshape(3, len(tags)))
    for c, tag in enumerate(tags):
        z = np.load(f"results/{a.prefix}_{tag}.npz")
        j = json.load(open(f"results/{a.prefix}_{tag}.json"))
        wd = j["stats"]["wdir"]
        ang = np.radians(wd)
        ux, uy = np.sin(ang), np.cos(ang)          # unit vector pointing UPWIND
        d = np.arange(0.0, a.rmax + dxg, dxg / 2.0)
        ii = np.round(it + d * ux / dxg).astype(int) % nx
        jj = np.round(jt + d * uy / dxg).astype(int) % ny
        th, zr, cl = topo[jj, ii], z0[jj, ii], lc[jj, ii]

        ax = axs[0, c]
        sm = np.convolve(z["fy"], np.ones(5) / 5.0, mode="same")
        ax.plot(z["fy_xc"], z["fy"] * 1e3, lw=0.9, color="#1f77b4", alpha=0.35)
        ax.plot(z["fy_xc"], sm * 1e3, lw=2.3, color="#1f77b4", label="LES + LPDM")
        ax.plot(z["fy_xc"], z["fy_kljun"] * 1e3, lw=2.0, ls="--", color="#d62728",
                label="Kljun et al. (2015)")
        for h, ls in (("fy1", ":"), ("fy2", "-.")):
            ax.plot(z["fy_xc"], np.convolve(z[h], np.ones(5) / 5.0, mode="same") * 1e3,
                    lw=1.0, ls=ls, color="#1f77b4", alpha=0.75,
                    label="window halves" if h == "fy1" else None)
        ax.set_xlim(-50, a.rmax); ax.grid(alpha=0.25)
        ax.set_ylabel("$f_y$  ($10^{-3}$ m$^{-1}$)" if c == 0 else "")
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_title(f"{a.prefix} {tag}: wind FROM {wd:.0f}$\\degree$", fontsize=11)

        ax = axs[1, c]
        ax.plot(d, th, lw=2.0, color="#5b3a1a")
        ax.fill_between(d, th.min() - 3, th, color="#a0764a", alpha=0.30)
        ax.set_ylabel("terrain (m, about the taper datum)" if c == 0 else "")
        ax.grid(alpha=0.25)
        axr = ax.twinx()
        axr.step(d, zr, where="mid", lw=1.6, color="#2a7f2a")
        axr.set_yscale("log"); axr.set_ylim(5e-3, 3.0)
        axr.set_ylabel("$z_0$ (m)", color="#2a7f2a")
        axr.tick_params(axis="y", colors="#2a7f2a")

        ax = axs[2, c]
        for k in np.unique(cl):
            msk = cl == k
            ax.fill_between(d, 0, 1, where=msk, step="mid",
                            color=COL.get(int(k), "0.6"), lw=0)
        ax.set_yticks([]); ax.set_xlim(-50, a.rmax)
        ax.set_xlabel("upwind distance along the transect (m)")
        # Label the longest contiguous run of each class rather than its mean position,
        # which for an interleaved transect lands the text on a different class.
        for k in np.unique(cl):
            msk = (cl == k).astype(int)
            edges = np.diff(np.concatenate(([0], msk, [0])))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            if not len(starts):
                continue
            b = int(np.argmax(ends - starts))
            if (ends[b] - starts[b]) * (d[1] - d[0]) < 130.0:
                continue                      # too narrow to hold a label
            ax.text(0.5 * (d[starts[b]] + d[min(ends[b], len(d) - 1)]), 0.5,
                    NAME.get(int(k), str(int(k))), ha="center", va="center",
                    fontsize=8, color="k")

    fig.suptitle("Crosswind-integrated footprint against the surface it came from\n"
                 "same upwind ray, same case; roughness in green on a log scale",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=0.4)
    p = os.path.join(a.outdir, f"{a.prefix}_transect_{'_'.join(tags)}.png")
    fig.savefig(p, dpi=145); plt.close(fig)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
