#!/usr/bin/env python3
"""Stage 6 on the static domain: what the tower sees as the wind turns.

The geography never moves. Only the wind does. So every difference between these panels is
flow, not a resampling artifact -- which is the whole reason the domain was made static.

The solar array CONTAINS the tower (60 m E/W, 250 m N, 100 m S), so its upwind reach is
250 m for a northerly and 60 m for an easterly, and the footprint share it takes should
swing by two orders of magnitude. That swing is the Stage 6 gate.

usage: fig_directional.py [--outdir figures] [--grid data/grid]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CASES = [("wN", "wind FROM the north", 0.0), ("wE", "wind FROM the east", 90.0),
         ("wS", "wind FROM the south", 180.0), ("wW", "wind FROM the west", 270.0)]
RES = 60.0


def cover_shares(tag):
    """Parse the footprint-weighted land-cover block out of a stage-5 report."""
    f = f"results/g24_{tag}.txt"
    if not os.path.exists(f):
        return {}
    out, on = {}, False
    for line in open(f):
        if "land-cover share" in line:
            on = True
            continue
        if on:
            m = re.match(r"\s+(.+?)\s+([\d.]+)%\s+\(domain area share\s+([\d.]+)%\)", line)
            if not m:
                break
            out[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    return out


def load(tag):
    p = f"results/g24_{tag}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    j = json.load(open(f"results/g24_{tag}.json"))
    f = z["les"]
    cap = float(f.sum()) * RES * RES
    return dict(xc=z["xc"], yc=z["yc"], xe=z["xe"], ye=z["ye"],
                les=f / cap if cap else f, kljun=z["kljun"], cap=cap, j=j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--grid", default="data/grid")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    meta = np.load(os.path.join(a.grid, "meta.npy"), allow_pickle=True).item()
    dxg, it_, jt_ = meta["dx"], meta["itower"], meta["jtower"]
    water = np.load(os.path.join(a.grid, "water.npy")) > 0.5
    array = np.load(os.path.join(a.grid, "array.npy")) > 0.5
    topo = np.load(os.path.join(a.grid, "topo.npy"))

    have = [(t, lab, d) for t, lab, d in CASES if load(t)]
    if not have:
        print("  no directional results yet"); return 1

    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(2, len(have), height_ratios=[1.25, 1.0], hspace=0.34,
                          wspace=0.24, left=0.055, right=0.975, top=0.845, bottom=0.085)

    vmax = max(load(t)["les"].max() for t, _, _ in have)
    norm = LogNorm(vmin=vmax / 1e4, vmax=vmax)
    shares = {}
    for c, (tag, lab, wdir) in enumerate(have):
        d = load(tag)
        st = d["j"]["stats"]
        ang = np.arctan2(st["V"], st["U"])
        ca, sa = np.cos(ang), np.sin(ang)
        ny, nx = water.shape
        ii = np.arange(-nx, 2 * nx); jj = np.arange(-ny, 2 * ny)
        I, J = np.meshgrid(ii, jj)
        dxm = (I - it_) * dxg; dym = (J - jt_) * dxg
        X = -(dxm * ca + dym * sa); Y = -dxm * sa + dym * ca
        tile = lambda m: m[J % ny, I % nx].astype(float)

        ax = fig.add_subplot(gs[0, c])
        ext = [d["xe"][0], d["xe"][-1], d["ye"][0], d["ye"][-1]]
        im = ax.imshow(np.ma.masked_less_equal(d["les"], 0), origin="lower", extent=ext,
                       aspect="equal", cmap="magma", norm=norm)
        ax.contourf(X, Y, tile(water), levels=[0.5, 1.5], colors=["#00b8ff"], alpha=0.18)
        ax.contour(X, Y, tile(water), levels=[0.5], colors="#00b8ff", linewidths=1.0)
        ax.contourf(X, Y, tile(array), levels=[0.5, 1.5], colors=["#39ff14"], alpha=0.45)
        ax.contour(X, Y, tile(array), levels=[0.5], colors="#39ff14", linewidths=1.8)
        ax.plot(0, 0, marker="*", ms=15, mfc="w", mec="k", mew=0.8, ls="none", zorder=6)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("upwind distance (m)")
        if c == 0:
            ax.set_ylabel("crosswind distance (m)")
        sh = cover_shares(tag); shares[tag] = sh
        arr = sh.get("solar array", (np.nan, np.nan))
        ax.set_title(f"{lab}\nachieved {st['wdir']:.0f}$\\degree$   |   "
                     f"array {arr[0]:.2f}% of the footprint", fontsize=10)
        if c == len(have) - 1:
            cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.035)
            cb.set_label("$f$ (m$^{-2}$), renormalised", fontsize=8)
        # inset: the near field, where the array lives
        axi = ax.inset_axes([0.56, 0.60, 0.42, 0.37])
        axi.imshow(np.ma.masked_less_equal(d["les"], 0), origin="lower", extent=ext,
                   aspect="equal", cmap="magma", norm=norm)
        axi.contourf(X, Y, tile(array), levels=[0.5, 1.5], colors=["#39ff14"], alpha=0.5)
        axi.contour(X, Y, tile(array), levels=[0.5], colors="#39ff14", linewidths=1.4)
        axi.plot(0, 0, marker="*", ms=9, mfc="w", mec="k", mew=0.6, ls="none")
        axi.set_xlim(-400, 700); axi.set_ylim(-450, 450)
        axi.set_xticks([]); axi.set_yticks([])
        for s_ in axi.spines.values():
            s_.set_edgecolor("w")

    # ---- the gate: array and water share vs direction --------------------------------
    axa = fig.add_subplot(gs[1, 0:2])
    tags = [t for t, _, _ in have]
    arr = [shares[t].get("solar array", (0, 0))[0] for t in tags]
    arr_a = [shares[t].get("solar array", (0, 1))[1] for t in tags]
    wat = [shares[t].get("water", (0, 0))[0] for t in tags]
    wat_a = [shares[t].get("water", (0, 1))[1] for t in tags]
    x = np.arange(len(tags))
    axa.bar(x - 0.19, arr, 0.36, color="#2ca02c", label="solar array, footprint share")
    axa.bar(x + 0.19, wat, 0.36, color="#1f77b4", label="open water, footprint share")
    axa.plot(x - 0.19, arr_a, "k_", ms=22, mew=2.0, label="area share of the domain")
    axa.plot(x + 0.19, wat_a, "k_", ms=22, mew=2.0)
    for i, v in enumerate(arr):
        axa.text(i - 0.19, v + 0.6, f"{v:.2f}%", ha="center", fontsize=8.5, color="#1a6b1a")
    for i, v in enumerate(wat):
        axa.text(i + 0.19, v + 0.6, f"{v:.1f}%", ha="center", fontsize=8.5, color="#14507f")
    axa.set_xticks(x)
    axa.set_xticklabels([f"{t}\n(array reaches "
                         f"{ {'wN':250,'wS':100,'wE':60,'wW':60}.get(t,0) } m upwind)"
                         for t in tags], fontsize=9)
    axa.set_ylabel("% of the flux footprint")
    axa.grid(alpha=0.25, axis="y")
    axa.legend(frameon=False, fontsize=9)
    axa.set_title("THE GATE — one fixed patch, one fixed lake, only the wind turns",
                  fontsize=10.5)

    # ---- crosswind-integrated, all directions ----------------------------------------
    axb = fig.add_subplot(gs[1, 2:])
    for tag, lab, _ in have:
        d = load(tag)
        f = d["les"].sum(axis=0) * RES
        axb.plot(d["xc"], np.convolve(f, np.ones(5) / 5, mode="same"), lw=2.0, label=lab)
    d0 = load(have[0][0])
    kl = d0["kljun"] / (d0["kljun"].sum() * RES * RES)
    axb.plot(d0["xc"], kl.sum(axis=0) * RES, lw=2.0, ls="--", color="k",
             label="Kljun (from the northerly's inputs)")
    axb.axvspan(0, 250, color="#39ff14", alpha=0.22)
    axb.text(125, axb.get_ylim()[1] * 0.02, "array reach,\nnortherly", ha="center",
             fontsize=8, color="#1a6b1a")
    axb.set_xlim(-50, 2600); axb.grid(alpha=0.25)
    axb.set_xlabel("upwind distance (m)"); axb.set_ylabel("$f_y$ (m$^{-1}$)")
    axb.legend(frameon=False, fontsize=8.5)
    axb.set_title("Crosswind-integrated footprint by direction", fontsize=10.5)

    fig.suptitle("Stage 6 — static 186 x 186 @ 24 m domain, four directions from ONE "
                 "spun-up state (90$\\degree$ re-indexing)\n"
                 "Terrain, roughness and the array are bit-identical in all four panels; "
                 "only the geostrophic wind was rotated.", fontsize=12)
    p = os.path.join(a.outdir, "g24_directional.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
