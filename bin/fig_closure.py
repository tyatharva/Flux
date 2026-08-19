#!/usr/bin/env python3
"""The sub-grid closure experiment, in one figure.

Same LES fields, same seed, same releases; only the sub-grid velocity variance handed to
the Langevin model changes. Kljun is fixed, so it is a stationary reference in every panel.

The point of the figure is that the physically motivated change goes the WRONG way, and by
roughly the amount the sigma_w deficit predicts -- which is what turns a vague "the near
field is under-resolved" into a specific, measured, correctable quantity.

usage: fig_closure.py [--outdir figures]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = 60.0
VARIANTS = [
    ("fv_stage5", "isotropic $(2/3)e_{sgs}$\n(baseline, Weil et al. 2004)", "#1f77b4", "-"),
    ("fv_aniso", "surface-layer ANISOTROPIC split\n$r_u{:}r_v{:}r_w$ = 1.64:0.95:0.41",
     "#d62728", "-"),
    ("fv_sgs135", "variance $\\times$1.349 (tuned scalar)", "#7f7f7f", ":"),
    ("fv_most2", "MOST-anchored floor  [ADOPTED]", "#2ca02c", "-"),
]


def load(tag):
    z = np.load(f"results/{tag}.npz")
    j = json.load(open(f"results/{tag}.json"))
    f = z["les"]
    cap = float(f.sum()) * RES * RES
    return z["xc"], f / cap, cap, j


def fy(f):
    return f.sum(axis=0) * RES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    xc, _, _, j0 = load("fv_stage5")
    zk = np.load("results/fv_stage5.npz")
    kl = zk["kljun"]
    kl = kl / (float(kl.sum()) * RES * RES)

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.36, wspace=0.30,
                          left=0.062, right=0.975, top=0.845, bottom=0.085)

    # ---- (a) crosswind-integrated footprints ---------------------------------------
    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(xc, fy(kl), lw=2.6, color="k", ls="--", label="Kljun et al. (2015)")
    rows = []
    for tag, lab, col, ls in VARIANTS:
        try:
            x, f, cap, j = load(tag)
        except FileNotFoundError:
            continue
        c = fy(f)
        sm = np.convolve(c, np.ones(5) / 5.0, mode="same")
        ax.plot(x, sm, lw=2.2, color=col, ls=ls, label=lab)
        rows.append((lab.split("\n")[0], j["les"]["peak_x"], j["les"]["centroid_x"],
                     j["les"]["area80_cells"] * RES * RES / 1e4,
                     j["overlap_kljun"] * 100, j["integral_les"], col))
    ax.axvline(j0["kljun"]["peak_x"], color="k", lw=0.8, ls=":")
    ax.set_xlim(-50, 2600)
    ax.set_xlabel("upwind distance (m)")
    ax.set_ylabel("$f_y$ (m$^{-1}$), renormalised")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Crosswind-integrated flux footprint — identical fields, identical seed, "
                 "only the sub-grid variance differs", fontsize=10.5)

    # ---- (b) the sigma_w argument ---------------------------------------------------
    axb = fig.add_subplot(gs[0, 2])
    st = j0["stats"]
    us = st["ustar"]
    res_w = st["sigma_w_resolved"]
    e = st["e_sgs"]
    names = ["anisotropic", "isotropic\n(baseline)", "MOST floor\n(adopted)",
             "scalar\n$\\times$1.349"]
    facs = [0.4104, 1.0, 1.242, 1.349]
    sw = [np.sqrt(res_w ** 2 + f * (2 / 3) * e) / us for f in facs]
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f"]
    axb.bar(range(4), sw, color=cols, alpha=0.85)
    axb.axhline(1.25, color="k", ls="--", lw=1.6)
    axb.text(-0.45, 1.31, "surface-layer $\\sigma_w/u_*$ = 1.25", ha="left", fontsize=8.5)
    axb.set_xticks(range(4)); axb.set_xticklabels(names, fontsize=8)
    axb.set_ylabel("$\\sigma_w / u_*$ at the receptor")
    axb.set_ylim(0, 1.52); axb.grid(alpha=0.25, axis="y")
    axb.set_title("What each variant does to $\\sigma_w$", fontsize=10)
    for i, v in enumerate(sw):
        axb.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.5)

    # ---- (c) peak error vs sigma_w --------------------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    pk = [1170.0, 390.0, 270.0, 270.0]
    axc.plot(sw, pk, "o-", color="0.3", lw=1.4, ms=9, zorder=1)
    for i, (x_, y_) in enumerate(zip(sw, pk)):
        axc.plot(x_, y_, "o", color=cols[i], ms=11, zorder=2)
    axc.axhline(j0["kljun"]["peak_x"], color="k", ls="--", lw=1.4)
    axc.text(0.80, j0["kljun"]["peak_x"] + 40, "Kljun 210 m", fontsize=8.5)
    axc.axvline(1.25, color="k", ls=":", lw=1.0)
    axc.set_xlabel("$\\sigma_w / u_*$ at the receptor")
    axc.set_ylabel("footprint peak (m)")
    axc.grid(alpha=0.25)
    axc.set_title("Peak location is set by $\\sigma_w$", fontsize=10)

    # ---- (d) the numbers ------------------------------------------------------------
    axd = fig.add_subplot(gs[1, 1:])
    axd.axis("off")
    hdr = f"{'variant':<34}{'peak':>8}{'vs Kljun':>10}{'80% area':>11}{'overlap':>9}{'integral':>10}"
    lines = [hdr, "-" * len(hdr),
             f"{'Kljun et al. (2015)':<34}{j0['kljun']['peak_x']:>7.0f} m{'---':>10}"
             f"{26.64:>10.1f} ha{'---':>9}{j0['integral_kljun']:>10.3f}"]
    for lab, p, c, ar, ov, it, col in rows:
        d = (p - j0["kljun"]["peak_x"]) / j0["kljun"]["peak_x"] * 100
        lines.append(f"{lab:<34}{p:>7.0f} m{d:>9.0f}%{ar:>10.1f} ha{ov:>8.1f}%{it:>10.3f}")
    axd.text(0.0, 0.98, "\n".join(lines), family="monospace", fontsize=9,
             va="top", transform=axd.transAxes)
    axd.text(0.0, 0.30,
             "The anisotropic split is the physically motivated change — near a wall,\n"
             "blocking suppresses $w$. It made the footprint FOUR TIMES worse, in the\n"
             "direction and by roughly the amount a $\\sigma_w$ deficit predicts. So the\n"
             "isotropic split was not the error; it was compensating for one.\n\n"
             "Adopted is the MOST-anchored floor, not the better-scoring scalar: the\n"
             "scalar is a constant fitted to one case at one height, with no rule for\n"
             "transferring it to terrain or to other stabilities.",
             fontsize=8.8, va="top", transform=axd.transAxes)

    fig.suptitle("Where the near-field footprint error actually comes from\n"
                 "flat, uniform, neutral window; the receptor sits at $z/\\Delta$ = 1.5, "
                 "so 88% of $\\sigma_w^2$ there is sub-grid", fontsize=12)
    p = os.path.join(a.outdir, "closure_experiment.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
