#!/usr/bin/env python3
"""The wind-aligned cone mask, as a figure. Nine panels: how k was chosen, and what it did.

bin/mask_cone.py replaces the retired half-plane cut. The half-plane could only test the
SIGN of the along-wind projection, and the periodic fold is per-axis and independent, so a
particle that wraps in one axis alone lands back UPWIND as a thin off-axis streak and
survives it. The cone tests crosswind distance against Kljun's own sigma_y instead, which is
the quantity that actually bounds where real material can be.

The panel that justifies k is the middle-left of row 2: the LES mass distribution against
q = |y'|/sigma_y(x') is bimodal with an EMPTY valley between the footprint and the wrap.
k sits in that valley, which is why the answer barely moves when k does.

usage: fig_cone_mask.py [--h5 corpus/corpus.h5] [--out figures/cone_mask_effect.png]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_cone as mc                                     # noqa: E402
import fig_corpus_pairs as fcp                             # noqa: E402

SPLIT_COLOUR = fcp.SPLIT_COLOUR


def cone_boundary(sin_w, cos_w, sy_of_x, k, y_min, smax=2400.0):
    """The cone edge in the north-up frame, as two polylines.

    The wind-frame map (x', y') = (x sin + y cos, x cos - y sin) is an involution, so the
    inverse is the same expression: x = x' sin + y' cos, y = x' cos - y' sin.
    """
    xp = np.linspace(-y_min, smax, 400)
    yp = np.maximum(k * sy_of_x(xp), y_min)
    out = []
    for sgn in (+1, -1):
        yy = sgn * yp
        out.append((xp * sin_w + yy * cos_w, xp * cos_w - yy * sin_w))
    return out


def case_row(fig, gs, h5, d, run_id, surf, k, y_min):
    """Raw target, target_cone, and what the cone deleted -- for one named record."""
    import h5py
    i = int(np.where(d["run_id"] == run_id)[0][0])
    with h5py.File(h5, "r") as f:
        t = f["target"][i].astype(np.float64)
        tc = f["target_cone"][i].astype(np.float64)
        sc = f["scalars"][i]
    X, Y = mc.axis_grids()
    sin_w, cos_w = float(sc[4]), float(sc[5])
    xw, yw = mc.wind_frame(X, Y, sin_w, cos_w)
    removed = t - tc
    prof = mc.kljun_ffp.ffp_profile(mc.Z_RECEPTOR, float(sc[0]), float(sc[3]), float(sc[1]),
                                    float(sc[2]), umean=float(d["u_mean"][i]))
    sy_of_x = lambda x: np.interp(np.maximum(x, 0.0), prof["x"], prof["sigy"])
    xc, xe = fcp.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    vmax = float(np.abs(t).max())
    lognorm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    symnorm = SymLogNorm(linthresh=vmax * 1e-4, vmin=-vmax, vmax=vmax, base=10)

    axs = []
    for c, (F, name, norm, cmap, floor, fg) in enumerate((
            (t, "target (raw LES)", lognorm, "magma", lognorm.vmin, "w"),
            (tc, "target_cone -- THE TRAINING TARGET", lognorm, "magma", lognorm.vmin, "w"),
            (removed, "what the cone DELETED", symnorm, "RdBu_r", None, "k"))):
        ax = fig.add_subplot(gs[0, c])
        axs.append(ax)
        im = fcp.raster(ax, F, norm, cmap, ext, mask_below=floor)
        fcp.draw_frame(ax, surf, fg=fg)
        fcp.draw_wind(ax, float(d["wdir"][i]), colour=fg)
        for bx, by in cone_boundary(sin_w, cos_w, sy_of_x, k, y_min):
            ax.plot(bx, by, color="#39ff14", lw=1.2, ls="--", zorder=8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("east  [m]", fontsize=8)
        if c == 0:
            ax.set_ylabel("north  [m]", fontsize=8)
        ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(xe[0], xe[-1])
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6.5)

    a = np.abs(t)
    up = (np.abs(removed) > 0) & (xw >= 0)
    fcp.stamp(axs[0], f"integral {d['I_raw'][i]:.3f}\nasympt.  {d['asym'][i]:.3f}",
              loc="upper left")
    fcp.stamp(axs[1], f"integral {d['I_cone'][i]:.3f}\nchange   "
                      f"{d['I_cone'][i] - d['I_raw'][i]:+.3f}", loc="upper left")
    axs[2].text(0.02, 0.98,
                f"|mass| removed {100 * d['rm_abs'][i]:.1f}% of |f|\n"
                f"  of it UPWIND {100 * a[up].sum() / a.sum():.1f}%  <- the half-plane\n"
                f"                       missed this\n"
                f"within 200 m   {100 * d['rm_near'][i]:.3f}%",
                transform=axs[2].transAxes, va="top", fontsize=6.5, family="monospace",
                bbox=dict(fc="w", ec="none", alpha=0.78, boxstyle="round,pad=0.25"))
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="corpus/corpus.h5")
    ap.add_argument("--npz-dir", default="corpus/pairs_npz")
    ap.add_argument("--out", default="figures/cone_mask_effect.png")
    ap.add_argument("--surface", default="data/grid30_raised")
    ap.add_argument("--k", type=float, default=mc.K_DEFAULT)
    ap.add_argument("--y-min", type=float, default=mc.YMIN_DEFAULT)
    ap.add_argument("--zm", type=float, default=30.0)
    ap.add_argument("--near-m", type=float, default=200.0)
    ap.add_argument("--case", default="case_2022030716")
    ap.add_argument("--dpi", type=int, default=130)
    a = ap.parse_args()

    print("finding the valley ...")
    edges, Hl, Hk, nsamp = mc.choose_k(a.h5, a.npz_dir)
    print(f"measuring at k = {a.k:g}, y_min = {a.y_min:g} ...")
    d = mc.measure(a.h5, a.k, a.y_min, a.zm, a.near_m, a.npz_dir)
    surf = fcp.load_surface(a.surface)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    fig = plt.figure(figsize=(14.4, 13.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.30, 1.0, 1.0], hspace=0.36, wspace=0.27,
                          left=0.055, right=0.965, top=0.902, bottom=0.075)
    case_row(fig, gs, a.h5, d, a.case, surf, a.k, a.y_min)

    order = ["train", "val", "test"]
    sp = d["split"]

    # --- THE PANEL THAT JUSTIFIES k -------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    fin = np.isfinite(ctr)
    ax.step(ctr[fin], 100 * Hl[fin], where="mid", color="#4c72b0", lw=1.5, label="LES target")
    ax.step(ctr[fin], 100 * Hk[fin], where="mid", color="#c44e52", lw=1.2,
            label="Kljun input")
    ax.axvspan(5, 11, color="#39ff14", alpha=0.18, lw=0, label="the empty valley")
    ax.axvline(a.k, color="k", lw=1.6, label=f"k = {a.k:g}")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 30)
    ax.set_xlim(0, float(ctr[fin].max()))
    ax.set_xlabel("$q = |y'| / \\sigma_y(x')$", fontsize=8.5)
    ax.set_ylabel("% of $|f|$ per bin", fontsize=8.5)
    ax.set_title("HOW k WAS CHOSEN: the distribution is bimodal", fontsize=8.4,
                 color="#1a6b1a")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.3, frameon=False, loc="lower left")
    lo_i = int(np.searchsorted(edges, 5.0)) - 1
    hi_i = int(np.searchsorted(edges, 11.0)) - 1
    ax.text(0.98, 0.95, f"LES mass in q=[5,11):\n  {100 * Hl[lo_i:hi_i].sum():.4f}%\n"
                        f"then it RISES again",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6, family="monospace")

    # --- does it eat wide footprints? -------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for s_ in order:
        m = sp == s_
        ax.scatter(d["sigma_v"][m], 100 * d["rm_abs"][m], s=5, alpha=0.45, lw=0,
                   color=SPLIT_COLOUR[s_], label=s_)
    z = np.polyfit(d["sigma_v"], 100 * d["rm_abs"], 1)
    xs = np.linspace(d["sigma_v"].min(), d["sigma_v"].max(), 20)
    ax.plot(xs, np.polyval(z, xs), "k-", lw=1.5, label=f"slope {z[0]:+.2f} %/(m/s)")
    hi = d["sigma_v"] > np.percentile(d["sigma_v"], 90)
    lo = d["sigma_v"] < np.percentile(d["sigma_v"], 10)
    ax.set_xlabel("$\\sigma_v$  [m s$^{-1}$]  (wide footprints to the right)", fontsize=8.5)
    ax.set_ylabel("|mass| removed  [% of $|f|$]", fontsize=8.5)
    ax.set_title(f"top/bottom $\\sigma_v$ decile removed: "
                 f"{np.median(d['rm_abs'][hi]) / np.median(d['rm_abs'][lo]):.2f}x",
                 fontsize=8.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.3, frameon=False, markerscale=2)

    # --- what the half-plane missed ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(100 * d["rm_abs"], bins=60, color="#8172b3", ec="k", lw=0.3,
            label="all removed")
    ax.hist(100 * d["rm_up"], bins=60, color="#dd8452", ec="k", lw=0.3,
            label="removed but UPWIND")
    ax.set_yscale("log")
    ax.set_xlabel("removed  [% of $|f|$]", fontsize=8.5)
    ax.set_ylabel("records", fontsize=8.5)
    ax.set_title("orange is what the half-plane could not see", fontsize=8.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.3, frameon=False)
    ax.text(0.97, 0.72, f"upwind, median {100 * np.median(d['rm_up']):.2f}%\n"
                        f"        max    {100 * d['rm_up'].max():.2f}%\n"
                        f"nonzero on {int((d['rm_up'] > 1e-6).sum())}/{d['n']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, family="monospace")

    # --- integral before/after ----------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    for s_ in order:
        m = sp == s_
        ax.scatter(d["e_raw"][m], d["e_cone"][m], s=5, alpha=0.45, lw=0,
                   color=SPLIT_COLOUR[s_])
    lim = [min(d["e_raw"].min(), d["e_cone"].min()) - 0.05,
           max(d["e_raw"].max(), d["e_cone"].max()) + 0.05]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.axhline(0, color="r", lw=0.9, ls="--")
    ax.axvline(0, color="r", lw=0.9, ls=":")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("integral $-$ asymptote, RAW", fontsize=8.5)
    ax.set_ylabel("integral $-$ asymptote, CONE", fontsize=8.5)
    mr, mcx = np.median(np.abs(d["e_raw"])), np.median(np.abs(d["e_cone"]))
    ax.set_title(f"median |error| {mr:.4f} -> {mcx:.4f}  "
                 f"({'improved' if mcx < mr else 'DEGRADED'})", fontsize=8.4)
    ax.tick_params(labelsize=7)
    r = float(np.corrcoef(d["rm_abs"], d["e_raw"])[0, 1])
    ax.text(0.98, 0.03, f"r(removed, raw error) = {r:+.3f}\n"
                        f"not an integral correction",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            family="monospace")

    # --- integral distributions ----------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    bins = np.linspace(0.3, 2.1, 70)
    ax.hist(d["I_raw"], bins=bins, histtype="step", lw=1.5, color="#4c72b0", label="raw")
    ax.hist(d["I_cone"], bins=bins, histtype="step", lw=1.5, color="#c44e52", label="cone")
    ax.axvline(np.median(d["asym"]), color="k", lw=1.4,
               label=f"asymptote (median {np.median(d['asym']):.3f})")
    ax.set_xlabel("footprint integral", fontsize=8.5)
    ax.set_ylabel("records", fontsize=8.5)
    ax.set_title("the cone shifts the distribution down, uniformly", fontsize=8.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.3, frameon=False)

    # --- negative lobe -------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    for s_ in order:
        m = sp == s_
        ax.scatter(100 * d["neg"][m], 100 * d["neg_cone"][m], s=5, alpha=0.45, lw=0,
                   color=SPLIT_COLOUR[s_])
    hi_ = 100 * max(d["neg"].max(), d["neg_cone"].max()) * 1.05
    ax.plot([0, hi_], [0, hi_], "k-", lw=0.8)
    ax.set_xlim(0, hi_); ax.set_ylim(0, hi_)
    ax.set_xlabel("negative lobe, RAW  [% of $|f|$]", fontsize=8.5)
    ax.set_ylabel("negative lobe, CONE", fontsize=8.5)
    ax.set_title(f"median {100 * np.median(d['neg']):.2f}% -> "
                 f"{100 * np.median(d['neg_cone']):.2f}%", fontsize=8.4)
    ax.tick_params(labelsize=7)

    fig.suptitle(
        f"The wind-aligned cone: $|y'| \\leq \\max({a.k:g}\\,\\sigma_y(x'), {a.y_min:g}"
        f"\\,\\mathrm{{m}})$   (n = {d['n']}; top row: {a.case}, wind FROM "
        f"{d['wdir'][int(np.where(d['run_id'] == a.case)[0][0])]:.0f}$\\degree$)\n"
        f"removes a median {100 * np.median(d['rm_abs']):.2f}% of $|f|$, of which "
        f"{100 * np.median(d['rm_up']):.2f}% sits UPWIND where the half-plane could not "
        f"reach it; Kljun loses {100 * d['klj_rm'].max():.5f}%",
        fontsize=12.5, y=0.972)
    fig.text(0.055, 0.020,
             "Green dashed: the cone boundary. Production retires trajectories at ONE "
             "domain length, so no particle can wrap twice — every wrapped particle lands "
             "off-axis or downwind, and the cone catches all of it.\n"
             "Full numbers in results/cone_mask_validation.txt, per record in "
             "results/cone_mask_per_record.tsv.",
             fontsize=7.2, va="bottom", linespacing=1.5)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="w")
    plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
