#!/usr/bin/env python3
"""The wraparound-mask validation, as a figure. Nine panels, one conclusion.

bin/mask_wrap.py measures whether the downwind mass in the target is periodic-wrap
double-counting. It concludes that it is not -- not because there is nothing downwind, but
because what is there does not behave like the thing that inflates the integral. This draws
the evidence for that, so the conclusion can be checked by eye rather than taken on the
strength of a correlation coefficient in a text file.

The decisive panel is the middle one of row 2. If downwind mass were wrap double-counting,
records that lose the most of it would be the records whose integral was most inflated --
an UPWARD slope. It slopes down.

usage: fig_wrap_mask.py [--h5 corpus/corpus.h5] [--out figures/wrap_mask_effect.png]
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
import mask_wrap as mw                                    # noqa: E402
import fig_corpus_pairs as fcp                            # noqa: E402

SPLIT_COLOUR = fcp.SPLIT_COLOUR


def case_row(fig, gs, h5, d, run_id, surf):
    """Raw target, masked target, and what the mask deleted -- for one named record."""
    import h5py
    i = int(np.where(d["run_id"] == run_id)[0][0])
    with h5py.File(h5, "r") as f:
        t = f["target"][i].astype(np.float64)
        tm = f["target_masked"][i].astype(np.float64)
        sc = f["scalars"][i]
    X, Y = mw.axis_grids()
    s = mw.upwind_projection(X, Y, float(sc[4]), float(sc[5]))
    removed = np.where(s < 0, t, 0.0)
    xc, xe = fcp.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    vmax = float(np.abs(t).max())
    lognorm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    symnorm = SymLogNorm(linthresh=vmax * 1e-4, vmin=-vmax, vmax=vmax, base=10)

    axs = []
    panels = [(t, "target (raw)", lognorm, "magma", lognorm.vmin, "w"),
              (tm, "target_masked", lognorm, "magma", lognorm.vmin, "w"),
              (removed, "what the mask DELETED", symnorm, "RdBu_r", None, "k")]
    for c, (F, name, norm, cmap, floor, fg) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        axs.append(ax)                    # keep the PANEL axes: fig.axes also holds the
        im = fcp.raster(ax, F, norm, cmap, ext, mask_below=floor)  # colourbars
        fcp.draw_frame(ax, surf, fg=fg)
        fcp.draw_wind(ax, float(d["wdir"][i]), colour=fg)
        # the mask boundary: the crosswind line through the receptor
        a = np.radians(float(d["wdir"][i]))
        L = 2600.0
        ax.plot([-L * np.cos(a), L * np.cos(a)], [L * np.sin(a), -L * np.sin(a)],
                color="#39ff14", lw=1.1, ls="--", zorder=8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("east  [m]", fontsize=8)
        if c == 0:
            ax.set_ylabel("north  [m]", fontsize=8)
        ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(xe[0], xe[-1])
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6.5)
    tot_abs = d["pos_removed"][i] - d["neg_removed"][i]
    fcp.stamp(axs[0],
              f"integral {d['I_raw'][i]:.3f}\nasympt.  {d['asym'][i]:.3f}", loc="upper left")
    fcp.stamp(axs[1],
              f"integral {d['I_mask'][i]:.3f}\nchange   {d['I_mask'][i] - d['I_raw'][i]:+.3f}",
              loc="upper left")
    axs[2].text(0.02, 0.98,
                      f"|mass| removed {100 * d['rm_abs'][i]:.1f}% of |f|\n"
                      f"net removed    {d['pos_removed'][i] + d['neg_removed'][i]:+.3f}\n"
                      f"positive share {d['pos_removed'][i] / max(tot_abs, 1e-30):.2f}",
                      transform=axs[2].transAxes, va="top", fontsize=6.6,
                      family="monospace",
                      bbox=dict(fc="w", ec="none", alpha=0.75, boxstyle="round,pad=0.25"))
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="corpus/corpus.h5")
    ap.add_argument("--out", default="figures/wrap_mask_effect.png")
    ap.add_argument("--surface", default="data/grid30_raised")
    ap.add_argument("--zm", type=float, default=30.0)
    ap.add_argument("--near-m", type=float, default=200.0)
    ap.add_argument("--case", default="case_2022020316")
    ap.add_argument("--dpi", type=int, default=130)
    a = ap.parse_args()

    print(f"measuring {a.h5} ...")
    d = mw.measure(a.h5, a.zm, a.near_m)
    surf = fcp.load_surface(a.surface)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    fig = plt.figure(figsize=(14.4, 13.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.30, 1.0, 1.0], hspace=0.34, wspace=0.26,
                          left=0.055, right=0.965, top=0.905, bottom=0.075)
    i = case_row(fig, gs, a.h5, d, a.case, surf)

    order = ["train", "val", "test"]
    sp = d["split"]

    # --- error before vs after --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for s_ in order:
        m = sp == s_
        ax.scatter(d["e_raw"][m], d["e_mask"][m], s=5, alpha=0.45, lw=0,
                   color=SPLIT_COLOUR[s_], label=s_)
    lim = [min(d["e_raw"].min(), d["e_mask"].min()) - 0.05,
           max(d["e_raw"].max(), d["e_mask"].max()) + 0.05]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.axhline(0, color="r", lw=0.9, ls="--")
    ax.axvline(0, color="r", lw=0.9, ls=":")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("integral $-$ asymptote, RAW", fontsize=8.5)
    ax.set_ylabel("integral $-$ asymptote, MASKED", fontsize=8.5)
    ax.set_title("the mask shifts everything down by about the same amount", fontsize=8.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.4, frameon=False, markerscale=2, loc="upper left")
    closer = np.abs(d["e_mask"]) < np.abs(d["e_raw"])
    ax.text(0.98, 0.03, f"closer to the asymptote: {100 * closer.mean():.1f}%\n"
                        f"median |err| {np.median(np.abs(d['e_raw'])):.3f} -> "
                        f"{np.median(np.abs(d['e_mask'])):.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
            family="monospace")

    # --- THE DECISIVE PANEL -----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for s_ in order:
        m = sp == s_
        ax.scatter(100 * d["rm_abs"][m], d["e_raw"][m], s=5, alpha=0.45, lw=0,
                   color=SPLIT_COLOUR[s_])
    r = float(np.corrcoef(d["rm_abs"], d["e_raw"])[0, 1])
    z = np.polyfit(100 * d["rm_abs"], d["e_raw"], 1)
    xs = np.linspace(0, 100 * d["rm_abs"].max(), 20)
    ax.plot(xs, np.polyval(z, xs), "k-", lw=1.6, label=f"fit, slope {z[0]:+.4f}/%")
    ax.axhline(0, color="r", lw=0.9, ls="--")
    ax.annotate("wrap double-counting\nwould slope THIS way",
                xy=(0.62, 0.90), xytext=(0.30, 0.60), xycoords="axes fraction",
                textcoords="axes fraction", fontsize=6.8, color="0.35",
                arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.0))
    ax.set_xlabel("|mass| removed by the mask  [% of $|f|$]", fontsize=8.5)
    ax.set_ylabel("integral $-$ asymptote, RAW", fontsize=8.5)
    ax.set_title(f"DECISIVE: r = {r:+.3f}, and the SIGN is wrong", fontsize=8.4,
                 color="#b22222")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper right")

    # --- integral distributions --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(0.3, 2.1, 70)
    ax.hist(d["I_raw"], bins=bins, histtype="step", lw=1.5, color="#4c72b0", label="raw")
    ax.hist(d["I_mask"], bins=bins, histtype="step", lw=1.5, color="#c44e52",
            label="masked")
    ax.axvline(np.median(d["asym"]), color="k", lw=1.4,
               label=f"asymptote (median {np.median(d['asym']):.3f})")
    ax.set_xlabel("footprint integral", fontsize=8.5)
    ax.set_ylabel("records", fontsize=8.5)
    ax.set_title("masking overshoots: half the corpus lands below", fontsize=8.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.4, frameon=False)
    ax.text(0.02, 0.72, f"below asymptote\n  raw    {int((d['I_raw'] < d['asym']).sum())}"
                        f"\n  masked {int((d['I_mask'] < d['asym']).sum())}",
            transform=ax.transAxes, va="top", fontsize=6.6, family="monospace")

    # --- how much is removed -----------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(100 * d["rm_abs"], bins=60, color="#8172b3", ec="k", lw=0.3, label="|mass|")
    ax.hist(100 * d["rm_near"], bins=60, color="#dd8452", ec="k", lw=0.3,
            label=f"within {a.near_m:.0f} m")
    ax.set_yscale("log")
    ax.set_xlabel("removed  [% of $|f|$]", fontsize=8.5)
    ax.set_ylabel("records", fontsize=8.5)
    ax.set_title("what is removed, and how little of it is near field", fontsize=8.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.4, frameon=False)

    # --- negative lobe ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    for s_ in order:
        m = sp == s_
        ax.scatter(100 * d["neg"][m], 100 * d["neg_mask"][m], s=5, alpha=0.45, lw=0,
                   color=SPLIT_COLOUR[s_])
    hi = 100 * max(d["neg"].max(), d["neg_mask"].max()) * 1.05
    ax.plot([0, hi], [0, hi], "k-", lw=0.8)
    ax.plot([0, hi], [0, hi / 2], "r--", lw=0.8, label="half survives")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("negative lobe, RAW  [% of $|f|$]", fontsize=8.5)
    ax.set_ylabel("negative lobe, MASKED", fontsize=8.5)
    ax.set_title(f"median {100 * np.median(d['neg']):.2f}% -> "
                 f"{100 * np.median(d['neg_mask']):.2f}%", fontsize=8.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left")

    # --- the Steinfeld side test ---------------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    nr = d["neg_right"][np.isfinite(d["neg_right"])]
    ax.hist(nr, bins=np.linspace(0, 1, 51), color="#55a868", ec="k", lw=0.3)
    ax.axvline(0.5, color="k", lw=1.4)
    ax.axvline(float(np.median(nr)), color="r", lw=1.4, ls="--",
               label=f"median {np.median(nr):.3f}")
    ax.set_xlabel("surviving negative mass on the RIGHT of the upwind axis", fontsize=8.5)
    ax.set_ylabel("records", fontsize=8.5)
    ax.set_title("Steinfeld predicts a right-hand majority; there is none", fontsize=8.4)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.4, frameon=False)

    fig.suptitle(
        f"Does removing the downwind half-plane fix the footprint integral?   "
        f"No.   (n = {d['n']}; top row: {a.case})\n"
        f"masking removes a median {100 * np.median(d['rm_abs']):.1f}% of $|f|$ and "
        f"{np.median(d['I_raw'] - d['I_mask']):.3f} of the integral from every record "
        f"alike, and r(removed, excess) = {r:+.3f}",
        fontsize=12.5, y=0.972)
    fig.text(0.055, 0.020,
             "Green dashed line: the mask boundary, the crosswind line through the "
             "receptor. Everything on its downwind side is set to zero.  "
             "Full numbers in results/wrap_mask_validation.txt, per record in "
             "results/wrap_mask_per_record.tsv.",
             fontsize=7.2, va="bottom")
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="w")
    plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
