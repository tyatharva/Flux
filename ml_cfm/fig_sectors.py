"""Metrics by wind direction: one panel per metric, the four cardinal 90-degree sectors on the
x axis, Kljun / FNO / CFM as points with 95% record-bootstrap intervals, the perfect value as
a dashed line, n per sector in the tick labels. Reads the per-record file report_metrics wrote.

    python -m ml_cfm.fig_sectors [--split val]
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml_cfm import final_recipe as FR         # noqa: E402
from ml_cfm import report_metrics as RM       # noqa: E402

COL = dict(Kljun="#c0392b", FNO="#2e8b57", CFM="#7b3fa0")
PANELS = (("peak_x", "peak distance RMSE [m]", "rmse", 0), ("centroid", "centroid RMSE [m]", "rmse", 0), ("integral", "integral RMSE", "rmse", 0),
          ("overlap80", "overlap80 (Jaccard)", "mean", 1), ("rel_l2", "rel. L2", "mean", 0), ("sw1_m", "sliced W1 [m]", "mean", 0),
          ("js_dist", "JS distance [bits]", "mean", 0), ("ms_ssim", "MS-SSIM (log grid)", "mean", 1))


def stat(x, how):
    return float(np.sqrt(np.nanmean(x ** 2))) if how == "rmse" else float(np.nanmean(x))


def boot(x, how, rng, n=2000):
    x = x[np.isfinite(x)]
    b = [stat(x[rng.integers(0, len(x), len(x))], how) for _ in range(n)]
    return stat(x, how), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def sector_masks(wdir):
    return {c: np.abs((wdir - deg + 180) % 360 - 180) <= 45 for c, deg in RM.CARDINAL}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(FR.OUT, f"sectors_{a.split}.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(FR.OUT, f"metrics_{a.split}_per_record.npz"))
    masks = sector_masks(z["wdir_deg"])
    rng = np.random.default_rng(0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.6))
    plt.subplots_adjust(left=0.045, right=0.985, top=0.86, bottom=0.09, wspace=0.28, hspace=0.42)
    for ax, (key, label, how, perfect) in zip(axes.ravel(), PANELS):
        for k in range(4):
            if k % 2 == 0:
                ax.axvspan(k - 0.5, k + 0.5, color="#f3f3f3", zorder=0)
        for j, name in enumerate(("Kljun", "FNO", "CFM")):
            xs, ys, lo, hi = [], [], [], []
            for k, (c, m) in enumerate(masks.items()):
                v, l, h = boot(z[f"{name}__{key}"][m], how, rng)
                xs.append(k + (j - 1) * 0.24); ys.append(v); lo.append(v - l); hi.append(h - v)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=7, color=COL[name], ecolor=COL[name], elinewidth=1.6, capsize=4, capthick=1.4,
                        mec="white", mew=0.8, label=name, zorder=3)
        ax.axhline(perfect, color="k", lw=1.0, ls="--", zorder=2)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{c}\nn = {int(m.sum())}" for c, m in masks.items()], fontsize=10)
        ax.set_title(label, fontsize=12, pad=8); ax.grid(axis="y", alpha=0.3, lw=0.6); ax.tick_params(axis="y", labelsize=9)
        ax.set_xlim(-0.6, 3.6)
        if perfect == 1:
            ax.set_ylim(top=1.0 + 0.02 * (1.0 - ax.get_ylim()[0]))
        else:
            ax.set_ylim(bottom=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], color="k", ls="--", lw=1.0)], l + ["perfect (the LES scored against itself)"],
               loc="upper center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.935))
    year = {"val": "validation year 2024", "test": "test year 2025"}.get(a.split, a.split)
    fig.suptitle(f"Metrics by wind sector, {year}: 90° sectors centred on N, E, S, W.  RMSE over the sector's records for the three errors, "
                 "mean for the five scores; whiskers = 95% record-bootstrap interval.", fontsize=12.5, y=0.985)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
