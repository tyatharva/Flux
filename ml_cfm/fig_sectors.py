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
PANELS = (("peak_x", "Peak distance RMSE [m]", "rmse", 0), ("centroid", "Centroid RMSE [m]", "rmse", 0), ("integral", "Integral RMSE", "rmse", 0),
          ("overlap80", "80% source-area overlap (Jaccard)", "mean", 1), ("rel_l2", "Relative L2 error", "mean", 0),
          ("sw1_m", "Sliced Wasserstein-1 distance [m]", "mean", 0), ("js_dist", "Jensen–Shannon distance [bits]", "mean", 0),
          ("ms_ssim", "MS-SSIM on the log grid", "mean", 1))


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
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(FR.OUT, f"sectors_{a.split}.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(FR.OUT, f"metrics_{a.split}_per_record.npz"))
    masks = sector_masks(z["wdir_deg"])
    rng = np.random.default_rng(0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6))
    plt.subplots_adjust(left=0.05, right=0.985, top=0.875, bottom=0.10, wspace=0.30, hspace=0.45)
    for ax, (key, label, how, perfect) in zip(axes.ravel(), PANELS):
        for k in range(4):
            if k % 2 == 0:
                ax.axvspan(k - 0.5, k + 0.5, color="#f3f3f3", zorder=0)
        for j, name in enumerate(("Kljun", "FNO", "CFM")):
            xs, ys, lo, hi = [], [], [], []
            for k, (c, m) in enumerate(masks.items()):
                v, l, h = boot(z[f"{name}__{key}"][m], how, rng)
                xs.append(k + (j - 1) * 0.24); ys.append(v); lo.append(v - l); hi.append(h - v)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=8, color=COL[name], ecolor=COL[name], elinewidth=1.8, capsize=4.5, capthick=1.6,
                        mec="white", mew=0.8, label=name, zorder=3)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{c}\nn = {int(m.sum())}" for c, m in masks.items()], fontsize=11)
        ax.set_title(f"{label}\n(perfect = {perfect})", fontsize=12.5, pad=8); ax.grid(axis="y", alpha=0.3, lw=0.6); ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim(-0.6, 3.6)
        if perfect == 1:
            ax.set_ylim(top=1.0 + 0.02 * (1.0 - ax.get_ylim()[0]))
        else:
            ax.set_ylim(bottom=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.995), markerscale=1.2)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=a.dpi)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
