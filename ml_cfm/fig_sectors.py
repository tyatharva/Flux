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
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.6))
    for ax, (key, label, how, perfect) in zip(axes.ravel(), PANELS):
        for j, name in enumerate(("Kljun", "FNO", "CFM")):
            xs, ys, lo, hi = [], [], [], []
            for k, (c, m) in enumerate(masks.items()):
                v, l, h = boot(z[f"{name}__{key}"][m], how, rng)
                xs.append(k + (j - 1) * 0.22); ys.append(v); lo.append(v - l); hi.append(h - v)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=5, color=COL[name], ecolor=COL[name], elinewidth=1.2, capsize=3, label=name)
        ax.axhline(perfect, color="k", lw=0.8, ls="--", label="perfect (LES)" if key == "peak_x" else None)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{c}\nn = {int(m.sum())}" for c, m in masks.items()], fontsize=8.5)
        ax.set_title(label, fontsize=10.5); ax.grid(axis="y", alpha=0.25, lw=0.5); ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(-0.6, 3.6)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    year = {"val": "validation year 2024", "test": "test year 2025"}.get(a.split, a.split)
    fig.suptitle(f"Metrics by wind sector, {year}: 90° sectors centred on N, E, S, W; points with 95% record-bootstrap intervals", fontsize=12, y=0.985)
    fig.text(0.5, 0.935, "RMSE over the sector's records for the three errors; mean over records for the five scores. Dashed line = the LES scored against itself.",
             ha="center", fontsize=8.5, color="#333333")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    rose(z, masks, out.replace("sectors_", "sectors_rose_"), a.split)
    return 0


def rose(z, masks, out, split_name):
    """The same numbers as a wind rose: one polar panel per metric, the four sectors as 90-degree
    wedges (N up), three bars per wedge (Kljun / FNO / CFM), bar length = the metric, whiskers =
    the 95% record-bootstrap interval. The radial axis of every panel is zoomed to its own
    values (the inner ring is not zero) and the value is printed at every bar tip."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 4, figsize=(20, 11.2), subplot_kw=dict(projection="polar"))
    plt.subplots_adjust(left=0.03, right=0.98, top=0.82, bottom=0.03, wspace=0.28, hspace=0.42)
    ang = {"N": 0.0, "E": np.pi / 2, "S": np.pi, "W": 3 * np.pi / 2}
    names = ("Kljun", "FNO", "CFM")
    for ax, (key, label, how, perfect) in zip(axes.ravel(), PANELS):
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        est = {(n, c): boot(z[f"{n}__{key}"][m], how, rng) for n in names for c, m in masks.items()}
        vals = np.array([e[0] for e in est.values()]); los = np.array([e[1] for e in est.values()]); his = np.array([e[2] for e in est.values()])
        span = his.max() - los.min()
        r0 = max(0.0, los.min() - 0.35 * span); r1 = his.max() + 0.30 * span
        if perfect == 1:
            r1 = min(1.0, r1)
        for c in masks:
            ax.bar(ang[c], r1 - r0, width=np.pi / 2, bottom=r0, color="#f2f2f2", ec="#dddddd", lw=0.6, zorder=1)
        for j, n in enumerate(names):
            for c in masks:
                v, lo, hi = est[(n, c)]
                th = ang[c] + (j - 1) * 0.40
                ax.bar(th, v - r0, width=0.36, bottom=r0, color=COL[n], alpha=0.9, label=n if c == "N" else None, zorder=3)
                ax.plot([th, th], [lo, hi], color="k", lw=1.1, zorder=4)
                fmt = "{:.0f}" if v >= 20 else ("{:.2f}" if v < 2 else "{:.1f}")
                ax.text(th, hi + 0.04 * span, fmt.format(v), ha="center", va="center", fontsize=7.5, zorder=5,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
        ax.set_rlim(r0, r1)
        ax.set_xticks(list(ang.values())); ax.set_xticklabels([f"{c}\nn = {int(m.sum())}" for c, m in masks.items()], fontsize=9.5)
        ax.set_yticks(np.linspace(r0, r1, 4)[1:-1]); ax.set_yticklabels([])
        ax.grid(alpha=0.4, lw=0.5); ax.spines["polar"].set_linewidth(0.8)
        ax.set_title(f"{label}\nradial axis {r0:.3g} to {r1:.3g}" + ("  (perfect = 1)" if perfect == 1 else "  (perfect = 0)"), fontsize=10.5, pad=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.905))
    year = {"val": "validation year 2024", "test": "test year 2025"}.get(split_name, split_name)
    fig.suptitle(f"Metrics by wind sector, {year}: 90° sectors centred on N, E, S, W\nbar = the metric over the sector's records, "
                 "whisker = 95% record-bootstrap interval, value at the tip; each panel's radial axis is zoomed to its own range", fontsize=12, y=0.985)
    fig.savefig(out, dpi=130); print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
