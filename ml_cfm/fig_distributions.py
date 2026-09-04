"""The per-record distribution of every metric over the whole split as empirical cumulative
distributions: one panel per metric, one curve per model, y = the fraction of records with a
value at or below x. A curve further towards the perfect value is better; the vertical
separation at any x is the difference in the fraction of records that reach it.

    python -m ml_cfm.fig_distributions [--split val]
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
from ml_cfm import fig_sectors as FS          # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(FR.OUT, f"distributions_{a.split}.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(FR.OUT, f"metrics_{a.split}_per_record.npz"))
    n = len(z["run_id"])
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    names = ("Kljun", "FNO", "CFM")
    for ax, (key, label, how, perfect) in zip(axes.ravel(), FS.PANELS):
        xlabel = label.replace(" RMSE", "")
        allv = np.concatenate([z[f"{n}__{key}"][np.isfinite(z[f"{n}__{key}"])] for n in names])
        for name in names:
            d = np.sort(z[f"{name}__{key}"][np.isfinite(z[f"{name}__{key}"])])
            y = np.arange(1, len(d) + 1) / len(d)
            ax.step(d, y, where="post", color=FS.COL[name], lw=2.0, label=f"{name}  ({'RMSE' if how == 'rmse' else 'mean'} {FS.stat(d, how):.3g})")
            ax.plot([np.median(d)], [0.5], marker="o", ms=8, color=FS.COL[name], mec="k", mew=0.6, zorder=5)
        ax.axhline(0.5, color="#999999", lw=0.7, ls=":")
        ax.set_ylim(0, 1); ax.set_ylabel("fraction of records ≤ x", fontsize=9.5)
        if key == "peak_x":
            ax.set_xlim(-5, 185); ax.set_xticks([0, 30, 60, 90, 120, 150, 180]); ax.tick_params(axis="x", labelsize=8.5)
            ax.axvline(perfect, color="k", lw=0.8, ls="--")
            xlabel += "  (30 m steps; cut at 180 m)"
        else:
            lo, hi = np.percentile(allv, [5, 95])
            ax.set_xlim(lo, hi)
            if lo <= perfect <= hi:
                ax.axvline(perfect, color="k", lw=0.8, ls="--")
            else:
                xlabel += f"  (perfect = {perfect}, off the {'left' if perfect < lo else 'right'} edge)"
        ax.set_xlabel(xlabel, fontsize=9)
        ax.tick_params(labelsize=8.5); ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=8.5, frameon=False, loc="lower right" if perfect == 0 else "upper left")
    year = {"val": "validation year 2024", "test": "test year 2025"}.get(a.split, a.split)
    fig.suptitle(f"Every record's score, {year} (n = {n}): cumulative distributions per model; dot = median; dashed = perfect where it is in range; "
                 "every axis is cut to the central 90% of records", fontsize=11.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
