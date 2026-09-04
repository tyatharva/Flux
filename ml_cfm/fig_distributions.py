"""The per-record distribution of every metric over the whole split: one panel per metric, a
violin per model with the median marked, the perfect value as a dashed line. The table reports
RMSEs and means, which the tail drives; this shows the tail.

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
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.6))
    names = ("Kljun", "FNO", "CFM")
    for ax, (key, label, how, perfect) in zip(axes.ravel(), FS.PANELS):
        data = [z[f"{name}__{key}"][np.isfinite(z[f"{name}__{key}"])] for name in names]
        vp = ax.violinplot(data, positions=range(3), showmedians=True, showextrema=False, widths=0.8)
        for body, name in zip(vp["bodies"], names):
            body.set_facecolor(FS.COL[name]); body.set_alpha(0.55); body.set_edgecolor("none")
        vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(1.2)
        for j, d in enumerate(data):
            ax.scatter(j + (np.random.default_rng(j).uniform(-0.12, 0.12, len(d))), d, s=3, color="k", alpha=0.18, lw=0)
        ax.axhline(perfect, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=9)
        agg = [FS.stat(d, how) for d in data]
        ax.set_title(label.replace(" RMSE", "") + "   " + ("RMSE " if how == "rmse" else "mean ") + " / ".join(f"{v:.3g}" for v in agg), fontsize=9.5)
        ax.grid(axis="y", alpha=0.25, lw=0.5); ax.tick_params(axis="y", labelsize=8)
    year = {"val": "validation year 2024", "test": "test year 2025"}.get(a.split, a.split)
    fig.suptitle(f"Per-record distribution of every metric, {year} (n = {n}): violins with the median (black), every record as a dot, dashed = perfect", fontsize=12, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
