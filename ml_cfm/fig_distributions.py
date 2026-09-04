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
    LOGX = ("centroid", "integral", "rel_l2", "sw1_m", "js_dist")
    for ax, (key, label, how, perfect) in zip(axes.ravel(), FS.PANELS):
        xlabel = label.replace(" RMSE", "")
        for name in names:
            raw = np.sort(z[f"{name}__{key}"][np.isfinite(z[f"{name}__{key}"])])
            d = 1 - raw[::-1] if key == "ms_ssim" else raw          # MS-SSIM drawn as 1 - MS-SSIM (log), perfect at the left
            y = np.arange(1, len(d) + 1) / len(d)
            ax.step(d, y, where="post", color=FS.COL[name], lw=1.8, label=f"{name}  ({'RMSE' if how == 'rmse' else 'mean'} {FS.stat(raw, how):.3g})")
            ax.plot([np.median(d)], [0.5], marker="o", ms=5, color=FS.COL[name], mec="k", mew=0.5, zorder=5)
        ax.axhline(0.5, color="#999999", lw=0.6, ls=":")
        ax.set_ylim(0, 1); ax.set_ylabel("fraction of records ≤ x", fontsize=9)
        if key in LOGX:
            allv = np.concatenate([z[f"{n}__{key}"][np.isfinite(z[f"{n}__{key}"])] for n in names])
            allv = allv[allv > 0]
            ax.set_xscale("log"); ax.set_xlim(allv.min() * 0.8, allv.max() * 1.25)
            xlabel += "  (log axis; perfect = 0, off the left edge)"
        elif key == "ms_ssim":
            ax.set_xscale("log"); xlabel = "1 − MS-SSIM  (log axis; MS-SSIM on the ticks)"
            tk = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
            ax.set_xticks(tk); ax.set_xticklabels([f"{1 - t:g}" for t in tk]); ax.set_xlim(0.004, 0.25)
        elif key == "overlap80":
            ax.set_xlim(0.25, 0.9); ax.axvline(perfect, color="k", lw=0.8, ls="--")
            xlabel += "  (perfect = 1, off the right edge)"
        else:
            ax.set_xlim(-5, 305); ax.set_xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
            ax.set_xticklabels(["0", "30", "60", "90", "", "150", "", "210", "", "270", ""], fontsize=8)
            ax.axvline(perfect, color="k", lw=0.8, ls="--")
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.tick_params(labelsize=8); ax.grid(alpha=0.3, lw=0.5)
        ax.legend(fontsize=8, frameon=False, loc="lower right" if key != "overlap80" else "upper left")
    year = {"val": "validation year 2024", "test": "test year 2025"}.get(a.split, a.split)
    fig.suptitle(f"Every record's score, {year} (n = {n}): cumulative distributions per model; dot = median; dashed = perfect where it is in range; "
                 "peak errors are multiples of the 30 m cell; error axes are logarithmic", fontsize=11.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
