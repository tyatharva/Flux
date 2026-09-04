"""The generative figure, one case, two panels: (a) the 80% source-area outline of every
sample with the LES's and the CFM mean's; (b) the crosswind-integrated footprint with the 90%
sample band against the LES, FNO and Kljun.

    python -m ml_cfm.fig_generative [--split val] [--allow-test] [--case run_id]
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml import data as D                      # noqa: E402
from ml_cfm import final_recipe as FR         # noqa: E402
from ml_cfm import report_metrics as RM       # noqa: E402
from ml_cfm import figstyle as FS             # noqa: E402


def level80(f):
    v = np.sort(np.maximum(f, 0).ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    return float(v[np.searchsorted(c, 0.8)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--case", default=None, help="run_id; default = the strongest-array N record")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="run_ids not to pick")
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.allow_test:
        raise SystemExit("refusing the test split without --allow-test")
    out = a.out or os.path.join(FR.OUT, f"generative_{a.split}.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    import fig_corpus_pairs as FCP

    split = D.load_split(a.split, allow_test=a.allow_test)
    st = D.load_statics()
    valid = split.valid_mask.astype(np.float32)
    fields, les, samples = RM.recipe_fields(split, valid)
    if a.case:
        i = int(np.where(split.meta["run_id"].astype(str) == a.case)[0][0])
    else:
        ok = ~np.isin(split.meta["run_id"].astype(str), list(a.exclude))
        north = np.where((split.octant.astype(str) == "N") & ok)[0]
        i = int(north[np.argmax(split.meta["array_share"][north])])
    wd = float(split.wdir_deg[i])
    S = samples[:, i]
    mean, kl, fno, tgt = fields["CFM"][i], fields["Kljun"][i], fields["FNO"][i], les[i]
    dt = str(split.meta["datetime"][i]).replace("T", " ")[:16]
    half = 1200.0
    xc = (np.arange(D.N) - D.IJ_RECEPTOR) * D.DX
    x0, x1, y0, y1 = D.ARRAY_XY

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    plt.subplots_adjust(left=0.03, right=0.985, top=0.92, bottom=0.11, wspace=0.10)
    # (b) the 80% source-area outline of each sample, with the LES's and the CFM mean's
    ax = axes[0]
    ax.set_facecolor("white")
    FS.background(ax, D.N * D.DX / 2)
    for s in S:
        ax.contour(xc, xc, s, levels=[level80(s)], colors=[FS.COL["cfm"]], linewidths=0.7, alpha=0.30, zorder=4)
    ax.contour(xc, xc, mean, levels=[level80(mean)], colors=["#ff1493"], linewidths=2.8, zorder=6)
    ax.contour(xc, xc, tgt, levels=[level80(tgt)], colors=[FS.COL["les"]], linewidths=2.6, zorder=6)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ff00ff", lw=1.6, zorder=6))
    ax.plot(0, 0, marker="*", ms=11, mfc="w", mec="k", mew=0.8, zorder=7)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Line2D([], [], color=FS.COL["cfm"], lw=0.9, alpha=0.6, label="80% source area of each of the 80 samples"),
                       Line2D([], [], color="#ff1493", lw=2.8, label="80% source area of the CFM mean"),
                       Line2D([], [], color=FS.COL["les"], lw=2.6, label="80% source area of the LES target")],
              fontsize=10.5, loc="lower right", framealpha=0.9, edgecolor="none")
    ax.set_title(f"{dt} UTC, wind from {wd:.0f}° ({split.octant[i]}): 80% source areas", fontsize=12, pad=10)
    # (b) crosswind-integrated footprint with bands
    ax = axes[1]
    prof = np.stack([FCP.crosswind_integrated(s, wd)[1] for s in S]) * 1e3
    s_ = FCP.crosswind_integrated(S[0], wd)[0]
    p5, p25, p50, p75, p95 = np.percentile(prof, [5, 25, 50, 75, 95], axis=0)
    FS.crosswind_panel(ax, [("les", tgt, "LES target"), ("fno", fno, "FNO"), ("kljun", kl, "Kljun")], wd, letter=None,
                       legend=False, bands=(s_, p5, p25, p50, p75, p95), tick_size=10, label_size=11.5)
    ax.plot(s_, FCP.crosswind_integrated(mean, wd)[1] * 1e3, color="#ff1493", lw=1.8, label="CFM mean")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_title("Crosswind-integrated footprint with the 90% band of the 80 samples", fontsize=12, pad=10)
    ax.set_xlabel("Upwind distance from the tower [m]", fontsize=11.5); ax.set_ylabel(r"$f_y$  [10$^{-3}$ m$^{-1}$]", fontsize=11.5)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=a.dpi)
    print("wrote", out, str(split.meta["run_id"][i]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
