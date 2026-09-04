"""The generative figure, one case, three panels: (a) the CFM field the recipe reports, the
mean of 80 samples; (b) the probability that a cell lies inside the 80% source area across the
80 samples, with the LES and FNO 80% areas drawn on it; (c) the crosswind-integrated footprint
with the 50% and 90% sample bands against the LES, FNO and Kljun.

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
        north = np.where(split.octant.astype(str) == "N")[0]
        i = int(north[np.argmax(split.meta["array_share"][north])])
    wd = float(split.wdir_deg[i])
    S = samples[:, i]
    mean, kl, fno, tgt = fields["CFM"][i], fields["Kljun"][i], fields["FNO"][i], les[i]
    dt = str(split.meta["datetime"][i]).replace("T", " ")[:16]
    half = 1200.0
    xc = (np.arange(D.N) - D.IJ_RECEPTOR) * D.DX
    x0, x1, y0, y1 = D.ARRAY_XY

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.4))
    plt.subplots_adjust(left=0.03, right=0.985, top=0.88, bottom=0.16, wspace=0.12)
    # (a) the mean
    ax = axes[0]
    m = FS.footprint_panel(ax, mean, float(max(tgt.max(), mean.max())), st, wd, letter="a", half=half)
    ax.set_xticks([-1000, 0, 1000]); ax.set_yticks([-1000, 0, 1000])
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03)
    lv = FS.levels(float(max(tgt.max(), mean.max())))
    ticks = np.arange(np.ceil(lv[0]), np.floor(lv[-1]) + 0.5)
    cb.set_ticks(ticks); cb.set_ticklabels([f"$10^{{{int(t)}}}$" for t in ticks]); cb.set_label("flux footprint [m$^{-2}$]", fontsize=10)
    ax.set_title("CFM footprint: the mean of 80 samples", fontsize=12.5, pad=10)
    # (b) P(inside the 80% source area)
    ax = axes[1]
    inside = np.stack([s >= level80(s) for s in S]).mean(0)
    ax.set_facecolor("#ececec")
    im = ax.imshow(np.ma.masked_less(inside, 0.005), origin="lower", extent=[xc[0] - 15, xc[-1] + 15] * 2, cmap="viridis", vmin=0, vmax=1, interpolation="nearest", zorder=2)
    ax.contour(xc, xc, st["water"] > 0.5, levels=[0.5], colors="#5fd0ff", linewidths=1.0, zorder=3)
    ax.contour(xc, xc, tgt, levels=[level80(tgt)], colors="w", linewidths=2.2, zorder=5)
    ax.contour(xc, xc, fno, levels=[level80(fno)], colors=[FS.COL["fno"]], linewidths=1.8, zorder=5)
    ax.contour(xc, xc, kl, levels=[level80(kl)], colors=[FS.COL["kljun"]], linewidths=1.6, linestyles="--", zorder=5)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#39ff14", lw=1.5, zorder=6))
    ax.plot(0, 0, marker="*", ms=10, mfc="w", mec="k", mew=0.7, zorder=7)
    FCP.draw_wind(ax, wd, colour="k")
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1000, 0, 1000]); ax.set_yticks([-1000, 0, 1000])
    ax.set_xticklabels(["-1 km", "0", "+1 km"]); ax.set_yticklabels(["-1 km", "0", "+1 km"])
    ax.tick_params(axis="x", direction="in", labeltop=True, labelbottom=False, top=True, pad=-14, labelsize=8.5, length=3)
    ax.tick_params(axis="y", direction="in", pad=-30, labelsize=8.5, length=3)
    for lab in ax.get_yticklabels():
        lab.set_ha("left")
    ax.grid(color="w", alpha=0.45, lw=0.5, zorder=4)
    ax.text(0.03, 0.965, "b", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left", zorder=8)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03)
    cb.set_label("fraction of the 80 samples whose 80% source area contains the cell", fontsize=10)
    ax.legend(handles=[Line2D([], [], color="w", lw=2.2, label="LES 80% source area"), Line2D([], [], color=FS.COL["fno"], lw=1.8, label="FNO 80% source area"),
                       Line2D([], [], color=FS.COL["kljun"], lw=1.6, ls="--", label="Kljun 80% source area")],
              fontsize=8.5, loc="lower right", facecolor="#444444", labelcolor="w", framealpha=0.85, edgecolor="none")
    ax.set_title("Where the samples put the 80% source area", fontsize=12.5, pad=10)
    # (c) crosswind-integrated footprint with bands
    ax = axes[2]
    prof = np.stack([FCP.crosswind_integrated(s, wd)[1] for s in S]) * 1e3
    s_ = FCP.crosswind_integrated(S[0], wd)[0]
    p5, p25, p50, p75, p95 = np.percentile(prof, [5, 25, 50, 75, 95], axis=0)
    FS.crosswind_panel(ax, [("les", tgt, "LES target"), ("fno", fno, "FNO"), ("kljun", kl, "Kljun")], wd, letter="c",
                       legend=False, bands=(s_, p5, p25, p50, p75, p95))
    ax.plot(s_, p50, color=FS.COL["cfm"], lw=1.6, label="CFM: sample median")
    ax.set_box_aspect(None); ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title("Crosswind-integrated footprint with the sample bands", fontsize=12.5, pad=10)
    ax.set_xlabel("upwind distance from the tower [m]", fontsize=10); ax.set_ylabel(r"$f_y$  [10$^{-3}$ m$^{-1}$]", fontsize=10)
    year = {"val": "validation year 2024", "test": "test year 2025"}.get(a.split, a.split)
    fig.suptitle(f"The CFM's 80 samples for one case: {dt} UTC, wind from {wd:.0f}° ({split.octant[i]}), {year}. "
                 "Each sample is one plausible LES footprint; their spread is the model's own error bar.", fontsize=12, y=0.97)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out, str(split.meta["run_id"][i]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
