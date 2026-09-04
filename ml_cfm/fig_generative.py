"""The generative figure: what the CFM's 80 samples look like on one case, and what their
spread means. Row 1: three individual samples, the sample mean (the recipe's field), the LES
target. Row 2: the probability that a cell is inside the 80% source area over the samples,
the 80% source-area contour of every sample, the crosswind-integrated profile with its
50% / 90% sample bands, and the histograms of the array share and of the integral over the
samples with Kljun, FNO and the LES marked.

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

COL = dict(les="#1f4e79", kljun="#c0392b", fno="#2e8b57", cfm="#7b3fa0")


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
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    import fig_corpus_pairs as FCP

    split = D.load_split(a.split, allow_test=a.allow_test)
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    fields, les, samples = RM.recipe_fields(split, valid)
    if a.case:
        i = int(np.where(split.meta["run_id"].astype(str) == a.case)[0][0])
    else:
        north = np.where(split.octant.astype(str) == "N")[0]
        i = int(north[np.argmax(split.meta["array_share"][north])])
    wd = float(split.wdir_deg[i])
    S = samples[:, i]                                        # (80,128,128) m^-2, uncut
    mean, kl, fno, tgt = fields["CFM"][i], fields["Kljun"][i], fields["FNO"][i], les[i]
    dt = str(split.meta["datetime"][i]).replace("T", " ").replace("Z", " UTC")

    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    vmax = float(max(tgt.max(), S.max(), kl.max()))
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    water = st["water"] > 0.5
    x0, x1, y0, y1 = D.ARRAY_XY
    half = 1200.0

    def decorate(ax, title):
        ax.set_facecolor("#f4f4f4")
        ax.contour(xc, xc, water, levels=[0.5], colors="#2a9fd6", linewidths=0.9)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#2ecc40", lw=1.4, zorder=6))
        ax.plot(0, 0, marker="*", ms=8, mfc="w", mec="k", mew=0.6, zorder=7)
        FCP.draw_wind(ax, wd, colour="#444444")
        ax.set_xlim(-half, half); ax.set_ylim(-half, half)
        ax.set_xticks([-1000, 0, 1000]); ax.set_yticks([-1000, 0, 1000]); ax.tick_params(labelsize=7, length=2)
        ax.set_title(title, fontsize=10.5, pad=6)

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(23, 9.6))
    gs = fig.add_gridspec(2, 5, left=0.04, right=0.92, top=0.86, bottom=0.07, wspace=0.16, hspace=0.30)
    # row 1: three samples, the mean, the LES
    for c, (fld, title) in enumerate([(S[0], "sample 1 of 80"), (S[1], "sample 2 of 80"), (S[2], "sample 3 of 80"),
                                      (mean, "CFM: mean of the 80 samples (cut)"), (tgt, "LES target (positive-only)")]):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(np.ma.masked_less_equal(fld, norm.vmin), origin="lower", extent=ext, cmap="magma", norm=norm, interpolation="nearest")
        decorate(ax, title)
        if c > 0:
            ax.set_yticklabels([])
    # row 2, panel 1: P(cell inside the 80% source area) over the samples
    ax = fig.add_subplot(gs[1, 0])
    inside = np.stack([s >= level80(s) for s in S]).mean(0)
    im2 = ax.imshow(np.ma.masked_less(inside, 0.01), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.contour(xc, xc, tgt, levels=[level80(tgt)], colors="w", linewidths=1.4)
    ax.contour(xc, xc, kl, levels=[level80(kl)], colors=COL["kljun"], linewidths=1.1, linestyles="--")
    decorate(ax, "P(cell in the 80% source area) over samples\nwhite: LES 80% area; red dashed: Kljun")
    cb = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=7)
    # panel 2: the 80% contour of every sample
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor("#f4f4f4")
    for s in S:
        ax.contour(xc, xc, s, levels=[level80(s)], colors=[COL["cfm"]], linewidths=0.5, alpha=0.35)
    ax.contour(xc, xc, tgt, levels=[level80(tgt)], colors=[COL["les"]], linewidths=1.8)
    ax.contour(xc, xc, fno, levels=[level80(fno)], colors=[COL["fno"]], linewidths=1.4)
    ax.contour(xc, xc, kl, levels=[level80(kl)], colors=[COL["kljun"]], linewidths=1.2, linestyles="--")
    decorate(ax, "80% source-area contour of each of the 80 samples\nblue: LES; green: FNO; red dashed: Kljun")
    ax.set_yticklabels([])
    # panel 3: crosswind-integrated profile with sample bands
    ax = fig.add_subplot(gs[1, 2])
    prof = np.stack([FCP.crosswind_integrated(s, wd)[1] for s in S]) * 1e3
    s_ = FCP.crosswind_integrated(S[0], wd)[0]
    p5, p25, p50, p75, p95 = np.percentile(prof, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(s_, p5, p95, color=COL["cfm"], alpha=0.15, lw=0, label="CFM 90% of samples")
    ax.fill_between(s_, p25, p75, color=COL["cfm"], alpha=0.30, lw=0, label="CFM 50% of samples")
    ax.plot(s_, p50, color=COL["cfm"], lw=1.3, label="CFM sample median")
    for key, fld, lab in (("les", tgt, "LES target"), ("fno", fno, "FNO"), ("kljun", kl, "Kljun")):
        ax.plot(*np.array(FCP.crosswind_integrated(fld, wd)) * [[1], [1e3]], color=COL[key], lw=1.6 if key == "les" else 1.2, label=lab)
    ax.set_xlim(-100, 1500); ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel("upwind distance from the tower [m]", fontsize=8.5); ax.set_ylabel(r"$f_y$  [10$^{-3}$ m$^{-1}$]", fontsize=8.5)
    ax.tick_params(labelsize=7); ax.legend(fontsize=7.5, frameon=False); ax.set_title("Crosswind-integrated footprint with sample bands", fontsize=10.5, pad=6)
    # panels 4-5: histograms of the array share and the integral over the samples
    tot = S.sum((1, 2)); share = 100 * (S * arr).sum((1, 2)) / tot; integ = tot * D.DX * D.DX
    ref = {"array share [%]": (share, 100 * (tgt * arr).sum() / tgt.sum(), 100 * (kl * arr).sum() / kl.sum(), 100 * (fno * arr).sum() / fno.sum()),
           "integral": (integ, tgt.sum() * 900, kl.sum() * 900, fno.sum() * 900)}
    for c, (lab, (x, r_les, r_kl, r_fno)) in enumerate(ref.items()):
        ax = fig.add_subplot(gs[1, 3 + c])
        ax.hist(x, bins=16, color=COL["cfm"], alpha=0.55, label="80 CFM samples")
        for val, key, name in ((r_les, "les", "LES"), (r_fno, "fno", "FNO"), (r_kl, "kljun", "Kljun"), (float(np.mean(x)), "cfm", "CFM mean")):
            ax.axvline(val, color=COL[key], lw=1.8 if key == "les" else 1.3, ls="-" if key != "cfm" else ":", label=name)
        lo, hi = np.percentile(x, [5, 95])
        ax.set_title(f"{lab} over the samples\n90% of samples in [{lo:.2f}, {hi:.2f}]", fontsize=10.5, pad=6)
        ax.set_xlabel(lab, fontsize=8.5); ax.set_ylabel("samples", fontsize=8.5); ax.tick_params(labelsize=7)
        ax.legend(fontsize=7.5, frameon=False)
    cax = fig.add_axes([0.935, 0.55, 0.010, 0.30])
    cb = fig.colorbar(im, cax=cax); cb.set_label("flux footprint  [m$^{-2}$]", fontsize=9); cb.ax.tick_params(labelsize=7.5)
    year = {"val": "validation year 2024", "test": "test year 2025"}.get(a.split, a.split)
    fig.text(0.04, 0.955, f"The CFM as a generative model: {dt}, wind from {wd:.0f}° ({split.octant[i]}), {year}", fontsize=13, weight="bold", va="center")
    fig.text(0.04, 0.92, "Each sample is one plausible LES footprint for these six scalars. The recipe reports their mean; the spread across samples is the "
             "model's own error bar, shown here as the probability map, the contour cloud, the profile bands and the histograms.",
             fontsize=8.5, va="center", color="#333333")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out, str(split.meta["run_id"][i]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
