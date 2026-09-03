"""The showcase figure: five val cases from five wind directions, one per row, labelled by
datetime; columns Kljun | FNO | CFM | LES | crosswind-integrated footprint; the frozen recipe's
fields (ml_cfm/final_recipe.py); one global log colour scale; the relative L2 error against
the LES in the corner of every model panel.

    python -m ml_cfm.fig_showcase [--out results/ml_cfm/final_recipe/showcase_val.png]
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
from ml import metrics as M                   # noqa: E402
from ml_cfm import tailthresh as TT           # noqa: E402
from ml_cfm import final_recipe as FR         # noqa: E402

OCTS = ("N", "NE", "NW", "W", "SW")
COL = dict(les="#1f4e79", kljun="#c0392b", fno="#2e8b57", cfm="#7b3fa0")


def pick_cases(split):
    oc = split.octant.astype(str)
    integ = split.meta["integral"]
    rows = []
    for o in OCTS:
        idx = np.where(oc == o)[0]
        if o == "N":
            rows.append(int(idx[np.argmax(split.meta["array_share"][idx])]))
        else:
            rows.append(int(idx[np.argmin(np.abs(integ[idx] - np.median(integ[idx])))]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(FR.OUT, "showcase_val.png"))
    ap.add_argument("--half", type=float, default=1830.0, help="half-width shown [m]; 1830 = full domain")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    import fig_corpus_pairs as FCP

    split = D.load_split("val")
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    R = FR.RECIPE
    samples = FR.cfm_samples(split, R["cfm_seeds"], R["samples_per_seed"], R["tau"], valid)
    cfm, _ = TT.threshold_stack(samples.mean(0), R["cut_frac"])
    fno_raw = np.mean([np.load(os.path.join(REPO, "results", "ml", "final", sd, "pred_val.npz"))["fno"] for sd in R["fno_seeds"]], axis=0).astype(np.float32)
    fno, _ = TT.threshold_stack(fno_raw, R["cut_frac"])
    les = np.maximum(split.target, 0).astype(np.float32)
    kl = split.kljun
    rows = pick_cases(split)
    sc = M.score_fields({"kljun": kl[rows], "fno": fno[rows], "cfm": cfm[rows]}, les[rows], split.wdir_deg[rows], arr, split.asymptote[rows])
    share = lambda f: 100 * D.raster_array_share(f, arr)

    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    vmax = float(max(les[rows].max(), kl[rows].max(), cfm[rows].max(), fno[rows].max()))
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    water = st["water"] > 0.5
    x0, x1, y0, y1 = D.ARRAY_XY

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "regular"})
    fig = plt.figure(figsize=(21.5, 4.3 * len(rows) + 1.6))
    gs = fig.add_gridspec(len(rows), 5, left=0.055, right=0.895, top=0.895, bottom=0.04, wspace=0.08, hspace=0.24, width_ratios=[1, 1, 1, 1, 1.25])
    heads = ["Kljun et al. (2015)", "FNO emulator", "CFM emulator", "LES target", "Crosswind-integrated footprint"]
    for r, i in enumerate(rows):
        wd = float(split.wdir_deg[i])
        dt = str(split.meta["datetime"][i]).replace("T", " ").replace("Z", " UTC")
        panels = [("kljun", kl[i]), ("fno", fno[i]), ("cfm", cfm[i]), ("les", les[i])]
        for c, (key, fld) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(np.ma.masked_less_equal(fld, norm.vmin), origin="lower", extent=ext, cmap="magma", norm=norm, interpolation="nearest")
            ax.set_facecolor("#f4f4f4")
            ax.contour(xc, xc, water, levels=[0.5], colors="#2a9fd6", linewidths=0.9)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#2ecc40", lw=1.4, zorder=6))
            ax.plot(0, 0, marker="*", ms=8, mfc="w", mec="k", mew=0.6, zorder=7)
            FCP.draw_wind(ax, wd, colour="#444444")
            ax.set_xlim(-a.half, a.half); ax.set_ylim(-a.half, a.half)
            ax.set_xticks([-1500, 0, 1500]); ax.set_yticks([-1500, 0, 1500])
            ax.tick_params(labelsize=7, length=2)
            if r == 0:
                ax.set_title(heads[c], fontsize=11, pad=8)
            if c == 0:
                ax.set_ylabel(f"{dt}\nwind from {wd:.0f}° ({split.octant[i]})   z/L {split.zL[i]:.2f}   $z_i$ {split.scalars[i, 0]:.0f} m",
                              fontsize=8.5, labelpad=8)
            else:
                ax.set_yticklabels([])
            if r < len(rows) - 1:
                ax.set_xticklabels([])
            ax.text(0.03, 0.03, f"array share {share(fld):.1f}%", transform=ax.transAxes, fontsize=8, color="k",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85), va="bottom", ha="left")
            if key != "les":
                ax.text(0.97, 0.97, f"rel. L2 = {sc[key]['rel_l2'][r]:.3f}", transform=ax.transAxes, fontsize=8.5, color="k",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), va="top", ha="right")
        ax = fig.add_subplot(gs[r, 4])
        for key, fld, lab in (("les", les[i], "LES target"), ("kljun", kl[i], "Kljun"), ("fno", fno[i], "FNO"), ("cfm", cfm[i], "CFM")):
            s_, fy = FCP.crosswind_integrated(fld, wd)
            ax.plot(s_, fy * 1e3, color=COL[key], lw=1.6 if key == "les" else 1.3, label=lab, zorder=5 if key == "les" else 4)
        p5, p95 = np.percentile(np.stack([FCP.crosswind_integrated(s, wd)[1] for s in samples[:, i]]), [5, 95], axis=0)
        ax.fill_between(s_, p5 * 1e3, p95 * 1e3, color=COL["cfm"], alpha=0.18, lw=0, label="CFM 90% sample band")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(-100, 1500); ax.tick_params(labelsize=7, length=2)
        ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
        ax.set_ylabel(r"$f_y$  [10$^{-3}$ m$^{-1}$]", fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        if r == 0:
            ax.set_title(heads[4], fontsize=11, pad=8)
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        if r == len(rows) - 1:
            ax.set_xlabel("upwind distance from the tower [m]", fontsize=8.5)
        else:
            ax.set_xticklabels([])
    cax = fig.add_axes([0.935, 0.25, 0.011, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("flux footprint  [m$^{-2}$]", fontsize=9)
    cb.ax.tick_params(labelsize=7.5)
    fig.text(0.055, 0.962, "Kegonsa Solar Array tower, validation year 2024: flux footprints from Kljun, the FNO emulator, the CFM emulator, and the LES target",
             fontsize=13, weight="bold", va="center")
    sub = ("Receptor 30 m. Maps 3.66 x 3.66 km, north up, one logarithmic colour scale (floor 10$^{-4}$ of the peak). "
           "White star = tower; green rectangle = solar array; blue = Lake Kegonsa shoreline; arrow = direction the air moves.\n"
           "FNO: mean of 4 seeds. CFM: mean of 80 samples from 4 seeds, spread scaled by 1.19; the shaded band is the 5-95% range of those samples. "
           "Both emulators are cut at their own 99.5% source area; the LES target is shown positive-only.\n"
           "rel. L2 = ||model $-$ LES|| / ||LES|| over the map, lower is better; two LES realisations of the same case differ by 0.40.")
    fig.text(0.055, 0.933, sub, fontsize=8.5, va="center", color="#333333", linespacing=1.5)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out, [str(split.meta["run_id"][i]) for i in rows])
    return 0


if __name__ == "__main__":
    sys.exit(main())
