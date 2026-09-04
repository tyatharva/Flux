"""The showcase figure: four cases, each from a different month and a different cardinal wind
sector, one per row, labelled by datetime; columns Kljun | FNO | CFM | LES | the eight metrics
of that case | crosswind-integrated footprint; the frozen recipe's fields
(ml_cfm/final_recipe.py); one global log colour scale.

    python -m ml_cfm.fig_showcase [--split val] [--allow-test]
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
from ml_cfm import final_recipe as FR         # noqa: E402
from ml_cfm import report_metrics as RM       # noqa: E402

SECTORS = ("N", "W", "S", "E")
COL = dict(les="#1f4e79", kljun="#c0392b", fno="#2e8b57", cfm="#7b3fa0")
ROWS = (("peak_x", "peak distance error [m]", "{:.0f}", False), ("centroid", "centroid error [m]", "{:.0f}", False),
        ("integral", "integral error", "{:.3f}", False), ("overlap80", "overlap80 (Jaccard)", "{:.3f}", True),
        ("rel_l2", "rel. L2", "{:.3f}", False), ("sw1_m", "sliced W1 [m]", "{:.0f}", False),
        ("js_dist", "JS distance [bits]", "{:.3f}", False), ("ms_ssim", "MS-SSIM (log)", "{:.3f}", True))


def pick_cases(split):
    """One record per cardinal sector, months all different: the strongest-array record for N,
    the median-integral record otherwise."""
    wd = split.wdir_deg
    months = np.array([str(d)[:7] for d in split.meta["datetime"]])
    sect = {c: np.abs((wd - deg + 180) % 360 - 180) <= 45 for c, deg in RM.CARDINAL}
    used, rows = set(), []
    for c in SECTORS:
        idx = np.where(sect[c] & ~np.isin(months, list(used)))[0]
        if c == "N":
            i = idx[np.argmax(split.meta["array_share"][idx])]
        else:
            integ = split.meta["integral"][idx]
            i = idx[np.argmin(np.abs(integ - np.median(integ)))]
        rows.append(int(i)); used.add(months[i])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--half", type=float, default=1830.0, help="half-width shown [m]; 1830 = full domain")
    a = ap.parse_args(argv)
    if a.split == "test" and not a.allow_test:
        raise SystemExit("refusing the test split without --allow-test")
    out = a.out or os.path.join(FR.OUT, f"showcase_{a.split}.png")
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
    v = RM._interior()
    xc0 = (np.arange(D.N) - D.IJ_RECEPTOR) * D.DX
    X, Y = np.meshgrid(xc0, xc0)
    fields, les, _ = RM.recipe_fields(split, valid)
    kl, fno, cfm = fields["Kljun"], fields["FNO"], fields["CFM"]
    rows = pick_cases(split)
    sc = M.score_fields({k: f[rows] for k, f in fields.items()}, les[rows], split.wdir_deg[rows], arr, split.asymptote[rows])
    fm = {k: [RM.field_metrics(f[i], les[i], v, X, Y) for i in rows] for k, f in fields.items()}

    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    vmax = float(max(les[rows].max(), kl[rows].max(), cfm[rows].max(), fno[rows].max()))
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    water = st["water"] > 0.5
    x0, x1, y0, y1 = D.ARRAY_XY

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "regular"})
    fig = plt.figure(figsize=(24.5, 4.3 * len(rows) + 1.6))
    gs = fig.add_gridspec(len(rows), 6, left=0.05, right=0.905, top=0.885, bottom=0.045, wspace=0.08, hspace=0.24,
                          width_ratios=[1, 1, 1, 1, 1.05, 1.25])
    heads = ["Kljun et al. (2015)", "FNO emulator", "CFM emulator", "LES target", "Metrics for this case", "Crosswind-integrated footprint"]
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
        # the metrics of this case
        ax = fig.add_subplot(gs[r, 4]); ax.axis("off")
        cells, colours = [], []
        for key, label, fmt, hi in ROWS:
            vals = [(sc[k][key][r] if key in sc[k] else fm[k][r][key]) for k in ("Kljun", "FNO", "CFM")]
            best = int(np.nanargmax(vals) if hi else np.nanargmin(vals))
            cells.append([label] + [fmt.format(x) for x in vals])
            colours.append(["white"] + ["#e8f4e8" if j == best else "white" for j in range(3)])
        tb = ax.table(cellText=cells, colLabels=["", "Kljun", "FNO", "CFM"], cellColours=colours, loc="center", cellLoc="center",
                      colWidths=[0.46, 0.18, 0.18, 0.18])
        tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.32)
        for (ri, ci), cell in tb.get_celld().items():
            cell.set_edgecolor("#cccccc"); cell.set_linewidth(0.6)
            if ci == 0:
                cell.get_text().set_ha("left"); cell.PAD = 0.03
            if ri == 0:
                cell.set_text_props(weight="bold")
        if r == 0:
            ax.set_title(heads[4], fontsize=11, pad=8)
        # the crosswind-integrated footprint
        ax = fig.add_subplot(gs[r, 5])
        for key, fld, lab in (("les", les[i], "LES target"), ("kljun", kl[i], "Kljun"), ("fno", fno[i], "FNO"), ("cfm", cfm[i], "CFM")):
            s_, fy = FCP.crosswind_integrated(fld, wd)
            ax.plot(s_, fy * 1e3, color=COL[key], lw=1.6 if key == "les" else 1.3, label=lab, zorder=5 if key == "les" else 4)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(-100, 1500); ax.tick_params(labelsize=7, length=2)
        ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
        ax.set_ylabel(r"$f_y$  [10$^{-3}$ m$^{-1}$]", fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        if r == 0:
            ax.set_title(heads[5], fontsize=11, pad=8)
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        if r == len(rows) - 1:
            ax.set_xlabel("upwind distance from the tower [m]", fontsize=8.5)
        else:
            ax.set_xticklabels([])
    cax = fig.add_axes([0.945, 0.25, 0.010, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("flux footprint  [m$^{-2}$]", fontsize=9)
    cb.ax.tick_params(labelsize=7.5)
    year = {"val": "validation year 2024", "test": "test year 2025"}.get(a.split, a.split)
    fig.text(0.05, 0.958, f"Kegonsa Solar Array tower, {year}: flux footprints from Kljun, the FNO emulator, the CFM emulator, and the LES target",
             fontsize=13, weight="bold", va="center")
    sub = ("Receptor 30 m. Maps 3.66 x 3.66 km, north up, one logarithmic colour scale (floor 10$^{-4}$ of the peak). "
           "White star = tower; green rectangle = solar array; blue = Lake Kegonsa shoreline; arrow = direction the air moves.\n"
           "FNO: mean of 4 seeds. CFM: mean of 80 samples from 4 seeds. Both emulators are cut at their own 99.5% source area; the LES target is shown positive-only. "
           "Metrics: errors against the LES (0 = perfect) except overlap80 and MS-SSIM (1 = perfect); shaded = best of the three.")
    fig.text(0.05, 0.925, sub, fontsize=8.5, va="center", color="#333333", linespacing=1.5)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out, [str(split.meta["run_id"][i]) for i in rows])
    return 0


if __name__ == "__main__":
    sys.exit(main())
