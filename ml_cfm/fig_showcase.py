"""The showcase figure: four cases, each from a different month and a different cardinal wind
sector, one per row; columns Kljun | FNO | CFM | LES | the eight metrics of that case |
crosswind-integrated footprint; the frozen recipe's fields (ml_cfm/final_recipe.py); filled
log10 contours on the turbo map, one scale, horizontal colour bar underneath.

    python -m ml_cfm.fig_showcase [--split val] [--allow-test]
"""
import argparse
import os
import string
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
from ml_cfm import figstyle as FS             # noqa: E402

SECTORS = ("N", "W", "S", "E")


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
    a = ap.parse_args(argv)
    if a.split == "test" and not a.allow_test:
        raise SystemExit("refusing the test split without --allow-test")
    out = a.out or os.path.join(FR.OUT, f"showcase_{a.split}.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    vmax = float(max(les[rows].max(), kl[rows].max(), cfm[rows].max(), fno[rows].max()))

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    nr, nc = len(rows), 6
    box, gap, left, top_pad, bottom_pad = 4.2, 0.22, 1.05, 0.55, 1.35        # inches
    W = left + nc * box + (nc - 1) * gap + 0.9
    H = top_pad + nr * box + (nr - 1) * gap + bottom_pad
    fig = plt.figure(figsize=(W, H))
    axes = np.empty((nr, nc), object)
    for r in range(nr):
        for c in range(nc):
            x = (left + c * (box + gap)) / W
            y = (bottom_pad + (nr - 1 - r) * (box + gap)) / H
            axes[r, c] = fig.add_axes([x, y, box / W, box / H])
    heads = ["Kljun et al. (2015)", "FNO emulator", "CFM emulator", "LES target", "Metrics for this case", "Crosswind-integrated footprint"]
    letters = iter(string.ascii_lowercase)
    for r, i in enumerate(rows):
        wd = float(split.wdir_deg[i])
        dt = str(split.meta["datetime"][i]).replace("T", " ")[:16]
        for c, fld in enumerate((kl[i], fno[i], cfm[i], les[i])):
            m = FS.footprint_panel(axes[r, c], fld, vmax, st, wd, letter=next(letters))
        vals = {key: [(sc[k][key][r] if key in sc[k] else fm[k][r][key]) for k in ("Kljun", "FNO", "CFM")] for key, _, _, _ in FS.TABLE_ROWS}
        FS.table_panel(axes[r, 4], vals, letter=next(letters), fontsize=11.5)
        FS.crosswind_panel(axes[r, 5], [("les", les[i], "LES target"), ("kljun", kl[i], "Kljun"), ("fno", fno[i], "FNO"), ("cfm", cfm[i], "CFM")],
                           wd, letter=next(letters), legend=(r == 0), xlabel=(r == nr - 1), ylabel=True)
        axes[r, 5].yaxis.tick_right(); axes[r, 5].yaxis.set_label_position("right")
        axes[r, 0].annotate(f"{dt} UTC\nwind from {wd:.0f}° ({split.octant[i]}),  z/L = {split.zL[i]:.2f},  $z_i$ = {split.scalars[i, 0]:.0f} m",
                            xy=(-0.03, 0.5), xycoords="axes fraction", ha="right", va="center", rotation=90, fontsize=12.5)
    for ax, h in zip(axes[0], heads):
        ax.annotate(h, xy=(0.5, 1.03), xycoords="axes fraction", ha="center", va="bottom", fontsize=15)
    cax = fig.add_axes([left / W, 0.55 / H, (nc * box + (nc - 1) * gap) / W, 0.22 / H])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal")
    lv = FS.levels(vmax)
    ticks = np.arange(np.ceil(lv[0]), np.floor(lv[-1]) + 0.5)
    cb.set_ticks(ticks); cb.set_ticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
    cb.set_label("flux footprint  [m$^{-2}$]      star = tower,  magenta = solar array,  blue = Lake Kegonsa,  arrow = wind,  background = terrain (USGS 3DEP), 5 m steps", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out, [str(split.meta["run_id"][i]) for i in rows])
    return 0


if __name__ == "__main__":
    sys.exit(main())
