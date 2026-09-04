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
# the cases the figure shows, pinned per split (picked by pick_cases and then adjusted on request)
PINNED = {"val": ("case_2024050319", "case_2024021820", "case_2024103018", "case_2024122419"),
          "test": ("case_2025053115", "case_2025020906", "case_2025110421", "case_2025042016")}


def pick_cases(split, exclude=()):
    """One record per cardinal sector, months all different: the strongest-array record for N,
    the median-integral record otherwise. `exclude`: run_ids not to use."""
    wd = split.wdir_deg
    months = np.array([str(d)[:7] for d in split.meta["datetime"]])
    ok = ~np.isin(split.meta["run_id"].astype(str), list(exclude))
    sect = {c: np.abs((wd - deg + 180) % 360 - 180) <= 45 for c, deg in RM.CARDINAL}
    used, rows = set(), []
    for c in SECTORS:
        idx = np.where(sect[c] & ok & ~np.isin(months, list(used)))[0]
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
    ap.add_argument("--exclude", nargs="*", default=[], help="run_ids not to pick")
    ap.add_argument("--cases", nargs="*", default=None, help="explicit run_ids, one per row; default = PINNED[split], else the picker")
    ap.add_argument("--size", nargs=2, type=float, default=(18.0, 12.0), help="figure width and height [in]")
    ap.add_argument("--dpi", type=int, default=300)
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
    cases = a.cases or PINNED.get(a.split)
    if cases:
        ids = split.meta["run_id"].astype(str)
        rows = [int(np.where(ids == c)[0][0]) for c in cases]
    else:
        rows = pick_cases(split, a.exclude)
    sc = M.score_fields({k: f[rows] for k, f in fields.items()}, les[rows], split.wdir_deg[rows], arr, split.asymptote[rows])
    fm = {k: [RM.field_metrics(f[i], les[i], v, X, Y) for i in rows] for k, f in fields.items()}
    vmax = float(max(les[rows].max(), kl[rows].max(), cfm[rows].max(), fno[rows].max()))

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    nr, nc = len(rows), 6
    W, H = a.size
    # geometry: equal square boxes sized to fit the width, text scaled with the box
    left_frac, right_frac, gap_frac = 0.045, 0.035, 0.012
    top_pad, cbar_pad = 0.05 * H, 0.115 * H                    # column titles above, colour bar below
    gap = W * gap_frac
    box_w = (W * (1 - left_frac - right_frac) - (nc - 1) * gap) / nc
    box_h = (H - top_pad - cbar_pad - (nr - 1) * gap) / nr
    box = min(box_w, box_h)                                    # square boxes that fit both ways
    left = (W - nc * box - (nc - 1) * gap) / 2 + 0.5 * (W * left_frac - W * right_frac)
    bottom_pad = H - top_pad - nr * box - (nr - 1) * gap
    k = W / 18.0                                               # font scale: point sizes set for an 18 in wide print
    F = dict(head=12 * k, row=9 * k, letter=13 * k, table=8.5 * k, tick=8 * k, legend=9.5 * k, cbar=10 * k)
    fig = plt.figure(figsize=(W, H))
    axes = np.empty((nr, nc), object)
    for r in range(nr):
        for c in range(nc):
            x = (left + c * (box + gap)) / W
            y = (bottom_pad + (nr - 1 - r) * (box + gap)) / H
            axes[r, c] = fig.add_axes([x, y, box / W, box / H])
    heads = ["Kljun et al. (2015)", "FNO emulator", "CFM emulator", "LES target", "Metrics",
             "Crosswind-integrated $f_y$  [10$^{-3}$ m$^{-1}$]"]
    letters = iter(string.ascii_lowercase)
    for r, i in enumerate(rows):
        wd = float(split.wdir_deg[i])
        dt = str(split.meta["datetime"][i]).replace("T", " ")[:16]
        for c, fld in enumerate((kl[i], fno[i], cfm[i], les[i])):
            m = FS.footprint_panel(axes[r, c], fld, vmax, st, wd, letter=next(letters), letter_size=F["letter"])
        vals = {key: [(sc[k_][key][r] if key in sc[k_] else fm[k_][r][key]) for k_ in ("Kljun", "FNO", "CFM")] for key, _, _, _ in FS.TABLE_ROWS}
        FS.table_panel(axes[r, 4], vals, letter=next(letters), fontsize=F["table"], letter_size=F["letter"])
        FS.crosswind_panel(axes[r, 5], [("les", les[i], "LES target"), ("kljun", kl[i], "Kljun"), ("fno", fno[i], "FNO"), ("cfm", cfm[i], "CFM")],
                           wd, letter=next(letters), legend=(r == 0), xlabel=(r == nr - 1), ylabel=False,
                           letter_size=F["letter"], tick_size=F["tick"], legend_size=F["legend"], label_size=F["tick"] + 1)
        axes[r, 5].yaxis.tick_right()
        axes[r, 0].annotate(f"{dt} UTC\nwind from {wd:.0f}° ({split.octant[i]})\nz/L = {split.zL[i]:.2f},  $z_i$ = {split.scalars[i, 0]:.0f} m",
                            xy=(-0.03, 0.5), xycoords="axes fraction", ha="right", va="center", rotation=90, fontsize=F["row"], linespacing=1.15)
    for ax, h in zip(axes[0], heads):
        ax.annotate(h, xy=(0.5, 1.03), xycoords="axes fraction", ha="center", va="bottom", fontsize=F["head"])
    cb_h = 0.12 * box
    cax = fig.add_axes([left / W, (bottom_pad - 0.58 * cbar_pad) / H, (nc * box + (nc - 1) * gap) / W, cb_h / H])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal")
    lv = FS.levels(vmax)
    ticks = np.arange(np.ceil(lv[0]), np.floor(lv[-1]) + 0.5)
    cb.set_ticks(ticks); cb.set_ticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
    cb.set_label("flux footprint  [m$^{-2}$]", fontsize=F["cbar"], labelpad=2)
    cb.ax.tick_params(labelsize=F["cbar"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=a.dpi)
    print("wrote", out, [str(split.meta["run_id"][i]) for i in rows])
    return 0


if __name__ == "__main__":
    sys.exit(main())
