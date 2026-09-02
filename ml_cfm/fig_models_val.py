"""Five val cases, full domain, one global log colour scale:
Kljun | FNO ensemble | CFM mean | CFM mean with the spread scaled by tau | LES target.

    python -m ml_cfm.fig_models_val [--tau 1.19] [--out results/ml_cfm/calib/final/models_val.png]

The tau column is the mean of the temperature-scaled samples (T' = mean + tau (T - mean) in
asinh space, then back to m^-2). tau leaves the mean nearly unchanged by construction; what
it changes is the spread, so each CFM title carries the array share with its 90% sample
interval. Cases: the strongest-array N record and the median-integral record of N, NE, NW
and W. The test split is never read.
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml import data as D                      # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tau", type=float, default=1.19)
    ap.add_argument("--seeds", nargs="+", default=sorted(glob.glob(os.path.join(REPO, "results", "ml_cfm", "final", "seed?"))))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ml_cfm", "calib", "final", "models_val.png"))
    ap.add_argument("--floor", type=float, default=1e-4, help="colour floor as a fraction of the global peak")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import fig_corpus_pairs as FCP

    split = D.load_split("val")
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    oc = split.octant.astype(str)
    integ = split.meta["integral"]
    north = np.where(oc == "N")[0]
    rows = [int(north[np.argmax(split.meta["array_share"][north])])]
    for o in ("N", "NE", "NW", "W"):
        idx = np.where((oc == o) & ~np.isin(np.arange(split.n), rows))[0]
        rows.append(int(idx[np.argmin(np.abs(integ[idx] - np.median(integ[idx])))]))

    T, s_out = [], None
    for sd in a.seeds:
        with np.load(os.path.join(sd, "samples_val.npz")) as z:
            assert np.array_equal(z["run_id"], split.meta["run_id"])
            T.append(z["samples_T"][:, rows].astype(np.float32))
            s_out = z["s_out"].astype(np.float32)[rows]
    T = np.concatenate(T)                                              # (S, 5, 128, 128)
    phys = lambda TT: s_out[None, :, None, None] * np.sinh(np.clip(TT, -20, 20)) * valid
    F = phys(T)
    m = T.mean(0, keepdims=True)
    Ft = phys(m + a.tau * (T - m))
    cfm, cfm_t = F.mean(0), Ft.mean(0)
    fno = np.mean([np.load(p)["fno"][rows] for p in sorted(glob.glob(os.path.join(REPO, "results", "ml", "final", "seed*", "pred_val.npz")))], axis=0)
    les, kl = split.target[rows], split.kljun[rows]

    vmax = float(max(les.max(), kl.max(), cfm.max(), fno.max()))
    norm = LogNorm(vmin=vmax * a.floor, vmax=vmax)
    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    surf = dict(water=st["water"] > 0.5, array=arr)
    share = lambda f: 100 * D.raster_array_share(f, arr)
    cols = [("Kljun", kl), ("FNO ensemble (5 seeds)", fno), ("CFM mean (5 seeds x 32 samples)", cfm),
            (f"CFM mean, spread x tau = {a.tau}", cfm_t), ("LES target", les)]
    fig, axes = plt.subplots(len(rows), 5, figsize=(21, 4.15 * len(rows)), squeeze=False)
    for r, i in enumerate(rows):
        wd = float(split.wdir_deg[i])
        for c, (name, fld) in enumerate(cols):
            ax = axes[r][c]
            im = FCP.raster(ax, fld[r], norm, "magma", ext, mask_below=norm.vmin)
            FCP.draw_frame(ax, surf, fg="w")
            FCP.draw_wind(ax, wd)
            ttl = f"{name}\narray share {share(fld[r]):.1f}%"
            if c in (2, 3):
                sh = share(F[:, r]) if c == 2 else share(Ft[:, r])
                lo, hi = np.percentile(sh, [5, 95])
                ttl += f"  90% interval [{lo:.1f}, {hi:.1f}]%"
            ax.set_title(ttl, fontsize=7.5)
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(f"{split.meta['run_id'][i]}  {oc[i]} {wd:.0f} deg\nz/L {split.zL[i]:.2f}  z_i {split.scalars[i, 0]:.0f} m", fontsize=7.5)
    fig.subplots_adjust(left=0.04, right=0.93, top=0.95, bottom=0.03, wspace=0.12, hspace=0.22)
    cax = fig.add_axes([0.945, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("footprint [m$^{-2}$], one log scale for every panel", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Val: Kljun / FNO / CFM / CFM with tau / LES on the full 3660 m domain; global scale, floor {a.floor:g} x peak. "
                 "Green = array, cyan = water, dotted = last real cell, arrow = mean flow", fontsize=9)
    fig.savefig(a.out, dpi=100)
    print("wrote", a.out, "cases", [split.meta["run_id"][i] for i in rows])
    return 0


if __name__ == "__main__":
    sys.exit(main())
