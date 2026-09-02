"""The two FNO-style figures the evaluator does not draw: north_residuals.png (LES - Kljun
against CFM mean - Kljun and FNO - Kljun) and octant_ratios.png (CFM/Kljun and CFM/FNO)."""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    sys.path.insert(0, _p)
from ml import data as D                  # noqa: E402
from ml import metrics as M               # noqa: E402


def main(outdir=os.path.join(REPO, "results", "ml_cfm", "eval", "final")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    import fig_corpus_pairs as FCP
    split = D.load_split("val")
    arr = D.load_statics()["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    cfm = np.mean([(np.load(p)["s_out"][:, None, None] * np.sinh(np.load(p)["samples_T"].astype(np.float32).clip(-20, 20)) * valid).mean(0)
                   for p in sorted(glob.glob(os.path.join(REPO, "results/ml_cfm/final/seed[0-4]/samples_val.npz")))], axis=0)
    fno = np.mean([np.load(p)["fno"] for p in sorted(glob.glob(os.path.join(REPO, "results/ml/final/seed*/pred_val.npz")))], axis=0)
    e = json.load(open(os.path.join(outdir, "eval.json")))
    oc = split.octant.astype(str)
    north = np.where(oc == "N")[0]
    top = north[np.argsort(-split.meta["array_share"][north])][:4]
    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    surf = dict(water=np.zeros_like(arr, bool), array=arr)
    fig, axes = plt.subplots(len(top), 3, figsize=(12, 3.6 * len(top)), squeeze=False)
    for r, i in enumerate(top):
        klj, tgt = split.kljun[i], split.target[i]
        vmax = float(max(np.abs(tgt - klj).max(), np.abs(cfm[i] - klj).max(), np.abs(fno[i] - klj).max(), 1e-12))
        norm = SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax, base=10)
        for c, (Fld, name, f_) in enumerate(((tgt - klj, "LES - Kljun (the truth's correction)", tgt),
                                             (cfm[i] - klj, "CFM mean - Kljun", cfm[i]),
                                             (fno[i] - klj, "FNO - Kljun", fno[i]))):
            ax = axes[r][c]
            im = FCP.raster(ax, Fld, norm, "RdBu_r", ext)
            FCP.draw_frame(ax, surf, fg="k")
            FCP.draw_wind(ax, float(split.wdir_deg[i]), colour="k")
            ax.set_xlim(-900, 900); ax.set_ylim(-900, 900); ax.tick_params(labelsize=6)
            ax.set_title(name, fontsize=7.5)
            sl = M.shape_l1_2d(klj if c == 0 else f_, tgt)
            ax.text(0.02, 0.02, f"shape L1 vs LES {sl:.2f}", transform=ax.transAxes, fontsize=6.5,
                    bbox=dict(fc="w", alpha=0.7, ec="none"))
        fig.colorbar(im, ax=axes[r][2], fraction=0.046, pad=0.02).ax.tick_params(labelsize=5.5)
        axes[r][0].set_ylabel(f"{split.meta['run_id'][i]}  N {split.wdir_deg[i]:.0f} deg", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "north_residuals.png"), dpi=110); plt.close(fig)

    keys = list(M.METRIC_KEYS)
    octs = ["oct_" + o for o in D.OCTANTS]
    fig, axes = plt.subplots(2, len(keys), figsize=(3.1 * len(keys), 6), sharey=False)
    for row, (pair, lab) in enumerate((("cfm_vs_kljun", "CFM / Kljun"), ("cfm_vs_fno", "CFM / FNO"))):
        for ax, k in zip(axes[row], keys):
            vals = [e["compare"][pair][g][k]["ratio"] if g in e["compare"][pair] else np.nan for g in octs]
            ns = [e["compare"][pair][g][k]["n"] for g in octs]
            ax.bar(range(8), vals, color=["#c44e52" if o in ("N", "NE", "NW") else "#4c72b0" for o in D.OCTANTS])
            ax.axhline(1.0, color="k", lw=0.8)
            ax.set_xticks(range(8)); ax.set_xticklabels([f"{o}\n{n}" for o, n in zip(D.OCTANTS, ns)], fontsize=7)
            ax.set_title(f"{k}: median|err| {lab}", fontsize=8)
            top_ = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 2
            ax.set_ylim(0, max(2.0, top_ * 1.1))
    fig.suptitle("< 1 the CFM mean wins; red octants carry the array signal (peak_x ratios are 0/0 where both are exact)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "octant_ratios.png"), dpi=130); plt.close(fig)
    print("written")


if __name__ == "__main__":
    main()
