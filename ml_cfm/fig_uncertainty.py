"""2-D error bars for the CFM: how the sample spread is shown on a footprint map.

    python -m ml_cfm.fig_uncertainty [--tau 1.19] [--out results/ml_cfm/calib/final/uncertainty_2d.png]

Uses the five final seeds' val samples (32 each, Euler 16) with the post-hoc temperature
tau applied in asinh space. Four rows (the strongest-array N records), five panels each:
  1. sample mean on the log scale with the LES 80% source-area contour;
  2. per-cell 5-95% width relative to the mean (the pointwise error bar, as a ratio);
  3. P(cell inside the 80% source area) over the samples, with the LES and Kljun 80% contours
     -- the contour's error bar;
  4. crosswind-integrated profile: median and 50% / 90% pointwise bands, LES and Kljun;
  5. array-share histogram of the samples with LES, Kljun and FNO marked.
The test split is never read.
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
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ml_cfm", "calib", "final", "uncertainty_2d.png"))
    ap.add_argument("--n-rows", type=int, default=4)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_corpus_pairs as FCP

    split = D.load_split("val")
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    oc = split.octant.astype(str)
    north = np.where(oc == "N")[0]
    rows = north[np.argsort(-split.meta["array_share"][north])][: a.n_rows]

    T, s_out = [], None
    for sd in a.seeds:
        with np.load(os.path.join(sd, "samples_val.npz")) as z:
            assert np.array_equal(z["run_id"], split.meta["run_id"])
            T.append(z["samples_T"][:, rows].astype(np.float32))
            s_out = z["s_out"].astype(np.float32)[rows]
    T = np.concatenate(T)                                   # (S, m, 128, 128)
    m = T.mean(0, keepdims=True)
    T = m + a.tau * (T - m)                                  # the temperature
    F = s_out[None, :, None, None] * np.sinh(np.clip(T, -20, 20)) * valid   # m^-2
    S = F.shape[0]
    fno = np.mean([np.load(p)["fno"][rows] for p in sorted(glob.glob(os.path.join(REPO, "results", "ml", "final", "seed*", "pred_val.npz")))], axis=0)

    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    surf = dict(water=np.zeros_like(arr, bool), array=arr)
    fig, axes = plt.subplots(len(rows), 5, figsize=(22, 3.9 * len(rows)), squeeze=False)
    for r, i in enumerate(rows):
        wd = float(split.wdir_deg[i])
        les, kl = split.target[i], split.kljun[i]
        f = F[:, r]
        mean = f.mean(0)
        lognorm, _, _ = FCP.pair_norms(kl, np.maximum(les, mean))
        _, l80_les = FCP.source_area_levels(les)
        _, l80_kl = FCP.source_area_levels(kl)
        _, l80_mean = FCP.source_area_levels(mean)

        ax = axes[r][0]
        FCP.raster(ax, mean, lognorm, "magma", ext, mask_below=lognorm.vmin)
        ax.contour(xc, xc, les, levels=[l80_les], colors="#4c72b0", linewidths=1.0)
        ax.contour(xc, xc, mean, levels=[l80_mean], colors="w", linewidths=0.8, linestyles="--")
        ax.set_title(f"{split.meta['run_id'][i]}  wdir {wd:.0f}  z/L {split.zL[i]:.2f}\nsample mean ({S} samples, tau {a.tau}); blue = LES 80% area, white = mean's", fontsize=7.5)

        ax = axes[r][1]
        lo, hi = np.percentile(f, [5, 95], axis=0)
        width = np.where(mean > 0.02 * mean.max(), (hi - lo) / np.maximum(mean, 1e-12), np.nan)
        im = ax.imshow(width, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=2)
        ax.contour(xc, xc, les, levels=[l80_les], colors="w", linewidths=0.8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        ax.set_title("pointwise error bar: (p95 - p5) / mean, where mean > 2% of its peak", fontsize=7.5)

        ax = axes[r][2]
        inside = np.zeros_like(mean)
        for s in range(S):
            _, l80 = FCP.source_area_levels(f[s])
            inside += (f[s] >= l80)
        inside /= S
        im = ax.imshow(np.where(inside > 0, inside, np.nan), origin="lower", extent=ext, cmap="cividis", vmin=0, vmax=1)
        ax.contour(xc, xc, les, levels=[l80_les], colors="#4c72b0", linewidths=1.0)
        ax.contour(xc, xc, kl, levels=[l80_kl], colors="#c44e52", linewidths=1.0, linestyles=":")
        ax.contour(xc, xc, inside, levels=[0.5], colors="w", linewidths=0.8, linestyles="--")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        ax.set_title("P(cell in the 80% source area) over samples\nblue LES, red dotted Kljun, white dashed P = 0.5", fontsize=7.5)

        for ax in axes[r][:3]:
            FCP.draw_frame(ax, surf, fg="w")
            FCP.draw_wind(ax, wd)
            ax.set_xlim(-900, 900); ax.set_ylim(-900, 900); ax.tick_params(labelsize=6)

        ax = axes[r][3]
        prof = np.stack([FCP.crosswind_integrated(f[s], wd)[1] for s in range(S)])
        s_ax, _ = FCP.crosswind_integrated(f[0], wd)
        p5, p25, p50, p75, p95 = np.percentile(prof, [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(s_ax, p5, p95, color="#9467bd", alpha=0.2, label="90% band")
        ax.fill_between(s_ax, p25, p75, color="#9467bd", alpha=0.4, label="50% band")
        ax.plot(s_ax, p50, color="#9467bd", lw=1.2, label="sample median")
        ax.plot(s_ax, FCP.crosswind_integrated(les, wd)[1], color="#4c72b0", lw=1.1, label="LES")
        ax.plot(s_ax, FCP.crosswind_integrated(kl, wd)[1], color="#c44e52", lw=1.0, label="Kljun")
        ax.axhline(0, color="k", lw=0.5); ax.set_xlim(-200, 1500); ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False); ax.set_xlabel("upwind distance [m]", fontsize=7)
        ax.set_title("crosswind-integrated footprint: pointwise 50% / 90% bands", fontsize=7.5)

        ax = axes[r][4]
        sh = 100 * np.array([D.raster_array_share(f[s], arr) for s in range(S)])
        ax.hist(sh, bins=20, color="#9467bd", alpha=0.8)
        for val, col, nm in ((100 * D.raster_array_share(les, arr), "#4c72b0", "LES"),
                             (100 * D.raster_array_share(kl, arr), "#c44e52", "Kljun"),
                             (100 * D.raster_array_share(fno[r], arr), "#2ca02c", "FNO")):
            ax.axvline(val, color=col, lw=1.5, label=nm)
        lo5, hi95 = np.percentile(sh, [5, 95])
        ax.axvspan(lo5, hi95, color="#9467bd", alpha=0.12)
        ax.set_title(f"array share: mean {sh.mean():.1f}%, 90% interval [{lo5:.1f}, {hi95:.1f}]%", fontsize=7.5)
        ax.set_xlabel("array share [%]", fontsize=7); ax.tick_params(labelsize=6); ax.legend(fontsize=6, frameon=False)
    fig.suptitle(f"CFM error bars in 2-D on the four strongest-array N records of val: {S} samples from 5 seeds, "
                 f"Euler 16, spread scaled by tau = {a.tau}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=100)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
