"""THE FROZEN VAL RECIPE for the two emulators, decided 2026-09-03 on val and applied unchanged
to test when the user runs it:

  CFM   seeds 0-3 (seed 4 dropped as the worst on val), the first 20 stored samples per seed
        (80 pooled; the fitted saturation S_sat = 21, upper band 64, gives 70 -> 20 per seed),
        spread at tau = 1 (the sampler as trained; the 1.19 of 2026-09-03 was dropped on 2026-09-04), PHYSICAL-space
        mean, then the 99.5% source-area cut (cells below the level that keeps 99.5% of the
        positive mass -> 0; every negative cell goes with them).
  FNO   seeds 0-3 mean (seed 4 dropped as the worst on val), the same 99.5% cut.
  Kljun untouched (never negative).
  LES   positive-only: negative cells -> 0 (the negative lobe is incoherent between two
        windows of one run, i.e. sampling noise; the cut denies the models any negative
        structure, so the target must not carry any either).

Metrics on these four fields; the crosswind-integrated product before and after; the
calibration of the tau-scaled samples against the positive-only LES.

    python -m ml_cfm.final_recipe [--split val]      # --split test needs --allow-test; never passed here
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml import data as D                      # noqa: E402
from ml import metrics as M                   # noqa: E402
from ml import evaluate as E                  # noqa: E402
from ml_cfm import tailthresh as TT           # noqa: E402
from ml_cfm import crps as CR                 # noqa: E402
from ml_cfm import evaluate as E2             # noqa: E402

RECIPE = dict(cfm_seeds=("seed0", "seed1", "seed2", "seed3"), samples_per_seed=20, tau=1.0,
              estimator="physical mean", cut_frac=0.995, fno_seeds=("seed0", "seed1", "seed2", "seed3"),
              les="positive-only", kljun="raw", decided_on="val", date="2026-09-04")
OUT = os.path.join(REPO, "results", "ml_cfm", "final_recipe")
GROUPS = ("all", "north_N_NE_NW", "array_in_view_gt5pct", "not_north")
KEYS = M.METRIC_KEYS + M.SHAPE_KEYS + M.IMAGE_KEYS + ("x80",)


TEST_DIR = os.path.join(REPO, "results", "ml_cfm", "test")   # the test-split artifacts live here, never under final/


def sample_path(seed, split_name):
    """Stored CFM samples for a seed on a split: val under final/, test under results/ml_cfm/test/."""
    if split_name == "val":
        return os.path.join(REPO, "results", "ml_cfm", "final", seed, "samples_val.npz")
    return os.path.join(TEST_DIR, f"{seed}_samples_{split_name}.npz")


def fno_path(seed, split_name):
    if split_name == "val":
        return os.path.join(REPO, "results", "ml", "final", seed, "pred_val.npz")
    return os.path.join(TEST_DIR, f"fno_{seed}_pred_{split_name}.npz")


def fno_mean(split, seeds):
    out = []
    for sd in seeds:
        with np.load(fno_path(sd, split.name)) as z:
            assert np.array_equal(z["run_id"], split.meta["run_id"])
            out.append(z["fno"].astype(np.float32))
    return np.mean(out, axis=0).astype(np.float32)


def cfm_samples(split, seeds, k, tau, valid):
    """(k*len(seeds), n, H, W) m^-2, the first k stored samples per seed, tau-scaled per seed."""
    out = []
    for sd in seeds:
        with np.load(sample_path(sd, split.name)) as z:
            assert np.array_equal(z["run_id"], split.meta["run_id"])
            T = z["samples_T"][:k].astype(np.float32)
            s_out = z["s_out"].astype(np.float32)
        m = T.mean(0, keepdims=True)
        T = m + tau * (T - m)
        out.append((s_out[None, :, None, None] * np.sinh(np.clip(T, -20, 20)) * valid[None, None]).astype(np.float32))
    return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--allow-test", action="store_true", help="the ONLY way the test split can be read; never passed by ml_cfm/")
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args(argv)
    if a.split != "val":
        sys.exit("this script evaluates the frozen recipe on val; the test run is the user's, with samples for that split written first")
    os.makedirs(a.outdir, exist_ok=True)
    split = D.load_split(a.split, allow_test=a.allow_test)
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    tr = D.load_split("train")
    groups = E.breakouts(split, np.isin(split.seed_key, list(set(tr.seed_key))))
    del tr

    samples = cfm_samples(split, RECIPE["cfm_seeds"], RECIPE["samples_per_seed"], RECIPE["tau"], valid)
    cfm_raw = samples.mean(0)
    cfm, icut = TT.threshold_stack(cfm_raw, RECIPE["cut_frac"])
    fno_raw = np.mean([np.load(os.path.join(REPO, "results", "ml", "final", sd, "pred_val.npz"))["fno"] for sd in RECIPE["fno_seeds"]], axis=0).astype(np.float32)
    fno, ifno = TT.threshold_stack(fno_raw, RECIPE["cut_frac"])
    les = np.maximum(split.target, 0).astype(np.float32)
    neg_les = (-np.minimum(split.target, 0).sum((1, 2))) / np.abs(split.target).sum((1, 2))
    fields = {"cfm": cfm, "fno": fno, "kljun": split.kljun}
    sc = M.score_fields(fields, les, split.wdir_deg, arr, split.asymptote)
    # the unmodified comparison for the 1-D product and the reference row
    sc_raw = M.score_fields({"cfm": cfm_raw, "fno": fno_raw, "kljun": split.kljun}, split.target, split.wdir_deg, arr, split.asymptote)

    cmp = {f"{m}_vs_{r}": {g: E.compare(sc[m], sc[r], groups[g], KEYS) for g in GROUPS}
           for m, r in (("cfm", "kljun"), ("cfm", "fno"), ("fno", "kljun"))}
    comp = {m: {g: M.composite(sc[m], sc["kljun"], groups[g])[0] for g in GROUPS} for m in ("cfm", "fno")}
    comp["cfm_vs_fno"] = {g: M.composite(sc["cfm"], sc["fno"], groups[g])[0] for g in GROUPS}
    comp_raw = {m: {g: M.composite(sc_raw[m], sc_raw["kljun"], groups[g])[0] for g in GROUPS} for m in ("cfm", "fno")}

    # calibration of the tau-scaled samples against the positive-only LES
    sm = E2.sample_metrics(samples, split, arr)
    les_scores = sc["les"]
    cal = {g: CR.calibration_table(sm, les_scores, groups[g]) for g in ("all", "north_N_NE_NW", "array_in_view_gt5pct")}

    out = dict(recipe=RECIPE, split=a.split, n=split.n, S=int(samples.shape[0]),
               cut=dict(cfm_mass_removed_median_pct=float(100 * np.median(icut["mass_removed_frac"])),
                        fno_mass_removed_median_pct=float(100 * np.median(ifno["mass_removed_frac"])),
                        les_negative_mass_zeroed_median_pct=float(100 * np.median(neg_les)),
                        les_negative_mass_zeroed_p95_pct=float(100 * np.percentile(neg_les, 95))),
               composite=comp, composite_unmodified=comp_raw, compare=cmp,
               medians={m: {g: {k: float(np.nanmedian(M.error_of(sc[m], k)[groups[g]])) for k in KEYS} for g in GROUPS} for m in fields},
               medians_unmodified={m: {g: {k: float(np.nanmedian(M.error_of(sc_raw[m], k)[groups[g]])) for k in KEYS} for g in GROUPS} for m in fields},
               calibration=cal, groups={g: int(groups[g].sum()) for g in GROUPS})
    with open(os.path.join(a.outdir, "recipe.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    L = [f"# The frozen recipe on {a.split} ({split.n} records)", "",
         "CFM: seeds 0-3, 20 stored samples each (80), tau 1 (no spread scaling), physical-space mean, 99.5% source-area cut. "
         "FNO: seeds 0-3 mean, 99.5% cut. Kljun raw. LES positive-only (negatives -> 0).", "",
         f"Cut removed (val medians): CFM {out['cut']['cfm_mass_removed_median_pct']:.2f}% of |mass|, FNO {out['cut']['fno_mass_removed_median_pct']:.2f}%; "
         f"LES negative mass zeroed {out['cut']['les_negative_mass_zeroed_median_pct']:.2f}% (p95 {out['cut']['les_negative_mass_zeroed_p95_pct']:.2f}%).", "",
         "## Composite vs Kljun (geometric mean of the five production-metric ratios; < 1 beats Kljun)", "",
         "| group | n | CFM | FNO | CFM/FNO | CFM unmodified vs raw LES | FNO unmodified vs raw LES |", "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        L.append(f"| {g} | {int(groups[g].sum())} | {comp['cfm'][g]:.3f} | {comp['fno'][g]:.3f} | {comp['cfm_vs_fno'][g]:.3f} | {comp_raw['cfm'][g]:.3f} | {comp_raw['fno'][g]:.3f} |")
    for pair, an, bn in (("cfm_vs_kljun", "CFM", "Kljun"), ("cfm_vs_fno", "CFM", "FNO"), ("fno_vs_kljun", "FNO", "Kljun")):
        L += ["", f"## {an} vs {bn}", ""]
        for g in ("all", "north_N_NE_NW", "array_in_view_gt5pct"):
            L += E2.fmt_table(cmp[pair][g], f"{g} ({int(groups[g].sum())})", an, bn)
    L += ["## The crosswind-integrated product: median errors, recipe fields vs unmodified fields (all records)", "",
          "| field | peak_x [m] | x80 [m] | shape_1d | integral | | peak_x unmod. | x80 unmod. | shape_1d unmod. | integral unmod. |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in fields:
        d, u = out["medians"][m]["all"], out["medians_unmodified"][m]["all"]
        L.append(f"| {m} | {d['peak_x']:.0f} | {d['x80']:.1f} | {d['shape_1d']:.4f} | {d['integral']:.4f} | | {u['peak_x']:.0f} | {u['x80']:.1f} | {u['shape_1d']:.4f} | {u['integral']:.4f} |")
    L += ["", "## Calibration of the 80 tau-scaled samples against the positive-only LES", "",
          "| group | metric | n | cover50 | cover90 | z sd | PIT KS p | CRPS | spread/skill |", "|---|---|---|---|---|---|---|---|---|"]
    for g, cg in cal.items():
        for k in ("array_share", "integral"):
            c = cg[k]
            L.append(f"| {g} | {k} | {c['n']} | {c['cover50']:.2f} | {c['cover90']:.2f} | {c['z_sd']:.2f} | {c['pit_ks_p']:.2g} | {c['crps_mean']:.3f} | {c['spread_skill']:.2f} |")
    with open(os.path.join(a.outdir, "recipe.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))

    # the figure: five cases, Kljun | FNO (cut) | CFM (cut) | LES positive-only, one scale
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import fig_corpus_pairs as FCP
    from ml_cfm.cut_sweep import cases
    rows = cases(split)
    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    surf = dict(water=st["water"] > 0.5, array=arr)
    share = lambda f: 100 * D.raster_array_share(f, arr)
    vmax = float(max(les[rows].max(), split.kljun[rows].max(), cfm[rows].max(), fno[rows].max()))
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    cols = [("Kljun", split.kljun), ("FNO, seeds 0-3, 99.5% cut", fno), ("CFM, seeds 0-3 x 20, tau 1, 99.5% cut", cfm), ("LES, positive-only", les)]
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 4.15 * len(rows)), squeeze=False)
    for r, i in enumerate(rows):
        wd = float(split.wdir_deg[i])
        for c, (name, fld) in enumerate(cols):
            ax = axes[r][c]
            im = FCP.raster(ax, fld[i], norm, "magma", ext, mask_below=norm.vmin)
            FCP.draw_frame(ax, surf, fg="w"); FCP.draw_wind(ax, wd)
            ttl = f"{name}\narray share {share(fld[i]):.1f}%"
            if c == 2:
                sh = share(samples[:, i])
                ttl += f"  90% interval [{np.percentile(sh, 5):.1f}, {np.percentile(sh, 95):.1f}]%"
            ax.set_title(ttl, fontsize=7.5); ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(f"{split.meta['run_id'][i]}  {split.octant[i]} {wd:.0f} deg\nz/L {split.zL[i]:.2f}  z_i {split.scalars[i, 0]:.0f} m", fontsize=7.5)
    fig.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.03, wspace=0.12, hspace=0.22)
    cax = fig.add_axes([0.935, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cax); cb.set_label("footprint [m$^{-2}$], one log scale", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle("The frozen recipe on val: Kljun / FNO / CFM / LES (positive-only); full 3660 m domain, floor 1e-4 x peak", fontsize=9)
    fig.savefig(os.path.join(a.outdir, "recipe_val.png"), dpi=100)
    print("wrote", os.path.join(a.outdir, "recipe_val.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
