"""Val evaluation of the prior-anchored CFM: the sample mean against Kljun AND the FNO
ensemble on the same records with the same estimators (ml.metrics, ml.evaluate), with and
without the connected-component filter; the sample spread against the realisation floors;
and the calibration of the spread against the 235 val LES targets.

    python -m ml_cfm.evaluate --seeds results/ml_cfm/final/seed0 ... --tag final

The samples come from each seed's samples_val.npz (written by ml_cfm.train at the best
epoch). The test split is refused by ml.data unless --allow-test is given; nothing in this
repository passes it.
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml import data as D                  # noqa: E402
from ml import metrics as M               # noqa: E402
from ml import evaluate as E              # noqa: E402
from ml_cfm import ccfilter as CC         # noqa: E402

OUT_DEFAULT = os.path.join(REPO, "results", "ml_cfm", "eval")
NORTH = ("N", "NE", "NW")
SPREAD_KEYS = ("array_share", "integral", "peak_x", "centroid_dist")
FLOOR_ROWS = [  # (quantity, floor, n, source)
    ("array_share [pp]", "5.34 between the two windows of case_2023111718", "1 pair",
     "results/ml/eval/floor/pair_floor.json"),
    ("array_share [pp]", "4.58 and 0.67 run-to-run (5.65->1.07, 1.14->0.47)", "2 runs x 2 cases",
     "results/les_realisation_spread.txt"),
    ("array_share [pp]", "0.19 median within-window SE", "~1000 records",
     "corpus/pairs_npz meta array_share_se"),
    ("integral", "0.0007 between the two windows; 0.444 and 0.148 run-to-run", "1 pair; 2 x 2",
     "pair_floor.json; les_realisation_spread.txt"),
    ("peak_x [m]", "0 between the two windows; 30 (one cell) run-to-run", "1 pair; 2 x 2", "same"),
    ("centroid [m]", "51 between the two windows; 46 run-to-run", "1 pair; 2 x 1", "same"),
    ("overlap80", "0.507 between the two windows; 0.56 two LPDM seeds", "1 pair; 1", "same"),
    ("shape_l1_2d", "0.630 between the two windows; 0.41 two LPDM seeds", "1 pair; 1", "same"),
]


def _rec(args):
    f, wd, asym = args
    return M.record_metrics(f, wd, _ARR, asym)


_ARR = None


def sample_metrics(samples, split, arr, workers=16):
    """samples (S,n,H,W) m^-2 -> dict key -> (S,n)."""
    global _ARR
    _ARR = arr
    S, n = samples.shape[:2]
    args = [(samples[s, i], float(split.wdir_deg[i]), float(split.asymptote[i]))
            for s in range(S) for i in range(n)]
    with get_context("fork").Pool(workers) as pool:
        rows = pool.map(_rec, args, chunksize=64)
    keys = ("array_share", "integral", "peak_x", "centroid_dist", "area80_ha")
    return {k: np.array([r[k] for r in rows], float).reshape(S, n) for k in keys}


def _pair(args):
    a, b, wd, asym = args
    e = M.pair_errors(a, b, wd, _ARR, asym)
    return e["overlap80"], e["shape_l1_2d"], e["centroid"], e["rel_l2"]


def between_sample_spread(samples, split, arr, n_pairs=8, seed=0, workers=16):
    global _ARR
    _ARR = arr
    rng = np.random.default_rng(seed)
    S, n = samples.shape[:2]
    args, owner = [], []
    for i in range(n):
        for _ in range(n_pairs):
            a, b = rng.choice(S, 2, replace=False)
            args.append((samples[a, i], samples[b, i], float(split.wdir_deg[i]),
                         float(split.asymptote[i])))
            owner.append(i)
    with get_context("fork").Pool(workers) as pool:
        rows = np.array(pool.map(_pair, args, chunksize=64))
    owner = np.array(owner)
    out = {}
    for j, k in enumerate(("overlap80", "shape_l1_2d", "centroid", "rel_l2")):
        out[k] = np.array([np.nanmedian(rows[owner == i, j]) for i in range(n)])
    return out


def calibration(sm, les, mask=None):
    """PIT, z-score and coverage of the LES value against the S samples, per key."""
    from scipy.stats import kstest
    out = {}
    for k in SPREAD_KEYS:
        s = sm[k]                                   # (S, n)
        l = les["centroid_dist" if k == "centroid_dist" else k]
        if k == "array_share":
            s, l = s * 100.0, l * 100.0
        if mask is not None:
            s, l = s[:, mask], l[mask]
        ok = np.isfinite(l) & np.isfinite(s).all(0)
        s, l = s[:, ok], l[ok]
        S = s.shape[0]
        pit = ((s < l[None]).sum(0) + 0.5 * (s == l[None]).sum(0)) / S
        sd = s.std(0, ddof=1)
        z = np.where(sd > 0, (l - s.mean(0)) / np.where(sd > 0, sd, 1), np.nan)
        lo50, hi50 = np.percentile(s, [25, 75], axis=0)
        lo90, hi90 = np.percentile(s, [5, 95], axis=0)
        out[k] = dict(
            n=int(ok.sum()), S=int(S),
            pit_ks_p=float(kstest(pit, "uniform").pvalue) if len(pit) > 5 else np.nan,
            pit_mean=float(pit.mean()), pit_hist=np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
            z_sd=float(np.nanstd(z)), z_mean=float(np.nanmean(z)),
            z_median_abs=float(np.nanmedian(np.abs(z))),
            cover50=float(np.mean((l >= lo50) & (l <= hi50))),
            cover90=float(np.mean((l >= lo90) & (l <= hi90))),
            sample_sd_median=float(np.median(sd)), sample_range90_median=float(np.median(hi90 - lo90)),
            les_minus_mean_median_abs=float(np.median(np.abs(l - s.mean(0)))))
    return out


def radial_spectrum(fT):
    """Mean over records of the radially averaged power spectrum of the asinh field."""
    n = fT.shape[-1]
    F = np.abs(np.fft.fftshift(np.fft.fft2(fT, axes=(-2, -1)), axes=(-2, -1))) ** 2
    ky, kx = np.meshgrid(np.arange(n) - n // 2, np.arange(n) - n // 2, indexing="ij")
    kr = np.hypot(kx, ky).astype(int)
    P = F.reshape(-1, n, n).mean(0)
    prof = np.bincount(kr.ravel(), P.ravel()) / np.maximum(np.bincount(kr.ravel()), 1)
    return prof[: n // 2]


def fmt_table(cmp_all, title, a_name, b_name):
    lines = [f"### {title}", "",
             f"| metric | unit | n | {a_name} median | {b_name} median | ratio | {a_name} wins | "
             "Wilcoxon p | median diff 95% CI |", "|---|---|---|---|---|---|---|---|---|"]
    for k, c in cmp_all.items():
        lo, hi = c["median_diff_ci95"]
        lines.append(f"| {k} | {E.UNITS.get(k, '')} | {c['n']} | {c['model_median']:.3f} | "
                     f"{c['ref_median']:.3f} | {c['ratio']:.2f} | {100*c['win_frac']:.0f}% | "
                     f"{c['wilcoxon_p']:.2g} | [{lo:+.3f}, {hi:+.3f}] |")
    if "overlap80" in cmp_all:
        lines.append(f"\noverlap80 is 1 - Jaccard here; raw medians {a_name} "
                     f"{cmp_all['overlap80']['model_overlap_median']:.3f} / {b_name} "
                     f"{cmp_all['overlap80']['ref_overlap_median']:.3f}.")
    return lines + [""]


def figures(outdir, split, fields, samples_phys, sm, sc, arr, spectra, cal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_corpus_pairs as FCP
    oc = split.octant.astype(str)
    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    surf = dict(water=np.zeros_like(arr, bool), array=arr)

    def raster_row(axes, i, names_fields, profile=True):
        wd = float(split.wdir_deg[i])
        tgt = split.target[i]
        lognorm, _, _ = FCP.pair_norms(split.kljun[i], np.maximum(tgt, fields["cfm"][i]))
        for ax, (name, Fld) in zip(axes, names_fields):
            FCP.raster(ax, Fld, lognorm, "magma", ext, mask_below=lognorm.vmin)
            l50, l80 = FCP.source_area_levels(Fld)
            if np.isfinite(l80):
                ax.contour(xc, xc, Fld, levels=[l80], colors="w", linewidths=0.6, linestyles="--")
            FCP.draw_frame(ax, surf, fg="w")
            FCP.draw_wind(ax, wd)
            ax.set_xlim(-900, 900); ax.set_ylim(-900, 900); ax.tick_params(labelsize=6)
            ax.set_title(f"{name}  array {100*float(D.raster_array_share(Fld, arr)):.1f}%", fontsize=7.5)
        if profile:
            ax = axes[-1]
            for Fld, col, nm in ((tgt, "#4c72b0", "LES"), (split.kljun[i], "#c44e52", "Kljun"),
                                 (fields["fno"][i], "#2ca02c", "FNO"), (fields["cfm"][i], "#9467bd", "CFM mean")):
                s_, fy = FCP.crosswind_integrated(Fld, wd)
                ax.plot(s_, fy, color=col, lw=1.1, label=nm)
            ax.axhline(0, color="k", lw=0.5); ax.set_xlim(-200, 1500); ax.tick_params(labelsize=6)
            ax.legend(fontsize=6, frameon=False)
            ax.set_title(f"{split.meta['run_id'][i]} {oc[i]} {wd:.0f} deg z/L {split.zL[i]:.2f}", fontsize=7)

    picks = []
    for o in D.OCTANTS:
        idx = np.where(oc == o)[0]
        if len(idx):
            integ = sc["les"]["integral"][idx]
            picks.append(int(idx[np.argsort(np.abs(integ - np.median(integ)))[0]]))
    fig, axes = plt.subplots(len(picks), 7, figsize=(23, 3.2 * len(picks)), squeeze=False)
    for r, i in enumerate(picks):
        raster_row(axes[r], i, [("LES target", split.target[i]), ("Kljun", split.kljun[i]),
                                ("FNO ensemble", fields["fno"][i]), ("CFM mean", fields["cfm"][i]),
                                ("CFM sample 1", samples_phys[0, i]), ("CFM sample 2", samples_phys[1, i])])
    fig.suptitle("One typical record per octant: LES / Kljun / FNO / CFM mean / two CFM samples, "
                 "one log scale; dashed = 80% source area", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.985]); fig.savefig(os.path.join(outdir, "octant_examples.png"), dpi=100)
    plt.close(fig)

    north = np.where(oc == "N")[0]
    if len(north):
        top = north[np.argsort(-split.meta["array_share"][north])][:4]
        fig, axes = plt.subplots(len(top), 6, figsize=(20, 3.4 * len(top)), squeeze=False)
        for r, i in enumerate(top):
            raster_row(axes[r][:5], i, [("LES target", split.target[i]), ("CFM mean", fields["cfm"][i]),
                                        ("sample 1", samples_phys[0, i]), ("sample 2", samples_phys[1, i]),
                                        ("sample 3", samples_phys[2, i])], profile=False)
            ax = axes[r][5]
            ax.hist(100 * sm["array_share"][:, i], bins=20, color="#9467bd", alpha=0.8)
            ax.axvline(100 * sc["les"]["array_share"][i], color="#4c72b0", lw=2, label="LES")
            ax.axvline(100 * sc["kljun"]["abs_array_share"][i], color="#c44e52", lw=1.5, label="Kljun")
            ax.axvline(100 * sc["fno"]["abs_array_share"][i], color="#2ca02c", lw=1.5, label="FNO")
            ax.set_xlabel("array share [%]", fontsize=7); ax.tick_params(labelsize=6)
            ax.legend(fontsize=6, frameon=False)
            ax.set_title(f"{split.meta['run_id'][i]}  {sm['array_share'].shape[0]} samples", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "north_samples.png"), dpi=100); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, k in zip(axes, SPREAD_KEYS):
        c = cal["all"][k]
        ax.bar(np.arange(10) / 10 + 0.05, c["pit_hist"], width=0.1, color="#9467bd")
        ax.axhline(c["n"] / 10, color="k", lw=0.8)
        ax.set_title(f"{k}: PIT of LES among samples\nKS p {c['pit_ks_p']:.2g}, cover50 {c['cover50']:.2f}, "
                     f"cover90 {c['cover90']:.2f}, z sd {c['z_sd']:.2f}", fontsize=7.5)
        ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "calibration.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for k, v in spectra.items():
        ax.loglog(np.arange(1, len(v)), v[1:], label=k, lw=1.1)
    ax.set_xlabel("wavenumber [cycles / 128 cells]", fontsize=8); ax.set_ylabel("power (asinh space)", fontsize=8)
    ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "spectra.png"), dpi=130); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", nargs="+", required=True, help="final/seedN directories")
    ap.add_argument("--fno", default=os.path.join(REPO, "results", "ml", "final"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--allow-test", action="store_true",
                    help="the ONLY way the test split can be read. Never passed by ml_cfm/.")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    outdir = os.path.join(a.outdir, a.tag)
    os.makedirs(outdir, exist_ok=True)
    split = D.load_split(a.split, allow_test=a.allow_test)
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)

    # ---- fields ---------------------------------------------------------------------
    samp, per_seed_mean, s_out = [], {}, None
    for sd in a.seeds:
        with np.load(os.path.join(sd, "samples_val.npz")) as z:
            if not np.array_equal(z["run_id"], split.meta["run_id"]):
                sys.exit(f"{sd}: run_id order does not match the split")
            T = z["samples_T"].astype(np.float32)
            s_out = z["s_out"].astype(np.float32)
            ph = (s_out[None, :, None, None] * np.sinh(np.clip(T, -20, 20)) * valid).astype(np.float32)
            samp.append(ph)
            per_seed_mean[os.path.basename(sd.rstrip("/"))] = ph.mean(0)
    samples = np.concatenate(samp)              # (S_total, n, 128, 128)
    S_total = samples.shape[0]
    cfm = samples.mean(0)
    fno_paths = sorted(glob.glob(os.path.join(a.fno, "seed*", "pred_val.npz")))
    fno = []
    for p in fno_paths:
        with np.load(p) as z:
            if not np.array_equal(z["run_id"], split.meta["run_id"]):
                sys.exit(f"{p}: run_id order mismatch")
            fno.append(z["fno"].astype(np.float32))
    fno = np.mean(fno, axis=0).astype(np.float32)
    fields = {"cfm": cfm, "fno": fno, "kljun": split.kljun}
    print(f"loaded {S_total} samples from {len(a.seeds)} seeds, FNO ensemble of {len(fno_paths)}")

    # ---- the filter -------------------------------------------------------------------
    filt, finfo = {}, {}
    for name in ("cfm", "fno", "kljun"):
        filt[name + "_f"], finfo[name] = CC.filter_stack(fields[name], "A")
    _, finfo["les"] = CC.filter_stack(split.target, "A")
    taus = np.array([CC.connectivity_level(f) for f in split.target])
    tau_star = float(np.nanmedian(taus))
    ruleB = {}
    for name in ("cfm", "fno", "kljun", "les"):
        src = split.target if name == "les" else fields[name]
        _, ib = CC.filter_stack(src, "B", tau=tau_star)
        ruleB[name] = dict(median_mass_removed_pct=float(100 * np.median(ib["mass_removed_frac"])),
                           mean_mass_removed_pct=float(100 * ib["mass_removed_frac"].mean()))
    filter_report = dict(
        ruleA={n: dict(median_mass_removed_pct=float(100 * np.median(i["mass_removed_frac"])),
                       mean_mass_removed_pct=float(100 * i["mass_removed_frac"].mean()),
                       max_mass_removed_pct=float(100 * i["mass_removed_frac"].max()),
                       median_components=float(np.median(i["n_components"])),
                       median_kept=float(np.median(i["n_kept"])),
                       peak_kept_frac=float(i["peak_kept"].mean())) for n, i in finfo.items()},
        ruleB=dict(tau_star=tau_star, tau_iqr=[float(np.nanpercentile(taus, 25)),
                                               float(np.nanpercentile(taus, 75))], fields=ruleB))
    print("filter:", json.dumps(filter_report["ruleA"], indent=None)[:400])
    all_fields = dict(fields, **filt)

    # ---- metrics of the means ----------------------------------------------------------
    sc = M.score_fields(all_fields, split.target, split.wdir_deg, arr, split.asymptote)
    for k, v in per_seed_mean.items():
        sc["seed_" + k] = M.score_fields({"x": v}, split.target, split.wdir_deg, arr, split.asymptote)["x"]
    # S-dependence of the mean
    s_dep = {}
    for S in (1, 4, 8, 32, S_total):
        m = samples[:S].mean(0)
        s = M.score_fields({"x": m}, split.target, split.wdir_deg, arr, split.asymptote)["x"]
        s_dep[S] = dict(composite=M.composite(s, sc["kljun"])[0],
                        medians={k: float(np.nanmedian(s[k])) for k in M.METRIC_KEYS + M.SHAPE_KEYS + ("rel_l2",)})
    print("means scored", round(time.time() - t0), "s")

    tr = D.load_split("train")
    shared = np.isin(split.seed_key, list(set(tr.seed_key)))
    groups = E.breakouts(split, shared)
    keys_all = M.METRIC_KEYS + M.SHAPE_KEYS + M.IMAGE_KEYS
    cmp = {}
    for name in ("cfm", "cfm_f"):
        for ref in ("kljun", "fno", "kljun_f", "fno_f"):
            cmp[f"{name}_vs_{ref}"] = {g: E.compare(sc[name], sc[ref], m, keys_all) for g, m in groups.items()}
    cmp["fno_vs_kljun"] = {g: E.compare(sc["fno"], sc["kljun"], m, keys_all) for g, m in groups.items()}
    cmp["fno_f_vs_kljun_f"] = {g: E.compare(sc["fno_f"], sc["kljun_f"], m, keys_all) for g, m in groups.items()}
    comp = {name: {g: M.composite(sc[name], sc["kljun"], m)[0] for g, m in groups.items()}
            for name in ("cfm", "cfm_f", "fno", "fno_f")}
    comp_vs_fno = {g: M.composite(sc["cfm"], sc["fno"], m)[0] for g, m in groups.items()}
    seed_comp = {k: M.composite(sc["seed_" + k], sc["kljun"])[0] for k in per_seed_mean}
    raw = {name: {k: float(np.nanmedian(sc[name][k])) for k in keys_all} for name in all_fields}
    asym = {name: float(np.nanmedian(sc[name]["integral_asym"])) for name in all_fields}
    asym["les"] = float(np.nanmedian(np.abs(sc["les"]["integral_asym_err"])))

    # ---- spread and calibration ------------------------------------------------------
    sm = sample_metrics(samples, split, arr)
    print("per-sample metrics", round(time.time() - t0), "s")
    bs = between_sample_spread(samples, split, arr)
    print("between-sample pairs", round(time.time() - t0), "s")
    cal = {g: calibration(sm, sc["les"], groups[g]) for g in
           ("all", "north_N_NE_NW", "array_in_view_gt5pct", "not_north")}
    spread = {}
    for g in ("all", "north_N_NE_NW", "array_in_view_gt5pct"):
        m = groups[g]
        spread[g] = dict(
            n=int(m.sum()),
            array_share_sd_pp=float(np.median(100 * sm["array_share"][:, m].std(0, ddof=1))),
            array_share_range90_pp=float(np.median(100 * (np.percentile(sm["array_share"][:, m], 95, 0)
                                                            - np.percentile(sm["array_share"][:, m], 5, 0)))),
            integral_sd=float(np.median(sm["integral"][:, m].std(0, ddof=1))),
            integral_range90=float(np.median(np.percentile(sm["integral"][:, m], 95, 0)
                                             - np.percentile(sm["integral"][:, m], 5, 0))),
            peak_x_sd_m=float(np.median(sm["peak_x"][:, m].std(0, ddof=1))),
            centroid_sd_m=float(np.median(np.hypot(*[sm["centroid_dist"][:, m].std(0, ddof=1)] * 2) / np.sqrt(2))),
            between_samples=dict(overlap80=float(np.nanmedian(bs["overlap80"][m])),
                                 shape_l1_2d=float(np.nanmedian(bs["shape_l1_2d"][m])),
                                 centroid_m=float(np.nanmedian(bs["centroid"][m])),
                                 rel_l2=float(np.nanmedian(bs["rel_l2"][m]))))
    # ---- sharpness ------------------------------------------------------------------
    s_ref = M._s_ref()
    sl = slice(D.PAD, D.PAD + D.NG)
    spectra = {k: radial_spectrum(np.arcsinh(v[:, sl, sl] / s_ref)).tolist() for k, v in
               (("LES", split.target), ("Kljun", split.kljun), ("FNO", fno), ("CFM mean", cfm),
                ("CFM sample", samples[0]))}
    grad = {k: float(np.mean(np.abs(np.gradient(np.arcsinh(v[:, sl, sl] / s_ref), axis=(1, 2))))) for k, v in
            (("les", split.target), ("kljun", split.kljun), ("fno", fno), ("cfm_mean", cfm), ("cfm_sample", samples[0]))}
    hi = {k: float(np.sum(v[32:])) / float(np.sum(v[1:])) for k, v in spectra.items()}
    sharp = dict(mean_abs_gradient_T=grad, high_k_power_fraction=hi)

    # ---- cost -----------------------------------------------------------------------
    cost = {}
    for sd in a.seeds:
        with open(os.path.join(sd, "run.json")) as fh:
            r = json.load(fh)
        cost[os.path.basename(sd.rstrip("/"))] = dict(r["sampling"], n_params=r["n_params"],
                                                       best_epoch=r["best_epoch"], epochs_run=r["epochs_run"],
                                                       val_mse_ref=r["val_mse_ref"], gap=r["gap"]["loss_ratio"],
                                                       wall_s=r["wall_s"])

    out = dict(tag=a.tag, split=a.split, n=split.n, S_total=S_total, seeds=a.seeds, fno_members=len(fno_paths),
               groups={g: int(m.sum()) for g, m in groups.items()}, filter=filter_report, compare=cmp,
               composite_vs_kljun=comp, composite_cfm_vs_fno=comp_vs_fno, seed_composites=seed_comp,
               S_dependence=s_dep, raw_medians=raw, integral_vs_asymptote=asym, spread=spread,
               calibration=cal, sharpness=sharp, cost=cost, floors=FLOOR_ROWS, wall_s=round(time.time() - t0))
    with open(os.path.join(outdir, "eval.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    # per-record table
    cols = ["run_id", "octant", "wdir_deg", "zL", "les_array_share_pct", "cfm_share_mean_pct", "cfm_share_sd_pp",
            "cfm_share_p5", "cfm_share_p95", "pit_share", "fno_share_pct", "kljun_share_pct"]
    with open(os.path.join(outdir, "per_record.tsv"), "w") as fh:
        fh.write("\t".join(cols + [f"{n}_{k}" for n in ("cfm", "cfm_f", "fno", "kljun") for k in M.METRIC_KEYS]) + "\n")
        sh = 100 * sm["array_share"]
        for i in range(split.n):
            les = 100 * sc["les"]["array_share"][i]
            row = [split.meta["run_id"][i], split.octant[i], f"{split.wdir_deg[i]:.1f}", f"{split.zL[i]:.4f}",
                   f"{les:.3f}", f"{sh[:, i].mean():.3f}", f"{sh[:, i].std(ddof=1):.3f}",
                   f"{np.percentile(sh[:, i], 5):.3f}", f"{np.percentile(sh[:, i], 95):.3f}",
                   f"{np.mean(sh[:, i] < les):.3f}", f"{100*sc['fno']['abs_array_share'][i]:.3f}",
                   f"{100*sc['kljun']['abs_array_share'][i]:.3f}"]
            row += [f"{sc[n][k][i]:.5g}" for n in ("cfm", "cfm_f", "fno", "kljun") for k in M.METRIC_KEYS]
            fh.write("\t".join(row) + "\n")

    # ---- markdown ----------------------------------------------------------------------
    Lm = [f"# CFM evaluation `{a.tag}` on {a.split}: {split.n} records, {S_total} samples "
          f"({len(a.seeds)} seeds x {S_total // len(a.seeds)}), FNO ensemble of {len(fno_paths)}", ""]
    for pair, an, bn in (("cfm_vs_kljun", "CFM mean", "Kljun"), ("cfm_vs_fno", "CFM mean", "FNO"),
                         ("cfm_f_vs_fno_f", "CFM mean (filtered)", "FNO (filtered)")):
        Lm += [f"## {an} vs {bn}", ""]
        for g, title in (("all", "all records"), ("north_N_NE_NW", "N/NE/NW"),
                         ("array_in_view_gt5pct", "array in view (LES share > 5%)")):
            Lm += fmt_table(cmp[pair][g], f"{title} ({groups[g].sum()})", an, bn)
    Lm += ["## Composite (geometric mean of the five production-metric ratios) by group", "",
           "| group | n | CFM/Kljun | CFM filtered/Kljun | FNO/Kljun | FNO filtered/Kljun | CFM/FNO |", "|---|---|---|---|---|---|---|"]
    for g, m in groups.items():
        Lm.append(f"| {g} | {int(m.sum())} | {comp['cfm'][g]:.3f} | {comp['cfm_f'][g]:.3f} | {comp['fno'][g]:.3f} | "
                  f"{comp['fno_f'][g]:.3f} | {comp_vs_fno[g]:.3f} |")
    Lm += ["", "## Per-seed composites vs Kljun (each seed's own 32-sample mean)", "",
           "| seed | composite |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in seed_comp.items()]
    Lm += ["", "## Dependence of the mean on the sample count S", "", "| S | composite vs Kljun | rel_l2 | shape_l1_2d | overlap80 (1-J) | array_share pp |", "|---|---|---|---|---|---|"]
    for S, v in s_dep.items():
        m = v["medians"]
        Lm.append(f"| {S} | {v['composite']:.3f} | {m['rel_l2']:.3f} | {m['shape_l1_2d']:.3f} | {m['overlap80']:.3f} | {m['array_share']:.3f} |")
    Lm += ["", "## The connected-component filter (rule A, 99.9% of |mass|)", "",
           "| field | median mass removed | mean | max | median components | median kept | peak kept |", "|---|---|---|---|---|---|---|"]
    for n_, i in filter_report["ruleA"].items():
        Lm.append(f"| {n_} | {i['median_mass_removed_pct']:.3f}% | {i['mean_mass_removed_pct']:.3f}% | {i['max_mass_removed_pct']:.2f}% | "
                  f"{i['median_components']:.0f} | {i['median_kept']:.0f} | {100*i['peak_kept_frac']:.0f}% |")
    rb = filter_report["ruleB"]
    Lm += ["", f"Rule B: the LES target becomes single-connected at a median level tau* = {rb['tau_star']:.2e} of its peak "
           f"(IQR {rb['tau_iqr'][0]:.1e}-{rb['tau_iqr'][1]:.1e}); at that level the mass removed is " +
           ", ".join(f"{k} {v['median_mass_removed_pct']:.2f}%" for k, v in rb["fields"].items()) + " (medians).", ""]
    Lm += ["## Sample spread against the realisation floors", "",
           "| group | n | array-share sd [pp] | 5-95% range [pp] | integral sd | integral 5-95% | peak_x sd [m] | between-sample overlap80 | shape L1 | centroid [m] | rel L2 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, v in spread.items():
        b = v["between_samples"]
        Lm.append(f"| {g} | {v['n']} | {v['array_share_sd_pp']:.2f} | {v['array_share_range90_pp']:.2f} | {v['integral_sd']:.3f} | "
                  f"{v['integral_range90']:.3f} | {v['peak_x_sd_m']:.0f} | {b['overlap80']:.3f} | {b['shape_l1_2d']:.3f} | {b['centroid_m']:.0f} | {b['rel_l2']:.3f} |")
    Lm += ["", "| quantity | floor | independent realisations | source |", "|---|---|---|---|"]
    Lm += [f"| {q} | {f} | {n_} | `{s}` |" for q, f, n_, s in FLOOR_ROWS]
    Lm += ["", "## Calibration: the LES target as one more draw from the sample set", "",
           "| group | metric | n | PIT KS p | PIT mean | z sd | median |z| | cover 50% | cover 90% | sample sd (median) | |LES - mean| (median) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, cg in cal.items():
        for k, c in cg.items():
            Lm.append(f"| {g} | {k} | {c['n']} | {c['pit_ks_p']:.2g} | {c['pit_mean']:.3f} | {c['z_sd']:.2f} | {c['z_median_abs']:.2f} | "
                      f"{c['cover50']:.2f} | {c['cover90']:.2f} | {c['sample_sd_median']:.3f} | {c['les_minus_mean_median_abs']:.3f} |")
    Lm += ["", "## Sharpness (asinh space, interior)", "",
           "| field | mean |grad| | high-k power fraction (k >= 32) |", "|---|---|---|"]
    for k in grad:
        kk = {"les": "LES", "kljun": "Kljun", "fno": "FNO", "cfm_mean": "CFM mean", "cfm_sample": "CFM sample"}[k]
        Lm.append(f"| {kk} | {grad[k]:.4f} | {hi[kk]:.4f} |")
    Lm += ["", "## Cost", "", "| seed | params | best epoch / run | val_mse_ref | gap | S | steps | solver | ms / record / sample | wall s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in cost.items():
        Lm.append(f"| {k} | {c['n_params']/1e6:.2f} M | {c['best_epoch']}/{c['epochs_run']} | {c['val_mse_ref']:.6f} | x{c['gap']:.2f} | {c['S']} | {c['steps']} | {c['solver']} | {c['ms_per_record_per_sample']:.2f} | {c['wall_s']:.0f} |")
    Lm += ["", "## Integral vs the asymptote 1 - z_m/z_i (median |error|)", ""] + [f"- {k}: {v:.4f}" for k, v in asym.items()]
    with open(os.path.join(outdir, "eval.md"), "w") as fh:
        fh.write("\n".join(Lm) + "\n")
    print("\n".join(Lm[:60]))
    if not a.no_figures:
        figures(outdir, split, fields, samples, sm, sc, arr, {k: np.array(v) for k, v in spectra.items()}, cal)
    print("done", round(time.time() - t0), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
