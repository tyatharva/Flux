"""Evaluate an emulator against Kljun on the same records, with the production metrics,
the realisation floors beside each, and the regime / wind-sector breakouts.

    python -m ml.evaluate --ckpt results/ml/final/seed0/best.pt [--ckpt ...] --tag final
    python -m ml.evaluate --floor            # the two-window pair, scored by this evaluator
    python -m ml.evaluate ... --split test --allow-test     # THE USER RUNS THIS, ONCE

Several --ckpt paths are averaged in physical units (a seed ensemble). Outputs land in
results/ml/eval/<tag>/: eval.json, eval.md, per_record.tsv and figures.

The val split is the default. The test split is refused by ml.data unless --allow-test is
given; nothing in this repository passes it.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml import data as D                  # noqa: E402
from ml import features as F              # noqa: E402
from ml import metrics as M               # noqa: E402

OUT_DEFAULT = os.path.join(REPO, "results", "ml", "eval")
NORTH = ("N", "NE", "NW")

# Realisation floors from the record, quoted beside each metric. n is the number of
# independent realisations behind each number (PROJECT_BRIEF.md standing rule 5).
FLOORS = {
    "peak_x": [("1 cell (30 m) run-to-run, both cases", "2 runs x 2 cases",
                "results/les_realisation_spread.txt"),
               ("0-24 m half-vs-half, convective", "4 cases",
                "docs/results/FOURTH_PASS_RESULTS.md:547-560")],
    "centroid": [("46 m run-to-run (334 vs 380 m), convective", "2 runs x 1 case",
                  "results/les_realisation_spread.txt"),
                 ("15-90 m half-vs-half, convective", "4 cases",
                  "docs/results/FOURTH_PASS_RESULTS.md:547-560"),
                 ("336 m p90 at 22.5 min of sub-windows", "18 sub-windows x 1 window",
                  "docs/results/STAGE2-6_RESULTS_V2.md:310-380")],
    "overlap80": [("0.592 half-vs-half at this grid", "1 window",
                   "docs/results/STAGE2-6_RESULTS_V2.md:296-305"),
                  ("0.43-0.51 half-vs-half, convective", "4 cases",
                   "docs/results/FOURTH_PASS_RESULTS.md:547-560"),
                  ("0.56 two LPDM seeds on the same fields", "1 case",
                   "results/les_realisation_spread.txt:30")],
    "array_share": [("5.65 -> 1.07 pp and 1.14 -> 0.47 pp run-to-run", "2 runs x 2 cases",
                     "results/les_realisation_spread.txt"),
                    ("0.19 pp median within-window SE (release groups)", "~1000 records",
                     "corpus/pairs_npz meta array_share_se, train+val")],
    "integral": [("1.44x and 1.20x run-to-run", "2 runs x 2 cases",
                  "results/les_realisation_spread.txt"),
                 ("5.5% two LPDM seeds on the same fields", "1 case",
                  "results/les_realisation_spread.txt:30")],
    "shape_l1_2d": [("0.41 two LPDM seeds on the same fields", "1 case",
                     "results/les_realisation_spread.txt:30"),
                    ("0.92 two release ensembles, retired 60 m grid", "1 case",
                     "results/stage5.txt:36; docs/results/STAGE2-6_RESULTS.md:520-545")],
    "shape_1d": [("the two-window pair scored by this evaluator", "1 pair",
                  "results/ml/eval/floor/pair_floor.json")],
}
UNITS = dict(peak_x="m", centroid="m", overlap80="Jaccard", array_share="pp", integral="",
             shape_l1_2d="", shape_1d="", rel_l2="", rel_l2_T="", mae_T="asinh", rmse_T="asinh",
             pearson_T="r", ssim_T="", psnr_T="dB")
for _k in M.IMAGE_KEYS:
    FLOORS[_k] = [("the two-window pair scored by this evaluator", "1 pair",
                   "results/ml/eval/floor/pair_floor.json")]
FLOORS["rel_l2"].insert(0, ("per-cell L1 0.41 (two LPDM seeds) to 0.92 (two release "
                            "ensembles, retired grid)", "1 case each",
                            "results/les_realisation_spread.txt:30; results/stage5.txt:36"))


def load_checkpoint(path):
    import torch
    from ml.train import TrainConfig
    from ml.model import build_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ck["config"])
    model = build_model(cfg, ck["n_channels"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return cfg, model, ck


def predict_split(ckpt_paths, split, statics, norm, dev=None):
    """Mean over checkpoints of the physical-space prediction, (n,128,128) float32."""
    import torch
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    for p in ckpt_paths:
        cfg, model, _ = load_checkpoint(p)
        model = model.to(dev)
        fx = F.Features(split, statics, norm, cfg.feature_spec())
        const = torch.from_numpy(fx.const).to(dev)
        out = []
        with torch.no_grad():
            for i in range(0, split.n, 64):
                sl = slice(i, min(split.n, i + 64))
                x = torch.from_numpy(fx.x_in[sl]).to(dev)
                s = torch.from_numpy(fx.scal[sl]).to(dev)
                r = model(x, const, s)
                base = torch.from_numpy(fx.base_T[sl]).to(dev)
                pT = base + r if cfg.head == "residual" else r
                out.append(pT.cpu().numpy())
        preds.append(fx.to_physical(np.concatenate(out)))
    return np.mean(preds, axis=0).astype(np.float32), len(preds)


def bootstrap_median_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        diffs.append(np.median(a[i]) - np.median(b[i]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def compare(sc_model, sc_ref, mask=None, keys=M.METRIC_KEYS):
    """Per metric: medians, means, win fraction, Wilcoxon p, bootstrap CI of the median
    difference (model - ref; negative is better)."""
    from scipy.stats import wilcoxon
    out = {}
    for k in keys:
        a = M.error_of(sc_model, k)
        b = M.error_of(sc_ref, k)
        if mask is not None:
            a, b = a[mask], b[mask]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        d = a - b
        try:
            p = float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else np.nan
        except ValueError:
            p = np.nan
        lo, hi = bootstrap_median_diff(a, b) if len(a) > 5 else (np.nan, np.nan)
        out[k] = dict(n=int(len(a)), model_median=float(np.median(a)),
                      ref_median=float(np.median(b)), model_mean=float(a.mean()),
                      ref_mean=float(b.mean()), win_frac=float(np.mean(a < b)),
                      tie_frac=float(np.mean(a == b)), wilcoxon_p=p,
                      median_diff_ci95=[lo, hi],
                      ratio=float(np.median(a) / np.median(b)) if np.median(b) > 0 else np.nan)
        if k == "overlap80":     # report the overlap itself too, not only 1 - J
            out[k]["model_overlap_median"] = float(np.median(1 - a))
            out[k]["ref_overlap_median"] = float(np.median(1 - b))
    return out


def breakouts(split, shared_seed=None):
    oc = split.octant.astype(str)
    groups = {"all": np.ones(split.n, bool)}
    for o in D.OCTANTS:
        groups["oct_" + o] = oc == o
    groups["north_N_NE_NW"] = np.isin(oc, NORTH)
    groups["not_north"] = ~groups["north_N_NE_NW"]
    groups["array_in_view_gt5pct"] = split.meta["array_share"] > 0.05
    groups["array_absent_le5pct"] = ~groups["array_in_view_gt5pct"]
    zl = split.zL
    q = np.percentile(zl, [100 / 3, 200 / 3])
    groups["zL_tercile_most_unstable"] = zl <= q[0]
    groups["zL_tercile_middle"] = (zl > q[0]) & (zl <= q[1])
    groups["zL_tercile_least_unstable"] = zl > q[1]
    zi = split.scalars[:, 0]
    q = np.percentile(zi, [100 / 3, 200 / 3])
    groups["zi_tercile_shallow"] = zi <= q[0]
    groups["zi_tercile_middle"] = (zi > q[0]) & (zi <= q[1])
    groups["zi_tercile_deep"] = zi > q[1]
    if shared_seed is not None:
        groups["seed_shared_with_train"] = shared_seed
        groups["seed_not_in_train"] = ~shared_seed
    return groups


def fmt_table(cmp_all, title):
    lines = [f"### {title}", "",
             "| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | "
             "ratio | FNO wins | Wilcoxon p | median diff 95% CI |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in cmp_all.items():
        lo, hi = c["median_diff_ci95"]
        lines.append(f"| {k} | {UNITS[k]} | {c['n']} | {c['model_median']:.3f} | "
                     f"{c['ref_median']:.3f} | {c['model_mean']:.3f} | {c['ref_mean']:.3f} | "
                     f"{c['ratio']:.2f} | {100*c['win_frac']:.0f}% | {c['wilcoxon_p']:.2g} | "
                     f"[{lo:+.3f}, {hi:+.3f}] |")
    lines.append("")
    if "overlap80" in cmp_all:
        lines.append("overlap80 is reported as 1 - Jaccard (smaller is better); the raw "
                     "medians are FNO {:.3f} / Kljun {:.3f}.".format(
                         cmp_all["overlap80"]["model_overlap_median"],
                         cmp_all["overlap80"]["ref_overlap_median"]))
    return lines


def pair_floor():
    """Score the two-window pair validation_pairs_30m/case_2023111718_w{0,1} with THIS
    evaluator, after cropping both to the cone by the file's rule. Both are train-split
    records (2023-11-17), so reading them is allowed."""
    import mask_cone as mc
    st = D.load_statics()
    arr = st["array"] > 0.5
    ga = D.read_grid_attrs()
    recs = []
    for w in (0, 1):
        p = os.path.join(REPO, "validation_pairs_30m", f"case_2023111718_w{w}.npz")
        with np.load(p, allow_pickle=True) as z:
            sc = np.asarray(z["scalars"], np.float64)
            tg = np.asarray(z["target"], np.float64)
            kl = np.asarray(z["kljun"], np.float64)
            meta = json.loads(str(z["meta"]))
        if meta.get("split_key", meta.get("split")) not in ("train", "case_2023111718") \
                and meta.get("split") != "train":
            raise RuntimeError("the validation pair is not a train-split record")
        X, Y = mc.axis_grids()
        xw, yw = mc.wind_frame(X, Y, float(sc[4]), float(sc[5]))
        sy = mc.sigma_y_field(sc, float(meta["u_mean_ms"]), xw)
        keep = mc.cone_keep(xw, yw, sy, float(ga["cone_mask_k"]), float(ga["cone_mask_y_min_m"]),
                            float(ga["cone_mask_x_min_m"]))
        recs.append(dict(target=np.where(keep, tg, 0.0), kljun=kl, wdir=float(meta["wdir_deg"]),
                         asym=1.0 - D.Z_RECEPTOR / float(sc[0]), meta=meta))
    e = M.pair_errors(recs[1]["target"], recs[0]["target"], recs[0]["wdir"], arr, recs[0]["asym"])
    ek = [M.pair_errors(r["kljun"], r["target"], r["wdir"], arr, r["asym"]) for r in recs]
    return dict(case="case_2023111718", wdir_deg=recs[0]["wdir"],
                w1_vs_w0={k: float(v) for k, v in e.items()},
                kljun_vs_w={k: [float(x[k]) for x in ek] for k in ek[0]},
                note="two windows of one run: a LOWER bound on the realisation floor "
                     "(near-duplicates at 0.19-0.33 of the half-vs-half floor, "
                     "PROJECT_BRIEF.md N_WINDOWS)")


def _panel_row(axes, split, fields, sc, arr, i, FCP, half=900):
    """LES / Kljun / FNO rasters on one shared log scale, plus the crosswind-integrated
    profiles of all three on the wind axis. The visual-match panel."""
    import numpy as np
    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    klj, tgt, fno = split.kljun[i], split.target[i], fields["fno"][i]
    fnc = fields["fno_cone"][i]
    lognorm, _, vmax = FCP.pair_norms(klj, np.maximum(tgt, fno))
    surf = dict(water=np.zeros_like(arr, bool), array=arr)
    wd = float(split.wdir_deg[i])
    for c, (Fld, name, key) in enumerate(((tgt, "LES target (cone)", "les"),
                                          (klj, "Kljun", "kljun"), (fno, "FNO raw", "fno"),
                                          (fnc, "FNO, cone-cropped", "fno_cone"))):
        ax = axes[c]
        FCP.raster(ax, Fld, lognorm, "magma", ext, mask_below=lognorm.vmin)
        l50, l80 = FCP.source_area_levels(Fld)
        if np.isfinite(l50):
            ax.contour(xc, xc, Fld, levels=[l80], colors="w", linewidths=0.6, linestyles="--")
        FCP.draw_frame(ax, surf, fg="w")
        FCP.draw_wind(ax, wd)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.tick_params(labelsize=6)
        share = float(D.raster_array_share(Fld, arr))
        pk = sc[key]["peak_x" if key == "les" else "abs_peak_x"][i]
        ax.set_title(f"{name}  array {100*share:.1f}%  peak_x {pk:.0f} m", fontsize=7.5)
    ax = axes[4]
    for Fld, col, nm in ((tgt, "#4c72b0", "LES"), (klj, "#c44e52", "Kljun"),
                         (fnc, "#2ca02c", "FNO (cone)")):
        s_, fy = FCP.crosswind_integrated(Fld, wd)
        ax.plot(s_, fy, color=col, lw=1.2, label=nm)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(-200, 1500)
    ax.set_xlabel("upwind distance [m]", fontsize=7)
    ax.set_ylabel("f_y [m^-1]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(f"{split.meta['run_id'][i]}  {split.octant[i]} {wd:.0f} deg  "
                 f"z/L {split.zL[i]:.2f}  z_i {split.scalars[i, 0]:.0f} m", fontsize=7)


def figures(outdir, split, fields, sc, cmp_groups, arr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    import fig_corpus_pairs as FCP
    oc = split.octant.astype(str)
    # 0. one typical record per octant: the record whose LES integral is the octant median
    picks = []
    for o in D.OCTANTS:
        idx = np.where(oc == o)[0]
        if len(idx):
            integ = sc["les"]["integral"][idx]
            picks.append(int(idx[np.argsort(np.abs(integ - np.median(integ)))[0]]))
    fig, axes = plt.subplots(len(picks), 5, figsize=(17, 3.3 * len(picks)), squeeze=False)
    for r, i in enumerate(picks):
        _panel_row(axes[r], split, fields, sc, arr, i, FCP)
    fig.suptitle("One typical record per octant (LES integral at the octant median): "
                 "LES / Kljun / FNO raw / FNO cone-cropped on one log scale; "
                 "dashed = 80% source area", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(outdir, "octant_examples.png"), dpi=110)
    plt.close(fig)
    # 0b. residual panels for the four N-wind records with the largest LES array share:
    # what the LES adds to Kljun against what the FNO adds to Kljun.
    north = np.where(oc == "N")[0]
    if len(north):
        top = north[np.argsort(-split.meta["array_share"][north])][:4]
        xc, xe = FCP.axes_m()
        ext = [xe[0], xe[-1], xe[0], xe[-1]]
        surf = dict(water=np.zeros_like(arr, bool), array=arr)
        fig, axes = plt.subplots(len(top), 2, figsize=(8.2, 3.6 * len(top)), squeeze=False)
        for r, i in enumerate(top):
            klj, tgt, fno = split.kljun[i], split.target[i], fields["fno"][i]
            vmax = float(max(np.abs(tgt - klj).max(), np.abs(fno - klj).max(), 1e-12))
            norm = SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax, base=10)
            for c, (Fld, name) in enumerate(((tgt - klj, "LES - Kljun (the truth's correction)"),
                                             (fno - klj, "FNO - Kljun (the learned residual)"))):
                ax = axes[r][c]
                im = FCP.raster(ax, Fld, norm, "RdBu_r", ext)
                FCP.draw_frame(ax, surf, fg="k")
                FCP.draw_wind(ax, float(split.wdir_deg[i]), colour="k")
                ax.set_xlim(-900, 900)
                ax.set_ylim(-900, 900)
                ax.tick_params(labelsize=6)
                ax.set_title(name, fontsize=7.5)
                sl = M.shape_l1_2d(fno, tgt) if c else M.shape_l1_2d(klj, tgt)
                ax.text(0.02, 0.02, f"shape L1 vs LES {sl:.2f}", transform=ax.transAxes,
                        fontsize=6.5, color="k", bbox=dict(fc="w", alpha=0.7, ec="none"))
            fig.colorbar(im, ax=axes[r][1], fraction=0.046, pad=0.02).ax.tick_params(labelsize=5.5)
            axes[r][0].set_ylabel(f"{split.meta['run_id'][i]}  N {split.wdir_deg[i]:.0f} deg",
                                  fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "north_residuals.png"), dpi=110)
        plt.close(fig)
    # 1. per-octant ratio bars
    keys = list(M.METRIC_KEYS)
    octs = ["oct_" + o for o in D.OCTANTS]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.2), sharey=False)
    for ax, k in zip(axes, keys):
        vals = [cmp_groups[g][k]["ratio"] if g in cmp_groups else np.nan for g in octs]
        ns = [cmp_groups[g][k]["n"] if g in cmp_groups else 0 for g in octs]
        ax.bar(range(8), vals, color=["#c44e52" if o in NORTH else "#4c72b0" for o in D.OCTANTS])
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(range(8))
        ax.set_xticklabels([f"{o}\n{n}" for o, n in zip(D.OCTANTS, ns)], fontsize=7)
        ax.set_title(f"{k}: median|err| FNO / Kljun", fontsize=8)
        ax.set_ylim(0, max(2.0, np.nanmax(vals) * 1.1 if np.isfinite(np.nanmax(vals)) else 2))
    fig.suptitle("< 1 the FNO beats Kljun; red octants carry the array signal", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "octant_ratios.png"), dpi=130)
    plt.close(fig)
    # 2. four N-wind records with the largest LES array share: LES / Kljun / FNO
    north = np.where(split.octant.astype(str) == "N")[0]
    if len(north):
        top = north[np.argsort(-split.meta["array_share"][north])][:4]
        xc, xe = FCP.axes_m()
        ext = [xe[0], xe[-1], xe[0], xe[-1]]
        surf = dict(water=split.valid_mask & False, array=arr)
        fig, axes = plt.subplots(len(top), 3, figsize=(11, 3.5 * len(top)), squeeze=False)
        for r, i in enumerate(top):
            klj, tgt, fno = split.kljun[i], split.target[i], fields["fno"][i]
            lognorm, _, vmax = FCP.pair_norms(klj, np.maximum(tgt, fno))
            for c, (Fld, name) in enumerate(((tgt, "LES target (cone)"), (klj, "Kljun"),
                                             (fno, "FNO"))):
                ax = axes[r][c]
                FCP.raster(ax, Fld, lognorm, "magma", ext, mask_below=lognorm.vmin)
                FCP.draw_frame(ax, surf, fg="w")
                FCP.draw_wind(ax, float(split.wdir_deg[i]))
                ax.set_xlim(-900, 900)
                ax.set_ylim(-900, 900)
                ax.tick_params(labelsize=6)
                share = D.raster_array_share(Fld, arr)
                ax.set_title(f"{name}  array {100*float(share):.1f}%  "
                             f"peak_x {sc['les' if c == 0 else ('kljun' if c == 1 else 'fno')][('peak_x' if c == 0 else 'abs_peak_x')][i]:.0f} m",
                             fontsize=7.5)
            axes[r][0].set_ylabel(f"{split.meta['run_id'][i]}\nwdir {split.wdir_deg[i]:.0f}",
                                  fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "north_examples.png"), dpi=120)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--pred", default=None, help="pred_val.npz instead of checkpoints")
    ap.add_argument("--split", default="val")
    ap.add_argument("--allow-test", action="store_true",
                    help="the ONLY way the test split can be read. Never passed by ml/.")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    ap.add_argument("--floor", action="store_true", help="score the two-window pair only")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    outdir = os.path.join(a.outdir, a.tag)
    os.makedirs(outdir, exist_ok=True)
    if a.floor:
        fl = pair_floor()
        with open(os.path.join(outdir, "pair_floor.json"), "w") as fh:
            json.dump(fl, fh, indent=1)
        print(json.dumps(fl, indent=1))
        return 0
    if not a.ckpt and not a.pred:
        sys.exit("give --ckpt (one or more) or --pred")

    split = D.load_split(a.split, allow_test=a.allow_test)
    st = D.load_statics()
    norm = D.read_norm()
    arr = st["array"] > 0.5
    if a.pred:
        with np.load(a.pred) as z:
            fno = z["fno"].astype(np.float32)
            if not np.array_equal(z["run_id"], split.meta["run_id"]):
                sys.exit("pred file's run_id order does not match the split")
        n_members = 1
    else:
        fno, n_members = predict_split(a.ckpt, split, st, norm)
    keep = D.cone_masks(split)
    fields = {"fno": fno, "fno_cone": np.where(keep, fno, 0.0).astype(np.float32),
              "kljun": split.kljun}
    sc = M.score_fields(fields, split.target, split.wdir_deg, arr, split.asymptote)

    shared = None
    if a.split == "val":
        tr = D.load_split("train")
        shared = np.isin(split.seed_key, list(set(tr.seed_key)))
    groups = breakouts(split, shared)
    cmp = {name: {g: compare(sc[name], sc["kljun"], m) for g, m in groups.items()}
           for name in ("fno", "fno_cone")}
    shape = {name: {g: compare(sc[name], sc["kljun"], m, M.SHAPE_KEYS + M.IMAGE_KEYS)
                    for g, m in (("all", groups["all"]),
                                 ("north_N_NE_NW", groups["north_N_NE_NW"]))}
             for name in ("fno", "fno_cone")}
    comp = {name: {g: M.composite(sc[name], sc["kljun"], m)[0] for g, m in groups.items()}
            for name in ("fno", "fno_cone")}
    # Kljun and each field against the asymptote (integral), all records
    asym = {name: dict(median_abs=float(np.nanmedian(sc[name]["integral_asym"])))
            for name in fields}
    asym["les"] = dict(median_abs=float(np.nanmedian(np.abs(sc["les"]["integral_asym_err"]))))

    raw = {name: {k: float(np.nanmedian(sc[name][k])) for k in
                  M.METRIC_KEYS + M.SHAPE_KEYS + M.IMAGE_KEYS} for name in fields}
    out = dict(tag=a.tag, split=a.split, n=split.n, n_members=n_members, ckpts=a.ckpt,
               groups={g: int(m.sum()) for g, m in groups.items()}, compare=cmp, shape=shape,
               raw_medians=raw,
               composite=comp, integral_vs_asymptote=asym, floors=FLOORS,
               cone_keep_fraction_of_fno_mass=float(
                   np.abs(fields["fno_cone"]).sum() / max(np.abs(fno).sum(), 1e-30)))
    with open(os.path.join(outdir, "eval.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    # per-record table
    cols = ["run_id", "octant", "wdir_deg", "zL", "zi_m", "les_array_share", "seed_shared"]
    rows = []
    for i in range(split.n):
        r = [split.meta["run_id"][i], split.octant[i], f"{split.wdir_deg[i]:.1f}",
             f"{split.zL[i]:.4f}", f"{split.scalars[i, 0]:.0f}",
             f"{sc['les']['array_share'][i]:.5f}",
             int(shared[i]) if shared is not None else ""]
        for name in ("fno", "fno_cone", "kljun"):
            for k in M.METRIC_KEYS:
                r.append(f"{sc[name][k][i]:.5g}")
        rows.append(r)
    hdr = cols + [f"{name}_{k}" for name in ("fno", "fno_cone", "kljun") for k in M.METRIC_KEYS]
    with open(os.path.join(outdir, "per_record.tsv"), "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    # the markdown
    L_ = [f"# Evaluation `{a.tag}` on {a.split} ({split.n} records, {n_members} member(s))", ""]
    for name in ("fno", "fno_cone"):
        L_ += [f"## {name}", ""]
        L_ += fmt_table(cmp[name]["all"], "all records")
        L_ += fmt_table(cmp[name]["north_N_NE_NW"],
                        f"N/NE/NW only ({int(groups['north_N_NE_NW'].sum())} records)")
        L_ += fmt_table(cmp[name]["array_in_view_gt5pct"],
                        f"array in view, LES share > 5% ({int(groups['array_in_view_gt5pct'].sum())})")
        L_ += fmt_table(shape[name]["all"], "shape and 2-D field metrics, all records (not "
                        "in the composite; per-cell agreement sits on the noise floor)")
        L_ += fmt_table(shape[name]["north_N_NE_NW"], "shape and 2-D field metrics, N/NE/NW only")
        L_ += ["", "Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are "
               "tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw "
               "medians are in eval.json under `raw_medians`.", ""]
        L_ += ["", "### composite (geometric mean of the five ratios) by group", "",
               "| group | n | composite |", "|---|---|---|"]
        for g, m in groups.items():
            L_.append(f"| {g} | {int(m.sum())} | {comp[name][g]:.3f} |")
        L_ += [""]
    L_ += ["## Realisation floors beside each metric", "",
           "| metric | floor | independent realisations | source |", "|---|---|---|---|"]
    for k, fl in FLOORS.items():
        for txt, n_, src in fl:
            L_.append(f"| {k} | {txt} | {n_} | `{src}` |")
    L_ += ["", "## Integral against the asymptote 1 - z_m/z_i (median |error|)", "",
           "| field | median abs error |", "|---|---|"]
    for k, v in asym.items():
        L_.append(f"| {k} | {v['median_abs']:.4f} |")
    with open(os.path.join(outdir, "eval.md"), "w") as fh:
        fh.write("\n".join(L_) + "\n")
    print("\n".join(L_))
    if not a.no_figures:
        figures(outdir, split, fields, sc, cmp["fno"], arr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
