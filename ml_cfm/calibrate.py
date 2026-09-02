"""Calibration of CFM variants on val: coverage, CRPS and spread-skill per group, the mean
metrics against Kljun and against the baseline, and the post-hoc temperature fit.

    python -m ml_cfm.calibrate --tag final \\
        --models fm_seed1=results/ml_cfm/final/seed1 \\
                 fm_seed1_sig0.3=results/ml_cfm/final/seed1,sigma=0.3 \\
                 crps_pure_ft=results/ml_cfm/calib/runs/crps_pure_ft,steps=2 ...

Each model spec is name=dir[,sigma=..][,steps=..][,solver=..]; S samples are drawn from
dir/best.pt on the val split (never test: ml.data refuses it, nothing here asks). The first
model is the baseline: every other model's mean is compared with it, and the temperature
scaling is fitted on its samples.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml import data as D                      # noqa: E402
from ml import metrics as M                   # noqa: E402
from ml import evaluate as E                  # noqa: E402
from ml_cfm import evaluate as E2             # noqa: E402
from ml_cfm import crps as CR                 # noqa: E402
from ml_cfm import infer as I                 # noqa: E402

OUT_DEFAULT = os.path.join(REPO, "results", "ml_cfm", "calib")
GROUPS = ("all", "north_N_NE_NW", "array_in_view_gt5pct", "not_north")
REPORT_GROUPS = ("all", "north_N_NE_NW", "array_in_view_gt5pct")


def parse_spec(s):
    name, rest = s.split("=", 1)
    parts = rest.split(",")
    spec = dict(name=name, dir=parts[0], sigma=None, steps=None, solver=None)
    for p in parts[1:]:
        k, v = p.split("=")
        spec[k] = float(v) if k == "sigma" else (int(v) if k == "steps" else v)
    return spec


def bands(n):
    """+-2 binomial sd around the nominal coverage."""
    return dict(cover50=[0.5 - 2 * np.sqrt(0.25 / n), 0.5 + 2 * np.sqrt(0.25 / n)],
                cover90=[0.9 - 2 * np.sqrt(0.09 / n), min(1.0, 0.9 + 2 * np.sqrt(0.09 / n))])


def draw(spec, split, st, norm, dev, S, seed):
    cfg, model, ck = I.load_checkpoint(os.path.join(spec["dir"], "best.pt"), dev)
    prep = I.Prepared(cfg, split, st, norm, dev)
    steps = spec["steps"] or (cfg.crps_steps if cfg.loss != "fm" else cfg.steps_final)
    solver = spec["solver"] or cfg.solver
    sigma = cfg.sigma if spec["sigma"] is None else spec["sigma"]
    sT, secs = prep.samples(model, cfg, S, steps, solver, seed, sigma=sigma)
    used = dict(sigma_train=cfg.sigma, sigma_sample=sigma, steps=steps, solver=solver, loss=cfg.loss,
                init_from=cfg.init_from, target_thresh=cfg.target_thresh, S=S,
                ms_per_record_per_sample=1000 * secs / (split.n * S), n_params=model.n_params())
    del model
    torch.cuda.empty_cache()
    return sT, prep, used


def evaluate_samples(sT, prep, split, arr, groups, sc_kljun, mask_t, tgt_T):
    """sT (S,n,H,W) torch on device -> everything the tables need."""
    S = sT.shape[0]
    field_crps = CR.crps_field_eval(sT, tgt_T, mask_t).cpu().numpy()
    samples = prep.physical(sT.cpu().numpy())                      # (S,n,H,W) m^-2
    mean = samples.mean(0)
    sc = M.score_fields({"m": mean}, split.target, split.wdir_deg, arr, split.asymptote)
    sm = E2.sample_metrics(samples, split, arr)
    les = sc["les"]
    out = dict(S=S, calibration={g: CR.calibration_table(sm, les, groups[g]) for g in GROUPS},
               field_crps_median={g: float(np.median(field_crps[groups[g]])) for g in GROUPS},
               composite_vs_kljun={g: M.composite(sc["m"], sc_kljun, groups[g])[0] for g in GROUPS},
               medians={g: {k: float(np.nanmedian(M.error_of(sc["m"], k)[groups[g]]))
                            for k in M.METRIC_KEYS + ("rel_l2", "shape_l1_2d")} for g in GROUPS},
               spread={g: dict(array_share_sd_pp=float(np.median(100 * sm["array_share"][:, groups[g]].std(0, ddof=1))),
                               integral_sd=float(np.median(sm["integral"][:, groups[g]].std(0, ddof=1))))
                       for g in GROUPS},
               mean_share_pct=(100 * D.raster_array_share(mean, arr)).tolist())
    return out, sc["m"], sm


def z_sd_share(sT_sub, prep, split, arr, idx, tau):
    """Array-share z sd of the LES among temperature-scaled samples for records idx."""
    m = sT_sub.mean(0, keepdim=True)
    scaled = m + tau * (sT_sub - m)
    ph = prep.physical(scaled.cpu().numpy(), idx)
    share = np.stack([D.raster_array_share(p, arr) for p in ph]) * 100      # (S, m)
    les = 100 * D.raster_array_share(split.target[idx], arr)
    sd = share.std(0, ddof=1)
    ok = sd > 0
    z = (les[ok] - share.mean(0)[ok]) / sd[ok]
    return float(np.std(z))


def fit_tau(sT, prep, split, arr, idx, lo=0.3, hi=6.0, it=18):
    """Bisection on tau so that the array-share z sd on records idx is 1 (z sd falls with tau)."""
    sub = sT[:, idx]
    f = lambda tau: z_sd_share(sub, prep, split, arr, idx, tau) - 1.0
    flo, fhi = f(lo), f(hi)
    if flo < 0:
        return lo
    if fhi > 0:
        return hi
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def temperature_study(sT, prep, split, arr, groups, sc_kljun, mask_t, tgt_T, mean_share_pct):
    """Global and predictor-grouped tau, in-sample and 2-fold cross-fitted; the scaled
    samples evaluated like a model."""
    n = split.n
    all_idx = np.arange(n)
    pred_in = np.asarray(mean_share_pct) > 5.0            # predictor-side group (no LES needed)
    folds = [np.where(all_idx % 2 == 0)[0], np.where(all_idx % 2 == 1)[0]]
    res = {}
    # in-sample
    tau_g = fit_tau(sT, prep, split, arr, all_idx)
    tau_in = fit_tau(sT, prep, split, arr, np.where(pred_in)[0])
    tau_out = fit_tau(sT, prep, split, arr, np.where(~pred_in)[0])
    res["tau_global_in_sample"] = tau_g
    res["tau_grouped_in_sample"] = dict(pred_array_in_view=tau_in, pred_array_absent=tau_out,
                                        n_pred_in_view=int(pred_in.sum()))
    # cross-fitted: tau from the other fold
    tau_cf_g = np.empty(n)
    tau_cf_grp = np.empty(n)
    for a, b in ((0, 1), (1, 0)):
        fa, fb = folds[a], folds[b]
        tg = fit_tau(sT, prep, split, arr, fa)
        ti = fit_tau(sT, prep, split, arr, fa[pred_in[fa]])
        to = fit_tau(sT, prep, split, arr, fa[~pred_in[fa]])
        tau_cf_g[fb] = tg
        tau_cf_grp[fb] = np.where(pred_in[fb], ti, to)
        res[f"fold{a}_fit"] = dict(tau_global=tg, tau_in_view=ti, tau_absent=to)
    m = sT.mean(0, keepdim=True)
    variants = {
        "temp_global_in_sample": np.full(n, tau_g),
        "temp_grouped_in_sample": np.where(pred_in, tau_in, tau_out),
        "temp_global_crossfit": tau_cf_g,
        "temp_grouped_crossfit": tau_cf_grp,
    }
    evals = {}
    for name, taus in variants.items():
        tt = torch.from_numpy(taus.astype(np.float32)).to(sT.device).view(1, n, 1, 1)
        scaled = m + tt * (sT - m)
        ev, _, _ = evaluate_samples(scaled, prep, split, arr, groups, sc_kljun, mask_t, tgt_T)
        ev["tau"] = dict(median=float(np.median(taus)), min=float(taus.min()), max=float(taus.max()))
        evals[name] = ev
        del scaled
    res["variants"] = evals
    return res


def in_band(c, n):
    b = bands(n)
    return dict(cover50=bool(b["cover50"][0] <= c["cover50"] <= b["cover50"][1]),
                cover90=bool(b["cover90"][0] <= c["cover90"] <= b["cover90"][1]))


def md_tables(models, groups, base_name, base_range):
    L = []
    L += ["## Calibration of the array share (LES as one more draw among S samples)", "",
          "| model | group | n | S | cover50 | cover90 | in band | z sd | PIT KS p | CRPS [pp] | MAE of mean [pp] | spread/skill | sample sd [pp] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, ev in models.items():
        for g in REPORT_GROUPS:
            c = ev["calibration"][g]["array_share"]
            ib = in_band(c, c["n"])
            L.append(f"| {name} | {g} | {c['n']} | {c['S']} | {c['cover50']:.2f} | {c['cover90']:.2f} | "
                     f"{'y' if ib['cover50'] else 'n'}/{'y' if ib['cover90'] else 'n'} | {c['z_sd']:.2f} | {c['pit_ks_p']:.2g} | "
                     f"{c['crps_mean']:.3f} | {c['mae_of_mean']:.3f} | {c['spread_skill']:.2f} | {c['sample_sd_median']:.2f} |")
    L += ["", "## Calibration of the integral", "",
          "| model | group | n | cover50 | cover90 | in band | z sd | CRPS | MAE of mean | spread/skill |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, ev in models.items():
        for g in REPORT_GROUPS:
            c = ev["calibration"][g]["integral"]
            ib = in_band(c, c["n"])
            L.append(f"| {name} | {g} | {c['n']} | {c['cover50']:.2f} | {c['cover90']:.2f} | "
                     f"{'y' if ib['cover50'] else 'n'}/{'y' if ib['cover90'] else 'n'} | {c['z_sd']:.2f} | "
                     f"{c['crps_mean']:.4f} | {c['mae_of_mean']:.4f} | {c['spread_skill']:.2f} |")
    L += ["", "## Peak and centroid (z sd, cover90)", "", "| model | group | peak_x z sd | peak_x cover90 | centroid z sd | centroid cover90 |", "|---|---|---|---|---|---|"]
    for name, ev in models.items():
        for g in REPORT_GROUPS:
            p, c = ev["calibration"][g]["peak_x"], ev["calibration"][g]["centroid_dist"]
            L.append(f"| {name} | {g} | {p['z_sd']:.2f} | {p['cover90']:.2f} | {c['z_sd']:.2f} | {c['cover90']:.2f} |")
    L += ["", "## Field CRPS (asinh space, cone cells, median over records) and the mean's metrics", "",
          f"Baseline composite range over the five final seeds: [{base_range[0]:.3f}, {base_range[1]:.3f}] (rule: a mean has not regressed if inside it and val_mse_ref <= 1.20e-4).", "",
          "| model | group | field CRPS | composite vs Kljun | vs baseline composite (p) | array share [pp] | centroid [m] | overlap80 (1-J) | integral | peak_x [m] | rel L2 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, ev in models.items():
        for g in REPORT_GROUPS:
            md = ev["medians"][g]
            vb = ev.get("vs_baseline", {}).get(g)
            vbs = f"{vb['composite']:.3f} (p {vb['p_min']:.2g})" if vb else "-"
            L.append(f"| {name} | {g} | {ev['field_crps_median'][g]:.5f} | {ev['composite_vs_kljun'][g]:.3f} | {vbs} | "
                     f"{md['array_share']:.3f} | {md['centroid']:.1f} | {md['overlap80']:.3f} | {md['integral']:.3f} | {md['peak_x']:.0f} | {md['rel_l2']:.3f} |")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    ap.add_argument("--no-temperature", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--score-target", default="none", choices=["none", "sa99"],
                    help="sa99: score against the 99%% source-area-thresholded LES target and Kljun")
    a = ap.parse_args(argv)
    t0 = time.time()
    outdir = os.path.join(a.outdir, a.tag)
    os.makedirs(outdir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = D.load_split("val")
    st = D.load_statics()
    norm = D.read_norm()
    arr = st["array"] > 0.5
    thresh_info = None
    if a.score_target == "sa99":
        from ml_cfm import tailthresh as TT
        split.target, ti = TT.threshold_stack(split.target, 0.99)
        split.kljun, tk = TT.threshold_stack(split.kljun, 0.99)
        thresh_info = dict(les_median_mass_removed_pct=float(100 * np.median(ti["mass_removed_frac"])),
                           kljun_median_mass_removed_pct=float(100 * np.median(tk["mass_removed_frac"])))
    tr = D.load_split("train")
    groups = E.breakouts(split, np.isin(split.seed_key, list(set(tr.seed_key))))
    del tr
    sc_kljun = M.score_fields({"k": split.kljun}, split.target, split.wdir_deg, arr, split.asymptote)["k"]
    s_ref = float(norm["target_scale"])
    tgt_T = torch.from_numpy(np.arcsinh(split.target / s_ref).astype(np.float32)).to(dev)
    with open(os.path.join(REPO, "results", "ml_cfm", "eval", "final", "eval.json")) as fh:
        seed_comp = json.load(fh)["seed_composites"]
    base_range = (min(seed_comp.values()), max(seed_comp.values()))

    specs = [parse_spec(s) for s in a.models]
    results, base_sc, base_name = {}, None, specs[0]["name"]
    temperature = None
    for k, spec in enumerate(specs):
        sT, prep, used = draw(spec, split, st, norm, dev, a.S, a.seed)
        mask_t = prep.mask if prep.mask.dim() == 3 else prep.mask.unsqueeze(0).expand(split.n, -1, -1)
        ev, sc_m, sm = evaluate_samples(sT, prep, split, arr, groups, sc_kljun, mask_t, tgt_T)
        ev["sampling"] = used
        with open(os.path.join(spec["dir"], "run.json")) as fh:
            r = json.load(fh)
        ev["run"] = dict(val_mse_ref=r["val_mse_ref"], val_crps=r.get("val_crps"), best_epoch=r["best_epoch"],
                         epochs_run=r["epochs_run"], gap=r["gap"]["loss_ratio"], wall_s=r["wall_s"],
                         config={kk: r["config"][kk] for kk in ("loss", "lam_crps", "lam_share", "crps_S", "crps_steps",
                                                                "sigma", "init_from", "target_thresh", "select", "lr", "batch")
                                 if kk in r["config"]})
        if k == 0:
            base_sc = sc_m
        else:
            ev["vs_baseline"] = {}
            for g in GROUPS:
                cmpg = E.compare(sc_m, base_sc, groups[g], M.METRIC_KEYS)
                ev["vs_baseline"][g] = dict(composite=M.composite(sc_m, base_sc, groups[g])[0],
                                            p_min=float(np.nanmin([c["wilcoxon_p"] for c in cmpg.values()])),
                                            per_metric={kk: dict(ratio=c["ratio"], p=c["wilcoxon_p"], win=c["win_frac"])
                                                        for kk, c in cmpg.items()})
        ev["regressed"] = not (base_range[0] <= ev["composite_vs_kljun"]["all"] <= base_range[1] + 1e-9
                               and r["val_mse_ref"] <= 1.20e-4)
        results[spec["name"]] = ev
        print(f"{spec['name']}: {used} | composite {ev['composite_vs_kljun']['all']:.3f} | "
              f"array-in-view cover50/90 {ev['calibration']['array_in_view_gt5pct']['array_share']['cover50']:.2f}/"
              f"{ev['calibration']['array_in_view_gt5pct']['array_share']['cover90']:.2f} | {round(time.time()-t0)} s", flush=True)
        if k == 0 and not a.no_temperature:
            temperature = temperature_study(sT, prep, split, arr, groups, sc_kljun, mask_t, tgt_T, ev["mean_share_pct"])
            for name, ev2 in temperature["variants"].items():
                ev2["sampling"] = dict(used, temperature=name)
                results[base_name + "+" + name] = ev2
            print("temperature:", {kk: v for kk, v in temperature.items() if kk != "variants"}, flush=True)
        del sT, prep
        torch.cuda.empty_cache()

    # verdicts
    verd = {}
    for name, ev in results.items():
        c = {g: in_band(ev["calibration"][g]["array_share"], ev["calibration"][g]["array_share"]["n"]) for g in REPORT_GROUPS}
        verd[name] = dict(coverage_in_band=c, all_in_band=all(v["cover50"] and v["cover90"] for v in c.values()),
                          regressed=ev.get("regressed", False))
    temp_fix = None
    if temperature is not None:
        v = verd[base_name + "+temp_global_crossfit"]
        vg = verd[base_name + "+temp_grouped_crossfit"]
        temp_fix = dict(global_crossfit_fixes_coverage=v["all_in_band"], grouped_crossfit_fixes_coverage=vg["all_in_band"],
                        tau_global=temperature["tau_global_in_sample"], tau_grouped=temperature["tau_grouped_in_sample"])
    out = dict(tag=a.tag, S=a.S, seed=a.seed, baseline=base_name, score_target=a.score_target, score_target_info=thresh_info, base_composite_range=base_range,
               groups={g: int(groups[g].sum()) for g in GROUPS}, bands={g: bands(int(groups[g].sum())) for g in GROUPS},
               models=results, verdicts=verd, temperature=temperature, temperature_verdict=temp_fix,
               wall_s=round(time.time() - t0))
    with open(os.path.join(outdir, "calib.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    L = [f"# CFM calibration `{a.tag}`: {len(results)} model variants, S = {a.S} samples each, val ({split.n} records)", "",
         "Bands: +-2 binomial sd around nominal — " + "; ".join(
             f"{g} (n {int(groups[g].sum())}): cover50 [{bands(int(groups[g].sum()))['cover50'][0]:.2f}, {bands(int(groups[g].sum()))['cover50'][1]:.2f}], "
             f"cover90 [{bands(int(groups[g].sum()))['cover90'][0]:.2f}, {bands(int(groups[g].sum()))['cover90'][1]:.2f}]" for g in REPORT_GROUPS), "",
         "| model | loss | init | sigma train / sample | steps | target | best epoch | val_mse_ref | val CRPS | ms/record/sample |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, ev in results.items():
        u, rr = ev["sampling"], ev.get("run", {})
        L.append(f"| {name} | {u['loss']} | {os.path.basename(os.path.dirname(u['init_from'])) if u['init_from'] else 'scratch'} | "
                 f"{u['sigma_train']} / {u['sigma_sample']} | {u['steps']} | {u['target_thresh']} | {rr.get('best_epoch', '-')} | "
                 f"{rr.get('val_mse_ref', float('nan')):.3e} | {(rr.get('val_crps') or float('nan')):.4f} | {u['ms_per_record_per_sample']:.1f} |")
    L += [""] + md_tables(results, groups, base_name, base_range)
    L += ["", "## Verdicts", ""]
    for name, v in verd.items():
        c = v["coverage_in_band"]
        L.append(f"- **{name}**: array-share coverage in band — " + ", ".join(
            f"{g} {'50✓' if c[g]['cover50'] else '50✗'}/{'90✓' if c[g]['cover90'] else '90✗'}" for g in REPORT_GROUPS)
            + f"; mean regressed: {'YES' if v['regressed'] else 'no'}")
    if temp_fix:
        L += ["", f"**Temperature scaling alone fixes coverage (cross-fitted): global tau = {temp_fix['tau_global']:.2f} -> "
              f"{'YES' if temp_fix['global_crossfit_fixes_coverage'] else 'NO'}; grouped tau "
              f"(pred. array in view {temp_fix['tau_grouped']['pred_array_in_view']:.2f} / absent {temp_fix['tau_grouped']['pred_array_absent']:.2f}) -> "
              f"{'YES' if temp_fix['grouped_crossfit_fixes_coverage'] else 'NO'}.**"]
    with open(os.path.join(outdir, "calib.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    if not a.no_figures:
        figures(outdir, results, groups)
    print("done", round(time.time() - t0), "s")
    return 0


def figures(outdir, results, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, g in zip(axes, REPORT_GROUPS):
        n = int(groups[g].sum())
        b = bands(n)
        c50 = [results[m]["calibration"][g]["array_share"]["cover50"] for m in names]
        c90 = [results[m]["calibration"][g]["array_share"]["cover90"] for m in names]
        x = np.arange(len(names))
        ax.axhspan(*b["cover50"], color="#4c72b0", alpha=0.12); ax.axhspan(*b["cover90"], color="#c44e52", alpha=0.12)
        ax.axhline(0.5, color="#4c72b0", lw=0.8); ax.axhline(0.9, color="#c44e52", lw=0.8)
        ax.plot(x, c50, "o", color="#4c72b0", label="cover 50%"); ax.plot(x, c90, "s", color="#c44e52", label="cover 90%")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6.5)
        ax.set_title(f"array share, {g} (n = {n}); bands = ±2 binomial sd", fontsize=8); ax.tick_params(labelsize=7)
    axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "coverage.png"), dpi=130); plt.close(fig)
    fig, axes = plt.subplots(2, len(names), figsize=(2.6 * len(names), 5), squeeze=False)
    for j, m in enumerate(names):
        for i, g in enumerate(("all", "array_in_view_gt5pct")):
            c = results[m]["calibration"][g]["array_share"]
            axes[i][j].bar(np.arange(10) / 10 + 0.05, c["pit_hist"], width=0.1, color="#9467bd")
            axes[i][j].axhline(c["n"] / 10, color="k", lw=0.8)
            axes[i][j].set_title(f"{m}\n{g}: z sd {c['z_sd']:.2f}, c90 {c['cover90']:.2f}", fontsize=6.5)
            axes[i][j].tick_params(labelsize=6)
    fig.suptitle("PIT of the LES array share among the samples (flat = calibrated)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(outdir, "pit.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
