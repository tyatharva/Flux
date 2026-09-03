"""How many samples does the CFM mean need? Draw extra samples per seed, score the sample
mean against the LES at every S, fit the convergence law, report the asymptote and the S
where the curve is within its own Monte-Carlo band of it.

    python -m ml_cfm.sample_count [--extra 128] [--steps 16]

Each sample is one ODE integration from an independent noise draw z_0 = x_prior + eps. The
S-sample mean estimates the model's conditional expectation with a Monte-Carlo error that
falls as 1/sqrt(S); its error against the LES therefore behaves as err(S) = a + b / sqrt(S)
(or b / S for a quadratic metric), where a is the floor (model bias plus the LES's own
realisation noise) that no S can remove. Per seed and pooled over seeds, so the sampling
Monte-Carlo error and the seed-to-seed model error are separated. Val only; the test
split is never read.
"""
import argparse
import glob
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
from ml_cfm import infer as I                 # noqa: E402

OUT = os.path.join(REPO, "results", "ml_cfm", "calib", "samples")
KEYS = ("array_share", "centroid", "overlap80", "integral", "shape_l1_2d", "rel_l2")


def composite_curve(samples_phys, split, arr, sc_k, groups, S_list, rng, n_rep):
    """samples_phys (S_total, n, H, W) -> {S: {group: {composite: (mean, sd), medians...}}}."""
    S_total = samples_phys.shape[0]
    out = {}
    for S in S_list:
        reps = []
        for r in range(n_rep if S < S_total else 1):
            idx = rng.choice(S_total, S, replace=False) if S < S_total else np.arange(S_total)
            mean = samples_phys[idx].mean(0)
            sc = M.score_fields({"m": mean}, split.target, split.wdir_deg, arr, split.asymptote)["m"]
            reps.append({g: dict(composite=M.composite(sc, sc_k, groups[g])[0],
                                 **{k: float(np.nanmedian(M.error_of(sc, k)[groups[g]])) for k in KEYS})
                         for g in groups})
        out[S] = {g: {k: [float(np.mean([r[g][k] for r in reps])), float(np.std([r[g][k] for r in reps], ddof=1) if len(reps) > 1 else 0.0)]
                      for k in ("composite",) + KEYS} for g in groups}
        print(f"  S {S:4d}: composite {out[S]['all']['composite'][0]:.4f} +- {out[S]['all']['composite'][1]:.4f}", flush=True)
    return out


def fit_law(S, y, power=0.5):
    """y = a + b S^-power by least squares; returns a, b."""
    X = np.stack([np.ones_like(S, float), np.asarray(S, float) ** -power], 1)
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return float(coef[0]), float(coef[1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", nargs="+", default=sorted(glob.glob(os.path.join(REPO, "results", "ml_cfm", "final", "seed?"))))
    ap.add_argument("--extra", type=int, default=128, help="new samples per seed (added to the stored 32)")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=4243)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--outdir", default=OUT)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(a.outdir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = D.load_split("val")
    st = D.load_statics()
    norm = D.read_norm()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    tr = D.load_split("train")
    groups = {g: m for g, m in E.breakouts(split, np.isin(split.seed_key, list(set(tr.seed_key)))).items()
              if g in ("all", "north_N_NE_NW", "array_in_view_gt5pct")}
    del tr
    sc_k = M.score_fields({"k": split.kljun}, split.target, split.wdir_deg, arr, split.asymptote)["k"]

    per_seed, secs_total = {}, 0.0
    for sd in a.seeds:
        name = os.path.basename(sd.rstrip("/"))
        extra_path = os.path.join(sd, f"samples_val_extra{a.extra}_e{a.steps}.npz")
        with np.load(os.path.join(sd, "samples_val.npz")) as z:
            assert np.array_equal(z["run_id"], split.meta["run_id"])
            T0 = z["samples_T"].astype(np.float32)
            s_out = z["s_out"].astype(np.float32)
        if os.path.exists(extra_path):
            with np.load(extra_path) as z:
                T1, secs = z["samples_T"].astype(np.float32), float(z["seconds"])
        else:
            cfg, model, _ = I.load_checkpoint(os.path.join(sd, "best.pt"), dev)
            prep = I.Prepared(cfg, split, st, norm, dev)
            sT, secs = prep.samples(model, cfg, a.extra, a.steps, "euler", a.seed)
            T1 = sT.cpu().numpy().astype(np.float32)
            np.savez_compressed(extra_path, samples_T=T1.astype(np.float16), run_id=split.meta["run_id"],
                                s_out=s_out, seconds=secs, steps=a.steps, seed=a.seed)
            del model, prep, sT
            torch.cuda.empty_cache()
        secs_total += secs
        T = np.concatenate([T0, T1])
        per_seed[name] = (s_out[None, :, None, None] * np.sinh(np.clip(T, -20, 20)) * valid[None, None]).astype(np.float32)
        print(f"{name}: {T.shape[0]} samples ({secs:.0f} s for the {a.extra} new)", flush=True)

    rng = np.random.default_rng(0)
    S_seed = [1, 2, 4, 8, 16, 32, 64, 128, per_seed[next(iter(per_seed))].shape[0]]
    curves = {"per_seed": {}, "pooled": None}
    for name, ph in per_seed.items():
        print(name, flush=True)
        curves["per_seed"][name] = composite_curve(ph, split, arr, sc_k, groups, S_seed, rng, a.reps)
    pooled = np.concatenate(list(per_seed.values()))
    S_pool = [5, 10, 20, 40, 80, 160, 320, 640, pooled.shape[0]]
    print("pooled", pooled.shape[0], flush=True)
    curves["pooled"] = composite_curve(pooled, split, arr, sc_k, groups, S_pool, rng, a.reps)

    # fits: composite (median-of-L1-type metrics -> 1/sqrt(S)), rel_l2 (quadratic -> also fit 1/S)
    fits = {}
    for scope, cv in (("pooled", curves["pooled"]),) + tuple((f"per_seed/{k}", v) for k, v in curves["per_seed"].items()):
        S = np.array(sorted(cv))
        fits[scope] = {}
        for g in groups:
            for k in ("composite", "rel_l2", "shape_l1_2d", "array_share"):
                y = np.array([cv[s][g][k][0] for s in S])
                a5, b5 = fit_law(S, y, 0.5)
                a1, b1 = fit_law(S, y, 1.0)
                res5 = float(np.sqrt(np.mean((y - (a5 + b5 * S ** -0.5)) ** 2)))
                res1 = float(np.sqrt(np.mean((y - (a1 + b1 / S)) ** 2)))
                best = "sqrt" if res5 <= res1 else "inv"
                a_, b_ = (a5, b5) if best == "sqrt" else (a1, b1)
                p = 0.5 if best == "sqrt" else 1.0
                # S at which the fitted excess over the asymptote is 1% and 2% of the asymptote
                s1 = float((b_ / (0.01 * a_)) ** (1 / p)) if a_ > 0 and b_ > 0 else float("nan")
                s2 = float((b_ / (0.02 * a_)) ** (1 / p)) if a_ > 0 and b_ > 0 else float("nan")
                fits[scope][f"{g}/{k}"] = dict(law=best, asymptote=a_, b=b_, rms_resid=min(res5, res1),
                                               S_within_1pct=s1, S_within_2pct=s2,
                                               at_max_S=float(y[-1]), excess_at_max_S_pct=float(100 * (y[-1] - a_) / a_) if a_ > 0 else float("nan"))
    out = dict(seeds=a.seeds, extra_per_seed=a.extra, steps=a.steps, S_per_seed=int(pooled.shape[0] // len(per_seed)),
               S_pooled=int(pooled.shape[0]), sampling_seconds_total=secs_total,
               ms_per_record_per_sample=1000 * secs_total / (split.n * a.extra * len(per_seed)),
               curves={"per_seed": {k: {str(s): v for s, v in cv.items()} for k, cv in curves["per_seed"].items()},
                       "pooled": {str(s): v for s, v in curves["pooled"].items()}},
               fits=fits, wall_s=round(time.time() - t0))
    with open(os.path.join(a.outdir, "sample_count.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    L = [f"# Sample count study: {len(per_seed)} seeds x {out['S_per_seed']} samples (Euler {a.steps}), val", "",
         f"Sampling cost {out['ms_per_record_per_sample']:.1f} ms per record per sample. Bands: sd over {a.reps} random subsets.", "",
         "## Pooled over seeds: composite vs Kljun of the S-sample mean", "",
         "| S | " + " | ".join(groups) + " | rel_l2 (all) | shape_l1_2d (all) | array_share pp (all) |", "|---|" + "---|" * (len(groups) + 3)]
    for s in sorted(curves["pooled"]):
        c = curves["pooled"][s]
        L.append(f"| {s} | " + " | ".join(f"{c[g]['composite'][0]:.3f} ± {c[g]['composite'][1]:.3f}" for g in groups)
                 + f" | {c['all']['rel_l2'][0]:.3f} | {c['all']['shape_l1_2d'][0]:.3f} | {c['all']['array_share'][0]:.3f} |")
    L += ["", "## Per seed: composite vs Kljun (all records)", "", "| S | " + " | ".join(curves["per_seed"]) + " |", "|---|" + "---|" * len(curves["per_seed"])]
    for s in S_seed:
        L.append(f"| {s} | " + " | ".join(f"{cv[s]['all']['composite'][0]:.3f}" for cv in curves["per_seed"].values()) + " |")
    L += ["", "## Convergence law fits, err(S) = a + b S^-p (p = 1/2 or 1, whichever fits better)", "",
          "| scope | group / metric | law | asymptote a | at max S | excess at max S | S for +1% | S for +2% | rms resid |", "|---|---|---|---|---|---|---|---|---|"]
    for scope, f in fits.items():
        for key, v in f.items():
            if key.endswith("/composite") or key == "all/rel_l2" or key == "all/array_share":
                L.append(f"| {scope} | {key} | 1/{'sqrt(S)' if v['law'] == 'sqrt' else 'S'} | {v['asymptote']:.4f} | {v['at_max_S']:.4f} | "
                         f"{v['excess_at_max_S_pct']:.1f}% | {v['S_within_1pct']:.0f} | {v['S_within_2pct']:.0f} | {v['rms_resid']:.4f} |")
    with open(os.path.join(a.outdir, "sample_count.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, g in zip(axes, groups):
        for name, cv in curves["per_seed"].items():
            S = sorted(cv)
            ax.errorbar(S, [cv[s][g]["composite"][0] for s in S], [cv[s][g]["composite"][1] for s in S], fmt="o-", ms=3, lw=0.8, alpha=0.6, label=name)
        S = sorted(curves["pooled"])
        ax.errorbar(S, [curves["pooled"][s][g]["composite"][0] for s in S], [curves["pooled"][s][g]["composite"][1] for s in S], fmt="s-", color="k", ms=4, lw=1.4, label="pooled")
        f = fits["pooled"][f"{g}/composite"]
        Sx = np.logspace(0, np.log10(max(S)) + 0.5, 100)
        ax.plot(Sx, f["asymptote"] + f["b"] * Sx ** (-0.5 if f["law"] == "sqrt" else -1.0), "--", color="gray", lw=0.9, label=f"fit, a = {f['asymptote']:.3f}")
        ax.axhline(f["asymptote"], color="gray", lw=0.6)
        ax.set_xscale("log"); ax.set_xlabel("samples S in the mean"); ax.set_title(f"composite vs Kljun, {g} (n = {int(groups[g].sum())})", fontsize=9)
        ax.tick_params(labelsize=7); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "sample_count.png"), dpi=130)
    print("done", round(time.time() - t0), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
