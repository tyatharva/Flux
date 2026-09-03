"""From the S-curve of ml_cfm.sample_count: the val noise floor, the saturation S, and
bootstrap bands -- then the headline tables at that S.

    python -m ml_cfm.sample_saturation [--S-use N]

Floor = sd of the composite under a bootstrap over the val RECORDS (the metric's own
sampling noise on this split). Saturation S_sat = the S at which the fitted improvement
still to come, b S^-p, equals the floor: beyond it the curve is inside the noise of the
split and a lower val value is selection, not estimation. Bands: parametric bootstrap of the
curve (y_S ~ N(mean_S, sd_S) from the subset repeats), refit, 2.5-97.5%. S is then chosen
from the fit (the upper band of S_sat, rounded up), never as the val argmin. Val only.
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
from ml_cfm import sample_count as SC         # noqa: E402

DIR = os.path.join(REPO, "results", "ml_cfm", "calib", "samples")


def record_bootstrap(sc_m, sc_k, mask, n_boot, rng):
    idx = np.where(mask)[0]
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        mm = np.zeros(len(mask), bool)
        # composite() takes a mask; a bootstrap needs repeats, so index the score arrays
        sm = {k: v[b] for k, v in sc_m.items()}
        sk = {k: v[b] for k, v in sc_k.items()}
        vals.append(M.composite(sm, sk)[0])
    return float(np.std(vals, ddof=1)), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", nargs="+", default=sorted(glob.glob(os.path.join(REPO, "results", "ml_cfm", "final", "seed?"))))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-curve", type=int, default=1000)
    ap.add_argument("--S-use", type=int, default=None, help="override the S chosen from the fit")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    with open(os.path.join(DIR, "sample_count.json")) as fh:
        J = json.load(fh)
    split = D.load_split("val")
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    tr = D.load_split("train")
    groups = {g: np.array(m) for g, m in E.breakouts(split, np.isin(split.seed_key, list(set(tr.seed_key)))).items()
              if g in ("all", "north_N_NE_NW", "array_in_view_gt5pct")}
    del tr
    sc_k = M.score_fields({"k": split.kljun}, split.target, split.wdir_deg, arr, split.asymptote)["k"]

    # pooled samples: stored 32 + the extra per seed
    pooled = []
    for sd in a.seeds:
        with np.load(os.path.join(sd, "samples_val.npz")) as z:
            T0, s_out = z["samples_T"].astype(np.float32), z["s_out"].astype(np.float32)
        ex = sorted(glob.glob(os.path.join(sd, "samples_val_extra*.npz")))
        T1 = [np.load(p)["samples_T"].astype(np.float32) for p in ex]
        T = np.concatenate([T0] + T1)
        pooled.append((s_out[None, :, None, None] * np.sinh(np.clip(T, -20, 20)) * valid[None, None]).astype(np.float32))
    pooled = np.concatenate(pooled)
    S_max = pooled.shape[0]
    sc_full = M.score_fields({"m": pooled.mean(0)}, split.target, split.wdir_deg, arr, split.asymptote)["m"]

    out = {"S_max": int(S_max), "groups": {}}
    for g, m in groups.items():
        floor, band = record_bootstrap(sc_full, sc_k, m, a.n_boot, rng)
        f = J["fits"]["pooled"][f"{g}/composite"]
        law = f["law"]
        p = 0.5 if law == "sqrt" else 1.0
        cv = J["curves"]["pooled"]
        S = np.array(sorted(int(s) for s in cv))
        mu = np.array([cv[str(s)][g]["composite"][0] for s in S])
        sd = np.array([cv[str(s)][g]["composite"][1] for s in S])
        sd = np.where(sd > 0, sd, sd[sd > 0].min() if (sd > 0).any() else 1e-4)
        A, B, Ssat = [], [], []
        for _ in range(a.n_curve):
            y = rng.normal(mu, sd)
            aa, bb = SC.fit_law(S, y, p)
            A.append(aa); B.append(bb)
            Ssat.append((bb / floor) ** (1 / p) if bb > 0 else np.nan)
        A, B, Ssat = np.array(A), np.array(B), np.array(Ssat)
        s_sat = float((f["b"] / floor) ** (1 / p)) if f["b"] > 0 else float("nan")
        out["groups"][g] = dict(
            n=int(m.sum()), law=law, composite_at_S_max=float(M.composite(sc_full, sc_k, m)[0]),
            composite_record_bootstrap_ci95=band, val_noise_floor_sd=floor,
            asymptote=f["asymptote"], asymptote_ci95=[float(np.nanpercentile(A, 2.5)), float(np.nanpercentile(A, 97.5))],
            b=f["b"], S_sat=s_sat, S_sat_ci95=[float(np.nanpercentile(Ssat, 2.5)), float(np.nanpercentile(Ssat, 97.5))],
            S_half_floor=float((f["b"] / (0.5 * floor)) ** (1 / p)) if f["b"] > 0 else float("nan"),
            excess_at_S_max=float(f["b"] * S_max ** -p), excess_at_S_max_over_floor=float(f["b"] * S_max ** -p / floor))
    # the S to use: the upper band of S_sat on the 'all' group, rounded up to a multiple of 10
    s_up = out["groups"]["all"]["S_sat_ci95"][1]
    S_use = a.S_use or int(min(S_max, 10 * np.ceil(s_up / 10)))
    out["S_use"] = S_use
    out["S_use_rule"] = "upper 97.5% band of S_sat on all records, rounded up to a multiple of 10, capped at S_max"

    # headline at S_use: a fixed subset (the first S_use pooled samples, seeds interleaved) vs Kljun and the FNO
    # ten random S_use-subsets: composites as mean +- sd over subsets (the estimator's own noise at
    # S_use, which by construction is about one floor); the paired comparisons from the first subset
    fno = np.mean([np.load(pth)["fno"] for pth in sorted(glob.glob(os.path.join(REPO, "results", "ml", "final", "seed*", "pred_val.npz")))], axis=0)
    keys = M.METRIC_KEYS + M.SHAPE_KEYS + ("rel_l2",)
    n_sub = 10
    comps = {g: dict(cfm=[], fno=None, cfm_vs_fno=[]) for g in groups}
    sc0 = None
    for r in range(n_sub):
        idx = rng.choice(S_max, S_use, replace=False)
        sc = M.score_fields({"cfm": pooled[idx].mean(0), "fno": fno.astype(np.float32)}, split.target, split.wdir_deg, arr, split.asymptote)
        sc0 = sc0 or sc
        for g, m in groups.items():
            comps[g]["cfm"].append(M.composite(sc["cfm"], sc_k, m)[0])
            comps[g]["fno"] = M.composite(sc["fno"], sc_k, m)[0]
            comps[g]["cfm_vs_fno"].append(M.composite(sc["cfm"], sc["fno"], m)[0])
    head = {}
    for g, m in groups.items():
        head[g] = dict(n=int(m.sum()), n_subsets=n_sub,
                       composite_cfm=float(np.mean(comps[g]["cfm"])), composite_cfm_sd=float(np.std(comps[g]["cfm"], ddof=1)),
                       composite_fno=comps[g]["fno"],
                       composite_cfm_vs_fno=float(np.mean(comps[g]["cfm_vs_fno"])), composite_cfm_vs_fno_sd=float(np.std(comps[g]["cfm_vs_fno"], ddof=1)),
                       vs_kljun=E.compare(sc0["cfm"], sc_k, m, keys), vs_fno=E.compare(sc0["cfm"], sc0["fno"], m, keys))
    out["headline_at_S_use"] = head
    with open(os.path.join(DIR, "sample_saturation.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    L = [f"# Saturation of the CFM sample mean on val (pooled {S_max} samples from {len(a.seeds)} seeds)", "",
         "Floor = record-bootstrap sd of the composite (2000 resamples of the val records). S_sat = the S at which the fitted "
         "remaining improvement b S^-p equals the floor. Bands: parametric bootstrap of the curve (1000 refits), 2.5-97.5%.", "",
         "| group | n | composite at S_max [record-bootstrap 95%] | floor (sd) | law | asymptote [95%] | S_sat [95%] | S at half the floor | excess at S_max / floor |",
         "|---|---|---|---|---|---|---|---|---|"]
    for g, v in out["groups"].items():
        L.append(f"| {g} | {v['n']} | {v['composite_at_S_max']:.3f} [{v['composite_record_bootstrap_ci95'][0]:.3f}, {v['composite_record_bootstrap_ci95'][1]:.3f}] | "
                 f"{v['val_noise_floor_sd']:.4f} | 1/{'sqrt(S)' if v['law'] == 'sqrt' else 'S'} | {v['asymptote']:.3f} [{v['asymptote_ci95'][0]:.3f}, {v['asymptote_ci95'][1]:.3f}] | "
                 f"{v['S_sat']:.0f} [{v['S_sat_ci95'][0]:.0f}, {v['S_sat_ci95'][1]:.0f}] | {v['S_half_floor']:.0f} | {v['excess_at_S_max_over_floor']:.2f} |")
    L += ["", f"**S chosen from the fit: {S_use}** ({out['S_use_rule']}).", "",
          f"## Headline at S = {S_use}: composite over {n_sub} random {S_use}-subsets (mean ± sd), paired tests from the first subset", "",
          "| group | n | CFM | FNO | CFM/FNO | array share pp CFM / FNO / Kljun (p CFM vs FNO) | centroid m (p) | shape L1 (p) | rel L2 (p) |", "|---|---|---|---|---|---|---|---|---|"]
    for g, h in head.items():
        vk, vf = h["vs_kljun"], h["vs_fno"]
        L.append(f"| {g} | {h['n']} | {h['composite_cfm']:.3f} ± {h['composite_cfm_sd']:.3f} | {h['composite_fno']:.3f} | {h['composite_cfm_vs_fno']:.3f} ± {h['composite_cfm_vs_fno_sd']:.3f} | "
                 f"{vf['array_share']['model_median']:.3f} / {vf['array_share']['ref_median']:.3f} / {vk['array_share']['ref_median']:.3f} ({vf['array_share']['wilcoxon_p']:.2g}) | "
                 f"{vf['centroid']['model_median']:.1f} vs {vf['centroid']['ref_median']:.1f} ({vf['centroid']['wilcoxon_p']:.2g}) | "
                 f"{vf['shape_l1_2d']['model_median']:.3f} vs {vf['shape_l1_2d']['ref_median']:.3f} ({vf['shape_l1_2d']['wilcoxon_p']:.2g}) | "
                 f"{vf['rel_l2']['model_median']:.3f} vs {vf['rel_l2']['ref_median']:.3f} ({vf['rel_l2']['wilcoxon_p']:.2g}) |")
    with open(os.path.join(DIR, "sample_saturation.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
