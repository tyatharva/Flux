"""The reporting metrics, on the frozen recipe (ml_cfm/final_recipe.py): the training losses
(stated), the composite of the five production quantities, and four field metrics on the
log-scale grid -- log-MSE, sliced Wasserstein-1, KL(LES || model), MS-SSIM -- each against
the LES target and each beside its two-window realisation floor.

    python -m ml_cfm.report_metrics [--split val]

Processing before scoring, as frozen: Kljun raw; FNO (mean of 4 seeds) and CFM (mean of 80
tau-scaled samples from 4 seeds) cut at their own 99.5% source area; LES positive-only.
Field metrics are over the 122^2 interior. Refuses --split test without --allow-test.
"""
import argparse
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
from ml_cfm import final_recipe as FR         # noqa: E402

EPS = 1e-9            # m^-2: the log floor. ~4e-5 of the median LES peak (2.6e-5), 1.5 decades under the 99.5% cut level
N_PROJ = 64           # sliced-Wasserstein projections
MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)   # Wang et al. 2003
GROUPS = ("all", "north_N_NE_NW", "array_in_view_gt5pct")
FIELD_KEYS = ("log_mse", "log_mse_body", "sw1_m", "kl_nats", "ms_ssim", "rel_l2")

LOSSES = {
    "FNO": "masked MSE + 0.03 x masked MAE, both in asinh space (global target scale) over the 122^2 interior, "
           "on the residual to Kljun; unweighted; selection on val masked MSE.",
    "CFM": "conditional flow matching, velocity parameterisation: z_t = x_K + t (x_LES - x_K) + (1 - t) eps, "
           "eps ~ N(0, 0.1^2) on the cone cells, target v = (x_LES - x_K) - eps, MSE over the cone cells, t ~ U(0,1); "
           "asinh space anchored on Kljun; unweighted; selection on the val MSE of the 16-step Euler sample mean.",
}


def _interior():
    v = np.zeros((D.N, D.N), bool)
    v[D.PAD:D.PAD + D.NG, D.PAD:D.PAD + D.NG] = True
    return v


def log_field(f):
    return np.log10(np.maximum(np.asarray(f, np.float64), 0) + EPS)


def log_mse(f, ref, v):
    return float(((log_field(f) - log_field(ref))[v] ** 2).mean())


def log_mse_body(f, ref, v, frac=0.995):
    """log-MSE on the cells inside the reference's own 99.5% source area (the trusted body)."""
    body = v & (np.maximum(np.asarray(ref, np.float64), 0) >= TT.source_area_level(ref, frac))
    return float(((log_field(f) - log_field(ref))[body] ** 2).mean())


def kl_nats(ref, f, v):
    """KL(P_ref || Q_f) in nats over the interior; both positive parts, Q smoothed by EPS."""
    p = np.maximum(np.asarray(ref, np.float64), 0)[v]; p = p / p.sum()
    q = np.maximum(np.asarray(f, np.float64), 0)[v] + EPS; q = q / q.sum()
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


_PROJ = None


def sliced_w1(f, ref, v, X, Y):
    """Mean over N_PROJ fixed directions of the 1-D W1 [m] between the unit-mass positive parts."""
    global _PROJ
    if _PROJ is None:
        th = np.linspace(0, np.pi, N_PROJ, endpoint=False)
        _PROJ = np.stack([np.cos(th), np.sin(th)], 1)                         # (P,2)
    a = np.maximum(np.asarray(f, np.float64), 0)[v]; a = a / a.sum()
    b = np.maximum(np.asarray(ref, np.float64), 0)[v]; b = b / b.sum()
    xy = np.stack([X[v], Y[v]], 1)                                            # (n,2)
    proj = xy @ _PROJ.T                                                       # (n,P)
    order = np.argsort(proj, axis=0)
    ps = np.take_along_axis(proj, order, 0)
    Fa = np.cumsum(a[order], axis=0)
    Fb = np.cumsum(b[order], axis=0)
    return float((np.abs(Fa - Fb)[:-1] * np.diff(ps, axis=0)).sum(0).mean())


def ms_ssim(f, ref, v):
    """Multi-scale SSIM (Wang, Simoncelli & Bovik 2003) on the log10 grid, 5 scales, 2x2 mean pooling."""
    from scipy.ndimage import gaussian_filter
    sl = slice(D.PAD, D.PAD + D.NG)
    a, b = log_field(f)[sl, sl], log_field(ref)[sl, sl]
    L = float(max(a.max(), b.max()) - np.log10(EPS))
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    vals = []
    for k in range(len(MS_WEIGHTS)):
        g = lambda x: gaussian_filter(x, 1.5, truncate=3.5, mode="reflect")
        mu_a, mu_b = g(a), g(b)
        saa, sbb, sab = g(a * a) - mu_a ** 2, g(b * b) - mu_b ** 2, g(a * b) - mu_a * mu_b
        cs = ((2 * sab + c2) / (saa + sbb + c2)).mean()
        lum = ((2 * mu_a * mu_b + c1) / (mu_a ** 2 + mu_b ** 2 + c1)).mean()
        vals.append((lum, cs))
        if k < len(MS_WEIGHTS) - 1:
            h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
            a = a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
            b = b[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
    out = vals[-1][0] ** MS_WEIGHTS[-1]
    for (lum, cs), wgt in zip(vals, MS_WEIGHTS):
        out *= max(cs, 1e-12) ** wgt
    return float(out)


def field_metrics(f, ref, v, X, Y):
    return dict(log_mse=log_mse(f, ref, v), log_mse_body=log_mse_body(f, ref, v), sw1_m=sliced_w1(f, ref, v, X, Y), kl_nats=kl_nats(ref, f, v),
                ms_ssim=ms_ssim(f, ref, v), rel_l2=M.image_metrics(f, ref)["rel_l2"])


def recipe_fields(split, valid):
    R = FR.RECIPE
    samples = FR.cfm_samples(split, R["cfm_seeds"], R["samples_per_seed"], R["tau"], valid)
    cfm, _ = TT.threshold_stack(samples.mean(0), R["cut_frac"])
    fno_raw = np.mean([np.load(os.path.join(REPO, "results", "ml", "final", sd, f"pred_{split.name}.npz"))["fno"]
                       for sd in R["fno_seeds"]], axis=0).astype(np.float32)
    fno, _ = TT.threshold_stack(fno_raw, R["cut_frac"])
    les = np.maximum(split.target, 0).astype(np.float32)
    return {"Kljun": split.kljun, "FNO": fno, "CFM": cfm}, les


def pair_floor(arr, v, X, Y):
    """The two-window pair (a train record), both windows positive-only, w1 scored against w0."""
    ws, keep, _, _ = TT._load_pair()
    w0, w1 = (np.maximum(w["target"], 0) for w in ws)
    wd = float(ws[0]["meta"]["wdir_deg"])
    asym = 1.0 - D.Z_RECEPTOR / float(ws[0]["scalars"][0])
    prod = M.pair_errors(w1, w0, wd, arr, asym)
    out = {k: float(prod[k]) for k in M.METRIC_KEYS + M.SHAPE_KEYS}
    out.update(field_metrics(w1, w0, v, X, Y))
    return out


def main(argv=None):
    global EPS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--outdir", default=FR.OUT)
    a = ap.parse_args(argv)
    if a.split == "test" and not a.allow_test:
        raise SystemExit("refusing the test split without --allow-test")
    split = D.load_split(a.split)
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    v = _interior()
    xc = (np.arange(D.N) - D.IJ_RECEPTOR) * D.DX
    X, Y = np.meshgrid(xc, xc)
    tr = D.load_split("train")
    groups = {g: np.array(m) for g, m in E.breakouts(split, np.isin(split.seed_key, list(set(tr.seed_key)))).items() if g in GROUPS}
    del tr

    fields, les = recipe_fields(split, valid)
    sc = M.score_fields(fields, les, split.wdir_deg, arr, split.asymptote)
    fm = {}
    for name, f in fields.items():
        rows = [field_metrics(f[i], les[i], v, X, Y) for i in range(split.n)]
        fm[name] = {k: np.array([r[k] for r in rows]) for k in FIELD_KEYS}
    floor = pair_floor(arr, v, X, Y)

    out = dict(split=a.split, recipe=FR.RECIPE, losses=LOSSES, eps_m2=EPS, n_proj=N_PROJ, floor_pair=floor, groups={})
    L = [f"# Reporting metrics on {a.split} (frozen recipe, {split.n} records)", "",
         "## Training losses", ""] + [f"- **{k}**: {s}" for k, s in LOSSES.items()] + ["",
         "## Composite = geometric mean over the five production quantities of median|error| / Kljun's median|error| (< 1 beats Kljun)", ""]
    for g, m in groups.items():
        og = out["groups"][g] = dict(n=int(m.sum()), composite={}, production={}, field={})
        L += [f"### {g} (n = {int(m.sum())})", "",
              "| model | composite | peak_x [m] | centroid [m] | 1 - overlap80 | array share [pp] | integral |", "|---|---|---|---|---|---|---|"]
        for name in fields:
            comp, ratios = M.composite(sc[name], sc["Kljun"], m)
            og["composite"][name] = dict(value=comp, ratios=ratios)
            med = {k: float(np.nanmedian(M.error_of(sc[name], k)[m])) for k in M.METRIC_KEYS}
            og["production"][name] = med
            L.append(f"| {name} | {comp:.3f} | {med['peak_x']:.1f} | {med['centroid']:.1f} | {med['overlap80']:.3f} | "
                     f"{med['array_share']:.2f} | {med['integral']:.3f} |")
        L.append(f"| two-window floor | - | {floor['peak_x']:.1f} | {floor['centroid']:.1f} | {1-floor['overlap80']:.3f} | "
                 f"{floor['array_share']:.2f} | {floor['integral']:.3f} |")
        L += ["", "| model | log-MSE (dex²) | log-MSE, LES body | sliced W1 [m] | KL(LES‖model) [nats] | MS-SSIM (log grid) | rel. L2 |", "|---|---|---|---|---|---|---|"]
        for name in fields:
            med = {k: float(np.nanmedian(fm[name][k][m])) for k in FIELD_KEYS}
            og["field"][name] = med
            L.append(f"| {name} | {med['log_mse']:.3f} | {med['log_mse_body']:.3f} | {med['sw1_m']:.1f} | {med['kl_nats']:.3f} | {med['ms_ssim']:.3f} | {med['rel_l2']:.3f} |")
        L.append(f"| two-window floor | {floor['log_mse']:.3f} | {floor['log_mse_body']:.3f} | {floor['sw1_m']:.1f} | {floor['kl_nats']:.3f} | {floor['ms_ssim']:.3f} | {floor['rel_l2']:.3f} |")
        L.append("")
    L += ["Medians over the group's records. Production errors are |model - LES| per record (overlap80 as 1 - Jaccard of the 80% source areas). "
          f"Field metrics on the 122² interior with log floor ε = {EPS:.0e} m⁻²: log-MSE = mean (log10(f+ε) - log10(LES+ε))² over the interior, and over the cells inside the LES's own 99.5% source area ('body'); "
          f"sliced W1 = mean over {N_PROJ} directions of the 1-D Wasserstein-1 between the unit-mass positive parts (placement/shape, blind to amplitude); "
          "KL = Σ P log(P/Q) with P = LES/ΣLES and Q = (model+ε)/Σ; MS-SSIM = 5-scale SSIM on the log10 grid (1 = identical). "
          "The floor row is window 1 vs window 0 of the one two-window case (a train record, n = 1), processed identically."]
    # sensitivity of the log-grid metrics to the floor, on every 4th record
    eps0, idx = EPS, np.arange(0, split.n, 4)
    L += ["", f"## Floor sensitivity (every 4th record, n = {len(idx)}): medians of log-MSE / MS-SSIM / KL at four values of ε", "",
          "| ε [m⁻²] | " + " | ".join(f"{n} log-MSE / MS-SSIM / KL" for n in fields) + " |", "|---|" + "---|" * len(fields)]
    out["eps_sensitivity"] = {}
    for eps in (1e-9, 1e-8, 1e-7, 1e-6):
        EPS = eps
        row = {n: [float(np.median([fn(fields[n][i], les[i], v) for i in idx])) for fn in
                   (log_mse, ms_ssim, lambda f, r, vv: kl_nats(r, f, vv))] for n in fields}
        out["eps_sensitivity"][f"{eps:.0e}"] = row
        L.append(f"| {eps:.0e} | " + " | ".join(f"{r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}" for r in row.values()) + " |")
    EPS = eps0
    L += ["", "Read: log-MSE and MS-SSIM are dominated by the cells that sit at the floor in both fields, so their level is set by ε and the model "
          "ordering is inside the noise; KL's smoothing bias at identical fields is 0.008 nats at ε = 1e-9 and grows with ε. "
          "Sliced W1 and the composite do not depend on ε."]
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, f"metrics_{a.split}.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(a.outdir, f"metrics_{a.split}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(a.outdir, f"metrics_{a.split}_per_record.npz"), run_id=split.meta["run_id"],
                        **{f"{n}__{k}": fm[n][k] for n in fm for k in FIELD_KEYS},
                        **{f"{n}__{k}": sc[n][k] for n in fields for k in M.METRIC_KEYS})
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
