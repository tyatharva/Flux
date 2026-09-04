"""The reporting metrics, on the frozen recipe (ml_cfm/final_recipe.py): the training losses
(stated); peak distance, centroid and integral as RMSE over records against the LES; the 80%
source-area overlap (Jaccard), rel. L2, sliced Wasserstein-1, Jensen-Shannon distance and
MS-SSIM on the log grid as means over records. Every number beside its two-window realisation
floor. Printed for the whole split; the four cardinal 90-degree sectors and the eight octants
go to the JSON and the per-record .npz for the graphs.

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
from ml_cfm import tailthresh as TT           # noqa: E402
from ml_cfm import final_recipe as FR         # noqa: E402

EPS = 1e-9            # m^-2: the log floor. ~4e-5 of the median LES peak (2.6e-5), 1.5 decades under the 99.5% cut level
N_PROJ = 64           # sliced-Wasserstein projections
MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)   # Wang et al. 2003
CARDINAL = (("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", 270.0))     # 90-degree sectors centred on each
GROUPS = ("all",) + tuple("sector_" + c for c, _ in CARDINAL) + tuple("oct_" + o for o in D.OCTANTS)
PRINT_GROUPS = ("all",)
FIELD_KEYS = ("overlap80", "rel_l2", "sw1_m", "js_dist", "ms_ssim")   # per-record scores, mean over records
RMSE_KEYS = ("peak_x", "centroid", "integral")                         # per-record errors, RMSE over records

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


def js_distance(f, ref, v):
    """Jensen-Shannon distance in bits, sqrt(JS divergence), between the unit-mass positive parts
    over the interior: 0 = identical, 1 = no shared support. Symmetric; no smoothing needed."""
    p = np.maximum(np.asarray(f, np.float64), 0)[v]; p = p / p.sum()
    q = np.maximum(np.asarray(ref, np.float64), 0)[v]; q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a: float((a[a > 0] * np.log2(a[a > 0] / m[a > 0])).sum())
    return float(np.sqrt(max(0.5 * kl(p) + 0.5 * kl(q), 0.0)))


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
    return dict(rel_l2=M.image_metrics(f, ref)["rel_l2"], sw1_m=sliced_w1(f, ref, v, X, Y), js_dist=js_distance(f, ref, v),
                ms_ssim=ms_ssim(f, ref, v))


def recipe_fields(split, valid):
    R = FR.RECIPE
    samples = FR.cfm_samples(split, R["cfm_seeds"], R["samples_per_seed"], R["tau"], valid)
    cfm, _ = TT.threshold_stack(samples.mean(0), R["cut_frac"])
    fno_raw = np.mean([np.load(os.path.join(REPO, "results", "ml", "final", sd, f"pred_{split.name}.npz"))["fno"]
                       for sd in R["fno_seeds"]], axis=0).astype(np.float32)
    fno, _ = TT.threshold_stack(fno_raw, R["cut_frac"])
    les = np.maximum(split.target, 0).astype(np.float32)
    return {"Kljun": split.kljun, "FNO": fno, "CFM": cfm}, les, samples


def perfect_row(les, wd, arr, asym, v, X, Y):
    """The LES scored against itself on one record: the perfect value of every column, computed
    rather than written down so the identities are checked on a real field."""
    prod = M.pair_errors(les, les, wd, arr, asym)
    out = {k: float(prod[k]) for k in RMSE_KEYS + ("overlap80",)}
    out.update(field_metrics(les, les, v, X, Y))
    return out


def main(argv=None):
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
    oc = split.octant.astype(str)
    groups = {"all": np.ones(split.n, bool)}
    for c, deg in CARDINAL:
        groups["sector_" + c] = np.abs((split.wdir_deg - deg + 180) % 360 - 180) <= 45
    for o in D.OCTANTS:
        groups["oct_" + o] = oc == o

    fields, les, samples = recipe_fields(split, valid)
    sc = M.score_fields(fields, les, split.wdir_deg, arr, split.asymptote)

    fm = {}
    for name, f in fields.items():
        rows = [field_metrics(f[i], les[i], v, X, Y) for i in range(split.n)]
        fm[name] = {k: np.array([r[k] for r in rows]) for k in FIELD_KEYS if k != "overlap80"}
        fm[name]["overlap80"] = sc[name]["overlap80"]
    perfect = perfect_row(les[0], float(split.wdir_deg[0]), arr, float(split.asymptote[0]), v, X, Y)

    out = dict(split=a.split, recipe=FR.RECIPE, losses=LOSSES, eps_m2=EPS, n_proj=N_PROJ, perfect=perfect, groups={})
    L = [f"# Reporting metrics on {a.split} (frozen recipe, {split.n} records)", "",
         "## Training losses", ""] + [f"- **{k}**: {s}" for k, s in LOSSES.items()] + [""]
    for g, m in groups.items():
        og = out["groups"][g] = dict(n=int(m.sum()), rmse={}, mean={}, median={})
        for name in fields:
            og["rmse"][name] = {k: float(np.sqrt(np.nanmean(sc[name][k][m] ** 2))) for k in RMSE_KEYS}
            og["mean"][name] = {k: float(np.nanmean(fm[name][k][m])) for k in FIELD_KEYS}
            og["median"][name] = {k: float(np.nanmedian(fm[name][k][m])) for k in FIELD_KEYS}
        if g not in PRINT_GROUPS:
            continue
        L += [f"## {g} (n = {int(m.sum())})", "",
              "| model | peak distance RMSE [m] | centroid RMSE [m] | integral RMSE | overlap80 (Jaccard) | rel. L2 | sliced W1 [m] | JS distance [bits] | MS-SSIM (log grid) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name in fields:
            r, e = og["rmse"][name], og["mean"][name]
            L.append(f"| {name} | {r['peak_x']:.1f} | {r['centroid']:.1f} | {r['integral']:.3f} | {e['overlap80']:.3f} | {e['rel_l2']:.3f} | "
                     f"{e['sw1_m']:.1f} | {e['js_dist']:.3f} | {e['ms_ssim']:.3f} |")
        L.append(f"| LES (perfect) | {perfect['peak_x']:.1f} | {perfect['centroid']:.1f} | {perfect['integral']:.3f} | {perfect['overlap80']:.3f} | "
                 f"{perfect['rel_l2']:.3f} | {perfect['sw1_m']:.1f} | {perfect['js_dist']:.3f} | {perfect['ms_ssim']:.3f} |")
        L.append("")
    L += ["Direction groups (" + ", ".join(f"{g.split('_', 1)[1]} n={out['groups'][g]['n']}" for g in GROUPS if g != "all") +
          ") are in the JSON and the per-record .npz, for the wind-rose graphs; sectors are 90 degrees centred on N/E/S/W, octants 45 degrees. "
          "Peak distance, centroid and integral: RMSE over records of the per-record error against the LES (|upwind peak distance difference|, "
          "distance between the mass centroids, |integral difference|). The rest are means over records of per-record scores: "
          "overlap80 = Jaccard of the two 80% source areas (1 = identical); rel. L2 = ||model - LES|| / ||LES|| on the 122² interior (0 = identical); "
          f"sliced W1 = mean over {N_PROJ} directions of the 1-D Wasserstein-1 between the unit-mass positive parts [m] (0 = identical); "
          "JS distance = sqrt of the Jensen-Shannon divergence in bits between the unit-mass positive parts (0 = identical, 1 = disjoint); "
          f"MS-SSIM = 5-scale SSIM on the log10 grid with floor ε = {EPS:.0e} m⁻² (1 = identical). "
          "The LES row is the LES target scored against itself: the perfect value of every column."]
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, f"metrics_{a.split}.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(a.outdir, f"metrics_{a.split}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(a.outdir, f"metrics_{a.split}_per_record.npz"), run_id=split.meta["run_id"],
                        **{f"{n}__{k}": fm[n][k] for n in fm for k in FIELD_KEYS},
                        octant=split.octant.astype(str), wdir_deg=split.wdir_deg,
                        **{f"{n}__{k}": sc[n][k] for n in fields for k in RMSE_KEYS})
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
