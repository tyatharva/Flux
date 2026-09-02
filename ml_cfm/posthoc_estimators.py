"""Post-hoc point estimators from the CFM's samples, scored on val against the LES.

    python -m ml_cfm.posthoc_estimators [--out results/ml_cfm/calib/final/estimators.md]

Estimators that need no LES (usable in production): mean in m^-2 (the current one), mean in
asinh space, per-cell median, the medoid sample (smallest summed asinh-L2 to the others),
the sample nearest the asinh mean, and the mean thresholded at its own 99% source area.
Plus the ORACLE best sample per record (smallest asinh L2 to the LES) as the ceiling only.
The test split is never read.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml import data as D                      # noqa: E402
from ml import metrics as M                   # noqa: E402
from ml import evaluate as E                  # noqa: E402
from ml_cfm import tailthresh as TT           # noqa: E402

GROUPS = ("all", "north_N_NE_NW", "array_in_view_gt5pct")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", nargs="+", default=sorted(glob.glob(os.path.join(REPO, "results", "ml_cfm", "final", "seed?"))))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ml_cfm", "calib", "final", "estimators.md"))
    a = ap.parse_args(argv)
    t0 = time.time()
    split = D.load_split("val")
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    tr = D.load_split("train")
    groups = E.breakouts(split, np.isin(split.seed_key, list(set(tr.seed_key))))
    del tr
    s_ref = float(D.read_norm()["target_scale"])

    T, s_out = [], None
    for sd in a.seeds:
        with np.load(os.path.join(sd, "samples_val.npz")) as z:
            assert np.array_equal(z["run_id"], split.meta["run_id"])
            T.append(z["samples_T"].astype(np.float32))
            s_out = z["s_out"].astype(np.float32)
    T = np.concatenate(T)                                  # (S, n, 128, 128)
    S, n = T.shape[:2]
    phys = lambda TT_: (s_out[:, None, None] * np.sinh(np.clip(TT_, -20, 20)) * valid).astype(np.float32)   # (n,H,W)
    phys_i = lambda TT_, i: (s_out[i] * np.sinh(np.clip(TT_, -20, 20)) * valid).astype(np.float32)        # (S,H,W) of record i
    les_T = np.arcsinh(split.target / s_ref).astype(np.float32)
    fno = np.mean([np.load(p)["fno"] for p in sorted(glob.glob(os.path.join(REPO, "results", "ml", "final", "seed*", "pred_val.npz")))], axis=0)

    est = {}
    est["mean_phys"] = np.stack([phys_i(T[:, i], i).mean(0) for i in range(n)])
    Tbar = T.mean(0)
    est["mean_asinh"] = phys(Tbar)
    est["median_cell"] = np.stack([np.median(phys_i(T[:, i], i), axis=0) for i in range(n)])
    medoid = np.empty((n, 128, 128), np.float32)
    nearest = np.empty_like(medoid)
    oracle = np.empty_like(medoid)
    which = dict(medoid=[], nearest=[], oracle=[])
    for i in range(n):
        X = T[:, i].reshape(S, -1).astype(np.float64)
        sq = (X * X).sum(1)
        d2 = sq[:, None] + sq[None] - 2 * X @ X.T
        k = int(np.argmin(d2.sum(1)))
        medoid[i] = phys_i(T[k, i][None], i)[0]
        which["medoid"].append(k)
        k2 = int(np.argmin(((X - Tbar[i].ravel()[None]) ** 2).sum(1)))
        nearest[i] = phys_i(T[k2, i][None], i)[0]
        which["nearest"].append(k2)
        k3 = int(np.argmin(((X - les_T[i].ravel()[None]) ** 2).sum(1)))
        oracle[i] = phys_i(T[k3, i][None], i)[0]
        which["oracle"].append(k3)
    est["medoid_sample"] = medoid
    est["nearest_to_mean_sample"] = nearest
    est["mean_phys_thr99"], _ = TT.threshold_stack(est["mean_phys"], 0.99)
    est["ORACLE_best_sample"] = oracle
    est["one_sample"] = phys(T[0])
    est["fno"] = fno.astype(np.float32)
    est["kljun"] = split.kljun
    print("estimators built", round(time.time() - t0), "s", flush=True)

    sc = M.score_fields(est, split.target, split.wdir_deg, arr, split.asymptote)
    print("scored", round(time.time() - t0), "s", flush=True)
    keys = M.METRIC_KEYS + M.SHAPE_KEYS + ("rel_l2", "ssim_T")
    L = [f"# Post-hoc point estimators from {S} CFM samples ({len(a.seeds)} seeds), val, {n} records", "",
         "Composite = geometric mean of the five production-metric median-|error| ratios vs Kljun (< 1 beats Kljun). "
         "'vs mean' = the same composite against the current estimator (mean in m^-2), with the smallest paired Wilcoxon p over the five metrics. "
         "The ORACLE row needs the LES and is the ceiling, not a method.", ""]
    for g in GROUPS:
        m = groups[g]
        L += [f"## {g} (n = {int(m.sum())})", "",
              "| estimator | composite vs Kljun | vs mean (p) | " + " | ".join(keys) + " |",
              "|---|---|---|" + "---|" * len(keys)]
        for name in est:
            comp = M.composite(sc[name], sc["kljun"], m)[0]
            if name == "mean_phys":
                vs = "-"
            else:
                c = E.compare(sc[name], sc["mean_phys"], m, M.METRIC_KEYS)
                vs = f"{M.composite(sc[name], sc['mean_phys'], m)[0]:.3f} ({min(v['wilcoxon_p'] for v in c.values()):.1g})"
            meds = " | ".join(f"{np.nanmedian(M.error_of(sc[name], k)[m]):.3f}" for k in keys)
            L.append(f"| {name} | {comp:.3f} | {vs} | {meds} |")
        L.append("")
    L += ["Units: peak_x, centroid [m]; array_share [pp]; overlap80 is 1 - Jaccard; ssim_T is 1 - SSIM; shape_1d, shape_l1_2d, rel_l2 dimensionless.", ""]
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("done", round(time.time() - t0), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
