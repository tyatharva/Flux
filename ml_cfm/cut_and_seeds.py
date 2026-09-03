"""(a) What the 99% source-area cut removes from each field against an absolute floor, with
the metrics before and after, at S = 800 and at the fitted S; (b) the seed-ensemble curve
from the five seeds on disk. Val only; the test split is never read.

    python -m ml_cfm.cut_and_seeds
"""
import glob
import itertools
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

OUT = os.path.join(REPO, "results", "ml_cfm", "calib", "samples")
GROUPS = ("all", "north_N_NE_NW", "array_in_view_gt5pct")


def cut_abs(fields, floor):
    out = np.where(fields >= floor, fields, 0.0).astype(np.float32)
    a = np.abs(fields).sum((1, 2))
    rem = 1 - np.abs(out).sum((1, 2)) / a
    area = ((fields != 0).sum((1, 2)) - (out != 0).sum((1, 2))) / np.maximum((fields != 0).sum((1, 2)), 1)
    return out, rem, area


def cut_sa(fields, frac=0.99):
    out, info = TT.threshold_stack(fields, frac)
    area = ((fields != 0).sum((1, 2)) - (out != 0).sum((1, 2))) / np.maximum((fields != 0).sum((1, 2)), 1)
    return out, info["mass_removed_frac"], area


def main():
    split = D.load_split("val")
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    tr = D.load_split("train")
    groups = {g: m for g, m in E.breakouts(split, np.isin(split.seed_key, list(set(tr.seed_key)))).items() if g in GROUPS}
    del tr
    sc_k = M.score_fields({"k": split.kljun}, split.target, split.wdir_deg, arr, split.asymptote)["k"]
    seeds = sorted(glob.glob(os.path.join(REPO, "results", "ml_cfm", "final", "seed?")))
    per_seed = []
    for sd in seeds:
        with np.load(os.path.join(sd, "samples_val.npz")) as z:
            T0, s_out = z["samples_T"].astype(np.float32), z["s_out"].astype(np.float32)
        T1 = [np.load(p)["samples_T"].astype(np.float32) for p in sorted(glob.glob(os.path.join(sd, "samples_val_extra*.npz")))]
        T = np.concatenate([T0] + T1)
        per_seed.append((s_out[None, :, None, None] * np.sinh(np.clip(T, -20, 20)) * valid[None, None]).astype(np.float32))
    fno = np.mean([np.load(p)["fno"] for p in sorted(glob.glob(os.path.join(REPO, "results", "ml", "final", "seed*", "pred_val.npz")))], axis=0).astype(np.float32)
    mean800 = np.mean([p.mean(0) for p in per_seed], axis=0)
    rng = np.random.default_rng(0)
    mean70 = np.concatenate([p[rng.choice(p.shape[0], 14, replace=False)] for p in per_seed]).mean(0)
    one = per_seed[0][0]
    base = {"kljun": split.kljun, "fno": fno, "cfm_mean800": mean800, "cfm_mean70": mean70, "cfm_one_sample": one}

    # (a) the cuts
    rows, fields = [], {}
    for name, f in base.items():
        fields[name] = f
        c99, r99, a99 = cut_sa(f, 0.99)
        cab, rab, aab = cut_abs(f, 1e-8)
        fields[name + "_sa99"] = c99
        fields[name + "_abs1e-8"] = cab
        rows.append((name, r99, a99, rab, aab))
    les_c, les_r, les_a = cut_sa(split.target, 0.99)
    _, les_rab, les_aab = cut_abs(split.target, 1e-8)
    rows.append(("LES target", les_r, les_a, les_rab, les_aab))
    sc = M.score_fields(fields, split.target, split.wdir_deg, arr, split.asymptote)
    comp = {n: {g: M.composite(sc[n], sc_k, groups[g])[0] for g in GROUPS} for n in fields}
    peak_frac = {n: float(np.median(1e-8 / np.abs(f).reshape(f.shape[0], -1).max(1))) for n, f in base.items()}

    L = ["# The cut: what it removes, and the metrics before and after (val medians)", "",
         "| field | 99% cut: |mass| removed | 99% cut: non-zero cells removed | 1e-8 floor: |mass| removed | 1e-8 floor: cells removed | 1e-8 as a fraction of the peak (median) |",
         "|---|---|---|---|---|---|"]
    for name, r99, a99, rab, aab in rows:
        pf = f"{peak_frac[name]:.1e}" if name in peak_frac else "-"
        L.append(f"| {name} | {100*np.median(r99):.2f}% | {100*np.median(a99):.0f}% | {100*np.median(rab):.2f}% | {100*np.median(aab):.0f}% | {pf} |")
    L += ["", "| field | composite all | north | in view |", "|---|---|---|---|"]
    for n in fields:
        L.append(f"| {n} | {comp[n]['all']:.3f} | {comp[n]['north_N_NE_NW']:.3f} | {comp[n]['array_in_view_gt5pct']:.3f} |")

    # (b) seed ensembles: every subset of k seeds, each seed contributing its 160 samples
    L += ["", "# Seed ensembles: composite vs Kljun of the pooled mean over k seeds (160 samples each), mean ± sd over the C(5,k) subsets", "",
          "| k | n subsets | all | north | in view | best subset (all) | worst subset (all) |", "|---|---|---|---|---|---|---|"]
    ens = {}
    for k in range(1, len(per_seed) + 1):
        vals = {g: [] for g in GROUPS}
        for sub in itertools.combinations(range(len(per_seed)), k):
            m = np.mean([per_seed[j].mean(0) for j in sub], axis=0)
            s = M.score_fields({"m": m}, split.target, split.wdir_deg, arr, split.asymptote)["m"]
            for g in GROUPS:
                vals[g].append(M.composite(s, sc_k, groups[g])[0])
        ens[k] = {g: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, float(min(v)), float(max(v))) for g, v in vals.items()}
        e = ens[k]
        L.append(f"| {k} | {len(vals['all'])} | {e['all'][0]:.3f} ± {e['all'][1]:.3f} | {e['north_N_NE_NW'][0]:.3f} ± {e['north_N_NE_NW'][1]:.3f} | "
                 f"{e['array_in_view_gt5pct'][0]:.3f} ± {e['array_in_view_gt5pct'][1]:.3f} | {e['all'][2]:.3f} | {e['all'][3]:.3f} |")
    with open(os.path.join(OUT, "cut_and_seeds.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "cut_and_seeds.json"), "w") as fh:
        json.dump(dict(composites=comp, cuts={r[0]: dict(sa99_mass=float(np.median(r[1])), sa99_area=float(np.median(r[2])),
                                                        abs_mass=float(np.median(r[3])), abs_area=float(np.median(r[4]))) for r in rows},
                       seed_ensembles=ens), fh, indent=1)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
