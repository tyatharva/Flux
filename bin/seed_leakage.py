#!/usr/bin/env python3
"""Do two cases that share a SEED have more similar footprints than two that do not?

WHY IT MATTERS. ~30 spun-up seeds serve the whole corpus, so the same seed is restarted from
in train, val and test. If a seed's turbulence realisation left a fingerprint in the
footprint, cases sharing a seed would resemble each other for a reason that has nothing to
do with meteorology, and a train/test split that does not group by seed would leak it.

THE COMPARISON, and it uses only footprints already on disk. Shape is the crosswind-
integrated footprint `f_y(x)` -- WIND-ALIGNED and therefore invariant to the 90-degree seed
rotation, which is what makes cases at different achieved directions comparable at all. Each
profile is normalised to unit area, so the metric is shape and not magnitude:

    D(A, B) = 0.5 * sum |fyA/sum(fyA) - fyB/sum(fyB)|          (0 = identical, 1 = disjoint)

THE FLOOR IS EACH CASE'S OWN HALF-VS-HALF DIFFERENCE, `fy1` vs `fy2`: two halves of the same
release ensemble in the same window, which differ by sampling alone. That is the smallest
difference the estimator can resolve, and a same-seed similarity only means something if it
is below it.

  usage: bin/seed_leakage.py [--dir results/corpus] [--json results/seed_leakage.json]
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np


def shape(fy):
    fy = np.asarray(fy, dtype=np.float64)
    s = np.abs(fy).sum()
    return fy / s if s > 0 else fy


def d_shape(a, b):
    return 0.5 * float(np.abs(shape(a) - shape(b)).sum())


def load(pairdir, corpusdir):
    out = []
    for f in sorted(glob.glob(os.path.join(pairdir, "*.json"))):
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        tag = os.path.basename(f)[:-5]
        npz = os.path.join(corpusdir, tag + ".npz")
        s = d.get("seed") or {}
        if not (s.get("job") and os.path.exists(npz)):
            continue
        with np.load(npz) as z:
            if not {"fy", "fy1", "fy2", "fy_xc"} <= set(z.files):
                continue
            fy, fy1, fy2 = (np.asarray(z["fy"]), np.asarray(z["fy1"]),
                            np.asarray(z["fy2"]))
            xax = np.asarray(z["fy_xc"])
        out.append({
            "tag": tag, "seed": s["job"], "rot": s.get("rot"),
            "regime": ("neutral" if "nbl" in s["job"] else
                       "convective" if "cbl" in s["job"] else "?"),
            "valid_time": (d.get("forcing") or {}).get("valid_time"),
            "z_recept": (d.get("site") or {}).get("z_recept_m"),
            "n": len(xax), "fy": fy, "fy1": fy1, "fy2": fy2,
            "floor": d_shape(fy1, fy2),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs")
    ap.add_argument("--dir", default="results/corpus")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    recs = load(a.pairs, a.dir)
    if not recs:
        print("SKIPPED: no footprint on disk carries both a seed and an f_y profile.")
        return 0

    # COMPARABLE ONLY WITHIN ONE GRID. Two 122-cell rasters at different cell sizes have the
    # same shape and a different meaning, so they are grouped by receptor height, which is
    # what distinguishes the grids in this project's records.
    groups = {}
    for r in recs:
        groups.setdefault(round(float(r["z_recept"] or 0), 1), []).append(r)

    report = {"groups": []}
    print(f"=== {len(recs)} footprint(s) on disk carry a seed and an f_y ===")
    for zr, rs in sorted(groups.items()):
        seeds = {}
        for r in rs:
            seeds.setdefault(r["seed"], []).append(r)
        same = [(x, y) for v in seeds.values() for x, y in itertools.combinations(v, 2)
                if x["valid_time"] != y["valid_time"]]
        diff = [(x, y) for x, y in itertools.combinations(rs, 2)
                if x["seed"] != y["seed"]]
        print(f"\n--- receptor {zr} m: {len(rs)} case(s), {len(seeds)} seed(s) ---")
        for r in rs:
            print(f"    {r['tag']:<24}{r['seed']:<24}rot {r['rot']}  "
                  f"{r['regime']:<11}{r['valid_time']}  own floor {r['floor']:.3f}")
        if not same or not diff:
            print(f"\n  SKIPPED at this receptor: {len(same)} same-seed pair(s) and "
                  f"{len(diff)} different-seed pair(s); the comparison needs at least one "
                  f"of each.")
            report["groups"].append({"z_recept": zr, "n": len(rs), "skipped": True,
                                     "n_same": len(same), "n_diff": len(diff)})
            continue

        def tab(pairs, label):
            vals = [d_shape(x["fy"], y["fy"]) for x, y in pairs]
            fl = [max(x["floor"], y["floor"]) for x, y in pairs]
            print(f"\n  {label}: {len(pairs)} pair(s)")
            for (x, y), v, f in zip(pairs, vals, fl):
                print(f"      {x['tag'][:20]:<21}{y['tag'][:20]:<21}"
                      f"D {v:.3f}   floor {f:.3f}   D/floor {v / f:6.2f}")
            return vals, fl

        vs, fs = tab(same, "SAME seed, different datetimes")
        vd, fd = tab(diff, "DIFFERENT seeds")
        rs_ = float(np.median(np.array(vs) / np.array(fs)))
        rd_ = float(np.median(np.array(vd) / np.array(fd)))
        print(f"\n  median D/floor  same-seed {rs_:.2f}   different-seed {rd_:.2f}")
        # The regimes present on each side, because a same-seed set drawn from one rung is
        # also a same-REGIME set, and that would explain a similarity just as well.
        reg_s = {(x["regime"], y["regime"]) for x, y in same}
        reg_d = {(x["regime"], y["regime"]) for x, y in diff}
        confounded = all(x == y for p in reg_s for x, y in [p]) and \
            any(x != y for p in reg_d for x, y in [p])
        report["groups"].append({
            "z_recept": zr, "n": len(rs), "skipped": False,
            "same_seed": [{"a": x["tag"], "b": y["tag"], "D": v, "floor": f}
                          for (x, y), v, f in zip(same, vs, fs)],
            "diff_seed": [{"a": x["tag"], "b": y["tag"], "D": v, "floor": f}
                          for (x, y), v, f in zip(diff, vd, fd)],
            "median_ratio_same": rs_, "median_ratio_diff": rd_,
            "regime_confounded": bool(confounded),
        })
        print()
        if confounded:
            print("  *** THE COMPARISON IS CONFOUNDED AND CANNOT SETTLE THE QUESTION.")
            print("      Every same-seed pair here is also a same-REGIME pair, and the")
            print("      different-seed pairs cross regimes. A seed serves one rung, so on")
            print("      a corpus this small 'same seed' and 'same regime' are the same")
            print("      variable. Separating them needs two cases of the SAME regime off")
            print("      DIFFERENT seeds -- which does not exist on disk.")
        if min(rs_, rd_) > 1.0:
            print(f"  Both classes differ by MORE than the within-case sampling floor "
                  f"({rs_:.2f}x and {rd_:.2f}x), so neither resembles the other more than "
                  f"the estimator can resolve.")
        if rs_ < 1.0:
            print(f"  *** same-seed pairs agree to WITHIN the sampling floor ({rs_:.2f}x). "
                  f"On a larger corpus that would be evidence of a seed fingerprint.")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1, default=float)
        print(f"\n  -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
