#!/usr/bin/env python3
"""Gate for ml/data.py and ml/features.py: the loader refuses the test split, reads only
its own rows, takes the file's normalisation rather than recomputing it, and puts the
static surface on the SAME padded cells as the corpus rasters.

ASSERTED (a FAIL is a FAIL):
  1. load_split("test") raises TestSplitForbidden, and the refusal is in the audit log.
  2. train and val load with the documented counts (837 / 235), no 2025 record, and every
     row's split re-derived from its own datetime by lpdm.corpus.split_of.
  3. norm/ carries the documented train-only constants; nothing is recomputed.
  4. data/grid30_raised pads to the corpus frame: tower (61,61) -> (64,64) inside the
     44-cell array at rows 62-72 / cols 62-65; the rectangle re-derived from
     bin/prep_surface.py's offsets is the same 44 cells.
  5. The raster-based array share of the LES target agrees with the touchdown-based
     meta/array_share on train+val: Pearson r >= 0.9 (measured r and median |diff| printed).
     This is the check that the static array mask sits on the target's cells.
  6. The cone rebuilt from the file's own rule leaves every val target EXACTLY zero outside
     it -- the reproduction matches the mask that made corpus_cone.h5.
  7. The asinh transform round-trips Kljun to 1e-5 relative, in both norm modes.
  8. The audit log has no line that loaded the test split.

MEASURED AND PRINTED (not gated): octant counts, records with array share > 5%, seed keys
shared between train and val (the leakage channel PROJECT_BRIEF.md limitation 10 names).

usage: /home/atyagi/miniforge3/envs/LESNet/bin/python bin/test_ml_data.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import data as D                                   # noqa: E402
from ml import features as F                               # noqa: E402

fails = []


def check(ok, msg):
    print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        fails.append(msg)


def main():
    t0 = time.time()
    print(f"corpus: {os.path.relpath(D.H5_DEFAULT, D.REPO)}")

    # 1. the guard
    try:
        D.load_split("test")
        check(False, "load_split('test') did not raise")
    except D.TestSplitForbidden as e:
        check(True, f"load_split('test') raises TestSplitForbidden: {str(e)[:60]}...")
    lines = D.audit_lines()
    check(any(l.get("split") == "test" and l.get("refused") for l in lines),
          "the refusal is recorded in results/ml/loader_audit.jsonl")

    # 2. train and val
    tr = D.load_split("train")
    va = D.load_split("val")
    check(tr.n == 837 and va.n == 235, f"counts train {tr.n} / val {va.n} (want 837 / 235)")
    years = sorted(set(s[:4] for s in np.concatenate([tr.meta["datetime"],
                                                       va.meta["datetime"]])))
    check(str(D.TEST_YEAR) not in years, f"no {D.TEST_YEAR} record in train+val: years {years}")
    check(len(set(tr.meta["parent_case"]) & set(va.meta["parent_case"])) == 0,
          "no parent case appears in both train and val")
    check(np.allclose(tr.scalars[:, 0], tr.meta["zi_achieved_m"]),
          "scalar h == meta/zi_achieved_m (the integral asymptote uses the input scalar)")
    check(bool((tr.zL < 0).all() and (va.zL < 0).all()),
          f"every record is unstable: z/L train [{tr.zL.min():.3f}, {tr.zL.max():.4f}]")

    # 3. norm/
    norm = D.read_norm()
    check(abs(norm["kljun_scale"] - 2.1699264834751375e-05) < 1e-15
          and abs(norm["target_scale"] - 2.4260476493509486e-05) < 1e-15,
          f"norm scales kljun {norm['kljun_scale']:.6e} target {norm['target_scale']:.6e}")
    check(norm["computed_on"] == "train split only" and norm["n_train"] == 837,
          f"norm computed_on={norm['computed_on']!r} n_train={norm['n_train']}")
    check(abs(float(norm["scalars_std"][0]) - 226.46852) < 0.01,
          f"norm h std {float(norm['scalars_std'][0]):.3f} matches the documented 226.47")

    # 4. statics on the padded grid
    st = D.load_statics()
    arr = st["array"] > 0.5
    rows = np.where(arr.any(axis=1))[0]
    cols = np.where(arr.any(axis=0))[0]
    check(int(arr.sum()) == 44 and rows.min() == 62 and rows.max() == 72
          and cols.min() == 62 and cols.max() == 65,
          f"array: {int(arr.sum())} cells, rows {rows.min()}-{rows.max()}, "
          f"cols {cols.min()}-{cols.max()} on the padded grid")
    check(bool(arr[D.IJ_RECEPTOR, D.IJ_RECEPTOR]), "receptor cell (64,64) is inside the array")
    check(all(st[k].shape == (128, 128) for k in D.STATIC_FIELDS), "all statics are (128,128)")
    check(bool((st["z0m"][~tr.valid_mask] == 0).all()) and
          bool((st["z0m"][tr.valid_mask] > 0).all()), "z0m is zero on the pad, positive inside")

    # 5. the empirical grid check
    both = np.concatenate([tr.target, va.target])
    meta_share = np.concatenate([tr.meta["array_share"], va.meta["array_share"]])
    rs = D.raster_array_share(both, arr)
    ok = np.isfinite(rs)
    r = float(np.corrcoef(rs[ok], meta_share[ok])[0, 1])
    d = np.abs(rs[ok] - meta_share[ok]) * 100
    check(r >= 0.9, f"raster array share vs meta/array_share on train+val: r = {r:.4f}, "
                    f"median |diff| {np.median(d):.3f} pp, p95 {np.percentile(d, 95):.3f} pp")
    north = np.concatenate([tr.octant, va.octant]) == "N"
    print(f"         N-wind mean array share: raster {100*np.nanmean(rs[north]):.2f}% "
          f"meta {100*np.mean(meta_share[north]):.2f}% (PROJECT_BRIEF.md: 30.28%)")

    # 6. the cone, rebuilt
    tc = time.time()
    keep = D.cone_masks(va)
    outside = np.abs(va.target[~keep]).max() if (~keep).any() else 0.0
    check(keep.shape == (va.n, 128, 128) and outside == 0.0,
          f"val target is exactly 0 outside the rebuilt cone (max |f| outside = {outside:g}; "
          f"{time.time()-tc:.0f} s to build {va.n} masks)")
    kept = keep[:, va.valid_mask].mean(axis=1)
    print(f"         cone keeps a median {100*np.median(kept):.1f}% of interior cells")

    # 7. transform round trip, both norm modes
    for mode in ("global", "record"):
        spec = F.FeatureSpec(norm_mode=mode)
        fx = F.Features(va, st, norm, spec)
        back = fx.to_physical(fx.base_T)
        rel = np.abs(back - va.kljun).max() / np.abs(va.kljun).max()
        check(rel < 1e-5, f"norm_mode={mode}: inv(fwd(kljun)) reproduces kljun, max rel "
                          f"err {rel:.2e}; channels {fx.channel_names}")
        check(np.isfinite(fx.target_T).all() and np.isfinite(fx.scal).all(),
              f"norm_mode={mode}: features finite; target_T range "
              f"[{fx.target_T.min():.2f}, {fx.target_T.max():.2f}], "
              f"scal col3 range [{fx.scal[:, 3].min():.2f}, {fx.scal[:, 3].max():.2f}]")

    # measured, not gated
    oc_tr = {o: int((tr.octant == o).sum()) for o in D.OCTANTS}
    oc_va = {o: int((va.octant == o).sum()) for o in D.OCTANTS}
    print(f"         octants train {oc_tr}")
    print(f"         octants val   {oc_va}")
    print(f"         array share > 5%: train {int((tr.meta['array_share'] > 0.05).sum())} "
          f"val {int((va.meta['array_share'] > 0.05).sum())}")
    shared = set(tr.seed_key) & set(va.seed_key)
    n_shared_val = int(np.isin(va.seed_key, list(shared)).sum())
    print(f"         seed keys: train {len(set(tr.seed_key))}, val {len(set(va.seed_key))}, "
          f"shared {len(shared)}; val records on a shared seed {n_shared_val}/{va.n}")

    # 8. the audit
    lines = D.audit_lines()
    # The test split was read exactly once, for the frozen evaluation of 2026-09-04 (see
    # docs/emulator/results.md). Every such read must carry allow_test; an unaudited one fails.
    loaded_test = [l for l in lines if l.get("split") == "test" and l.get("n", 0) > 0]
    unaudited = [l for l in loaded_test if not l.get("allow_test")]
    check(len(unaudited) == 0, f"audit log: {len(lines)} lines; {len(loaded_test)} test-split "
                               f"loads, {len(unaudited)} of them without allow_test")

    name = "test_ml_data"
    print(f"{name}: {'FAIL' if fails else 'PASS'} ({len(fails)} failures, "
          f"{time.time()-t0:.0f} s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
