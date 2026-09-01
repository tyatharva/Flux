#!/usr/bin/env python3
"""Does storing the LPDM's input fields as fp16 change the footprint?

docs/PLAN.md Stage 3 makes fp16-on-write the storage fix (81.7 GB -> 16.2 GB per window at the
24 m grid) but requires the quantisation be shown small against the LPDM's own noise
BEFORE it is accepted. This settles it on real fields and without any LES: load a window,
round u, v, w and TKE_0 through float16 and back, and recompute the footprint with an
identical seed.

fp16 has a 10-bit mantissa, so the relative spacing is 2^-11 = 4.9e-4. For w with
sigma_w ~ 0.36 m/s that is ~2e-4 m/s of quantisation, three orders of magnitude below the
Monte-Carlo scatter of the estimator itself. The point of the test is to confirm that with
the actual estimator rather than by arithmetic.

usage: fp16_test.py <outdir> --dt <dt> [--res 60] [--tback 900]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.fields import FieldSet, dump_series
from lpdm.driver import compute_footprint
from lpdm.footprint import source_area_overlap


def describe(g, xc):
    f = g.normalised("flux")
    fy = f.sum(axis=0)
    tot = fy.sum()
    return dict(peak=float(xc[int(np.argmax(fy))]),
                centroid=float((fy * xc).sum() / tot),
                integral=float(g.integral()), field=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--res", type=float, default=60.0)
    ap.add_argument("--tback", type=float, default=900.0)
    ap.add_argument("--nrel", type=int, default=700)
    a = ap.parse_args()

    paths = dump_series(a.outdir)
    out = {}
    for tag in ("fp32", "fp16"):
        t0 = time.time()
        fs = FieldSet(paths, a.dt, verbose=False,
                      store_dtype=np.float16 if tag == "fp16" else None)
        r = compute_footprint(fs, paths, n_per_release=a.nrel, t_back=a.tback,
                              grid_res=a.res, seed=0, split_halves=False, verbose=False)
        g = r["grid"]
        out[tag] = describe(g, g.xc)
        print(f"  {tag}: peak {out[tag]['peak']:.0f} m  centroid "
              f"{out[tag]['centroid']:.1f} m  integral {out[tag]['integral']:.4f}  "
              f"({time.time()-t0:.0f} s)")
        del fs

    A, B = out["fp32"], out["fp16"]
    ov = source_area_overlap(np.maximum(B["field"], 0), np.maximum(A["field"], 0))
    d = B["field"] - A["field"]
    l1 = np.abs(d).sum() / np.abs(A["field"]).sum()
    print("\n  fp16 vs fp32, same seed and same releases:")
    print(f"    peak difference      {B['peak']-A['peak']:+.1f} m")
    print(f"    centroid difference  {B['centroid']-A['centroid']:+.2f} m")
    print(f"    integral difference  {B['integral']-A['integral']:+.5f}")
    print(f"    80% source-area overlap  {ov*100:.2f}%")
    print(f"    per-cell L1 difference   {l1*100:.3f}% of the total")
    print("\n  Stage 5 error floor for comparison: 59.2% overlap between two halves of one")
    print("  window, 60 m peak, 99 m centroid. fp16 must be far inside that to be free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
