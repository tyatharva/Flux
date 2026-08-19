#!/usr/bin/env python3
"""Why does the LES+LPDM footprint peak further upwind than Kljun? Three suspects, one load.

1. C0, the Langevin constant. Weil et al. (2004) use 3 for the LES sub-grid term, but the
   choice is not sharp and it sets the Lagrangian timescale T_L = 2 sigma_s^2/(C0 eps)
   directly. A longer T_L means slower vertical mixing and a longer footprint.
2. The 2/|w_td| weight's heavy tail. A particle that barely crosses the touchdown level
   contributes 2/|w| with |w| near zero. If those land preferentially far upwind, they
   inflate the tail and drag the centroid out. A floor on |w_td| tests it directly.
3. Monte-Carlo noise. The footprint grid resolution and particle count set how much of the
   per-cell scatter is estimator noise rather than turbulence.

All three are evaluated from ONE field-cache load, because loading 361 dumps costs more
than the trajectories do.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm import kljun
from lpdm.driver import compute_footprint
from lpdm.fields import FieldSet, dump_series


def main(outdir="runs/s30_w1/output", dt=0.0625):
    paths = dump_series(outdir)
    fs = FieldSet(paths, dt, verbose=False)
    print(f"  {len(paths)} dumps, cache {fs.mem_gb:.2f} GB\n")
    base = dict(n_per_release=1500, dt_release=4.0, t_back=900.0, grid_res=60.0,
                split_halves=False, verbose=False)
    runs = [
        ("baseline C0=3, res 60 m", dict(c0=3.0)),
        ("C0=6 (shorter T_L)", dict(c0=6.0)),
        ("C0=1.5 (longer T_L)", dict(c0=1.5)),
        ("C0=3, |w_td| floor 0.05", dict(c0=3.0, w_floor=0.05)),
        ("C0=3, seed 1 (noise)", dict(c0=3.0, seed=1)),
    ]
    print(f"  {'case':<28} {'peak_x':>8} {'cent_x':>8} {'int f':>7} {'80% ha':>8} {'top0.1%':>8}")
    ref = None
    for label, kw in runs:
        t0 = time.time()
        r = compute_footprint(fs, paths, **base, **kw)
        g = r["grid"]
        m = g.metrics("flux")
        tc = g.tail_concentration()
        print(f"  {label:<28} {m['peak_x']:8.0f} {m['centroid_x']:8.0f} "
              f"{g.integral():7.3f} {m['area80_cells']*g.area/1e4:8.2f} "
              f"{tc['top0p1pct_share']*100:7.1f}%   ({time.time()-t0:.0f} s)")
        if ref is None:
            ref = r
    st = ref["stats"]
    print(f"\n  Kljun x_max for these scalars: "
          f"{kljun.peak_distance(ref['z_agl'], st['h'], st['ustar'], umean=st['u_mean']):.0f} m")
    print(f"  LES scalars: U={st['u_mean']:.2f}  u*={st['ustar']:.3f}  h={st['h']:.0f}  "
          f"sigma_w={st['sigma_w']:.3f} (= {st['sigma_w']/st['ustar']:.2f} u*)  "
          f"sigma_v={st['sigma_v']:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
