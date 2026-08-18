#!/usr/bin/env python3
"""Stage 4 gate: LPDM well-mixed test, plus the backward transit-time check.

usage: stage4_wellmixed.py <output_dir> [--dt 0.0625] [--n 40000] [--tlimit 600]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.fields import FieldSet, dump_series
from lpdm.model import LPDM
from lpdm import wellmixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--dt", type=float, default=0.0625)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--tlimit", type=float, default=600.0)
    ap.add_argument("--zlid", type=float, default=500.0)
    ap.add_argument("--c0", type=float, default=3.0)
    ap.add_argument("--ztouch", type=float, default=2.0)
    a = ap.parse_args()

    paths = dump_series(a.outdir)
    print(f"  {len(paths)} dumps: {os.path.basename(paths[0])} .. {os.path.basename(paths[-1])}")
    t0 = time.time()
    fs = FieldSet(paths, a.dt)
    print(f"  field cache {fs.mem_gb:.2f} GB, {fs.nx}x{fs.ny}x{fs.nz}, "
          f"dt_dump={fs.dt_dump:.2f} s, window {fs.t[0]:.0f}-{fs.t[-1]:.0f} s "
          f"({time.time()-t0:.0f} s to load)")

    lp = LPDM(fs, c0=a.c0, z_touch=a.ztouch)
    ok = True
    for direction, label in ((-1, "BACKWARD (the mode footprints use)"),
                             (+1, "FORWARD (control)")):
        t0 = time.time()
        out = wellmixed.run_test(lp, fs, n=a.n, z_lid=a.zlid, t_limit=a.tlimit,
                                 direction=direction)
        ok &= wellmixed.report(out, label)
        print(f"  ({out['iters']} integrator steps, {time.time()-t0:.0f} s)")

    # ---- second gate: backward transit time from the 30 m receptor to the surface
    print("\n  --- backward transit time from the receptor ---")
    k_r = int(np.argmin(np.abs(fs.zk - 30.0)))
    zr = float(fs.zk[k_r])
    n = 20000
    rng = np.random.default_rng(7)
    xr = fs.x0 + 0.75 * fs.Lx
    yr = fs.y0 + 0.5 * fs.Ly
    res = lp.run(np.full(n, xr), np.full(n, yr), np.full(n, zr),
                 float(fs.t[-1]), direction=-1, t_limit=a.tlimit,
                 reflect_touchdown=False, record_touchdown=True)
    tt = res["td_t"]
    frac = len(tt) / n
    print(f"  receptor z = {zr:.2f} m (level k={k_r})")
    print(f"  reached the surface within {a.tlimit:.0f} s: {frac*100:.1f}% of {n} particles")
    if frac > 0:
        q = np.percentile(tt, [5, 25, 50, 75, 95])
        print("  transit time (s): " + "  ".join(f"p{p}={v:.0f}" for p, v in
                                                 zip((5, 25, 50, 75, 95), q)))
        print(f"  median {q[2]/60:.1f} min. PLAN.md expects 1-5 min unstable, "
              f"10-15 min stable; neutral sits between.")
    print(f"\n  STAGE 4 GATE: {'PASS' if ok and frac > 0.5 else 'FAIL'}")
    return 0 if (ok and frac > 0.5) else 1


if __name__ == "__main__":
    sys.exit(main())
