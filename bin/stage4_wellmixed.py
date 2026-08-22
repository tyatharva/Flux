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
    ap.add_argument("--tlimit", type=float, default=900.0)
    # Score the footprint-relevant layer; release well above it so no artificial
    # boundary sits near the scored region (see lpdm/wellmixed.py).
    ap.add_argument("--zscore", type=float, default=400.0)
    ap.add_argument("--zrelease", type=float, default=1200.0)
    ap.add_argument("--c0", type=float, default=3.0)
    ap.add_argument("--ztouch", type=float, default=2.0)
    ap.add_argument("--z-target", type=float, default=10.0,
                    help="receptor height, m AGL, for the MOST floor and the transit-time "
                         "check. Was hard-coded at 30.0 in two places.")
    ap.add_argument("--sgs-most", action="store_true",
                    help="run the gate WITH the MOST-anchored variance floor in place. "
                         "The floor rescales sigma^2 by a height-dependent factor, and a "
                         "rescaling that the Thomson drift does not know about breaks "
                         "well-mixedness -- so the gate has to be run in the configuration "
                         "the footprints are actually computed in, not only in the "
                         "unmodified one.")
    ap.add_argument("--fp16-cache", action="store_true")
    a = ap.parse_args()

    paths = dump_series(a.outdir)
    print(f"  {len(paths)} dumps: {os.path.basename(paths[0])} .. {os.path.basename(paths[-1])}")
    t0 = time.time()
    fs = FieldSet(paths, a.dt,
                  cache_dtype=np.float16 if a.fp16_cache else np.float32)
    print(f"  field cache {fs.mem_gb:.2f} GB, {fs.nx}x{fs.ny}x{fs.nz}, "
          f"dt_dump={fs.dt_dump:.2f} s, window {fs.t[0]:.0f}-{fs.t[-1]:.0f} s "
          f"({time.time()-t0:.0f} s to load)")

    sgs = 1.0
    if a.sgs_most:
        from lpdm.les_stats import window_stats
        k_r = int(np.argmin(np.abs(fs.zk - a.z_target)))
        st = window_stats(paths[::max(1, len(paths) // 40)], k_r)
        zl = np.asarray(st["zlev"], dtype=np.float64)
        wwp = np.asarray(st["ww_prof"], dtype=np.float64)
        esp = np.asarray(st["esgs_prof"], dtype=np.float64)
        h = float(st["h"]); Lv = float(st["L"])
        zeta = zl / Lv if np.isfinite(Lv) and abs(Lv) > 1e-6 else np.zeros_like(zl)
        phi = np.where(zeta < 0.0, np.maximum(1.0 - 3.0 * zeta, 1.0) ** (1.0 / 3.0),
                       1.0 + 0.2 * np.minimum(zeta, 2.0))
        tgt2 = (1.25 * phi * float(st["ustar"])
                * np.maximum(1.0 - zl / max(h, 1.0), 0.0) ** 0.75) ** 2
        need = np.maximum(tgt2 - wwp, 0.0)
        have = np.maximum((2.0 / 3.0) * esp, 1e-9)
        taper = np.clip((0.2 * h - zl) / (0.1 * h), 0.0, 1.0)
        fac = 1.0 + taper * np.maximum(need / have - 1.0, 0.0)
        sgs = (zl, fac)
        print(f"  MOST floor ON: factor {fac.min():.2f}-{fac.max():.2f} over the column")
    lp = LPDM(fs, c0=a.c0, z_touch=a.ztouch, sgs_scale=sgs)
    ok = True
    for direction, label in ((-1, "BACKWARD (the mode footprints use)"),
                             (+1, "FORWARD (control)")):
        t0 = time.time()
        out = wellmixed.run_test(lp, fs, n=a.n, z_score_top=a.zscore,
                                 z_release_top=a.zrelease, t_limit=a.tlimit,
                                 direction=direction)
        ok &= wellmixed.report(out, label)
        print(f"  ({out['iters']} integrator steps, {time.time()-t0:.0f} s)")

    # ---- second gate: backward transit time from the 30 m receptor to the surface
    print(f"\n  --- backward transit time from the {a.z_target:.0f} m receptor ---")
    k_r = int(np.argmin(np.abs(fs.zk - a.z_target)))
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
        print(f"  median {q[2]/60:.1f} min at z = {zr:.1f} m. Transit scales roughly as "
              f"z/sigma_w, so the 30 m receptor's 180-290 s should fall to ~60-95 s here; "
              f"this median is what sizes t_back and therefore the window.")
    print(f"\n  STAGE 4 GATE: {'PASS' if ok and frac > 0.5 else 'FAIL'}")
    return 0 if (ok and frac > 0.5) else 1


if __name__ == "__main__":
    sys.exit(main())
