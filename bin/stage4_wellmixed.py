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
    ap.add_argument("--sgs-most-legacy", action="store_true",
                    help="reproduce the RETIRED 0.1h-0.2h factor taper, so the bias it "
                         "introduced can be measured against the same fields.")
    ap.add_argument("--sgs-most-mode", default="surface",
                    choices=("surface", "mixed", "blend"))
    ap.add_argument("--no-subgrid-weight", action="store_true",
                    help="DIAGNOSTIC: drop the sub-grid-fraction weighting.")
    ap.add_argument("--no-eps-consistent", action="store_true",
                    help="DIAGNOSTIC: raise sigma^2 without raising eps.")
    ap.add_argument("--sgs-scale", type=float, default=1.0,
                    help="DIAGNOSTIC: a CONSTANT multiplier on the sub-grid variance, "
                         "with no height dependence and therefore no dsc/dz term. It "
                         "separates two hypotheses that the height-dependent floor "
                         "confounds -- whether the gate fails because of the floor's "
                         "GRADIENT or simply because of its MAGNITUDE.")
    ap.add_argument("--sgs-most-form", default="multiplicative",
                    choices=("additive", "multiplicative"),
                    help="how the floor reaches the drift. Additive is production; the "
                         "multiplicative factor amplifies the field's own gradient error "
                         "by sc (up to 10x convectively) and is kept for comparison.")
    ap.add_argument("--dmap", default=None,
                    help="displacement map, for the receptor-column d the floor needs.")
    ap.add_argument("--d-recept", type=float, default=None)
    ap.add_argument("--tag", default=None,
                    help="write the per-direction verdict to results/<tag>.json")
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

    sgs, off = a.sgs_scale, None
    if a.sgs_scale != 1.0:
        print(f"  DIAGNOSTIC: constant sub-grid variance x{a.sgs_scale:.3f} "
              f"(no height dependence, so the drift's dsc/dz term is exactly zero)")
    if a.sgs_most:
        # THE GATE MUST RUN THE PRODUCTION CLOSURE, and it must run the SAME CODE that
        # produces it. This block used to be a second copy of the floor that had already
        # drifted from lpdm/driver.py's (it never received the displacement correction),
        # so the gate was validating a closure the footprints do not use.
        from lpdm.les_stats import window_stats
        from lpdm.sgs_floor import check_monotone, most_floor
        k_r = int(np.argmin(np.abs(fs.zk - a.z_target)))
        st = window_stats(paths[::max(1, len(paths) // 40)], k_r)
        d_r = 0.0
        if a.dmap:
            dm = np.load(a.dmap)
            d_r = float(dm[fs.ny // 2, fs.nx // 2]) if a.d_recept is None else a.d_recept
        elif a.d_recept is not None:
            d_r = float(a.d_recept)
        fl = most_floor(st, d_r=d_r, mode=a.sgs_most_mode, legacy=a.sgs_most_legacy,
                        subgrid_weight=not a.no_subgrid_weight)
        n_new, worst = check_monotone(fl)
        # Same delivery choice as production, for the same reason: the gate has to run
        # the closure the footprints run, down to how the correction reaches the drift.
        if a.sgs_most_legacy or a.sgs_most_form == "multiplicative":
            sgs = (fl["zl"], fl["fac"])
        else:
            off = (fl["zl"], fl["delta"])
        kk = int(np.argmin(np.abs(fl["zl"] - a.z_target)))
        print(f"  MOST floor ON{' [LEGACY TAPER]' if a.sgs_most_legacy else ''}"
              f" [{'multiplicative' if off is None else 'additive'}"
              f"{'' if a.no_subgrid_weight else ', f_sgs-weighted'}"
              f"{'' if a.no_eps_consistent else ', eps-consistent'}]: "
              f"factor {fl['fac'].min():.2f}-{fl['fac'].max():.2f} over the column, "
              f"{fl['fac'][kk]:.3f} at the receptor (d={d_r:.2f} m)")
        if not a.sgs_most_legacy:
            print(f"  floor active below z={fl['zl'][fl['kpk']]:.0f} m "
                  f"(the model's own sigma_w^2 peak, {fl['base'][fl['kpk']]:.4f} m2/s2)")
        print(f"  floor-induced turnovers in sigma_w^2 below the peak: {n_new} "
              f"(worst {worst:+.2%}) -- must be 0")
        # The profile itself, because "it failed" without the profile is not a diagnosis.
        print(f"  {'z(m)':>8} {'resolved':>10} {'(2/3)e':>10} {'factor':>8} "
              f"{'sigma_w^2':>10} {'d/dz':>10}")
        sel = [i for i in range(len(fl["zl"])) if fl["zl"][i] <= max(2.0 * fl["zl"][fl["kpk"]], 150.0)]
        dz = np.gradient(fl["sig2"], fl["zl"])
        for i in sel[::max(1, len(sel) // 12)]:
            print(f"  {fl['zl'][i]:8.1f} {fl['wwp'][i]:10.4f} {fl['have'][i]:10.4f} "
                  f"{fl['fac'][i]:8.2f} {fl['sig2'][i]:10.4f} {dz[i]:+10.5f}")
    lp = LPDM(fs, c0=a.c0, z_touch=a.ztouch, sgs_scale=sgs, sgs_offset=off,
              sgs_eps_consistent=not a.no_eps_consistent)
    ok = True
    verdict = {}
    for direction, label in ((-1, "BACKWARD (the mode footprints use)"),
                             (+1, "FORWARD (control)")):
        t0 = time.time()
        out = wellmixed.run_test(lp, fs, n=a.n, z_score_top=a.zscore,
                                 z_release_top=a.zrelease, t_limit=a.tlimit,
                                 direction=direction)
        ok &= wellmixed.report(out, label)
        verdict["backward" if direction < 0 else "forward"] = out["metrics"]
        print(f"  ({out['iters']} integrator steps, {time.time()-t0:.0f} s)")
    # A CORRECT LAGRANGIAN STOCHASTIC MODEL IN A STATIONARY FIELD IS WELL MIXED IN BOTH
    # DIRECTIONS. Backward-only agreement is not a pass: the retired floor passed backward
    # at 7.51% rms and failed forward at 1.258 in the lowest three bins, and the footprints
    # only ever run backward, so a one-sided test could not have seen it.
    print(f"\n  BOTH DIRECTIONS: {'PASS' if ok else 'FAIL'}"
          f"  (asymmetry is the primary diagnostic; backward alone proves nothing)")

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
    if a.tag:
        import json
        verdict["config"] = dict(sgs_most=bool(a.sgs_most), legacy=bool(a.sgs_most_legacy),
                                 mode=a.sgs_most_mode, z_target=a.z_target,
                                 form=("multiplicative" if off is None else "additive"),
                                 subgrid_weight=not a.no_subgrid_weight,
                                 eps_consistent=not a.no_eps_consistent,
                                 outdir=a.outdir, transit_frac=float(frac))
        if a.sgs_most:
            verdict["floor"] = dict(
                zl=[float(v) for v in fl["zl"]], fac=[float(v) for v in fl["fac"]],
                sig2=[float(v) for v in fl["sig2"]], base=[float(v) for v in fl["base"]],
                kpk=int(fl["kpk"]), n_new_turnovers=int(n_new))
        verdict["pass"] = bool(ok and frac > 0.5)
        with open(f"results/{a.tag}.json", "w") as f:
            json.dump(verdict, f, indent=2, default=float)
        print(f"  -> results/{a.tag}.json")
    return 0 if (ok and frac > 0.5) else 1


if __name__ == "__main__":
    sys.exit(main())
