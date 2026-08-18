#!/usr/bin/env python3
"""Unit test of the footprint estimator, independent of any LES.

Replaces the LES fields with HOMOGENEOUS turbulence -- constant U, constant sigma_s^2,
constant eps, zero gradients, zero resolved w -- and a reflecting surface. For a
horizontally uniform surface source in that flow, every bit of surface flux crosses the
measurement height, so

    integral of f_flux over all touchdowns  ->  1

as the backward integration time grows. Any constant factor wrong in the 2/|w_td| weight,
in the w_release weighting, or in the particle bookkeeping shows up here as a number that
is not 1, with nothing about the LES or the SGS closure able to hide it.

The touchdown height is set to the lowest LES level so the MOST sub-layer never engages --
this is a test of the estimator, not of the sub-layer.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.fields import FieldSet, dump_series
from lpdm.footprint import FootprintGrid
from lpdm.model import LPDM

U0, E0, C0, TL_TARGET = 5.0, 0.60, 3.0, 20.0
NSEED = 8
LIDS = (60.0, 100.0)


def main(outdir, t_backs=(300.0, 600.0, 900.0, 1800.0), n=20000):
    fs = FieldSet(dump_series(outdir)[:3], 0.0625, verbose=False)
    sig2 = (2.0 / 3.0) * E0
    eps0 = 2.0 * sig2 / (C0 * TL_TARGET)
    fs.u[:] = U0; fs.v[:] = 0.0; fs.w[:] = 0.0
    fs.e[:] = E0; fs.eps[:] = eps0; fs.dsig2dz[:] = 0.0
    fs.ustar[:] = 0.4; fs.z0m[:] = 0.03; fs.invL[:] = 0.0
    z_touch = float(fs.zk[0])
    zr = float(fs.zk[1])
    print(f"  homogeneous: U={U0} m/s  sigma_w={np.sqrt(sig2):.3f} m/s  "
          f"T_L={TL_TARGET} s  eps={eps0:.5f}")
    print(f"  receptor {zr:.1f} m, touchdown (= lowest LES level) {z_touch:.1f} m, "
          f"{n:,} particles per t_back\n")
    print(f"  {'t_back (s)':>11} {'touchdowns':>12} {'int f_flux':>11} {'int f_conc':>11}"
          f" {'peak_x (m)':>11}")
    lp = LPDM(fs, c0=C0, z_touch=z_touch, seed=3)
    for tb in t_backs:
        res = lp.run(np.full(n, fs.x0 + 0.5 * fs.Lx), np.full(n, fs.y0 + 0.5 * fs.Ly),
                     np.full(n, zr), float(fs.t[-1]), direction=-1, t_limit=tb,
                     reflect_touchdown=True, record_touchdown=True)
        g = FootprintGrid(-600.0, 20000.0, -3000.0, 3000.0, 20.0)
        r = dict(res)
        r["td_x"] = -(res["td_x"] - (fs.x0 + 0.5 * fs.Lx))
        r["td_y"] = res["td_y"] - (fs.y0 + 0.5 * fs.Ly)
        g.add(r, 0.0, 0.0)
        m = g.metrics("flux")
        print(f"  {tb:11.0f} {len(res['td_x']):12,} {g.integral_all('flux'):11.3f}"
              f" {g.integral_all('conc'):11.3f} {m.get('peak_x', np.nan):11.0f}")
    print("\n  int f_flux must approach 1. It approaches from below because a finite\n"
          "  backward time truncates the far-field tail -- that shortfall is physical\n"
          "  bookkeeping, not an estimator error.")

    # Sharper variant: cap the slab with a reflecting lid. This is NOT a test that the
    # answer is 1 -- in a closed slab it is not. With a surface source and no flux through
    # the lid, the whole slab accumulates uniformly at Q/H, so the flux surviving to height
    # z is Q(1 - (z - z_td)/(z_lid - z_td)): a straight line from Q at the floor to 0 at the
    # lid. That gives a KNOWN, lid-dependent target instead of an asymptote, so a wrong
    # constant in the 2/|w_td| weight cannot hide behind slow convergence. The lids are
    # chosen shallow enough that the slab mixes in H^2/(sigma^2 T_L) << t_back.
    print(f"\n  --- closed slab: predicted flux is Q(1 - (z_r-z_td)/H), not Q ---")
    print(f"  {'lid (m)':>8} {'mix time (s)':>13} {'predicted':>10} {'measured':>10}"
          f" {'s.e.':>7} {'sigmas':>8}")
    K = sig2 * TL_TARGET
    for lid in LIDS:
        vals = []
        for sd in range(NSEED):
            lp2 = LPDM(fs, c0=C0, z_touch=z_touch, seed=200 + sd)
            r = lp2.run(np.full(n, fs.x0 + 0.5 * fs.Lx), np.full(n, fs.y0 + 0.5 * fs.Ly),
                        np.full(n, zr), float(fs.t[-1]), direction=-1,
                        t_limit=t_backs[-1], reflect_touchdown=True,
                        record_touchdown=True, z_ceil=lid)
            wt = r["w_release"][r["td_particle"]] * 2.0 / np.maximum(r["td_w"], 1e-6)
            vals.append(wt.sum() / n)
        v = np.array(vals); se = v.std(ddof=1) / np.sqrt(len(v))
        pred = 1.0 - (zr - z_touch) / (lid - z_touch)
        tmix = (lid - z_touch) ** 2 / K
        print(f"  {lid:8.0f} {tmix:13.0f} {pred:10.3f} {v.mean():10.3f} {se:7.3f}"
              f" {(v.mean()-pred)/se:+8.2f}")

    # Single-realisation scatter is large: the flux estimator is a signed sum, so its
    # variance is set by the near-cancellation of up- and down-released trajectories.
    # Independent seeds separate estimator BIAS from that scatter.
    print(f"\n  --- bias: {NSEED} independent seeds at t_back={t_backs[-1]:.0f} s ---")
    vals = []
    for sd in range(NSEED):
        lp = LPDM(fs, c0=C0, z_touch=z_touch, seed=100 + sd)
        r = lp.run(np.full(n, fs.x0 + 0.5 * fs.Lx), np.full(n, fs.y0 + 0.5 * fs.Ly),
                   np.full(n, zr), float(fs.t[-1]), direction=-1, t_limit=t_backs[-1],
                   reflect_touchdown=True, record_touchdown=True)
        wt = r["w_release"][r["td_particle"]] * 2.0 / np.maximum(r["td_w"], 1e-6)
        vals.append(wt.sum() / n)
        print(f"   seed {sd}: {vals[-1]:.4f}", flush=True)
    v = np.array(vals)
    se = v.std(ddof=1) / np.sqrt(len(v))
    print(f"  mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  standard error {se:.4f}")
    print(f"  distance from 1: {(v.mean()-1)/se:+.2f} standard errors "
          f"(plus a negative offset from the truncated tail)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/s30_spinup/output")
