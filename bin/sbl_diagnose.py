#!/usr/bin/env python3
"""WHICH stable-boundary-layer failure is this? Decoupling, or resolution starvation?

Two different things kill a stable LES and they need different responses, so telling them
apart is worth a script rather than a glance at u*.

  RUNAWAY SURFACE DECOUPLING (FASTEDDY_TRAPS.md 15). A cold start has no turbulence, so a
  prescribed cooling builds a near-discontinuous inversion before any can develop and the
  stratification then prevents it developing. FIXABLE -- warm up neutrally first.
  Signature: Ri_g explodes (~1e8 measured), the mean wind aloft sits at EXACTLY the
  geostrophic value with no Ekman turning, dtheta/dz at the first level reaches thousands
  of K/km, and the boundary layer detaches from the flow.

  RESOLUTION STARVATION. The Ozmidov scale falls to a few times Delta, so there is no band
  between the filter width and the largest overturning eddy. NOT fixable by forcing: the
  surface layer stays perfectly healthy while resolved variance drains out of it and
  reappears aloft as internal waves. Signature: Ri_g stays WELL BELOW critical, Ekman
  backing is normal, the inversion is ordinary -- and the height containing 95% of the
  column TKE marches upward into the free atmosphere while u* decays.

The second signature is the useful one, and it is why this reports zTKE95 rather than the
TKE integral: the integral RISES during a collapse (FASTEDDY_TRAPS.md 15) because wave
energy aloft replaces turbulence below. Asking WHERE the energy is separates them.

usage: sbl_diagnose.py <job_or_run_dir> --dt 0.01461988 [--wth -0.012] [--G 10.0]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from netCDF4 import Dataset

VONK, G_ACC, TH0 = 0.4, 9.81, 290.0
RI_CRIT = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("d", help="directory of dumps")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--wth", type=float, default=-0.012, help="kinematic surface heat flux")
    ap.add_argument("--geo", type=float, default=10.0, help="geostrophic wind magnitude")
    ap.add_argument("--zm", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()

    ps = sorted(glob.glob(os.path.join(a.d, "*.[0-9]*")),
                key=lambda q: int(q.rsplit(".", 1)[1]))
    if len(ps) < 3:
        print(f"FATAL: only {len(ps)} dumps in {a.d}", file=sys.stderr)
        return 2
    sel = ps[::a.stride]
    if ps[-1] not in sel:
        sel.append(ps[-1])

    print(f"=== stable-collapse diagnosis: {a.d} ===")
    print(f"  w'th' = {a.wth:+.4f} K m/s, G = {a.geo:.1f} m/s, receptor {a.zm:.0f} m")
    print(f"  {'t_h':>6}{'u*':>8}{'z/L':>8}{'U(zm)':>8}{'backing':>9}{'|U-G| aloft':>12}"
          f"{'Ri_g@20m':>10}{'dth/dz@2m':>11}{'zTKE95':>8}")
    rows = []
    for q in sel:
        with Dataset(q) as ds:
            g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
            u, v, w, th = g("u"), g("v"), g("w"), g("theta")
            z = g("zPos")[:, 0, 0]
            ust = float(g("fricVel").mean())
        ub, vb, thb = u.mean(axis=(-2, -1)), v.mean(axis=(-2, -1)), th.mean(axis=(-2, -1))
        sp = np.hypot(ub, vb)
        kr = int(np.argmin(np.abs(z - a.zm)))
        zl = (VONK * G_ACC * a.zm * (-a.wth) / max(ust, 1e-9) ** 3 / TH0)
        wdir = (270.0 - np.degrees(np.arctan2(vb[kr], ub[kr]))) % 360.0
        kfree = int(np.argmin(np.abs(z - 300.0)))
        dev = abs(sp[kfree] - a.geo)
        dudz, dvdz = np.gradient(ub, z), np.gradient(vb, z)
        dthdz = np.gradient(thb, z)
        k20 = int(np.argmin(np.abs(z - 20.0)))
        sh2 = dudz[k20] ** 2 + dvdz[k20] ** 2
        rig = (G_ACC / thb[k20] * dthdz[k20]) / max(sh2, 1e-12)
        pr = lambda arr: arr - arr.mean(axis=(-2, -1), keepdims=True)
        tk = 0.5 * ((pr(u) ** 2 + pr(v) ** 2 + pr(w) ** 2).mean(axis=(-2, -1)))
        cum = np.cumsum(tk * np.gradient(z))
        ztk = float(z[min(np.searchsorted(cum, 0.95 * cum[-1]), len(z) - 1)])
        t = int(q.rsplit(".", 1)[1]) * a.dt / 3600.0
        if t <= 0:
            continue
        rows.append((t, ust, rig, ztk, dev))
        print(f"  {t:6.2f}{ust:8.4f}{zl:8.3f}{sp[kr]:8.3f}{wdir:9.1f}{dev:12.4f}"
              f"{rig:10.3f}{dthdz[0]*1000:11.1f}{ztk:8.0f}")

    t = np.array([r[0] for r in rows]); ust = np.array([r[1] for r in rows])
    rig = np.array([r[2] for r in rows]); ztk = np.array([r[3] for r in rows])
    dev = np.array([r[4] for r in rows])
    print()
    # BOTH ARE SCORED, AND THEY ARE STAGES RATHER THAN ALTERNATIVES. Measured on the two
    # runs that died: the weakly-stable one is starving with Ri_g at 0.033 and the flow
    # aloft still DEPARTING from geostrophic, while the GABLS1-regime one -- further down
    # the same road -- has Ri_g at 0.198 and the flow PINNED to 0.002 m/s of geostrophic.
    # Starvation comes first, while the surface layer is still perfectly healthy;
    # decoupling is where it ends up. Forcing a single label on that would lose the fact
    # that the earlier, diagnosable stage is the one that matters.
    starved = (ztk[-1] > 3.0 * np.median(ztk[:max(len(ztk)//3, 1)])
               and ust[-1] < 0.7 * ust.max())
    decoupled = (rig.max() > RI_CRIT) or dev[-1] < 5e-3
    print(f"  Ri_g at 20 m peaks at {rig.max():.3f} against a critical {RI_CRIT}")
    print(f"  the flow at 300 m ends {dev[-1]:.3f} m/s from geostrophic "
          f"({'departing -- still coupled' if dev[-1] > 1e-2 else 'PINNED -- decoupled'})")
    print(f"  zTKE95 runs {ztk[0]:.0f} -> {ztk[-1]:.0f} m; u* runs {ust[0]:.4f} -> "
          f"{ust[-1]:.4f} ({100*ust[-1]/max(ust.max(),1e-30):.0f}% of peak)")
    print()
    print(f"  RESOLUTION STARVATION : {'YES' if starved else 'no'}  -- resolved variance")
    print(f"      draining out of the layer and reappearing aloft while the surface layer")
    print(f"      itself stays healthy. This is the ROOT stage and it is NOT fixable by")
    print(f"      changing the forcing: it is Delta against the Ozmidov scale (bin/ozmidov.py).")
    print(f"  SURFACE DECOUPLING    : {'YES' if decoupled else 'not yet'}  -- Ri_g at or past")
    print(f"      critical and the flow aloft pinned to exactly geostrophic. This is the END")
    print(f"      stage. Reached from a COLD START it is a separate, FIXABLE fault (warm up")
    print(f"      neutrally first, FASTEDDY_TRAPS.md 15); reached after starvation it is not.")
    if starved and not decoupled:
        print("  -> starving, not yet decoupled: the earlier and more diagnosable stage.")
    elif starved and decoupled:
        print("  -> starved AND decoupled: further down the same road.")
    elif decoupled and not starved:
        print("  -> decoupled without starving: suspect the COLD START, which is fixable.")
    else:
        print("  -> neither signature is clear; report the numbers, not a label.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
