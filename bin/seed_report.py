#!/usr/bin/env python3
"""What one seed job actually cost and achieved. The numbers the other 17 get rented on.

A seed job is projected before it runs -- 0.0149 s/step, ~1.02 GPU-h per simulated hour,
73.3 MB home, all seven stationarity limits passing by 3.0 h. This measures each of those
against the run, because the remaining seventeen are budgeted on them and a 10% error in
the wall-to-sim ratio is a 5-hour error across the library.

Reported and not assumed:

  wall-to-sim        measured s/step against the 0.0149 the planner used
  the seven limits   which PASS, and WHICH BINDS LAST -- the one closest to its threshold
                     is the one that sets the 3.0 h budget, and it is worth knowing which
  laminarisation     a stable LES can stop being turbulent. TKE trend over the run, and
                     the resolved fraction at the receptor. This is the failure mode that
                     looks like a converged run.
  achieved direction against the base angle after Ekman backing -- the forcing angle is
                     not the achieved one, and the library is labelled by the achieved
  k0/k1              the accuracy-CFL check; ~9 means dt past the boundary
  artifact size      against 73.3 MB

usage: seed_report.py jobs/seed_sbl_a030 [--wall-seconds N] [--out FILE]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--wall-seconds", type=float, default=None)
    ap.add_argument("--sps-planned", type=float, default=0.0149)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.job, "manifest.json")))
    ret = os.path.join(a.job, "return")
    st_path = os.path.join(ret, "stationarity.json")
    if not os.path.exists(st_path):
        print(f"FATAL: {st_path} does not exist", file=sys.stderr)
        return 2
    st = json.load(open(st_path))
    run = man["run"]

    o = []
    p = o.append
    p(f"=== seed {man['job']}: {man['rung']} ({man['regime']}), base angle "
      f"{man['base_angle_deg']:.0f} deg ===")

    # ---- cost --------------------------------------------------------------------
    steps = run["steps_total"]
    sim_h = steps * run["dt"] / 3600.0
    p(f"\n--- cost, against the projection the other 17 are budgeted on ---")
    p(f"  {steps} steps x dt {run['dt']} = {sim_h:.3f} simulated hours")
    if a.wall_seconds:
        sps = a.wall_seconds / steps
        p(f"  measured wall {a.wall_seconds/3600:.3f} h  ->  {sps*1000:.3f} ms/step")
        p(f"  planned                              {a.sps_planned*1000:.3f} ms/step"
          f"   ({100*(sps/a.sps_planned-1):+.1f}%)")
        p(f"  wall-to-sim ratio {a.wall_seconds/3600/sim_h:.3f} GPU-h per simulated hour"
          f"   (projection 1.02, i.e. {100*(a.wall_seconds/3600/sim_h/1.02-1):+.1f}%)")
        p(f"  the library at this rate: 18 x {a.wall_seconds/3600:.2f} h = "
          f"{18*a.wall_seconds/3600:.1f} GPU-h  (projected 52)")
    else:
        p("  (no --wall-seconds given; cost not measured)")

    # ---- the seven limits --------------------------------------------------------
    p(f"\n--- the seven stationarity limits, scored on the last {st['score_h']} h ---")
    p(f"  {'quantity':<30}{'mean':>10}{'trend':>11}{'limit':>8}{'used':>8}  verdict")
    rows = sorted(st["gated"], key=lambda r: -abs(r["trend_pct_per_h"]) / r["limit"])
    for r in rows:
        frac = abs(r["trend_pct_per_h"]) / r["limit"]
        p(f"  {r['name']:<30}{r['mean']:10.4f}{r['trend_pct_per_h']:+10.2f}%"
          f"{r['limit']:8.1f}{100*frac:7.0f}%  {'ok' if r['ok'] else 'DRIFTING'}")
    p(f"  ALL SEVEN: {'PASS' if st['pass'] else 'FAIL'}")
    tight = rows[0]
    p(f"  BINDS LAST: {tight['name']} at {100*abs(tight['trend_pct_per_h'])/tight['limit']:.0f}% "
      f"of its limit ({tight['trend_pct_per_h']:+.2f} %/h against {tight['limit']:.1f})")
    p(f"  reported, not gated: " + ", ".join(
        f"{r['name']} {r['trend_pct_per_h']:+.2f} %/h" for r in st["reported"]))

    # ---- laminarisation ----------------------------------------------------------
    dumps = sorted(glob.glob(os.path.join(a.job, "output", "*.[0-9]*")),
                   key=lambda q: int(q.rsplit(".", 1)[1]))
    p(f"\n--- is the stable boundary layer still turbulent? ---")
    if len(dumps) < 3:
        p(f"  only {len(dumps)} dumps on disk; cannot judge")
    else:
        from netCDF4 import Dataset
        ts, tkes, usts, ww0 = [], [], [], []
        for q in dumps:
            with Dataset(q) as ds:
                g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
                u, v, w = g("u"), g("v"), g("w")
                z = g("zPos")[:, 0, 0]
                e = np.maximum(g("TKE_0"), 0.0)
                usts.append(float(g("fricVel").mean()))
            pr = lambda arr: arr - arr.mean(axis=(-2, -1), keepdims=True)
            tk = 0.5 * ((pr(u) ** 2 + pr(v) ** 2 + pr(w) ** 2).mean(axis=(-2, -1)))
            tkes.append(float(np.trapezoid(tk, z)))
            ww = (pr(w) ** 2).mean(axis=(-2, -1))
            ww0.append(float(ww[0] / max(ww[1], 1e-30)))
            ts.append(int(q.rsplit(".", 1)[1]) * run["dt"] / 3600.0)
        ts, tkes, usts = np.array(ts), np.array(tkes), np.array(usts)
        half = ts >= ts[-1] / 2
        sl = np.polyfit(ts[half], tkes[half], 1)[0]
        p(f"  column-integrated TKE {tkes[0]:.3f} -> {tkes[-1]:.3f} m3/s2;"
          f" trend over the last half {100*sl/max(abs(tkes[half].mean()),1e-30):+.1f} %/h")
        p(f"  u* {usts[0]:.4f} -> {usts[-1]:.4f} m/s   (min over the run {usts.min():.4f})")
        # u* IS THE TEST, NOT TKE. The first version keyed on column TKE and said "no"
        # on a run whose u* had fallen 58% and whose z/L had reached 2.67 -- because the
        # TKE integral is dominated by gravity-wave variance aloft, which GROWS as the
        # turbulence dies (FASTEDDY_TRAPS.md 15). u* is a genuine turbulent flux and it
        # cannot be faked by waves.
        u_slope = np.polyfit(ts[half], usts[half], 1)[0]
        u_rel = 100 * u_slope / max(abs(usts[half].mean()), 1e-30)
        p(f"  u* trend over the last half {u_rel:+.1f} %/h; peak-to-final "
          f"{100*(usts[-1]/max(usts.max(), 1e-30) - 1):+.0f}%")
        lam = usts[-1] < 0.6 * usts.max() or usts[-1] < 0.05 or u_rel < -20.0
        p(f"  DECAYING TURBULENCE: "
          f"{'YES -- u* is collapsing; this state is not a usable seed' if lam else 'no'}")
        p(f"  k0/k1 at the last dump {ww0[-1]:.3f}   (must be < 1; ~9 means dt too large)")

    # ---- achieved vs asked -------------------------------------------------------
    ach = man.get("achieved", {})
    tgt = man["target"]
    p(f"\n--- achieved vs asked ---")
    if ach:
        gd = tgt["G_dir_from_deg"]
        wd = ach.get("wdir")
        p(f"  geostrophic forcing FROM {gd:.1f} deg (base angle {man['base_angle_deg']:.0f})")
        if wd is not None:
            back = ((gd - wd + 180.0) % 360.0) - 180.0
            p(f"  achieved receptor wind FROM {wd:.1f} deg  ->  Ekman backing "
              f"{back:+.1f} deg  (measured 12-24 deg neutrally, 7-13 convectively)")
        p(f"  z_i {ach.get('zi', float('nan')):.0f} m against a {tgt['zi_m']:.0f} m target")
        p(f"  u* {ach.get('ustar', float('nan')):.4f},  U {ach.get('U', float('nan')):.3f} m/s,"
          f"  sigma_w {ach.get('sigma_w', float('nan')):.4f}")
    else:
        p("  the manifest carries no `achieved` block")

    # ---- what comes home ---------------------------------------------------------
    p(f"\n--- what comes home ---")
    tot = 0
    for f in sorted(glob.glob(os.path.join(ret, "*"))):
        sz = os.path.getsize(f)
        tot += sz
        p(f"  {os.path.basename(f):<26}{sz/1e6:9.2f} MB")
    rst = os.path.join(ret, "seed_restart.nc")
    if os.path.exists(rst):
        sz = os.path.getsize(rst) / 1e6
        p(f"  restart {sz:.1f} MB against the 73.3 MB estimate ({100*(sz/73.3-1):+.1f}%)")
    p(f"  return/ total {tot/1e6:.1f} MB")

    txt = "\n".join(o)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
