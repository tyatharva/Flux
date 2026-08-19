#!/usr/bin/env python3
"""Items 1 and 2: which reference frame, and why is the terrain footprint integral != 1?

Loads one sampling window and computes the flux-footprint integral under

  * MODEL frame, raw          -- weight by w                    (no mean removal)
  * MODEL frame, mean removed -- weight by w - <w>              (Reynolds, what we had)
  * STREAMLINE frame          -- weight by the double-rotated w (what an EC tower reports)

across a range of backward integration times, and reports alongside them the concentration
integral and the wrap-around fraction. Together those separate the three candidates:

  integral grows with t_back and does not saturate  -> periodic wrap-around double counting
  integral saturates away from 1                    -> mean-subtraction artifact
  integral set by the mean wind, not by t_back      -> genuine advective non-closure

The mean-subtraction term is w_bar times the CONCENTRATION integral, and the concentration
integral is unbounded, so printing it next to the flux integral makes that mechanism
visible rather than inferred.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.fields import FieldSet, dump_series
from lpdm.model import LPDM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--dt", type=float, default=0.0625)
    ap.add_argument("--ztarget", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--dtrel", type=float, default=5.0)
    ap.add_argument("--wfloor", type=float, default=0.02)
    ap.add_argument("--maxdisp", type=float, default=0.0, help="0 = off")
    ap.add_argument("--tbacks", default="300,600,900,1500")
    a = ap.parse_args()

    paths = dump_series(a.outdir)
    fs = FieldSet(paths, a.dt, verbose=False)
    k_r = int(np.argmin(np.abs(fs.zk - a.ztarget)))
    i_r, j_r = int(round(0.75 * fs.nx)), int(round(0.5 * fs.ny))
    xr, yr = fs.x0 + i_r * fs.dx, fs.y0 + j_r * fs.dy
    zg_r = float(fs.ground(np.array([float(i_r)]), np.array([float(j_r)]))[0])
    zr = float(fs.height(np.array([float(k_r)]), np.array([float(i_r)]),
                         np.array([float(j_r)]))[0])
    print(f"  {len(paths)} dumps, window {fs.t[0]:.0f}-{fs.t[-1]:.0f} s, cache {fs.mem_gb:.2f} GB")
    print(f"  receptor k={k_r}  z={zr:.3f} m ASL = {zr-zg_r:.3f} m AGL   "
          f"domain {fs.Lx:.0f} x {fs.Ly:.0f} m")

    lp = LPDM(fs, c0=3.0, z_touch=2.0, seed=0)
    md = a.maxdisp if a.maxdisp > 0 else None
    print(f"\n  {'t_back':>7} {'releases':>9} {'model raw':>10} {'model -mean':>12}"
          f" {'STREAMLINE':>11} {'conc int':>9} {'w_bar*C':>9} {'wrapped':>8}")
    for tb in [float(v) for v in a.tbacks.split(",")]:
        t_first = float(fs.t[0]) + tb
        if t_first >= fs.t[-1]:
            print(f"  {tb:7.0f}  window too short"); continue
        times = np.arange(t_first, float(fs.t[-1]) + 1e-9, a.dtrel)
        t = np.repeat(times, a.n)
        n = len(t)
        sel = (fs.t >= t_first - 1e-6)
        Ub = float(fs.u[sel, k_r, j_r, i_r].mean())
        Vb = float(fs.v[sel, k_r, j_r, i_r].mean())
        Wb = float(fs.w[sel, k_r, j_r, i_r].mean())
        th = np.arctan2(Vb, Ub); ph = np.arctan2(Wb, np.hypot(Ub, Vb))

        res = lp.run(np.full(n, xr), np.full(n, yr), np.full(n, zr), t, direction=-1,
                     t_limit=tb, reflect_touchdown=True, record_touchdown=True,
                     max_disp=md)
        p = res["td_particle"]
        S = 2.0 / np.maximum(res["td_w"], a.wfloor)
        conc = S.sum() / n
        raw = (res["rel_w"][p] * S).sum() / n
        dem = ((res["rel_w"][p] - Wb) * S).sum() / n
        wsf = (res["rel_w"] * np.cos(ph)
               - (res["rel_u"] * np.cos(th) + res["rel_v"] * np.sin(th)) * np.sin(ph))
        sfl = (wsf[p] * S).sum() / n
        wrapped = (np.abs(res["td_x"] - xr) > fs.Lx).mean() if len(res["td_x"]) else 0.0
        print(f"  {tb:7.0f} {len(times):9d} {raw:10.3f} {dem:12.3f} {sfl:11.3f}"
              f" {conc:9.2f} {-Wb*conc:9.3f} {wrapped*100:7.1f}%")
    print(f"\n  (w_bar*C is the mean-subtraction term: it is what separates 'model raw' from")
    print(f"   'model -mean'. It grows with the concentration integral, which is unbounded.)")


if __name__ == "__main__":
    sys.exit(main())
