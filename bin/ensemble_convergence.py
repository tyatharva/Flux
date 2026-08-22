#!/usr/bin/env python3
"""Item 4: how many independent sub-windows does a footprint need to converge?

Stage 5 measured that two 15-minute halves of one window disagree as much as two entirely
separate runs. That makes sub-windows of a single long run the right ensemble unit -- they
are already independent, and they cost one LES run rather than N.

Method, from ONE backward integration:

  * release particles continuously across the whole available period
  * bin each touchdown by the RELEASE sub-window it came from, giving M independent
    footprints for the price of one integration
  * split the M sub-windows into a held-out reference half and a training half
  * for n = 1..M/2, average n randomly drawn training sub-windows and measure how far the
    result's peak and centroid sit from the held-out reference, over many draws

Also measures the sub-window autocorrelation, because "independent" is an assumption worth
checking: if adjacent sub-windows are correlated, the effective ensemble size is smaller
than the count and the convergence number is optimistic.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.fields import FieldSet, dump_series
from lpdm.footprint import FootprintGrid, source_area_overlap
from lpdm.model import LPDM


def metrics(f, xc, yc, res):
    tot = f.sum()
    if tot <= 0:
        return np.nan, np.nan
    fy = f.sum(axis=0)
    return float(xc[int(np.argmax(fy))]), float((fy * xc).sum() / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--dt", type=float, default=0.0340136)
    ap.add_argument("--ztarget", type=float, default=10.0)
    ap.add_argument("--tback", type=float, default=900.0)
    ap.add_argument("--nsub", type=int, default=18)
    ap.add_argument("--nper", type=int, default=900, help="particles per release time")
    ap.add_argument("--dtrel", type=float, default=5.0)
    ap.add_argument("--res", type=float, default=60.0)
    ap.add_argument("--wfloor", type=float, default=0.02)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--tag", default="ensemble")
    a = ap.parse_args()

    paths = dump_series(a.outdir)
    fs = FieldSet(paths, a.dt, verbose=False)
    k_r = int(np.argmin(np.abs(fs.zk - a.ztarget)))
    i_r, j_r = int(round(0.75 * fs.nx)), int(round(0.5 * fs.ny))
    xr, yr = fs.x0 + i_r * fs.dx, fs.y0 + j_r * fs.dy
    zr = float(fs.height(np.array([float(k_r)]), np.array([float(i_r)]),
                         np.array([float(j_r)]))[0])
    t_first = float(fs.t[0]) + a.tback
    t_last = float(fs.t[-1])
    span = t_last - t_first
    sub = span / a.nsub
    print(f"  {len(paths)} dumps, window {fs.t[0]:.0f}-{t_last:.0f} s, cache {fs.mem_gb:.2f} GB")
    print(f"  receptor k={k_r} z={zr:.3f} m;  release period {t_first:.0f}-{t_last:.0f} s "
          f"= {span:.0f} s split into {a.nsub} sub-windows of {sub:.0f} s")

    times = np.arange(t_first, t_last + 1e-9, a.dtrel)
    Ub = float(fs.u[fs.t >= t_first, k_r, j_r, i_r].mean())
    Vb = float(fs.v[fs.t >= t_first, k_r, j_r, i_r].mean())
    Wb = float(fs.w[fs.t >= t_first, k_r, j_r, i_r].mean())
    th = np.arctan2(Vb, Ub); ph = np.arctan2(Wb, np.hypot(Ub, Vb))
    ca, sa = np.cos(th), np.sin(th)

    lp = LPDM(fs, c0=3.0, z_touch=2.0, seed=0)
    grids = [FootprintGrid(-600.0, 4500.0, -1500.0, 1500.0, a.res) for _ in range(a.nsub)]
    t0 = time.time()
    B = 12
    for b0 in range(0, len(times), B):
        tb = times[b0:b0 + B]
        n = len(tb) * a.nper
        t = np.repeat(tb, a.nper)
        res = lp.run(np.full(n, xr), np.full(n, yr), np.full(n, zr), t, direction=-1,
                     t_limit=a.tback, reflect_touchdown=True, record_touchdown=True,
                     max_disp=fs.Lx)   # see lpdm/driver.py on periodic wrap-around
        wsf = (res["rel_w"] * np.cos(ph)
               - (res["rel_u"] * ca + res["rel_v"] * sa) * np.sin(ph))
        dx_ = res["td_x"] - xr; dy_ = res["td_y"] - yr
        X = -(dx_ * ca + dy_ * sa); Y = -dx_ * sa + dy_ * ca
        sidx = np.clip(((t[res["td_particle"]] - t_first) / sub).astype(int), 0, a.nsub - 1)
        for m in range(a.nsub):
            sel = sidx == m
            if not sel.any():
                continue
            r = dict(res)
            r["w_release"] = wsf
            r["td_particle"] = res["td_particle"][sel]
            r["td_w"] = res["td_w"][sel]
            r["td_x"] = X[sel]; r["td_y"] = Y[sel]
            r["n"] = int(((t >= t_first + m * sub) & (t < t_first + (m + 1) * sub)).sum())
            grids[m].add(r, 0.0, 0.0, w_floor=a.wfloor)
        print(f"    batch {b0//B+1}/{-(-len(times)//B)}  {time.time()-t0:.0f} s", flush=True)

    F = np.array([g.normalised("flux") for g in grids])
    xc, yc = grids[0].xc, grids[0].yc
    pk = np.array([metrics(f, xc, yc, a.res)[0] for f in F])
    ct = np.array([metrics(f, xc, yc, a.res)[1] for f in F])
    print(f"\n  --- per sub-window ({sub:.0f} s each) ---")
    print(f"  peak_x   mean {np.nanmean(pk):7.1f}  sd {np.nanstd(pk):6.1f} m")
    print(f"  centroid mean {np.nanmean(ct):7.1f}  sd {np.nanstd(ct):6.1f} m")

    print(f"\n  --- are sub-windows independent? lag autocorrelation of the metrics ---")
    for name, v in (("peak", pk), ("centroid", ct)):
        v0 = v - np.nanmean(v)
        den = np.nansum(v0 * v0)
        ac = [np.nansum(v0[:-L] * v0[L:]) / den if den > 0 else np.nan
              for L in range(1, min(5, a.nsub))]
        print(f"   {name:9s} " + "  ".join(f"lag{L}={x:+.2f}" for L, x in enumerate(ac, 1)))
    print(f"   (|r| below ~2/sqrt(M) = {2/np.sqrt(a.nsub):.2f} is consistent with independence)")

    half = a.nsub // 2
    rng = np.random.default_rng(0)
    ref = F[half:].mean(axis=0)
    rpk, rct = metrics(ref, xc, yc, a.res)
    print(f"\n  --- convergence against a held-out reference "
          f"({a.nsub-half} sub-windows, peak {rpk:.0f} m, centroid {rct:.0f} m) ---")
    print(f"  {'n':>3} {'|d peak| med':>13} {'p90':>7} {'|d centroid| med':>17} {'p90':>7}"
          f" {'80% overlap':>12}")
    conv_pk = conv_ct = None
    for n in range(1, half + 1):
        dp, dc, ov = [], [], []
        for _ in range(a.draws):
            pick = rng.choice(half, size=n, replace=False)
            m = F[pick].mean(axis=0)
            p, c = metrics(m, xc, yc, a.res)
            dp.append(abs(p - rpk)); dc.append(abs(c - rct))
            ov.append(source_area_overlap(np.maximum(m, 0), np.maximum(ref, 0)))
        dp, dc = np.array(dp), np.array(dc)
        print(f"  {n:3d} {np.median(dp):13.1f} {np.percentile(dp,90):7.1f} "
              f"{np.median(dc):17.1f} {np.percentile(dc,90):7.1f} {np.mean(ov)*100:11.1f}%")
        if conv_pk is None and np.percentile(dp, 90) <= a.res:
            conv_pk = n
        if conv_ct is None and np.percentile(dc, 90) <= 100.0:
            conv_ct = n
    print(f"\n  CORPUS PARAMETER")
    print(f"    peak location stable to one grid cell ({a.res:.0f} m) at the 90th "
          f"percentile:  n = {conv_pk if conv_pk else '>'+str(half)} sub-windows "
          f"({(conv_pk or half)*sub/60:.1f} min of sampling)")
    print(f"    centroid stable to 100 m at the 90th percentile:                 "
          f"n = {conv_ct if conv_ct else '>'+str(half)} sub-windows "
          f"({(conv_ct or half)*sub/60:.1f} min)")
    np.savez_compressed(f"results/{a.tag}.npz", F=F, xc=xc, yc=yc, pk=pk, ct=ct,
                        sub=sub, ref=ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
