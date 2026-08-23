#!/usr/bin/env python3
"""Is the window stationary, and is the resolved field vertically neutral?

The well-mixed condition presumes a STATIONARY field. A convective boundary layer grows by
entrainment and a compressible solver lets a warming layer expand, so both assumptions have
to be checked rather than assumed -- and a forward/backward asymmetry in the well-mixed
test is exactly what a non-stationary field or a non-zero slab-mean w would produce, since
the two integrations traverse the same window in opposite time order.

Reports, per dump: the slab-mean resolved w profile (which must be ~0 for a uniform release
to stay uniform), z_i from the minimum heat-flux level, and the domain-mean sigma_w.

usage: window_stationarity.py <windowdir> --dt DT [--every N]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.fields import dump_series
from netCDF4 import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windir")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--every", type=int, default=40)
    a = ap.parse_args()
    paths = dump_series(a.windir)[:: a.every]
    print(f"  {len(paths)} dumps sampled from {a.windir}")

    zs, rows = None, []
    for p in paths:
        with Dataset(p) as ds:
            step = int(p.rsplit(".", 1)[1])
            if zs is None:
                zp = np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))
                zs = zp[:, 0, 0] if zp.ndim == 3 else zp
            w = np.squeeze(np.asarray(ds["w"][:], dtype=np.float64))
            th = np.squeeze(np.asarray(ds["theta"][:], dtype=np.float64))
            wb = w.mean(axis=(-2, -1))
            wp = w - wb[:, None, None]
            thp = th - th.mean(axis=(-2, -1), keepdims=True)
            wth = (wp * thp).mean(axis=(-2, -1))
            k_zi = int(np.argmin(wth[: len(zs) // 2]))
            rows.append((step * a.dt, zs[k_zi], float(wth.min()),
                         float(np.sqrt((wp ** 2).mean(axis=(-2, -1))[2])),
                         wb.copy()))
    print(f"\n  {'t(s)':>8} {'z_i(m)':>8} {'min w-th':>10} {'sig_w(10m)':>11} "
          f"{'mean w @10m':>12} {'mean w @100m':>13} {'|mean w| max':>13}")
    for t, zi, mwt, sw, wb in rows:
        k100 = int(np.argmin(np.abs(zs - 100.0)))
        print(f"  {t:8.0f} {zi:8.1f} {mwt:10.4f} {sw:11.4f} {wb[2]:12.2e} "
              f"{wb[k100]:13.2e} {np.abs(wb[:80]).max():13.2e}")
    zi0, zi1 = rows[0][1], rows[-1][1]
    dt_h = (rows[-1][0] - rows[0][0]) / 3600.0
    print(f"\n  z_i {zi0:.0f} -> {zi1:.0f} m over {dt_h*60:.0f} min "
          f"= {(zi1-zi0)/max(dt_h,1e-9):+.0f} m/h ({100*(zi1-zi0)/zi0:+.1f}% of the window)")
    print(f"  sigma_w at the receptor {rows[0][3]:.4f} -> {rows[-1][3]:.4f} "
          f"({100*(rows[-1][3]-rows[0][3])/rows[0][3]:+.1f}%)")
    wbar = np.mean([r[4] for r in rows], axis=0)
    print(f"  window-mean slab w: max |w| below 400 m = {np.abs(wbar[:70]).max():.3e} m/s")
    print(f"    a uniform release stays uniform only if this is small against sigma_w "
          f"({rows[0][3]:.3f} m/s); ratio = {np.abs(wbar[:70]).max()/max(rows[0][3],1e-9):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
