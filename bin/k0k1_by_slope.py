#!/usr/bin/env python3
"""The accuracy-CFL check, conditioned on terrain slope.

WHY THE STANDING CHECK IS NOT ENOUGH OVER TERRAIN. docker/k0k1_check.py averages the
resolved w variance over the WHOLE plane at k=0 and k=1. That is the right test on flat
ground, where every column runs at the same effective CFL and grid-scale acoustic noise
appears everywhere at once. Over terrain it is not: the amplification is LOCAL,

    CFL_eff ~ CFL_3d sqrt(1 + (slope dx/dz)^2)

so a few steep columns can be ringing while the domain mean stays clean. On this domain
only 1.7% of cells exceed slope 0.14, and they cannot move a 14,884-cell average.

So condition on the local slope. Grid-scale acoustic noise is a ratio near 9 in the
affected bins; a monotone rise from ~0.4 to ~0.7 with slope is not that -- it is resolved
vertical motion over topography, which is real.

usage: k0k1_by_slope.py <dump.nc> [--grid data/grid16] [--box 3]

NOTE WHAT THIS CANNOT SEE. Conditioning k0/k1 on slope makes it terrain-aware; it does not
make it a physics check. The ratio is between two levels, so it stays healthy when both go
quiet together -- it read 0.442 on a fully collapsed stable boundary layer. Run
docker/turb_alive.py alongside it, always. bin/run_pass5.sh (retired 2026-09-04; see docs/history/pass-5.md) does.
"""
import argparse
import os
import sys

import numpy as np
from netCDF4 import Dataset

LIMIT = 2.0          # per-bin median; the flat check uses 1.0 on the domain mean
EDGES = [0.0, 0.02, 0.05, 0.08, 0.11, 0.14, 0.20, 1.0]


def boxvar(a, n):
    """Variance in a (2n+1)^2 neighbourhood, periodic. A single cell is too noisy."""
    p = np.pad(a, n, mode="wrap")
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(p, (2 * n + 1, 2 * n + 1))
    return win.var(axis=(-2, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--grid", default="data/grid16")
    ap.add_argument("--box", type=int, default=3)
    a = ap.parse_args()

    topo = np.load(os.path.join(a.grid, "topo.npy"))
    meta = np.load(os.path.join(a.grid, "meta.npy"), allow_pickle=True).item()
    dx = float(meta["dx"])
    gy, gx = np.gradient(topo, dx)
    slope = np.hypot(gx, gy)

    with Dataset(a.dump) as ds:
        w = np.squeeze(np.asarray(ds["w"][:], dtype=np.float64))
    # inf is not NaN and a NaN passes every > test, so check finiteness FIRST
    if not np.isfinite(w).all():
        print(f"  FAIL: w is not finite in {a.dump}")
        return 1
    r = boxvar(w[0], a.box) / np.maximum(boxvar(w[1], a.box), 1e-30)

    print(f"  {a.dump}")
    print(f"  domain-mean k0/k1 (what the standing check reports): "
          f"{w[0].var() / max(w[1].var(), 1e-30):.3f}")
    print(f"\n  {'slope bin':>16}{'cells':>8}{'median':>9}{'p95':>8}{'max':>8}   verdict")
    bad = 0
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (slope >= lo) & (slope < hi)
        if m.sum() < 20:
            continue
        med = float(np.median(r[m]))
        ok = med < LIMIT
        bad += (not ok)
        print(f"  {lo:6.2f}-{hi:<8.2f}{m.sum():8d}{med:9.2f}"
              f"{np.percentile(r[m], 95):8.2f}{r[m].max():8.2f}   "
              f"{'ok' if ok else 'RINGING'}")
    print(f"\n  {'PASS -- no slope bin shows the acoustic signature' if not bad else 'FAIL -- steep columns are ringing; lower the terrain dt'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
