#!/usr/bin/env python3
"""Diagnose the near-surface w artifact found during Stage 2 spinup.

Reports, per FastEddy dump:
  * per-field variance in the lowest levels and the k0/k1 ratio, which isolates
    WHICH field is affected (it is w only; u, v, theta are clean);
  * lag-1 spatial autocorrelation of w per level, which distinguishes resolved
    turbulence (corr near +1) from grid-scale noise (corr near 0) and from a
    2-cell checkerboard (corr strongly negative).

Background: resolved w'w' at the first interior level runs ~8-9x the second and
decays upward. u, v, theta show no such step, and FastEddy's own fricVel is
correct, so the surface momentum flux is fine -- the defect is confined to the
resolved w field in roughly the lowest three levels (z < 35 m).

Ruled out by controlled single-variable tests: vertical stretching, thread-block
shape (4x4x16 vs 1x4x64), roughness length (0.03 vs 0.10), spanwise domain width
(1.46 vs 2.90 km). The k0/k1 ratio came out at 8.94 in every one.

usage: diag_near_surface.py <dump.nc> [<dump.nc> ...]
"""
import sys

import numpy as np
from netCDF4 import Dataset

FIELDS = ("u", "v", "w", "theta")


def diagnose(path):
    print(f"\n=== {path} ===")
    with Dataset(path) as ds:
        z = np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))[:, 0, 0]
        var, fld = {}, {}
        for v in FIELDS:
            a = np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
            fld[v] = a
            ap = a - a.mean(axis=(-2, -1), keepdims=True)
            var[v] = (ap ** 2).mean(axis=(-2, -1))
        ustar = float(np.squeeze(np.asarray(ds["fricVel"][:], dtype=np.float64)).mean())

    print(f"  u* (FastEddy fricVel) = {ustar:.4f} m/s")
    print(f"  {'k':>3} {'z(m)':>7} " + "".join(f"{v + '_var':>12}" for v in FIELDS))
    for k in range(8):
        print(f"  {k:>3} {z[k]:7.1f} " + "".join(f"{var[v][k]:12.5g}" for v in FIELDS))
    print("  k0/k1 ratio: " + "  ".join(
        f"{v}={var[v][0] / max(var[v][1], 1e-30):.2f}" for v in FIELDS)
        + "   <- w is the outlier; others ~1")

    print(f"\n  lag-1 spatial autocorrelation of w"
          f"  (+1 = resolved turbulence, 0 = grid noise, <0 = checkerboard)")
    w = fld["w"]
    for k in (0, 1, 2, 4, 8, 16):
        if k >= w.shape[0]:
            continue
        f = w[k] - w[k].mean()
        denom = max(f.var(), 1e-30)
        rx = (f[:, :-1] * f[:, 1:]).mean() / denom
        ry = (f[:-1, :] * f[1:, :]).mean() / denom
        print(f"   k={k:<3} z={z[k]:7.1f} m  std={w[k].std():8.4f}  "
              f"corr_x={rx:+.3f}  corr_y={ry:+.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        diagnose(p)
