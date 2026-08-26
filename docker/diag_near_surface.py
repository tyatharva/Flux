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

Ruled out by source inspection:
  * Discrete impermeability IS correctly enforced. cuda_advectionDevice.cu:106-109
    explicitly zeroes the ground face velocity ("Ensure ground and ceiling face
    vertical velocity component is set to 0"), so there is no spurious advective
    mass flux through the surface.
  * Explicit filtering is IDENTICAL to the NBL tutorial (filterSelector = 1,
    filter_6thdiff_vert = 1, filter_6thdiff_vert_coeff = 0.03), inherited from it.

Still notable in the bottom BC (cuda_BCsDevice.cu:284-291): below-ground halo
cells are set to w = 0 for the vertical momentum component, whereas u and v get
a zero-gradient copy of the first interior cell. So the cell-centred w at the
first interior level is unconstrained while the face flux is pinned to zero, and
in a compressible solver there is no elliptic projection to remove grid-scale
divergent noise -- only acoustic propagation and the explicit filter. That is
consistent with the observed signature but is NOT yet demonstrated to be the
cause, since the same code path serves the NCAR tutorials.

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
        + "\n  (healthy: w ratio < 1, others ~1. A w ratio near 9 means dt exceeds the\n"
        "   ACCURACY CFL limit -- ~1.51 at 122^3 @ 16 m, ~1.64 at the retired 24-30 m\n"
        "   grids. THE BOUNDARY IS NOT THE SAME NUMBER AT EVERY GRID: it moves with grid\n"
        "   ANISOTROPY, so re-measure it whenever dx/dz_sfc changes. See PROJECT_BRIEF.md.\n"
        "   Not a stability failure: the run will still exit 0 with no CORRUPTED report.\n"
        "\n"
        "   AND A HEALTHY RATIO HERE IS NOT A HEALTHY BOUNDARY LAYER. This is a dt check.\n"
        "   Two collapsed stable runs scored 0.442 and 0.72 on it while their turbulence\n"
        "   was gone. Run docker/turb_alive.py for the physics question.)")

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
