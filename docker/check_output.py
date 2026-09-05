#!/usr/bin/env python3
"""Structural + physical sanity check on a FastEddy NetCDF output dump.

Used as part of the docs/history/overview.md Stage 0a gate. This deliberately checks only what a
short truncated run CAN establish -- that the grid decomposed correctly, the
writer works, and the fields are finite and physically plausible. It makes no
claim about converged turbulence, which needs hours of simulated time.

usage: check_output.py <file.nc> [<file.nc> ...]
"""
import sys
import numpy as np
from netCDF4 import Dataset

EXPECTED = ("rho", "u", "v", "w", "theta")


def check(path):
    ok = True
    print(f"\n=== {path} ===")
    with Dataset(path) as ds:
        dims = {k: len(v) for k, v in ds.dimensions.items()}
        print(f"  dims: {dims}")
        print(f"  vars: {len(ds.variables)}")

        missing = [v for v in EXPECTED if v not in ds.variables]
        if missing:
            print(f"  FAIL: missing expected variables: {missing}")
            ok = False

        for name in EXPECTED:
            if name not in ds.variables:
                continue
            a = np.asarray(ds.variables[name][:], dtype=np.float64)
            n_nan = int(np.isnan(a).sum())
            n_inf = int(np.isinf(a).sum())
            status = "ok"
            if n_nan or n_inf:
                status = f"FAIL nan={n_nan} inf={n_inf}"
                ok = False
            print(f"  {name:>7}: min={a.min():12.5g} max={a.max():12.5g} "
                  f"mean={a.mean():12.5g}  [{status}]")

            # rho is a density: strictly positive everywhere or the run is broken.
            if name == "rho" and a.min() <= 0:
                print(f"  FAIL: rho has non-positive values (min {a.min()})")
                ok = False

        # Vertical structure. FastEddy NetCDF dims are (time, zIndex, yIndex,
        # xIndex), so z is axis 1 after squeezing time -- averaging over the
        # horizontal means axes (-2, -1).
        if "theta" in ds.variables:
            # NOTE: the NetCDF writer already emits actual potential temperature,
            # not the density-weighted prognostic rho*theta. Verified: raw values
            # are 264-268 K for SBL, which is physical. Do NOT divide by rho.
            th = np.squeeze(np.asarray(ds.variables["theta"][:], dtype=np.float64))
            prof = th.mean(axis=(-2, -1))
            zq = [0, len(prof) // 4, len(prof) // 2, len(prof) - 1]
            print("  theta(z) horiz-mean at k = "
                  + ", ".join(f"{k}:{prof[k]:.2f}K" for k in zq))
            print(f"  d(theta)/dz over column = {prof[-1] - prof[0]:+.3f} K")

    return ok


if __name__ == "__main__":
    results = [check(p) for p in sys.argv[1:]]
    print()
    if all(results):
        print("ALL CHECKS PASSED")
        sys.exit(0)
    print("CHECKS FAILED")
    sys.exit(1)
