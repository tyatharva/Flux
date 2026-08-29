#!/usr/bin/env python3
"""Zero htFlux (and optionally other IO-registered surface fields) inside a restart file.

WHY THIS EXISTS, and why it is not optional. htFlux is an IO-REGISTERED FIELD, so the
restart read at FEMAIN/FastEddy.c:221 -- which runs AFTER hydro_coreInit() at :157 --
overwrites whatever the .in asked for with whatever the file holds (FASTEDDY_TRAPS.md 17).
The Steinfeld spin-up accelerator runs a burn-in at +0.05 K m/s and then continues at zero;
restarting from the burn-in dump without this would silently continue the burn-in flux for
the whole neutral spin-up, and every diagnostic would still print a number. It cost a whole
segment of a stable seed once, running at zero flux while its .in asked for -0.012.

usage: zero_htflux.py <restart.nc> [--value 0.0]
"""
import argparse, sys
import numpy as np
from netCDF4 import Dataset

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--value", type=float, default=0.0)
a = ap.parse_args()
with Dataset(a.path, "a") as ds:
    if "htFlux" not in ds.variables:
        print(f"FATAL: {a.path} carries no htFlux", file=sys.stderr); raise SystemExit(1)
    before = float(np.asarray(ds["htFlux"][:]).mean())
    ds["htFlux"][:] = a.value
# ASSERT ON THE ARTIFACT: re-open and read it back, because the point of this script is
# that a value written into a file is not the same thing as a value asked for.
with Dataset(a.path) as ds:
    after = float(np.asarray(ds["htFlux"][:]).mean())
if abs(after - a.value) > 1e-9:
    print(f"FATAL: htFlux reads back {after:+.9f}, wrote {a.value:+.9f}", file=sys.stderr)
    raise SystemExit(1)
print(f"  htFlux in {a.path}: {before:+.6f} -> {after:+.6f} (verified by re-read)")
