#!/usr/bin/env python3
"""Standing accuracy-CFL check: resolved w variance ratio between the first two levels.

WHY: FastEddy has no CFL machinery, and its ACCURACY limit sits BELOW its stability
limit (PROJECT_BRIEF.md). Between the two the model exits 0, prints no warning, and silently
replaces resolved near-surface w with grid-scale acoustic noise. The only cheap
detector is the ratio of resolved w variance at k=0 to k=1: physically w variance must
GROW away from the wall (the surface is impermeable), so the ratio must be < 1. When dt
is above the accuracy limit the ratio jumps to ~9.

Emits: OK / FAIL / SKIP.  SKIP when turbulence has not developed yet -- the ratio of two
near-zero variances is meaningless, so an early dump must not be scored.

usage: k0k1_check.py <dump.nc> [<dump.nc> ...]
"""
import sys

import os
import numpy as np
from netCDF4 import Dataset

# Below this, resolved w is still laminar noise and the ratio carries no information.
WW_FLOOR = 1.0e-5   # m2/s2 at k=1
LIMIT = 1.0

def _zpos(path):
    """Level heights, falling back to the first dump of the same run if this one is lean.

    ioLPDMmode (the kegonsa fork) writes the static coordinate geometry into the FIRST
    file a run produces and omits it from every later one, because xPos/yPos/zPos are
    rewritten byte-identically each dump and cost more than the packed prognostic fields
    they accompany. So a mid-window dump legitimately has no zPos, and this check must not
    abort a run over it.
    """
    import glob
    with Dataset(path) as ds:
        if "zPos" in ds.variables:
            return np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))[:, 0, 0]
    fam = os.path.basename(path).rsplit(".", 1)[0]   # the anchor's OWN run, not the dir's
    sibs = sorted((q for q in glob.glob(os.path.join(os.path.dirname(path) or ".",
                                                     "*.[0-9]*"))
                   if os.path.basename(q).rsplit(".", 1)[0] == fam),
                  key=lambda q: int(q.rsplit(".", 1)[1]))
    for q in sibs:
        with Dataset(q) as ds:
            if "zPos" in ds.variables:
                return np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))[:, 0, 0]
    raise KeyError("no zPos in this dump or any sibling of the same run")


def check(path):
    with Dataset(path) as ds:
        w = np.squeeze(np.asarray(ds["w"][:], dtype=np.float64))
    z = _zpos(path)
    wp = w - w.mean(axis=(-2, -1), keepdims=True)
    ww = (wp ** 2).mean(axis=(-2, -1))
    ratio = ww[0] / max(ww[1], 1e-30)
    tag = f"k0/k1={ratio:6.3f}  ww[0]={ww[0]:.3e} ww[1]={ww[1]:.3e} (z={z[0]:.1f},{z[1]:.1f} m)"
    # NON-FINITE FIRST. A NaN passes every `>` and `>=` test, so a gate written only as
    # "fail if ratio >= LIMIT" reports OK on a field that is entirely NaN or inf -- which
    # is exactly what a rho-skipped ioLPDMmode dump looked like (docs/FASTEDDY_TRAPS.md #1).
    # inf is not NaN, and FastEddy's own ****CORRUPTED*** banner tests only for NaN, so
    # this is the only place an inf field is caught at all.
    nbad = int((~np.isfinite(w)).sum())
    if nbad or not np.isfinite(ratio):
        return "FAIL", (f"  FAIL: NON-FINITE w in {path}\n"
                        f"        {nbad:,} of {w.size:,} cells are NaN or inf; "
                        f"ratio={ratio}\n"
                        f"        Note FastEddy exits 0 and prints no CORRUPTED banner for\n"
                        f"        inf. See docs/FASTEDDY_TRAPS.md #1.")
    if ww[1] < WW_FLOOR:
        return "SKIP", f"  k0/k1 SKIP (turbulence undeveloped, ww[1] < {WW_FLOOR:g}): {tag}"
    if ratio >= LIMIT:
        return "FAIL", (f"  FAIL: near-surface w variance ratio {ratio:.2f} >= {LIMIT} in {path}\n"
                        f"        {tag}\n"
                        f"        dt is above the ACCURACY CFL limit -- resolved near-surface w is\n"
                        f"        grid-scale acoustic noise. Lower dt; see PROJECT_BRIEF.md.")
    return "OK", f"  k0/k1 OK ({ratio:.3f} < {LIMIT}): {tag}"


if __name__ == "__main__":
    rc = 0
    for p in sys.argv[1:]:
        status, msg = check(p)
        print(msg)
        if status == "FAIL":
            rc = 1
    sys.exit(rc)
