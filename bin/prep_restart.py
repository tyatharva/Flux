#!/usr/bin/env python3
"""Build a production restart: rotate the spun-up FLOW, inject the STATIC surface.

The domain is fixed to the map, so wind direction is changed by turning the flow rather
than the geography. A square, doubly periodic, FLAT, UNIFORM spin-up with dx = dy is
exactly equivariant under a 90-degree rotation of the grid, so one spun-up state re-indexes
into four directions at no extra GPU cost. The rotation is applied to every 3-D field and,
for the horizontal wind, to the vector components as well; the geostrophic forcing must be
rotated with it (see --print-ug).

    rotation by m * 90 deg counter-clockwise:  (x, y) -> (-y, x) per turn
    index map:                                 new[j, i] = old[(-i) % N, j]
    vector components:                         (u, v) -> (-v, u)

Surface fields are NOT rotated. That is the entire point: terrain, roughness and the solar
array are bit-identical for every direction, so a directional difference in the footprint
is flow and cannot be a resampling artifact.

The surface is injected by overwriting the restart file, which is the only way to give
FastEddy v5.0.1 a spatially varying surface: hydro_coreInit() runs BEFORE
ioReadNetCDFinFileSingleTime(), and the read walks the whole registered variable list --
which includes zPos, topoPos, z0m, z0t and htFlux. No source change is needed.

usage: prep_restart.py <spinup_dump> <outfile> [--rot 0|1|2|3] [--grid data/grid]
                       [--flat]   (rotate only, keep the flat uniform surface)
"""
import argparse
import os
import shutil
import sys

import numpy as np
from netCDF4 import Dataset


def rot_index(n, m):
    """Index arrays (jj, ii) such that new[j,i] = old[jj[j,i], ii[j,i]] for m 90-deg turns."""
    J, I = np.mgrid[0:n, 0:n]
    for _ in range(m % 4):
        J, I = (-I) % n, J.copy()
    return J, I


def rotate_stack(a, m):
    """Rotate the trailing (y, x) plane of a (..., ny, nx) array by m * 90 degrees."""
    if m % 4 == 0:
        return a
    n = a.shape[-1]
    if a.shape[-2] != n:
        raise ValueError("90-degree re-indexing needs a square grid")
    J, I = rot_index(n, m)
    return a[..., J, I]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rot", type=int, default=0, choices=(0, 1, 2, 3))
    ap.add_argument("--grid", default="data/grid")
    ap.add_argument("--flat", action="store_true",
                    help="rotate the flow but leave the surface flat and uniform")
    ap.add_argument("--ug", type=float, default=10.0,
                    help="geostrophic magnitude of the spin-up (for the rotated report)")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    shutil.copy2(a.src, a.dst)
    m = a.rot % 4

    with Dataset(a.dst, "a") as ds:
        nx = len(ds.dimensions["xIndex"]); ny = len(ds.dimensions["yIndex"])
        if m and nx != ny:
            raise SystemExit(f"rotation needs nx == ny, got {nx} x {ny}")

        if m:
            # Every PROGNOSTIC and diagnostic field turns with the flow. xPos/yPos/zPos and
            # topoPos are geometry and are rewritten below (or left alone for --flat).
            flow3d = ["rho", "u", "v", "w", "theta", "pressure", "TKE_0"]
            flow2d = ["tskin", "fricVel", "htFlux", "invOblen"]
            cache = {v: np.asarray(ds[v][:]) for v in flow3d + flow2d if v in ds.variables}
            for v, arr in cache.items():
                ds[v][:] = rotate_stack(arr, m)
            # Horizontal wind is a VECTOR: rotating the grid is not enough, the components
            # rotate too. (u, v) -> (-v, u) per counter-clockwise turn.
            u = np.asarray(ds["u"][:]).copy()
            v = np.asarray(ds["v"][:]).copy()
            for _ in range(m):
                u, v = -v, u.copy()
            ds["u"][:] = u
            ds["v"][:] = v
            print(f"  rotated the flow by {90*m} deg CCW "
                  f"({len(cache)} fields re-indexed, u/v components rotated)")

        if a.flat:
            print("  --flat: surface left as the spin-up wrote it")
        else:
            topo = np.load(os.path.join(a.grid, "topo.npy"))
            z0 = np.load(os.path.join(a.grid, "z0m.npy"))
            wth = np.load(os.path.join(a.grid, "htFlux.npy"))
            if topo.shape != (ny, nx):
                raise SystemExit(f"surface is {topo.shape}, restart is {(ny, nx)}")
            # Terrain-following zPos, rebuilt from the FLAT column the spin-up carries.
            zp = np.asarray(ds["zPos"][:]); shp = zp.shape
            zcol = np.squeeze(zp)[:, 0, 0].astype(np.float64)
            zC = float(zcol[-1])
            new = zcol[:, None, None] * (zC - topo[None]) / zC + topo[None]
            ds["zPos"][:] = new.reshape(shp).astype(np.float32)
            t2 = np.asarray(ds["topoPos"][:])
            ds["topoPos"][:] = topo.reshape(t2.shape).astype(np.float32)
            ds["z0m"][:] = z0.reshape(t2.shape).astype(np.float32)
            if "z0t" in ds.variables:
                ds["z0t"][:] = np.minimum(z0, 0.01).reshape(t2.shape).astype(np.float32)
            if "htFlux" in ds.variables:
                ds["htFlux"][:] = wth.reshape(t2.shape).astype(np.float32)
            print(f"  injected the static surface: terrain {topo.min():.1f}..{topo.max():.1f} m, "
                  f"z0 {z0.min():.0e}..{z0.max():.2f} m  (NOT rotated -- it is the geography)")

    # ---- READ IT BACK. PROJECT_BRIEF.md's standing rule, applied where it is load-bearing ----
    #
    # The surface is injected by OVERWRITING the restart file, because that is the only way
    # to give FastEddy v5.0.1 a spatially varying surface -- and a write that silently does
    # not land produces a case that runs to completion on the WRONG surface and says
    # nothing. That is not hypothetical here: for a NEUTRAL corpus case the array's entire
    # signal is z0m (0.25 against cropland's 0.10 on grid16_raised); there is no thermal
    # contrast at all, so a failed z0m injection is a case with no array in it, reported as
    # a clean run. Gate B5 measured this once, on one file, in a separate script; the
    # corpus runs 1370 of them and the check belongs in the writer.
    if not a.flat:
        with Dataset(a.dst) as ds:
            g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
            got_t, got_z0, got_zp = g("topoPos"), g("z0m"), g("zPos")
            got_h = g("htFlux") if "htFlux" in ds.variables else None
        checks = [("topoPos", got_t, topo),
                  ("z0m", got_z0, z0),
                  ("zPos", got_zp,
                   (np.squeeze(new).astype(np.float32).astype(np.float64)))]
        if got_h is not None:
            checks.append(("htFlux", got_h, wth))
        worst = []
        for nm, got, want in checks:
            want = np.asarray(want, dtype=np.float64).reshape(got.shape)
            if not np.isfinite(got).all():
                raise SystemExit(f"FATAL: {nm} in {a.dst} is not finite after injection")
            sc = max(float(np.abs(want).max()), 1e-30)
            rel = float(np.abs(got - want).max()) / sc
            worst.append((nm, rel))
            # float32 storage, so ~1e-7 relative is the floor; anything above 1e-5 means
            # the write did not land, not that it rounded.
            if rel > 1e-5:
                raise SystemExit(
                    f"FATAL: {nm} read back from {a.dst} differs from what was injected by "
                    f"{rel:.2e} relative -- the write did not land. The run would have used "
                    f"the spin-up's surface and reported nothing.")
        print("  read back: " + ", ".join(f"{n} {r:.1e}" for n, r in worst)
              + "   (float32 storage, so ~1e-7 is the floor)")
        # AND SAY WHAT THE ARRAY ITSELF GOT, not what the whole roughness map looks like.
        # 29% of this domain is rougher than 0.2 m -- tree and built cells -- so a
        # threshold count says nothing about the array. Read the array mask.
        amask = os.path.join(a.grid, "array.npy")
        if os.path.exists(amask):
            am = np.load(amask).astype(bool).reshape(got_z0.shape)
            if am.any():
                z0a = float(np.median(got_z0[am]))
                z0o = float(np.median(got_z0[~am]))
                print(f"    the array's {int(am.sum())} cells read back at z0 = "
                      f"{z0a:.3f} m against a domain median of {z0o:.3f} m "
                      f"({z0a/max(z0o,1e-12):.2f}x)"
                      + ("  -- for a NEUTRAL case this ratio IS the array's entire signal"
                         if got_h is not None and float(np.abs(got_h).max()) < 1e-6
                         else ""))
                if z0a <= z0o * 1.001:
                    print("    WARNING: the array is aerodynamically identical to the "
                          "surface around it; a neutral case built on this grid carries "
                          "NO array signal at all (bin/prep_surface.py --raise-topo)")

    # The forcing must turn with the flow, or the state and the forcing disagree and the
    # run spends its adjustment budget turning back.
    ug, vg = a.ug, 0.0
    for _ in range(m):
        ug, vg = -vg, ug
    frm = (np.degrees(np.arctan2(-ug, -vg)) % 360)
    # ADVISORY, NOT AN INSTRUCTION -- and it used to read like one. This is where the
    # ROTATION puts a --ug spin-up's forcing, which is what the retired per-bin campaign
    # needed: it rotated one flat state into four directions and had to be told the matching
    # U_g/V_g. A sounding-forced corpus case already carries its own forcing in the .in that
    # bin/sounding_to_forcing.py wrote (U_g = -2.788 on the first end-to-end case, against
    # the -10.0000 this line reports), and nothing acts on what is printed here.
    print(f"  the rotation puts a {a.ug:.1f} m/s spin-up's forcing at "
          f"U_g = {ug:.4f}, V_g = {vg:.4f}  (FROM {frm:.0f} deg)")
    print(f"    ADVISORY: a corpus case keeps the forcing already in its own .in. Only a "
          f"re-indexed spin-up needs these; the surface wind is then backed from that "
          f"direction by the Ekman angle (measured ~12-19 deg here).")
    print(f"  wrote {a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
