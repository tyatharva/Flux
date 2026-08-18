#!/usr/bin/env python3
"""Stage 6 preprocessing: wind-aligned terrain + roughness map, and a consistent restart.

Two products:

1. `topo.bin` -- FastEddy's `topoFile`. Format read at SRC/GRID/grid.c:378-420: two
   int32 extents (Nx, Ny) followed by Nx*Ny float32 in [j][i] order (y slowest is what
   the code's `ji = j*Nx + i` indexing implies). Elevation in metres; we subtract the
   domain mean so the terrain is a perturbation about zero rather than ~265 m ASL, which
   would otherwise consume 10% of the domain depth.

2. A modified restart netCDF. This is necessary, not cosmetic. FastEddy's restart read
   (FEMAIN/FastEddy.c:221) runs AFTER gridInit and walks the whole registered variable
   list -- which includes `zPos`, `topoPos` and `z0m`. So restarting a FLAT spin-up with a
   topoFile set leaves the solver with correct terrain-following metrics (computed in
   gridInit) but overwrites the diagnostic `zPos`/`topoPos` in every subsequent dump with
   the flat values from the spin-up file. The LPDM reads `zPos`, so it would place every
   particle at the wrong height. Writing the terrain into the restart file itself makes
   the read a no-op and keeps grid metrics, output coordinates and the LPDM consistent.

   The same mechanism is what makes the solar-array bulk patch reachable at all: `z0m` is
   initialised from the scalar `surflayer_z0` and v5.0.1 exposes no spatially varying
   roughness input, but `z0m` IS in the registered list, so a restart file carries it.
   No FastEddy source change is required.

Rotation happens entirely here (CLAUDE.md): the DEM is resampled into a frame whose +x is
the direction the wind blows TOWARD, so the LES needs no code change and its x axis is the
mean-wind axis by construction.
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
import sys

import numpy as np
import rasterio
from rasterio.transform import rowcol

# ---------------------------------------------------------------- site assumptions
# SURROGATE SITE COORDINATE -- see data/README.md.
# The surveyed tower position is not in this repository and could not be established from
# public sources (the published descriptions give "3725 Schneider Dr, Stoughton WI" and
# "west of Lake Kegonsa" but no coordinates). The first estimate, -89.2450/42.9686, lands
# 6 m from open water: the DEM there reads 256.64 m, which is Lake Kegonsa's surface.
# What is used instead is the nearest tower position whose 4380 x 1500 m westerly domain
# contains no water at all -- an explicit, reproducible rule, not a guess dressed up as a
# location. It sits 810 m from the shore with 38 m of relief across the domain, matching
# CLAUDE.md's "~30 m of elevation change across the area".
# REPLACE THIS with the surveyed coordinate before any result is treated as site-specific.
TOWER_LON, TOWER_LAT = -89.2539, 42.9419
ARRAY_ALONG_WIND = 100.0     # m, extent along the wind axis
ARRAY_CROSS_WIND = 400.0     # m, extent across the wind axis
ARRAY_OFFSET_UPWIND = 200.0  # m, centre distance upwind of the tower
Z0_GRASS = 0.03
Z0_ARRAY = 0.20              # bulk patch: panels raise z0 by roughly an order of magnitude
D_ARRAY = 1.2                # displacement height (m); folded into the terrain, see below


def taper_weights(n, pad):
    """1 in the interior, smoothly 0 at both ends over `pad` cells (raised cosine)."""
    w = np.ones(n)
    if pad <= 0:
        return w
    r = 0.5 * (1.0 - np.cos(np.pi * (np.arange(pad) + 0.5) / pad))
    w[:pad] = r
    w[-pad:] = r[::-1]
    return w


def sample_rotated(dem_path, tower_lon, tower_lat, wind_from_deg, nx, ny, dx, dy,
                   i_tower, j_tower):
    """Bilinear DEM sample on the wind-aligned LES grid."""
    from pyproj import Transformer
    with rasterio.open(dem_path) as src:
        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        tx, ty = tr.transform(tower_lon, tower_lat)
        band = src.read(1).astype(np.float64)
        inv = ~src.transform

        # +x of the LES is the direction the wind blows TOWARD
        bearing = np.radians((wind_from_deg + 180.0) % 360.0)
        ex = np.array([np.sin(bearing), np.cos(bearing)])   # east, north components
        ey = np.array([-ex[1], ex[0]])                      # right-handed with z up

        i = np.arange(nx); j = np.arange(ny)
        s = (i - i_tower) * dx
        c = (j - j_tower) * dy
        S, Cc = np.meshgrid(s, c)                            # (ny, nx)
        X = tx + S * ex[0] + Cc * ey[0]
        Y = ty + S * ex[1] + Cc * ey[1]

        col, row = inv * (X, Y)
        r0 = np.floor(row).astype(int); c0 = np.floor(col).astype(int)
        fr = row - r0; fc = col - c0
        r0 = np.clip(r0, 0, band.shape[0] - 2); c0 = np.clip(c0, 0, band.shape[1] - 2)
        z = ((1 - fr) * (1 - fc) * band[r0, c0] + (1 - fr) * fc * band[r0, c0 + 1]
             + fr * (1 - fc) * band[r0 + 1, c0] + fr * fc * band[r0 + 1, c0 + 1])
        return z, (tx, ty), (ex, ey)


def write_topofile(path, topo):
    """topo is (ny, nx). See SRC/GRID/grid.c:378-420 for the format."""
    ny, nx = topo.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", nx, ny))
        np.ascontiguousarray(topo, dtype=np.float32).tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default="data/dem/kegonsa_30m_wtm.tif")
    ap.add_argument("--restart-in", required=True, help="flat spin-up dump to base on")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--wind-from", type=float, default=270.0)
    ap.add_argument("--nx", type=int, default=146)
    ap.add_argument("--ny", type=int, default=50)
    ap.add_argument("--dx", type=float, default=30.0)
    ap.add_argument("--itower", type=int, default=109)
    ap.add_argument("--jtower", type=int, default=25)
    ap.add_argument("--taper-x", type=int, default=12, help="cells (12 x 30 m = 360 m)")
    ap.add_argument("--taper-y", type=int, default=8)
    ap.add_argument("--no-array", action="store_true")
    ap.add_argument("--flat", action="store_true", help="roughness patch only, no terrain")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    z, (tx, ty), (ex, ey) = sample_rotated(a.dem, TOWER_LON, TOWER_LAT, a.wind_from,
                                           a.nx, a.ny, a.dx, a.dx, a.itower, a.jtower)
    print(f"  tower EPSG:3071 ({tx:.1f}, {ty:.1f});  wind from {a.wind_from:.0f} deg "
          f"-> LES +x bearing {(a.wind_from+180)%360:.0f} deg")
    print(f"  raw DEM on grid: min {z.min():.1f}  max {z.max():.1f}  "
          f"range {z.max()-z.min():.1f} m  mean {z.mean():.1f} m")

    topo = z - z.mean()
    if a.flat:
        topo[:] = 0.0
    # Taper to the domain-mean plane at both periodic seams, else the wrap is a cliff.
    wx = taper_weights(a.nx, a.taper_x)
    wy = taper_weights(a.ny, a.taper_y)
    topo = topo * (wx[None, :] * wy[:, None])
    print(f"  after mean removal + seam taper: min {topo.min():.2f}  max {topo.max():.2f}"
          f"  range {np.ptp(topo):.2f} m")
    print(f"  seam discontinuity  x: {np.abs(topo[:,0]-topo[:,-1]).max():.3e} m"
          f"   y: {np.abs(topo[0,:]-topo[-1,:]).max():.3e} m")

    # --- roughness map -------------------------------------------------------------
    xg = (np.arange(a.nx) - a.itower) * a.dx
    yg = (np.arange(a.ny) - a.jtower) * a.dx
    z0 = np.full((a.ny, a.nx), Z0_GRASS)
    if not a.no_array:
        inx = np.abs(xg + ARRAY_OFFSET_UPWIND) <= 0.5 * ARRAY_ALONG_WIND
        iny = np.abs(yg) <= 0.5 * ARRAY_CROSS_WIND
        patch = iny[:, None] & inx[None, :]
        z0[patch] = Z0_ARRAY
        # The displacement height is not a FastEddy input. Represent it the only way the
        # model can feel it: raise the effective surface by d over the array. This is a
        # deliberate approximation and is the reason the array shows up in `topoPos`.
        topo[patch] += D_ARRAY
        print(f"  solar array patch: {patch.sum()} cells "
              f"({patch.sum()*a.dx**2/1e4:.2f} ha), z0 {Z0_GRASS} -> {Z0_ARRAY} m, "
              f"d = {D_ARRAY} m, centred {ARRAY_OFFSET_UPWIND:.0f} m upwind")
    z0 = z0 * (wx[None, :] * wy[:, None]) + Z0_GRASS * (1 - wx[None, :] * wy[:, None])

    write_topofile(os.path.join(a.outdir, "topo.bin"), topo)
    np.save(os.path.join(a.outdir, "topo.npy"), topo)
    np.save(os.path.join(a.outdir, "z0m.npy"), z0)
    print(f"  wrote {a.outdir}/topo.bin  ({a.nx}x{a.ny} float32 + 2 int32 header)")

    # --- restart file with terrain, terrain-following zPos, and the roughness map ----
    from netCDF4 import Dataset
    dst = os.path.join(a.outdir, os.path.basename(a.restart_in))
    shutil.copy2(a.restart_in, dst)
    with Dataset(dst, "a") as ds:
        zp = np.asarray(ds["zPos"][:])
        shp = zp.shape                      # (t, z, y, x) or (z, y, x)
        zcol = np.squeeze(zp)[:, 0, 0].astype(np.float64)
        zC = float(zcol[-1])                # flat spin-up: zCeiling
        Fk = zcol
        new = (Fk[:, None, None] * (zC - topo[None, :, :]) / zC + topo[None, :, :])
        ds["zPos"][:] = new.reshape(shp).astype(np.float32)
        tp = np.asarray(ds["topoPos"][:])
        ds["topoPos"][:] = topo.reshape(tp.shape).astype(np.float32)
        ds["z0m"][:] = z0.reshape(tp.shape).astype(np.float32)
    print(f"  wrote {dst}: zPos made terrain-following, topoPos and z0m replaced")
    print("  NOTE lat/lon are left at the flat spin-up's uniform values -- they drive "
          "Coriolis only, and\n       the rotation is already baked into the resampling, "
          "so a single-direction run\n       needs no rotated lat(y,x).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
