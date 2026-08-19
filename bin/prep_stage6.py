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

Rotation happens entirely here (PROJECT_BRIEF.md): the DEM is resampled into a frame whose +x is
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

# ---------------------------------------------------------------- site definition
# Surveyed tower coordinate.
TOWER_LON, TOWER_LAT = -89.292362, 42.957160

# The solar array is a GEOGRAPHIC object, so it is defined by an offset from the tower in
# map coordinates and rotated with everything else. Defining it relative to the wind (as an
# "upwind distance") silently moves the array when the wind direction changes, which is
# exactly wrong for a multi-direction corpus: for a westerly the array is upwind, for an
# easterly it must be downwind and out of the footprint.
# PROJECT_BRIEF.md: array footprint ~100 m x 400 m, at 270 deg from the tower.
ARRAY_BEARING = 270.0        # deg, direction from the tower to the array centre
ARRAY_DISTANCE = 200.0       # m
ARRAY_EW = 100.0             # m, east-west extent
ARRAY_NS = 400.0             # m, north-south extent

# Land-cover classes. z0 in metres.
Z0_WATER = 1.0e-4            # open water, aerodynamically smooth
Z0_GRASS = 0.03              # site grass
Z0_ARRAY = 0.20              # bulk solar-array patch
D_ARRAY = 1.2                # displacement height (m) for the array
WTH_WATER = 0.0              # kinematic sensible heat flux over water (neutral case)
WTH_LAND = 0.0               # ... and over land; both zero is what "neutral" means

# Water classification, from the sub-cell elevation spread measured by docker/prep_dem30.py.
# The histogram of that spread is strongly bimodal -- 12,214 cells below 0.01 m, a gap, then
# land from 0.02 m upward -- so this threshold sits in the gap rather than being tuned.
WATER_STD_MAX = 0.02         # m
WATER_LEVEL_TOL = 1.0        # m about the modal water elevation


def taper_weights(n, pad):
    """1 in the interior, smoothly 0 at both ends over `pad` cells (raised cosine)."""
    w = np.ones(n)
    if pad <= 0:
        return w
    r = 0.5 * (1.0 - np.cos(np.pi * (np.arange(pad) + 0.5) / pad))
    w[:pad] = r
    w[-pad:] = r[::-1]
    return w


def rotated_map_coords(tower_lon, tower_lat, crs, wind_from_deg, nx, ny, dx, dy,
                       i_tower, j_tower):
    """Map coordinates (EPSG of `crs`) of every LES cell centre, wind-aligned."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    tx, ty = tr.transform(tower_lon, tower_lat)
    bearing = np.radians((wind_from_deg + 180.0) % 360.0)   # LES +x = where wind blows TO
    ex = np.array([np.sin(bearing), np.cos(bearing)])
    ey = np.array([-ex[1], ex[0]])
    s = (np.arange(nx) - i_tower) * dx
    c = (np.arange(ny) - j_tower) * dy
    S, C = np.meshgrid(s, c)
    X = tx + S * ex[0] + C * ey[0]
    Y = ty + S * ex[1] + C * ey[1]
    return X, Y, (tx, ty), (ex, ey)


def bilinear(src_ds, band, X, Y):
    inv = ~src_ds.transform
    col, row = inv * (X, Y)
    r0 = np.clip(np.floor(row).astype(int), 0, band.shape[0] - 2)
    c0 = np.clip(np.floor(col).astype(int), 0, band.shape[1] - 2)
    fr, fc = row - r0, col - c0
    return ((1 - fr) * (1 - fc) * band[r0, c0] + (1 - fr) * fc * band[r0, c0 + 1]
            + fr * (1 - fc) * band[r0 + 1, c0] + fr * fc * band[r0 + 1, c0 + 1])


def write_topofile(path, topo):
    """topo is (ny, nx). See SRC/GRID/grid.c:378-420 for the format."""
    ny, nx = topo.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", nx, ny))
        np.ascontiguousarray(topo, dtype=np.float32).tofile(f)


def main():
    import rasterio
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default="data/dem/kegonsa_30m_wtm.tif")
    ap.add_argument("--std", default="data/dem/kegonsa_30m_std.tif")
    ap.add_argument("--restart-in", required=True, help="flat spin-up dump to base on")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--wind-from", type=float, default=270.0)
    ap.add_argument("--nx", type=int, default=146)
    ap.add_argument("--ny", type=int, default=50)
    ap.add_argument("--dx", type=float, default=30.0)
    ap.add_argument("--itower", type=int, default=110)
    ap.add_argument("--jtower", type=int, default=25)
    ap.add_argument("--taper-x", type=int, default=12, help="cells (12 x 30 m = 360 m)")
    ap.add_argument("--taper-y", type=int, default=8)
    ap.add_argument("--no-array", action="store_true")
    ap.add_argument("--flat", action="store_true", help="land cover only, no terrain")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    with rasterio.open(a.dem) as dm, rasterio.open(a.std) as ds_:
        X, Y, (tx, ty), (ex, ey) = rotated_map_coords(
            TOWER_LON, TOWER_LAT, dm.crs, a.wind_from, a.nx, a.ny, a.dx, a.dx,
            a.itower, a.jtower)
        z = bilinear(dm, dm.read(1).astype(np.float64), X, Y)
        sd = bilinear(ds_, ds_.read(1).astype(np.float64), X, Y)
        allz = dm.read(1).astype(np.float64); allsd = ds_.read(1).astype(np.float64)

    # ---- land cover: water first, from the measured sub-cell spread -----------------
    wlev = float(np.median(allz[np.isfinite(allsd) & (allsd < WATER_STD_MAX)]))
    water = (sd < WATER_STD_MAX) & (np.abs(z - wlev) < WATER_LEVEL_TOL)
    print(f"  tower {TOWER_LON:.6f}, {TOWER_LAT:.6f} -> EPSG:3071 ({tx:.1f}, {ty:.1f})")
    print(f"  wind from {a.wind_from:.0f} deg -> LES +x points to bearing "
          f"{(a.wind_from+180)%360:.0f} deg (upwind is -x)")
    print(f"  water level {wlev:.2f} m; WATER cells in domain: {water.sum()} "
          f"({100*water.mean():.1f}%, {water.sum()*a.dx**2/1e6:.2f} km2)")
    xg = (np.arange(a.nx) - a.itower) * a.dx      # +x downwind, so upwind distance is -xg
    yg = (np.arange(a.ny) - a.jtower) * a.dx
    if water.any():
        wi = np.where(water.any(axis=0))[0]
        print(f"    water spans along-wind {-xg[wi.max()]:.0f} .. {-xg[wi.min()]:.0f} m "
              f"upwind of the tower")
        up = water[:, xg < 0]
        print(f"    fraction of the UPWIND half of the domain that is water: "
              f"{100*up.mean():.1f}%")

    # ---- terrain --------------------------------------------------------------------
    topo = z - z.mean()
    if a.flat:
        topo[:] = 0.0
    wx = taper_weights(a.nx, a.taper_x); wy = taper_weights(a.ny, a.taper_y)
    W2 = wx[None, :] * wy[:, None]
    topo = topo * W2
    print(f"  terrain: raw {z.min():.1f}..{z.max():.1f} m (range {np.ptp(z):.1f}); after "
          f"mean removal + seam taper {topo.min():.2f}..{topo.max():.2f} m")
    print(f"    seam discontinuity x {np.abs(topo[:,0]-topo[:,-1]).max():.3f} m, "
          f"y {np.abs(topo[0,:]-topo[-1,:]).max():.3f} m")
    tw = water & (W2 < 0.999)
    print(f"    water cells whose TERRAIN is seam-tapered: {tw.sum()} of {water.sum()} "
          f"(their z0 and heat flux are untouched -- see below)")

    # ---- roughness and surface heat flux --------------------------------------------
    z0 = np.full((a.ny, a.nx), Z0_GRASS)
    z0[water] = Z0_WATER
    wth = np.full((a.ny, a.nx), WTH_LAND)
    wth[water] = WTH_WATER
    if not a.no_array:
        # array centre in map coordinates, then in LES coordinates
        b = np.radians(ARRAY_BEARING)
        ax_, ay_ = tx + ARRAY_DISTANCE * np.sin(b), ty + ARRAY_DISTANCE * np.cos(b)
        dxm, dym = X - ax_, Y - ay_
        # the array's own extents are east-west / north-south, i.e. MAP aligned
        patch = (np.abs(dxm) <= ARRAY_EW / 2) & (np.abs(dym) <= ARRAY_NS / 2)
        patch &= ~water
        z0[patch] = Z0_ARRAY
        topo[patch] += D_ARRAY
        if patch.any():
            pi_ = np.where(patch.any(axis=0))[0]
            print(f"  solar array: {patch.sum()} cells ({patch.sum()*a.dx**2/1e4:.2f} ha), "
                  f"z0 {Z0_GRASS} -> {Z0_ARRAY} m, d = {D_ARRAY} m")
            print(f"    along-wind position: {-xg[pi_.max()]:.0f} .. {-xg[pi_.min()]:.0f} m "
                  f"upwind of the tower "
                  f"({'UPWIND' if -xg[pi_.min()] > 0 else 'DOWNWIND'})")
        else:
            print("  solar array: NOT in the domain for this rotation")
    # The LAND COVER is deliberately NOT tapered, unlike the terrain. Terrain height enters
    # the coordinate transform and its metric tensor, so a step at the periodic seam is a
    # numerical cliff. Roughness and surface heat flux are local boundary conditions: a step
    # at the seam is a coastline, which is a thing that exists. Tapering them would erase
    # the water from the upwind edge of exactly the easterly cases that are supposed to
    # sample it.

    write_topofile(os.path.join(a.outdir, "topo.bin"), topo)
    for nm, arr in (("topo", topo), ("z0m", z0), ("water", water.astype(np.float32)),
                    ("htFlux", wth)):
        np.save(os.path.join(a.outdir, nm + ".npy"), arr)
    print(f"  wrote {a.outdir}/topo.bin ({a.nx}x{a.ny} float32 + 2 int32 header)")

    # ---- restart file ----------------------------------------------------------------
    from netCDF4 import Dataset
    dst = os.path.join(a.outdir, os.path.basename(a.restart_in))
    shutil.copy2(a.restart_in, dst)
    with Dataset(dst, "a") as ds:
        zp = np.asarray(ds["zPos"][:]); shp = zp.shape
        zcol = np.squeeze(zp)[:, 0, 0].astype(np.float64)
        zC = float(zcol[-1]); Fk = zcol
        new = Fk[:, None, None] * (zC - topo[None, :, :]) / zC + topo[None, :, :]
        ds["zPos"][:] = new.reshape(shp).astype(np.float32)
        t2 = np.asarray(ds["topoPos"][:])
        ds["topoPos"][:] = topo.reshape(t2.shape).astype(np.float32)
        ds["z0m"][:] = z0.reshape(t2.shape).astype(np.float32)
        if "z0t" in ds.variables:
            ds["z0t"][:] = np.minimum(z0, 0.01).reshape(t2.shape).astype(np.float32)
        if "htFlux" in ds.variables:
            ds["htFlux"][:] = wth.reshape(t2.shape).astype(np.float32)
    print(f"  wrote {dst}: zPos terrain-following; topoPos, z0m, z0t, htFlux replaced")
    print("  NOTE albedo has no pathway: FastEddy in this configuration has NO radiation")
    print("       scheme (surflayerSelector=1 prescribes the kinematic heat flux directly),")
    print("       so what albedo would have controlled is subsumed by htFlux, which IS")
    print("       per-cell -- cuda_surfaceLayerDevice.cu:191 reuses the array when")
    print("       surflayer_idealsine=0. The built-in surflayer_offshore wave-roughness")
    print("       path is a GLOBAL switch and cannot be applied to water cells only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
