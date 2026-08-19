#!/usr/bin/env python3
"""Build the STATIC surface for the Kegonsa domain: terrain, roughness, heat flux.

Replaces the rotated preprocessing of bin/prep_stage6.py. The domain no longer turns with
the wind -- the wind turns instead, via FastEddy's (U_g, V_g) -- so the surface is built
ONCE, on a fixed, north-up, map-projected grid, and is bit-identical for every direction.
Any directional difference in a footprint is therefore flow and cannot be a resampling
artifact. It also makes the class of bug that put the solar array at a fixed *upwind
distance* structurally impossible: the array is a rectangle in EPSG:3071 and nothing else.

Inputs (both gitignored, see data/README.md):
  data/raw/output_USGS10m.tif                             USGS 3DEP 1/3 arcsec, EPSG:4269
  data/raw/ESA_WorldCover_10m_2021_v200_N42W090_Map.tif   ESA WorldCover v200, EPSG:4326

Terrain is resampled with `average` (a height field the model feels through its mean) and
land cover with `mode` (a class map must never be averaged -- the mean of "water" and
"forest" is not a land-cover class).

usage: prep_surface.py --outdir data/grid [--nx 186 --dx 24 --pad 20]
"""
import argparse
import os
import struct
import subprocess
import sys

import numpy as np
import rasterio
from pyproj import Transformer

# ---------------------------------------------------------------- site definition
# Surveyed tower coordinate. Single source of truth for the whole project.
TOWER_LON, TOWER_LAT = -89.292362, 42.957160
SITE_CRS = "EPSG:3071"       # NAD83(HARN) / Wisconsin Transverse Mercator, metres

# THE SOLAR ARRAY CONTAINS THE TOWER. It is not a patch at some distance -- the tower is
# inside it, near the south end. Offsets are in map metres from the tower:
#     60 m east and west, 250 m north, 100 m south  ->  120 m x 350 m = 4.20 ha
# This is a geographic rectangle and nothing about it depends on the wind. Its upwind
# reach is therefore a strong function of direction (250 m for a northerly, 60 m for an
# easterly), which is exactly the Stage 6 signal.
ARRAY_DX_W, ARRAY_DX_E = -60.0, 60.0
ARRAY_DY_S, ARRAY_DY_N = -100.0, 250.0

# ESA WorldCover v200 class -> aerodynamic roughness length (m).
# Water is aerodynamically smooth; the built-in surflayer_offshore wave-roughness path in
# FastEddy is a GLOBAL switch and cannot be applied to water cells only, so a per-cell z0
# is used instead (see data/README.md).
WORLDCOVER_Z0 = {
    10: 1.00,    # tree cover
    20: 0.20,    # shrubland
    30: 0.03,    # grassland
    40: 0.10,    # cropland  (mid-season; CDL would split corn/soy, see PLAN)
    50: 0.50,    # built-up
    60: 0.01,    # bare / sparse vegetation
    70: 0.001,   # snow and ice
    80: 1.0e-4,  # PERMANENT WATER
    90: 0.05,    # herbaceous wetland
    95: 0.20,    # mangroves (not present here)
    100: 0.02,   # moss and lichen
}
WATER_CLASS = 80
Z0_ARRAY = 0.20              # bulk solar-array patch
D_ARRAY = 1.2                # displacement height (m); recorded, not yet used by FastEddy
Z0_FALLBACK = 0.05           # any unmapped/no-data class
WTH_UNIFORM = 0.0            # kinematic sensible heat flux; 0 everywhere == neutral


def taper_weights(n, pad):
    """1 in the interior, smoothly 0 at both ends over `pad` cells (raised cosine)."""
    w = np.ones(n)
    if pad <= 0:
        return w
    r = 0.5 * (1.0 - np.cos(np.pi * (np.arange(pad) + 0.5) / pad))
    w[:pad] = r
    w[-pad:] = r[::-1]
    return w


def warp(src, dst, bounds, res, resampling, dtype, extra=()):
    x0, y0, x1, y1 = bounds
    cmd = ["gdalwarp", "-overwrite", "-t_srs", SITE_CRS,
           "-te", f"{x0}", f"{y0}", f"{x1}", f"{y1}",
           "-tr", f"{res}", f"{res}", "-r", resampling, "-ot", dtype,
           "-co", "COMPRESS=LZW", *extra, src, dst]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return dst


def smooth121(a, npass):
    """Separable 1-2-1 passes.

    WHY. The terrain dt is set by the STEEPEST cell, through the terrain-following metric
    amplification CFL_eff ~ CFL_3d sqrt(1 + (slope dx/dz)^2) (CLAUDE.md). The raw 24 m
    field reaches |grad z| = 0.37 -- a 9 m drop across one cell -- which the grid cannot
    represent as anything but an aliased step, and which alone would cost a 1.44x smaller
    dt on every terrain run. Two passes take the maximum to 0.245 (amplification 1.21) for
    an rms terrain change of 0.44 m against 61 m of relief. This is filtering topography
    to the resolvable scale, which is standard for terrain-following LES -- not a fudge to
    buy speed, though it does buy speed.
    """
    for _ in range(int(npass)):
        b = a.copy()
        b[1:-1, :] = 0.25 * a[:-2, :] + 0.5 * a[1:-1, :] + 0.25 * a[2:, :]
        a = b.copy()
        b[:, 1:-1] = 0.25 * a[:, :-2] + 0.5 * a[:, 1:-1] + 0.25 * a[:, 2:]
        a = b
    return a


def write_topofile(path, topo):
    """FastEddy topoFile: two int32 dimensions then the field, row-major, float32."""
    ny, nx = topo.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", nx, ny))
        f.write(topo.astype("<f4").tobytes(order="C"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/grid")
    ap.add_argument("--nx", type=int, default=186)
    ap.add_argument("--ny", type=int, default=186)
    ap.add_argument("--dx", type=float, default=24.0)
    ap.add_argument("--pad", type=int, default=20,
                    help="taper ring width in cells (terrain only)")
    ap.add_argument("--smooth", type=int, default=2,
                    help="1-2-1 smoothing passes on the terrain (see note in source)")
    ap.add_argument("--dem", default="data/raw/output_USGS10m.tif")
    ap.add_argument("--lc", default="data/raw/"
                    "ESA_WorldCover_10m_2021_v200_N42W090_Map.tif")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    tf = Transformer.from_crs("EPSG:4326", SITE_CRS, always_xy=True)
    tx, ty = tf.transform(TOWER_LON, TOWER_LAT)
    # Snap the box to whole cells so the grid is reproducible bit-for-bit, and centre the
    # tower: with periodic BCs the tower could sit anywhere, but centring maximises the
    # radius of REAL geography before the taper ring in every direction at once.
    x0 = np.floor((tx - a.nx * a.dx / 2) / a.dx) * a.dx
    y0 = np.floor((ty - a.ny * a.dx / 2) / a.dx) * a.dx
    x1, y1 = x0 + a.nx * a.dx, y0 + a.ny * a.dx
    bounds = (x0, y0, x1, y1)
    print(f"  tower {TOWER_LON:.6f}, {TOWER_LAT:.6f} -> {SITE_CRS} ({tx:.1f}, {ty:.1f})")
    print(f"  domain {a.nx} x {a.ny} @ {a.dx:.0f} m = {a.nx*a.dx:.0f} x {a.ny*a.dx:.0f} m")
    print(f"  bounds {x0:.0f} {y0:.0f} -> {x1:.0f} {y1:.0f}")
    it = int(round((tx - x0) / a.dx - 0.5))
    jt = int(round((ty - y0) / a.dx - 0.5))
    print(f"  tower cell (i,j) = ({it}, {jt}) of ({a.nx}, {a.ny})")

    # ---- terrain -------------------------------------------------------------------
    dem = warp(a.dem, os.path.join(a.outdir, "dem24.tif"), bounds, a.dx,
               "average", "Float32")
    with rasterio.open(dem) as ds:
        # rasterio row 0 is the NORTH edge; the model's j index increases northward.
        z = np.flipud(ds.read(1).astype(np.float64))
    if not np.isfinite(z).all():
        raise SystemExit("DEM has no-data inside the domain -- widen the source tile")
    print(f"  terrain {z.min():.2f} .. {z.max():.2f} m  (relief {np.ptp(z):.1f} m), "
          f"at tower {z[jt, it]:.2f} m")

    # ---- land cover ----------------------------------------------------------------
    lc = warp(a.lc, os.path.join(a.outdir, "lc24.tif"), bounds, a.dx,
              "mode", "Byte")
    with rasterio.open(lc) as ds:
        cls = np.flipud(ds.read(1))
    z0 = np.full(cls.shape, Z0_FALLBACK)
    for k, v in WORLDCOVER_Z0.items():
        z0[cls == k] = v
    water = cls == WATER_CLASS
    print("  land cover (WorldCover v200, mode-resampled to 24 m):")
    for k, n in sorted(zip(*np.unique(cls, return_counts=True)),
                       key=lambda p: -p[1]):
        print(f"    class {int(k):3d}  {n:6d} cells  {100*n/cls.size:5.2f}%  "
              f"z0 = {WORLDCOVER_Z0.get(int(k), Z0_FALLBACK):g} m")

    # ---- solar array: a rectangle in map coordinates, overriding the land cover -----
    # WorldCover 2021 labels the array as cropland -- it does not see photovoltaics -- so
    # the patch has to be imposed. Cell CENTRES against the rectangle.
    xc = x0 + (np.arange(a.nx) + 0.5) * a.dx
    yc = y0 + (np.arange(a.ny) + 0.5) * a.dx
    XX, YY = np.meshgrid(xc - tx, yc - ty)
    array = ((XX >= ARRAY_DX_W) & (XX <= ARRAY_DX_E) &
             (YY >= ARRAY_DY_S) & (YY <= ARRAY_DY_N))
    z0[array] = Z0_ARRAY
    water &= ~array
    print(f"  solar array: {array.sum()} cells "
          f"({array.sum()*a.dx**2/1e4:.2f} ha, nominal 4.20 ha), z0 -> {Z0_ARRAY} m, "
          f"d = {D_ARRAY} m")
    print(f"    extent from the tower: {ARRAY_DX_W:+.0f} .. {ARRAY_DX_E:+.0f} m E-W, "
          f"{ARRAY_DY_S:+.0f} .. {ARRAY_DY_N:+.0f} m N-S "
          f"-- the tower is INSIDE it")
    print(f"    contains the tower cell: {bool(array[jt, it])}")

    # ---- taper the TERRAIN only ----------------------------------------------------
    # Terrain height enters the coordinate transform and its metric tensor, so a step at
    # the periodic seam is a numerical cliff. Roughness and heat flux are local boundary
    # conditions where a seam is a coastline -- a thing that exists -- and tapering them
    # would erase the lake from exactly the easterly cases meant to sample it.
    w2d = np.outer(taper_weights(a.ny, a.pad), taper_weights(a.nx, a.pad))
    edge = np.concatenate([z[0], z[-1], z[:, 0], z[:, -1]])
    base = float(np.median(edge))
    topo = base + (z - base) * w2d
    if a.smooth:
        raw_topo = topo.copy()
        topo = smooth121(topo, a.smooth)
        print(f"  terrain smoothing: {a.smooth} x (1-2-1), rms change "
              f"{np.sqrt(((topo-raw_topo)**2).mean()):.3f} m, "
              f"max {np.abs(topo-raw_topo).max():.2f} m")
    print(f"  taper: {a.pad} cells ({a.pad*a.dx:.0f} m) to a constant {base:.2f} m; "
          f"real geography to ~{(min(it, a.nx-it, jt, a.ny-jt) - a.pad)*a.dx:.0f} m "
          f"from the tower")
    print(f"    terrain after taper {topo.min():.2f} .. {topo.max():.2f} m, "
          f"at tower {topo[jt, it]:.2f} m")

    # ---- slope statistics: these set the terrain dt (CLAUDE.md CFL amplification) ----
    gy, gx = np.gradient(topo, a.dx)
    slope = np.hypot(gx, gy)
    q = np.percentile(slope, [50, 90, 99]), slope.max()
    ampl = float(np.sqrt(1.0 + (q[1] * a.dx / 8.558) ** 2))
    print("  slope |grad z|: p50 %.4f  p90 %.4f  p99 %.4f  max %.4f"
          % (q[0][0], q[0][1], q[0][2], q[1]))
    print("  terrain CFL amplification at the steepest cell (dx/dz = %.2f): %.3f"
          % (a.dx / 8.558, ampl))
    print("    -> terrain dt must be the flat dt divided by %.3f" % ampl)

    wth = np.full(topo.shape, WTH_UNIFORM)
    write_topofile(os.path.join(a.outdir, "topo.bin"), topo)
    for nm, arr in (("topo", topo), ("z0m", z0), ("water", water.astype(np.float32)),
                    ("array", array.astype(np.float32)), ("htFlux", wth),
                    ("lcclass", cls.astype(np.int16))):
        np.save(os.path.join(a.outdir, nm + ".npy"), arr)
    meta = dict(nx=a.nx, ny=a.ny, dx=a.dx, x0=x0, y0=y0, itower=it, jtower=jt,
                tower_x=tx, tower_y=ty, pad=a.pad, base=base,
                slope_p50=float(q[0][0]), slope_p90=float(q[0][1]),
                slope_p99=float(q[0][2]), slope_max=float(q[1]),
                cfl_amplification=ampl, smooth=a.smooth)
    np.save(os.path.join(a.outdir, "meta.npy"), meta, allow_pickle=True)
    print(f"  wrote {a.outdir}/topo.bin and the .npy surface fields")
    return 0


if __name__ == "__main__":
    sys.exit(main())
