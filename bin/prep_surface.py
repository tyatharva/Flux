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
Z0_ARRAY = 0.10              # bulk solar-array patch; see --z0-array and the RSL note below
D_ARRAY = 1.5                # displacement height (m) of the array
Z0_FALLBACK = 0.05           # any unmapped/no-data class
WTH_UNIFORM = 0.0            # kinematic VIRTUAL heat flux; 0 everywhere == neutral

# Displacement height per class, d ~ 0.7 h_c with h_c ~ 10 z0 (the standard pair of
# rules-of-thumb, applied consistently rather than tabulated independently). This is an
# LPDM-side quantity ONLY: FastEddy v5.0.1 carries surflayer_z0 and surflayer_z0t and no
# displacement height at all, so nothing here reaches the LES unless --raise-topo puts it
# into the terrain.
D_PER_Z0 = 7.0               # d = 7 z0, i.e. 0.7 * (10 z0)

# Convective cases only (--wth > 0): per-class MULTIPLIER on the reference land value.
# FastEddy in this configuration has no radiation scheme -- surflayerSelector = 1
# prescribes the kinematic surface heat flux directly -- so htFlux is the single channel
# through which surface type reaches the thermodynamics, and it is per-cell
# (cuda_surfaceLayerDevice.cu: with surflayer_idealsine = 0 the kernel comments
# "reuse *htFlux array values" and never overwrites what the restart injected).
#
# Water is the important one and it is directional: within 4 km the water sits almost
# entirely in the E and NE octants, so an easterly fetch is over a surface with roughly a
# tenth of the land's sensible heat flux. A lake in a Wisconsin summer afternoon runs
# H of order 0-20 W/m2 against 100-200 over the crop, because its heat goes into storage
# and into latent flux, not into the air.
WORLDCOVER_WTH = {
    10: 1.10,    # tree cover -- rough, dry canopy, Bowen ratio above cropland
    20: 1.10,    # shrubland
    30: 1.10,    # grassland
    40: 1.00,    # cropland  -- the REFERENCE
    50: 1.50,    # built-up  -- little transpiration, most of Rn into H
    60: 1.40,    # bare
    70: 0.30,    # snow and ice
    80: 0.12,    # PERMANENT WATER -- storage and latent, not sensible
    90: 0.60,    # herbaceous wetland
    95: 0.60,
    100: 1.00,
}
# The solar array. PROJECT_BRIEF.md lists the elevated heat source as an accepted omission, and
# with a per-cell htFlux it no longer has to be one. PV modules are darker than the crop
# they replaced (albedo ~0.1 against ~0.2) and they do not transpire, so essentially all of
# the absorbed shortwave that is not exported as electricity leaves as sensible heat. Field
# studies of utility-scale arrays report a daytime sensible-flux enhancement of order 1.5-2
# over the adjacent vegetation. 1.6 is that, stated as an assumption and swept if it
# matters. This is also the pathway PROJECT_BRIEF.md identifies for albedo: with no radiation
# scheme, htFlux is what albedo would have controlled.
WTH_ARRAY = 1.60
WTH_FALLBACK = 1.00

# BOWEN RATIO PER CLASS, and why this table has to exist.
#
# The multipliers above are SENSIBLE-flux ratios -- that is what the field studies they
# come from report, and what "PV modules do not transpire" is an argument about. But the
# field FastEddy is given must be the VIRTUAL heat flux, because the run is dry and
# buoyancy is what htFlux is for (PROJECT_BRIEF.md, "Boundary and initial conditions"):
#
#     w'th_v' = w'th' + 0.61 th w'q' = w'th' * (1 + 0.61 th c_p / (B L_v))
#
# with B the Bowen ratio. The conversion factor is therefore CLASS-DEPENDENT, and applying
# sensible ratios to a virtual field is not a small error in the right direction -- it
# gets the contrast wrong. Water and wet crops have small B and gain a lot of buoyancy
# from their latent flux; panels and pavement have large B and gain almost none. Working
# in virtual flux COMPRESSES the wet-dry buoyancy contrast, which is physically correct
# and is exactly what the decision to run dry is trading for.
#
# The fourth pass prescribed the CONUS404 sensible flux directly and never applied any of
# this; PROJECT_BRIEF.md predicted that would cost 5-10% in z_i and w*.
WORLDCOVER_BOWEN = {
    10: 0.4,     # tree cover -- transpiring deciduous canopy
    20: 0.8,     # shrubland
    30: 0.5,     # grassland
    40: 0.4,     # cropland -- midsummer corn/soy, well watered. THE REFERENCE.
    50: 2.0,     # built-up
    60: 2.0,     # bare
    70: 0.5,     # snow and ice
    80: 0.15,    # PERMANENT WATER -- almost all of it latent
    90: 0.2,     # herbaceous wetland
    95: 0.2,
    100: 0.8,
}
BOWEN_ARRAY = 4.0            # PV modules do not transpire; the shaded ground contributes little
BOWEN_FALLBACK = 0.5
# 0.61 * theta * c_p / L_v at theta = 300 K, c_p = 1004.5, L_v = 2.5e6
VIRT_COEFF = 0.61 * 300.0 * 1004.5 / 2.5e6


def virtual_factor(bowen):
    """w'th_v'/w'th' for a surface with the given Bowen ratio."""
    return 1.0 + VIRT_COEFF / np.maximum(np.asarray(bowen, dtype=np.float64), 1e-3)


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
    amplification CFL_eff ~ CFL_3d sqrt(1 + (slope dx/dz)^2) (PROJECT_BRIEF.md). The raw 24 m
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
    ap.add_argument("--nx", type=int, default=122)
    ap.add_argument("--ny", type=int, default=122)
    ap.add_argument("--dx", type=float, default=16.0)
    ap.add_argument("--pad", type=int, default=12,
                    help="taper ring width in cells (terrain only). 12 measured as the knee "
                         "on this domain: 20 -> 12 buys 128 m of real terrain (656 -> 784 m "
                         "from the tower, which is what covers Kljun's x90 here) for +0.8% "
                         "CFL amplification, while 10 and 8 make the TAPER the steepest "
                         "cell in the domain (slope_max 0.184 -> 0.210 -> 0.222).")
    ap.add_argument("--wth", type=float, default=0.0,
                    help="reference cropland kinematic VIRTUAL heat flux (K m/s). "
                         "0 = neutral. Per-class virtual multipliers scale it. This is "
                         "w'th_v', NOT the sensible flux -- use --wth-sensible to pass a "
                         "CONUS404 number and have it converted.")
    ap.add_argument("--wth-sensible", type=float, default=None,
                    help="reference cropland SENSIBLE heat flux (K m/s), converted to "
                         "virtual with the cropland Bowen ratio. Overrides --wth.")
    ap.add_argument("--bowen-crop", type=float, default=None,
                    help="cropland Bowen ratio (default from WORLDCOVER_BOWEN). The array's "
                         "virtual multiplier is insensitive to its OWN Bowen ratio "
                         "(1.37-1.40 over B = 2-6) and sensitive to this one "
                         "(1.31-1.45 over B = 0.3-0.6), so sweep it here.")
    ap.add_argument("--z0-array", type=float, default=Z0_ARRAY,
                    help="bulk roughness of the solar-array patch")
    ap.add_argument("--d-array", type=float, default=D_ARRAY,
                    help="displacement height of the solar-array patch")
    ap.add_argument("--raise-topo", action="store_true",
                    help="RAISE topoPos by the displacement height over the array, so the "
                         "model ground IS the effective surface there and the first model "
                         "level clears panel top. dmap.npy is then written as ZEROS, "
                         "because the displacement is already in the terrain and the LPDM "
                         "must not count it twice. The receptor must then be released with "
                         "stage5_footprint.py --exact-agl, or it sits 10 m above the "
                         "PANELS rather than 10 m above bare ground.")
    ap.add_argument("--d-cap", type=float, default=None,
                    help="cap on the per-cell displacement height, m. Defaults to the "
                         "first model level, so the LPDM's sub-layer log-law anchor always "
                         "stays above the displacement surface. Tall tree cover is the "
                         "reason: d ~ 0.7 h_c is metres there, the LES resolves none of "
                         "it, and it is nowhere near the tower.")
    ap.add_argument("--dz-sfc", type=float, default=None,
                    help="surface layer thickness, m; sets the first model level (dz/2), "
                         "the default --d-cap and the terrain CFL amplification. Defaults "
                         "to the value bin/vgrid.py solves for --nz/--zceiling.")
    ap.add_argument("--nz", type=int, default=122)
    ap.add_argument("--zceiling", type=float, default=2500.0)
    ap.add_argument("--receptor", type=float, default=10.0)
    ap.add_argument("--receptor-k", type=int, default=2,
                    help="cell index the receptor sits on. NOT derivable from the height: "
                         "dz_sfc = receptor/(k+0.5), so k is the choice that sets the "
                         "surface spacing (bin/vgrid.py). k=2 is the production grid.")
    ap.add_argument("--smooth", type=int, default=2,
                    help="1-2-1 smoothing passes on the terrain (see note in source)")
    ap.add_argument("--dem", default="data/raw/output_USGS10m.tif")
    ap.add_argument("--lc", default="data/raw/"
                    "ESA_WorldCover_10m_2021_v200_N42W090_Map.tif")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # The vertical grid is not decoration here: it sets the first model level (which caps
    # the displacement height and decides whether the array's log law has any room) and
    # the dz that the terrain CFL amplification is measured against. Solve it from
    # FastEddy's own zDeform rather than restating a number from a comment.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vgrid import solve_c1, z_of_zeta              # noqa: E402
    if a.dz_sfc is None:
        d_zeta = a.zceiling / (a.nz - 0.5)
        zeta = (np.arange(a.nz) + 0.5) * d_zeta
        c1 = solve_c1(zeta[a.receptor_k], a.receptor, a.zceiling)
        zlev = z_of_zeta(zeta, c1, a.zceiling)
        assert abs(zlev[a.receptor_k] - a.receptor) < 1e-6, "vertical grid solve failed"
        a.dz_sfc = float(2.0 * zlev[0])
    z_first = 0.5 * a.dz_sfc
    if a.d_cap is None:
        a.d_cap = z_first
    print(f"  vertical grid: Nz={a.nz}, zCeiling={a.zceiling:.0f} m, receptor {a.receptor:.1f} m "
          f"at k={a.receptor_k}\n    -> dz_sfc = {a.dz_sfc:.4f} m, first level "
          f"{z_first:.4f} m, dx/dz = {a.dx/a.dz_sfc:.3f}, d capped at {a.d_cap:.4f} m")

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
    print(f"  land cover (WorldCover v200, mode-resampled to {a.dx:.0f} m):")
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
    z0[array] = a.z0_array
    water &= ~array
    print(f"  solar array: {array.sum()} cells "
          f"({array.sum()*a.dx**2/1e4:.2f} ha, nominal 4.20 ha), z0 -> {a.z0_array} m, "
          f"d = {a.d_array} m")
    # THE ARRAY CAN BE AERODYNAMICALLY INVISIBLE, and silently. WorldCover labels the
    # array as cropland, whose z0 is also 0.10 m -- so at --z0-array 0.10 the override
    # changes nothing at all and the array's entire NEUTRAL signal is zero, leaving only
    # the convective heat-flux contrast. Say so rather than let it pass.
    z0_crop = WORLDCOVER_Z0[40]
    if abs(a.z0_array - z0_crop) < 1e-6:
        print(f"    WARNING: z0_array == z0_cropland == {z0_crop} m, so the array has NO "
              f"roughness\n             contrast with the surface it replaced. In the "
              f"neutral regime it is\n             invisible. Use --raise-topo with a "
              f"larger --z0-array to restore it.")
    print(f"    ln(z_first/z0_array) = {np.log(z_first/a.z0_array):.2f} at the first model "
          f"level ({z_first:.2f} m)"
          + (f" -- raised to {np.log((z_first)/a.z0_array):.2f} above panel top by "
             f"--raise-topo" if a.raise_topo else ""))
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
    # TERRAIN IS A PERTURBATION ABOUT ZERO, not absolute elevation. FastEddy's
    # terrain-following map is z(k) = zeta_k * (zC - h_g)/zC + h_g, so a ground at 269 m
    # ASL would compress every level by 9% and put the 30 m receptor at 27.3 m AGL. It
    # would also be inconsistent with the base state, which is hydrostatically built for a
    # ground at z = 0 and which the flat spin-up already carries.
    topo = (z - base) * w2d
    if a.smooth:
        raw_topo = topo.copy()
        topo = smooth121(topo, a.smooth)
        print(f"  terrain smoothing: {a.smooth} x (1-2-1), rms change "
              f"{np.sqrt(((topo-raw_topo)**2).mean()):.3f} m, "
              f"max {np.abs(topo-raw_topo).max():.2f} m")
    print(f"  taper: {a.pad} cells ({a.pad*a.dx:.0f} m) to zero; datum {base:.2f} m ASL; "
          f"real geography to ~{(min(it, a.nx-it, jt, a.ny-jt) - a.pad)*a.dx:.0f} m "
          f"from the tower")
    print(f"    terrain after taper {topo.min():.2f} .. {topo.max():.2f} m (about 0), "
          f"at tower {topo[jt, it]:+.2f} m")

    # ---- displacement height -----------------------------------------------------------
    # d ~ 0.7 h_c with h_c ~ 10 z0. Capped, because tree cover reaches d of several metres
    # while the first model level is at ~2 m: uncapped, the LPDM's sub-layer log-law anchor
    # (z_ref - d) would go negative over exactly the cells the LES represents as a bulk
    # roughness and nothing more. The cap is reported, not hidden.
    dmap = np.minimum(D_PER_Z0 * z0, a.d_cap)
    dmap[array] = min(a.d_array, a.d_cap)
    n_cap = int((D_PER_Z0 * z0 > a.d_cap).sum())
    print(f"  displacement height: d = {D_PER_Z0:.1f} z0, capped at {a.d_cap:.3f} m "
          f"({n_cap} cells = {100*n_cap/dmap.size:.1f}% capped, mostly tree cover)")
    print(f"    range {dmap.min():.3f} .. {dmap.max():.3f} m; mean {dmap.mean():.3f} m; "
          f"array {dmap[array].mean() if array.any() else float('nan'):.3f} m; "
          f"at tower {dmap[jt, it]:.3f} m")

    # ---- optionally put the array's displacement surface INTO the terrain ---------------
    if a.raise_topo:
        # One 1-2-1 pass on the raise alone. A bare 1-cell step of 1.5 m over 16 m is a
        # slope of 0.094 that the grid can only represent as an aliased cliff, and it
        # enters the metric tensor; ramping it over ~1 cell each side keeps the same
        # displacement surface without handing the terrain dt an artificial steepest cell.
        raise_f = smooth121(np.where(array, a.d_array, 0.0), 1)
        topo = topo + raise_f
        # The displacement is now the model's ground wherever it was added, so the LPDM
        # must NOT apply it again there. Everywhere else d is unchanged.
        dmap = np.maximum(dmap - raise_f, 0.0)
        print(f"  --raise-topo: array terrain raised by up to {raise_f.max():.3f} m "
              f"(1-2-1 ramped); tower cell {raise_f[jt, it]:+.3f} m")
        print(f"    the first model level is now {z_first:.2f} m above the RAISED surface "
              f"= {z_first + raise_f[jt,it]:.2f} m above bare ground, "
              f"ln(z/z0) = {np.log(z_first/a.z0_array):.2f}")
        print(f"    dmap over the array is now {dmap[array].mean():.3f} m (the rest is in "
              f"topoPos)")
        print(f"    THE RECEPTOR MUST BE RELEASED WITH --exact-agl: the instrument is 10 m "
              f"above BARE\n    GROUND, so over the raised patch it sits at "
              f"{a.receptor - raise_f[jt,it]:.3f} m above the model surface, between "
              f"levels.")
    zC = a.zceiling
    print(f"    receptor at {a.receptor:.1f} m will sit at "
          f"{a.receptor * (zC - topo[jt, it]) / zC:.4f}"
          f" m above the tower cell's ground (terrain-following compression)")

    # ---- slope statistics: these set the terrain dt (PROJECT_BRIEF.md CFL amplification) ----
    gy, gx = np.gradient(topo, a.dx)
    slope = np.hypot(gx, gy)
    q = np.percentile(slope, [50, 90, 99]), slope.max()
    aniso = a.dx / a.dz_sfc
    ampl = float(np.sqrt(1.0 + (q[0][1] * aniso) ** 2))
    ampl_max = float(np.sqrt(1.0 + (q[1] * aniso) ** 2))
    print("  slope |grad z|: p50 %.4f  p90 %.4f  p99 %.4f  max %.4f"
          % (q[0][0], q[0][1], q[0][2], q[1]))
    print("  terrain CFL amplification (dx/dz = %.3f): %.3f at p90, %.3f at the steepest cell"
          % (aniso, ampl, ampl_max))
    print("    -> terrain dt is at most the flat dt divided by %.3f. THIS IS A PROJECTION,"
          % ampl_max)
    print("       not a substitute for bisecting it: the amplification formula sets the")
    print("       search bracket, and k0/k1 < 1 sets the answer (PROJECT_BRIEF.md).")

    # ---- surface heat flux: SENSIBLE ratios converted to VIRTUAL multipliers ----------
    bowen_crop = a.bowen_crop if a.bowen_crop is not None else WORLDCOVER_BOWEN[40]
    f_crop = float(virtual_factor(bowen_crop))
    if a.wth_sensible is not None:
        a.wth = a.wth_sensible * f_crop
    wth = np.full(topo.shape, WTH_UNIFORM)
    vratio = {}
    for k, sr in WORLDCOVER_WTH.items():
        b = bowen_crop if k == 40 else WORLDCOVER_BOWEN.get(k, BOWEN_FALLBACK)
        vratio[k] = sr * float(virtual_factor(b)) / f_crop
    vr_array = WTH_ARRAY * float(virtual_factor(BOWEN_ARRAY)) / f_crop
    if a.wth > 0.0:
        f = np.full(cls.shape, WTH_FALLBACK)
        for k, v in vratio.items():
            f[cls == k] = v
        f[array] = vr_array
        wth = a.wth * f
        print(f"  surface heat flux: VIRTUAL, w'th_v' = w'th'*(1 + {VIRT_COEFF:.4f}/B)")
        print(f"    cropland B = {bowen_crop:.2f} -> s->v factor {f_crop:.3f}; reference "
              f"sensible {a.wth/f_crop:.4f} -> virtual {a.wth:.4f} K m/s "
              f"({a.wth*1.15*1004.5:.0f} W/m2)")
        print(f"    {'class':<10} {'B':>6} {'s->v':>7} {'sens':>7} {'virt':>7} {'K m/s':>9}")
        for nm, k in (("tree", 10), ("grass", 30), ("cropland", 40), ("built", 50),
                      ("bare", 60), ("wetland", 90), ("water", 80)):
            b = bowen_crop if k == 40 else WORLDCOVER_BOWEN.get(k, BOWEN_FALLBACK)
            print(f"    {nm:<10} {b:6.2f} {float(virtual_factor(b)):7.3f} "
                  f"{WORLDCOVER_WTH[k]:7.3f} {vratio[k]:7.3f} {a.wth*vratio[k]:9.4f}")
        print(f"    {'ARRAY':<10} {BOWEN_ARRAY:6.2f} "
              f"{float(virtual_factor(BOWEN_ARRAY)):7.3f} {WTH_ARRAY:7.3f} "
              f"{vr_array:7.3f} {a.wth*vr_array:9.4f}"
              f"   ({a.wth*vr_array*1.15*1004.5:.0f} W/m2)")
        print(f"    range {wth.min():.4f} .. {wth.max():.4f} K m/s")
        print(f"    DOMAIN MEAN {wth.mean():.4f} K m/s  <-- set surflayer_wth to THIS in the")
        print(f"      flat spin-up. The .in scalar fills htFlux before the restart read")
        print(f"      overrides it per cell, so for a spin-up (which has no injection) the")
        print(f"      scalar IS the flux; using the cropland reference instead would spin")
        print(f"      up a boundary layer at the wrong z_i for the run it feeds.")
        print(f"    NOT tapered at the seams -- it is a local boundary condition, "
              f"like the roughness")
    write_topofile(os.path.join(a.outdir, "topo.bin"), topo)
    for nm, arr in (("topo", topo), ("z0m", z0), ("water", water.astype(np.float32)),
                    ("array", array.astype(np.float32)), ("htFlux", wth),
                    ("dmap", dmap), ("lcclass", cls.astype(np.int16))):
        np.save(os.path.join(a.outdir, nm + ".npy"), arr)
    meta = dict(nx=a.nx, ny=a.ny, dx=a.dx, x0=x0, y0=y0, itower=it, jtower=jt,
                tower_x=tx, tower_y=ty, pad=a.pad, base=base,
                slope_p50=float(q[0][0]), slope_p90=float(q[0][1]),
                slope_p99=float(q[0][2]), slope_max=float(q[1]),
                cfl_amplification=ampl, cfl_amplification_max=ampl_max,
                smooth=a.smooth, wth_ref=a.wth, wth_ref_sensible=a.wth / f_crop,
                wth_mean=float(wth.mean()), bowen_crop=bowen_crop,
                vratio_array=vr_array, z0_array=a.z0_array, d_array=a.d_array,
                raise_topo=bool(a.raise_topo), d_cap=a.d_cap,
                nz=a.nz, zceiling=a.zceiling, dz_sfc=a.dz_sfc, receptor=a.receptor)
    np.save(os.path.join(a.outdir, "meta.npy"), meta, allow_pickle=True)
    print(f"  wrote {a.outdir}/topo.bin and the .npy surface fields "
          f"(topo, z0m, water, array, htFlux, dmap, lcclass, meta)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
