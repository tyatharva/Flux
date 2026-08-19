#!/usr/bin/env python3
"""Resample the Dane County LiDAR DEM to 30 m AND classify open water, in one pass.

Two products on the same 30 m grid:

  kegonsa_30m_wtm.tif      mean bare-earth elevation
  kegonsa_30m_std.tif      standard deviation of the 0.4572 m elevations WITHIN each cell

The second one is the water detector, and it is a measurement rather than a guess. A LiDAR
bare-earth surface over open water is specular: returns are sparse and interpolated to a
level plane, so the sub-cell elevation spread collapses to a few millimetres. Land -- even
a mown field -- keeps centimetres to metres of sub-cell relief. Classifying on
"flat at 30 m" alone would sweep in ploughed fields and road corridors; classifying on
sub-cell spread separates them cleanly, and the histogram of that spread is bimodal, which
is what makes the threshold defensible instead of tuned.

Aggregation assigns each source pixel to a 30 m cell by its centre coordinate. 30 m is not
an integer multiple of 0.4572 m, so the cells do not nest and a resampling filter would
smear the very quantity being measured.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import rasterio
from rasterio.transform import Affine

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/dem/Dane2024_beDEM45cm_ELm_WTM.tif")
    ap.add_argument("--out-mean", default="data/dem/kegonsa_30m_wtm.tif")
    ap.add_argument("--out-std", default="data/dem/kegonsa_30m_std.tif")
    ap.add_argument("--radius", type=float, default=4000.0)
    ap.add_argument("--res", type=float, default=30.0)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--lat", type=float, default=None)
    a = ap.parse_args()

    if a.lon is None or a.lat is None:
        from prep_stage6 import TOWER_LON, TOWER_LAT
        a.lon = TOWER_LON if a.lon is None else a.lon
        a.lat = TOWER_LAT if a.lat is None else a.lat

    with rasterio.open(a.src) as src:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        tx, ty = tr.transform(a.lon, a.lat)
        r, res = a.radius, a.res
        x0 = np.floor((tx - r) / res) * res
        y0 = np.floor((ty - r) / res) * res
        nx = int(round(2 * r / res))
        ny = int(round(2 * r / res))
        x1, y1 = x0 + nx * res, y0 + ny * res
        print(f"  tower {a.lon:.6f}, {a.lat:.6f}  ->  EPSG:3071 ({tx:.1f}, {ty:.1f})")
        print(f"  window {x0:.0f} {y0:.0f} -> {x1:.0f} {y1:.0f}   {nx} x {ny} cells at {res} m")

        inv = ~src.transform
        cmin, rmax = inv * (x0, y0)
        cmax, rmin = inv * (x1, y1)
        c0, c1 = int(np.floor(cmin)), int(np.ceil(cmax))
        r0, r1 = int(np.floor(rmin)), int(np.ceil(rmax))
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, src.width), min(r1, src.height)
        nodata = src.nodata
        print(f"  reading source rows {r0}-{r1}, cols {c0}-{c1} "
              f"({(r1-r0)*(c1-c0)/1e6:.0f} Mpx) in bands")

        acc_n = np.zeros((ny, nx), dtype=np.int64)
        acc_s = np.zeros((ny, nx), dtype=np.float64)
        acc_q = np.zeros((ny, nx), dtype=np.float64)
        step = 4096
        sx = src.transform.a; sy = src.transform.e
        ox = src.transform.c; oy = src.transform.f
        cols = np.arange(c0, c1)
        cx = ox + sx * (cols + 0.5)
        ci = np.floor((cx - x0) / res).astype(np.int64)
        okc = (ci >= 0) & (ci < nx)
        ci = ci[okc]
        for rs in range(r0, r1, step):
            re = min(rs + step, r1)
            arr = src.read(1, window=((rs, re), (c0, c1)))
            rows = np.arange(rs, re)
            ry = oy + sy * (rows + 0.5)
            rj = np.floor((y0 + ny * res - ry) / res).astype(np.int64)
            rj = ny - 1 - rj
            okr = (rj >= 0) & (rj < ny)
            arr = arr[okr][:, okc]
            rj = rj[okr]
            good = np.isfinite(arr)
            if nodata is not None:
                good &= arr != nodata
            good &= arr > 1.0
            flat = (rj[:, None] * nx + ci[None, :]).ravel()
            v = arr.ravel().astype(np.float64)
            g = good.ravel()
            np.add.at(acc_n.ravel(), flat[g], 1)
            np.add.at(acc_s.ravel(), flat[g], v[g])
            np.add.at(acc_q.ravel(), flat[g], v[g] * v[g])
            print(f"    rows {rs}-{re}", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(acc_n > 0, acc_s / np.maximum(acc_n, 1), np.nan)
        var = np.where(acc_n > 1, acc_q / np.maximum(acc_n, 1) - mean ** 2, np.nan)
    std = np.sqrt(np.maximum(var, 0.0))
    mean = mean[::-1]
    std = std[::-1]
    print(f"  cells with data: {(acc_n>0).mean()*100:.1f}%, "
          f"median source pixels per cell {np.median(acc_n[acc_n>0]):.0f}")
    print(f"  elevation {np.nanmin(mean):.2f} .. {np.nanmax(mean):.2f} m")
    print(f"  sub-cell std {np.nanmin(std):.4f} .. {np.nanmax(std):.4f} m, "
          f"median {np.nanmedian(std):.3f} m")

    transform = Affine(res, 0, x0, 0, -res, y0 + ny * res)
    prof = dict(driver="GTiff", height=ny, width=nx, count=1, dtype="float32",
                crs="EPSG:3071", transform=transform, compress="LZW", nodata=np.nan)
    for path, data in ((a.out_mean, mean), (a.out_std, std)):
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(data.astype("float32"), 1)
        print(f"  wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
