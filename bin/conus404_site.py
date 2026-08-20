#!/usr/bin/env python3
"""Extract a stratified hourly CONUS404 sample AT THE TOWER, straight over HTTP.

WHAT THIS IS FOR, and what it is NOT for. CONUS404 is used here as a **climatology of the
site**, to set the ranges and the sampling density of the LES corpus -- how deep the
boundary layer gets, how large the surface heat flux gets, how the wind rose is shaped, how
often the site is unstable. It is NOT used to force any individual run: there is no
per-case sounding, no projection matching, no time-varying boundary condition. Each LES
case remains one idealised, quasi-stationary state; CONUS404 only says which states are
worth simulating and how many of each.

That distinction is what keeps the emulator honest. A footprint model conditioned on
Kljun's scalars must be trained across the range of those scalars the site actually
produces -- z_i above all, since Kljun takes it as an input and a constant z_i would make
that channel uninformative -- but the LES itself gains nothing from mesoscale forcing it
cannot sustain in a 4.5 km doubly-periodic box.

Source: HyTEST's CONUS404 zarr on the USGS Open Storage Network pod. Anonymous, no egress
charge, S3 API over plain HTTPS -- so this needs no cloud SDK and no credentials, just
urllib and a zstd decompressor.

Sampling: zarr chunks are 144 hours (6 days) x 175 x 175, so a chunk is the natural
atomic unit. Taking every `--stride`-th chunk gives 6 contiguous days out of every
6*stride, evenly spread across 45 water years -- every month of every year is represented,
and the diurnal cycle is complete within each block. Contiguity matters: the joint
distribution of (z_i, flux, wind) is strongly autocorrelated within a day, and a block
sample preserves that structure where an hour-by-hour random sample would not.

usage: conus404_site.py [--stride 10] [--out data/raw/conus404_site.npz]
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zstandard as zstd
from pyproj import CRS, Transformer

BASE = "https://usgs.osn.mghpcc.org/hytest/conus404/conus404_hourly.zarr"
TOWER_LON, TOWER_LAT = -89.292362, 42.957160
# Everything needed to reconstruct (z_i, w'theta', u*, U, direction, stability) at a point.
TVARS = ["PBLH", "ACSHFLSM", "U10", "V10", "Z", "TK", "PSFC"]
SVARS = ["COSALPHA", "SINALPHA", "HGT", "LANDMASK", "LU_INDEX", "lat", "lon"]


def fetch(key, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"{BASE}/{key}", timeout=180) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2.0 * (k + 1))


def chunk(meta, v, key):
    za = meta[v + "/.zarray"]
    n = int(np.prod(za["chunks"]))
    raw = fetch(key)
    d = zstd.ZstdDecompressor().decompress(raw, max_output_size=n * 8 + 4096)
    return np.frombuffer(d, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="data/raw/conus404_site.npz")
    a = ap.parse_args()

    cache = "/tmp/claude-1000/zmeta.json"
    if not os.path.exists(cache):
        with open(cache, "wb") as f:
            f.write(fetch(".zmetadata"))
    meta = json.load(open(cache))["metadata"]

    nt, ny, nx = meta["PBLH/.zarray"]["shape"]
    ct, cy, cx = meta["PBLH/.zarray"]["chunks"]
    xs = np.concatenate([chunk(meta, "x", f"x/{i}") for i in range(-(-nx // cx))])[:nx]
    ys = np.concatenate([chunk(meta, "y", f"y/{i}") for i in range(-(-ny // cy))])[:ny]
    crs = CRS.from_wkt(meta["crs/.zattrs"]["crs_wkt"])
    X, Y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(TOWER_LON, TOWER_LAT)
    ix = int(np.argmin(np.abs(xs - X))); iy = int(np.argmin(np.abs(ys - Y)))
    jy, jx = iy // cy, ix // cx
    ly, lx = iy % cy, ix % cx
    print(f"tower -> CONUS404 cell (y,x) = ({iy},{ix}), offset "
          f"({xs[ix]-X:+.0f}, {ys[iy]-Y:+.0f}) m; chunk ({jy},{jx}) local ({ly},{lx})")

    static = {}
    for v in SVARS:
        c = meta[v + "/.zarray"]["chunks"]
        static[v] = float(chunk(meta, v, f"{v}/{jy}.{jx}").reshape(c)[ly, lx])
    print("  static: " + "  ".join(f"{k}={v:.4f}" for k, v in static.items()))

    nct = -(-nt // ct)
    cts = list(range(0, nct, a.stride))
    tvals = np.concatenate([chunk(meta, "time", f"time/{i}")
                            for i in range(-(-nt // meta["time/.zarray"]["chunks"][0]))])[:nt]
    print(f"  {len(cts)} of {nct} time-chunks ({100*len(cts)/nct:.0f}%), "
          f"{len(cts)*ct:,} hourly samples, {len(cts)*len(TVARS)*15.5/1e3:.1f} GB to stream")

    out = {v: np.full((len(cts), ct), np.nan, dtype=np.float32) for v in TVARS}
    t0 = time.time()
    done = [0]

    def job(args):
        n, c = args
        for v in TVARS:
            sh = meta[v + "/.zarray"]["chunks"]
            # U/V here are unstaggered 10 m fields; Z/TK/PSFC are mass points. Every
            # variable in TVARS shares the (time, y, x) mass grid, so one local index
            # serves all of them.
            arr = chunk(meta, v, f"{v}/{c}.{jy}.{jx}").reshape(sh)
            out[v][n, :arr.shape[0]] = arr[:, ly, lx]
        done[0] += 1
        if done[0] % 20 == 0:
            el = time.time() - t0
            print(f"    {done[0]}/{len(cts)} chunks  {el/60:.1f} min  "
                  f"eta {el/done[0]*(len(cts)-done[0])/60:.1f} min", flush=True)

    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        list(ex.map(job, list(enumerate(cts))))

    tsel = np.concatenate([tvals[c*ct:(c+1)*ct] if (c+1)*ct <= nt else
                           np.pad(tvals[c*ct:], (0, (c+1)*ct-nt), constant_values=-1)
                           for c in cts])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, time_hours=tsel.astype(np.int64),
                        time_units="hours since 1979-10-01 00:00:00",
                        **{v: out[v].ravel() for v in TVARS},
                        **{f"static_{k}": v for k, v in static.items()},
                        iy=iy, ix=ix, stride=a.stride)
    print(f"  wrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB, "
          f"{len(tsel):,} hourly records, {(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
