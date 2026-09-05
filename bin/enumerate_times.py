#!/usr/bin/env python3
"""Every candidate hour at the tower, screened. Stage 0 of corpus time selection.

WHY ENUMERATE RATHER THAN RETRY. The corpus takes at most one case per day, and a day's
midday hour is accepted only ~71% of the time. Picking midday and retrying elsewhere when
it fails is neither deterministic nor reproducible, and it silently biases WHICH times
survive. Enumerating every hour first and choosing from the valid set is both, and it makes
the choice a COVERAGE decision instead of an accident (`bin/select_times.py`).

=== HRRR ANALYSES ARE HOURLY, SO THERE ARE 24 CANDIDATES PER DAY, NOT 48 ===

The tower's averaging periods are half-hourly and a footprint is stamped period-ending, so
a day holds 48 of them. But the forcing for a case comes from **the HRRR analysis whose
valid time equals the footprint timestamp** (docs/les/case-generation.md), and HRRR analyses run hourly. A
:30 timestamp has no analysis behind it. HRRR's `subh` product carries 15-minute FORECAST
output, but the pseudo-sounding needs the `nat` hybrid-level profile, which is hourly-only
-- so the case timestamps are hourly whatever the surface cadence is. This is a property of
the data source, not a choice. 24x is still a 24-fold larger pool than one midday.

=== WHAT IS SCREENED, AND WHAT IT COSTS ===

Four fields, not six: `HPBL` for depth, `SHTFL` for the stability sign, and the 10 m wind
for direction. `LHTFL` and `PRES` are dropped -- they are needed for the Bowen conversion,
which `bin/hrrr_sounding.py` does for the SELECTED time only, and they are a third of the
download. GRIB byte-range subsetting is per MESSAGE, so each field is a full CONUS grid at
~1.9 MB whatever you do with it: **~7.5 MB per hour**, and the GRIB is deleted after the
one gridpoint is read.

A full five-year enumeration is therefore ~330 GB of transfer over ~20 h. That is a
run-once cost and it is stated here so nobody starts it by accident. `--stride` samples
every Nth day for a statistically useful subset at a fraction of it.

=== dz_i/dt IS SCREENED SEPARATELY FROM z_i, AND THAT IS THE POINT ===

Widening the acceptable z_i band pushes selection towards morning and evening, because
those are the hours when a day whose midday is too deep still has a valid z_i. They are
also exactly when z_i changes fastest, so a naive widening trades a domain violation for a
STATIONARITY violation and reports neither. The growth rate is therefore computed per hour
from the day's own z_i series and screened on its own threshold, independent of the z_i
value -- see `--max-dzidt-rel`.

usage: enumerate_times.py 2023-01-01 2023-12-31 [--stride 4] [--out results/candidates.tsv]
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TOWER_LAT = 42.957160
TOWER_LON = -89.292362
RD, CP = 287.05, 1004.5

# === THE SCREEN DOES NOT USE THE WIND, SO THE SCREEN DOES NOT FETCH IT ==================
# MEASURED against the real archive (2023-07-15 19Z): the four-message pattern is 9.19 MB
# and the two the screen actually reads are 4.66 MB. GRIB byte-range subsetting is per
# MESSAGE and each message is a full CONUS field (1799x1059) whatever a caller wants from
# it, so a field nobody reads costs its whole 2.3 MB.
#
# It matters because of WHERE the fetches are. An accepted day draws ~1.5 hours and costs
# ~4.5 fetches, but a day that exhausts its pool costs ~26 -- so the screening term is
# dominated by the days that yield nothing, and halving it takes ~8.8 GB off a machine.
#
# lpdm/corpus.py:screen() takes hpbl, shtfl and dz_i/dt and nothing else. The 10 m wind is
# recorded on the ACCEPTED hour only, as a label, so it is fetched once at acceptance
# instead of on every candidate.
SCREEN = r":(?:HPBL|SHTFL):surface:"
SCREEN_WIND = r":[UV]GRD:10 m above ground:"
# cfgrib names, in the order we try them
NAMEMAP = {"blh": "hpbl", "hpbl": "hpbl", "HPBL": "hpbl",
           "ishf": "shtfl", "sshf": "shtfl", "SHTFL": "shtfl",
           "u10": "u10", "UGRD": "u10", "v10": "v10", "VGRD": "v10"}


def screen_hour(ts, save_dir, keep_grib=False, with_wind=True):
    """The screening fields at the tower for one analysis hour.

    `with_wind=False` fetches only what lpdm/corpus.py:screen() reads -- 4.66 MB instead of
    9.19 -- and returns no u10/v10. The hour draw uses that for every CANDIDATE and asks
    for the wind once, on the hour it accepts.
    """
    from herbie import Herbie
    import pandas as pd
    from hrrr_sounding import crs_of, meridian_convergence, rotate_to_earth

    H = Herbie(ts, model="hrrr", product="sfc", fxx=0, save_dir=save_dir)
    ds = H.xarray(SCREEN + ("|" + SCREEN_WIND if with_wind else ""),
                  remove_grib=not keep_grib)
    dss = list(ds) if isinstance(ds, (list, tuple)) else [ds]
    pts = pd.DataFrame({"latitude": [TOWER_LAT], "longitude": [TOWER_LON]})
    out, crs = {}, None
    for d in dss:
        if crs is None:
            try:
                crs = crs_of(d)
            except Exception:
                pass
        p = d.herbie.pick_points(pts, method="nearest", k=1)
        for v in p.variables:
            key = NAMEMAP.get(v)
            if key:
                out[key] = float(np.asarray(p[v].values).ravel()[0])
    need = {"hpbl", "u10", "v10"} if with_wind else {"hpbl"}
    if not need <= set(out):
        raise KeyError(f"screen missing {need} - set(out) at {ts}")
    if with_wind:
        # WINDS ARE GRID-RELATIVE. 5.11 deg here, invisible in the speed, and it is most of
        # a direction bin -- which is the axis this whole selection is stratifying on.
        gamma = meridian_convergence(crs, TOWER_LON, TOWER_LAT) if crs is not None else 0.0
        u, v = rotate_to_earth(out["u10"], out["v10"], gamma)
        out["u10"], out["v10"] = float(u), float(v)
        # INSIDE the guard: u, v and gamma only exist on this branch, and these three lines
        # sat outside it -- a NameError on the first wind-free screen, i.e. on every
        # candidate hour of every day, on the rented box.
        out["wdir"] = float((270.0 - np.degrees(np.arctan2(v, u))) % 360.0)
        out["wspd"] = float(np.hypot(u, v))
        out["gamma"] = float(gamma)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--stride", type=int, default=1, help="sample every Nth day")
    ap.add_argument("--hours", default="0-23", help="e.g. 0-23 or 6,9,12,15,18")
    ap.add_argument("--cache", default="data/hrrr")
    ap.add_argument("--out", default="results/candidates.tsv")
    ap.add_argument("--keep-grib", action="store_true")
    a = ap.parse_args()

    if "-" in a.hours and "," not in a.hours:
        lo, hi = (int(x) for x in a.hours.split("-"))
        hours = list(range(lo, hi + 1))
    else:
        hours = [int(x) for x in a.hours.split(",")]

    d0 = dt.date.fromisoformat(a.start)
    d1 = dt.date.fromisoformat(a.end)
    days = []
    d = d0
    while d <= d1:
        days.append(d)
        d += dt.timedelta(days=a.stride)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # Append-and-resume: the full run is ~20 h and must survive a kill.
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            f = ln.rstrip("\n").split("\t")
            if len(f) > 1 and f[0] != "date":
                done.add((f[0], f[1]))
    else:
        with open(a.out, "w") as f:
            f.write("date\thour\tzi_m\tdzidt_m_per_h\tdzidt_rel_per_h\tshtfl_wm2"
                    "\twdir_deg\twspd_ms\n")

    print(f"{len(days)} day(s) x {len(hours)} hour(s) = {len(days)*len(hours)} analyses"
          f"  (stride {a.stride}); {len(done)} already on disk")
    nf = 0
    for d in days:
        rows = {}
        for h in hours:
            key = (d.isoformat(), f"{h:02d}")
            ts = dt.datetime(d.year, d.month, d.day, h)
            if key in done:
                continue
            try:
                rows[h] = screen_hour(ts, a.cache, a.keep_grib)
            except Exception as e:
                nf += 1
                print(f"  {d} {h:02d}Z: {type(e).__name__}: {e}", file=sys.stderr)
        if not rows:
            continue
        # dz_i/dt from the DAY'S OWN series, centred where both neighbours exist. A
        # one-sided difference at the ends is still a rate; a missing neighbour is not.
        hs = sorted(rows)
        zs = np.array([rows[h]["hpbl"] for h in hs], float)
        if len(hs) >= 2:
            grad = np.gradient(zs, np.array(hs, float))
        else:
            grad = np.array([np.nan])
        with open(a.out, "a") as f:
            for i, h in enumerate(hs):
                r = rows[h]
                g = float(grad[i])
                rel = g / max(r["hpbl"], 1.0)
                f.write(f"{d.isoformat()}\t{h:02d}\t{r['hpbl']:.1f}\t{g:+.1f}\t"
                        f"{100*rel:+.1f}\t{r.get('shtfl', float('nan')):.1f}\t"
                        f"{r['wdir']:.1f}\t{r['wspd']:.2f}\n")
        print(f"  {d}: {len(hs)} hours, z_i {zs.min():.0f}-{zs.max():.0f} m")
    if nf:
        print(f"\n  {nf} analysis hour(s) failed to fetch; they are simply absent from "
              f"{a.out} and the selector treats them as unavailable")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
