"""Release / rotate / accumulate: turn a FieldSet into a wind-aligned flux footprint."""
from __future__ import annotations

import time

import numpy as np

from .footprint import FootprintGrid
from .les_stats import window_stats
from .model import LPDM


def receptor_indices(fs, z_target=30.0, x_frac=0.75, y_frac=0.5):
    k = int(np.argmin(np.abs(fs.zk - z_target)))
    i = int(round(x_frac * fs.nx))
    j = int(round(y_frac * fs.ny))
    return i, j, k


def make_releases(fs, n_per_release, t_first, t_last, dt_release, xr, yr, zr):
    times = np.arange(t_first, t_last + 1e-9, dt_release)
    t = np.repeat(times, n_per_release)
    n = len(t)
    return (np.full(n, xr), np.full(n, yr), np.full(n, zr), t, times)


def compute_footprint(fs, paths, z_target=30.0, n_per_release=700, dt_release=4.0,
                      t_back=900.0, c0=3.0, z_touch=2.0, grid_res=20.0,
                      grid_x=(-600.0, 4500.0), grid_y=(-1500.0, 1500.0),
                      seed=0, split_halves=True, batch_releases=12, verbose=True):
    """Release, integrate backward, rotate into the wind frame, accumulate.

    Releases are processed in batches of `batch_releases` release times rather than as one
    ensemble. The field cache is several GB and every integrator step does scattered
    4-D interpolation into it; a batch whose release times span ~1 minute touches a narrow
    slab of the time axis and stays resident, while one ensemble spanning the whole window
    touches all of it on every step. Same answer, far less memory traffic.
    """
    i_r, j_r, k_r = receptor_indices(fs, z_target)
    xr = fs.x0 + i_r * fs.dx
    yr = fs.y0 + j_r * fs.dy
    # ABSOLUTE height of that cell centre. Over terrain this is NOT fs.zk[k_r]: the
    # vertical coordinate is terrain-following, so the level sits at zg + ~z_target above
    # the local ground. Releasing at the flat-column height would put the receptor 20 m
    # underground on the hill.
    zr = float(fs.height(np.array([float(k_r)]), np.array([float(i_r)]),
                         np.array([float(j_r)]))[0])
    zg_r = float(fs.ground(np.array([float(i_r)]), np.array([float(j_r)]))[0])
    st = window_stats(paths, k_r)

    t_first = float(fs.t[0]) + t_back
    t_last = float(fs.t[-1])
    if t_last <= t_first:
        raise ValueError(f"window too short: need > {t_back:.0f} s of history before the "
                         f"first release (have {fs.t[-1]-fs.t[0]:.0f} s)")
    times = np.arange(t_first, t_last + 1e-9, dt_release)
    tmid = 0.5 * (times[0] + times[-1])
    if verbose:
        print(f"  receptor  (i,j,k)=({i_r},{j_r},{k_r})  x={xr:.0f} y={yr:.0f}  "
              f"z={zr:.3f} m ASL = {zr-zg_r:.3f} m AGL (ground {zg_r:.2f} m)")
        print(f"  releases  {len(times)} times over {t_first:.0f}-{t_last:.0f} s, "
              f"{n_per_release} each = {len(times)*n_per_release:,} particles; "
              f"t_back={t_back:.0f} s")

    ang = np.arctan2(st["V"], st["U"])
    ca, sa = np.cos(ang), np.sin(ang)
    mkgrid = lambda: FootprintGrid(grid_x[0], grid_x[1], grid_y[0], grid_y[1], grid_res)
    full, h1, h2 = mkgrid(), mkgrid(), mkgrid()

    lp = LPDM(fs, c0=c0, z_touch=z_touch, seed=seed)
    t_start = time.time()
    n_td = 0
    for b0 in range(0, len(times), batch_releases):
        tb = times[b0:b0 + batch_releases]
        n = len(tb) * n_per_release
        t = np.repeat(tb, n_per_release)
        res = lp.run(np.full(n, xr), np.full(n, yr), np.full(n, zr), t,
                     direction=-1, t_limit=t_back,
                     reflect_touchdown=True, record_touchdown=True)
        dx = res["td_x"] - xr
        dy = res["td_y"] - yr
        X = -(dx * ca + dy * sa)          # upwind-positive
        Y = -dx * sa + dy * ca
        r = dict(res); r["td_x"] = X; r["td_y"] = Y
        full.add(r, 0.0, 0.0)
        n_td += len(X)
        if split_halves:
            rel = t[res["td_particle"]]
            n_h1 = int((tb <= tmid).sum()) * n_per_release
            for g, m, nn in ((h1, rel <= tmid, n_h1), (h2, rel > tmid, n - n_h1)):
                rr = dict(res)
                rr["td_particle"] = res["td_particle"][m]
                rr["td_w"] = res["td_w"][m]
                rr["td_x"] = X[m]; rr["td_y"] = Y[m]
                rr["n"] = nn
                g.add(rr, 0.0, 0.0)
        if verbose:
            print(f"    batch {b0//batch_releases+1}/"
                  f"{-(-len(times)//batch_releases)}  {n_td:,} touchdowns  "
                  f"{time.time()-t_start:.0f} s", flush=True)

    out = dict(stats=st, grid=full, receptor=(xr, yr, zr), z_agl=zr - zg_r, k_recept=k_r,
               n_particles=full.n_particles, n_touchdown=n_td,
               wind_angle=float(np.degrees(ang)))
    if split_halves:
        out["halves"] = [h1, h2]
    return out
