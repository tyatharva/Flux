#!/usr/bin/env python3
"""Stage 2 gate: TKE stationarity + profile comparison against NCAR's NBL validation case.

Reference values read off the published NBL tutorial figures
(docs/Tutorials/images/{TURB,MEAN}-PROF-neutral.png), which show the state at t=6-7 h:

  sigma_w^2 : ~0 at the surface, RISING to a peak ~0.125 m2/s2 at z ~ 130 m,
              decaying to ~0 by z ~ 650 m
  <U'W'>    : ~0.17 m2/s2 at the surface  ->  u* ~ 0.41 m/s
  => sigma_w^2 peak / u*^2  ~  0.73
  U(z)      : ~4.3 m/s at the first level, reaching the 10 m/s geostrophic value by ~500 m
  phi(z)    : veers ~25 deg (245 -> 270) between surface and free stream
  theta(z)  : 300 K constant to ~490 m, inversion to ~312 K by ~680 m

NOISY_LEVELS excludes the lowest levels of resolved w from the peak search. It is 0:
the dt correction removed the near-surface w artifact (docker/diag_near_surface.py),
and the standing k0/k1 check in docker/check_run.sh now catches it if it returns.

usage: stage2_gate.py <dump.nc> [<dump.nc> ...]
"""
import sys

import numpy as np
from netCDF4 import Dataset

DT = float(__import__("os").environ.get("FE_DT", "0.0625"))  # override with FE_DT
NOISY_LEVELS = 0          # dt fix removed the near-surface w artifact; no exclusion needed
REF_PEAK_OVER_USTAR2 = 0.73
REF_PEAK_Z = 130.0


def load(path):
    with Dataset(path) as ds:
        g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
        z = g("zPos")[:, 0, 0]
        u, v, w, th = g("u"), g("v"), g("w"), g("theta")
        ustar = float(g("fricVel").mean())
    prime = lambda a: a - a.mean(axis=(-2, -1), keepdims=True)
    ww = (prime(w) ** 2).mean(axis=(-2, -1))
    tke = 0.5 * ((prime(u) ** 2) + (prime(v) ** 2) + (prime(w) ** 2)).mean(axis=(-2, -1))
    U = u.mean(axis=(-2, -1))
    V = v.mean(axis=(-2, -1))
    return dict(z=z, ww=ww, tke=tke, ustar=ustar,
                spd=np.hypot(U, V),
                phi=(270.0 - np.degrees(np.arctan2(V, U))) % 360.0,
                th=th.mean(axis=(-2, -1)))


def main(paths):
    paths = sorted(paths, key=lambda p: int(p.split(".")[-1]))
    hist = []
    print("  --- TKE stationarity ---")
    print(f"  {'step':>8} {'t(min)':>8} {'TKE_col':>10} {'u*':>8}")
    for p in paths:
        st = int(p.split(".")[-1])
        d = load(p)
        col = float(d["tke"].mean())
        hist.append((st, col, d))
        print(f"  {st:>8} {st*DT/60:8.2f} {col:10.5f} {d['ustar']:8.4f}")

    if len(hist) >= 4:
        v = np.array([h[1] for h in hist[-4:]])
        # Per-dump growth rate is far less window-sensitive than an endpoint drift:
        # a 4-dump endpoint difference swings wildly as the window slides over noise.
        rate = (v[-1] / max(v[0], 1e-30)) ** (1.0 / (len(v) - 1)) - 1.0
        drift = (v[-1] - v[0]) / max(abs(v.mean()), 1e-30)
        print(f"\n  TKE last 4 dumps: {np.array2string(v, precision=5)}")
        print(f"  growth per dump: {rate*100:+.2f}%   endpoint drift: {drift*100:+.2f}%")
        print(f"  PLATEAU: {'YES' if abs(rate) < 0.01 else 'NO'} "
              f"(criterion |growth per dump| < 1%)")

    d = hist[-1][2]
    z, ww, us = d["z"], d["ww"], d["ustar"]
    k = NOISY_LEVELS + int(np.argmax(ww[NOISY_LEVELS:]))
    print(f"\n  --- vs NBL validation (t=6-7 h reference) ---")
    print(f"  {'quantity':<34} {'ours':>12} {'NBL ref':>12}")
    print(f"  {'sigma_w^2 peak / u*^2':<34} {ww[k]/us**2:12.3f} {REF_PEAK_OVER_USTAR2:12.3f}")
    print(f"  {'height of sigma_w^2 peak (m)':<34} {z[k]:12.0f} {REF_PEAK_Z:12.0f}")
    print(f"  {'u* (m/s)':<34} {us:12.3f} {0.41:12.3f}")
    print(f"  {'wind speed, first level (m/s)':<34} {d['spd'][0]:12.2f} {4.3:12.2f}")
    print(f"  {'wind veering, sfc->free (deg)':<34} "
          f"{(d['phi'][0]-d['phi'][min(len(z)-1,60)]+540)%360-180:12.1f} {-25.0:12.1f}")
    hi = np.where(ww > 0.05 * ww[k])[0]
    print(f"  {'sigma_w^2 -> 0 by (m)':<34} {z[hi[-1]] if len(hi) else np.nan:12.0f} {650:12.0f}")

    print(f"\n  --- sigma_w^2 / u*^2 profile ---")
    n = ww / max(us**2, 1e-30)
    for j in list(range(0, 40, 2)) + [44, 50, 56, 62]:
        if j >= len(z):
            break
        flag = "  <-- artifact" if j < NOISY_LEVELS else ""
        bar = "#" * int(min(n[j], 1.2) / 1.2 * 44)
        print(f"   z={z[j]:7.1f} m  {n[j]:7.3f} |{bar}{flag}")


if __name__ == "__main__":
    main(sys.argv[1:])
