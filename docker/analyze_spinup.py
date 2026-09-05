#!/usr/bin/env python3
"""Stage 2 gate analysis: TKE stationarity and resolved turbulence profiles.

Gate criteria (docs/history/overview.md Stage 2):
  1. Domain-averaged TKE reaches a plateau.
  2. Resolved w'w'(z) profile has the recognizable NBL/CBL validation shape.

Perturbations are deviations from the HORIZONTAL (x-y) mean at each level, which is
the standard LES decomposition for a horizontally homogeneous flat-terrain case.

usage: analyze_spinup.py <dump.nc> [<dump.nc> ...]
"""
import re
import sys

import numpy as np
from netCDF4 import Dataset


def step_of(path):
    m = re.search(r"\.(\d+)$", path)
    return int(m.group(1)) if m else -1


def profiles(path):
    with Dataset(path) as ds:
        z = np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))[:, 0, 0]
        out = {}
        for v in ("u", "v", "w"):
            a = np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
            out[v] = a - a.mean(axis=(-2, -1), keepdims=True)   # deviation from horiz mean
        sgs = np.squeeze(np.asarray(ds["TKE_0"][:], dtype=np.float64))
        uu = (out["u"] ** 2).mean(axis=(-2, -1))
        vv = (out["v"] ** 2).mean(axis=(-2, -1))
        ww = (out["w"] ** 2).mean(axis=(-2, -1))
        tke_res = 0.5 * (uu + vv + ww)
        # uw momentum flux -> friction velocity from the lowest level
        uw = (out["u"] * out["w"]).mean(axis=(-2, -1))
        vw = (out["v"] * out["w"]).mean(axis=(-2, -1))
        ustar = (uw[0] ** 2 + vw[0] ** 2) ** 0.25
        return dict(z=z, ww=ww, uu=uu, vv=vv, tke_res=tke_res,
                    sgs=sgs.mean(axis=(-2, -1)), ustar=ustar,
                    tke_col=float(tke_res.mean()), sgs_col=float(sgs.mean()))


def main(paths):
    paths = sorted(paths, key=step_of)
    rows = []
    for p in paths:
        d = profiles(p)
        rows.append((step_of(p), d))
        print(f"  step {step_of(p):>7}  t={step_of(p)*0.0275/60:7.2f} min   "
              f"TKE_res={d['tke_col']:.5f}  TKE_sgs={d['sgs_col']:.5f}  "
              f"u*={d['ustar']:.4f} m/s  max_ww={d['ww'].max():.5f}")

    if len(rows) >= 4:
        vals = np.array([r[1]["tke_col"] for r in rows])
        last = vals[-4:]
        drift = (last[-1] - last[0]) / max(abs(last.mean()), 1e-30)
        print(f"\n  TKE over last 4 dumps: {np.array2string(last, precision=5)}")
        print(f"  relative drift across them: {drift*100:+.2f}%")
        print(f"  PLATEAU: {'YES' if abs(drift) < 0.05 else 'NO — still developing'}"
              f"  (criterion |drift| < 5%)")

    d = rows[-1][1]
    z, ww, ustar = d["z"], d["ww"], d["ustar"]
    print(f"\n  === resolved w'w'(z) at final dump, normalized by u*^2 (u*={ustar:.4f}) ===")
    n = ww / max(ustar ** 2, 1e-30)
    kpk = int(np.argmax(ww))
    for k in range(0, min(len(z), 60), 4):
        bar = "#" * int(min(n[k], 3.0) / 3.0 * 46)
        print(f"   z={z[k]:7.1f} m  ww/u*^2={n[k]:6.3f} |{bar}")
    print(f"\n  peak w'w' at z={z[kpk]:.1f} m  (ww/u*^2={n[kpk]:.3f})")
    inbl = z < 800
    if inbl.sum():
        print(f"  BL-depth proxy: highest z with ww/u*^2 > 0.1 = "
              f"{z[np.where(n > 0.1)[0][-1]] if (n>0.1).any() else float('nan'):.0f} m")


if __name__ == "__main__":
    main(sys.argv[1:])
