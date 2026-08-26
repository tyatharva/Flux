#!/usr/bin/env python3
"""Is the 90-degree re-index EXACT? The library's x4 multiplier rests on this.

Fifteen seeds cover twelve headings only because a square doubly-periodic flat uniform
domain with dx = dy is exactly equivariant under a 90-degree rotation, so each spun-up
state re-indexes into four directions at no GPU cost. If that re-index were a resampling
rather than a permutation it would smear the turbulence it is supposed to be reusing, and
the cost of the library would be 4x what is budgeted.

Three things are checked, and the first is the one that matters:

  1. PERMUTATION, NOT INTERPOLATION. The multiset of values in each rotated 3-D field must
     be BIT-IDENTICAL to the original's. Sorting both and comparing exactly is a complete
     test of that -- any interpolation, any averaging, any change of dtype shows up
     immediately, and no tolerance has to be chosen.
  2. THE VECTOR ROTATES WITH THE GRID. Re-indexing alone would leave a westerly labelled
     as a westerly in a rotated frame. (u, v) -> (-v, u) per counter-clockwise turn, so the
     domain-mean wind direction must move by EXACTLY 90m degrees and its magnitude must be
     preserved to fp32 roundoff.
  3. ROUND TRIP. Four turns must return the original, bit for bit.

Second moments are reported alongside: TKE is a scalar under rotation and must be
invariant, which is the same claim Gate B6 makes dynamically (after 200 steps) and this
makes kinematically (at zero steps).

usage: rot_check.py <restart.nc> [--tmp runs/rotcheck]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

FIELDS3D = ("u", "v", "w", "theta", "TKE_0")


def prep(src, dst, rot):
    r = subprocess.run(["./docker/pyrun.sh", "bin/prep_restart.py", src, dst,
                        "--rot", str(rot), "--flat"], capture_output=True, text=True)
    if not os.path.exists(dst):
        print(r.stdout + r.stderr, file=sys.stderr)
        raise SystemExit(f"prep_restart --rot {rot} produced nothing")


def read(path):
    out = {}
    with Dataset(path) as ds:
        for v in FIELDS3D:
            if v in ds.variables:
                out[v] = np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
    return out


def wind(d):
    ub = d["u"].mean(axis=(-2, -1))
    vb = d["v"].mean(axis=(-2, -1))
    k = int(np.argmax(np.hypot(ub, vb)))
    return float(np.hypot(ub[k], vb[k])), float((270.0 - np.degrees(np.arctan2(vb[k], ub[k]))) % 360.0), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("restart")
    ap.add_argument("--tmp", default="runs/rotcheck")
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    base = os.path.join(a.tmp, "rot0.nc")
    shutil.copy(a.restart, base)
    d0 = read(base)
    sp0, dir0, k0 = wind(d0)
    print(f"=== 90-degree re-index check: {os.path.basename(a.restart)} ===")
    print(f"  reference: |U| {sp0:.6f} m/s FROM {dir0:.4f} deg at k={k0}\n")
    print(f"  {'rot':>4}  {'permutation exact':<20}{'|U|':>12}{'d|U|':>11}"
          f"{'dir':>10}{'turn':>9}{'TKE err':>11}")
    ok = True
    for m in (1, 2, 3):
        p = os.path.join(a.tmp, f"rot{m}.nc")
        prep(base, p, m)
        d = read(p)
        # 1. permutation: the sorted multiset must be bit-identical.
        #
        # SCALARS ONLY FOR A DIRECT COMPARE. u and v are the components of a VECTOR and
        # rotate WITH the grid -- (u, v) -> (-v, u) per counter-clockwise turn -- so
        # sorted(u_rot) matches sorted(-v_orig), not sorted(u_orig). Comparing u to u
        # reports every correct rotation as an interpolation, which is what the first
        # version of this check did.
        exp_uv = {1: (lambda D: -D["v"], lambda D: D["u"]),
                  2: (lambda D: -D["u"], lambda D: -D["v"]),
                  3: (lambda D: D["v"], lambda D: -D["u"])}[m]
        perm = all(np.array_equal(np.sort(d0[v], axis=None), np.sort(d[v], axis=None))
                   for v in ("w", "theta", "TKE_0") if v in d0 and v in d)
        perm &= np.array_equal(np.sort(exp_uv[0](d0), axis=None), np.sort(d["u"], axis=None))
        perm &= np.array_equal(np.sort(exp_uv[1](d0), axis=None), np.sort(d["v"], axis=None))
        sp, dr, _ = wind(d)
        turn = ((dir0 - dr + 180.0) % 360.0) - 180.0
        # 2. TKE is a scalar under rotation
        t0 = 0.5 * sum(float(((d0[v] - d0[v].mean(axis=(-2, -1), keepdims=True)) ** 2).mean())
                       for v in ("u", "v", "w"))
        t1 = 0.5 * sum(float(((d[v] - d[v].mean(axis=(-2, -1), keepdims=True)) ** 2).mean())
                       for v in ("u", "v", "w"))
        terr = abs(t1 - t0) / max(t0, 1e-30)
        # expected signed turn for m quarter-turns, wrapped to (-180, 180]
        exp = ((90.0 * m + 180.0) % 360.0) - 180.0
        turned = min(abs(turn - exp), abs(turn + exp)) < 1e-3
        print(f"  {m:>4}  {'YES bit-identical' if perm else 'NO -- INTERPOLATED':<20}"
              f"{sp:>12.6f}{sp-sp0:>+11.2e}{dr:>10.4f}{turn:>+9.2f}{terr:>11.2e}"
              f"  {'ok' if (perm and turned) else 'FAIL'}")
        ok &= perm and turned and abs(sp - sp0) < 1e-5 and terr < 1e-12
    # 3. round trip
    cur = base
    for m in range(4):
        nxt = os.path.join(a.tmp, f"rt{m}.nc")
        prep(cur, nxt, 1)
        cur = nxt
    d4 = read(cur)
    rt = all(np.array_equal(d0[v], d4[v]) for v in d0 if v in d4)
    print(f"\n  round trip (four 90-deg turns): {'BIT-FOR-BIT IDENTICAL' if rt else 'DIFFERS'}")
    ok &= rt
    print(f"\n  ROTATION CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
