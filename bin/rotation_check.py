#!/usr/bin/env python3
"""Is the 90-degree re-index of THIS seed artifact exact? A static, no-GPU gate.

The seed library is 5 rungs x 3 base angles = 15 states presenting 60 (state, direction)
options, and the x4 comes entirely from re-indexing a square, doubly periodic, FLAT,
UNIFORM spin-up. Gate B6 already showed the rotation is exact and that 200 steps of
FastEddy from a rotated state agree with the unrotated one -- but B6 ran on fifth-pass
spin-ups, and docs/reference/standing-rules.md's standing rule is to validate the state the model actually
LOADED. This scores the artifact a seed job actually returns, before 60 cases are built
on it, and it costs seconds rather than GPU time.

THE PRODUCTION FUNCTION IS IMPORTED, NEVER REIMPLEMENTED. bin/prep_restart.py owns the
index map; a gate carrying its own copy is the mistake stage4_wellmixed.py already made
with the sigma_w floor. Better still, the FILE prep_restart.py writes is compared against
that function's output, so the check covers the whole production path -- the copy, the
netCDF write, the dtype round trip -- and not just the arithmetic.

Four things are scored:

  1. FOUR TURNS ARE THE IDENTITY, bit-for-bit. Not "to a tolerance": the map is a pure
     permutation of indices, so anything but equality means the permutation is wrong.
  2. THE FILE MATCHES THE FUNCTION, bit-for-bit, for every rotated 3-D field.
  3. THE HORIZONTAL WIND VECTOR TURNS EXACTLY 90 DEGREES per turn: the slab mean must go
     (u, v) -> (-v, u) at every level. This is the part a pure re-index would get wrong if
     the vector components were not rotated with the grid, and it is invisible in any
     scalar diagnostic.
  4. SCALARS ARE INVARIANT. Slab mean and variance of theta, w and the sub-grid TKE are
     properties of a level, not of an orientation, so they must survive to fp64 roundoff.

usage: rotation_check.py <restart.nc> [--tmp runs/rotchk] [--json FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_restart import rotate_stack          # THE production index map, imported

# The 3-D prognostic/diagnostic fields a rotation must move. u and v are also rotated as
# VECTOR COMPONENTS, which is checked separately.
FIELDS3D = ("u", "v", "w", "theta", "TKE_0", "rho", "pressure")
VEC = ("u", "v")


def _get(ds, name):
    return np.squeeze(np.asarray(ds[name][:], dtype=np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--tmp", default="runs/rotchk")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if not os.path.exists(a.src):
        print(f"FATAL: no restart at {a.src}", file=sys.stderr)
        return 2
    os.makedirs(a.tmp, exist_ok=True)
    out, ok_all = [], True
    p = out.append
    p(f"=== 90-degree rotation check on {os.path.basename(a.src)} ===")

    with Dataset(a.src) as ds:
        present = [v for v in FIELDS3D if v in ds.variables]
        src = {v: _get(ds, v) for v in present}
    p(f"  fields present: {', '.join(present)}   shape {src[present[0]].shape}")
    for v, arr in src.items():
        if not np.isfinite(arr).all():          # inf is not NaN, and NaN passes every >
            print(f"FATAL: {v} in the source is not finite", file=sys.stderr)
            return 2

    # ---- 1. four turns are the identity, bit-for-bit --------------------------------
    bad = []
    for v, arr in src.items():
        r = arr
        for _ in range(4):
            r = rotate_stack(r, 1)
        if not np.array_equal(r, arr):
            bad.append(v)
    ok = not bad
    ok_all &= ok
    p(f"\n  1. four 90-deg turns == identity, bit-for-bit: "
      f"{'PASS' if ok else 'FAIL on ' + ', '.join(bad)}")

    # ---- 2/3/4. per rotation --------------------------------------------------------
    rows = []
    for m in (1, 2, 3):
        dst = os.path.join(a.tmp, f"FE_ROT{m}.0")
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "prep_restart.py"), a.src, dst,
               "--rot", str(m), "--flat"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(dst):
            p(f"\n  rot {m}: prep_restart.py FAILED\n{r.stderr[-600:]}")
            ok_all = False
            continue
        with Dataset(dst) as ds:
            got = {v: _get(ds, v) for v in present}

        # 2. the FILE against the FUNCTION, bit-for-bit
        want = {v: rotate_stack(src[v], m) for v in present}
        want["u"], want["v"] = (-rotate_stack(src["v"], m) if "v" in src else None,
                                rotate_stack(src["u"], m) if "u" in src else None)
        # m turns of (u,v) -> (-v,u) is a rotation by m*90 of the vector:
        c, s = int(round(np.cos(np.pi * m / 2))), int(round(np.sin(np.pi * m / 2)))
        want["u"] = c * rotate_stack(src["u"], m) - s * rotate_stack(src["v"], m)
        want["v"] = s * rotate_stack(src["u"], m) + c * rotate_stack(src["v"], m)
        diff = {v: int(np.count_nonzero(got[v] != want[v])) for v in present}
        exact = all(n == 0 for n in diff.values())
        ok_all &= exact

        # 3. the slab-mean wind vector turns exactly 90*m degrees
        um, vm = src["u"].mean(axis=(-2, -1)), src["v"].mean(axis=(-2, -1))
        gum, gvm = got["u"].mean(axis=(-2, -1)), got["v"].mean(axis=(-2, -1))
        wu, wv = c * um - s * vm, s * um + c * vm
        scale = max(float(np.abs(np.hypot(um, vm)).max()), 1e-30)
        dvec = float(np.abs(np.hypot(gum - wu, gvm - wv)).max()) / scale
        # THE COMPASS BEARING MOVES THE OTHER WAY, and the sign is what pick_seed.py's
        # rotation arithmetic depends on. Rotating the flow VECTOR 90 deg counter-clockwise
        # takes (u, v) -> (-v, u); compass bearings increase CLOCKWISE, so the bearing the
        # wind blows from DECREASES by 90 per turn. pick_seed.py says exactly this
        # ("SUBTRACT 90 per turn") and picks every corpus case on it, so it is checked here
        # against the artifact rather than left as a comment: want (-90*m) % 360.
        k = int(np.argmax(np.hypot(um, vm)))
        b0 = (np.degrees(np.arctan2(um[k], vm[k])) + 180.0) % 360.0     # blows FROM
        b1 = (np.degrees(np.arctan2(gum[k], gvm[k])) + 180.0) % 360.0
        dbear = float((b1 - b0) % 360.0)
        want_bear = float((-90 * m) % 360)
        ok_bear = abs(((dbear - want_bear + 180.0) % 360.0) - 180.0) < 1e-6
        ok_all &= ok_bear

        # 4. scalars are a property of the level, not of the orientation
        sc = {}
        for v in ("theta", "w", "TKE_0"):
            if v not in src:
                continue
            for lab, f in (("mean", lambda x: x.mean(axis=(-2, -1))),
                           ("var", lambda x: x.var(axis=(-2, -1)))):
                x, y = f(src[v]), f(got[v])
                sc[f"{v}.{lab}"] = float(np.abs(y - x).max()
                                         / max(float(np.abs(x).max()), 1e-30))
        worst = max(sc.values()) if sc else 0.0
        okv = dvec < 1e-13 and worst < 1e-13
        ok_all &= okv
        p(f"\n  rot {m}  ({90*m} deg CCW of the flow)")
        p(f"    2. file == rotate_stack(src), bit-for-bit: "
          f"{'PASS' if exact else 'FAIL ' + str(diff)}")
        p(f"    3. slab-mean wind vector: max relative departure from the exact turn "
          f"{dvec:.2e}")
        p(f"       the FROM bearing moved {dbear:+.6f} deg; pick_seed.py subtracts "
          f"{90*m} -> want {want_bear:.0f}: {'PASS' if ok_bear else 'FAIL'}")
        p(f"    4. scalar slab moments invariant: worst relative change {worst:.2e}"
          f"   ({', '.join(f'{k} {v:.1e}' for k, v in sorted(sc.items()))})")
        rows.append({"rot": m, "file_matches_function": exact, "n_differing": diff,
                     "vec_rel_departure": dvec, "bearing_shift_deg": dbear,
                     "bearing_shift_want_deg": want_bear, "bearing_ok": bool(ok_bear),
                     "scalar_worst_rel": worst,
                     "pass": bool(exact and okv and ok_bear)})
        os.remove(dst)

    p(f"\n  ROTATION CHECK: {'PASS' if ok_all else 'FAIL'}")
    txt = "\n".join(out)
    print(txt)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"src": os.path.abspath(a.src), "pass": bool(ok_all),
                   "identity_bitwise": ok, "rotations": rows},
                  open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
