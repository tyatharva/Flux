#!/usr/bin/env python3
"""The parallel LPDM must be BIT-IDENTICAL to the serial one, not merely similar.

Worker count is a performance knob and nothing else. The ensemble is always cut into the
same chunks with the same per-chunk seeds, so 1 worker and 16 must produce the same
trajectories, the same touchdowns and the same footprint -- otherwise every result carries
a silent dependence on how busy the machine was.

usage: test_parallel_lpdm.py <windowdir> --dt DT
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.fields import FieldSet, dump_series
from lpdm.model import LPDM

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windir")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    paths = dump_series(a.windir)[::12]
    fs = FieldSet(paths, a.dt, verbose=False, cache_dtype=np.float16)
    print(f"  {len(paths)} dumps, cache {fs.mem_gb:.2f} GB, {a.n} particles")

    rng = np.random.default_rng(3)
    x = rng.uniform(fs.x0, fs.x0 + fs.Lx, a.n)
    y = rng.uniform(fs.y0, fs.y0 + fs.Ly, a.n)
    z = rng.uniform(2.0, 800.0, a.n)
    t0 = float(fs.t[-1])

    res = {}
    for w in (1, a.workers):
        lp = LPDM(fs, seed=11)
        t = time.time()
        res[w] = lp.run(x, y, z, t0, direction=-1, t_limit=300.0,
                        reflect_touchdown=True, record_touchdown=True, n_workers=w)
        res[w]["_wall"] = time.time() - t
        print(f"  {w:2d} worker(s): {res[w]['_wall']:6.1f} s, "
              f"{len(res[w]['td_x']):,} touchdowns")

    a1, a2 = res[1], res[a.workers]
    for k in ("td_x", "td_y", "td_w", "td_t", "td_particle", "x", "y", "z", "w_release"):
        same = (len(a1[k]) == len(a2[k])) and bool(np.array_equal(
            np.nan_to_num(a1[k], nan=-1e30), np.nan_to_num(a2[k], nan=-1e30)))
        check(f"{k} identical", same,
              f"{len(a1[k])} vs {len(a2[k])}" if len(a1[k]) != len(a2[k]) else "")
    check("particle counts identical", a1["n"] == a2["n"])
    # td_particle must index the FULL ensemble, not a chunk. If the offset were missing
    # the indices would still be in range and the error would be silent.
    check("td_particle spans the whole ensemble",
          int(a2["td_particle"].max()) > a.n // 2,
          f"max index {int(a2['td_particle'].max())} of {a.n}")
    sp = a1["_wall"] / max(a2["_wall"], 1e-9)
    print(f"\n  speed-up {sp:.1f}x on {a.workers} workers "
          f"({a1['_wall']:.0f} s -> {a2['_wall']:.0f} s)")
    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: " + "; ".join(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
