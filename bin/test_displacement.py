#!/usr/bin/env python3
"""Focused checks on the displacement-height and receptor-height changes.

Three properties, each of which would be expensive to discover from a footprint:

  1. WITH NO DISPLACEMENT MAP, NOTHING CHANGES. The sub-layer log law gains a `- d` and a
     floor on its anchor; with d = 0 and any z0 below ~2 m both reduce to the expression
     that was there before, and the result must be bit-identical, not merely close.
  2. window_stats on an INTEGER level is the old slice. The fractional path is new and
     runs unconditionally, so it has to collapse exactly onto a[k] when the level is whole,
     or every previous result silently moves.
  3. The fractional receptor lands where it is asked to. On the production grid a cell
     centre sits at exactly 10.000000 m, so exact_agl must return k = 2 to roundoff over
     flat ground -- and over RAISED ground it must return the fractional level that keeps
     the receptor a fixed height above BARE ground, which is the whole reason it exists.

usage: test_displacement.py [<flat_dump_dir>]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.driver import receptor_indices
from lpdm.fields import FieldSet, dump_series
from lpdm.les_stats import window_stats
from lpdm.model import LPDM

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def main(outdir):
    paths = dump_series(outdir)[:2]
    fs = FieldSet(paths, 0.0328947, verbose=False)
    print(f"  {fs.nx} x {fs.ny} x {fs.nz}, levels {fs.zk[0]:.4f} {fs.zk[1]:.4f} "
          f"{fs.zk[2]:.4f} m, ground {fs.zg.min():.2f}..{fs.zg.max():.2f} m\n")

    # ---- 1. d = 0 must be bit-identical ------------------------------------------------
    rng = np.random.default_rng(0)
    n = 4000
    x = fs.x0 + rng.uniform(0, fs.Lx, n)
    y = fs.y0 + rng.uniform(0, fs.Ly, n)
    # deliberately BELOW the first level, which is the only branch that changed
    z = rng.uniform(0.2, float(fs.zk[0]) * 0.98, n)
    t = np.full(n, float(fs.t[0]))
    lp = LPDM(fs, seed=1)
    check("the sub-layer branch is actually exercised",
          bool((np.clip(z - fs.ground(*fs.hindex(x, y)), lp.z_touch * 0.5, None)
                < lp.z_ref).any()))
    base = lp._local(x, y, z, t)
    fs.set_displacement(np.zeros((fs.ny, fs.nx)))
    zero = lp._local(x, y, z, t)
    same = all(np.array_equal(a, b) for a, b in zip(base, zero))
    check("d = 0 map is bit-identical to no map", same)

    # ---- and a real d must move the wind, downward, monotonically ----------------------
    fs.set_displacement(np.full((fs.ny, fs.nx), 1.0))
    withd = lp._local(x, y, z, t)
    u0, u1 = base[0], withd[0]
    moved = np.abs(u1 - u0) > 0
    check("a non-zero d changes the sub-layer wind", bool(moved.any()),
          f"{100*moved.mean():.0f}% of particles")
    # d raises (z-d)/z0 toward 1 from above -> ln shrinks -> the continued wind WEAKENS
    sgn = np.sign(u0)
    check("displacement weakens the sub-layer wind (|u| decreases)",
          bool(np.all(np.abs(u1[moved]) <= np.abs(u0[moved]) + 1e-12)),
          f"max increase {np.max(np.abs(u1[moved])-np.abs(u0[moved])):.3e}")
    check("no NaN or inf anywhere with d in play",
          all(np.isfinite(a).all() for a in withd))
    # a particle INSIDE the canopy (zagl < d) must still be finite
    fs.set_displacement(np.full((fs.ny, fs.nx), float(fs.zk[0]) * 1.5))
    deep = lp._local(x, y, z, t)
    check("d above the first level is still finite (canopy-interior particles)",
          all(np.isfinite(a).all() for a in deep))
    fs.set_displacement(None)

    # ---- 2. window_stats on a whole level is the old slice -----------------------------
    k = 2
    st_i = window_stats(paths, k)
    st_f = window_stats(paths, float(k))
    same = all(np.allclose(st_i[q], st_f[q], rtol=0, atol=0)
               for q in ("u_mean", "sigma_v", "sigma_w", "z_recept", "e_sgs"))
    check("window_stats(int k) == window_stats(float k)", same)
    st_h = window_stats(paths, k + 0.5)
    st_k1 = window_stats(paths, k + 1)
    mid = 0.5 * (st_i["z_recept"] + st_k1["z_recept"])
    check("fractional level interpolates the receptor height",
          abs(st_h["z_recept"] - mid) < 1e-9,
          f"{st_h['z_recept']:.6f} vs midpoint {mid:.6f}")
    between = min(st_i["u_mean"], st_k1["u_mean"]) - 1e-9 <= st_h["u_mean"] \
        <= max(st_i["u_mean"], st_k1["u_mean"]) + 1e-9
    check("fractional level's mean wind lies between its neighbours", bool(between),
          f"{st_i['u_mean']:.4f} < {st_h['u_mean']:.4f} < {st_k1['u_mean']:.4f}")

    # ---- 3. the fractional receptor lands where asked ----------------------------------
    zt = float(fs.zk[2])
    i, j, kk = receptor_indices(fs, zt, ij=(fs.nx // 2, fs.ny // 2), exact_agl=True)
    zg = float(fs.ground(np.array([float(i)]), np.array([float(j)]))[0])
    zz = float(fs.height(np.array([kk]), np.array([float(i)]), np.array([float(j)]))[0])
    check("exact_agl reproduces a cell centre over its own ground",
          abs((zz - zg) - zt) < 1e-6, f"k = {kk:.6f}, z-zg = {zz-zg:.9f} vs {zt:.9f}")
    # now RAISE the ground under that column and demand the receptor stay put
    fs.zg = fs.zg.copy()
    fs.zg[j, i] += 1.5
    i2, j2, k2 = receptor_indices(fs, zt, ij=(i, j), exact_agl=True)
    zg2 = float(fs.ground(np.array([float(i2)]), np.array([float(j2)]))[0])
    z2 = float(fs.height(np.array([k2]), np.array([float(i2)]), np.array([float(j2)]))[0])
    check("over raised ground the receptor holds its height ABOVE THAT GROUND",
          abs((z2 - zg2) - zt) < 1e-3, f"k = {k2:.4f} (was {kk:.4f}), "
          f"z-zg = {z2-zg2:.6f}")
    check("and the fractional level actually moved", abs(k2 - kk) > 1e-6,
          f"{kk:.4f} -> {k2:.4f}")

    print(f"\n  {'ALL CHECKS PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/g24_flat/output"))
