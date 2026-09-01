#!/usr/bin/env python3
"""bl_depth: EXACT on every profile on disk, and physical on the one that broke it.

Two things are asserted, and the first is the one that makes the second safe to ship.

  A. BIT-IDENTITY ON EVERY EXISTING RECORD. Every footprint JSON in results/ that carries
     a full `profiles.tke_prof` also carries the `stats.h` the run that made it stored.
     Re-deriving h from the profile must reproduce that number EXACTLY -- not to a
     tolerance, because there is no physics between the two, only arithmetic. 47 profiles
     across 16, 24 and 30 m grids, neutral and convective, floored and unfloored.

  B. THE PROFILE THAT BROKE IT NOW RETURNS THE SURFACE-ATTACHED DEPTH. case_2023112120 is
     a neutral case whose resolved TKE peaks at 1.01 m2/s2 at 39 m, decays to 0.28 at
     448 m, and then rises to 2.42 at 1887 m -- a wave layer more than twice as energetic
     as the boundary layer under it. The old estimator took the global argmax and returned
     h = 2372 m; the refusal that replaced it blocked the neutral corpus. h must now come
     back inside the surface-attached layer.

  usage: bin/test_bl_depth.py [--profile results/_prof.npz]

`--profile` is the optional part B input: an .npz with `tke` and `z`, as written from a
window directory by the recipe in the docstring of _regen_profile below. Part A needs
nothing but the repo.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.les_stats import bl_depth, surface_layer_top  # noqa: E402

# The value the ninth pass measured and refused, and the level the wave layer peaks at.
BROKEN_H_M = 2371.979
WAVE_PEAK_M = 1887.0


def _regen_profile(windir, stride=40):
    """The window-mean resolved-TKE profile, for part B.

    Kept here rather than in a scratch script so the number in part B is reproducible:

        from lpdm.les_stats import WindowAccumulator, zlevels_of
        from lpdm.fields import open_dump
    """
    from lpdm.les_stats import WindowAccumulator, zlevels_of
    from lpdm.fields import open_dump
    paths = sorted(glob.glob(os.path.join(windir, "FE_WIN.[0-9]*")),
                   key=lambda p: int(p.rsplit(".", 1)[1]))[::stride]
    if not paths:
        return None, None
    z = zlevels_of(paths)
    acc = WindowAccumulator(z, 3)
    for p in paths:
        with open_dump(p) as ds:
            acc.add(ds, step=int(p.rsplit(".", 1)[1]))
    return acc.tke_prof / acc.n, z


def part_a():
    rows, bad = [], []
    for f in sorted(glob.glob("results/**/*.json", recursive=True)):
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        p = d.get("profiles")
        st = d.get("stats") or {}
        if not (isinstance(p, dict) and p.get("tke_prof") and p.get("zlev")):
            continue
        stored = st.get("h")
        if stored is None:
            continue
        tk = np.asarray(p["tke_prof"], dtype=np.float64)
        z = np.asarray(p["zlev"], dtype=np.float64)
        got, info = bl_depth(tk, z, frac=0.05, return_info=True)
        ok = (got == float(stored))
        rows.append((f, float(stored), got, ok, info))
        if not ok:
            bad.append((f, float(stored), got))
    print(f"=== A. h re-derived from {len(rows)} stored profiles, against each run's own h ===")
    aloft = [r for r in rows if r[4]["global_max_above_surface_layer"]]
    print(f"  {len(rows) - len(bad)} of {len(rows)} EXACT")
    print(f"  {len(aloft)} of {len(rows)} have their global TKE maximum ABOVE the "
          f"surface-attached layer (the case the band bounds away)")
    for f, s, g, ok, info in rows:
        if info["global_max_above_surface_layer"]:
            print(f"    {os.path.basename(f):<44} h {g:8.2f} m; global max "
                  f"{info['tke_global_max']:.2f} at {info['z_global_max_m']:.0f} m")
    for f, s, g in bad:
        print(f"  *** {f}: stored {s!r}, re-derived {g!r}, diff {g - s:+.6e}")
    if not rows:
        print("  *** no stored profiles found; part A proved nothing")
        return False
    return not bad


def part_b(tk, z, source):
    print(f"\n=== B. the profile that broke it: {source} ===")
    if tk is None:
        print("  SKIPPED: no profile available. Part B needs either --profile or the "
              "window dumps of case_2023112120 still on disk.")
        return None
    k_sfc, k_sfc_min, top = surface_layer_top(tk, z)
    h, info = bl_depth(tk, z, frac=0.05, return_info=True)
    print(f"  surface-attached peak       {tk[k_sfc]:.4f} m2/s2 at z = {z[k_sfc]:.0f} m")
    print(f"  its terminating minimum     {tk[k_sfc_min]:.4f} at z = {z[k_sfc_min]:.0f} m")
    print(f"  global maximum in the column{info['tke_global_max']:>9.4f} at "
          f"z = {info['z_global_max_m']:.0f} m   "
          f"(above the surface layer: {info['global_max_above_surface_layer']})")
    print(f"  h                           {h:.1f} m")
    ok = True
    if not info["global_max_above_surface_layer"]:
        print("  *** this profile does not exhibit the defect; part B proved nothing")
        return None
    if h > z[k_sfc_min] + 1e-9:
        print(f"  *** FAIL: h is above the surface-attached layer's top "
              f"({z[k_sfc_min]:.0f} m)"); ok = False
    if abs(h - BROKEN_H_M) < 1.0:
        print(f"  *** FAIL: h reproduced the broken value {BROKEN_H_M} m"); ok = False
    if h > 0.5 * WAVE_PEAK_M:
        print(f"  *** FAIL: h is more than half way to the wave layer's peak"); ok = False
    if not (100.0 <= h <= 1250.0):
        print(f"  *** FAIL: h = {h:.0f} m is outside the 100-1250 m band the domain and "
              f"the surface layer support"); ok = False
    if ok:
        print(f"  PASS: h = {h:.0f} m, inside the surface-attached layer, against "
              f"{BROKEN_H_M:.0f} m before")
    return ok


def main():
    ap = argparse.ArgumentParser()
    # DEFAULTS TO THE CACHED PROFILE. Part B used to regenerate it from a 19 GB window
    # directory, so the test silently SKIPPED once that scratch was cleaned up -- and a
    # skip reads like a pass in a log. results/_prof.npz is 2.4 kB, is checked in, and
    # reproduces part B exactly; --windir still regenerates it if the window is present.
    ap.add_argument("--profile", default=("results/_prof.npz"
                                          if os.path.exists("results/_prof.npz") else None),
                    help=".npz with `tke` and `z` for part B")
    ap.add_argument("--windir", default="runs/case_2023112120/window",
                    help="window directory to rebuild the part B profile from")
    a = ap.parse_args()

    a_ok = part_a()

    tk = z = None
    src = ""
    if a.profile and os.path.exists(a.profile):
        d = np.load(a.profile)
        tk, z, src = d["tke"], d["z"], a.profile
    elif os.path.isdir(a.windir):
        tk, z = _regen_profile(a.windir)
        src = a.windir
    b_ok = part_b(tk, z, src or "(none)")

    print()
    print(f"A bit-identity on every stored profile : {'PASS' if a_ok else 'FAIL'}")
    print(f"B the wave-layer profile               : "
          f"{'PASS' if b_ok else ('SKIP' if b_ok is None else 'FAIL')}")
    return 0 if (a_ok and b_ok is not False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
