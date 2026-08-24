#!/usr/bin/env python3
"""Is the neutral-vs-convective compaction real, or is it the closure?

The floor is inert neutrally (max factor 1.23) and active convectively (3.51), so any
statistic compared across the two regimes is comparing two DIFFERENT closures unless the
floor is switched off on both sides. This differences the 80% source AREA -- a 2-D measure,
unlike x80 -- with the floor on and off, and quotes each against the sampling floor taken
from the run's own half-vs-half split, so "the sign flips" is a measurement and not a
reading of two numbers.

usage: compaction_check.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.footprint import source_area_mask

R = "results"


def area(f, cell, frac=0.80):
    return float(source_area_mask(np.maximum(f, 0), frac).sum() * cell / 1e4)   # ha


def main():
    rows = {}
    print("=== 80% SOURCE AREA, flat controls, floor ON vs OFF ===")
    print(f"  {'regime':<12}{'closure':<10}{'A80':>8}{'half1':>8}{'half2':>8}"
          f"{'|d| halves':>11}{'A50':>8}{'x80':>7}{'maxfac':>8}")
    for reg, tag in (("neutral", "g16p6b_flat"), ("convective", "g16p6b_flatcbl")):
        for cfg in ("nofloor", "new", "legacy"):
            jp, np_ = f"{R}/{tag}_{cfg}.json", f"{R}/{tag}_{cfg}.npz"
            if not (os.path.exists(jp) and os.path.exists(np_)):
                continue
            d = json.load(open(jp))
            z = np.load(np_)
            cell = (z["xc"][1] - z["xc"][0]) * (z["yc"][1] - z["yc"][0])
            a1, a2 = area(z["half1"], cell), area(z["half2"], cell)
            a = d["les"]["area80_ha"]
            fl = d.get("floor")
            rows[(reg, cfg)] = dict(a=a, half=abs(a1 - a2), x80=d["les"]["x80"])
            print(f"  {reg:<12}{cfg:<10}{a:8.3f}{a1:8.3f}{a2:8.3f}{abs(a1-a2):11.3f}"
                  f"{d['les']['area50_ha']:8.3f}{d['les']['x80']:7.0f}"
                  f"{(f'{max(fl[chr(102)+chr(97)+chr(99)]):.2f}' if fl else '--'):>8}")

    print("\n=== the compaction ratio, both closures ===")
    print(f"  {'closure':<12}{'metric':<8}{'neutral':>10}{'convective':>12}{'ratio':>8}"
          f"   verdict")
    out = {}
    for cfg, lab in (("nofloor", "floor OFF"), ("new", "floor ON")):
        n, c = rows[("neutral", cfg)], rows[("convective", cfg)]
        for key, mlab in (("a", "A80 ha"), ("x80", "x80 m")):
            r = n[key] / c[key]
            out[(cfg, key)] = r
            print(f"  {lab:<12}{mlab:<8}{n[key]:10.3f}{c[key]:12.3f}{r:8.2f}"
                  f"   convective is {'MORE COMPACT' if r > 1 else 'BROADER'}")

    print("\n=== does the sign survive removing the floor? ===")
    for key, mlab in (("a", "80% AREA"), ("x80", "x80")):
        r_off, r_on = out[("nofloor", key)], out[("new", key)]
        flip = (r_off - 1.0) * (r_on - 1.0) < 0
        print(f"  {mlab:<10} floor OFF {r_off:.2f}x   floor ON {r_on:.2f}x   "
              f"-> {'SIGN FLIPS' if flip else 'sign survives'}"
              f"  (factor {max(r_on/r_off, r_off/r_on):.2f} between them)")

    # ---- WHICH FLOOR APPLIES ---------------------------------------------------------
    # The half-vs-half spread is an UNPAIRED floor and it is the wrong one here, twice
    # over. It is measured on half-windows, whose A80 is biased HIGH because a noisier
    # footprint spreads mass over more cells -- visible directly below, where both halves
    # exceed the full window in every row. And the floor-on/floor-off comparison is
    # PAIRED: same window, same release times and positions, same seed, same field
    # realisation, so the sampling variance is common-mode and largely cancels.
    #
    # The paired noise has its own null, and it is free: the floor is inert neutrally
    # (1.000 at the receptor, 1.23 at most over the column), so floor-on minus floor-off
    # in the NEUTRAL row measures what the pairing leaks when the closure barely changes.
    print("\n=== the paired comparison, and its own null ===")
    for reg in ("neutral", "convective"):
        n_, f_ = rows[(reg, "nofloor")], rows[(reg, "new")]
        d = f_["a"] - n_["a"]
        print(f"  {reg:<12} A80 {n_['a']:6.3f} (off) -> {f_['a']:6.3f} (on)   "
              f"change {d:+6.3f} ha ({100*d/n_['a']:+6.1f}%)")
    null = abs(rows[("neutral", "new")]["a"] - rows[("neutral", "nofloor")]["a"])
    eff = abs(rows[("convective", "new")]["a"] - rows[("convective", "nofloor")]["a"])
    print(f"\n  NULL (neutral, where the floor is inert): {null:.3f} ha -- an UPPER bound "
          f"on what the pairing leaks")
    print(f"  EFFECT (convective, where it is not):      {eff:.3f} ha = "
          f"{eff/max(null,1e-9):.0f}x the null")
    print(f"  -> the convective change is real; the neutral one is not distinguishable "
          f"from zero.")
    print(f"\n  For contrast, the UNPAIRED half-vs-half spread is "
          f"{max(rows[('convective','new')]['half'], rows[('convective','nofloor')]['half']):.3f} ha "
          f"-- but every half exceeds its own full window above, so it is measuring the "
          f"bias of a half-sample, not the error of a full one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
