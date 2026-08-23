#!/usr/bin/env python3
"""Structural tests for the restructured sigma_w floor (lpdm/sgs_floor.py).

The defect this replaces was found only after eight production footprints had been
computed on it, by a diagnostic that ran once. These are the properties that make it
impossible rather than merely unobserved, checked on adversarial profiles as well as on
the measured one.

usage: test_sgs_floor.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.sgs_floor import check_monotone, most_floor

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def levels(nz=122, dz=3.9933, zc=2500.0, c1=0.194059):
    """The production vertical grid, from FastEddy's own zDeform."""
    dzeta = zc / (nz - 0.5)
    zeta = (np.arange(nz) + 0.5) * dzeta
    return ((1.0 - c1) / zc ** 2) * zeta ** 3 + c1 * zeta


def convective(zl, h=540.0, ust=0.42, L=-25.0, wth=0.18, th0=304.0):
    """A CBL column shaped like the measured one in FIFTH_PASS_RESULTS.md 5b."""
    zz = np.clip(zl / h, 1e-4, 2.0)
    wstar = (9.81 / th0 * wth * h) ** (1.0 / 3.0)
    ww = 1.8 * wstar ** 2 * zz ** (2.0 / 3.0) * np.maximum(1.0 - 0.8 * zz, 0.0) ** 2
    e = 0.55 * np.exp(-zl / 12.0) + 0.040 * np.exp(-zl / 90.0)
    return dict(zlev=zl, ww_prof=ww, esgs_prof=e, ustar=ust, L=L, h=h,
                htFlux=wth, theta0=th0)


def neutral(zl, h=414.0, ust=0.388):
    zz = np.clip(zl / h, 1e-4, 2.0)
    ww = (1.6 * ust) ** 2 * zz ** (2.0 / 3.0) * np.maximum(1.0 - zz, 0.0) ** 1.4
    e = 0.58 * np.exp(-zl / 9.0) + 0.030 * np.exp(-zl / 70.0)
    return dict(zlev=zl, ww_prof=ww, esgs_prof=e, ustar=ust, L=np.inf, h=h,
                htFlux=0.0, theta0=300.0)


def main():
    zl = levels()
    print(f"grid: {len(zl)} levels, first {zl[0]:.3f} m, receptor {zl[2]:.3f} m")

    cases = {"convective": convective(zl), "neutral": neutral(zl)}
    # Adversarial: a base profile that FALLS everywhere (all the energy sub-grid at the
    # wall and no resolved motion), and one whose peak is at the very first level.
    adv = convective(zl)
    adv2 = dict(adv); adv2["ww_prof"] = adv["ww_prof"] * 0.0
    cases["no resolved w"] = adv2
    adv3 = dict(adv); adv3["esgs_prof"] = adv["esgs_prof"] * 40.0
    cases["sub-grid dominated"] = adv3
    adv4 = dict(adv); adv4["ustar"] = 4.0            # an absurd target
    cases["absurd target"] = adv4

    for name, st in cases.items():
        print(f"\n=== {name} ===")
        for d_r in (0.0, 1.5):
            fl = most_floor(st, d_r=d_r)
            n_new, worst = check_monotone(fl)
            k = 2
            check(f"{name} d={d_r}: floor never lowers the variance",
                  bool(np.all(fl["fac"] >= 1.0 - 1e-12)),
                  f"min fac {fl['fac'].min():.6f}")
            check(f"{name} d={d_r}: no floor-induced turnover in sigma_w^2",
                  n_new == 0, f"{n_new} new drops, worst {worst:+.3%}")
            check(f"{name} d={d_r}: inactive at and above the resolved profile's peak",
                  abs(fl["fac"][fl["kpk"]] - 1.0) < 1e-9
                  and bool(np.allclose(fl["fac"][fl["kpk"]:], 1.0)),
                  f"peak at z={fl['zl'][fl['kpk']]:.0f} m")
            check(f"{name} d={d_r}: transported sigma_w^2 is continuous at the peak",
                  abs(fl["sig2"][fl["kpk"]] - fl["base"][fl["kpk"]]) < 1e-12)
            check(f"{name} d={d_r}: the floor still acts AT THE RECEPTOR when needed",
                  fl["zl"][fl["kpk"]] > 20.0,
                  f"active range 0-{fl['zl'][fl['kpk']]:.0f} m")
            print(f"        receptor {fl['zl'][k]:.2f} m: fac {fl['fac'][k]:.3f}, "
                  f"sigma_w/u* {np.sqrt(fl['base'][k])/fl['ustar']:.2f} -> "
                  f"{np.sqrt(fl['sig2'][k])/fl['ustar']:.2f}, "
                  f"max fac {fl['fac'].max():.2f}, peak z {fl['zl'][fl['kpk']]:.0f} m")

    # The retired form, on the same profile, must SHOW the defect -- otherwise this test
    # is not testing what it claims to.
    print("\n=== the retired taper, same fields (it must FAIL these) ===")
    fl_old = most_floor(cases["convective"], d_r=1.5, legacy=True)
    n_old, worst_old = check_monotone(fl_old)
    check("legacy form reproduces the defect (turnovers > 0)", n_old > 0,
          f"{n_old} drops, worst {worst_old:+.2%}, max fac {fl_old['fac'].max():.2f}")
    fl_new = most_floor(cases["convective"], d_r=1.5)
    print(f"  max factor: legacy {fl_old['fac'].max():.2f} at "
          f"z={fl_old['zl'][int(np.argmax(fl_old['fac']))]:.0f} m  vs  "
          f"restructured {fl_new['fac'].max():.2f} at "
          f"z={fl_new['zl'][int(np.argmax(fl_new['fac']))]:.0f} m")
    print(f"  receptor factor: legacy {fl_old['fac'][2]:.3f}  vs  "
          f"restructured {fl_new['fac'][2]:.3f}")

    # THE TWO DELIVERIES MUST DESCRIBE THE SAME VARIANCE. They differ only in how the
    # correction reaches the drift; if they disagree on sigma^2 itself then one of them
    # is not the floor that was designed, and the well-mixed comparison between them
    # would be measuring the wrong thing.
    print("\n=== additive vs multiplicative deliver the same sigma^2 ===")
    for name, st in (("convective", cases["convective"]), ("neutral", cases["neutral"])):
        fl = most_floor(st, d_r=1.5)
        have, wwp = fl["have"], fl["wwp"]
        mult = wwp + fl["fac"] * have          # sigma^2 via the factor
        addv = wwp + have + fl["delta"]        # sigma^2 via the offset
        check(f"{name}: the two deliveries agree on sigma^2",
              bool(np.allclose(mult, addv, rtol=1e-10, atol=1e-12)),
              f"max |diff| = {np.max(np.abs(mult - addv)):.2e}")
        check(f"{name}: the offset is non-negative (still a floor)",
              bool(np.all(fl["delta"] >= -1e-15)), f"min {fl['delta'].min():.2e}")
        check(f"{name}: the offset vanishes where the floor is inactive",
              bool(np.allclose(fl["delta"][fl["kpk"]:], 0.0)))

    # A floor that is never needed must be exactly inert -- bit for bit, so a run in
    # conditions the LES already resolves is unchanged by the closure change.
    print("\n=== inertness ===")
    st = neutral(zl); st = dict(st); st["ustar"] = 1e-6
    fl = most_floor(st, d_r=1.5)
    check("a target of zero leaves the model untouched",
          bool(np.allclose(fl["fac"], 1.0)) and bool(np.allclose(fl["sig2"], fl["base"])))

    # The interpolant's derivative, as the LPDM now computes it, must be the exact
    # derivative of the curve the LPDM samples.
    print("\n=== drift consistency ===")
    fl = most_floor(cases["convective"], d_r=1.5)
    z, f = fl["zl"], fl["fac"]
    slope = np.diff(f) / np.diff(z)
    zt = np.linspace(z[0] + 0.01, z[40], 4000)
    sc = np.interp(zt, z, f)
    idx = np.clip(np.searchsorted(z, zt) - 1, 0, len(slope) - 1)
    num = np.gradient(sc, zt)
    ana = slope[idx]
    m = np.abs(num - ana) > 1e-6 * np.maximum(np.abs(ana), 1.0)
    check("piecewise slope == d/dz of the sampled interpolant",
          m.sum() <= len(z),                       # only the knot cells differ
          f"{int(m.sum())} of {len(zt)} samples differ (knots: {len(z)})")
    cen = np.gradient(f, z)
    check("the retired central difference was a DIFFERENT curve",
          float(np.max(np.abs(cen[:40] - np.r_[slope[:40]]))) > 0.05,
          f"max |central - exact| = {np.max(np.abs(cen[:40] - slope[:40])):.3f}")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: " + "; ".join(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
