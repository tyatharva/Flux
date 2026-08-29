#!/usr/bin/env python3
"""Translate the tower's measured sigma_w from 10 m to the 30 m model receptor.

WHY THIS EXISTS. data/raw/H_and_sigma_w.csv is the only EXTERNAL check in the project --
one year of half-hourly eddy covariance at the real instrument, never used for training,
tuning or forcing. Everything else compares the model against itself. The seventh pass
moved the model receptor to 30 m, so the check cannot be applied as-is: the file is at
10 m and sigma_w grows with height in the surface layer.

THE TRANSLATION, and why it goes through u* rather than through a height ratio.

    sigma_w(z) = 1.25 u* phi_w(z/L)          (MOST, the same anchor lpdm/sgs_floor.py uses)

u* is CONSTANT through the surface layer -- that is what "constant-flux layer" means -- and
H is a SURFACE flux, so neither depends on the height the instrument sits at. All the
height dependence is in phi_w(z/L). So:

    1. invert   sigma_w(10) = 1.25 u* phi_w(10/L(u*, H))   for u*, given the measured pair
    2. predict  sigma_w(30) = 1.25 u* phi_w(30/L(u*, H))   with the SAME u*, H

L couples back into the inversion through u*^3, so step 1 is a fixed point, not a formula.
It converges in a handful of iterations for every record in the file.

    L = -u*^3 theta_v / (kappa g w'th_v')      w'th_v' = H / (rho c_p)

THIS IS MORE TRUSTWORTHY AT 30 m THAN AT 10 m, not less. MOST is what is being inverted,
and a 10 m sensor standing inside a 2-3 m solar array is plausibly inside the roughness
sublayer, where MOST does not hold (PROJECT_BRIEF.md). At 30 m the receptor is clear of a 5-15 m
RSL by a factor of 2-6, so the FORWARD half of the translation is on firmer ground than the
backward half. The residual risk sits in step 1 and is stated, not hidden: an RSL-affected
sigma_w(10) yields an RSL-affected u*.

phi_w is IMPORTED from lpdm/sgs_floor.py. A gate that reimplements the production function
is scoring a different model than the one it is gating (PROJECT_BRIEF.md, twice now).

usage:
  bin/sigma_w_tower.py                              # build results/sigma_w_curve_30m.json
  bin/sigma_w_tower.py --h 150 --les 0.75           # score one case against the curve
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.sgs_floor import phi_w_of_zeta            # noqa: E402

KAPPA = 0.4
GRAV = 9.81
THETA_V = 288.0          # K, a site-typical virtual potential temperature
RHO_CP = 1.2 * 1004.5    # J m^-3 K^-1, so H [W/m2] -> w'th' [K m/s]
Z_TOWER = 10.0
Z_MODEL = 30.0
ZETA_CAP = 2.0           # phi_w's own stable saturation; keep the inversion inside it
H_ABS_MAX = 500.0        # W/m2. THE FILE CARRIES SENTINELS: 9 rows at exactly -9999 and
# 18 below -500 W/m2, which is not a sensible heat flux any surface produces. Unscreened
# they land in the lowest H decile and drag its edge to -9999, so the bin the gate would
# look a nocturnal case up in is defined by nine bad rows. Screen on the ARTIFACT's
# plausibility, not on a magic number: |H| <= 500 W/m2 keeps every real record.
USTAR_QC = 0.15          # m/s, the project's standing QC (PROJECT_BRIEF.md, CONUS404 section)


def ustar_from_sigma_w(sig, h_wm2, z=Z_TOWER, iters=60, tol=1e-10):
    """Invert sigma_w(z) = 1.25 u* phi_w(z/L) for u*, by fixed-point iteration.

    Vectorised over records. Starts from the neutral solution (phi_w = 1), which is exact
    at H = 0 and within 30% everywhere this site goes, and relaxes.
    """
    sig = np.asarray(sig, dtype=np.float64)
    wth = np.asarray(h_wm2, dtype=np.float64) / RHO_CP
    ust = sig / 1.25                                       # neutral start
    for _ in range(iters):
        prev = ust
        zeta = zeta_of(ust, wth, z)
        ust = sig / (1.25 * phi_w_of_zeta(zeta))
        if np.nanmax(np.abs(ust - prev)) < tol:
            break
    return ust, zeta_of(ust, wth, z)


def zeta_of(ust, wth, z):
    """z/L, with L = -u*^3 theta_v/(kappa g w'th_v'). Capped at phi_w's own saturation."""
    ust = np.maximum(np.asarray(ust, dtype=np.float64), 1e-4)
    wth = np.asarray(wth, dtype=np.float64)
    # 1/L rather than L: the neutral limit is 1/L -> 0, which is finite, where L -> inf.
    inv_L = -KAPPA * GRAV * wth / (THETA_V * ust ** 3)
    return np.clip(z * inv_L, -1e4, ZETA_CAP)


def load(csv_path):
    H, S = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                h_, s_ = float(row["H"]), float(row["sigma_w"])
            except (TypeError, ValueError):
                continue
            # np.isfinite FIRST, never isnan: inf passes every > comparison (PROJECT_BRIEF.md).
            if np.isfinite(h_) and np.isfinite(s_) and s_ > 0 and abs(h_) <= H_ABS_MAX:
                H.append(h_); S.append(s_)
    return np.asarray(H), np.asarray(S)


def build_curve(H, S, n_bins=10, min_n=30, ustar_qc=USTAR_QC):
    """Per-H-decile quantiles of the translated sigma_w(30 m).

    `ustar_qc` screens on the INVERTED u*, which is the project's standing QC applied to
    the quantity this file can actually produce. It removes the calm-night tail where the
    inversion is least trustworthy (u* ~ 0.08, z/L ~ 1) and where MOST is furthest from
    the regime it was written for. Those hours are outside the corpus anyway -- stable is
    excluded (STABLE_REGIME_RESULT.md) -- so screening them costs the gate nothing.
    """
    ust, zeta10 = ustar_from_sigma_w(S, H, Z_TOWER)
    wth = H / RHO_CP
    zeta30 = zeta_of(ust, wth, Z_MODEL)
    s30 = 1.25 * ust * phi_w_of_zeta(zeta30)
    ok = np.isfinite(s30) & np.isfinite(ust) & (ust >= ustar_qc)
    edges = np.unique(np.percentile(H[ok], np.linspace(0, 100, n_bins + 1)))
    edges[0] -= 1e-9
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (H > lo) & (H <= hi)
        if m.sum() < min_n:
            continue
        q10, q30 = np.percentile(S[m], [25, 50, 75]), np.percentile(s30[m], [25, 50, 75])
        bins.append(dict(
            h_lo=float(lo), h_hi=float(hi), n=int(m.sum()), h_median=float(np.median(H[m])),
            sigma_w_10m=dict(p25=float(q10[0]), p50=float(q10[1]), p75=float(q10[2])),
            sigma_w_30m=dict(p25=float(q30[0]), p50=float(q30[1]), p75=float(q30[2])),
            ustar_median=float(np.median(ust[m])),
            zeta_10m_median=float(np.median(zeta10[m])),
            zeta_30m_median=float(np.median(zeta30[m])),
            lift=float(np.median(s30[m]) / np.median(S[m]))))
    return bins, ust, s30


def score(bins, h_case, les_sigma_w=None, z=Z_MODEL):
    """Look up the bin containing h_case and, if given, place les_sigma_w in it."""
    b = min(bins, key=lambda r: abs(r["h_median"] - h_case))
    key = "sigma_w_30m" if abs(z - Z_MODEL) < 1e-6 else "sigma_w_10m"
    q = b[key]
    out = dict(h_case=float(h_case), bin=b, iqr=[q["p25"], q["p75"]], median=q["p50"])
    if les_sigma_w is not None:
        out["les"] = float(les_sigma_w)
        out["inside_iqr"] = bool(q["p25"] <= les_sigma_w <= q["p75"])
        out["ratio_to_median"] = float(les_sigma_w / q["p50"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/H_and_sigma_w.csv")
    ap.add_argument("--out", default="results/sigma_w_curve_30m.json")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--ustar-qc", type=float, default=USTAR_QC,
                    help="drop records whose INVERTED u* falls below this (m/s)")
    ap.add_argument("--h", type=float, default=None, help="score one case at this H (W/m2)")
    ap.add_argument("--les", type=float, default=None, help="the LES sigma_w at 30 m")
    a = ap.parse_args()

    H, S = load(a.csv)
    bins, ust, s30 = build_curve(H, S, a.bins, ustar_qc=a.ustar_qc)
    if not bins:
        print("FATAL: no H bin held enough records", file=sys.stderr)
        return 2

    n_kept = sum(b["n"] for b in bins)
    print(f"=== tower sigma_w translated {Z_TOWER:.0f} m -> {Z_MODEL:.0f} m "
          f"({n_kept} of {H.size} half-hours; |H| <= {H_ABS_MAX:.0f} W/m2 and "
          f"inverted u* >= {a.ustar_qc:.2f} m/s) ===")
    print("  MOST inversion: sigma_w(10) -> u* -> sigma_w(30). u* is constant through the")
    print("  surface layer and H is a surface flux, so only phi_w(z/L) changes with height.")
    print(f"\n  {'H bin (W/m2)':>18} {'n':>6} {'sw10 p50':>9} {'sw30 p50':>9} "
          f"{'sw30 IQR':>16} {'u* p50':>7} {'z/L(30)':>8} {'lift':>6}")
    for b in bins:
        q = b["sigma_w_30m"]
        print(f"  {b['h_lo']:>8.0f}..{b['h_hi']:<8.0f} {b['n']:>6d} "
              f"{b['sigma_w_10m']['p50']:>9.3f} {q['p50']:>9.3f} "
              f"{'[%.3f, %.3f]' % (q['p25'], q['p75']):>16} "
              f"{b['ustar_median']:>7.3f} {b['zeta_30m_median']:>8.2f} {b['lift']:>6.3f}")

    lifts = np.array([b["lift"] for b in bins])
    print(f"\n  the 10 -> 30 m lift runs {lifts.min():.3f} - {lifts.max():.3f}x, and it is")
    print("  ABOVE 1 IN BOTH REGIMES, which is not obvious and is worth stating. The ratio")
    print("  is exactly phi_w(30/L)/phi_w(10/L) -- u* cancels -- and zeta triples with")
    print("  height, so it exceeds 1 whenever phi_w is increasing in |zeta|. Unstable:")
    print("  (1+3|z|)^(1/3) grows. Stable: (1+0.6 z)/(1+0.2 z) grows too.")
    print("  CAVEAT ON THE STABLE SIDE, stated because the form is being used outside what")
    print("  it is good for: surface-layer MOST makes sigma_w RISE with height at fixed u*,")
    print("  while a real SBL has u* falling with height and sigma_w falling with it. The")
    print("  translated stable bins are therefore an upper bound. It costs nothing here --")
    print("  the corpus contains no stable cases (STABLE_REGIME_RESULT.md) -- but a stable")
    print("  case scored against these bins would be scored against the wrong curve.")
    print("\n  READ THIS AS AN ORDER-OF-MAGNITUDE CHECK PLUS A GATE, NOT A VALIDATION. The")
    print("  file carries no wind speed, so conditioning on H alone leaves an IQR spanning")
    print("  a factor of ~2; a case inside it is consistent with the instrument and nothing")
    print("  finer. What it catches is a collapsed or closure-dominated near-surface")
    print("  variance, which is exactly what the 10 m configuration turned out to have.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(z_tower=Z_TOWER, z_model=Z_MODEL, n_records=int(H.size),
                       theta_v=THETA_V, rho_cp=RHO_CP, kappa=KAPPA,
                       h_abs_max=H_ABS_MAX, ustar_qc=a.ustar_qc,
                       n_kept=int(n_kept), bins=bins), f, indent=1)
    print(f"\n  wrote {a.out}")

    if a.h is not None:
        r = score(bins, a.h, a.les)
        print(f"\n=== case at H = {a.h:+.0f} W/m2 -> bin "
              f"[{r['bin']['h_lo']:.0f}, {r['bin']['h_hi']:.0f}], n = {r['bin']['n']} ===")
        print(f"  tower sigma_w(30 m): median {r['median']:.3f}, "
              f"IQR [{r['iqr'][0]:.3f}, {r['iqr'][1]:.3f}] m/s")
        if a.les is not None:
            print(f"  LES sigma_w(30 m) = {r['les']:.3f} -> "
                  f"{'INSIDE' if r['inside_iqr'] else 'OUTSIDE'} the IQR, "
                  f"{r['ratio_to_median']:.2f}x the median")
    return 0


if __name__ == "__main__":
    sys.exit(main())
