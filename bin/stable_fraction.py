#!/usr/bin/env python3
"""How much of this site's STABLE record is WEAKLY stable -- and therefore runnable?

WHY THIS NUMBER EXISTS. A stable seed at GABLS1's own regime (G = 8 m/s,
w'th' = -0.012 K m/s) was healthy for 1.75 simulated hours at dx = 16 m and then
collapsed: u* 0.236 -> 0.098, z/L -> +2.67. The cause is resolution, measured at the
HEALTHY dump: the Ozmidov scale L_O = sqrt(eps/N^3) -- the largest eddy stratification
permits to overturn -- was only 1.0-3.2 x Delta through the layer. GABLS1 runs the same
regime at dx = 6.25 m.

So the question is not "can this grid do stable" (it cannot, at z/L ~ 0.2) but "how much
of the site's stable record sits at stratification weak enough that it can". If the
weakly-stable band is most of the record, the exclusion is a minor limitation. If it is a
sliver, stable coverage is a major gap. That is what this measures, and it belongs in the
limitations section either way.

THREE INDEPENDENT SOURCES, deliberately. Each has a different failure mode and they are
reported side by side rather than averaged:

  1. THE TOWER'S OWN RECORD -- data/raw/H_and_sigma_w.csv, 17,520 half-hours, one year.
     This is a MEASUREMENT at the actual receptor. It carries H and sigma_w and no u*, so
     u* is inverted from sigma_w through the project's OWN closure,
     sigma_w = 1.25 u* phi_w(zeta), phi_w = 1 + 0.2 zeta stable (lpdm/sgs_floor.py).
     Using the project's own phi_w rather than a textbook one is the point: the answer is
     then consistent with the closure the LPDM actually runs.

  2. HRRR -- results/candidates.tsv, the corpus's own forcing source, 2208 hourly
     analyses at the tower. SHTFL and the 10 m wind, solved for u* through the full
     stability-corrected log law. NOT the neutral log law: at fixed wind, neutral
     inversion overestimates u*, which understates z/L and would bias this answer toward
     the very band it is testing.

  3. CONUS404 -- results/conus404_site.npz, 39,456 hourly records over 45 years. Carries
     u* DIRECTLY, so no inversion at all. The longest record and the only one with no
     surface-layer assumption in it, but 4 km and a reanalysis rather than the site.

THE u* >= 0.15 QC CUTS EXACTLY THE HOURS IN QUESTION, so every fraction is reported twice.
Strong stability suppresses u*, so a QC on u* preferentially deletes strongly stable hours
and inflates the weakly-stable share. Quoting only the QC'd number would be the answer
flattering itself.

usage: stable_fraction.py [--zm 10.0] [--out results/stable_fraction.txt]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

VONK = 0.4
G = 9.81
THETA0 = 290.0
RHO_CP = 1.15 * 1004.5      # W/m^2 -> K m/s, the same constant select_times.py uses
Z0 = 0.1435                 # geometric-mean WorldCover z0 of the 1952 m box
UST_QC = 0.15               # the standing QC in PROJECT_BRIEF.md

# The bands. The user's question is the middle one; the others exist so the answer cannot
# be read as "weakly stable is rare" when the band just below it is even more runnable.
BANDS = [
    (0.0,   0.03, "near-neutral stable  0    < z/L <= 0.03"),
    (0.03,  0.10, "WEAKLY stable        0.03 < z/L <= 0.10"),
    (0.10,  0.50, "moderately stable    0.10 < z/L <= 0.50"),
    (0.50,  np.inf, "strongly stable      0.50 < z/L"),
]


def zeta_from_ustar(ust, wth, zm):
    """z/L at zm from u* and the kinematic heat flux. Positive = stable."""
    ust = np.asarray(ust, dtype=np.float64)
    wth = np.asarray(wth, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -VONK * G * zm * wth / (np.maximum(ust, 1e-6) ** 3 * THETA0)
    return z


def ustar_from_sigmaw(sw, wth, zm):
    """Invert sigma_w = 1.25 u* (1 + 0.2 z/L) for u*, on the PHYSICAL branch.

    With a = kappa g zm (-w'th') / theta0 > 0 and zeta = a/u*^3,

        f(u*) = 1.25 u* + 0.25 a / u*^2 - sigma_w

    is convex with its minimum at u* = (0.4 a)^(1/3), i.e. at zeta = 2.5, and

        f_min = 1.875 (0.4 a)^(1/3).

    So sigma_w < 1.875 (0.4a)^(1/3) has NO consistent MOST state at all: the measured
    vertical velocity variance is too small for the measured cooling. Those half-hours are
    returned as NaN and counted separately rather than clipped -- clipping them would move
    them into a stability band, and which band would be an artifact of the clip.

    Of the two roots, the physical one is the LARGER u* (the smaller zeta). The small-u*
    root is the runaway branch: the same decoupled state the cold-started seed fell into.
    """
    sw = np.asarray(sw, dtype=np.float64)
    wth = np.asarray(wth, dtype=np.float64)
    a = VONK * G * zm * np.maximum(-wth, 0.0) / THETA0          # >= 0
    u_min = np.cbrt(0.4 * np.maximum(a, 1e-30))
    f_min = 1.875 * u_min
    ok = (a > 0) & (sw > f_min)
    # Bisect on [u_min, hi] where f is increasing. hi from the neutral limit with margin.
    lo = np.where(ok, u_min, np.nan)
    hi = np.maximum(sw / 1.25, u_min) * 4.0 + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1.25 * mid + 0.25 * a / mid ** 2 - sw
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    out = np.where(ok, 0.5 * (lo + hi), np.nan)
    # a == 0 (no cooling) is neutral-or-unstable, not a failure: u* = sigma_w/1.25.
    out = np.where(a <= 0, sw / 1.25, out)
    return out


def ustar_from_wind(U, wth, zm, z0=Z0):
    """Solve the stability-corrected log law for u*. Businger-Dyer, psi_m = -5 zeta stable.

        U = (u*/kappa) [ ln(zm/z0) + 5 (zm - z0)/L ]          (stable)
        U = (u*/kappa) [ ln(zm/z0) - psi_m(zeta) ]            (unstable)

    Fixed-point iterated with damping. NON-CONVERGENCE IS A RESULT, not an error: at
    strong enough cooling for a given wind there is no solution -- the classical
    surface-layer cutoff -- and those hours are exactly the ones this grid cannot run.
    They are returned NaN and counted, never clipped into a band.
    """
    U = np.asarray(U, dtype=np.float64)
    wth = np.asarray(wth, dtype=np.float64)
    lnz = np.log(zm / z0)
    ust = VONK * np.maximum(U, 1e-3) / lnz                     # neutral first guess
    for _ in range(200):
        zeta = zeta_from_ustar(ust, wth, zm)
        zeta = np.clip(zeta, -20.0, 20.0)
        x = (1.0 - 16.0 * np.minimum(zeta, 0.0)) ** 0.25
        psi_un = (2.0 * np.log(0.5 * (1.0 + x)) + np.log(0.5 * (1.0 + x * x))
                  - 2.0 * np.arctan(x) + 0.5 * np.pi)
        psi = np.where(zeta < 0.0, psi_un, -5.0 * zeta)
        denom = lnz - psi
        new = VONK * np.maximum(U, 1e-3) / np.maximum(denom, 0.05)
        ust = 0.7 * ust + 0.3 * new
    zeta = zeta_from_ustar(ust, wth, zm)
    # A converged state must reproduce the wind. Where the denominator has been driven to
    # the 0.05 floor the log law has no solution and the answer is meaningless.
    x = (1.0 - 16.0 * np.minimum(np.clip(zeta, -20, 20), 0.0)) ** 0.25
    psi_un = (2.0 * np.log(0.5 * (1.0 + x)) + np.log(0.5 * (1.0 + x * x))
              - 2.0 * np.arctan(x) + 0.5 * np.pi)
    psi = np.where(zeta < 0.0, psi_un, -5.0 * np.clip(zeta, -20, 20))
    resid = np.abs((ust / VONK) * (lnz - psi) - U) / np.maximum(U, 1e-3)
    bad = (~np.isfinite(ust)) | (resid > 0.02) | ((lnz - psi) <= 0.06)
    return np.where(bad, np.nan, ust)


def tabulate(name, zeta, ust, n_nonmost, tot, P):
    """One source's stable breakdown, reported twice: all hours, and u* >= 0.15.

    Returns the QC'd counts the closing summary needs: (n_qc, n_stable, n_runnable).
    """
    out_qc = (0, 0, 0)
    P("")
    P(f"--- {name} " + "-" * max(0, 74 - len(name)))
    P(f"    {tot:,} records; {n_nonmost:,} ({100.0*n_nonmost/max(tot,1):.1f}%) have NO "
      f"consistent MOST state")
    P(f"    (those are strongly stable by construction -- counted, never clipped into a band)")
    for label, mask in (("ALL hours", np.ones_like(zeta, dtype=bool)),
                        (f"u* >= {UST_QC} (the standing QC)", ust >= UST_QC)):
        m = mask & np.isfinite(zeta)
        stable = m & (zeta > 0.0)
        # the non-MOST hours belong with the stable set: they failed BECAUSE of cooling
        nm = n_nonmost if label == "ALL hours" else 0
        ns = int(stable.sum()) + nm
        P("")
        P(f"  [{label}]  {int(m.sum()):,} usable + {nm:,} non-MOST; "
          f"STABLE = {ns:,} ({100.0*ns/max(int(m.sum())+nm,1):.1f}% of the record)")
        if ns == 0:
            continue
        P(f"      {'band':<42} {'n':>7} {'% of stable':>12} {'% of all':>9}")
        cum = 0
        for lo, hi, lab in BANDS:
            k = int((stable & (zeta > lo) & (zeta <= hi)).sum())
            if lab.startswith("strongly"):
                k += nm
            cum += k
            P(f"      {lab:<42} {k:>7} {100.0*k/ns:>11.1f}% {100.0*k/max(int(m.sum())+nm,1):>8.1f}%")
        run = int((stable & (zeta > 0.0) & (zeta <= 0.10)).sum())
        if label.startswith("u* >="):
            out_qc = (int(m.sum()), ns, run)
        P(f"      {'-'*42} {'-'*7} {'-'*12} {'-'*9}")
        P(f"      {'RUNNABLE at this grid (z/L <= 0.10)':<42} {run:>7} "
          f"{100.0*run/ns:>11.1f}% {100.0*run/max(int(m.sum())+nm,1):>8.1f}%")
        z = zeta[stable]
        if z.size:
            P(f"      z/L over the stable set: p10 {np.percentile(z,10):.3f}  "
              f"p25 {np.percentile(z,25):.3f}  p50 {np.percentile(z,50):.3f}  "
              f"p75 {np.percentile(z,75):.3f}  p90 {np.percentile(z,90):.3f}")
        if label.startswith("u* >="):
            # CROSSWALK, so this does not read as contradicting PROJECT_BRIEF.md's climatology.
            # "Stable" here means z/L > 0 at 10 m, which is a WIDER set than the standing
            # class table's "stable + very stable" (z/L > 0.05, and quoted at 30 m where
            # z/L is 3x larger for the same L). Both splits are printed on the same rows
            # so the two numbers can be reconciled instead of argued about.
            zz = zeta[m]
            edges = [(-np.inf, -0.5, "very unstable  z/L < -0.5"),
                     (-0.5, -0.05, "unstable       -0.5 .. -0.05"),
                     (-0.05, 0.05, "near-neutral   -0.05 .. +0.05"),
                     (0.05, 0.5, "stable         +0.05 .. +0.5"),
                     (0.5, np.inf, "very stable    z/L > +0.5")]
            P("")
            P("      crosswalk to PROJECT_BRIEF.md's own class edges, evaluated HERE at 10 m:")
            for lo, hi, lab in edges:
                k = int(((zz > lo) & (zz <= hi)).sum())
                P(f"        {lab:<34} {k:>7} {100.0*k/max(zz.size,1):>6.1f}%")
    return out_qc


def _close(tab, P):
    """The one number the limitations section needs: what the restriction COSTS."""
    P("")
    P("=" * 86)
    P("WHAT RESTRICTING THE LIBRARY TO WEAK STABILITY COSTS  (u* >= 0.15 throughout)")
    P("=" * 86)
    P(f"  {'source':<34}{'stable':>9}{'kept':>9}{'kept/stable':>13}{'LOST/record':>13}")
    for name, (nq, ns, nr) in tab:
        if nq == 0:
            continue
        P(f"  {name:<34}{100.0*ns/nq:>8.1f}%{100.0*nr/nq:>8.1f}%"
          f"{100.0*nr/max(ns,1):>12.1f}%{100.0*(ns-nr)/nq:>12.1f}%")
    P("")
    P("  'stable'      = z/L > 0 at 10 m, as a share of the QC'd record")
    P("  'kept'        = z/L <= 0.10, the band the grid can carry")
    P("  'LOST/record' = stable but too strongly stratified to run: the exclusion's cost")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zm", type=float, default=10.0)
    ap.add_argument("--out", default="results/stable_fraction.txt")
    a = ap.parse_args()

    lines = []
    tab = []
    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 86)
    P(f"HOW MUCH OF THIS SITE'S STABLE RECORD IS WEAKLY STABLE?   z_m = {a.zm:.1f} m AGL")
    P("=" * 86)
    P("")
    P("The grid can run stable cases only where stratification is weak enough that the")
    P("Ozmidov scale stays above Delta = 10.09 m. Measured on the collapsed seed, that")
    P("fails at z/L ~ 0.2. The band being tested is 0.03 < z/L <= 0.10, and everything")
    P("BELOW it is more runnable still, so the operative figure is z/L <= 0.10.")

    # ---- 1. the tower's own record ---------------------------------------------------
    if os.path.exists("data/raw/H_and_sigma_w.csv"):
        H, SW = [], []
        with open("data/raw/H_and_sigma_w.csv") as fh:
            for row in csv.DictReader(fh):
                try:
                    h_, s_ = float(row["H"]), float(row["sigma_w"])
                except (TypeError, ValueError):
                    h_, s_ = np.nan, np.nan
                H.append(h_); SW.append(s_)
        H = np.asarray(H); SW = np.asarray(SW)
        good = np.isfinite(H) & np.isfinite(SW) & (SW > 0)
        wth = H[good] / RHO_CP
        ust = ustar_from_sigmaw(SW[good], wth, a.zm)
        zeta = zeta_from_ustar(ust, wth, a.zm)
        nonmost = int((~np.isfinite(ust)).sum())
        tab.append(("tower H + sigma_w (1 y, 30 min)", tabulate("1. THE TOWER'S OWN RECORD  (H_and_sigma_w.csv, 1 y of half-hours)",
                 zeta, np.nan_to_num(ust, nan=0.0), nonmost, int(good.sum()), P)))
    else:
        P("\n--- 1. tower record ABSENT (data/raw/H_and_sigma_w.csv) ---")

    # ---- 2. HRRR, the corpus's own forcing -------------------------------------------
    if os.path.exists("results/candidates.tsv"):
        sh, ws = [], []
        for ln in open("results/candidates.tsv"):
            f = ln.rstrip("\n").split("\t")
            if len(f) < 8 or f[0] == "date":
                continue
            sh.append(float(f[5])); ws.append(float(f[7]))
        sh = np.asarray(sh); ws = np.asarray(ws)
        wth = sh / RHO_CP
        ust = ustar_from_wind(ws, wth, a.zm)
        zeta = zeta_from_ustar(ust, wth, a.zm)
        nonmost = int((~np.isfinite(ust)).sum())
        tab.append(("HRRR (corpus forcing, hourly)", tabulate("2. HRRR  (candidates.tsv, the corpus's own forcing source)",
                 zeta, np.nan_to_num(ust, nan=0.0), nonmost, sh.size, P)))
    else:
        P("\n--- 2. HRRR candidates ABSENT (results/candidates.tsv) ---")

    # ---- 3. CONUS404, 45 years, u* carried directly ----------------------------------
    if os.path.exists("results/conus404_site.npz"):
        d = np.load("results/conus404_site.npz", allow_pickle=True)
        ust = np.asarray(d["ustar"], dtype=np.float64)
        wth = np.asarray(d["wth"], dtype=np.float64)
        zeta = zeta_from_ustar(ust, wth, a.zm)
        tab.append(("CONUS404 (45 y, u* direct)", tabulate("3. CONUS404  (45 y hourly, u* carried directly -- no inversion)",
                 zeta, ust, 0, ust.size, P)))
    else:
        P("\n--- 3. CONUS404 ABSENT (results/conus404_site.npz) ---")

    _close(tab, P)
    P("")
    P("=" * 86)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
