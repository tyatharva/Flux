#!/usr/bin/env python3
"""What states the site actually produces -- and therefore what the corpus must span.

Reads the stratified CONUS404 sample from bin/conus404_site.py and turns it into the five
numbers that size the LES corpus: boundary-layer depth, surface kinematic heat flux,
friction velocity, wind speed at the measurement height, and wind direction.

USED FOR SWEEP DESIGN ONLY. Nothing here forces an LES run. Each case stays one idealised
quasi-stationary state; this file only decides which states are worth the GPU time and how
many of each. In particular z_i MUST be swept -- Kljun takes it as an input, so a corpus at
one z_i leaves that input channel untrained and the emulator cannot learn what it does.

Derivations, all standard surface-layer relations, all stated so they can be checked:

    HFX   = ACSHFLSM * 1000 / 3600                      kJ m-2 per hour -> W m-2
    rho   = PSFC / (R_d T)                              T from the lowest model level
    w'th' = HFX / (rho c_p)                             kinematic, K m s-1
    u*    solved with U10, z0 = 0.05 m and MOST:
             u* = k U10 / [ ln(10/z0) - psi_m(10/L) ],  L = -u*^3 th / (k g w'th')
          iterated to convergence from the neutral value.
    U(30) = (u*/k) [ ln(30/z0) - psi_m(30/L) ]

Wind is rotated from the model's grid-relative frame to earth-relative with the
COSALPHA/SINALPHA the file carries; at this longitude that is a 5.5 deg correction, which
is a whole direction bin and cannot be skipped.

usage: conus404_dist.py [--in data/raw/conus404_site.npz] [--out results/conus404_site.txt]
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np

K, G, CP, RD = 0.4, 9.81, 1004.5, 287.05
Z0 = 0.05          # the site's geometric-mean WorldCover roughness (runs/g24_base/base.in)
ZM = 10.0          # EC measurement height (corrected from 30 m, 2026-08-21)
EPOCH = dt.datetime(1979, 10, 1)
OCT = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def psi_m(zl):
    """Integrated momentum stability correction; Businger-Dyer / Hogstrom."""
    zl = np.asarray(zl, float)
    out = np.where(zl > 0, -5.3 * zl, 0.0)
    neg = zl < 0
    if neg.any():
        x = (1.0 - 19.0 * zl[neg]) ** 0.25
        out[neg] = (np.log((1 + x ** 2) / 2) + 2 * np.log((1 + x) / 2)
                    - 2 * np.arctan(x) + np.pi / 2)
    return out


def solve_ustar(U10, wth, theta, n=12):
    """Joint solution of u* and L from a wind speed and a surface heat flux."""
    us = K * U10 / np.log(10.0 / Z0)
    L = np.full_like(us, np.inf)
    for _ in range(n):
        us = np.maximum(us, 1e-3)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.where(np.abs(wth) > 1e-6,
                         -us ** 3 * theta / (K * G * wth), np.inf)
        L = np.where(np.isfinite(L), L, 1e9)
        # keep the correction inside the range MOST is usable in; beyond |z/L| ~ 2 the
        # profile relation stops being informative and the iteration can run away
        zl = np.clip(10.0 / L, -2.0, 1.0)
        us = K * U10 / np.maximum(np.log(10.0 / Z0) - psi_m(zl), 0.5)
    return us, L


def pct(name, v, unit="", qs=(5, 25, 50, 75, 95, 99)):
    v = v[np.isfinite(v)]
    s = "  ".join(f"p{q}={np.percentile(v, q):.3g}" for q in qs)
    return f"  {name:<26} n={len(v):>6,}  mean={v.mean():.3g}{unit}   {s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/conus404_site.npz")
    ap.add_argument("--out", default="results/conus404_site.txt")
    a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=True)

    th = z["time_hours"].astype(np.int64)
    ok = th >= 0
    times = np.array([EPOCH + dt.timedelta(hours=int(h)) for h in th[ok]])
    month = np.array([t.month for t in times])
    hour = np.array([t.hour for t in times])          # UTC; local standard = UTC-6
    lst = (hour - 6) % 24

    pblh = z["PBLH"][ok]
    tk = z["TK"][ok]
    psfc = z["PSFC"][ok]
    u10g, v10g = z["U10"][ok], z["V10"][ok]
    ca, sa = float(z["static_COSALPHA"]), float(z["static_SINALPHA"])
    # grid-relative -> earth-relative (WRF convention)
    ue = u10g * ca - v10g * sa
    ve = v10g * ca + u10g * sa
    U10 = np.hypot(ue, ve)
    wdir = (270.0 - np.degrees(np.arctan2(ve, ue))) % 360.0      # meteorological, FROM

    hfx = z["ACSHFLSM"][ok] * 1000.0 / 3600.0                    # kJ m-2 h-1 -> W m-2
    rho = psfc / (RD * tk)
    wth = hfx / (rho * CP)
    us, L = solve_ustar(U10, wth, tk)
    with np.errstate(divide="ignore", invalid="ignore"):
        U30 = us / K * np.maximum(np.log(ZM / Z0) - psi_m(np.clip(ZM / L, -2.0, 1.0)), 0.5)
        zL = ZM / L
        ziL = pblh / L

    # Eddy-covariance quality control: u* < 0.15 m/s is the standard rejection, and it is
    # also the class the 4464 m domain cannot contain (its x90 exceeds the wrap length).
    qc = (us >= 0.15) & np.isfinite(pblh) & (pblh > 20)
    out = []
    P = out.append
    P("CONUS404 at the Kegonsa tower -- climatology for corpus design, not forcing")
    P("=" * 78)
    P(f"grid cell (y,x) = ({int(z['iy'])},{int(z['ix'])}), CONUS404 terrain "
      f"{float(z['static_HGT']):.1f} m, land mask {float(z['static_LANDMASK']):.0f}, "
      f"LU category {float(z['static_LU_INDEX']):.0f}")
    P(f"chunk stride {int(z['stride'])} -> 6 contiguous days out of every "
      f"{6*int(z['stride'])}; {ok.sum():,} hourly records, "
      f"{times[0]:%Y-%m-%d} to {times[-1]:%Y-%m-%d}")
    P(f"earth-relative rotation applied: COSALPHA={ca:.4f} SINALPHA={sa:.4f} "
      f"({np.degrees(np.arcsin(sa)):+.2f} deg)")
    P("")
    P("--- ALL HOURS " + "-" * 62)
    P(pct("z_i  (PBLH)", pblh, " m"))
    P(pct("w'theta'  surface", wth, " K m/s"))
    P(pct("H  sensible", hfx, " W/m2"))
    P(pct("U(10 m)", U10, " m/s"))
    P(pct("u*  (MOST, z0=0.05)", us, " m/s"))
    P(pct("U(30 m)  (MOST)", U30, " m/s"))
    P("")
    P(f"--- QUALITY-CONTROLLED (u* >= 0.15 m/s): {qc.sum():,} of {ok.sum():,} "
      f"= {100*qc.mean():.1f}% " + "-" * 12)
    P(pct("z_i", pblh[qc], " m"))
    P(pct("w'theta'", wth[qc], " K m/s"))
    P(pct("u*", us[qc], " m/s"))
    P(pct("U(30 m)", U30[qc], " m/s"))
    P(pct("z_i / L", ziL[qc], ""))
    P("")

    # ---- stability classes -----------------------------------------------------------
    cls = np.full(qc.sum(), "", dtype=object)
    z_ = zL[qc]
    edges = [(-np.inf, -0.5, "very unstable"), (-0.5, -0.05, "unstable"),
             (-0.05, 0.05, "near-neutral"), (0.05, 0.5, "stable"),
             (0.5, np.inf, "very stable")]
    P("--- STABILITY (z/L at 30 m), quality-controlled " + "-" * 29)
    for lo, hi, nm in edges:
        m = (z_ > lo) & (z_ <= hi)
        cls[m] = nm
        if m.any():
            P(f"  {nm:<14} {100*m.mean():5.1f}%   z_i p50 {np.median(pblh[qc][m]):5.0f} m"
              f"   u* p50 {np.median(us[qc][m]):.2f}   "
              f"w'th' p50 {np.median(wth[qc][m]):+.3f} K m/s")
    P("")

    # ---- wind rose, and the joint table that actually sizes the corpus ----------------
    oi = (np.floor(((wdir[qc] + 22.5) % 360) / 45.0)).astype(int)
    P("--- WIND ROSE x STABILITY, quality-controlled (% of all QC hours) " + "-" * 12)
    P("            " + "".join(f"{o:>7}" for o in OCT) + "     all")
    for lo, hi, nm in edges:
        m = (z_ > lo) & (z_ <= hi)
        row = [100 * ((oi == k) & m).mean() for k in range(8)]
        P(f"  {nm:<10}" + "".join(f"{v:7.2f}" for v in row) + f"{100*m.mean():8.2f}")
    row = [100 * (oi == k).mean() for k in range(8)]
    P("  " + "-" * 74)
    P(f"  {'all':<10}" + "".join(f"{v:7.2f}" for v in row) + f"{100.0:8.2f}")
    P("")
    P("  The array is upwind only on northerlies (PROJECT_BRIEF.md), so the N/NE/NW columns are")
    P("  the ones where the tower measures the array at all -- and they are where the")
    P("  emulator's site-specific skill has to come from.")
    P("")

    # ---- the recommendation ----------------------------------------------------------
    q = lambda v, p: float(np.percentile(v[qc][np.isfinite(v[qc])], p))
    P("--- RECOMMENDED SWEEP " + "-" * 55)
    P(f"  z_i          {q(pblh,5):.0f} - {q(pblh,95):.0f} m      "
      f"(p25 {q(pblh,25):.0f}, p50 {q(pblh,50):.0f}, p75 {q(pblh,75):.0f})")
    P(f"  w'theta'     {q(wth,5):+.3f} - {q(wth,95):+.3f} K m/s   "
      f"(p50 {q(wth,50):+.3f})")
    P(f"  u*           {q(us,5):.2f} - {q(us,95):.2f} m/s       (p50 {q(us,50):.2f})")
    P(f"  U(30 m)      {q(U30,5):.1f} - {q(U30,95):.1f} m/s       (p50 {q(U30,50):.1f})")
    P("  direction    all 8 octants, sampled in proportion to the rose above but with a")
    P("               floor, since N/NE/NW carry the array signal and must not be starved")
    P("")
    P("  Convective midday reference (local 10-16 h, w'theta' > 0.05):")
    md = qc & (lst >= 10) & (lst <= 16) & (wth > 0.05)
    if md.any():
        P(f"    n={md.sum():,}   z_i p25/p50/p75 = {np.percentile(pblh[md],25):.0f}/"
          f"{np.percentile(pblh[md],50):.0f}/{np.percentile(pblh[md],75):.0f} m"
          f"   w'th' p25/p50/p75 = {np.percentile(wth[md],25):.3f}/"
          f"{np.percentile(wth[md],50):.3f}/{np.percentile(wth[md],75):.3f} K m/s")
        P(f"    u* p50 {np.median(us[md]):.2f} m/s   U(30) p50 {np.median(U30[md]):.1f} m/s"
          f"   z_i/L p50 {np.median(ziL[md]):+.1f}")
        jja = md & np.isin(month, [6, 7, 8])
        if jja.any():
            P(f"    JJA only (n={jja.sum():,}): z_i p50 {np.median(pblh[jja]):.0f} m, "
              f"w'th' p50 {np.median(wth[jja]):.3f} K m/s, "
              f"z_i/L p50 {np.median(ziL[jja]):+.1f}")
    txt = "\n".join(out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(txt + "\n")
    print(txt)
    print(f"\n  wrote {a.out}")
    np.savez_compressed(a.out.replace(".txt", ".npz"),
                        pblh=pblh, wth=wth, ustar=us, U30=U30, wdir=wdir, zL=zL, ziL=ziL,
                        qc=qc, month=month, lst=lst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
