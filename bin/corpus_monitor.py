#!/usr/bin/env python3
"""Is this corpus case healthy? The per-case gate for an unattended 1370-case run.

=== WHY THE ARRAY SHARE IS NOT THE THING TO WATCH ===

The array share is the scientific result, and it is a poor fault detector. Measured on
case_2023031014: `h` fell through to the domain top, a SIX-fold error in the quantity that
sets the sigma_w floor's mixed-layer blend, and the share moved **0.8 points against its
own 3.66-point standard error** -- invisible. The same defect moved the near-field peak a
full raster cell (64 -> 48 m) and drove the floor factor to 9e4.

That asymmetry is not an accident. The share is an integral over ~1.03% of the box with a
sampling SE of 3-4 points, so it averages a near-field error away; the peak is a location,
converged to under one cell in a single 2.5-minute sub-window (PROJECT_BRIEF.md's ensemble table),
so it registers one. **Gate the sharp quantities. Report the blunt one.**

=== THE GATES, AND WHERE EACH NUMBER COMES FROM ===

Every band below is measured off the records already on disk, not chosen. The "production"
column is the 122^3 @ 16 m grid at a 10 m receptor -- g16r_* (4 directions x 2 regimes),
g16p6b_*, and the corpus cases -- and the "failures" column is what this project's own
historical defects actually produced.

  gate                      production        failures seen            band       margin
  ------------------------  ----------------  ---------------------  ---------  --------
  G1 f_sgs at floor peak    0.368 - 0.564     0.008 (h -> 2500 m)    >= 0.25       1.5x
  G2a integral saturation   |d| <= 3.4e-4     uncapped: climbs       <= 1e-3       2.9x
  G2b integral magnitude    0.858 - 1.208     1.641 (stage6_raw)     0.6 - 1.5     1.2x
  G3a half-vs-half dpeak    0 - 16 m (1 cell) 480 m (g24_wS_iso)     <= 32 m       2.0x
  G3b peak / Kljun peak     0.67 - 1.33       2.43 - 7.40            0.4 - 2.5     1.8x

G2a is the sharp one and G2b is coarse, deliberately. The wrap cap makes the integral
converge FROM BELOW, so saturation by 1.0 L is a statement about whether the cap is
binding -- and "an integral that crosses 1 and keeps climbing cannot be truncation, so it
is always a model inconsistency" (PROJECT_BRIEF.md). The magnitude, by contrast, is legitimately
not 1: over a slope the residual is w_bar times the concentration integral, which is the
advection non-closure that makes EC hard in complex terrain. So the band is wide and the
value is quoted against Kljun on the identical cells rather than against 1.

G3a's tolerance is two raster cells because the peak's own sampling p90 is **0 m** at this
receptor -- one cell is already the smallest disagreement the raster can express, so two is
the smallest tolerance that cannot fail on quantisation alone.

usage: corpus_monitor.py results/corpus/case_*.json [--json FILE] [--quiet]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.sgs_floor import FSGS_AT_PEAK_MIN            # noqa: E402

SAT_TOL = 1e-3          # |I(2L)/I(1L) - 1|; measured max 3.4e-4 over 60 capped records
INT_LO, INT_HI = 0.6, 1.5
DPEAK_CELLS = 2.0       # raster cells; the peak's own sampling p90 is 0 m
PEAK_RATIO_LO, PEAK_RATIO_HI = 0.4, 2.5


def score(d, cell_m=16.0):
    """Gate one case JSON. Returns (name, ok, detail) rows plus a reported dict."""
    les, klj = d.get("les") or {}, d.get("kljun") or {}
    fl = d.get("floor") or {}
    health = fl.get("health") or {}
    rows = []

    # ---- G1: the floor repaired a deficit, not a broken input ----------------------
    if health:
        ok = bool(health.get("ok"))
        f = health.get("f_sgs_at_peak", float("nan"))
        rows.append(("G1 floor health", ok,
                     f"f_sgs at the floor's peak {f:.3f} (>= {FSGS_AT_PEAK_MIN}), "
                     f"factor {health.get('fac_min', float('nan')):.3f}-"
                     f"{health.get('fac_max', float('nan')):.4g} peaking at "
                     f"z={health.get('z_fac_max', float('nan')):.0f} m, inert above "
                     f"z={health.get('z_inert', float('nan')):.0f} m"
                     + ("" if ok else "  || " + " || ".join(health.get("alarms", [])))))
    else:
        # NOT A PASS. A case with no health block was produced before the invariant
        # existed, or by a path that skipped the floor; either way it is unjudged, and
        # PROJECT_BRIEF.md's rule is that a regime with no evidence is no evidence, not good news.
        rows.append(("G1 floor health", None,
                     "no health block in this JSON -- the case predates the invariant or "
                     "ran without --sgs-most; UNJUDGED, not passed"))

    # ---- G2a: the wrap cap is binding ---------------------------------------------
    bd = d.get("by_disp") or []
    i1 = next((b["integral"] for b in bd if abs(b.get("frac_of_Lx", 0) - 1.0) < 1e-6), None)
    i2 = next((b["integral"] for b in bd if b.get("frac_of_Lx", 0) >= 1.9), None)
    if i1 and i2:
        sat = i2 / i1 - 1.0
        note = ("binding" if abs(sat) <= SAT_TOL else
                "NOT binding -- trajectories are still gaining influence past one domain "
                "length, which is wrap-around double counting, not truncation")
        rows.append(("G2a integral saturates", bool(abs(sat) <= SAT_TOL),
                     f"I(2L)/I(1L) - 1 = {sat:+.2e} (|.| <= {SAT_TOL:.0e}); "
                     f"the cap is {note}"))
    else:
        rows.append(("G2a integral saturates", None,
                     "no by_disp curve in this JSON; UNJUDGED"))

    # ---- G2b: magnitude, against Kljun on the identical cells AND the asymptote -----
    # THE FLUX-FOOTPRINT INTEGRAL DOES NOT ASYMPTOTE TO 1. Steinfeld et al. (2008), after
    # Horst & Weil (1992): the ceiling is 1 - z_m/z_i, because the fraction z_m/z_i of the
    # column lies below the receptor and its flux never crosses it. At the retired 10 m
    # receptor that was 1.25% and invisible; at 30 m in an 800 m CBL it is 3.75%, the size
    # of effects this project gates on. Kljun on the identical cells stays the PRIMARY
    # reference because it also carries the domain truncation, which the asymptote does
    # not; the asymptote is the physical ceiling the pair should sit under.
    I = d.get("integral_les")
    Ik = d.get("integral_kljun")
    A = d.get("integral_asymptote")
    if A is None:
        zm = d.get("zm_agl") or d.get("zm")
        h = ((d.get("stats") or {}).get("h"))
        if zm and h:
            A = 1.0 - float(zm) / float(h)
    if I is not None:
        extra = (f"; Kljun on the same cells {Ik:.3f}" if Ik else "")
        if A:
            extra += (f"; asymptote 1 - z_m/z_i = {A:.4f}, LES/asymptote {I/A:.3f}")
        rows.append(("G2b integral magnitude", bool(INT_LO <= I <= INT_HI),
                     f"{I:.3f} in [{INT_LO}, {INT_HI}]" + extra))
    # G2c is REPORTED, NOT GATED, and deliberately so. An integral above the asymptote is
    # not automatically wrong -- over sloping ground the residual is w_bar times the
    # concentration integral and the footprint genuinely need not integrate to the
    # asymptote (PROJECT_BRIEF.md) -- but it is the shape a broken closure makes, so the number
    # belongs in every record.
    if I is not None and A:
        rows.append(("G2c integral vs asymptote", None,
                     f"LES/asymptote {I/A:.3f}"
                     + (f", Kljun/asymptote {Ik/A:.3f}" if Ik else "")
                     + " -- REPORTED, not gated (a slope moves it legitimately)"))

    # ---- G3a: the peak is converged within the window ------------------------------
    dpk = (d.get("halves") or {}).get("dpeak")
    if dpk is not None:
        tol = DPEAK_CELLS * cell_m
        rows.append(("G3a peak converged", bool(abs(dpk) <= tol),
                     f"half-vs-half |dpeak| {abs(dpk):.0f} m <= {tol:.0f} m "
                     f"({DPEAK_CELLS:.0f} raster cells)"))

    # ---- G3b: the peak is where a near-field footprint puts it ---------------------
    pl, pk = les.get("peak_x"), klj.get("peak_x")
    if pl is not None and pk:
        r = pl / pk
        rows.append(("G3b peak vs Kljun", bool(PEAK_RATIO_LO <= r <= PEAK_RATIO_HI),
                     f"LES peak {pl:.0f} m / Kljun {pk:.0f} m = {r:.2f}x in "
                     f"[{PEAK_RATIO_LO}, {PEAK_RATIO_HI}]"))

    share = (d.get("cover_share") or {}).get("solar array")
    sh_se = (d.get("cover_share_se") or {}).get("solar array")
    reported = dict(
        array_share=share, array_share_se=sh_se,
        centroid_m=les.get("centroid_dist"), centroid_bearing=les.get("centroid_bearing"),
        area80_ha=les.get("area80_ha"), peak_x=pl, integral=I, integral_kljun=Ik,
        overlap_kljun=d.get("overlap_kljun"), overlap_halves=(d.get("halves") or {}).get("overlap"),
        wrapped_fraction=d.get("wrapped_fraction"),
        h=fl.get("h"), fac_max=health.get("fac_max"),
        f_sgs_at_peak=health.get("f_sgs_at_peak"), z_fac_max=health.get("z_fac_max"))
    return rows, reported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--cell-m", type=float, default=16.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true", help="one line per case")
    a = ap.parse_args()

    paths = sorted({p for c in a.cases for p in (glob.glob(c) or [c])})
    out, n_ok, n_bad, n_unjudged = [], 0, 0, 0
    for p in paths:
        try:
            d = json.load(open(p))
        except (OSError, ValueError) as e:
            print(f"  {os.path.basename(p)}: UNREADABLE ({e})")
            n_bad += 1
            continue
        # THE CELL SIZE IS IN THE RECORD. stage5_footprint.py writes `res`, the raster
        # resolution it actually used; taking the CLI default instead scored a 24 m grid
        # against a 16 m cell -- conservative here, but the same defect as
        # docs/FASTEDDY_TRAPS.md 19 and it would be the wrong sign on a coarser grid.
        rows, rep = score(d, float(d.get("res") or a.cell_m))
        bad = [r for r in rows if r[1] is False]
        unj = [r for r in rows if r[1] is None]
        verdict = "FAIL" if bad else ("UNJUDGED" if unj else "OK")
        n_ok += verdict == "OK"
        n_bad += verdict == "FAIL"
        n_unjudged += verdict == "UNJUDGED"
        out.append(dict(case=os.path.basename(p), verdict=verdict,
                        gates=[dict(name=r[0], ok=r[1], detail=r[2]) for r in rows],
                        reported=rep))
        if a.quiet:
            print(f"  {verdict:8} {os.path.basename(p):32} peak {rep['peak_x'] or -1:4.0f} m  "
                  f"I {rep['integral'] or float('nan'):.3f}  "
                  f"share {100*(rep['array_share'] or 0):5.1f}%  "
                  f"fac<={rep['fac_max'] or float('nan'):.3g}")
            for r in bad + unj:
                print(f"           -> {r[0]}: {r[2]}")
            continue
        print(f"\n=== {os.path.basename(p)}: {verdict} ===")
        for nm, ok, det in rows:
            tag = {True: "ok  ", False: "FAIL", None: "??  "}[ok]
            print(f"  [{tag}] {nm:24} {det}")
        print("  --- REPORTED, not gated (the share is the result, not the detector) ---")
        print(f"    array share {100*(rep['array_share'] or 0):.2f}%"
              + (f" +/- {100*rep['array_share_se']:.2f}" if rep["array_share_se"] else "")
              + f", centroid {rep['centroid_m'] or float('nan'):.0f} m at "
                f"{rep['centroid_bearing'] or float('nan'):.1f} deg, A80 "
                f"{rep['area80_ha'] or float('nan'):.2f} ha, overlap vs Kljun "
                f"{rep['overlap_kljun'] or float('nan'):.2f}, wrapped "
                f"{100*(rep['wrapped_fraction'] or 0):.1f}%, h {rep['h'] or float('nan'):.0f} m")

    print(f"\n=== {len(paths)} case(s): {n_ok} OK, {n_bad} FAIL, {n_unjudged} UNJUDGED ===")
    # ---- cross-case spread: what a corpus monitor is actually for ------------------
    if len(out) > 1:
        keys = ("peak_x", "integral", "array_share", "centroid_m", "area80_ha",
                "f_sgs_at_peak", "fac_max")
        print(f"  {'metric':16}{'min':>10}{'median':>10}{'max':>10}{'spread':>10}")
        for k in keys:
            v = np.array([o["reported"].get(k) for o in out
                          if o["reported"].get(k) is not None], float)
            if v.size < 2:
                continue
            sp = (v.max() / v.min()) if v.min() > 0 else float("nan")
            print(f"  {k:16}{v.min():10.4g}{np.median(v):10.4g}{v.max():10.4g}{sp:10.2f}x")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(dict(n=len(paths), ok=n_ok, fail=n_bad, unjudged=n_unjudged,
                       cases=out), open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
