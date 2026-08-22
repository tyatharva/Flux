#!/usr/bin/env python3
"""Read t_back off the capture curve the control window already produced.

A WINDOW IS (30 min + t_back), so t_back sizes every production run -- and it cannot be
guessed, because the descent time scales as z/sigma_w and the receptor just moved from 30 m
to 10 m. It also cannot be measured before a window exists, which is why the control window
is run with a deliberately GENEROUS t_back: a window too short to contain the answer cannot
report that it was too short.

The curve is free. compute_footprint masks the touchdowns it already has on their AGE, so
every shorter t_back is scored from the same ensemble with no extra integration.

CONVERGENCE, defined against the run's own noise rather than a round number: the smallest
t_back at which the crosswind-integrated peak and x80 have both settled to within the
window's half-vs-half sampling floor of their fully-converged values, AND the flux integral
has reached `--frac` of the full-t_back value. Reporting a t_back tighter than the sampling
floor would be claiming a precision the estimator does not have.

usage: pick_tback.py results/g16_flat.json [--frac 0.97] [--out results/tback_production.txt]
"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--frac", type=float, default=0.97)
    ap.add_argument("--out", default="results/tback_production.txt")
    ap.add_argument("--pad", type=float, default=1.25,
                    help="safety factor on the converged value; production pays a little "
                         "more backward time than the minimum so a case with weaker "
                         "sigma_w than the control is still covered")
    a = ap.parse_args()
    d = json.load(open(a.json))
    rows = d.get("capture") or []
    if not rows:
        print("  no capture curve in the json -- rerun with --tback-marks"); return 1
    rows = sorted(rows, key=lambda r: r["t_back"])
    full = rows[-1]
    h = d.get("halves", {}) or {}
    # the run's own sampling floor, and never finer than one raster cell
    res = float(d.get("res", 16.0))
    tol_peak = max(abs(h.get("dpeak", 0.0) or 0.0), res)
    tol_x80 = max(abs(h.get("dx80", 0.0) or 0.0), res)
    print(f"  full t_back {full['t_back']:.0f} s: integral {full['integral']:.3f}, "
          f"peak {full['peak_x']:.0f} m, x80 {full['x80']:.0f} m")
    print(f"  tolerances from this window's own halves: peak {tol_peak:.0f} m, "
          f"x80 {tol_x80:.0f} m; integral must reach {100*a.frac:.0f}% of the full value")
    print(f"\n  {'t_back':>8}{'integral':>10}{'frac':>8}{'peak':>8}{'d peak':>9}"
          f"{'x80':>8}{'d x80':>9}   converged")
    pick = None
    for r in rows:
        dp = abs(r["peak_x"] - full["peak_x"])
        dx = abs(r["x80"] - full["x80"])
        conv = (dp <= tol_peak and dx <= tol_x80 and r["frac"] >= a.frac)
        if conv and pick is None:
            pick = r["t_back"]
        print(f"  {r['t_back']:8.0f}{r['integral']:10.3f}{r['frac']:8.3f}"
              f"{r['peak_x']:8.0f}{dp:9.0f}{r['x80']:8.0f}{dx:9.0f}   "
              f"{'yes' if conv else 'no'}{'  <-- first' if conv and pick == r['t_back'] else ''}")
    if pick is None:
        pick = full["t_back"]
        print(f"\n  NOT CONVERGED inside the curve -- falling back to the full "
              f"{pick:.0f} s. The control window was not long enough; rerun it longer.")
    prod = round(pick * a.pad / 50.0) * 50.0
    print(f"\n  converged at {pick:.0f} s; production t_back = {pick:.0f} x {a.pad} "
          f"= {prod:.0f} s (rounded to 50 s)")
    print(f"  -> a production window is 1800 + {prod:.0f} = {1800+prod:.0f} s")
    with open(a.out, "w") as f:
        f.write(f"{prod:.0f}\n")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
