#!/usr/bin/env python3
"""Does the floor invariant actually catch the defect it was written for, and only that?

THE DEFECT. On the first corpus case `h` fell through to the domain top (2500 m instead of
559). `h` sets the sigma_w floor's mixed-layer blend, so the floor ran at 3-20x between 35
and 200 m where it should have been 1.0, peaking near 9e4. It produced a plausible
footprint: the near-field peak was wrong by a full raster cell while the array share moved
0.8 points against its own 3.66-point SE. Nothing errored, and the receptor factor -- the
one number the driver printed prominently -- read 1.000 in both the broken and the correct
run.

WHAT THIS TEST ASSERTS, and both halves matter equally:

  1. SENSITIVITY. Replaying the SAME window with h forced to 800-2500 m must alarm.
  2. SPECIFICITY. Every production record on disk must pass -- including the four
     convective cases that legitimately reach a floor FACTOR of 5-10, which is why the
     invariant is not written on the factor.

Run: docker/pyrun.sh bin/test_floor_health.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.sgs_floor import FSGS_AT_PEAK_MIN, floor_health, most_floor   # noqa: E402


def st_from(rec):
    """Rebuild the most_floor input from a persisted case JSON."""
    fl, s0 = rec["floor"], rec.get("stats", {})
    return dict(zlev=np.array(fl["zl"], float), ww_prof=np.array(fl["wwp"], float),
                esgs_prof=np.array(fl["have"], float) * 1.5, ustar=fl["ustar"],
                L=fl["L"], h=fl["h"], htFlux=s0.get("htFlux", 0.0),
                theta0=s0.get("theta0", 300.0)), fl


def usable(p):
    try:
        d = json.load(open(p))
    except (OSError, ValueError):
        return None
    fl = d.get("floor")
    if not isinstance(fl, dict) or "wwp" not in fl or "have" not in fl or fl.get("legacy"):
        return None
    if d.get("sgs_subgrid_weight") is not True:
        return None          # production is the weighted closure; the rest is history
    return d


def main():
    recs = [(p, d) for p in sorted(glob.glob("results/**/*.json", recursive=True))
            if (d := usable(p)) is not None]
    if len(recs) < 8:
        print(f"FATAL: only {len(recs)} production floor records found; this test needs "
              f"the results/ tree", file=sys.stderr)
        return 2

    fails = []
    print(f"=== SPECIFICITY: {len(recs)} production records must all pass ===")
    print(f"  {'record':34}{'f_sgs@peak':>11}{'z_peak':>8}{'fac_max':>10}{'verdict':>9}")
    band = []
    for p, d in recs:
        st, fl = st_from(d)
        hh = floor_health(most_floor(st, d_r=fl["d_r"], mode=fl["mode"]))
        band.append(hh["f_sgs_at_peak"])
        v = "ok" if hh["ok"] else "ALARM"
        if not hh["ok"]:
            fails.append(f"specificity: {p} alarmed -- {'; '.join(hh['alarms'])}")
        print(f"  {p.replace('results/',''):34}{hh['f_sgs_at_peak']:11.3f}"
              f"{hh['z_delta_max']:8.1f}{hh['fac_max']:10.4g}{v:>9}")
    band = np.array(band, float)
    print(f"  band {band.min():.3f}-{band.max():.3f} against a {FSGS_AT_PEAK_MIN} "
          f"threshold: margin {band.min()/FSGS_AT_PEAK_MIN:.2f}x")

    # ---- SENSITIVITY: replay the real defect on the real window ---------------------
    src = "results/corpus/case_2023031014.json"
    print(f"\n=== SENSITIVITY: {os.path.basename(src)} replayed with h forced wrong ===")
    d = usable(src)
    if d is None:
        fails.append("sensitivity: the reference case is not on disk")
    else:
        st, fl = st_from(d)
        print(f"  {'h (m)':>8}{'f_sgs@peak':>11}{'z_peak':>8}{'fac_max':>12}{'verdict':>10}")
        for hh_m, want_alarm in ((fl["h"], False), (700., True), (800., True),
                                 (1000., True), (1500., True), (2500., True)):
            H = floor_health(most_floor(st | {"h": float(hh_m)}, d_r=fl["d_r"],
                                        mode=fl["mode"]))
            got = not H["ok"]
            tag = "ALARM" if got else "ok"
            print(f"  {hh_m:8.0f}{H['f_sgs_at_peak']:11.3f}{H['z_delta_max']:8.1f}"
                  f"{H['fac_max']:12.4g}{tag:>10}"
                  + ("" if got == want_alarm else "   <-- WRONG"))
            if got != want_alarm:
                fails.append(f"sensitivity: h={hh_m:.0f} expected "
                             f"{'ALARM' if want_alarm else 'ok'}, got {tag}")

    # ---- the INERT arm: a wrong h can switch the floor off instead of up ------------
    print(f"\n=== SENSITIVITY (inert arm): a neutral window whose floor h switches OFF ===")
    d2 = usable("results/g16r_nbl_wN.json")
    if d2 is not None:
        st2, fl2 = st_from(d2)
        for hh_m in (fl2["h"], 2000., 2500.):
            H = floor_health(most_floor(st2 | {"h": float(hh_m)}, d_r=fl2["d_r"],
                                        mode=fl2["mode"]))
            print(f"  h={hh_m:7.0f}  inflation_max {H['inflation_max']:.3e}  "
                  f"{'ALARM: ' + H['alarms'][0][:70] if not H['ok'] else 'ok'}")
            if hh_m > 1900 and H["ok"]:
                fails.append(f"inert arm: h={hh_m:.0f} switched the floor off silently")

    print()
    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        print(f"\n  FLOOR HEALTH TEST: FAIL ({len(fails)})")
        return 1
    print("  FLOOR HEALTH TEST: PASS -- sensitive to the real defect, silent on every "
          "production record including the convective ones with factor 5-10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
