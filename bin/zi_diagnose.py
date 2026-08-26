#!/usr/bin/env python3
"""Is a z_i trend the LAYER moving, or the THRESHOLD moving? They look identical.

z_i is diagnosed as the height where the resolved TKE profile falls below 5% of its own
peak (bin/seed_stationarity.py:81). That threshold is NORMALISED BY THE PEAK, so when the
peak changes the threshold changes with it and the reported depth moves even if the layer
does not. FASTEDDY_TRAPS.md 16 recorded one direction of this -- the peak GREW 25x during
a spin-up and z_i appeared to FALL 154 -> 76 m, and a healthy run was killed over it.

THE MIRROR IMAGE IS WORSE, because it fails a run instead of killing it, and it is what
seed_nbl-shallow_a000 hit: in a neutral Ekman layer u* falls for the first quarter of the
17.6 h inertial period (PROJECT_BRIEF.md: "Neutral stationarity is a statement about u(z_m)/u*,
not about u*"), the resolved TKE peak falls with it, the 5% threshold falls, and z_i is
reported as RISING at +11.67 %/h against a limit of 3.

This prints both diagnoses side by side and, crucially, the correlation between each and
the peak it might be inheriting. A depth that is strongly anti-correlated with its own
normaliser is measuring the normaliser.

IT DOES NOT PROPOSE A VERDICT. Which threshold the gate should use is a decision about
what "stationary" means for this project, not something a diagnostic gets to settle -- and
changing a gate immediately after it fails is how a gate stops meaning anything.

usage: zi_diagnose.py <output_dir> --dt 0.01461988 [--score-h 1.5] [--abs 0.01]
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
from netCDF4 import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("d")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--score-h", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=0.05, help="the gate's peak fraction")
    ap.add_argument("--abs", type=float, default=0.01,
                    help="a FIXED TKE threshold, m2/s2, for the comparison diagnosis")
    a = ap.parse_args()

    ps = sorted(glob.glob(f"{a.d}/*.[0-9]*"), key=lambda q: int(q.rsplit(".", 1)[1]))
    T, Z5, ZA, PK, US = [], [], [], [], []
    for q in ps:
        st = int(q.rsplit(".", 1)[1])
        if st == 0:
            continue
        with Dataset(q) as ds:
            g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
            u, v, w = g("u"), g("v"), g("w")
            z = g("zPos")[:, 0, 0]
            ust = float(g("fricVel").mean())
        pr = lambda arr: arr - arr.mean(axis=(-2, -1), keepdims=True)
        tk = 0.5 * ((pr(u) ** 2 + pr(v) ** 2 + pr(w) ** 2).mean(axis=(-2, -1)))
        k = int(np.argmax(tk))
        b5 = np.where(tk[k:] < a.frac * tk[k])[0]
        ba = np.where(tk[k:] < a.abs)[0]
        Z5.append(float(z[k + b5[0]]) if len(b5) else float(z[-1]))
        ZA.append(float(z[k + ba[0]]) if len(ba) else float(z[-1]))
        PK.append(float(tk[k])); US.append(ust); T.append(st * a.dt / 3600.0)
    if len(T) < 4:
        print(f"FATAL: only {len(T)} usable dumps in {a.d}", file=sys.stderr)
        return 2
    T, Z5, ZA, PK, US = map(np.asarray, (T, Z5, ZA, PK, US))
    m = T >= T[-1] - a.score_h
    tr = lambda y: 100.0 * np.polyfit(T[m], y[m], 1)[0] / abs(y[m].mean())

    print(f"=== z_i diagnosis: {a.d}, last {a.score_h} h of {T[-1]:.2f} ===")
    print(f"  {'quantity':<42}{'mean':>10}{'trend %/h':>12}")
    print(f"  {'z_i, ' + str(int(100*a.frac)) + '% of the RUNNING PEAK  [GATED]':<42}"
          f"{Z5[m].mean():>10.1f}{tr(Z5):>+12.2f}")
    print(f"  {'z_i, fixed threshold ' + f'{a.abs:g} m2/s2':<42}{ZA[m].mean():>10.1f}{tr(ZA):>+12.2f}")
    print(f"  {'peak resolved TKE (the normaliser)':<42}{PK[m].mean():>10.4f}{tr(PK):>+12.2f}")
    print(f"  {'u*':<42}{US[m].mean():>10.4f}{tr(US):>+12.2f}")
    settled = T > 1.0
    if settled.sum() > 3:
        c5 = np.corrcoef(Z5[settled], PK[settled])[0, 1]
        ca = np.corrcoef(ZA[settled], PK[settled])[0, 1]
        print(f"\n  correlation with the peak, after t = 1.0 h:")
        print(f"    peak-normalised z_i  {c5:+.3f}"
              + ("   <- it is measuring the normaliser" if c5 < -0.7 else ""))
        print(f"    fixed-threshold z_i  {ca:+.3f}")
        print(f"\n  the peak-normalised depth takes {len(set(Z5[settled]))} distinct values "
              f"after t = 1.0 h: it is a staircase on model levels, and a straight line "
              f"fitted through a staircase reports a trend whatever the layer does.")
    print("\n  NO VERDICT IS OFFERED. Which threshold defines z_i is a decision about what")
    print("  stationarity means here; a diagnostic does not get to settle it, and changing")
    print("  a gate right after it fails is how a gate stops meaning anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
