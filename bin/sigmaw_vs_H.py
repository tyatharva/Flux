#!/usr/bin/env python3
"""What does the TOWER measure for sigma_w at a given surface heat flux?

An EXTERNAL check on the LES. Everything else in this project is internally consistent by
construction -- the LPDM is driven by FastEddy's own fields, the closure is anchored to
FastEddy's own sub-grid TKE, and the gates compare the model against itself. This compares
the LES against an INSTRUMENT: data/raw/H_and_sigma_w.csv is one year of half-hourly eddy
covariance at the actual receptor, and it is not used for training, tuning or forcing.

It is a weak check and that is stated rather than hidden. sigma_w depends on u* far more
than on H, and the file carries no wind speed, so conditioning on H alone leaves a wide
band -- the interquartile range at H ~ 0 spans roughly a factor of two. What it CAN catch
is an order-of-magnitude error, a sign error, or an LES whose near-surface variance has
collapsed. It cannot validate a 20% difference and must not be quoted as if it could.

usage: sigmaw_vs_H.py --h 0.0 [--band 10.0] [--les 0.35]
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, required=True, help="this case's sensible H, W/m2")
    ap.add_argument("--band", type=float, default=10.0, help="+/- W/m2 around it")
    ap.add_argument("--les", type=float, default=None, help="the LES sigma_w at 10 m, m/s")
    ap.add_argument("--csv", default="data/raw/H_and_sigma_w.csv")
    a = ap.parse_args()

    H, S = [], []
    for row in csv.DictReader(open(a.csv)):
        try:
            h_, s_ = float(row["H"]), float(row["sigma_w"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(h_) and np.isfinite(s_) and s_ > 0:
            H.append(h_); S.append(s_)
    H, S = np.asarray(H), np.asarray(S)
    m = np.abs(H - a.h) <= a.band
    if m.sum() < 30:
        print(f"FATAL: only {int(m.sum())} records within +/-{a.band} W/m2 of {a.h}",
              file=sys.stderr)
        return 2
    s = S[m]
    q = np.percentile(s, [5, 25, 50, 75, 95])
    print(f"=== tower sigma_w at H = {a.h:+.0f} +/- {a.band:.0f} W/m2 "
          f"({int(m.sum())} half-hours of {H.size}) ===")
    print(f"  p5 {q[0]:.3f}   p25 {q[1]:.3f}   MEDIAN {q[2]:.3f}   p75 {q[3]:.3f}   "
          f"p95 {q[4]:.3f}  m/s")
    print(f"  mean {s.mean():.3f}, sd {s.std(ddof=1):.3f}")
    if a.les is not None:
        pct = 100.0 * (s < a.les).mean()
        inside = q[1] <= a.les <= q[3]
        print(f"\n  LES sigma_w(10 m) = {a.les:.3f} m/s")
        print(f"  -> the {pct:.0f}th percentile of the measured distribution at this H")
        print(f"  -> {'INSIDE' if inside else 'OUTSIDE'} the measured interquartile range "
              f"[{q[1]:.3f}, {q[3]:.3f}]")
        print(f"  -> ratio to the measured median: {a.les/q[2]:.2f}x")
        print()
        print("  READ THIS AS AN ORDER-OF-MAGNITUDE CHECK, NOT A VALIDATION. sigma_w scales")
        print("  with u*, the file carries no wind speed, and the IQR here already spans a")
        print(f"  factor of {q[3]/q[1]:.2f}. A result inside the IQR is consistent with the")
        print("  instrument; it does not demonstrate agreement to better than that width.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
