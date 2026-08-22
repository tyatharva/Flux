#!/usr/bin/env python3
"""What fraction of the site's convective states fit in the box -- and how biased the rest is.

A doubly-periodic CBL wants L >= 4 z_i or the largest thermals lock to the domain, which
caps z_i at L/4. That cap is usually quoted as a coverage number and left there. It is
worth more than that, because z_i and surface heat flux are POSITIVELY CORRELATED: the
deep boundary layers the box cannot hold are the strongly-heated ones, which are exactly
the states where the solar array's sensible-flux enhancement is largest. A z_i-capped
corpus is therefore not a random subsample of convective conditions -- it is thinnest
precisely where the effect this project exists to measure is strongest.

This quantifies both halves: the coverage, and the bias of what is left out.

usage: zi_coverage.py [--in results/conus404_site.npz] [--out results/zi_coverage.txt]
                      [--L 1952] [--alt 3488]
"""
import argparse
import os
import sys

import numpy as np

G, THETA0 = 9.81, 300.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="results/conus404_site.npz")
    ap.add_argument("--out", default="results/zi_coverage.txt")
    ap.add_argument("--L", type=float, default=1952.0, help="domain length, m")
    ap.add_argument("--alt", type=float, default=3488.0,
                    help="the fallback domain length to compare against (218^2 @ 16 m)")
    a = ap.parse_args()

    d = np.load(a.src, allow_pickle=True)
    qc = d["qc"]
    zi, wth, zL = d["pblh"][qc], d["wth"][qc], d["zL"][qc]
    us, lst = d["ustar"][qc], d["lst"][qc]
    # The convective-midday reference used throughout PROJECT_BRIEF.md: local 10-16 h with a
    # surface heat flux clearly above the noise. This is the population the array signal
    # lives in, so it is the one the coverage question is really about.
    mid = (lst >= 10) & (lst < 16) & (wth > 0.05)
    pops = (("all QC hours", np.ones_like(zi, dtype=bool)),
            ("unstable (z/L < 0)", zL < 0),
            ("very unstable (z/L < -0.5)", zL < -0.5),
            ("convective midday (10-16 h, w'th' > 0.05)", mid))

    out = []
    p = out.append
    p(f"CONUS404 at the tower: {int(qc.sum()):,} quality-controlled hours "
      f"of {qc.size:,} ({100*qc.mean():.1f}%)")
    p("")
    p("=== z_i coverage: fraction of hours the box can hold ===")
    p(f"{'population':<44}" + "".join(f"{f'L>={r} z_i':>13}" for r in (4, 3, 2)))
    for L, tag in ((a.L, f"{a.L:.0f} m box"), (a.alt, f"{a.alt:.0f} m box (fallback)")):
        p(f"  -- {tag}: caps z_i at " +
          ", ".join(f"{L/r:.0f} m (L>={r}z_i)" for r in (4, 3, 2)))
        for name, sel in pops:
            row = "".join(f"{100*(zi[sel] < L/r).mean():12.1f}%" for r in (4, 3, 2))
            p(f"  {name:<42}" + row)
    p("")
    p("=== bias of what the cap excludes (convective midday) ===")
    p("z_i and surface heat flux are positively correlated, so the cap is not a random")
    p("subsample: it removes the strongly-heated states preferentially.")
    p(f"  rank correlation(z_i, w'th') over convective midday: "
      f"{np.corrcoef(np.argsort(np.argsort(zi[mid])), np.argsort(np.argsort(wth[mid])))[0,1]:+.3f}")
    p("")
    for r in (4, 2):
        cap = a.L / r
        inn, ext = mid & (zi < cap), mid & (zi >= cap)
        if not (inn.any() and ext.any()):
            continue
        p(f"  -- L >= {r} z_i, cap {cap:.0f} m --")
        hdr = "w'th' p50"
        p(f"     {'':<26}{'n':>7}{'z_i p50':>10}{hdr:>12}{'u* p50':>9}{'w* p50':>9}")
        wsm = {}
        for nm, m in (("representable (z_i<cap)", inn), ("EXCLUDED (z_i>=cap)", ext)):
            ws = (G / THETA0 * np.median(wth[m]) * np.median(zi[m])) ** (1.0 / 3.0)
            wsm[nm] = (float(np.median(wth[m])), ws)
            p(f"     {nm:<26}{int(m.sum()):7d}{np.median(zi[m]):9.0f}m"
              f"{np.median(wth[m]):12.4f}{np.median(us[m]):9.3f}{ws:9.2f}")
        (w0, s0), (w1, s1) = wsm["representable (z_i<cap)"], wsm["EXCLUDED (z_i>=cap)"]
        p(f"     excluded/representable at the medians:  w'th' {w1/w0:.2f}x   w* {s1/s0:.2f}x")
        p("")
    p("READ THIS AS: the cap costs coverage AND skews the corpus toward weak convection.")
    p("Whether it must be paid is a separate, measurable question -- see the domain")
    p("adequacy test (bin/domain_adequacy.py), which asks whether L = 2 z_i actually")
    p("corrupts a 10 m footprint or only the w*-scaling the 4 z_i rule was written for.")

    txt = "\n".join(out)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
