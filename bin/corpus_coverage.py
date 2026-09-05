#!/usr/bin/env python3
"""How many days does the domain actually accept, and which ones?

The 1952 m box holds `z_i <= 976 m` at `L >= 2 z_i` and needs `z_i >= 100 m` for the 10 m
receptor to stay in the surface layer. Both bounds bite, at opposite ends of the day and
of the year, and neither is a random trim:

  * `z_i` and surface heat flux correlate at **+0.43** at this site, so the deep hours the
    box cannot hold carry **1.51x** the heat flux and **1.58x** the `w*` of the ones it
    can. The corpus is thinnest exactly where the array's flux enhancement is largest.
  * The shallow rejections are nocturnal, and stable cases carry no array signal anyway
    (`bin/case_surface.py`) -- so losing them costs diversity in the flow, not in the
    array response.

docs/problem/site.md quotes 60.9% of convective midday from CONUS404's PBLH. This measures the same
thing on the diagnostic the corpus ACTUALLY filters with (HRRR's HPBL), over the hours the
corpus actually draws, which is the only version of the number that predicts a corpus size.

usage: corpus_coverage.py [--forcing results/forcing] [--glob 'yr_*'] [--out FILE]
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcing", default="results/forcing")
    ap.add_argument("--glob", default="yr_*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for p in sorted(_glob.glob(os.path.join(a.forcing, a.glob + ".json"))):
        d = json.load(open(p))
        vt = d["provenance"]["valid_time"]
        l = d["labels"]
        rows.append(dict(t=vt, month=int(vt[5:7]), hour=int(vt[11:13]),
                         zi=float(l["zi_m"]), wth=float(l["wth_virtual"]),
                         ok=bool(d.get("representable")),
                         lo=float(l["zi_min_m"]), hi=float(l["zi_max_m"])))
    if not rows:
        print(f"no forcing records matching {a.glob} under {a.forcing}", file=sys.stderr)
        return 2

    zi = np.array([r["zi"] for r in rows])
    ok = np.array([r["ok"] for r in rows])
    wth = np.array([r["wth"] for r in rows])
    hr = np.array([r["hour"] for r in rows])
    mo = np.array([r["month"] for r in rows])
    lo, hi = rows[0]["lo"], rows[0]["hi"]

    out = []
    p = out.append
    p(f"Corpus coverage from {len(rows)} sampled cases "
      f"({a.forcing}/{a.glob}.json)")
    p(f"  the box accepts {lo:.0f} m <= z_i <= {hi:.0f} m")
    p(f"  ACCEPTED {ok.sum()}/{len(rows)} = {100*ok.mean():.1f}%")
    p(f"    too deep  (z_i > {hi:.0f} m): {(zi > hi).sum():3d} = {100*(zi>hi).mean():.1f}%")
    p(f"    too shallow (z_i < {lo:.0f} m): {(zi < lo).sum():3d} = {100*(zi<lo).mean():.1f}%")
    p("")
    p("=== by UTC hour (local = UTC - 6 CST / - 5 CDT) ===")
    p(f"  {'hour':>5}{'n':>5}{'accepted':>10}{'z_i p50':>9}{'wth p50':>9}")
    for h in range(0, 24, 2):
        m = (hr // 2) == (h // 2)
        if not m.any():
            continue
        p(f"  {h:3d}-{h+1:<2d}{m.sum():5d}{100*ok[m].mean():9.0f}%"
          f"{np.median(zi[m]):9.0f}{np.median(wth[m]):9.4f}")
    p("")
    p("=== by month ===")
    p(f"  {'month':>6}{'n':>5}{'accepted':>10}{'z_i p50':>9}")
    for m_ in range(1, 13):
        m = mo == m_
        if not m.any():
            continue
        p(f"  {m_:6d}{m.sum():5d}{100*ok[m].mean():9.0f}%{np.median(zi[m]):9.0f}")
    p("")
    p("=== the bias of what is excluded ===")
    if (~ok).any() and ok.any():
        deep = zi > hi
        if deep.any():
            p(f"  rejected-as-too-deep carry {np.mean(wth[deep])/max(np.mean(wth[ok]),1e-9):.2f}x "
              f"the virtual heat flux of the accepted set "
              f"({np.mean(wth[deep]):.4f} vs {np.mean(wth[ok]):.4f} K m/s)")
        p(f"  rank correlation z_i vs flux over the sample: "
          f"{np.corrcoef(np.argsort(np.argsort(zi)), np.argsort(np.argsort(wth)))[0,1]:+.3f}")
    p("  THIS IS NOT A NEUTRAL TRIM. State it wherever the corpus is described.")
    p("")
    n5 = int(round(365.25 * 5 * ok.mean()))
    p(f"=== what a five-year, one-per-day corpus comes to ===")
    p(f"  {365.25*5:.0f} days x {100*ok.mean():.1f}% = ~{n5} usable cases")
    p(f"  at 1.132 GPU-h each that is ~{n5*1.132:.0f} GPU-h, plus 52 for the seed library")

    txt = "\n".join(out)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
