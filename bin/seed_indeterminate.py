#!/usr/bin/env python3
"""INDETERMINATE is a verdict, not a diagnosis. Is the limit TRENDING or is it NOISY?

THE QUESTION THIS ANSWERS. Every seed in the library returns INDETERMINATE on several of
the seven stationarity limits: the trend is inside its threshold but not by the three
standard errors the gate requires before it will assert anything. That is an honest verdict
and a useless one on its own, because it does not say what would fix it. Two very different
situations produce it:

  TRENDING  the quantity is genuinely still moving, the estimate is sharpening as the run
            lengthens, and a longer run resolves it -- into a PASS if the trend is decaying
            toward the threshold, into a FAIL if it is growing away from it. Longer runs
            buy something.
  NOISY     the trend estimate scatters about a value that is already inside the threshold,
            and what blocks the verdict is the STANDARD ERROR rather than the trend. If that
            standard error does not shrink as the scoring window widens -- because the
            quantity decorrelates on the eddy turnover rather than on the dump interval, so
            n_eff saturates -- then a longer run buys nothing and INDETERMINATE is simply
            the library's state.

THE DISCRIMINATOR IS THE STANDARD ERROR'S BEHAVIOUR, NOT THE TREND'S. For an ordinary least
squares slope over a window of duration T with n_eff effective samples,

    SE ~ sigma sqrt(12) / (T sqrt(n_eff))

so widening the window helps through T even when n_eff saturates -- UNLESS the residual
scatter sigma grows with T, which is what a slowly wandering mean flow does. Measuring
SE(T) therefore separates the two cases directly, and it costs nothing: the dumps are
already on disk.

usage: seed_indeterminate.py jobs24/seed_nbl-deep_a000 [jobs24/seed_cbl-deep_a000 ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# How far SE must fall over the swept width range before "widening helps" is a claim rather
# than scatter. The ideal 1/T scaling over 1.0 -> 2.5 h is a factor of 2.5; half of that is
# a generous bar that still excludes a flat SE.
SE_FALL_MIN = 1.25
# A trend counts as SYSTEMATIC when it moves monotonically across the swept widths by more
# than the SE at the widest one -- i.e. the movement is bigger than the noise on the best
# estimate available.
def classify(widths, trends, ses, limit):
    """TRENDING (toward / away), NOISY, or RESOLVED, from the width sweep."""
    t = np.asarray(trends, float); s = np.asarray(ses, float); w = np.asarray(widths, float)
    good = np.isfinite(t) & np.isfinite(s) & (s > 0)
    if good.sum() < 3:
        return "TOO FEW", "", np.nan, np.nan
    t, s, w = t[good], s[good], w[good]
    se_fall = s[0] / s[-1]
    margin = (limit - abs(t[-1])) / s[-1]          # SE of separation at the widest window
    # monotone in |trend|? use the signed value, since a sign flip IS scatter
    d = np.diff(t)
    monotone = bool(np.all(d > 0) or np.all(d < 0))
    moved = abs(t[-1] - t[0])
    systematic = monotone and moved > s[-1]
    if margin >= 3.0:
        return "RESOLVED (in band)", "", se_fall, margin
    if abs(t[-1]) > limit and margin <= -3.0:
        return "RESOLVED (drifting)", "", se_fall, margin
    if systematic:
        toward = abs(t[-1]) < abs(t[0])
        return ("TRENDING toward band" if toward else "TRENDING AWAY from band"), \
               f"|trend| {abs(t[0]):.2f} -> {abs(t[-1]):.2f} over {w[0]:.1f}-{w[-1]:.1f} h", \
               se_fall, margin
    if se_fall >= SE_FALL_MIN:
        return "NOISY, but SE is shrinking", \
               f"SE {s[0]:.2f} -> {s[-1]:.2f} ({se_fall:.1f}x); a longer run would resolve it", \
               se_fall, margin
    return "NOISY, SE-LIMITED", \
           f"SE {s[0]:.2f} -> {s[-1]:.2f} ({se_fall:.2f}x); widening buys nothing", \
           se_fall, margin


def sweep(job, widths):
    man = json.load(open(os.path.join(job, "manifest.json")))
    dt = float(man["run"]["dt"]); base = man["run"]["outFileBase"]
    g = man.get("gate", {}); grid = man.get("grid", {})
    paths = sorted(glob.glob(os.path.join(job, "output", base + ".[0-9]*")),
                   key=lambda p: int(p.rsplit(".", 1)[1]))
    out = {}
    here = os.path.dirname(os.path.abspath(__file__))
    for w in widths:
        with tempfile.TemporaryDirectory() as td:
            js = os.path.join(td, "g.json")
            subprocess.run([sys.executable, os.path.join(here, "seed_stationarity.py"),
                            os.path.join(job, "output"), "--dt", str(dt),
                            "--wth", str(man["target"]["wth_virtual"]),
                            "--zm", str(g.get("zm", 10.0)), "--k", str(g.get("k", 2)),
                            "--dx", str(grid.get("dx", 16.0)),
                            "--score-h", str(w), "--json", js], capture_output=True)
            if not os.path.exists(js):
                continue
            for r in json.load(open(js))["gated"]:
                out.setdefault(r["name"], dict(limit=r["limit"], w=[], t=[], se=[],
                                               neff=[]))
                out[r["name"]]["w"].append(w)
                out[r["name"]]["t"].append(r["trend_pct_per_h"])
                out[r["name"]]["se"].append(r["trend_se_pct_per_h"])
                out[r["name"]]["neff"].append(r.get("n_eff", np.nan))
    return man, out, len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--widths", default="1.0,1.5,2.0,2.5")
    ap.add_argument("--out", default="results/seed_indeterminate.txt")
    a = ap.parse_args()
    widths = [float(x) for x in a.widths.split(",")]

    lines = []
    P = lambda s="": (print(s), lines.append(s))
    P("INDETERMINATE: TRENDING OR NOISY?  Generated by bin/seed_indeterminate.py.")
    P("")
    P("  The gate refuses a verdict when the threshold sits within 3 SE of the trend. This")
    P("  asks WHY, by sweeping the scoring-window width and watching what moves: the trend")
    P("  (the quantity is still settling) or the standard error (the estimate is sharpening).")
    P("  SE ~ sigma sqrt(12)/(T sqrt(n_eff)), so a window twice as wide should halve the SE")
    P("  unless the residual scatter grows with it -- which is exactly what a slowly")
    P("  wandering mean flow does.")
    summary = {}
    for job in a.jobs:
        man, sw, ndump = sweep(job, widths)
        P("")
        P(f"=== {man['job']} ({man['regime']}, {ndump} dumps, "
          f"{man['run']['sim_hours']:.2f} sim-h) ===")
        P(f"  {'quantity':<28}{'limit':>6}  " +
          "".join(f"{'t@%.1f' % w:>9}" for w in widths) + "  " +
          "".join(f"{'SE@%.1f' % w:>8}" for w in widths) +
          f"{'n_eff':>16}  verdict")
        for nm, d in sw.items():
            v, why, se_fall, margin = classify(d["w"], d["t"], d["se"], d["limit"])
            summary.setdefault(v, []).append(f"{man['rung']}:{nm}")
            P(f"  {nm:<28}{d['limit']:6.1f}  "
              + "".join(f"{x:+9.2f}" for x in d["t"]) + "  "
              + "".join(f"{x:8.2f}" for x in d["se"])
              + f"  {min(d['neff']):5.1f}-{max(d['neff']):<5.1f}  {v}")
            if why:
                P(f"  {'':<34}{why}")
        P(f"  margin at the widest window is (limit - |trend|)/SE; the gate needs 3.")
    P("")
    P("=== WHAT FOLLOWS ===")
    for v in sorted(summary):
        P(f"  {v}:")
        for q in summary[v]:
            P(f"      {q}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
