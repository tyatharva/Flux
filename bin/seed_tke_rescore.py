#!/usr/bin/env python3
"""Is `TKE_BL/u*^2` measuring the turbulence, or measuring its own denominator?

    python3 bin/seed_tke_rescore.py [--library jobs30] [--out results/seed_tke_rescore.txt]

THE FOURTH INSTANCE OF THE SAME FAILURE CLASS. PROJECT_BRIEF.md's standing rule is that a
diagnostic whose DENOMINATOR or REFERENCE moves with anything but the quantity being
measured will report that movement as signal, and it lists three instances: `z_i` gated
against a running TKE peak, `TKE` averaged over the whole column, and `k0/k1` as a ratio of
two levels that die together. `TKE_BL/u*^2` IS the fix that was applied to the second of
those -- the column mean divides by a fixed 3000 m box and therefore rises with `z_i` even
in an equilibrated layer, so the average was moved inside the boundary layer.

That fix removed the `z_i` dependence of the DIVISOR. It did not make the diagnostic
scale-free, because two references are still moving:

    TKE_BL / u*^2  =  [ int_0^zi TKE dz / zi ]  /  u*^2
                        \______  ______/           \__ the inertial oscillation, which
                               \/                      falls ~10 %/h through the first
                        the averaging DEPTH,           quarter period and is NOT a
                        which entrains upward          statement about the turbulence

PROJECT_BRIEF.md already records that `u*` "fell for the first ~4.4 h (one quarter period)" of the
17.6 h inertial period at this latitude, and that this is why every OTHER gated limit is
written as a ratio -- `U/u*`, `sigma_v/u*`, `sigma_w/u*` -- whose numerator rides the
oscillation WITH the denominator and cancels it. `TKE_BL` does not ride it: it is an energy,
not a velocity scale carried by the mean flow, so `u*^2` in the denominator does not cancel.

WHAT THIS SCRIPT DOES, AND WHY IT NEEDS NO NEW GPU TIME. The returned seed artifacts carry
the gated ratio's trend and the reported trends of `u*`, `domain TKE` and `z_i`, but NOT the
absolute `TKE_BL` series -- `jobs/run_seed.sh` returns `stationarity.json`, which holds
verdicts and trends, not the per-dump series they were fitted to. The absolute numerator is
nonetheless recoverable from what IS there, by two routes with DISJOINT inputs:

    A.  trend(TKE_BL) = trend(TKE_BL/u*^2) + 2*trend(u*)
        Invert the ratio. To first order trend(N/D) = trend(N) - trend(D), and
        trend(u*^2) = 2*trend(u*).

    B.  trend(TKE_BL) = trend(domain TKE) - trend(z_i)
        De-normalise the column. Nearly all resolved TKE lies below z_i, so
        domain_TKE = int_0^H TKE dz / H ~= (zi/H) * TKE_BL.

A uses `u*`; B uses `domain TKE` and `z_i`. They share no input, so AGREEMENT BETWEEN THEM
IS THE EVIDENCE, and it is also the check on the linearisation: both are first-order in
(trend x window), and a window over which a quantity moves 60% is not obviously in that
regime. Where the two routes agree the recovery is trustworthy; where they do not, this
script says so and names `z_i` rather than reporting a mean of two disagreeing numbers.

THE VERDICT COLUMN IS NOT A GATE and nothing here re-accepts or re-refuses a seed. It
answers one question -- was the reported drift the turbulence or the reference -- so that
the library's real limitation is recorded in the currency it actually applies to.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

# The two routes must land within this of each other, in %/h, for the recovery to be
# reported as CLEAN. It is not a physics tolerance: it is the size of the second-order
# terms both routes drop, and it is set from the measured spread of (A - B) on the rungs
# where z_i is quiet -- 0.0 to 1.4 %/h -- rounded up.
AGREE_PCT_PER_H = 3.0

# Below this, in %/h, the absolute boundary-layer TKE is called STEADY. It is the gate's
# own limit on the ratio, reused so the two are read in the same units.
STEADY_PCT_PER_H = 5.0


def rung_of(job):
    return job.replace("seed_", "").rsplit("_", 1)[0]


def collect(library):
    """One row per seed, from the artifacts the seed run returned. No GPU, no re-run."""
    rows = []
    pats = (os.path.join(library, "*", "return", "stationarity.json"),
            os.path.join(library, "*", "stationarity.json"))
    seen = set()
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            j = json.load(open(p))
            job = os.path.basename(os.path.dirname(
                os.path.dirname(p) if p.endswith("return/stationarity.json") else p))
            if job in seen:
                continue
            seen.add(job)
            g = {x["name"]: x for x in j["gated"]}
            rep = {r["name"]: r for r in j["reported"]}
            need = ("TKE_BL/u*^2", "z_i")
            if any(k not in g for k in need) or "u*" not in rep or "domain TKE" not in rep:
                print(f"  SKIP {job}: the artifact does not carry the trends this needs",
                      file=sys.stderr)
                continue
            R = g["TKE_BL/u*^2"]["trend_pct_per_h"]
            U = rep["u*"]["trend_pct_per_h"]
            D = rep["domain TKE"]["trend_pct_per_h"]
            Z = g["z_i"]["trend_pct_per_h"]
            rows.append({
                "job": job, "rung": rung_of(job),
                "ratio": R, "ratio_se": g["TKE_BL/u*^2"]["trend_se_pct_per_h"],
                "ratio_verdict": g["TKE_BL/u*^2"]["verdict"],
                "ustar": U, "domain_tke": D, "zi": Z,
                "A": R + 2.0 * U, "B": D - Z,
            })
    return rows


def verdict(a, b):
    """(clean?, absolute trend, one-word reading). Never averages two disagreeing routes."""
    clean = abs(a - b) <= AGREE_PCT_PER_H
    v = 0.5 * (a + b)
    if not clean:
        return False, v, "UNRESOLVED"
    if abs(v) < STEADY_PCT_PER_H:
        return True, v, "STEADY"
    return True, v, "RISING" if v > 0 else "FALLING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="jobs30")
    ap.add_argument("--out", default="results/seed_tke_rescore.txt")
    ap.add_argument("--json", default="results/seed_tke_rescore.json")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    rows = collect(a.library)
    if not rows:
        raise SystemExit(f"FATAL: no seed artifacts with the needed trends under {a.library}")

    L = []
    P = L.append
    P("=== Is TKE_BL/u*^2 measuring the turbulence, or its own denominator? ===")
    P(f"  library {a.library}, {len(rows)} seeds, recovered from returned artifacts only.")
    P("")
    P("  A = trend(TKE_BL/u*^2) + 2*trend(u*)     invert the ratio")
    P("  B = trend(domain TKE)  -   trend(z_i)    de-normalise the column")
    P("  Disjoint inputs, so their AGREEMENT is the evidence. Both are first order in")
    P("  (trend x window); where they disagree by more than "
      f"{AGREE_PCT_PER_H:.0f} %/h the recovery is UNRESOLVED")
    P("  and is reported as such rather than averaged.")
    P("")
    P(f"  {'seed':24s} {'GATED':>7s} {'u*':>6s} {'z_i':>6s} | {'A':>7s} {'B':>7s} "
      f"{'|A-B|':>6s} | {'ABSOLUTE':>8s}  reading")
    P("  " + "-" * 96)
    cur = None
    for r in sorted(rows, key=lambda x: x["job"]):
        if r["rung"] != cur:
            cur = r["rung"]
            P(f"  --- {cur} ---")
        cl, v, w = verdict(r["A"], r["B"])
        P(f"  {r['job']:24s} {r['ratio']:+7.1f} {r['ustar']:+6.1f} {r['zi']:+6.1f} | "
          f"{r['A']:+7.1f} {r['B']:+7.1f} {abs(r['A']-r['B']):6.1f} | "
          f"{v:+8.1f}  {w}")

    P("")
    P("  === BY RUNG: what the gate reported, against what the turbulence did ===")
    P(f"  {'rung':14s} {'GATED':>7s} {'u*':>6s} {'z_i':>6s} | {'ABSOLUTE':>8s} {'|A-B|':>6s}"
      f"  reading")
    P("  " + "-" * 74)
    agg = defaultdict(list)
    for r in rows:
        agg[r["rung"]].append(r)
    out_rungs = {}
    for k, v in sorted(agg.items()):
        n = len(v)
        m = lambda f: sum(f(x) for x in v) / n            # noqa: E731
        A, B = m(lambda x: x["A"]), m(lambda x: x["B"])
        cl, val, w = verdict(A, B)
        out_rungs[k] = {"n": n, "gated": m(lambda x: x["ratio"]),
                        "ustar": m(lambda x: x["ustar"]), "zi": m(lambda x: x["zi"]),
                        "A": A, "B": B, "absolute": val, "reading": w, "clean": cl}
        P(f"  {k:14s} {m(lambda x: x['ratio']):+7.1f} {m(lambda x: x['ustar']):+6.1f} "
          f"{m(lambda x: x['zi']):+6.1f} | {val:+8.1f} {abs(A - B):6.1f}  {w}")

    P("")
    P("  === WHAT THIS SETTLES ===")
    P("  The gated ratio has THREE moving parts and only one of them is the turbulence:")
    P("")
    P("    numerator   int_0^zi TKE dz / zi   -- the averaging DEPTH entrains upward, so a")
    P("                                          layer of fixed integrated TKE reports a")
    P("                                          FALLING BL average as it deepens")
    P("    denominator u*^2                   -- falls through the first quarter of the")
    P("                                          17.6 h inertial period, which PROJECT_BRIEF.md")
    P("                                          already records at ~10 %/h")
    P("")
    P("  Moving the average inside the boundary layer removed the column mean's z_i")
    P("  dependence -- that fix was real -- but it did not make the diagnostic scale-free.")
    P("  It exchanged one reference for two.")
    P("")
    conv = [k for k in out_rungs if k.startswith("cbl")]
    neut = [k for k in out_rungs if k.startswith("nbl")]
    if conv:
        P("  CONVECTIVE rungs: the reported drift is the references, not the turbulence.")
        for k in sorted(conv):
            o = out_rungs[k]
            P(f"    {k:14s} gate {o['gated']:+6.1f} %/h   absolute {o['absolute']:+6.1f} %/h"
              f"   ({o['reading']}, u* {o['ustar']:+.1f}, z_i {o['zi']:+.1f})")
    if neut:
        P("")
        P("  NEUTRAL rungs: z_i and u* are both quiet, so the ratio DOES track the")
        P("  turbulence -- and it says these seeds are still spinning up at 2.0 sim-h.")
        for k in sorted(neut):
            o = out_rungs[k]
            P(f"    {k:14s} gate {o['gated']:+6.1f} %/h   absolute {o['absolute']:+6.1f} %/h"
              f"   ({o['reading']}, u* {o['ustar']:+.1f}, z_i {o['zi']:+.1f})")
    P("")
    P("  WHAT IS OWED FROM FUTURE SEED RUNS. This recovery is first order and it is")
    P("  arithmetic on trends, not a re-fit. `stationarity.json` returns verdicts and")
    P("  trends but NOT the per-dump series they came from, so the absolute TKE_BL trend")
    P("  cannot be fitted directly, its own standard error cannot be formed, and no")
    P("  verdict here carries an n_eff. Seed runs should return the scored series itself.")

    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    open(a.out, "w").write(txt)
    json.dump({"library": a.library, "agree_pct_per_h": AGREE_PCT_PER_H,
               "steady_pct_per_h": STEADY_PCT_PER_H,
               "seeds": rows, "rungs": out_rungs}, open(a.json, "w"), indent=1)
    print(txt)
    print(f"  -> {a.out}")
    print(f"  -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
