#!/usr/bin/env python3
"""One case per day, chosen for COVERAGE from the enumerated candidates. Deterministic.

`bin/enumerate_times.py` screens every hourly analysis; this picks which one becomes a
case. Three rules, in order:

  1. AT MOST ONE PER DAY. Two times from one day share the synoptic state, the soil
     moisture and the morning's history -- they are not independent units, and PROJECT_BRIEF.md's
     split-by-run rule would have to cover them jointly or the effective sample size is
     overstated. One per day makes the run id and the independent unit the same thing.

  2. VALID means the domain can hold it AND the state is quasi-stationary. Those are two
     separate screens and conflating them is the trap: widening the z_i band pushes
     selection towards morning and evening, because those are the hours when a day whose
     midday is too deep still has an acceptable z_i -- and they are also exactly when z_i
     changes fastest. A naive widening therefore trades a domain violation for a
     STATIONARITY violation and reports neither. dz_i/dt is screened on its own threshold.

  3. AMONG THE VALID, FILL THE THINNEST CELL. The candidate pool is ~24x larger than
     one-midday-per-day, so the binding constraint stops being validity and becomes
     coverage. Direction is the dominant skill axis and this site's rose is S/SW/W-heavy
     (S 16.0%, W 14.4%, NW 14.5%, SW 14.3% against N 10.6%, NE 10.2%, E 10.4%, SE 9.8%),
     so a northerly hour is worth more than a southerly one even though both are equally
     valid. Greedy thinnest-cell-first spends the surplus on that rather than on raw count.

NO RANDOMNESS ANYWHERE. Days are processed in date order and ties break on a fixed rule
(fewest in cell, then most stationary, then earliest hour), so the same candidate table
always yields the same corpus. A re-run is a no-op, not a re-roll.

usage: select_times.py [--candidates results/candidates.tsv] [--zi-min 100] [--zi-max 1200]
                       [--max-dzidt-rel 15] [--out results/selected_times.tsv]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

VONK, G, Z0 = 0.4, 9.81, 0.1435
RHO_CP = 1.15 * 1004.5          # W/m^2 -> K m/s
DIR_SECTORS = 12                # 30 deg, matching the seed library's 12 headings
DIR_NAMES = ["N", "NNE", "ENE", "E", "ESE", "SSE", "S", "SSW", "WSW", "W", "WNW", "NNW"]
# PROJECT_BRIEF.md's own classes: 27.2% very unstable (z/L < -0.5), 30.3% unstable,
# 13.3% near-neutral, 20.4% stable, 8.8% very stable.
STAB_EDGES = [-np.inf, -0.5, -0.02, 0.02, 0.5, np.inf]
STAB_NAMES = ["v.unstable", "unstable", "neutral", "stable", "v.stable"]
ZI_EDGES = [0, 300, 700, 1e9]
ZI_NAMES = ["shallow", "mid", "deep"]


def classify(zi, shtfl, wspd, wdir, zm=10.0):
    """(direction sector, stability class, z_i bin) for one candidate hour.

    u* is a surface-layer estimate from the 10 m wind and the domain z0 -- good to ~20%
    and used ONLY to order candidates into bins. Every quantity the corpus records comes
    off the LES itself; nothing here reaches a training record.
    """
    ust = VONK * max(wspd, 0.05) / np.log(zm / Z0)
    wth = shtfl / RHO_CP
    L = (-ust ** 3 * 290.0 / (VONK * G * wth)) if abs(wth) > 1e-5 else np.inf
    zoL = (zm / L) if np.isfinite(L) and L != 0 else 0.0
    d = int(((wdir + 180.0 / DIR_SECTORS) % 360.0) // (360.0 / DIR_SECTORS))
    s = int(np.searchsorted(STAB_EDGES, zoL, side="right") - 1)
    z = int(np.searchsorted(ZI_EDGES, zi, side="right") - 1)
    return d, min(max(s, 0), 4), min(max(z, 0), 2), float(zoL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="results/candidates.tsv")
    ap.add_argument("--zi-min", type=float, default=100.0)
    ap.add_argument("--zi-max", type=float, default=1200.0)
    ap.add_argument("--max-dzidt-rel", type=float, default=15.0,
                    help="percent per hour; screened INDEPENDENTLY of the z_i value")
    ap.add_argument("--out", default="results/selected_times.tsv")
    ap.add_argument("--report", default="results/time_selection.txt")
    a = ap.parse_args()

    if not os.path.exists(a.candidates):
        print(f"FATAL: {a.candidates} does not exist", file=sys.stderr)
        return 2
    rows = []
    for ln in open(a.candidates):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 8 or f[0] == "date":
            continue
        rows.append(dict(date=f[0], hour=int(f[1]), zi=float(f[2]),
                         dzidt=float(f[3]), dzidt_rel=float(f[4]),
                         shtfl=float(f[5]), wdir=float(f[6]), wspd=float(f[7])))
    if not rows:
        print(f"FATAL: no candidate rows in {a.candidates}", file=sys.stderr)
        return 2

    by_day = defaultdict(list)
    for r in rows:
        by_day[r["date"]].append(r)
    days = sorted(by_day)

    for r in rows:
        r["ok_zi"] = a.zi_min <= r["zi"] <= a.zi_max
        r["ok_dz"] = abs(r["dzidt_rel"]) <= a.max_dzidt_rel
        r["valid"] = r["ok_zi"] and r["ok_dz"]
        r["cell"] = classify(r["zi"], r["shtfl"], r["wspd"], r["wdir"])[:3]
        r["zoL"] = classify(r["zi"], r["shtfl"], r["wspd"], r["wdir"])[3]

    counts = Counter()
    picked, empty, empty_reason = [], [], Counter()
    for d in days:
        cand = [r for r in by_day[d] if r["valid"]]
        if not cand:
            empty.append(d)
            any_zi = any(r["ok_zi"] for r in by_day[d])
            empty_reason["no hour with an acceptable z_i" if not any_zi
                         else "z_i acceptable somewhere, but never stationary enough"] += 1
            continue
        # deterministic: thinnest cell, then most stationary, then earliest hour
        cand.sort(key=lambda r: (counts[r["cell"]], abs(r["dzidt_rel"]), r["hour"]))
        best = cand[0]
        counts[best["cell"]] += 1
        picked.append(best)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("date\thour\tzi_m\tdzidt_rel_per_h\tzm_over_L\twdir_deg\twspd_ms\t"
                "dir_sector\tstability\tzi_bin\n")
        for r in picked:
            d, s, z = r["cell"]
            f.write(f"{r['date']}\t{r['hour']:02d}\t{r['zi']:.1f}\t{r['dzidt_rel']:+.1f}\t"
                    f"{r['zoL']:+.4f}\t{r['wdir']:.1f}\t{r['wspd']:.2f}\t"
                    f"{DIR_NAMES[d]}\t{STAB_NAMES[s]}\t{ZI_NAMES[z]}\n")

    out = []
    p = out.append
    nv = sum(r["valid"] for r in rows)
    p(f"Time selection from {len(rows)} candidate hours over {len(days)} day(s)")
    p(f"  screens: {a.zi_min:.0f} m <= z_i <= {a.zi_max:.0f} m, "
      f"|dz_i/dt| <= {a.max_dzidt_rel:.0f} %/h")
    p(f"  candidate hours valid: {nv}/{len(rows)} = {100*nv/len(rows):.1f}%")
    p(f"    z_i out of range        : {sum(not r['ok_zi'] for r in rows):5d}")
    p(f"    z_i ok but drifting fast: {sum(r['ok_zi'] and not r['ok_dz'] for r in rows):5d}"
      f"   <-- the transition-bias screen; these would all be morning/evening")
    p("")
    p(f"  DAYS WITH A CASE : {len(picked)}/{len(days)} = {100*len(picked)/len(days):.1f}%")
    p(f"  DAYS WITH NONE   : {len(empty)}")
    for why, n in empty_reason.most_common():
        p(f"      {n:4d}  {why}")
    if empty:
        p(f"      e.g. {', '.join(empty[:6])}")
    p("")
    p("=== time of day chosen (UTC; local = UTC-6 CST / -5 CDT) ===")
    hh = Counter(r["hour"] for r in picked)
    mx = max(hh.values()) if hh else 1
    for h in range(24):
        n = hh.get(h, 0)
        p(f"  {h:02d}Z {n:4d} |{'#' * int(round(40 * n / mx))}")
    p("")
    p("=== coverage: direction x stability (cases) ===")
    p(f"  {'':>6}" + "".join(f"{s:>12}" for s in STAB_NAMES) + f"{'total':>8}")
    for di, dn in enumerate(DIR_NAMES):
        row = [sum(counts[(di, si, zi)] for zi in range(3)) for si in range(5)]
        p(f"  {dn:>6}" + "".join(f"{v:12d}" for v in row) + f"{sum(row):8d}")
    p(f"  {'total':>6}" + "".join(
        f"{sum(counts[(di, si, zi)] for di in range(12) for zi in range(3)):12d}"
        for si in range(5)) + f"{len(picked):8d}")
    p("")
    p("=== coverage: direction x z_i bin ===")
    p(f"  {'':>6}" + "".join(f"{z:>10}" for z in ZI_NAMES) + f"{'total':>8}")
    for di, dn in enumerate(DIR_NAMES):
        row = [sum(counts[(di, si, zi)] for si in range(5)) for zi in range(3)]
        p(f"  {dn:>6}" + "".join(f"{v:10d}" for v in row) + f"{sum(row):8d}")
    nz = sum(1 for c in counts.values() if c)
    p("")
    p(f"  {nz} of {12*5*3} cells occupied; thinnest occupied cell "
      f"{min([c for c in counts.values() if c], default=0)}, fullest {max(counts.values(), default=0)}")
    n_all = len(days)
    p("")
    p(f"=== scaled to five years ===")
    p(f"  {100*len(picked)/n_all:.1f}% of days carry a case -> "
      f"{round(365.25*5*len(picked)/n_all)} cases from {round(365.25*5)} days")

    txt = "\n".join(out)
    print(txt)
    if a.report:
        os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
        open(a.report, "w").write(txt + "\n")
        print(f"\n  wrote {a.out} and {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
