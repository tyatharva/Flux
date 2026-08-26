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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_fraction import ustar_from_wind, zeta_from_ustar   # noqa: E402

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
    and used ONLY to order candidates into bins and to apply the stability screen. Every
    quantity the corpus RECORDS comes off the LES itself; nothing here reaches a training
    record.

    THE NEUTRAL LOG LAW IS NOT GOOD ENOUGH HERE, and it used to be what this did. At a
    given wind, inverting neutrally OVERSTATES u*, which UNDERSTATES z/L -- and z/L is now
    a screen, not just a bin label, so that bias would quietly wave through exactly the
    strongly stable hours the screen exists to reject. bin/stable_fraction.py's
    ustar_from_wind solves the stability-corrected log law instead, and returns NaN where
    it has no solution at all: the classical surface-layer cutoff, which is the same set of
    hours by another name. NaN is mapped to the very-stable class, never to 0.
    """
    wth = shtfl / RHO_CP
    ust = float(ustar_from_wind(np.asarray(max(wspd, 0.05)), np.asarray(wth), zm))
    if not np.isfinite(ust) or ust <= 0.0:
        zoL = np.inf if wth < 0 else 0.0
    else:
        zoL = float(zeta_from_ustar(np.asarray(ust), np.asarray(wth), zm))
    if not np.isfinite(zoL):
        zoL = np.inf if wth < 0 else 0.0
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
    ap.add_argument("--max-zol", type=float, default=0.0,
                    help="reject stable hours above this z/L at 10 m. Default 0.0: the "
                         "grid cannot carry ANY of them. Two seeds were run, at z/L 0.12 "
                         "and 0.044, and both collapsed on the same timeline -- the "
                         "Ozmidov scale is 6.9 Delta at the receptor even at 0.044, "
                         "against 318 neutrally, so weakening the stratification does not "
                         "reach the problem (STABLE_REGIME_RESULT.md). Unstable hours are "
                         "never screened by this. Raise it only to re-measure the bound.")
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
        cl = classify(r["zi"], r["shtfl"], r["wspd"], r["wdir"])
        r["cell"], r["zoL"] = cl[:3], cl[3]
        r["ok_zi"] = a.zi_min <= r["zi"] <= a.zi_max
        r["ok_dz"] = abs(r["dzidt_rel"]) <= a.max_dzidt_rel
        # THE THIRD SCREEN, and it is one-sided on purpose. A convective hour is never
        # rejected for being convective; only the stable side has a resolution ceiling.
        r["ok_zl"] = r["zoL"] <= a.max_zol
        r["valid"] = r["ok_zi"] and r["ok_dz"] and r["ok_zl"]

    counts = Counter()
    picked, empty, empty_reason = [], [], Counter()
    for d in days:
        cand = [r for r in by_day[d] if r["valid"]]
        if not cand:
            empty.append(d)
            any_zi = any(r["ok_zi"] for r in by_day[d])
            any_zl = any(r["ok_zi"] and r["ok_zl"] for r in by_day[d])
            empty_reason["no hour with an acceptable z_i" if not any_zi
                         else ("z_i ok somewhere, but every such hour is too stable to run"
                               if not any_zl
                               else "z_i and z/L ok somewhere, but never stationary enough")] += 1
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
      f"|dz_i/dt| <= {a.max_dzidt_rel:.0f} %/h, z/L <= {a.max_zol:.2f} (stable side only)")
    p(f"  candidate hours valid: {nv}/{len(rows)} = {100*nv/len(rows):.1f}%")
    p(f"    z_i out of range        : {sum(not r['ok_zi'] for r in rows):5d}")
    p(f"    z_i ok, too stable to run: {sum(r['ok_zi'] and not r['ok_zl'] for r in rows):4d}"
      f"   <-- the resolution ceiling; z/L > {a.max_zol:.2f} laminarises at dx = 16 m")
    p(f"    z_i+z/L ok but drifting  : "
      f"{sum(r['ok_zi'] and r['ok_zl'] and not r['ok_dz'] for r in rows):5d}"
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
    p("=== TIME-OF-DAY SKEW: is the daytime convective population being thinned? ===")
    p("  Stationarity is the binding screen now, and it rejects the morning and evening")
    p("  transitions hardest. The convective midday is where the array's flux signal")
    p("  lives, so a screen that quietly ate it would be the worst possible failure --")
    p("  and it would look like a healthy 24-hour histogram while doing it.")
    p("")
    p("  LOCAL STANDARD TIME (UTC-6). Acceptance of every CANDIDATE hour, by local hour:")
    p(f"  {'LST':>5}{'cand':>7}{'z_i ok':>9}{'z/L ok':>9}{'dz/dt ok':>10}"
      f"{'ALL':>8}{'accept':>9}   {'convective share of candidates':<32}")
    for lh in range(24):
        sub = [r for r in rows if (r["hour"] - 6) % 24 == lh]
        if not sub:
            continue
        n = len(sub)
        nzi = sum(r["ok_zi"] for r in sub)
        nzl = sum(r["ok_zl"] for r in sub)
        ndz = sum(r["ok_dz"] for r in sub)
        nok = sum(r["valid"] for r in sub)
        nconv = sum(1 for r in sub if r["zoL"] < -0.02)
        p(f"  {lh:02d}h {n:6d}{100*nzi/n:8.0f}%{100*nzl/n:8.0f}%{100*ndz/n:9.0f}%"
          f"{nok:8d}{100*nok/n:8.0f}%   {'#' * int(round(32 * nconv / n))}")
    p("")
    # THE ACTUAL TEST. Not "does the histogram span 24 h" -- it does, and that is not
    # evidence. The question is whether the CONVECTIVE hours survive the screens at the
    # same rate as everything else, and whether the SELECTED set is convective in at
    # least the proportion the underlying record is.
    conv = [r for r in rows if r["zoL"] < -0.02]
    nonc = [r for r in rows if r["zoL"] >= -0.02]
    mid = [r for r in conv if 10 <= (r["hour"] - 6) % 24 <= 16]
    def rate(x):
        return 100.0 * sum(r["valid"] for r in x) / max(len(x), 1)
    p(f"  screen acceptance, convective (z/L < -0.02)      : {rate(conv):5.1f}%  "
      f"({sum(r['valid'] for r in conv)}/{len(conv)})")
    p(f"  screen acceptance, everything else               : {rate(nonc):5.1f}%  "
      f"({sum(r['valid'] for r in nonc)}/{len(nonc)})")
    p(f"  screen acceptance, CONVECTIVE MIDDAY (10-16 LST) : {rate(mid):5.1f}%  "
      f"({sum(r['valid'] for r in mid)}/{len(mid)})")
    p("")
    pop_conv = 100.0 * len(conv) / max(len(rows), 1)
    sel_conv = 100.0 * sum(1 for r in picked if r["zoL"] < -0.02) / max(len(picked), 1)
    val_conv = (100.0 * sum(1 for r in rows if r["valid"] and r["zoL"] < -0.02)
                / max(nv, 1))
    p(f"  convective share of ALL candidate hours   : {pop_conv:5.1f}%")
    p(f"  convective share of VALID candidate hours : {val_conv:5.1f}%")
    p(f"  convective share of SELECTED cases        : {sel_conv:5.1f}%")
    verdict = ("NOT THINNED -- selection is convective-richer than the record"
               if sel_conv >= pop_conv else
               f"THINNED by {pop_conv - sel_conv:.1f} points -- investigate")
    p(f"  VERDICT: {verdict}")
    p("")
    p("  local-hour histogram of SELECTED cases (LST):")
    lh_sel = Counter((r["hour"] - 6) % 24 for r in picked)
    mxl = max(lh_sel.values()) if lh_sel else 1
    for lh in range(24):
        n = lh_sel.get(lh, 0)
        night = " " if 6 <= lh <= 18 else "*"
        p(f"  {lh:02d}h{night}{n:4d} |{'#' * int(round(36 * n / mxl))}")
    p("       (* = outside 06-18 LST)")
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
