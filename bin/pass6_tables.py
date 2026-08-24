#!/usr/bin/env python3
"""The sixth pass's final tables: what the closure change did to the corpus.

Differences the regenerated production set against the frozen pre-fix record, and the
per-case legacy re-analysis against its own fields. The two are different questions and
are reported separately:

  * BEFORE vs AFTER mixes three changes -- the closure, the raised map, and a different
    turbulence realisation -- and is the corpus update.
  * new vs legacy ON THE SAME FIELDS isolates the closure alone, and is the attribution.

usage: pass6_tables.py
"""
import json
import os
import sys

import numpy as np

R = "results"
NBL = ["wN", "wE", "wS", "wW"]


def load(tag):
    p = os.path.join(R, f"{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def row(d):
    return dict(wdir=d["stats"]["wdir"], array=100 * d["cover_share"]["solar array"],
                integ=d["integral_les"], a80=d["les"]["area80_ha"],
                a80k=d["kljun"]["area80_ha"], peak=d["les"]["peak_x"],
                x80=d["les"]["x80"], ov=100 * d["overlap_kljun"])


def main():
    before = json.load(open(os.path.join(R, "pass6_before.json")))
    print("=" * 92)
    print("1. PRODUCTION, REGENERATED  (weighted floor, eps-consistent, --raise-topo, "
          "receptor 8.5 m + exact-agl)")
    print("=" * 92)
    hdr = (f"  {'case':<10}{'wdir':>7}{'array%':>9}{'d vs 5th':>10}{'integral':>10}"
           f"{'d':>8}{'A80 ha':>9}{'A80 Kljun':>11}{'ratio':>7}{'x80':>7}")
    agg = {}
    for reg in ("nbl", "cbl"):
        print(f"\n  --- {'neutral' if reg == 'nbl' else 'convective'} ---")
        print(hdr)
        cur, old = [], []
        for d in NBL:
            n = load(f"g16r_{reg}_{d}")
            o = before.get(f"g16_{reg}_{d}")
            if n is None:
                print(f"  g16r_{reg}_{d}: pending"); continue
            r = row(n); cur.append(r)
            da = r["array"] - 100 * o["array"] if o else np.nan
            di = r["integ"] - o["integral"] if o else np.nan
            if o: old.append(o)
            print(f"  {reg}_{d:<6}{r['wdir']:7.1f}{r['array']:9.2f}{da:+10.2f}"
                  f"{r['integ']:10.3f}{di:+8.3f}{r['a80']:9.3f}{r['a80k']:11.3f}"
                  f"{r['a80']/r['a80k']:7.2f}{r['x80']:7.0f}")
        if cur:
            agg[reg] = dict(
                array=np.mean([c["array"] for c in cur]),
                a80=np.mean([c["a80"] for c in cur]),
                integ=np.array([c["integ"] for c in cur]),
                array_old=np.mean([100 * o["array"] for o in old]) if old else np.nan,
                a80_old=np.mean([o["area80"] for o in old]) if old else np.nan,
                integ_old=np.array([o["integral"] for o in old]) if old else None)
            a = agg[reg]
            print(f"  {'mean':<10}{'':7}{a['array']:9.2f}"
                  f"{a['array'] - a['array_old']:+10.2f}"
                  f"{a['integ'].mean():10.3f}"
                  f"{a['integ'].mean() - a['integ_old'].mean():+8.3f}"
                  f"{a['a80']:9.3f}")

    if "nbl" in agg and "cbl" in agg:
        n, c = agg["nbl"], agg["cbl"]
        print(f"\n  NEUTRAL-vs-CONVECTIVE COMPACTION (mean 80% source area)")
        print(f"    fifth pass  {n['a80_old']:.3f} / {c['a80_old']:.3f} ha "
              f"= {n['a80_old']/c['a80_old']:.2f}x   [CONFOUNDED: the floor was active in "
              f"the convective half only]")
        print(f"    sixth pass  {n['a80']:.3f} / {c['a80']:.3f} ha "
              f"= {n['a80']/c['a80']:.2f}x   [same closure in both]")
        print(f"\n  CONVECTIVE INTEGRALS  fifth pass {c['integ_old'].min():.3f}-"
              f"{c['integ_old'].max():.3f}  ->  sixth pass {c['integ'].min():.3f}-"
              f"{c['integ'].max():.3f}")
        print(f"  NEUTRAL    INTEGRALS  fifth pass {n['integ_old'].min():.3f}-"
              f"{n['integ_old'].max():.3f}  ->  sixth pass {n['integ'].min():.3f}-"
              f"{n['integ'].max():.3f}")

    print("\n" + "=" * 92)
    print("2. THE CLOSURE ALONE  (same fields, same releases, same seed; only the floor "
          "differs)")
    print("=" * 92)
    print(f"  {'case':<12}{'array% new':>12}{'array% legacy':>15}{'d (points)':>12}"
          f"{'d rel':>8}{'integ new':>11}{'integ legacy':>14}{'x80 new':>9}{'x80 leg':>9}")
    ds = []
    for reg in ("nbl", "cbl"):
        for d in NBL:
            a, b = load(f"g16r_{reg}_{d}"), load(f"g16r_{reg}_{d}_legacy")
            if a is None or b is None:
                continue
            ra, rb = row(a), row(b)
            dd = rb["array"] - ra["array"]
            ds.append((reg, dd))
            print(f"  {reg}_{d:<8}{ra['array']:12.2f}{rb['array']:15.2f}{dd:+12.2f}"
                  f"{100*dd/max(ra['array'],1e-9):+7.1f}%{ra['integ']:11.3f}"
                  f"{rb['integ']:14.3f}{ra['x80']:9.0f}{rb['x80']:9.0f}")
    for reg, lab in (("nbl", "neutral"), ("cbl", "convective")):
        v = [d for r, d in ds if r == reg]
        if v:
            print(f"  {lab}: the retired closure was worth {np.mean(v):+.2f} points of "
                  f"array share on average (range {min(v):+.2f} to {max(v):+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
