#!/usr/bin/env python3
"""Which seed does this case restart from? Stage 4 of the corpus pipeline.

The library is 6 stability/depth rungs x 3 base angles, and each base angle re-indexes into
4 wind directions by a 90-degree rotation of the grid (bin/prep_restart.py --rot; Gate B6
measured the rotation exact to 1.2e-14). So 18 spun-up files present 72 (state, direction)
options, and this picks one.

=== THE METRIC IS "WHAT WILL 30 MINUTES FAIL TO CLOSE", AND NOTHING ELSE ===

The seed exists only to be adjusted away, so the only thing worth minimising is the part of
the gap the 30-minute adjustment will still be carrying when sampling starts. Measured on
this project's own runs:

  regime      a CBL turns over in ~8 T* ~ 1.2 h, so 30 min does NOT convert a stable seed
              into a convective one.                          -> a HARD CONSTRAINT, not a cost
  z_i         entrainment closes +79 m/h = +40 m in 30 min, out of a gap that can be 800.
                                                              -> COST, scaled by ln 2
  direction   the mean flow backs -5.4 deg/h = 2.7 deg in 30 min, out of a gap up to 15.
                                                              -> COST, scaled by the 30 deg spacing
  z/L, u*     the surface flux is PRESCRIBED and the surface layer is ~0.1 z_i deep, so it
              re-equilibrates in ~2 min at a 10 m receptor.   -> REPORTED, never costed

An earlier version of this file standardized every axis by the library's own sample spread.
That is wrong in a way worth recording: the spread is a property of the library, not of the
physics, so the NARROWEST axis gets the largest weight -- and with an unspun library whose
z/L values were all placeholder estimates within 0.01 of each other, z/L ended up weighted
5x more heavily than z_i, exactly inverting the table above. The scales below are fixed and
physical: a factor of 2 in z_i, and one direction bin.

REGIME COMES FROM THE PRESCRIBED SURFACE FLUX, not from a z/L estimate. It is the boundary
condition both sides were actually built from, so no u* estimate enters the choice at all --
and a u* estimate is precisely what there is no honest way to get before the LES has run.

SEEDS ARE MATCHED ON WHAT THEY ACHIEVED, NOT ON WHAT THEY WERE ASKED FOR. jobs/run_seed.sh
writes the measured z_i, u*, U and direction into manifest["achieved"]. PROJECT_BRIEF.md already
requires this for direction ("Achieved direction is not forcing direction"); the same
argument applies to depth, and a seed that entrained past its target simply IS a different
rung than the one it was aimed at.

A MISMATCH DOES NOT CORRUPT A CASE. Corpus inputs are read off the LES window
(lpdm/les_stats.py:window_stats), never off the sounding, so an imperfectly-closed gap moves
where a case LANDS in input space without making it wrong. Seed spacing is a COVERAGE
question, not a correctness one -- which is why the 30-minute adjustment study can follow
the pipeline rather than gate it.

usage: pick_seed.py results/forcing/<case>.json [--index jobs/index.json]
                    [--library jobs] [--json FILE]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sounding_to_forcing import ekman_backing_deg

VONK = 0.4
G = 9.81

WTH_NEUTRAL = 0.01       # |w'th_v'| below this is a neutral run, K m/s
ZI_SCALE = np.log(2.0)   # a factor of 2 in depth costs 1
DIR_SCALE = 30.0         # one library direction bin costs 1


def regime_of(wth):
    """stable / neutral / convective from the PRESCRIBED virtual heat flux."""
    if wth > WTH_NEUTRAL:
        return "convective"
    if wth < -WTH_NEUTRAL:
        return "stable"
    return "neutral"


def load_library(index_path, library_dir):
    """Prefer each job's RETURNED manifest (it carries `achieved`); fall back to the index.

    The fallback is not silent. A seed with no achieved block is matched on its TARGET, and
    that is reported per row and again on the chosen seed -- because a library matched on
    targets is a library whose spacing has never been measured.
    """
    # NAME THE ACTUAL CAUSE. The analysis runs inside a container that mounts ONLY the
    # repo root, so a library under /tmp is simply not there -- and the symptom was
    # "no usable seeds", which points at the library's contents rather than at its path.
    if not os.path.isdir(library_dir) and not os.path.exists(index_path):
        raise SystemExit(
            f"neither {library_dir} nor {index_path} exists from in here. If this is "
            f"running in the container, remember it mounts only the repo root -- a seed "
            f"library outside it is invisible. Put it under the repo and pass a relative "
            f"path.")
    seeds, have, rejected = [], set(), []
    if os.path.isdir(library_dir):
        for m in sorted(glob.glob(os.path.join(library_dir, "*", "return",
                                               "manifest.json"))):
            s = json.load(open(m))
            # A SEED THAT FAILED ITS OWN STATIONARITY GATE IS NOT A SEED. It is a state
            # still drifting in one of the footprint's controlling parameters, and
            # restarting a case from it starts the case mid-transient -- which the 30
            # minute adjustment is not there to absorb and would not announce.
            if s.get("achieved", {}).get("pass") is False:
                rejected.append(s["job"])
                have.add(s["job"])          # and do NOT fall back to its index entry
                continue
            seeds.append(s)
            have.add(s["job"])
    if os.path.exists(index_path):
        for s in json.load(open(index_path))["jobs"]:
            if s["job"] not in have:
                seeds.append(s)
    if rejected:
        print(f"  EXCLUDED {len(rejected)} seed(s) that failed their own stationarity "
              f"gate: {', '.join(sorted(rejected))}")
    if not seeds:
        raise SystemExit(f"no usable seeds under {library_dir} or in {index_path}"
                         + (f" ({len(rejected)} failed their gate)" if rejected else ""))
    return seeds


def seed_state(s, zm):
    """(z_i, receptor heading, z_m/L, source) for one seed.

    Both headings compared here must be RECEPTOR-LEVEL. An achieved seed reports one
    directly. An unspun seed only knows its geostrophic angle, so the same Ekman estimate
    the case used is applied to it -- comparing a geostrophic bearing against a
    receptor-level one would be off by the whole turning, which is 10-24 deg at this site
    and therefore most of a direction bin.
    """
    wth = float(s["target"]["wth_virtual"])
    ach = s.get("achieved")
    if ach and np.isfinite(ach.get("zi", np.nan)):
        ust, th0 = float(ach["ustar"]), float(ach["theta0"])
        L = (-ust ** 3 * th0 / (VONK * G * wth)) if abs(wth) > 1e-6 else np.inf
        return (float(ach["zi"]), float(ach["wdir"]),
                (zm / L if np.isfinite(L) else 0.0), "achieved")
    zi = float(s["target"]["zi_m"])
    # zi/L only to choose the Ekman angle; it never enters the cost
    zoL_bulk = -1.0 if wth > WTH_NEUTRAL else 0.0
    hdg = (float(s["target"]["G_dir_from_deg"]) - ekman_backing_deg(zoL_bulk)) % 360.0
    return zi, hdg, float("nan"), "target"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("forcing")
    ap.add_argument("--index", default="jobs/index.json")
    ap.add_argument("--library", default="jobs")
    ap.add_argument("--zm", type=float, default=10.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    fc = json.load(open(a.forcing))
    lab, par = fc["labels"], fc["params"]
    zi_c = float(lab["zi_m"])
    wth_c = float(par["surflayer_wth"])
    reg_c = regime_of(wth_c)
    # the heading the LES will actually blow FROM, which is what an achieved seed reports
    dir_c = float(lab["predicted_10m_dir_deg"])
    L_c = lab.get("L_estimate")
    zoL_c = (a.zm / float(L_c)) if L_c not in (None, 0) else float("nan")

    seeds = load_library(a.index, a.library)
    rows = []
    for s, st in ((s, seed_state(s, a.zm)) for s in seeds):
        zi_s, dir_s, zoL_s, src = st
        same = regime_of(float(s["target"]["wth_virtual"])) == reg_c
        for rot in range(4):
            # prep_restart.py rotates the FLOW rot*90 deg counter-clockwise, which turns the
            # bearing the wind blows FROM counter-clockwise too: SUBTRACT 90 per turn.
            d = (dir_s - 90.0 * rot) % 360.0
            dd = abs(((d - dir_c + 180.0) % 360.0) - 180.0)
            c_zi = abs(np.log(max(zi_s, 1.0) / max(zi_c, 1.0))) / ZI_SCALE
            cost = float(np.hypot(c_zi, dd / DIR_SCALE))
            rows.append({"job": s["job"], "rung": s["rung"], "rot": rot,
                         "regime": s["regime"], "regime_match": bool(same),
                         "seed_dir_deg": round(d, 2), "d_dir_deg": round(dd, 2),
                         "seed_zi_m": round(zi_s, 1),
                         "cost_zi": round(float(c_zi), 4),
                         "cost_dir": round(dd / DIR_SCALE, 4),
                         "cost": round(cost, 4),
                         "seed_zm_over_L": (None if not np.isfinite(zoL_s)
                                            else round(float(zoL_s), 4)),
                         "labelled_by": src})
    # regime is a constraint, so it sorts before cost rather than being folded into it
    rows.sort(key=lambda r: (not r["regime_match"], r["cost"]))
    best = rows[0]

    print(os.path.basename(a.forcing))
    print(f"  case: {reg_c} (w'th_v' {wth_c:+.4f}), z_i {zi_c:.0f} m, "
          f"heading {dir_c:.1f} deg, z_m/L "
          f"{'n/a' if not np.isfinite(zoL_c) else f'{zoL_c:+.4f}'}")
    print(f"  {'seed':<24}{'rot':>4}{'dir':>6}{'d_dir':>7}{'z_i':>7}"
          f"{'c_zi':>7}{'c_dir':>7}{'cost':>7}  {'regime':>10} {'label':>8}")
    for r in rows[:6]:
        print(f"  {r['job']:<24}{r['rot']:>4}{r['seed_dir_deg']:>6.0f}"
              f"{r['d_dir_deg']:>7.1f}{r['seed_zi_m']:>7.0f}{r['cost_zi']:>7.3f}"
              f"{r['cost_dir']:>7.3f}{r['cost']:>7.3f}  {r['regime']:>10} "
              f"{r['labelled_by']:>8}")
    print(f"\n  CHOSEN: {best['job']} --rot {best['rot']}")
    print(f"    heading {best['seed_dir_deg']:.0f} vs {dir_c:.0f} deg "
          f"(gap {best['d_dir_deg']:.1f}, and 30 min of backing closes ~2.7)")
    print(f"    z_i {best['seed_zi_m']:.0f} vs {zi_c:.0f} m "
          f"(gap {best['seed_zi_m']-zi_c:+.0f}, and 30 min of entrainment closes ~+40)")
    if not best["regime_match"]:
        print(f"  WARNING: no {reg_c} seed in the library; fell back to "
              f"{best['regime']}. 30 min will NOT convert one regime into the other.")
    if best["labelled_by"] == "target":
        print("  NOTE: matched on the seed's TARGET, not its achieved state -- this "
              "library has not been spun up, so the spacing is unmeasured.")
    if best["d_dir_deg"] > 15.0:
        print(f"  WARNING: {best['d_dir_deg']:.1f} deg exceeds half the 30 deg library "
              f"spacing; a base angle is missing or a seed drifted off its own.")

    out = {"forcing": os.path.abspath(a.forcing),
           "case": {"zi_m": zi_c, "wth_virtual": wth_c, "regime": reg_c,
                    "heading_deg": dir_c,
                    "zm_over_L": None if not np.isfinite(zoL_c) else zoL_c},
           "scales": {"zi": "ln 2", "direction_deg": DIR_SCALE},
           "chosen": best, "ranked": rows[:12]}
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
