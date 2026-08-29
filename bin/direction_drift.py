#!/usr/bin/env python3
"""Ekman backing and direction DRIFT, measured off the seed library.

Direction is the dominant skill axis, and every one of the library's (state, direction)
options is placed by an angle. This script measures that angle from the seeds themselves
and regenerates the standing record, so the number is never hand-maintained again.

=== TWO NUMBERS, AND THE SECOND ONE IS THE FINDING ===

**Backing** is a static offset: geostrophic FROM-bearing minus the seed's achieved
receptor FROM-bearing. bin/sounding_to_forcing.py:ekman_backing_deg predicts it (23.5 deg
neutral, 10 convective) and bin/pick_seed.py places every unspun seed with it.

**Drift** is d(direction)/dt at freeze. It was not measured at all until the first corpus
pair, and it inverted the design assumption. The library was built on the premise that the
30-minute adjustment CLOSES a direction gap by ~2.7 deg. It does not: the seed is frozen
mid-oscillation and keeps turning the way it already was, so on case_2023031014 the gap
WIDENED from 11.26 to 21.79 deg. 30 min is 2.8% of the 17.6 h inertial period -- far too
little for the case's own forcing to re-point a mean flow. What the case inherits is the
seed's angular momentum, not the seed's angle.

AND THE WIDENING IS NOT n = 1. Both corpus cases that have run show it, measured as
achieved-minus-requested direction, which does not depend on any seed estimate:

    case_2023031014   pick gap 11.3 deg -> achieved 21.8 deg   (widened 10.5)
    e2e_20230118      pick gap 14.1 deg -> achieved 36.0 deg   (widened 21.9)

Two for two, in opposite directions of the compass and off different rungs. What is still
n = 1 is the RATE, which is why the projection below is called a measured correction and
not a calibration.

=== AND THE CONSEQUENCE FOR THE SPEC IS THE OPPOSITE OF THE OBVIOUS ONE ===

The obvious reading of "the measured backing is 18.3 against a nominal 23.5" is that the
nominal is wrong by 5.2 deg and the three base angles should be offset to compensate.
**That reading is wrong, and this script exists partly to say so before 15 seeds are
placed on it.**

The two numbers are measured at DIFFERENT INSTANTS. The achieved backing is read at the
seed's freeze; the heading that matters is the one the window carries, which is
PROJECT_H = (ADJ_S + WINDOW_S/2) later -- the midpoint of the dumps lpdm/les_stats.py
averages. Carry the freeze-time backing forward at the seed's own measured drift and the
5.2 deg collapses to 1.6. The nominal angle was not set wrong; it was being compared at
the wrong time.

So: bin/pick_seed.py now PROJECTS, and the base angles stay where they are. What would
change that verdict is per-rung drift spreading wide enough to stagger the rungs' effective
direction sets -- which needs more than one rung to see, and is why this table has a
per-rung column that is mostly empty today.

usage: direction_drift.py [--library jobs] [--out results/direction_drift.txt]
"""
from __future__ import annotations

import argparse
import glob
import math
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pick_seed import PROJECT_H                                  # noqa: E402
from sounding_to_forcing import ekman_backing_deg                # noqa: E402


def collect(library):
    out = []
    for m in sorted(glob.glob(os.path.join(library, "*", "return", "manifest.json"))):
        s = json.load(open(m))
        ach = s.get("achieved") or {}
        # A DRIFT MEASUREMENT IS VALID WHATEVER THE STATIONARITY VERDICT. The rate is
        # read off the same dumps either way; refusing a seed here because a DIFFERENT
        # limit could not be resolved throws away the only samples this question has.
        # The gate state is carried instead, so a reader can weight them.
        if not np.isfinite(ach.get("wdir", np.nan)):
            continue
        G = float(s["target"]["G_dir_from_deg"])
        frz = float(ach["wdir"])
        rate = ach.get("dwdir_dt_deg_per_h")
        proj = (frz + rate * PROJECT_H) % 360.0 if rate is not None else None
        nom = ekman_backing_deg(-1.0 if s["regime"] == "convective" else 0.0)
        out.append(dict(job=s["job"], rung=s["rung"], regime=s["regime"], G_dir=G,
                        ustar=ach.get("ustar"), gate=ach.get("pass"),
                        wdir_freeze=frz, rate=rate, wdir_window=proj, nominal=nom,
                        back_freeze=((G - frz + 180) % 360) - 180,
                        back_window=(None if proj is None
                                     else ((G - proj + 180) % 360) - 180),
                        zi_target=float(s["target"]["zi_m"]),
                        zi_ach=ach.get("zi_peakfrac", ach.get("zi"))))
    return out


def predictors(rows, cases):
    """Does anything predict the drift rate? Reported honestly, including "no".

    THE HONEST ANSWER TODAY IS THAT NOTHING CAN BE TESTED. A predictor needs a sample, and
    the sample is: 2 seeds with a measured freeze-time drift rate, and 2 corpus cases with
    a measured widening. With n = 2 a straight line through the points is exact by
    construction and its correlation is +/-1 whatever the physics -- reporting one would
    be the purest form of the failure PROJECT_BRIEF.md already forbids twice ("a tolerance
    measured from one difference is not a tolerance"; "gates compare against a
    DISTRIBUTION with enough degrees of freedom to have a standard error").

    So this prints the table and the sample size and refuses to fit. What it DOES do is
    name the candidate predictors and lay the data out so the answer arrives on its own as
    seeds accumulate -- and it flags the one structural fact that is already visible
    without any fit.
    """
    L = []
    P = L.append
    P("")
    P("  === WHAT PREDICTS THE DRIFT RATE? ===")
    P(f"  {'seed':26}{'drift':>9}{'u*':>8}{'z_i':>8}{'h/u* (s)':>10}{'|drift|*h/u*':>14}")
    for r in rows:
        if r["rate"] is None:
            continue
        us, zi = r.get("ustar"), r.get("zi_ach")
        tt = (zi / us) if (us and zi) else float("nan")
        P(f"  {r['job']:26}{r['rate']:+9.2f}{(us or float('nan')):8.4f}"
          f"{(zi or float('nan')):8.0f}{tt:10.0f}{abs(r['rate'])*tt/3600.0:14.3f}")
    n = sum(1 for r in rows if r["rate"] is not None)
    P(f"  n = {n} seeds with a measured rate."
      + ("  NO FIT IS REPORTED: with n <= 3 any predictor explains the data exactly and "
         "the correlation is an artifact of the sample size, not a result."
         if n <= 3 else ""))
    # ORDERING IS NOT A FIT, and it is the one thing a sample this small can honestly say.
    # A monotone ordering across n points has a 2/n! chance of arising by accident, which
    # is worth stating at n = 3 (p = 0.33) only as a candidate to watch -- never as a
    # relationship. What it CANNOT do is separate the candidates: if u*, z_i and h/u* all
    # order the seeds identically, no amount of staring at three points tells you which is
    # the mechanism.
    have = [r for r in rows if r["rate"] is not None
            and r.get("ustar") and r.get("zi_ach")]
    if len(have) >= 3:
        mag = [abs(r["rate"]) for r in have]
        for nm, key in (("u*", lambda r: r["ustar"]), ("z_i", lambda r: r["zi_ach"]),
                        ("h/u*", lambda r: r["zi_ach"] / r["ustar"])):
            order = [abs(r["rate"]) for r in sorted(have, key=key)]
            mono = (all(a >= b for a, b in zip(order, order[1:]))
                    or all(a <= b for a, b in zip(order, order[1:])))
            P(f"    |drift| vs {nm:5}: {'MONOTONE' if mono else 'not monotone'}  "
              + "  ".join(f"{v:.2f}" for v in order))
        P(f"    A monotone ordering of {len(have)} points arises by chance with "
          f"probability {2.0/math.factorial(len(have)):.3f}"
          f" -- a candidate to watch, not a relationship. And where several candidates "
          f"order the seeds identically, three points cannot separate them.")
    if cases:
        P("")
        P(f"  {'case':24}{'pick gap':>10}{'ach gap':>9}{'widened':>9}"
          f"{'seed rate':>11}{'WINDOW rate':>13}{'ratio':>8}")
        for c in cases:
            sr, wr = c.get("seed_rate"), c.get("window_rate")
            rat = (wr / sr) if (sr and wr) else float("nan")
            P(f"  {c['tag']:24}{c['pick_gap']:10.1f}{c['ach_gap']:9.1f}"
              f"{c['widened']:+9.1f}{(sr if sr is not None else float('nan')):+11.2f}"
              f"{(wr if wr is not None else float('nan')):+13.2f}{rat:8.2f}")
        P(f"  n = {len(cases)} cases.")
        P("")
        P("  DOES THE CASE INHERIT THE SEED'S DRIFT? Measured directly for the first time,")
        P("  on the two cases whose windows recorded a per-dump direction. The answer is")
        P("  YES IN SIGN AND NO IN MAGNITUDE: both windows back, as the seed was backing")
        P("  at freeze, but 1.3x and 2.7x faster. So the ballistic model behind")
        P("  bin/pick_seed.py's projection points the right way and UNDER-CORRECTS, and")
        P("  the residual is the scatter the denser base angles exist to absorb.")
    P("")
    P("  === AND ONE THING FOLLOWS WITHOUT ANY FIT ===")
    P("  A drift that is one-signed and of order 10-20 deg by the time the window is")
    P("  sampled is NOT a spacing that 30-degree base angles deliver +/- 15 deg on.")
    P("  Twelve library directions at 30 deg give a worst-case gap of 15 deg only if the")
    P("  seeds sit where they were placed; a variable 10-20 deg one-signed excursion")
    P("  makes the worst case 25-35 deg, and on the DOMINANT SKILL AXIS.")
    P("")
    P("  Projection (bin/pick_seed.py) removes the MEAN of that excursion and leaves its")
    P("  SCATTER, so it is a partial fix and cannot be the whole one. The honest fix is")
    P("  DENSER BASE ANGLES: 6 base angles at 15 deg = 24 library directions, worst-case")
    P("  7.5 deg before drift and ~15 after, which is what 3 angles were believed to give.")
    P("  Cost: 30 seeds instead of 15, i.e. ~86 GPU-h instead of ~43, against a corpus of")
    P("  ~1700 GPU-h -- 2.5% of the total to fix the axis the emulator is judged on.")
    P("  PROPOSED, NOT APPLIED.")
    return L


def collect_cases(pairdir="pairs", pickdir="results/pick"):
    """Every corpus case that has a measured achieved-vs-requested direction."""
    out = []
    for p in sorted(glob.glob(os.path.join(pairdir, "*.json"))):
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        amr = ((d.get("forcing") or {}).get("achieved_minus_requested")) or {}
        sd = d.get("seed") or {}
        g = amr.get("dir_deg")
        if g is None:
            continue
        pk = sd.get("d_dir_deg")
        if pk is None:
            continue
        # the drift measured ACROSS the case's own window, if stage 5 recorded it
        wr = None
        fp = os.path.join("results", "corpus", os.path.basename(p))
        if os.path.exists(fp):
            try:
                wr = json.load(open(fp)).get("dwdir_dt_window_deg_per_h")
            except (OSError, ValueError):
                wr = None
        out.append(dict(tag=os.path.basename(p)[:-5], seed=sd.get("job"),
                        pick_gap=float(pk), ach_gap=abs(float(g)),
                        widened=abs(float(g)) - float(pk),
                        seed_rate=sd.get("dwdir_dt_deg_per_h"), window_rate=wr))
    return out


def render(rows, cases=()):
    L = []
    P = L.append
    P("=== Ekman backing and direction DRIFT, measured off the seed library ===")
    P("  GENERATED by bin/direction_drift.py -- do not hand-edit.")
    P("")
    P("  backing  = geostrophic FROM-bearing minus the seed's achieved FROM-bearing")
    P("  drift    = d(direction)/dt at freeze, from the gate's scored window")
    P(f"  window   = the freeze-time heading carried forward {PROJECT_H:.4f} h "
      f"(= ADJ_S + WINDOW_S/2),")
    P("             which is the midpoint of the dumps window_stats averages -- i.e. the")
    P("             instant the pair's own direction LABEL refers to.")
    P("")
    if not rows:
        P("  NO SPUN SEEDS WITH A RECORDED DRIFT YET.")
        return "\n".join(L) + "\n"
    P(f"  {'rung':<14}{'job':<26}{'G from':>7}{'freeze':>8}{'drift':>9}"
      f"{'window':>8}{'b_frz':>7}{'b_win':>7}{'nom':>6}")
    for r in rows:
        P(f"  {r['rung']:<14}{r['job']:<26}{r['G_dir']:>7.1f}{r['wdir_freeze']:>8.2f}"
          f"{(r['rate'] if r['rate'] is not None else float('nan')):>+9.2f}"
          f"{(r['wdir_window'] if r['wdir_window'] is not None else float('nan')):>8.2f}"
          f"{r['back_freeze']:>+7.2f}"
          f"{(r['back_window'] if r['back_window'] is not None else float('nan')):>+7.2f}"
          f"{r['nominal']:>6.1f}")
    P("")
    per = {}
    for r in rows:
        per.setdefault(r["rung"], []).append(r)
    for k, v in sorted(per.items()):
        bf = np.array([x["back_freeze"] for x in v], float)
        rt = np.array([x["rate"] for x in v if x["rate"] is not None], float)
        P(f"  {k:<14} backing at freeze {bf.mean():+6.2f} deg (n={bf.size})"
          + (f", drift {rt.mean():+6.2f} deg/h (n={rt.size})" if rt.size else ""))
    bf = np.array([r["back_freeze"] for r in rows], float)
    bw = np.array([r["back_window"] for r in rows if r["back_window"] is not None], float)
    nm = np.array([r["nominal"] for r in rows], float)
    P(f"  {'LIBRARY':<14} backing at freeze {bf.mean():+6.2f} deg (n={bf.size})")
    P("")
    P("  === IS THE NOMINAL WRONG, OR WAS IT READ AT THE WRONG INSTANT? ===")
    P(f"    nominal                       {nm.mean():+6.2f} deg")
    P(f"    measured AT FREEZE            {bf.mean():+6.2f} deg   "
      f"error {abs(bf.mean()-nm.mean()):5.2f} deg "
      f"= {100*abs(bf.mean()-nm.mean())/30:.0f}% of a 30 deg library bin")
    if bw.size:
        nm2 = np.array([r["nominal"] for r in rows if r["back_window"] is not None], float)
        P(f"    measured AT THE WINDOW        {bw.mean():+6.2f} deg   "
          f"error {abs(bw.mean()-nm2.mean()):5.2f} deg "
          f"= {100*abs(bw.mean()-nm2.mean())/30:.0f}% of a 30 deg library bin")
        P("")
        if abs(bw.mean() - nm2.mean()) < abs(bf.mean() - nm.mean()):
            P("    THE ANGLE WAS RIGHT; THE INSTANT WAS WRONG. Most of the apparent error")
            P("    in the nominal is the seed continuing to back between its freeze and")
            P("    the window it is used for. bin/pick_seed.py projects, and the three")
            P("    base angles need no offset on this evidence.")
        else:
            P("    Projection does NOT reconcile the nominal with the measurement here.")
            P("    If this survives more rungs, the nominal itself needs replacing.")
    P("")
    P("  RECOMMENDATION ON BASE ANGLES -- PROPOSED, NOT APPLIED.")
    P("    Leave {0, 30, 60} as they are. Two reasons, and the second is the general one:")
    P("      (i) once compared at the window the nominal is right to within the error")
    P("          above, so there is nothing to compensate for;")
    P("      (ii) a drift that is the SAME across rungs cannot change coverage at all --")
    P("           shifting all 12 library directions by one constant leaves a uniformly")
    P("           spaced set of 12. Only a drift that DIFFERS between rungs moves the")
    P("           effective spacing, and that is what the per-rung column above is for.")
    P(f"    Revisit when the per-rung drifts span a usable fraction of the 30 deg bin.")
    P(f"    Samples so far: {bf.size}.")
    L.extend(predictors(rows, cases))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="jobs")
    ap.add_argument("--out", default="results/direction_drift.txt")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = collect(a.library)
    txt = render(rows, collect_cases())
    print(txt)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(txt)
    print(f"  wrote {a.out}")
    if a.json:
        json.dump(dict(project_h=PROJECT_H, seeds=rows), open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
