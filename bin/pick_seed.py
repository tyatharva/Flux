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

# === HOW FAR FORWARD A FROZEN SEED IS PROJECTED, AND WHY THAT NUMBER ===============
# bin/run_corpus_case.sh runs ADJ_S = 1800 s of adjustment and then a WINDOW_S = 2400 s
# window, and lpdm/les_stats.py:window_stats averages over the dumps that SURVIVE the
# adjustment -- i.e. over [ADJ_S, ADJ_S + WINDOW_S] measured from the restart. Its
# midpoint is ADJ_S + WINDOW_S/2 = 3000 s, and the direction it reports is the label the
# pair actually carries. So that is the instant a seed's heading must be compared at.
#
# Do not "improve" this to the release-weighted midpoint (ADJ_S + TBACK + 900 = 3300 s).
# That is the right centre for the FOOTPRINT and the wrong one for the LABEL, and it is
# the label this file is minimising the gap in -- a mismatch does not corrupt a pair, it
# moves where the pair lands in input space.
ADJ_S, WINDOW_S = 1800.0, 2400.0
PROJECT_H = (ADJ_S + 0.5 * WINDOW_S) / 3600.0        # 0.8333 h


def regime_of(wth):
    """stable / neutral / convective from the PRESCRIBED virtual heat flux."""
    if wth > WTH_NEUTRAL:
        return "convective"
    if wth < -WTH_NEUTRAL:
        return "stable"
    return "neutral"


def load_library(index_path, library_dir, available_only=False):
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
    seeds, have, rejected, unbuilt, incomplete = [], set(), [], [], []
    if os.path.isdir(library_dir):
        for m in sorted(glob.glob(os.path.join(library_dir, "*", "return",
                                               "manifest.json"))):
            s = json.load(open(m))
            ret = os.path.dirname(m)
            # A SEED THAT FAILED ITS OWN STATIONARITY GATE IS NOT A SEED. It is a state
            # still drifting in one of the footprint's controlling parameters, and
            # restarting a case from it starts the case mid-transient -- which the 30
            # minute adjustment is not there to absorb and would not announce.
            #
            # AND THE VERDICT IS READ FROM THE GATE'S OWN JSON, NOT ONLY FROM THE
            # MANIFEST. jobs/run_seed.sh stamps `achieved` into the manifest as its LAST
            # step, so a job that died after the gate and before the stamp leaves a
            # manifest with no verdict at all -- and testing only `achieved.pass is False`
            # then reads a FAILED seed as an unjudged one and ranks it. Observed on
            # seed_sbl-weak_a030, whose stationarity.json says pass=false, whose manifest
            # says nothing, and which this function happily returned as the best available
            # seed in the library.
            gate = None
            gp = os.path.join(ret, "stationarity.json")
            if os.path.exists(gp):
                try:
                    gate = bool(json.load(open(gp)).get("pass"))
                except (ValueError, OSError):
                    gate = None
            ach = s.get("achieved") or {}
            if ach.get("pass") is False or gate is False:
                rejected.append(s["job"])
                have.add(s["job"])          # and do NOT fall back to its index entry
                continue
            # NO VERDICT AND NO ARTIFACT IS AN UNFINISHED JOB, not a seed and not an
            # unbuilt one. Falling back to its index entry would present a run that got
            # part way and stopped as though it had never started.
            if not ach and gate is None and not os.path.exists(
                    os.path.join(ret, "seed_restart.nc")):
                incomplete.append(s["job"])
                have.add(s["job"])
                continue
            if available_only and not os.path.exists(
                    os.path.join(ret, "seed_restart.nc")):
                unbuilt.append(s["job"])
                have.add(s["job"])
                continue
            seeds.append(s)
            have.add(s["job"])
    if os.path.exists(index_path):
        for s in json.load(open(index_path))["jobs"]:
            if s["job"] in have:
                continue
            # A SEED THAT HAS NOT BEEN SPUN UP CANNOT BE RESTARTED FROM, and while the
            # library is being built most of it has not. Two separate problems follow, and
            # --available-only answers both:
            #
            #   (i)  bin/run_corpus_case.sh refuses a pick whose return/seed_restart.nc
            #        does not exist, so a case matched to an unbuilt seed simply stops.
            #   (ii) an unspun seed's heading is an ESTIMATE -- its geostrophic angle minus
            #        a nominal Ekman backing -- while a spun one reports what it achieved.
            #        Ranking the two together compares a measurement against a guess, and
            #        the guess is only as good as the nominal angle. That is fine for
            #        planning the library and wrong for choosing a restart point.
            #
            # Default stays OFF so the full library can be costed before it exists.
            if available_only:
                unbuilt.append(s["job"])
                continue
            seeds.append(s)
    if rejected:
        print(f"  EXCLUDED {len(rejected)} seed(s) that failed their own stationarity "
              f"gate: {', '.join(sorted(rejected))}")
    if incomplete:
        print(f"  EXCLUDED {len(incomplete)} unfinished job(s) -- a return/ directory with "
              f"neither a gate verdict nor a restart: {', '.join(sorted(incomplete))}")
    if unbuilt:
        print(f"  --available-only: EXCLUDED {len(unbuilt)} seed(s) with no returned "
              f"artifact; ranking only the {len(seeds)} that have actually been spun up")
    if not seeds:
        raise SystemExit(f"no usable seeds under {library_dir} or in {index_path}"
                         + (f" ({len(rejected)} failed their gate)" if rejected else ""))
    return seeds


def measured_backing(seeds):
    """Ekman backing measured off the library itself, per rung and overall.

    THE NOMINAL IS AN ESTIMATE AND THE LIBRARY CAN NOW MEASURE IT. An unspun seed's
    heading is `G_dir_from - ekman_backing_deg(z_i/L)`, and every one of the 60
    (state, direction) options in the library is placed by that number -- direction is the
    dominant skill axis, so an error there is an error in the axis that matters most. The
    first spun seed put it at **18.3 deg against a nominal 23.5**, i.e. 5.2 deg, about a
    sixth of a library direction bin.

    So: as seeds are spun, use what THEY measured. Per rung first, because the backing
    depends on stability and depth and the rungs are exactly that axis; then the library
    mean; then the nominal. One spun seed is one sample and is reported as such -- this
    replaces a guess with a small sample, and says which it is.
    """
    per, by_reg = {}, {}
    for s in seeds:
        ach = s.get("achieved") or {}
        wd = ach.get("wdir")
        if wd is None or not np.isfinite(wd):
            continue
        b = ((float(s["target"]["G_dir_from_deg"]) - float(wd) + 180.0) % 360.0) - 180.0
        per.setdefault(s["rung"], []).append(b)
        by_reg.setdefault(s["regime"], []).append(b)
    # POOL WITHIN A REGIME, NEVER ACROSS ONE. Ekman turning is a function of stability --
    # ekman_backing_deg itself puts it at 23.5 deg neutral against 10.0 convective, a
    # 13.5 deg spread that is nearly half a library direction bin. A library MEAN pools
    # those, so with only neutral seeds spun it would place every convective seed using a
    # neutral measurement: strictly worse than the convective nominal it replaced, and
    # silently so. Observed on the first dry run of a convective case, which took the
    # neutral 18.3 deg.
    return ({k: (float(np.mean(v)), len(v)) for k, v in per.items()},
            {k: (float(np.mean(v)), len(v)) for k, v in by_reg.items()})


def seed_state(s, zm, meas=None):
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
        # MATCH IN THE CORPUS'S OWN CURRENCY. The stationarity gate switched its z_i to a
        # FIXED TKE threshold in 2026-08 because it scores a TREND and a peak-normalised
        # threshold moves with the peak. This function does something different: it
        # compares a VALUE against the case's requested depth, and the depth a case is
        # finally LABELLED with is lpdm/les_stats.py:window_stats's `h`, which is still the
        # 5%-of-peak fraction. The two definitions differ by 7-21% and the gap grows with
        # regime intensity, so mixing them would put a systematic offset into seed
        # selection that nothing downstream could see. Prefer the peak-fraction depth when
        # the seed reports it; fall back to the gated one for seeds spun before the split.
        zi_a = ach.get("zi_peakfrac", ach["zi"])
        # === PROJECT THE FREEZE-TIME DRIFT FORWARD; DO NOT BUDGET A CLOSURE ==========
        # MEASURED ON THE FIRST CORPUS PAIR, AND IT INVERTED THE DESIGN ASSUMPTION.
        # This file used to say "30 min of backing closes ~2.7 deg", i.e. that the
        # adjustment pulls the seed toward the case's own forcing. It does not. The seed
        # is frozen mid-oscillation and keeps turning the way it already was: the gap on
        # case_2023031014 WIDENED from 11.26 to 21.79 deg (341.72 -> 331.19 against a
        # 352.98 target) while the closure budget predicted it would shrink.
        #
        # 30 min is 2.8% of the 17.6 h inertial period. Nothing at that timescale can
        # re-point a mean flow; what the case inherits is the seed's angular momentum.
        # So the honest model is ballistic, not restoring:
        #
        #     heading at the window  =  heading at freeze  +  (d dir/dt) * PROJECT_H
        #
        # Checked on the one case that exists: -8.12 deg/h over 0.8333 h projects
        # 334.95 against a measured 331.19 -- residual 3.76 deg, against 10.53 for the
        # unprojected heading. n = 1, so this is a measured correction and not yet a
        # calibration; every seed from here reports its own rate.
        wd = float(ach["wdir"])
        rate = ach.get("dwdir_dt_deg_per_h")
        if rate is not None and np.isfinite(rate):
            wd = (wd + float(rate) * PROJECT_H) % 360.0
            src = "achieved+projected"
        else:
            # A SEED SPUN BEFORE THE DRIFT WAS RECORDED. Say so: its heading is a
            # freeze-time value being compared against a window-time one, and on the
            # measured rate that is ~7 deg, a quarter of a library direction bin.
            src = "achieved(frozen)"
        return (float(zi_a), wd, (zm / L if np.isfinite(L) else 0.0), src)
    zi = float(s["target"]["zi_m"])
    # zi/L only to choose the Ekman angle; it never enters the cost
    zoL_bulk = -1.0 if wth > WTH_NEUTRAL else 0.0
    back, src = ekman_backing_deg(zoL_bulk), "target"
    if meas:
        per, by_reg = meas
        if s["rung"] in per:
            back, src = per[s["rung"]][0], f"target/backing:rung(n={per[s['rung']][1]})"
        elif s["regime"] in by_reg:
            back, src = (by_reg[s["regime"]][0],
                         f"target/backing:{s['regime'][:4]}(n={by_reg[s['regime']][1]})")
    hdg = (float(s["target"]["G_dir_from_deg"]) - back) % 360.0
    return zi, hdg, float("nan"), src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("forcing")
    ap.add_argument("--index", default="jobs/index.json")
    ap.add_argument("--library", default="jobs")
    ap.add_argument("--zm", type=float, default=10.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--available-only", action="store_true",
                    help="rank only seeds that have a returned artifact on disk. Use "
                         "while the library is partially built: an unbuilt seed cannot be "
                         "restarted from, and its heading is an estimate rather than a "
                         "measurement, so ranking it against a spun seed compares a guess "
                         "with a number.")
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

    seeds = load_library(a.index, a.library, available_only=a.available_only)
    meas = measured_backing(seeds)
    per, by_reg = meas
    if by_reg:
        print(f"  Ekman backing MEASURED off the library: "
              + ", ".join(f"{k} {v:.1f} deg (n={n})" for k, (v, n) in sorted(per.items()))
              + "; by regime "
              + ", ".join(f"{k} {v:.1f} (n={n})" for k, (v, n) in sorted(by_reg.items()))
              + f"; nominal {ekman_backing_deg(0.0):.1f} neutral / "
              f"{ekman_backing_deg(-1.0):.1f} convective. An unspun seed takes its own "
              f"rung's measurement, then its own REGIME's, then the nominal -- never a "
              f"cross-regime mean, because the backing IS a function of stability.")
    rows = []
    for s, st in ((s, seed_state(s, a.zm, meas)) for s in seeds):
        zi_s, dir_s, zoL_s, src = st
        same = regime_of(float(s["target"]["wth_virtual"])) == reg_c
        for rot in range(4):
            # prep_restart.py rotates the FLOW rot*90 deg counter-clockwise, which turns the
            # bearing the wind blows FROM counter-clockwise too: SUBTRACT 90 per turn.
            d = (dir_s - 90.0 * rot) % 360.0
            dd = abs(((d - dir_c + 180.0) % 360.0) - 180.0)
            c_zi = abs(np.log(max(zi_s, 1.0) / max(zi_c, 1.0))) / ZI_SCALE
            cost = float(np.hypot(c_zi, dd / DIR_SCALE))
            ach_ = s.get("achieved") or {}
            _rate = ach_.get("dwdir_dt_deg_per_h")
            _froz = (None if _rate is None or not np.isfinite(_rate)
                     else round((float(ach_["wdir"]) - 90.0 * rot) % 360.0, 2))
            rows.append({"job": s["job"], "rung": s["rung"], "rot": rot,
                         "seed_dir_frozen_deg": _froz,
                         "dwdir_dt_deg_per_h": (None if _rate is None else float(_rate)),
                         "project_h": PROJECT_H,
                         "seed_G": float(s["target"].get("G", float("nan"))),
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
    print(f"    heading {best['seed_dir_deg']:.1f} vs {dir_c:.1f} deg "
          f"(gap {best['d_dir_deg']:.1f})")
    if best.get("seed_dir_frozen_deg") is not None:
        print(f"    ^ PROJECTED from a frozen {best['seed_dir_frozen_deg']:.1f} deg at "
              f"{best['dwdir_dt_deg_per_h']:+.2f} deg/h over {PROJECT_H:.3f} h "
              f"(= ADJ_S + WINDOW_S/2). The adjustment does NOT close a direction gap -- "
              f"measured, it widened one by 10.5 deg -- so the seed is carried forward "
              f"ballistically rather than assumed to relax toward the case's forcing.")
    elif best["labelled_by"] == "achieved(frozen)":
        print(f"    ^ NOT projected: this seed predates the drift measurement, so its "
              f"heading is a freeze-time value compared against a window-time one "
              f"(~7 deg on the one rate measured so far).")
    print(f"    z_i {best['seed_zi_m']:.0f} vs {zi_c:.0f} m "
          f"(gap {best['seed_zi_m']-zi_c:+.0f} m)")
    print(f"    ^ REPORTED, not projected, and the budget it replaces was too small by 3x."
          f" The design assumed +79 m/h of entrainment, i.e. ~+40 m of closure. Measured on"
          f" case_2023031014 the depth closed a 146 m gap AND overshot by 49 -- +233 m/h."
          f" Unlike direction, this rate is NOT a property of the seed and cannot be"
          f" carried forward: entrainment is set by the CASE's surface flux working"
          f" against the CASE's inversion, and that case's lid (2.61 K/km) is far weaker"
          f" than the seed's (+8 K/100 m). The library buys LESS convergence in direction"
          f" and MORE in depth than it was designed to.")
    # === THE GEOSTROPHIC SPEED IS REPORTED, NOT COSTED -- AND HERE IS WHY, WITH THE
    # === NUMBER THAT SAYS WHEN TO STOP BELIEVING IT.
    #
    # Everything Kljun sees is a RATIO. U(z_m) and u* both scale with G, so Pi_4 = U/u*
    # is nearly invariant under a speed mismatch -- measured on g16_spin, u* moved 18%
    # over five windows while U/u* moved 0.6%. That is exactly why this file costs
    # direction and depth and not speed.
    #
    # What 30 minutes does NOT do is close the gap. The mean flow accelerates at
    # f*(G_case - G_seed), so it closes f*dt = 9.94e-5 * 1800 = 17.9% of it, whatever the
    # gap is. A case therefore samples a flow somewhere between the two forcings, and the
    # window is labelled by what it ACHIEVED (lpdm/les_stats.py) rather than by either.
    # That is sound for a modest gap and an extrapolation for a large one, so the ratio is
    # printed and a factor of 2 is called out rather than left for someone to notice.
    G_s = float(best.get("seed_G", float("nan")))
    G_c = float(lab.get("G_speed", float("nan")))
    if np.isfinite(G_s) and np.isfinite(G_c) and G_s > 0:
        r = G_c / G_s
        print(f"    G {G_s:.1f} -> {G_c:.1f} m/s ({r:.2f}x): REPORTED, never costed -- "
              f"U and u* scale together so Kljun's U/u* is nearly invariant, and 30 min "
              f"closes only f*dt = 17.9% of a geostrophic gap whatever its size")
        if r > 2.0 or r < 0.5:
            print(f"  WARNING: the case is forced {r:.2f}x the seed's geostrophic wind. "
                  f"The ratio argument above is an extrapolation at this size: the window "
                  f"will be sampled mid-acceleration and its mean flow is a transient, "
                  f"not a state. The pair stays self-consistent (inputs are read off the "
                  f"LES) but do not quote this case as quasi-stationary in U.")
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
           "ekman_backing_measured": {k: {"deg": v, "n": n} for k, (v, n) in per.items()},
           "ekman_backing_by_regime": {k: {"deg": v, "n": n}
                                       for k, (v, n) in by_reg.items()},
           "ekman_backing_nominal": float(ekman_backing_deg(0.0)),
           "G_seed": G_s, "G_case": G_c,
           "G_ratio": (G_c / G_s if (np.isfinite(G_s) and G_s > 0) else None),
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
