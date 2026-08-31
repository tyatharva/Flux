#!/usr/bin/env python3
"""Which seed does this case restart from? Stage 4 of the corpus pipeline.

The library is 5 stability/depth rungs x 6 base angles, and each base angle re-indexes into
4 wind directions by a 90-degree rotation of the grid (bin/prep_restart.py --rot; Gate B6
measured the rotation exact to 1.2e-14). So 30 spun-up files present 120 (state, direction)
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
DIR_SCALE = 15.0         # one library direction bin costs 1; 6 base angles x 4
                         # rotations = 24 headings at 15 deg (approved 2026-08-27,
                         # was 30 -- see bin/make_seed_jobs.py:BASE_ANGLES)

# === HOW FAR FORWARD A FROZEN SEED IS PROJECTED, AND WHY THAT NUMBER ===============
# bin/run_corpus_case.sh runs ADJ_S seconds of adjustment and then a WINDOW_S window, and
# lpdm/les_stats.py:window_stats averages over the dumps that SURVIVE the adjustment --
# i.e. over [ADJ_S, ADJ_S + WINDOW_S] measured from the restart. Its midpoint is
# ADJ_S + WINDOW_S/2, and the direction it reports is the label the pair actually carries.
# So that is the instant a seed's heading must be compared at.
#
# Do not "improve" this to the release-weighted midpoint (ADJ_S + TBACK + REL_S/2). That is
# the right centre for the FOOTPRINT and the wrong one for the LABEL, and it is the label
# this file is minimising the gap in -- a mismatch does not corrupt a pair, it moves where
# the pair lands in input space.
#
# READ FROM THE ENVIRONMENT, because these ARE the driver's numbers and having them written
# down twice is how they drift apart. They did: the literals here stayed at the 16 m
# geometry's 1800/2400 after production moved to 1800/2700, leaving the projection 150 s
# short. At the measured -5.8 deg/h that is 0.24 deg against a 15 deg library spacing --
# harmless, and silent, which is the half worth fixing. bin/run_corpus.sh exports both.
ADJ_S = float(os.environ.get("ADJ_S") or 1800.0)
WINDOW_S = float(os.environ.get("WINDOW_S") or 2700.0)
PROJECT_H = (ADJ_S + 0.5 * WINDOW_S) / 3600.0        # 0.875 h at the production geometry


def regime_of(wth):
    """stable / neutral / convective from the PRESCRIBED virtual heat flux."""
    if wth > WTH_NEUTRAL:
        return "convective"
    if wth < -WTH_NEUTRAL:
        return "stable"
    return "neutral"


# === THE ONE DRIFTING LIMIT THE CORPUS ADMITS, AND ONLY ON THE RUNGS THAT CANNOT AVOID IT
# `zi-neutral` admits a seed whose ONLY drifting limit is z_i and whose rung is NEUTRAL.
# Nothing else: a neutral seed drifting in u*, sigma_w or a Kljun geometry term is still
# refused, and a CONVECTIVE seed drifting in z_i is still refused -- a CBL under a capping
# inversion and subsidence has a depth the box is holding, so drift there is a defect rather
# than a property of the flow.
DRIFT_MODES = ("off", "zi-neutral", "any")
ZI_LIMIT = "z_i"


def _drift_admitted(mode, drifting, wth):
    """Is this seed's DRIFTING set admissible under `mode`? -> (bool, reason)"""
    if mode == "any":
        return True, "--allow-drifting: any drifting limit, any rung"
    if mode != "zi-neutral":
        return False, ""
    if set(drifting) != {ZI_LIMIT}:
        return False, (f"drifting in {', '.join(sorted(drifting))}, not in z_i alone")
    if regime_of(wth) != "neutral":
        return False, (
            f"a {regime_of(wth)} rung drifting in z_i. The acceptance is specific to the "
            f"neutral rungs, where the depth has no equilibrium at any affordable spin-up; "
            f"a convective rung's depth is being HELD by its capping inversion and "
            f"subsidence, and a stable one's by its stratification, so drift there is a "
            f"defect rather than a property of the flow")
    return True, "a neutral rung drifting only in z_i"


def load_library(index_path, library_dir, available_only=False,
                 allow_indeterminate=True, allow_drifting="any", exclude=()):
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
    indeterminate, indeterminate_blocked, excluded_by_hand = [], [], []
    drifting_used = []
    if os.path.isdir(library_dir):
        for m in sorted(glob.glob(os.path.join(library_dir, "*", "return",
                                               "manifest.json"))):
            s = json.load(open(m))
            ret = os.path.dirname(m)
            # AN EXPLICIT, RECORDED EXCLUSION. Distinct from every automatic one above:
            # this is a human decision about a specific seed, and it is named in the
            # output so it cannot be mistaken for the library simply not containing it.
            if s["job"] in exclude:
                excluded_by_hand.append(s["job"])
                have.add(s["job"])
                continue
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
            gate, indet = None, []
            gp = os.path.join(ret, "stationarity.json")
            if os.path.exists(gp):
                try:
                    _g = json.load(open(gp))
                    gate = bool(_g.get("pass"))
                    # DRIFTING AND INDETERMINATE ARE NOT THE SAME REFUSAL. A seed with a
                    # DRIFTING limit is known to be moving in a footprint-controlling
                    # parameter -- restarting a case from it starts the case mid-transient.
                    # A seed with only INDETERMINATE limits is not known to be moving; its
                    # stationarity is UNESTABLISHED, because the trend estimator cannot
                    # resolve its own limit in a 3.0 h run (TKE_BL/u*^2 and z_i decorrelate
                    # on the eddy turnover, so n_eff saturates at 3-5 however finely the
                    # run is dumped). Both are refused by default. Only the second can be
                    # opted into, and only explicitly.
                    indet = list(_g.get("indeterminate") or [])
                    s["_drifting"] = list(_g.get("drifting") or [])
                    s["_indeterminate"] = indet
                    # === THE z_i ROW ITSELF, NOT ONLY WHICH BUCKET IT LANDED IN =========
                    # Whether z_i reads DRIFTING or INDETERMINATE depends on the SCORING
                    # WINDOW, not only on the flow: measured on seed_nbl-deep_a015, the
                    # same run gives +5.76 %/h DRIFTING over 2.0 h and +4.97 %/h
                    # INDETERMINATE over 1.5 h, because the shorter window cannot resolve
                    # the trend against its own SE. The 2.0 h seed ceiling makes 1.5 h the
                    # derived width, so the boolean below would quietly stop firing on
                    # seeds in exactly the state it was added for.
                    #
                    # So the TREND travels with every pair beside the verdict. A consumer
                    # checking the corpus's z_i distribution wants the number; the bucket is
                    # a threshold applied to it, and the threshold is not the evidence.
                    for _r in (_g.get("gated") or []):
                        if isinstance(_r, dict) and _r.get("name") == ZI_LIMIT:
                            s["_zi_verdict"] = _r.get("verdict")
                            s["_zi_trend"] = _r.get("trend_pct_per_h")
                            s["_zi_trend_se"] = _r.get("trend_se_pct_per_h")
                            s["_zi_limit"] = _r.get("limit")
                            break
                    s["_score_h"] = _g.get("score_h")
                except (ValueError, OSError):
                    gate = None
            ach = s.get("achieved") or {}
            if ach.get("pass") is False or gate is False:
                only_indet = bool(indet) and not s.get("_drifting")
                if only_indet and allow_indeterminate:
                    s["_gate_state"] = "INDETERMINATE"
                    indeterminate.append((s["job"], indet))
                elif s.get("_drifting") and _drift_admitted(
                        allow_drifting, s["_drifting"],
                        float((s.get("target") or {}).get("wth_virtual", 0.0)))[0]:
                    # A NARROW, LOUD, DEFAULT-OFF OPT-IN, AND IT IS NOT THE SAME
                    # CONCESSION AS --allow-indeterminate.
                    #
                    # Why it exists at all: PLAN.md item 0aa predicted, from a
                    # scoring-window sweep, that z_i in the NEUTRAL rungs is "TRENDING AWAY
                    # from band ... a longer run resolves these into a FAIL, not a pass".
                    # seed_nbl-deep_a015 at 2.917 sim-h duly resolved it -- +5.76 %/h
                    # against a 3 %/h limit -- so the refusal above makes the NEUTRAL half
                    # of the corpus unbuildable at any affordable spin-up, and a neutral
                    # Ekman layer's depth genuinely does keep growing for several inertial
                    # periods (35-50 simulated hours; PROJECT_BRIEF.md makes the same argument for
                    # u*, whose fix was to gate on a RATIO -- and z_i is the one gated
                    # quantity with no ratio to take).
                    #
                    # THE CORPUS SETS THIS, AT `zi-neutral`, AS OF 2026-08-30. The decision
                    # is the user's and it is recorded here rather than only in a log: the
                    # z_i limit is UNSATISFIABLE on a neutral rung, not failed, so refusing
                    # it refuses the neutral half of the corpus for a state no spin-up can
                    # reach. Letting z_i grow to a FIXED 2.0 sim-h ceiling is deterministic
                    # and reproducible, and z_i is a weak input at a 30 m receptor -- Kljun's
                    # only z_i channel, 1/(1 - z_m/h), spans ~5% over h = 400-1200 m. The
                    # achieved z_i is recorded per case so the distribution can be checked;
                    # if it turns out too narrow to train on, a per-case lid from the
                    # sounding is the fallback. NO CAPPING INVERSION was added, deliberately.
                    #
                    # What it does NOT do is call the seed stationary. gate_state is stamped
                    # DRIFTING on every pair and make_pair.py writes a warning into the
                    # training record. `any` remains a separate, wider, default-off opt-in.
                    s["_gate_state"] = "DRIFTING"
                    s["_drift_reason"] = _drift_admitted(
                        allow_drifting, s["_drifting"],
                        float((s.get("target") or {}).get("wth_virtual", 0.0)))[1]
                    drifting_used.append((s["job"], s["_drifting"], s["_drift_reason"]))
                else:
                    if only_indet:
                        indeterminate_blocked.append((s["job"], indet))
                    else:
                        # NAME WHY THE MODE DID NOT ADMIT IT. Under `zi-neutral` a refusal
                        # can now mean two quite different things -- the wrong limit, or the
                        # wrong rung -- and "DRIFTING" alone does not distinguish them.
                        why = _drift_admitted(
                            allow_drifting, s.get("_drifting") or [],
                            float((s.get("target") or {}).get("wth_virtual", 0.0)))[1]
                        rejected.append((s["job"], s.get("_drifting") or [], why))
                    have.add(s["job"])      # and do NOT fall back to its index entry
                    continue
            else:
                s["_gate_state"] = "PASS" if gate else "unjudged"
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
    if excluded_by_hand:
        print(f"  EXCLUDED BY REQUEST ({len(excluded_by_hand)}): "
              f"{', '.join(sorted(excluded_by_hand))} -- named on the command line, not "
              f"rejected by any gate.")
    # === THE WHOLE LIBRARY IS THE DEFAULT, AND IT IS SAID ONCE ==========================
    # With the gate no longer filtering, the per-seed notices below would print a paragraph
    # for every seed in the library on every case -- 30 of them, identical, on 1469 cases.
    # That is not disclosure, it is noise that trains the reader to skip the block where the
    # genuinely per-seed notices live. So the POLICY is stated once, with the counts, and
    # the per-seed loops below run only when a seed was admitted by a NARROWER opt-in than
    # "everything", which is the case where which seed got in is the information.
    whole_library = bool(allow_indeterminate) and allow_drifting == "any"
    if whole_library and (drifting_used or indeterminate):
        nd, ni = len(drifting_used), len(indeterminate)
        print(f"  SEED SELECTION USES THE WHOLE LIBRARY (the default since 2026-08-31): "
              f"{nd} seed(s) with a DRIFTING limit and {ni} INDETERMINATE are ranked "
              f"alongside any that passed. A seed is an INITIAL CONDITION, not a corpus "
              f"point -- the case adjusts 30 min under its own forcing and every ML input "
              f"is measured by window_stats over the same window as the footprint, so the "
              f"pair is self-consistent whatever the seed's drift state. gate_state is "
              f"still stamped on every pair. --strict-gate restores the refusal.")
    for job, lim, why in (drifting_used if not whole_library else ()):
        if allow_drifting == "zi-neutral":
            print(f"  *** USING A z_i-DRIFTING NEUTRAL SEED: {job} ({why}), admitted by "
                  f"--allow-drifting zi-neutral. The limit is UNSATISFIABLE on this rung "
                  f"rather than failed -- a neutral Ekman layer's depth grows for several "
                  f"inertial periods -- so the seed is frozen at a fixed 2.0 sim-h ceiling "
                  f"and its achieved z_i is recorded per pair. Every pair carries "
                  f"seed.gate_state = DRIFTING and zi_accepted_drifting = true.")
        else:
            print(f"  *** USING A DRIFTING SEED: {job} is DRIFTING in {', '.join(lim)} and "
                  f"was admitted by --allow-drifting {allow_drifting}. That is a STRONGER "
                  f"defect than INDETERMINATE: the seed is KNOWN to be moving in a "
                  f"footprint-controlling parameter, so this case starts mid-transient and "
                  f"the 30-minute adjustment is not there to absorb it. Every pair built on "
                  f"it carries seed.gate_state = DRIFTING.")
    if rejected:
        print(f"  EXCLUDED {len(rejected)} seed(s) whose gate found a limit DRIFTING: "
              f"{', '.join(sorted(j for j, _l, _w in rejected))}")
        if allow_drifting != "off":
            for j, lim, why in sorted(rejected):
                print(f"    {j}: DRIFTING in {', '.join(lim) or '?'} -- not admitted under "
                      f"--allow-drifting {allow_drifting}"
                      + (f" ({why})" if why else ""))
    for j, lim in sorted(indeterminate_blocked):
        print(f"  EXCLUDED {j}: stationarity UNESTABLISHED (INDETERMINATE on "
              f"{', '.join(lim)}) -- nothing is drifting, but the gate cannot resolve "
              f"these limits in a 3.0 h run. Pass --allow-indeterminate to use it anyway; "
              f"every pair built on it is stamped with this state.")
    for j, lim in (sorted(indeterminate) if not whole_library else ()):
        print(f"  *** USING AN INDETERMINATE SEED: {j} is INDETERMINATE on "
              f"{', '.join(lim)} and was admitted by --allow-indeterminate. Its "
              f"stationarity is UNESTABLISHED, not established. Every pair built on it "
              f"carries seed.gate_state = INDETERMINATE.")
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
    heading is `G_dir_from - ekman_backing_deg(z_i/L)`, and every one of the 120
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
    ap.add_argument("--exclude", default=None,
                    help="comma-separated seed job names to exclude explicitly, "
                         "regardless of their gate verdict. Named in the output.")
    ap.add_argument("--allow-drifting", nargs="?", const="any", default="any",
                    choices=DRIFT_MODES,
                    help="admit a seed with a DRIFTING limit. NOT the same concession as "
                         "--allow-indeterminate: a drifting seed is KNOWN to be moving in a "
                         "footprint-controlling parameter, where an indeterminate one is "
                         "merely unestablished. "
                         "`zi-neutral` (THE CORPUS DEFAULT since 2026-08-30) admits only a "
                         "NEUTRAL rung whose ONLY drifting limit is z_i -- the one limit "
                         "that is unsatisfiable rather than failed, because a neutral Ekman "
                         "layer's depth grows for several inertial periods and PLAN.md 0aa "
                         "measured it trending AWAY from band as the run lengthens. "
                         "`any` (the bare flag) admits any drifting limit on any rung and "
                         "stays a wide, loud, manual opt-in. `off` refuses both. "
                         "gate_state is stamped DRIFTING on every pair either way.")
    ap.add_argument("--allow-indeterminate", action="store_true", default=True,
                    help="admit seeds whose gate returned INDETERMINATE. ON BY DEFAULT "
                         "since 2026-08-31 -- see --strict-gate for the reasoning and for "
                         "how to restore the refusal. NOT a pass: the state is still "
                         "stamped onto every pair.")
    ap.add_argument("--strict-gate", action="store_true",
                    help="RESTORE THE PRE-2026-08-31 BEHAVIOUR: refuse every seed whose "
                         "stationarity gate did not return a clean PASS. Equivalent to "
                         "--allow-drifting off with --allow-indeterminate off. "
                         "THE DEFAULT IS NOW THE WHOLE LIBRARY, and the reason is that a "
                         "seed is an INITIAL CONDITION rather than a corpus point: the "
                         "case restarts from it, integrates 30 minutes of adjustment under "
                         "its OWN sounding's forcing, and every ML input is then measured "
                         "by window_stats over exactly the same window as the footprint. "
                         "So the pair is self-consistent whatever the seed's drift state, "
                         "and refusing a seed removes a RESTART POINT without removing any "
                         "error. What it removed, measured on the 30-seed library "
                         "(SEED_LIBRARY_RESULT.md): all six cbl-shallow seeds, leaving the "
                         "weakly-convective rung with no restart point at all; eight of "
                         "twelve neutral seeds, dropping the neutral half to four base "
                         "angles and firing this script's own half-spacing warning; and "
                         "the Ekman-backing calibration down to n = 1 and 2 on three of "
                         "five rungs. Use this flag to reproduce a pre-2026-08-31 pick, "
                         "not to build a corpus.")
    ap.add_argument("--available-only", action="store_true",
                    help="rank only seeds that have a returned artifact on disk. Use "
                         "while the library is partially built: an unbuilt seed cannot be "
                         "restarted from, and its heading is an estimate rather than a "
                         "measurement, so ranking it against a spun seed compares a guess "
                         "with a number.")
    a = ap.parse_args()
    if a.strict_gate:
        a.allow_drifting, a.allow_indeterminate = "off", False

    fc = json.load(open(a.forcing))
    lab, par = fc["labels"], fc["params"]
    zi_c = float(lab["zi_m"])
    wth_c = float(par["surflayer_wth"])
    reg_c = regime_of(wth_c)
    # the heading the LES will actually blow FROM, which is what an achieved seed reports
    dir_c = float(lab["predicted_10m_dir_deg"])
    L_c = lab.get("L_estimate")
    zoL_c = (a.zm / float(L_c)) if L_c not in (None, 0) else float("nan")

    seeds = load_library(a.index, a.library, available_only=a.available_only,
                         allow_indeterminate=a.allow_indeterminate,
                         allow_drifting=a.allow_drifting,
                         exclude=set(x for x in (a.exclude or "").split(",") if x))
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
                         "gate_state": s.get("_gate_state", "unjudged"),
                         "gate_indeterminate": s.get("_indeterminate") or [],
                         "gate_drifting": s.get("_drifting") or [],
                         # THE FLAG THE TRAINING RECORD FILTERS ON. Distinct from a bare
                         # gate_state = DRIFTING: it says the drift is the KNOWN,
                         # unsatisfiable z_i one on a neutral rung, accepted by design,
                         # rather than an unexplained one admitted by a wide manual flag.
                         "zi_accepted_drifting": bool(
                             s.get("_gate_state") == "DRIFTING"
                             and set(s.get("_drifting") or []) == {ZI_LIMIT}
                             and s["regime"] == "neutral"),
                         "drift_reason": s.get("_drift_reason"),
                         # z_i's own gate row, so the record carries the EVIDENCE and not
                         # only the bucket a threshold put it in at one scoring width.
                         "zi_gate_verdict": s.get("_zi_verdict"),
                         "zi_trend_pct_per_h": s.get("_zi_trend"),
                         "zi_trend_se_pct_per_h": s.get("_zi_trend_se"),
                         "zi_trend_limit_pct_per_h": s.get("_zi_limit"),
                         "gate_score_h": s.get("_score_h"),
                         # THE SEED'S OWN ACHIEVED DEPTH, IN BOTH CURRENCIES. With z_i left
                         # to grow to a fixed ceiling, where it got to is a property of the
                         # library that the corpus's z_i distribution inherits -- so it is
                         # recorded per pair rather than reconstructed from the job later.
                         "seed_zi_achieved_m": (
                             float((s.get("achieved") or {}).get("zi"))
                             if (s.get("achieved") or {}).get("zi") is not None else None),
                         "seed_zi_peakfrac_m": (
                             float((s.get("achieved") or {}).get("zi_peakfrac"))
                             if (s.get("achieved") or {}).get("zi_peakfrac") is not None
                             else None),
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
    print(f"  case: SEED-MATCHING regime {reg_c} (|w'th_v'| > {WTH_NEUTRAL} K m/s; "
          f"bin/case_surface.py answers a different question on a different threshold) (w'th_v' {wth_c:+.4f}), z_i {zi_c:.0f} m, "
          f"heading {dir_c:.1f} deg, z_m/L "
          f"{'n/a' if not np.isfinite(zoL_c) else f'{zoL_c:+.4f}'}")
    print(f"  {'seed':<24}{'rot':>4}{'dir':>6}{'d_dir':>7}{'z_i':>7}"
          f"{'c_zi':>7}{'c_dir':>7}{'cost':>7}  {'regime':>10} {'label':>8}")
    for r in rows[:6]:
        print(f"  {r['job']:<24}{r['rot']:>4}{r['seed_dir_deg']:>6.0f}"
              f"{r['d_dir_deg']:>7.1f}{r['seed_zi_m']:>7.0f}{r['cost_zi']:>7.3f}"
              f"{r['cost_dir']:>7.3f}{r['cost']:>7.3f}  {r['regime']:>10} "
              f"{r['labelled_by']:>8}")
    print(f"\n  CHOSEN: {best['job']} --rot {best['rot']}  [gate "
          f"{best.get('gate_state','unjudged')}]")
    if best.get("gate_state") == "INDETERMINATE":
        print(f"    *** its stationarity is UNESTABLISHED on "
              f"{', '.join(best['gate_indeterminate'])} -- not established, and not a "
              f"pass. This is recorded on the pair, not waved through.")
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
    if best["d_dir_deg"] > 0.5 * DIR_SCALE:
        print(f"  WARNING: {best['d_dir_deg']:.1f} deg exceeds half the {DIR_SCALE:.0f} deg "
              f"library spacing; a base angle is missing, or a seed drifted off its own "
              f"by more than the projection removed.")

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
