#!/usr/bin/env python3
"""Assemble ONE (input, target) training record. Stage 8, the last of the pipeline.

    input   Kljun's scalars, every one of them read off the LES window itself
    target  the 122 x 122 LPDM flux footprint on that same window

=== THE INPUTS COME FROM THE LES, NOT FROM THE SOUNDING ===

This is the load-bearing decision of the whole corpus design and it is worth stating
plainly. The sounding chooses which state to simulate; it does not describe the state that
results. u*, U(z_m), sigma_v, L, z_i and the wind direction are all read back off the
window by lpdm/les_stats.py:window_stats, so:

  * an imperfectly-matched seed, or an adjustment that has not fully closed, moves where a
    case LANDS in input space without making the pair wrong -- input and target are
    measured on the same fields, so they are consistent by construction;
  * the Ekman turning, the inertial oscillation and the entrainment growth are all
    absorbed as LABELS rather than errors, which is what PROJECT_BRIEF.md already requires for
    direction ("Achieved direction is not forcing direction");
  * and the emulator is being taught the map the LES actually realises, which is the only
    map the LPDM footprint corresponds to.

The sounding's own numbers are still carried, as `forcing`, so the achieved-vs-requested
gap is measurable across the corpus instead of assumed small.

=== SPLIT BY RUN, NEVER BY SAMPLE ===

`run_id` is written into every record and is the ONLY legitimate grouping key for a
train/test split. The effective sample size for generalisation is the number of LES runs,
not the number of footprint cells or sub-windows drawn from one run (PROJECT_BRIEF.md, ML model).

=== WHAT IS NOT AN INPUT ===

z_0 and the receptor height are properties of the site, identical in every case, so they
are recorded as provenance rather than offered as features -- a constant column is not a
predictor, and this is a single-tower emulator by design.

usage: make_pair.py --tag <case> --footprint results/<case>.json
                    [--forcing results/forcing/<case>.json] [--seed results/pick/<case>.json]
                    [--outdir pairs]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

# Kljun's scalar inputs, and nothing else. PROJECT_BRIEF.md: "Inputs are Kljun's scalars only."
KLJUN_INPUTS = ("u_mean", "ustar", "sigma_v", "h", "L", "wdir")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="the case id; becomes run_id")
    ap.add_argument("--grid", default=None,
                    help="the surface directory this case ran on. Its z0m.npy sets the "
                         "recorded geometric-mean z0; without it the field is written as "
                         "null rather than as a number from another grid.")
    ap.add_argument("--footprint", required=True,
                    help="the .json stage5_footprint.py wrote (its .npz sits beside it)")
    ap.add_argument("--forcing", default=None)
    ap.add_argument("--seed", default=None, help="the pick_seed.py JSON")
    ap.add_argument("--outdir", default="pairs")
    ap.add_argument("--copy-npz", action="store_true",
                    help="copy the footprint arrays into the pair rather than "
                         "referencing them; the corpus is then self-contained")
    a = ap.parse_args()

    if not os.path.exists(a.footprint):
        print(f"FATAL: {a.footprint} does not exist", file=sys.stderr)
        return 2
    fp = json.load(open(a.footprint))
    npz_path = a.footprint[:-5] + ".npz"
    if not os.path.exists(npz_path):
        print(f"FATAL: {npz_path} does not exist; the footprint arrays are the TARGET",
              file=sys.stderr)
        return 2

    st = fp.get("stats") or {}
    missing = [k for k in KLJUN_INPUTS if st.get(k) is None]
    if missing:
        print(f"FATAL: the window stats carry no {missing}; without them there is no "
              f"input vector and the record would be a target with no features",
              file=sys.stderr)
        return 2
    inputs = {k: float(st[k]) for k in KLJUN_INPUTS}
    # inf is a legitimate value of L (exactly neutral) and is NOT corruption -- but it
    # cannot go into a model, so it is carried as 1/L, which is finite everywhere and is
    # the form the similarity functions actually use.
    inputs["inv_L"] = (1.0 / inputs["L"]) if np.isfinite(inputs["L"]) else 0.0
    for k, v in inputs.items():
        if k != "L" and not np.isfinite(v):
            print(f"FATAL: input {k} = {v} is not finite", file=sys.stderr)
            return 2

    with np.load(npz_path) as z:
        target = np.asarray(z["les"], dtype=np.float64)
        kljun = np.asarray(z["kljun"], dtype=np.float64)
        xc, yc = np.asarray(z["xc"]), np.asarray(z["yc"])
    if not np.isfinite(target).all():
        print("FATAL: the footprint target is not finite", file=sys.stderr)
        return 2
    if target.shape != (len(yc), len(xc)) and target.shape != (len(xc), len(yc)):
        print(f"FATAL: target {target.shape} does not match the "
              f"{len(yc)} x {len(xc)} raster", file=sys.stderr)
        return 2

    # SPLIT BY PARENT CASE, NOT BY TAG. A case now runs TWO sampling windows inside one
    # FastEddy invocation and yields tags <case>_w0 and <case>_w1. They share a seed, an
    # adjustment, a sounding and a surface, so splitting on the tag would put one in train
    # and the other in validation and leak nearly everything that makes them what they are.
    # The effective sample size for generalisation is the number of PARENTS.
    parent = re.sub(r"_w\d+$", "", a.tag)
    widx = None
    m_ = re.search(r"_w(\d+)$", a.tag)
    if m_:
        widx = int(m_.group(1))
    z0_geom = None
    if a.grid:
        _z0p = os.path.join(a.grid, "z0m.npy")
        if os.path.exists(_z0p):
            z0_geom = float(np.exp(np.log(np.load(_z0p)).mean()))
        else:
            print(f"  WARNING: no z0m.npy in {a.grid}; z0_geometric_m recorded as null")
    else:
        print("  WARNING: no --grid given; z0_geometric_m recorded as null rather than "
              "as a number carried over from another grid")
    rec = {
        "run_id": a.tag,
        "parent": parent,
        "window_index": widx,
        "split_key": parent,     # SPLIT BY RUN. Never by sample, never by window.
        "inputs": inputs,
        "target": {"file": os.path.basename(npz_path) if a.copy_npz else npz_path,
                   "array": "les", "shape": list(target.shape),
                   "units": "1/m^2, normalised so the raster integrates to "
                            f"{float(fp.get('integral_les', float('nan'))):.4f}",
                   "reference": "kljun"},
        "site": {"z_recept_m": fp.get("zm"), "z_agl_m": fp.get("zm_agl"),
                 "d_recept_m": fp.get("d_recept"), "z_target_m": fp.get("z_target"),
                 # READ OFF THE GRID THIS CASE RAN ON, never a literal. It was 0.1435 --
                 # the 122^2 @ 16 m value -- and it is 0.0615 at 30 m, because the box
                 # takes in more lake. A constant that is really a grid property is
                 # FASTEDDY_TRAPS.md 19, and it has bitten five times in one day.
                 "z0_geometric_m": z0_geom,
                 "note": "constant across the corpus: a single-tower emulator"},
        "closure": {"sgs_most": fp.get("sgs_most"), "mode": fp.get("sgs_most_mode"),
                    "form": fp.get("sgs_most_form"),
                    "subgrid_weight": fp.get("sgs_subgrid_weight"),
                    "eps_consistent": fp.get("sgs_eps_consistent"),
                    "tback_s": fp.get("tback"), "rel_seconds": fp.get("rel_seconds"),
                    "note": "the near field is closure-dominated at z/Delta ~ 1; the "
                            "sigma_w anchor is worth 46-66% shape L1 against a 38% "
                            "sampling floor (PROJECT_BRIEF.md)"},
        "diagnostics": {"integral_les": fp.get("integral_les"),
                        "integral_kljun": fp.get("integral_kljun"),
                        # THE ASYMPTOTE IS 1 - z_m/z_i, NOT 1 (Steinfeld et al. 2008,
                        # after Horst & Weil 1992). At 30 m in an 800 m boundary layer
                        # that is 3.75%, the size of effects this project gates on.
                        "integral_asymptote": fp.get("integral_asymptote"),
                        "integral_over_asymptote": fp.get("integral_over_asymptote"),
                        "touchdowns": fp.get("touchdowns"),
                        "overlap80_kljun": fp.get("overlap_kljun"),
                        "cover_share": fp.get("cover_share", {}),
                        # THE SHARE TRAVELS WITH ITS OWN SAMPLING SPREAD. A share quoted
                        # without an SE cannot be compared to anything: the h defect moved
                        # the array share 0.8 points against a 3.66-point SE and looked
                        # like a result. PROJECT_BRIEF.md: score a second moment against its own
                        # sampling spread, never against a number you picked.
                        "cover_share_se": fp.get("cover_share_se", {}),
                        "cover_share_groups": fp.get("cover_share_groups_n"),
                        # THE PER-CASE QC STAMP, so a suspect pair is identifiable from
                        # the training record alone rather than only from a log that was
                        # deleted with the fields.
                        "floor_health": ((fp.get("floor") or {}).get("health") or {}),
                        "wind_angle": fp.get("wind_angle")},
    }

    if a.forcing and os.path.exists(a.forcing):
        fc = json.load(open(a.forcing))
        lab = fc["labels"]
        rec["forcing"] = {
            "valid_time": fc["provenance"].get("valid_time"),
            "representable": fc.get("representable"),
            "requested": {"zi_m": lab["zi_m"], "G": lab["G_speed"],
                          "G_dir_from_deg": lab["G_dir_from_deg"],
                          "predicted_10m_dir_deg": lab["predicted_10m_dir_deg"],
                          "hrrr_10m_dir_deg": lab["hrrr_10m_dir_deg"],
                          "wth_virtual": lab["wth_virtual"], "bowen": lab["bowen"]},
            # ACHIEVED MINUS REQUESTED. Not an error term -- the achieved value is the
            # input -- but the distribution of these across 1825 cases is the only way to
            # see whether 30 minutes of adjustment is actually enough.
            "achieved_minus_requested": {
                "zi_m": (None if st.get("h") is None
                         else round(float(st["h"]) - float(lab["zi_m"]), 2)),
                "dir_deg": (None if st.get("wdir") is None else round(
                    float(((float(st["wdir"]) - float(lab["predicted_10m_dir_deg"])
                            + 180.0) % 360.0) - 180.0), 2))},
        }
        if fc.get("representable") is False:
            rec.setdefault("warnings", []).append(
                "the sounding's z_i exceeds what the domain supports; this pair is "
                "domain-constrained and should be excluded or reported as such")

    if a.seed and os.path.exists(a.seed):
        pk = json.load(open(a.seed))
        ch = pk["chosen"]
        rec["seed"] = {"job": ch["job"], "rot": ch["rot"],
                       "d_dir_deg": ch["d_dir_deg"],
                       "seed_zi_m": ch["seed_zi_m"],
                       "labelled_by": ch["labelled_by"],
                       "regime_match": ch["regime_match"],
                       # THE SEED'S GATE STATE TRAVELS WITH EVERY PAIR. A pair built on a
                       # seed whose stationarity was never established is not wrong, but
                       # it is qualified, and the qualification has to survive in the
                       # training record rather than only in a log.
                       "gate_state": ch.get("gate_state", "unjudged"),
                       "gate_indeterminate": ch.get("gate_indeterminate", []),
                       "gate_drifting": ch.get("gate_drifting", []),
                       # WHERE THE SEED'S HEADING CAME FROM. The adjustment does not close
                       # a direction gap -- measured, it widened one by 10.5 deg -- so the
                       # seed is carried forward at its own drift rate. Recording the
                       # frozen heading, the rate and the horizon makes that reconstructable
                       # per pair instead of only in aggregate.
                       "seed_dir_deg": ch.get("seed_dir_deg"),
                       "seed_dir_frozen_deg": ch.get("seed_dir_frozen_deg"),
                       "dwdir_dt_deg_per_h": ch.get("dwdir_dt_deg_per_h"),
                       "project_h": ch.get("project_h")}
        # THE WARNING BELONGS WHERE `ch` EXISTS. It was written into the --forcing block,
        # which runs BEFORE the seed is read, so it raised UnboundLocalError and killed
        # stage 8 on two finished cases -- after the GPU and the LPDM had both succeeded.
        if ch.get("gate_state") == "INDETERMINATE":
            rec.setdefault("warnings", []).append(
                "the seed this pair restarts from has UNESTABLISHED stationarity: "
                + ", ".join(ch.get("gate_indeterminate") or [])
                + " could not be resolved against their own limits in a 3.0 h spin-up. "
                  "Nothing was drifting; nothing was established either.")

    os.makedirs(a.outdir, exist_ok=True)
    if a.copy_npz:
        shutil.copyfile(npz_path, os.path.join(a.outdir, os.path.basename(npz_path)))
    out = os.path.join(a.outdir, f"{a.tag}.json")
    json.dump(rec, open(out, "w"), indent=1, sort_keys=True)
    # One line per case, so 1825 records are enumerable without opening 1825 files.
    #
    # REWRITTEN, NOT APPENDED. bin/run_corpus.sh is resumable and a case can legitimately
    # be re-run -- a fixed seed, a corrected sounding, a killed analysis. A blind append
    # would put the same run_id in the index twice with different inputs, and the second
    # copy would be silently over-weighted by any consumer that iterates the file. The
    # line for this run_id is replaced in place and every other line is preserved
    # byte-for-byte, so a re-run costs nothing and concurrent cases do not clobber
    # each other's rows.
    idx = os.path.join(a.outdir, "index.jsonl")
    row = json.dumps({"run_id": a.tag, "record": os.path.basename(out),
                      "target": rec["target"]["file"],
                      "valid_time": rec.get("forcing", {}).get("valid_time"),
                      "inputs": inputs})
    keep, replaced = [], False
    if os.path.exists(idx):
        for ln in open(idx):
            ln = ln.rstrip("\n")
            if not ln:
                continue
            try:
                same = json.loads(ln).get("run_id") == a.tag
            except json.JSONDecodeError:
                same = False           # keep an unparseable line rather than lose it
            if same:
                replaced = True
                continue
            keep.append(ln)
    keep.append(row)
    tmp = idx + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write("\n".join(keep) + "\n")
    os.replace(tmp, idx)               # atomic, so a kill cannot truncate the index

    print(out)
    print(f"  index {idx}: {len(keep)} record(s)"
          f"{' (this run_id replaced)' if replaced else ''}")
    print(f"  run_id {a.tag}   target {target.shape}  "
          f"integral {rec['diagnostics']['integral_les']}")
    print(f"  inputs: " + "  ".join(f"{k} {v:.4g}" for k, v in inputs.items()))
    if "forcing" in rec:
        d = rec["forcing"]["achieved_minus_requested"]
        print(f"  achieved - requested: z_i {d['zi_m']} m, direction {d['dir_deg']} deg")
    if "seed" in rec:
        print(f"  from seed {rec['seed']['job']} rot {rec['seed']['rot']} "
              f"({rec['seed']['d_dir_deg']:.1f} deg away)")
    for w in rec.get("warnings", []):
        print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
