#!/usr/bin/env python3
"""Compare two runs of the SAME seed, scored identically at the SAME simulated time.

WHAT THIS IS FOR, AND WHAT IT CANNOT BE. The seed library moved to a new CUDA toolkit and a
new image (`FASTEDDY_TRAPS.md` §23), and a seed produced by the new one has to be shown to
be the same KIND of boundary layer as the one produced by the old. It cannot be shown to be
the SAME boundary layer: FastEddy is not bitwise reproducible run to run on one GPU with one
binary, and over two simulated hours two runs of one seed are two independent turbulence
realisations. `jobs30/*/manifest.json` says so in its own `reproducibility` field.

So this is not a regression test with a tolerance. It answers a narrower and answerable
question: **do the two runs land in the same place in the library's own coordinates** --
the achieved `u*`, `U`, direction, `sigma_v`, `sigma_w`, depth and the two Kljun geometry
terms that `bin/pick_seed.py` matches cases on. Those are what a seed IS, downstream.

TWO THINGS IT DOES TO MAKE THE COMPARISON FAIR, AND BOTH MATTER:

  * IT SCORES AT A MATCHED STEP. A run that stopped early and a run that ran to the ceiling
    are at different points of the inertial oscillation, and `u*` moves ~18% over it while
    `U/u*` moves 0.6% (PROJECT_BRIEF.md). Comparing end-of-run to end-of-run would report the
    oscillation.
  * IT USES THE PRODUCTION GATE, NOT A REIMPLEMENTATION. Both sides go through
    bin/seed_stationarity.py with identical arguments, because this project has already
    shipped one wrong result from a gate that carried its own copy of a production function.

    python3 bin/seed_compare.py --a jobs30/seed_cbl-mid_a015 --b /out/work/seed_cbl-mid_a015 \\
        --step 106920 --label-a "CUDA 11.8, 2026-08-30" --label-b "CUDA 13.0, in-image"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

# The achieved state pick_seed.py matches on, in the order a reader wants it.
KEYS = ["ustar", "U", "wdir", "sigma_v", "sigma_w", "zi", "zi_peakfrac", "tke_peak",
        "theta0", "x_peak", "x90"]
UNITS = {"ustar": "m/s", "U": "m/s", "wdir": "deg", "sigma_v": "m/s", "sigma_w": "m/s",
         "zi": "m", "zi_peakfrac": "m", "tke_peak": "m2/s2", "theta0": "K",
         "x_peak": "m", "x90": "m"}


def dumps(d, base):
    out = {}
    for p in glob.glob(os.path.join(d, f"{base}.[0-9]*")):
        try:
            out[int(p.rsplit(".", 1)[1])] = p
        except ValueError:
            pass
    return out


def score(job, step, tmp, py, side):
    """bin/seed_stationarity.py on the dumps up to `step`, through the production path.

    `side` NAMES THE SCRATCH DIRECTORY, AND IT HAS TO. The first version keyed it on
    os.path.basename(job) -- and the whole point of this script is to compare two runs of
    the SAME seed, so both sides had the same basename, both wrote into one directory, and
    the second side's symlinks were skipped as already existing. It then scored the FIRST
    side's dumps twice and reported +0.00% on every field and every ratio: a clean,
    plausible, completely empty result. Two turbulence realisations cannot agree to zero,
    and that is the tell -- the check below makes it an error instead of a reading.
    """
    man = json.load(open(os.path.join(job, "manifest.json")))
    r, g = man["run"], man.get("gate", {})
    base = r["outFileBase"]
    have = dumps(os.path.join(job, "output"), base)
    sel = {k: v for k, v in have.items() if k <= step}
    if not sel:
        raise SystemExit(f"FATAL: {job} has no dump at or below step {step} "
                         f"(has {sorted(have)[:3]}...{sorted(have)[-1:]})")
    if max(sel) != step:
        raise SystemExit(f"FATAL: {job}'s newest dump at or below {step} is {max(sel)}. "
                         f"A matched-step comparison needs the same step on both sides.")
    # SYMLINKS, NOT COPIES -- a dump is 73 MB and there are up to 24 of them. ONE RUN PER
    # DIRECTORY (FASTEDDY_TRAPS.md 18c): only this base name is linked, so an accelerator
    # burn-in's FE_SEED_ACC.* cannot interleave into the series.
    d = os.path.join(tmp, f"{side}_{os.path.basename(job.rstrip('/'))}")
    os.makedirs(d, exist_ok=True)
    for k, p in sel.items():
        dst = os.path.join(d, f"{base}.{k}")
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(p), dst)
    dt, sim_h = float(r["dt"]), step * float(r["dt"]) / 3600.0
    # The same rule jobs/run_seed.sh uses: the window must fit strictly inside the run, or
    # seed_stationarity.py refuses it for reaching step 0 where a cold start has u* = 0.
    score_h = min(2.0, max(0.5, sim_h - 0.5))
    js = os.path.join(tmp, os.path.basename(d) + ".json")
    cmd = [py, "bin/seed_stationarity.py", d, "--dt", str(dt),
           "--wth", str(man["target"]["wth_virtual"]),
           "--zm", str(g.get("zm", 30.0)), "--k", str(g.get("k", 3)),
           "--dx", str(man.get("grid", {}).get("dx", 30.0)),
           "--score-h", f"{score_h:.3f}", "--json", js, "--label", os.path.basename(d)]
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.isfile(js):
        print(r2.stdout[-2000:], r2.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"FATAL: the gate wrote no JSON for {job}")
    out = json.load(open(js))
    out["_sim_h"] = sim_h
    out["_score_h"] = score_h
    out["_n_dumps"] = len(sel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference job dir (manifest.json + output/)")
    ap.add_argument("--b", required=True, help="the new job dir")
    ap.add_argument("--step", type=int, default=0,
                    help="absolute step to score both at; 0 = the largest both have")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--json", default="results/seed_compare.json")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    if a.step == 0:
        ka = dumps(os.path.join(a.a, "output"),
                   json.load(open(os.path.join(a.a, "manifest.json")))["run"]["outFileBase"])
        kb = dumps(os.path.join(a.b, "output"),
                   json.load(open(os.path.join(a.b, "manifest.json")))["run"]["outFileBase"])
        common = sorted(set(ka) & set(kb))
        if not common:
            raise SystemExit("FATAL: the two runs share no dump step")
        a.step = common[-1]
        print(f"  --step not given; the largest step both runs have is {a.step}")

    tmp = tempfile.mkdtemp(prefix="seedcmp_", dir=os.path.join(root, "results"))
    try:
        A = score(a.a, a.step, tmp, sys.executable, "A")
        B = score(a.b, a.step, tmp, sys.executable, "B")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    fa, fb = A["final"], B["final"]
    # AN EXACT ZERO ON EVERY FIELD IS NOT AGREEMENT, IT IS THE SAME DATA TWICE.
    # FastEddy is not bitwise reproducible run to run, so two runs of one seed differing by
    # exactly 0.0 in u*, U, sigma_w, z_i AND the direction is impossible physics and a
    # certain plumbing fault. This script had that fault; the assertion is what stops the
    # next version of it from being read as a result.
    keys = [k for k in KEYS if k in fa and k in fb]
    if keys and all(float(fa[k]) == float(fb[k]) for k in keys):
        raise SystemExit(
            "FATAL: every field is EXACTLY equal. Two independent turbulence realisations\n"
            "       cannot do that -- the two sides are scoring the same dumps. Check that\n"
            "       --a and --b are different directories and that both actually hold their\n"
            "       own output/.")
    print(f"\n=== the same seed, two runs, scored at step {a.step} "
          f"= {A['_sim_h']:.3f} simulated hours ===")
    print(f"  A  {a.label_a:34s} {A['_n_dumps']:2d} dumps, gate window {A['_score_h']:.3f} h")
    print(f"  B  {a.label_b:34s} {B['_n_dumps']:2d} dumps, gate window {B['_score_h']:.3f} h")
    print("\n  THESE ARE TWO TURBULENCE REALISATIONS, NOT TWO EVALUATIONS OF ONE. FastEddy is")
    print("  not bitwise reproducible run to run on one GPU, so a difference here is a")
    print("  number to READ, not a tolerance to pass. What matters is whether the two land")
    print("  in the same place in the coordinates bin/pick_seed.py matches cases on.")
    print(f"\n  {'quantity':14s} {'A':>12s} {'B':>12s} {'B-A':>11s} {'rel':>8s}  unit")
    rows = {}
    for k in KEYS:
        if k not in fa or k not in fb:
            continue
        va, vb = float(fa[k]), float(fb[k])
        d = vb - va
        rel = d / va if va else float("nan")
        rows[k] = {"a": va, "b": vb, "diff": d, "rel": rel}
        print(f"  {k:14s} {va:12.4f} {vb:12.4f} {d:+11.4f} {rel:+8.2%}  {UNITS.get(k, '')}")

    # THE DIMENSIONLESS RATIOS ARE THE ONES THAT SHOULD AGREE, and they are the reason the
    # gate is written around them: numerator and denominator ride the 17.6 h inertial
    # oscillation together, so u* can move 18% while U/u* moves 0.6% (PROJECT_BRIEF.md).
    print(f"\n  {'ratio':14s} {'A':>12s} {'B':>12s} {'rel':>8s}   (these are the "
          f"oscillation-immune ones)")
    ratios = {}
    for name, num, den in (("U/u*", "U", "ustar"), ("sigma_v/u*", "sigma_v", "ustar"),
                           ("sigma_w/u*", "sigma_w", "ustar")):
        try:
            ra, rb = fa[num] / fa[den], fb[num] / fb[den]
        except (KeyError, ZeroDivisionError):
            continue
        ratios[name] = {"a": ra, "b": rb, "rel": (rb - ra) / ra}
        print(f"  {name:14s} {ra:12.4f} {rb:12.4f} {(rb - ra) / ra:+8.2%}")

    # THE LENGTH SCALES IN UNITS OF THE RASTER CELL THEY WILL BE USED AT.
    # x_peak and x90 are Kljun geometry terms evaluated from the seed's achieved state, and
    # bin/pick_seed.py matches cases on them -- but a footprint is binned on the LES grid,
    # so a difference smaller than a cell cannot appear in anything downstream. Quoting the
    # percentage alone makes a 3.7 m difference look like a 2.4% disagreement when it is an
    # eighth of one cell.
    dx = json.load(open(os.path.join(a.a, "manifest.json"))).get("grid", {}).get("dx", 30.0)
    print(f"\n  the two length scales in units of one {dx:.0f} m raster cell:")
    for k in ("x_peak", "x90"):
        if k in rows:
            print(f"  {k:14s} |B-A| = {abs(rows[k]['diff']) / dx:.2f} cells")

    print(f"\n  gate verdict   A: {'PASS' if A['pass'] else 'not PASS'}"
          f"   drifting {A.get('drifting')}   indeterminate {len(A.get('indeterminate', []))}/7")
    print(f"                 B: {'PASS' if B['pass'] else 'not PASS'}"
          f"   drifting {B.get('drifting')}   indeterminate {len(B.get('indeterminate', []))}/7")

    os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
    json.dump({"step": a.step, "sim_h": A["_sim_h"], "score_h": A["_score_h"],
               "a": {"job": a.a, "label": a.label_a, "final": fa,
                     "pass": A["pass"], "drifting": A.get("drifting"),
                     "indeterminate": A.get("indeterminate")},
               "b": {"job": a.b, "label": a.label_b, "final": fb,
                     "pass": B["pass"], "drifting": B.get("drifting"),
                     "indeterminate": B.get("indeterminate")},
               "fields": rows, "ratios": ratios},
              open(a.json, "w"), indent=1)
    print(f"  -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
