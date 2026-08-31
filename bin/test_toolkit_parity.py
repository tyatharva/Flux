#!/usr/bin/env python3
"""Did the CUDA toolkit upgrade move the physics? Compare against the REPRODUCIBILITY FLOOR.

THE PIN MOVED (2026-08-31). Every result this project has published came out of CUDA 11.8.
The RTX 5090 is sm_120, which nvcc 11.8 cannot target -- not with SASS and not even with
PTX, because 11.8's newest virtual architecture is compute_90 and a compiler cannot emit
code for an architecture that did not exist when it was written. So the deployable image
is built on a newer toolkit, and before any seed produced by it is allowed into the
library, the toolkit change has to be shown not to have moved the answer.

AND THE TEST IS NOT "ARE THEY IDENTICAL". FastEddy is NOT bitwise reproducible run to run
on ONE GPU with ONE binary -- two runs of the same case differ by ~1e-4 relative in
velocity and ~7e-4 K in theta after 200 steps (PROJECT_BRIEF.md). The source of that is on the
seed path by construction: lsf_horMnSubTerms=1 computes slab means through
cuda_singleRankHorizSlabMeans, whose atomicAdd accumulates in whatever order the blocks
retire. So old-vs-new CANNOT be zero, and comparing it against zero would fail a correct
change. PROJECT_BRIEF.md's standing rule: compare against the floor, not against zero.

    A   N steps, OLD toolkit, run 1
    C   N steps, OLD toolkit, run 2      <- the floor: the same binary against itself
    B   N steps, NEW toolkit, run 1

The claim is |A-B| <= ~|A-C|: the toolkit is worth no more than re-running the same case.

THE INITIAL CONDITION IS BIT-IDENTICAL ACROSS ALL THREE, AND THAT IS WHY THIS WORKS.
hydro_core.c:1881 draws the initial theta perturbation with rand(), seeded at
FastEddy.c:113 by srand(mpi_rank_world + 12345) -- a FIXED seed at one rank -- so every
cold start of a given case gets the same perturbation field. The sequence glibc returns for
a given seed is a libc property, and both images are Ubuntu 22.04, which is the reason the
distro is held fixed across the upgrade rather than moved to 24.04 at the same time. If the
distro moved, this test would be measuring rand().

RUN THIS ON THE HOST. It launches containers, and a container cannot launch a sibling.
The host python has numpy and netCDF4; this deliberately needs no scipy.

    python3 bin/test_toolkit_parity.py \
        --job jobs30/seed_nbl-deep_a015 --n 200 \
        --old-image flux-fasteddy:cuda118 --old-tree /path/to/an/11.8-built/FastEddy-tree \
        --new-image flux-seeds:latest

A tree given with --{old,new}-tree is bind-mounted at /fe and its binary taken from
/fe/SRC/FEMAIN/FastEddy. With no tree, the image's OWN baked-in binary is used
(/flux/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy), which is the point of the new image.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

FIELDS = ("u", "v", "w", "theta")
BAKED_BIN = "/flux/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy"


def write_in(src, dst, **kv):
    """Rewrite the named keys of a FastEddy .in, leaving every other line byte-identical."""
    out = []
    for ln in open(src):
        k = ln.split("=", 1)[0].strip() if "=" in ln else None
        if k in kv:
            out.append(f"{k} = {kv[k]}\n")
        else:
            out.append(ln)
    open(dst, "w").write("".join(out))


def run_fe(image, tree, case_host_dir, infile, log_path, gpu="0"):
    """One FastEddy invocation. Returns (rc, combined output).

    CUDA_VISIBLE_DEVICES is passed explicitly: this test is meaningful only if all three
    runs land on the SAME physical GPU. Two different cards would add a second difference
    to the one being measured.
    """
    cmd = ["docker", "run", "--rm", "--gpus", "all",
           "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
           "-e", "OMPI_MCA_plm=isolated",
           "-e", "HOME=/tmp",
           "--user", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{case_host_dir}:/case"]
    fe = BAKED_BIN
    if tree:
        cmd += ["-v", f"{os.path.abspath(tree)}:/fe"]
        fe = "/fe/SRC/FEMAIN/FastEddy"
    cmd += ["-w", "/case", "--entrypoint", "mpirun", image, "-np", "1", fe, f"./{infile}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    with open(log_path, "w") as f:
        f.write(" ".join(cmd) + "\n\n" + txt)
    return r.returncode, txt


def newest_dump(d, base):
    """The dump this run actually produced -- asserted on, never assumed."""
    files = glob.glob(os.path.join(d, f"{base}.[0-9]*"))
    if not files:
        return None
    return sorted(files, key=lambda p: int(p.rsplit(".", 1)[1]))[-1]


def cmp_dumps(p, q):
    """max|a-b| and max|a-b|/rms(a), per field, in float64."""
    out = {}
    with Dataset(p) as A, Dataset(q) as B:
        for v in FIELDS:
            a = np.squeeze(np.asarray(A[v][:], dtype=np.float64))
            b = np.squeeze(np.asarray(B[v][:], dtype=np.float64))
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise SystemExit(f"FATAL: non-finite values in {v} ({p} or {q})")
            d = np.abs(a - b)
            rms = float(np.sqrt((a ** 2).mean()))
            out[v] = {"absmax": float(d.max()),
                      "relmax": float(d.max() / max(rms, 1e-30)),
                      "rms_field": rms}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="jobs30/seed_nbl-deep_a015",
                    help="job directory whose seed.in supplies the configuration")
    ap.add_argument("--n", type=int, default=200, help="steps per run")
    ap.add_argument("--old-image", default="flux-fasteddy:cuda118")
    ap.add_argument("--old-tree", default="",
                    help="host FastEddy tree to mount at /fe; empty = the image's baked binary")
    ap.add_argument("--new-image", required=True)
    ap.add_argument("--new-tree", default="")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--work", default="results/toolkit_parity")
    ap.add_argument("--json", default="results/toolkit_parity.json")
    ap.add_argument("--tol", type=float, default=3.0,
                    help="PASS if every field's |A-B| is within this multiple of |A-C|")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    seed_in = os.path.join(a.job, "seed.in")
    if not os.path.isfile(seed_in):
        raise SystemExit(f"FATAL: no {seed_in}")

    work = os.path.abspath(a.work)
    shutil.rmtree(work, ignore_errors=True)

    # ONE RUN PER DIRECTORY, OR IT IS NOT A SERIES (FASTEDDY_TRAPS.md 18c). Three runs
    # writing one output/ would produce three families with overlapping step numbers.
    runs = [("A", a.old_image, a.old_tree), ("C", a.old_image, a.old_tree),
            ("B", a.new_image, a.new_tree)]
    base = "FE_PAR"
    results = {}
    for tag, image, tree in runs:
        d = os.path.join(work, tag)
        os.makedirs(os.path.join(d, "output"), exist_ok=True)
        # A COLD START: inPath/inFile empty, so no IO-registered field can be inherited
        # from a restart (FASTEDDY_TRAPS.md 17) and the .in is the whole configuration.
        write_in(seed_in, os.path.join(d, "run.in"),
                 Nt=a.n, NtBatch=a.n, frqOutput=a.n,
                 inPath="", inFile="", outFileBase=base, outPath="./output/")
        print(f"  --- run {tag}: {image}"
              f"{' (tree ' + tree + ')' if tree else ' (baked binary)'}, {a.n} steps")
        rc, txt = run_fe(image, tree, d, "run.in", os.path.join(d, "run.log"), a.gpu)
        # ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS (FASTEDDY_TRAPS.md 12): FastEddy
        # exits 0 on fully-NaN fields, and a missing restart makes it run to completion
        # writing only NaN.
        for pat in ("CORRUPTED", "#NaN", "#Inf"):
            if pat in txt:
                raise SystemExit(f"FATAL: run {tag} reported {pat}")
        dump = newest_dump(os.path.join(d, "output"), base)
        if dump is None:
            print(txt[-3000:], file=sys.stderr)
            raise SystemExit(f"FATAL: run {tag} wrote no dump (rc={rc})")
        step = int(dump.rsplit(".", 1)[1])
        if step != a.n:
            raise SystemExit(f"FATAL: run {tag} newest dump is step {step}, wanted {a.n}")
        results[tag] = dump
        print(f"      -> {os.path.relpath(dump, root)}")

    floor = cmp_dumps(results["A"], results["C"])   # old vs old   : the floor
    toolk = cmp_dumps(results["A"], results["B"])   # old vs new   : the question

    print(f"\n=== toolkit parity, {a.n} steps, {a.job} ===")
    print(f"  A, C : {a.old_image}" + (f"  tree {a.old_tree}" if a.old_tree else "  (baked)"))
    print(f"  B    : {a.new_image}" + (f"  tree {a.new_tree}" if a.new_tree else "  (baked)"))
    print()
    print(f"  {'field':6s} {'|A-B| max':>12s} {'|A-C| max':>12s} {'ratio':>8s}"
          f" {'|A-B| rel':>12s} {'|A-C| rel':>12s}")
    ok = True
    rows = {}
    for v in FIELDS:
        r = toolk[v]["absmax"] / max(floor[v]["absmax"], 1e-30)
        rows[v] = {"toolkit_absmax": toolk[v]["absmax"], "floor_absmax": floor[v]["absmax"],
                   "ratio": r, "toolkit_relmax": toolk[v]["relmax"],
                   "floor_relmax": floor[v]["relmax"]}
        if r > a.tol:
            ok = False
        print(f"  {v:6s} {toolk[v]['absmax']:12.4e} {floor[v]['absmax']:12.4e} {r:8.2f}"
              f" {toolk[v]['relmax']:12.4e} {floor[v]['relmax']:12.4e}")

    # THE FLOOR MUST BE NONZERO OR THE TEST ESTABLISHES NOTHING. If the two old-toolkit
    # runs came out bit-identical there is no floor to compare against, and a ratio
    # against zero is meaningless in both directions -- report it rather than dividing.
    degenerate = [v for v in FIELDS if floor[v]["absmax"] == 0.0]
    if degenerate:
        ok = False
        print(f"\n  *** the floor is EXACTLY ZERO for {degenerate}: the two same-binary runs")
        print("      were bit-identical, so this comparison has no scale and establishes")
        print("      NOTHING about the toolkit. Re-run with more steps.")

    verdict = "PASS" if ok else "FAIL"
    print(f"\n  VERDICT: {verdict} -- the toolkit change is worth "
          f"{min(rows[v]['ratio'] for v in FIELDS):.2f}-{max(rows[v]['ratio'] for v in FIELDS):.2f}x"
          f" the model's own run-to-run floor (tolerance {a.tol:.1f}x)")

    os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
    json.dump({"job": a.job, "steps": a.n, "gpu": a.gpu, "tol": a.tol,
               "old_image": a.old_image, "old_tree": a.old_tree,
               "new_image": a.new_image, "new_tree": a.new_tree,
               "fields": rows, "pass": ok},
              open(a.json, "w"), indent=1)
    print(f"  -> {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
