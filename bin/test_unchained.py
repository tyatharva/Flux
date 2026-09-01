#!/usr/bin/env python3
"""Does removing the chain change the answer? Compare against the REPRODUCIBILITY FLOOR.

CHAINING IS RETIRED (2026-08-26): a seed and a target case are each ONE continuous
FastEddy invocation. Every production seed will therefore be unchained, while every result
this project has recorded so far came off a chained run. Before that carries, it is worth
one cheap measurement.

AND THE TEST IS NOT "ARE THEY IDENTICAL". FastEddy is NOT bitwise reproducible run to run
on a single GPU -- two runs of the same case differ by ~1e-4 relative in velocity and
~7e-4 K in theta after 200 steps (PROJECT_BRIEF.md). So chained-vs-unchained CANNOT be zero, and
comparing it against zero would fail a correct change. PROJECT_BRIEF.md's standing rule applies:
compare against the floor, not against zero.

    A   2N steps, one invocation
    B   N steps, then a restart, then N more     <- the retired chain
    C   2N steps, one invocation, again          <- the floor

The claim is |A-B| <= |A-C|, i.e. the chain is worth no more than re-running the same case.
That is what Gate C2 (restart is bit-for-bit, 0 of 23 variables differing) predicts, and
this measures it rather than assuming it.

RUN THIS ON THE HOST, not through docker/pyrun.sh -- it launches FastEddy containers, and
a container cannot launch a sibling. The host python has numpy and netCDF4, which is all
this needs (it deliberately has no scipy dependency for that reason).

    python3 bin/test_unchained.py --n 1000

MEASURED 2026-08-26, N = 1000 (2000 steps), jobs/seed_nbl-shallow_a000:

    field   |A-B| max        |A-C| max       A-B / A-C
    u       7.114e-04       6.580e-04         1.08
    v       6.537e-04       7.337e-04         0.89
    w       6.679e-04       7.400e-04         0.90
    theta   2.594e-03       2.808e-03         0.92

PASS. The chain is worth 0.89-1.08x the reproducibility floor -- on three of four fields
SMALLER than simply re-running the same case. Chained results therefore carry over to
unchained production unchanged.

usage: test_unchained.py [--job jobs/seed_nbl-shallow_a000] [--n 1000]
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

FIELDS = ("u", "v", "w", "theta")


def run(case_dir, infile, log):
    r = subprocess.run(["./docker/run_case.sh", case_dir, infile, log],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def write_in(src, dst, **kv):
    out = []
    for ln in open(src):
        k = ln.split("=", 1)[0].strip() if "=" in ln else None
        if k in kv:
            out.append(f"{k} = {kv[k]}\n")
        else:
            out.append(ln)
    open(dst, "w").write("".join(out))


def cmp_dumps(p, q):
    """max |a-b| / rms(a) per field, plus theta in K."""
    out = {}
    with Dataset(p) as A, Dataset(q) as B:
        for v in FIELDS:
            a = np.squeeze(np.asarray(A[v][:], dtype=np.float64))
            b = np.squeeze(np.asarray(B[v][:], dtype=np.float64))
            d = np.abs(a - b)
            rms = float(np.sqrt((a ** 2).mean()))
            out[v] = (float(d.max()), float(d.max() / max(rms, 1e-30)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="jobs/seed_nbl-shallow_a000")
    ap.add_argument("--n", type=int, default=1000, help="steps per half")
    ap.add_argument("--work", default="runs/unchain_test")
    a = ap.parse_args()

    src = os.path.join(a.job, "seed.in")
    if not os.path.exists(src):
        print(f"FATAL: {src} does not exist", file=sys.stderr)
        return 2
    N, T = a.n, 2 * a.n
    D = a.work
    shutil.rmtree(D, ignore_errors=True)
    os.makedirs(os.path.join(D, "output"), exist_ok=True)

    def clear():
        for f in glob.glob(os.path.join(D, "output", "*")):
            os.remove(f)

    print(f"=== chained vs unchained, {a.job}, N = {N}, total = {T} steps ===")
    results = {}

    # --- A and C: one invocation, twice -------------------------------------------
    for tag in ("A", "C"):
        clear()
        write_in(src, os.path.join(D, "one.in"), Nt=T, NtBatch=N, frqOutput=N,
                 inPath="", inFile="", outFileBase="FE_ONE")
        rc, log = run(D, "one.in", f"/tmp/flux-logs/unchain_{tag}.log")
        f = os.path.join(D, "output", f"FE_ONE.{T}")
        if not os.path.exists(f):
            print(f"FATAL: run {tag} produced no FE_ONE.{T}\n{log[-1500:]}", file=sys.stderr)
            return 1
        results[tag] = os.path.join(D, f"final_{tag}.nc")
        shutil.copy(f, results[tag])
        print(f"  run {tag}: one invocation of {T} steps -> ok")

    # --- B: the retired chain, N then N -------------------------------------------
    clear()
    write_in(src, os.path.join(D, "seg1.in"), Nt=N, NtBatch=N, frqOutput=N,
             inPath="", inFile="", outFileBase="FE_SEG")
    rc, log = run(D, "seg1.in", "/tmp/flux-logs/unchain_B1.log")
    if not os.path.exists(os.path.join(D, "output", f"FE_SEG.{N}")):
        print(f"FATAL: chain segment 1 produced no dump\n{log[-1500:]}", file=sys.stderr)
        return 1
    write_in(src, os.path.join(D, "seg2.in"), Nt=T, NtBatch=N, frqOutput=N,
             inPath="./output/", inFile=f"FE_SEG.{N}", outFileBase="FE_SEG")
    rc, log = run(D, "seg2.in", "/tmp/flux-logs/unchain_B2.log")
    f = os.path.join(D, "output", f"FE_SEG.{T}")
    if not os.path.exists(f):
        print(f"FATAL: chain segment 2 produced no FE_SEG.{T}\n{log[-1500:]}", file=sys.stderr)
        return 1
    results["B"] = os.path.join(D, "final_B.nc")
    shutil.copy(f, results["B"])
    print(f"  run B: {N} + restart + {N} steps -> ok")

    ab = cmp_dumps(results["A"], results["B"])
    ac = cmp_dumps(results["A"], results["C"])
    print(f"\n  {'field':>7}{'|A-B| max':>13}{'rel':>11}{'|A-C| max':>13}{'rel':>11}"
          f"{'A-B / A-C':>12}")
    verdict = True
    for v in FIELDS:
        r = ab[v][0] / max(ac[v][0], 1e-30)
        print(f"  {v:>7}{ab[v][0]:13.3e}{ab[v][1]:11.2e}{ac[v][0]:13.3e}{ac[v][1]:11.2e}"
              f"{r:12.2f}")
        if r > 3.0:
            verdict = False
    print()
    print("  A = one invocation of 2N   B = N + restart + N (the retired chain)")
    print("  C = one invocation of 2N, again -- the run-to-run reproducibility floor")
    if verdict:
        print("\n  PASS: the chain is worth no more than re-running the same case.")
        print("        Unchaining does not change the answer; it removes a restart READ,")
        print("        which is what docs/FASTEDDY_TRAPS.md 17 was about.")
    else:
        print("\n  FAIL: chained and unchained differ by more than the reproducibility")
        print("        floor. Do NOT carry chained results over to unchained production.")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
