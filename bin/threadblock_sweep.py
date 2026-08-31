#!/usr/bin/env python3
"""Measure the fastest CUDA thread-block shape ON THIS GPU, before the library is spun.

WHY THIS EXISTS AND WHY IT RUNS AUTOMATICALLY. Every `.in` in this project carries
tBx=1, tBy=2, tBz=64, and PROJECT_BRIEF.md records where that came from: a sweep of nine legal
shapes at 122^3, 300 steps each, ON AN RTX 4080 (Ada, sm_89), 2026-08-22. 1x2x64 won at
0.01475 s/step, 1x2x32 and 1x8x16 were within 1%, 1x16x16 was 12% slower and 1x32x8 was
46% slower. NOTHING ABOUT THAT IS A PROPERTY OF THE GRID. It is a property of the memory
system of one architecture, and the deployable target is Blackwell (sm_120), which has a
different L2, a different shared-memory/L1 split and a different SM count. Carrying the
Ada winner onto it would be the same mistake this project already made with the accuracy
CFL boundary -- "it is a property of CFL_3d, not of the spacing" -- which turned out to be
a property of the grid and had to be re-measured on every one of them.

So the number is re-measured, on the machine, before any seed runs. It costs ~2 minutes
once and it prices ~30 GPU-hours.

THE LEGAL SET IS NOT A MATTER OF TASTE. SRC/GRID/grid.c:222-240 refuses the run outright
unless (Nx+2Nh) % tBx == (Ny+2Nh) % tBy == (Nz+2Nh) % tBz == 0, on PER-RANK halo-inclusive
extents. At 122^3 with Nh=3 all three are 128 = 2^7, so tBx, tBy and tBz must each be a
power of two. CUDA caps blockDim.z at 64 and a block at 1024 threads. Shapes outside that
are not slow, they are a CRITICAL ERROR before the first timestep.

tBx > 1 IS SWEPT ANYWAY, and that is deliberate. kStride = 1 while iStride = (Ny+6)(Nz+6),
so with tBx > 1 adjacent threads in a warp read addresses iStride floats apart and one
128-byte transaction becomes four 32-byte ones -- measured at 17% on Ada. That is an
argument about coalescing, which is architectural, so it is exactly the kind of claim that
should be re-tested rather than assumed on a new architecture. A couple of tBx > 1 shapes
are included so the answer is measured rather than inherited.

WHAT IS TIMED IS `Comp./step`, NOT `Time/step`. FastEddy prints both in its TIMESTEP
PERFORMANCE block; the difference is the netCDF write, which depends on the filesystem and
not on the block shape. Timing the wrong column would rank shapes by disk speed.

THE BLOCK SHAPE IS A PURE PERFORMANCE KNOB AND CANNOT MOVE THE PHYSICS. Every main kernel
computes its own cell with no cross-thread accumulation, and the one reduction that DOES
accumulate -- cuda_singleRankHorizSlabMeans, whose atomicAdd is the source of this model's
run-to-run nondeterminism -- is templated on tBx_red/tBy_red/tBz_red, compile-time
constants 2/8/1 in fecuda_PlugIns_cu.h. They do not follow tBx/tBy/tBz. So changing the
block shape changes how fast the answer arrives and not what it is.

AND THE WINNER IS NOT PICKED ON ONE MEASUREMENT. The first version of this script ran each
shape once and took the fastest, which on the development GPU meant choosing 2x2x32 over
1x2x64 on 0.7% -- a difference smaller than the repeat noise of a single 200-step run, and
exactly the failure PROJECT_BRIEF.md names: "A TOLERANCE MEASURED FROM ONE DIFFERENCE IS NOT A
TOLERANCE." The sweep is now two phases: SCREEN every candidate once, then CONFIRM everything
within a few percent of the best by repeating it, rank on the MEDIAN, and keep the incumbent
1x2x64 unless the challenger beats it by more than the measured noise.

    python3 bin/threadblock_sweep.py --template jobs30/seed_nbl-deep_a015/seed.in \\
        --steps 200 --json results/threadblock_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

PERF = re.compile(
    r"^\s*([0-9.]+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)", re.M)


def legal_shapes(nx, ny, nz, nh, max_x=4):
    """Every (tBx,tBy,tBz) FastEddy's own decomposition check accepts, plus CUDA's limits."""
    ex, ey, ez = nx + 2 * nh, ny + 2 * nh, nz + 2 * nh
    out = []
    for bx in [b for b in (1, 2, 4, 8) if b <= max_x and ex % b == 0]:
        for by in [2 ** k for k in range(0, 8)]:
            if ey % by:
                continue
            for bz in [2 ** k for k in range(0, 7)]:      # blockDim.z <= 64
                if ez % bz or bz > 64:
                    continue
                n = bx * by * bz
                if n < 32 or n > 1024:                    # a warp; CUDA's block cap
                    continue
                out.append((bx, by, bz))
    return out


def rewrite(src, dst, **kv):
    seen = set()
    with open(dst, "w") as f:
        for ln in open(src):
            k = ln.split("=", 1)[0].strip() if "=" in ln else None
            if k in kv:
                f.write(f"{k} = {kv[k]}\n")
                seen.add(k)
            else:
                f.write(ln)
    missing = set(kv) - seen
    if missing:
        raise SystemExit(f"FATAL: {src} has no line for {sorted(missing)}")


def run_once(root, work, fe_bin, template, shape, steps, gpu, timeout):
    bx, by, bz = shape
    d = os.path.join(work, f"tb_{bx}x{by}x{bz}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "output"))
    # ONE BATCH, so exactly one TIMESTEP PERFORMANCE block is printed and there is nothing
    # to average or to pick between. frqOutput = steps writes at 0 and at the end; the IO
    # is excluded from the column being read anyway.
    rewrite(template, os.path.join(d, "tb.in"),
            tBx=bx, tBy=by, tBz=bz, Nt=steps, NtBatch=steps, frqOutput=steps,
            inPath="", inFile="", outPath="./output/", outFileBase="FE_TB")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
               OMPI_MCA_plm="isolated")
    t0 = time.time()
    try:
        r = subprocess.run(["mpirun", "--bind-to", "none", "-np", "1", fe_bin, "./tb.in"],
                           cwd=d, capture_output=True, text=True, env=env, timeout=timeout)
        txt = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        txt, r = "TIMEOUT", None
    wall = time.time() - t0
    with open(os.path.join(d, "tb.log"), "w") as f:
        f.write(txt)

    # ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS. An illegal shape prints
    # "CRITICAL ERROR: Nz+2*Nh is not an exact multiple of tBz" and a launch that exceeds
    # the register budget prints "too many resources requested for launch" -- and this
    # project has FastEddy exiting 0 on fully-NaN fields, so the log is the evidence.
    for bad in ("CRITICAL ERROR", "too many resources", "CORRUPTED", "#NaN", "#Inf"):
        if bad in txt:
            return {"shape": f"{bx}x{by}x{bz}", "ok": False, "why": bad, "wall_s": wall}
    # THE LAST PERFORMANCE BLOCK IS NOT THE RUN'S. FastEddy prints one TIMESTEP
    # PERFORMANCE block per batch and then a FINAL one for the shutdown, with
    # `Batch Steps = 0`:
    #
    #        3.0472  |    200  |  0.0152  |  0.0144   |  0.165962     <- the run
    #        0.1651  |      0  |  0.0008  |  0.0002   |  0.130430     <- the tail
    #
    # Taking m[-1] read the tail. It is finite, it is plausible, and it is the same
    # ~0.0002 for EVERY shape -- so the sweep ranked fourteen block shapes on shutdown
    # noise and reported a spread of 1.00x. Nothing failed; a number came out.
    # Select on the artifact instead: the block whose Batch Steps is the count asked for.
    m = [r for r in PERF.findall(txt) if int(r[1]) == steps]
    if not m:
        got = sorted({int(r[1]) for r in PERF.findall(txt)})
        return {"shape": f"{bx}x{by}x{bz}", "ok": False,
                "why": f"no TIMESTEP PERFORMANCE block for {steps} steps (saw batches {got})",
                "wall_s": wall}
    total, nsteps, per_step, comp_step, io_s = m[-1]
    return {"shape": f"{bx}x{by}x{bz}", "tBx": bx, "tBy": by, "tBz": bz, "ok": True,
            "steps": int(nsteps), "s_per_step": float(per_step),
            "comp_per_step": float(comp_step), "io_s": float(io_s),
            "total_s": float(total), "wall_s": wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="a .in carrying the production grid")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--fe-bin",
                    default="/flux/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy")
    ap.add_argument("--work", default="runs/tb_sweep")
    ap.add_argument("--json", default="results/threadblock_sweep.json")
    ap.add_argument("--max-shapes", type=int, default=14,
                    help="cap the sweep; shapes are ordered by an Ada-informed prior so "
                         "the cap drops the least likely first")
    ap.add_argument("--repeats", type=int, default=2,
                    help="extra runs of each CONFIRM-phase shape (total = 1 + repeats)")
    ap.add_argument("--confirm-band", type=float, default=0.05,
                    help="screen-phase shapes within this fraction of the best are confirmed")
    ap.add_argument("--incumbent", default="1x2x64",
                    help="the shape the .in files already carry; kept unless beaten by "
                         "more than the measured repeat noise")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--keep", action="store_true", help="keep the scratch runs")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if not os.path.isfile(a.template):
        raise SystemExit(f"FATAL: no template {a.template}")
    if not os.access(a.fe_bin, os.X_OK):
        raise SystemExit(f"FATAL: no FastEddy binary at {a.fe_bin}")

    cfg = {}
    for ln in open(a.template):
        if "=" in ln:
            k, v = ln.split("=", 1)
            cfg[k.strip()] = v.split("#")[0].strip()
    nx, ny, nz, nh = (int(cfg[k]) for k in ("Nx", "Ny", "Nz", "Nh"))
    shapes = legal_shapes(nx, ny, nz, nh)

    # ORDER THE SWEEP, BECAUSE THE CAP HAS TO DROP SOMETHING. The prior is the Ada
    # measurement: tBx=1 first (coalescing), then blocks of 128-256 threads, then the rest.
    # A couple of tBx>1 shapes are pulled forward deliberately so the coalescing claim is
    # re-tested rather than inherited -- it is the one result here that is architectural.
    def prior(s):
        bx, by, bz = s
        n = bx * by * bz
        return (0 if bx == 1 else 1, abs(n - 128), -bz)
    shapes.sort(key=prior)
    forced = [s for s in shapes if s[0] > 1][:2]
    head = [s for s in shapes if s[0] == 1][:max(1, a.max_shapes - len(forced))]
    todo = head + [s for s in forced if s not in head]

    print(f"=== thread-block sweep, {nx}x{ny}x{nz} (Nh={nh}), {a.steps} steps, GPU {a.gpu} ===")
    print(f"  {len(shapes)} legal shapes; measuring {len(todo)}"
          f"{'' if len(todo) == len(shapes) else f' (--max-shapes {a.max_shapes})'}")
    if len(todo) < len(shapes):
        # NO SILENT CAPS. Say which shapes were not measured, so a later reader knows the
        # winner is the best of what was tried and not the best that exists.
        skipped = [f"{b[0]}x{b[1]}x{b[2]}" for b in shapes if b not in todo]
        print(f"  NOT measured ({len(skipped)}): {' '.join(skipped)}")

    def median(v):
        v = sorted(v)
        return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])

    rows, times = [], {}
    print("  --- phase 1: screen, one run each ---")
    for sh in todo:
        r = run_once(root, a.work, a.fe_bin, a.template, sh, a.steps, a.gpu, a.timeout)
        rows.append(r)
        if r["ok"]:
            times.setdefault(r["shape"], []).append(r["comp_per_step"])
            print(f"  {r['shape']:>10s}  comp {r['comp_per_step']:.5f} s/step"
                  f"   total {r['s_per_step']:.5f}   wall {r['wall_s']:.1f} s")
        else:
            print(f"  {r['shape']:>10s}  REJECTED: {r['why']}")
    if not times:
        raise SystemExit("FATAL: no shape produced a timing; refusing to guess a winner")

    # PHASE 2: repeat everything close enough to the leader that one run cannot separate
    # them, plus the incumbent whether or not it made the band -- the comparison that
    # decides whether to CHANGE anything is against the incumbent, so it needs the same
    # number of measurements as its challengers.
    best1 = min(v[0] for v in times.values())
    shape_of = {f"{b[0]}x{b[1]}x{b[2]}": b for b in todo}
    confirm = [k for k, v in times.items() if v[0] <= best1 * (1 + a.confirm_band)]
    # MEMBERSHIP IS IN `times`, NOT IN `shape_of`. shape_of holds every shape that was
    # ATTEMPTED; times holds only those that produced a timing. If the incumbent is
    # rejected in phase 1 -- 'too many resources requested for launch' under a different
    # architecture's register budget, a timeout, a transient CRITICAL ERROR -- then
    # appending it here makes phase 2 raise KeyError, no JSON is written, and
    # bin/run_seeds.py falls back to the .in's own shape with only a WARNING. The seed
    # library would then be spun on an unmeasured block shape after a sweep that crashed.
    if a.incumbent in times and a.incumbent not in confirm:
        confirm.append(a.incumbent)
    elif a.incumbent in shape_of and a.incumbent not in times:
        print(f"  NOTE: the incumbent {a.incumbent} produced no timing in phase 1, so it "
              f"cannot be the comparison. The winner will be taken on its own merits.")
    if a.repeats > 0 and confirm:
        print(f"  --- phase 2: confirm {len(confirm)} shape(s) within "
              f"{a.confirm_band:.0%} of the leader, {a.repeats} more run(s) each ---")
        for k in confirm:
            if k not in times:          # belt and braces for the same failure
                continue
            for _ in range(a.repeats):
                r = run_once(root, a.work, a.fe_bin, a.template, shape_of[k], a.steps,
                             a.gpu, a.timeout)
                rows.append(r)
                if r["ok"]:
                    times[k].append(r["comp_per_step"])
            v = times[k]
            print(f"  {k:>10s}  median {median(v):.5f}  n={len(v)}  "
                  f"[{min(v):.5f}, {max(v):.5f}]  spread {(max(v) - min(v)) / median(v):.2%}")

    stats = {k: {"n": len(v), "median": median(v), "min": min(v), "max": max(v),
                 "spread": (max(v) - min(v)) / median(v) if len(v) > 1 else None}
             for k, v in times.items()}
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["median"])
    win_shape, win_st = ranked[0]
    slow_shape, slow_st = ranked[-1]

    # THE MEASUREMENT'S OWN NOISE, from the repeats, pooled over every confirmed shape.
    spreads = [st["spread"] for st in stats.values() if st["spread"] is not None]
    noise = median(spreads) if spreads else 0.0
    inc_st = stats.get(a.incumbent)
    kept_incumbent = False
    if inc_st and win_shape != a.incumbent:
        gain = (inc_st["median"] - win_st["median"]) / win_st["median"]
        if gain <= noise:
            # A DIFFERENCE SMALLER THAN THE REPEAT NOISE IS NOT A DIFFERENCE. Changing the
            # library's block shape on it would be reporting measurement scatter as a
            # result -- and every .in in the project, and PROJECT_BRIEF.md, carry the incumbent.
            print(f"\n  {win_shape} is {gain:.2%} faster than the incumbent "
                  f"{a.incumbent}, which is inside the {noise:.2%} repeat noise of this "
                  f"measurement. KEEPING {a.incumbent}.")
            win_shape, win_st, kept_incumbent = a.incumbent, inc_st, True
    bx, by, bz = shape_of[win_shape]
    win = {"tBx": bx, "tBy": by, "tBz": bz, "shape": win_shape,
           "comp_per_step": win_st["median"], "n_runs": win_st["n"],
           "kept_incumbent": kept_incumbent}

    print(f"\n  WINNER {win_shape} at {win_st['median']:.5f} s/step compute "
          f"(median of {win_st['n']})")
    print(f"  spread over the measured set: {slow_st['median'] / win_st['median']:.2f}x"
          f"  ({slow_shape} is slowest)"
          f"   |  repeat noise {noise:.2%}")
    ref = stats.get("1x2x64")
    if ref:
        # THE ADA WINNER IS QUOTED WHETHER OR NOT IT WON, because "we re-measured and got
        # the same answer" is a result and needs its number.
        print(f"  the Ada winner 1x2x64 is {ref['median'] / win_st['median']:.3f}x the "
              f"winner here" + ("  (unchanged)" if win_shape == "1x2x64" else ""))
    xs = [(k, st) for k, st in stats.items() if shape_of[k][0] > 1]
    if xs:
        # THE COALESCING CLAIM, RE-TESTED RATHER THAN INHERITED. PROJECT_BRIEF.md records tBx > 1
        # costing 17% at 186^2 on Ada. Whatever this machine says, it is a measurement.
        b = min(xs, key=lambda kv: kv[1]["median"])
        print(f"  best tBx>1 shape here: {b[0]} at {b[1]['median'] / win_st['median']:.3f}x "
              f"the winner (PROJECT_BRIEF.md records tBx>1 costing ~17% at 186^2 on Ada)")

    os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
    json.dump({"grid": [nx, ny, nz], "Nh": nh, "steps": a.steps, "gpu": a.gpu,
               "template": a.template, "n_legal": len(shapes),
               "not_measured": [f"{b[0]}x{b[1]}x{b[2]}" for b in shapes if b not in todo],
               "rows": rows, "stats": stats, "repeat_noise": noise,
               "confirmed": confirm, "incumbent": a.incumbent,
               "winner": win,
               "ada_reference_1x2x64": (ref or {}).get("median")},
              open(a.json, "w"), indent=1)
    print(f"  -> {a.json}")
    if not a.keep:
        shutil.rmtree(a.work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
