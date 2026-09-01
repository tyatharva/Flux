#!/usr/bin/env python3
"""Does streaming the in-process hand-off change the answer, and does it stop the buildup?

TWO CLAIMS, SCORED DIFFERENTLY.

**A. Identity -- ASSERTED AT EXACTLY ZERO.** The streamed route consumes each snapshot once,
feeds it to the window-statistics accumulator and the field cache inside that one pass, and
releases it. The batched route reads every snapshot into a list, builds the cache from the
list, and then reads every snapshot AGAIN for the statistics. Those are two schedules over
identical arithmetic, so the correct tolerance is zero -- there is no physics between them.
This is the same standard `bin/test_dumpsrc.py` and `bin/test_ringsrc.py` already hold the
ring to, and for the same reason: a tolerance here would be hiding a real difference.

**B. Host residency -- MEASURED AND PRINTED.** The batched route holds the whole window:
541 snapshots x 36.5 MB = **19.7 GB**, on top of the 12.0 GB fp16 field cache built from
them. That is what the in-process hand-off was supposed to remove and did not -- it removed
the ~20 GB of disk and moved it to RAM. Streaming holds one or two snapshots instead. Both
peaks are reported; there is no threshold, because the useful output is the number.

**WHAT STREAMING CANNOT REACH, AND WHY IT IS SAID HERE RATHER THAN IMPLIED.** The 12.0 GB
field cache is not buildup -- it IS the window, and the production integrator
(`lpdm/driver.py:compute_footprint`) is a CPU integrator that random-accesses all of it on
every step. Host residency therefore floors at the cache, not at a snapshot. Getting to
~74 MB needs the window to live in VRAM and the integration to happen there, i.e. making
`lpdm/gpu.py:GpuLPDM` the production integrator -- which docs/PLAN.md carries as a deferred item
and which is an INTEGRATOR change, not a plumbing change.

usage: docker/pyrun.sh bin/test_streaming.py runs/<case>/window --dt 0.0295858 [--n 24]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpdm.dumpsrc import MemDump                                      # noqa: E402
from lpdm.fields import FieldSet, dump_series                         # noqa: E402
from lpdm.hostwatch import HostWatch, ShmGuard, _rss_bytes, dir_bytes  # noqa: E402
from lpdm.les_stats import WindowAccumulator, window_stats            # noqa: E402
from lpdm.ringsrc import RingConsumer                                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ringsrc import stage                                        # noqa: E402

CACHE_FIELDS = ("u", "v", "w", "e", "eps", "dsig2dz", "ustar", "z0m", "invL")


def diff(a, b):
    """Worst absolute difference between two window_stats dicts, and where."""
    assert set(a) == set(b), set(a) ^ set(b)
    worst, where = 0.0, None
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray):
            d = 0.0 if np.array_equal(x, y) else float(np.nanmax(np.abs(
                np.asarray(x, float) - np.asarray(y, float))))
        elif isinstance(x, list):
            d = 0.0 if x == y else max(abs(p - q) for p, q in zip(x, y))
        else:
            d = 0.0 if x == y else abs(float(x) - float(y))
        if d > worst:
            worst, where = d, k
    return worst, where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windir")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--n", type=int, default=24, help="snapshots to stage")
    ap.add_argument("--k", type=float, default=3.4166, help="receptor level, fractional")
    ap.add_argument("--stage-dir", default=None,
                    help="where to stage; defaults to a tmpdir under the ring root so the "
                         "test exercises the same filesystem production uses")
    ap.add_argument("--mode", choices=("identity", "batched", "streamed"),
                    default="identity",
                    help="identity (default) runs BOTH routes in this process and compares "
                         "them, which is the gate -- but its RSS numbers are not usable, "
                         "because holding a copy of one route's cache to compare against "
                         "the other's inflates the second by exactly that cache. For a "
                         "real residency number run each mode in its OWN process; the "
                         "driver does all three.")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    paths = dump_series(a.windir)[:a.n]
    if len(paths) < 4:
        print(f"FATAL: {a.windir} has {len(paths)} dumps; need at least 4", file=sys.stderr)
        return 2
    ctype = np.float16
    root = a.stage_dir or os.environ.get("FLUX_RINGROOT", "/dev/shm/flux")
    os.makedirs(root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="stream_", dir=root)
    rep = {"windir": a.windir, "n": len(paths), "k": a.k, "staging": tmp}
    try:
        # A REAL STAGING DIRECTORY, written in the producer's own format, so this exercises
        # RingConsumer rather than a stand-in for it.
        stage(paths, tmp, a.dt, pause_after=int(str(paths[-1]).rsplit(".", 1)[1]))

        # ---- the /dev/shm guard, on the filesystem production actually stages to --------
        ring = RingConsumer(tmp, verbose=False)
        g = ShmGuard(tmp, ring.meta.snap_bytes, ring.meta.queue)
        info = g.check(fatal=False)
        rep["shm"] = info
        print(f"staging {tmp}")
        print(f"  snapshot {ring.meta.snap_bytes/1e6:.1f} MB, producer queue depth "
              f"{ring.meta.queue}, so the staging dir is bounded at "
              f"{ring.meta.queue*ring.meta.snap_bytes/1e6:.0f} MB by the producer itself "
              f"(io_lpdmonline.c:189)")
        print(f"  filesystem: {info['free_bytes']/1e6:.0f} MB free of "
              f"{info['total_bytes']/1e6:.0f} MB, need {info['need_bytes']/1e6:.0f} MB "
              f"-> {'OK' if info['ok'] else 'TOO SMALL'}")
        print(f"  NOTE: this test's producer writes every snapshot up front and has NO "
              f"backpressure, so the staging peak below is the whole window rather than "
              f"the queue depth. The real producer blocks at depth "
              f"{ring.meta.queue}; the staging figure that means anything is the one "
              f"stage5_footprint.py reports against a LIVE LES.")

        if a.mode != "identity":
            # ONE ROUTE, ONE PROCESS, so the peak is that route's own and nothing else's.
            gc.collect()
            hw = HostWatch(tmp, poll_s=0.05).start()
            if a.mode == "batched":
                handles, _ = ring.drain_until_pause()
                fs = FieldSet(handles, a.dt, verbose=False, cache_dtype=ctype)
                st = window_stats(handles, a.k)
            else:
                fs = FieldSet(None, a.dt, verbose=False, cache_dtype=ctype,
                              defer_load=True, stream=ring.iter_until_pause(),
                              nt=len(paths),
                              geom_dump=MemDump(ring.geometry(), step=0))
                acc = WindowAccumulator(fs.zpos[:, 0, 0], a.k)
                fs.load(stats=acc, release_handles=True)
                st = acc.finish()
            hr = hw.stop()
            per = len(paths) * ring.meta.snap_bytes / 1e9
            print(f"\n{a.mode.upper()} in isolation, {len(paths)} snapshots")
            print(f"  peak RSS          {hr['peak_rss_gb']:.3f} GB")
            print(f"  field cache       {fs.mem_gb:.3f} GB")
            print(f"  snapshots on host {per:.3f} GB if retained, "
                  f"{2 * ring.meta.snap_bytes / 1e9:.3f} GB if streamed")
            print(f"  peak staging      {hr['peak_staging_mb']:.1f} MB")
            print(f"  h = {st['h']:.1f} m, u* = {st['ustar']:.4f}  (the route agrees; "
                  f"--mode identity is what proves it)")
            rep.update({"mode": a.mode, "watch": hr, "cache_gb": float(fs.mem_gb)})
            if a.json:
                os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
                json.dump(rep, open(a.json, "w"), indent=1, default=float)
                print(f"wrote {a.json}")
            return 0

        # ---- BATCHED: what the ring path did before --------------------------------------
        gc.collect()
        base = _rss_bytes()
        hw_b = HostWatch(tmp, poll_s=0.05).start()
        handles, pause = ring.drain_until_pause()
        rss_after_drain = _rss_bytes()
        fs_b = FieldSet(handles, a.dt, verbose=False, cache_dtype=ctype)
        st_b = window_stats(handles, a.k)          # the SECOND pass over the same window
        rep_b = hw_b.stop()
        print(f"\nBATCHED  drain-then-load, statistics in a second pass")
        print(f"  {len(handles)} snapshots retained = "
              f"{len(handles)*ring.meta.snap_bytes/1e6:.0f} MB, cache {fs_b.mem_gb:.3f} GB")
        print(f"  RSS after drain {(rss_after_drain-base)/1e6:+.0f} MB over baseline; "
              f"peak {rep_b['peak_rss_gb']:.3f} GB")
        print(f"  staging dir peaked at {rep_b['peak_staging_mb']:.1f} MB")
        cache_b = {n: np.array(getattr(fs_b, n), copy=True) for n in CACHE_FIELDS}
        t_b, dtd_b = fs_b.t.copy(), fs_b.dt_dump
        del fs_b, handles
        gc.collect()

        # ---- STREAMED: production ---------------------------------------------------------
        shutil.rmtree(tmp)
        stage(paths, tmp, a.dt, pause_after=int(str(paths[-1]).rsplit(".", 1)[1]))
        ring2 = RingConsumer(tmp, verbose=False)
        gc.collect()
        base2 = _rss_bytes()
        hw_s = HostWatch(tmp, poll_s=0.05).start()
        n_expect = len(paths)
        fs_s = FieldSet(None, a.dt, verbose=False, cache_dtype=ctype, defer_load=True,
                        stream=ring2.iter_until_pause(), nt=n_expect,
                        geom_dump=MemDump(ring2.geometry(), step=0))
        acc = WindowAccumulator(fs_s.zpos[:, 0, 0], a.k)
        fs_s.load(stats=acc, release_handles=True)
        st_s = acc.finish()
        rep_s = hw_s.stop()
        print(f"\nSTREAMED consume-and-release, statistics fused into the one pass")
        print(f"  {len(fs_s.paths)} snapshots consumed, cache {fs_s.mem_gb:.3f} GB")
        print(f"  peak RSS {rep_s['peak_rss_gb']:.3f} GB")
        print(f"  staging dir peaked at {rep_s['peak_staging_mb']:.1f} MB")
        print(f"  released handles: "
              f"{sum(1 for h in fs_s.paths if getattr(h, '_released', False))}"
              f"/{len(fs_s.paths)}")
        left = [f for f in os.listdir(tmp) if f.startswith("snap.")]
        print(f"  staging files left behind: {len(left)} "
              f"(the consumer deletes each after reading; that is what releases the "
              f"producer's backpressure)")

        # ---- A. identity, asserted at zero ------------------------------------------------
        bad = []
        for n in CACHE_FIELDS:
            A, B = cache_b[n], getattr(fs_s, n)
            if not np.array_equal(A, B):
                bad.append(f"{n} max|d|="
                           f"{np.nanmax(np.abs(A.astype('f8') - B.astype('f8'))):.3e}")
        worst, where = diff(st_b, st_s)
        t_ok = np.array_equal(t_b, fs_s.t) and dtd_b == fs_s.dt_dump
        print(f"\nA. IDENTITY (asserted at exactly zero)")
        print(f"  field cache, {len(CACHE_FIELDS)} arrays : "
              f"{'BIT-IDENTICAL' if not bad else '; '.join(bad)}")
        print(f"  time axis and cadence            : {'IDENTICAL' if t_ok else 'DIFFERS'}")
        print(f"  window_stats, {len(st_b)} fields         : worst |diff| {worst:.3e}"
              f"{'' if where is None else ' at ' + where}"
              f" -> {'BIT-IDENTICAL' if worst == 0.0 else 'DIFFERS'}")

        # ---- B. residency, measured ---------------------------------------------------------
        retained = len(paths) * ring.meta.snap_bytes
        print(f"\nB. HOST RESIDENCY -- IN-PROCESS, AND THEREFORE NOT THE REAL NUMBER.")
        print(f"  This run holds a COPY of the batched cache ({cache_b['u'].nbytes/1e9:.2f} "
              f"GB across 9 arrays) to compare the streamed one against, so the streamed "
              f"peak below carries the batched cache too. Run --mode batched and "
              f"--mode streamed in separate processes for the honest figure.")
        print(f"  {'':<26}{'batched':>12}{'streamed':>12}")
        print(f"  {'peak RSS (GB)':<26}{rep_b['peak_rss_gb']:12.3f}"
              f"{rep_s['peak_rss_gb']:12.3f}")
        print(f"  {'snapshots retained':<26}{len(paths):12d}{'1-2':>12}")
        print(f"  {'that is (MB)':<26}{retained/1e6:12.0f}"
              f"{2*ring.meta.snap_bytes/1e6:12.0f}")
        print(f"  {'peak staging (MB)':<26}{rep_b['peak_staging_mb']:12.1f}"
              f"{rep_s['peak_staging_mb']:12.1f}")
        print(f"\n  Extrapolated to a production 541-snapshot window: batched would retain "
              f"{541*ring.meta.snap_bytes/1e9:.1f} GB of snapshots on top of a "
              f"{541*fs_s.mem_gb/len(paths):.1f} GB cache; streamed retains the cache and "
              f"{2*ring.meta.snap_bytes/1e6:.0f} MB.")
        print(f"  The cache is not buildup -- it IS the window, and compute_footprint is a "
              f"CPU integrator that random-accesses all of it. Reaching ~74 MB needs the "
              f"window in VRAM and the integration there (lpdm/gpu.py), a deferred item.")

        rep.update({"batched": rep_b, "streamed": rep_s,
                    "cache_bit_identical": not bad,
                    "cache_diffs": bad,
                    "time_axis_identical": bool(t_ok),
                    "window_stats_worst_diff": float(worst),
                    "window_stats_worst_field": where,
                    "snapshots_retained_batched_bytes": int(retained),
                    "staging_files_left": len(left)})
        ok = (not bad) and t_ok and worst == 0.0 and not left
        print(f"\nA: {'PASS' if ok else 'FAIL'}")
        rep["pass"] = bool(ok)
        if a.json:
            os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
            json.dump(rep, open(a.json, "w"), indent=1, default=float)
            print(f"wrote {a.json}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
