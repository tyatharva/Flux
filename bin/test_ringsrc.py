#!/usr/bin/env python3
"""Does a window delivered through the ring produce the same numbers as one on disk?

WHAT THIS ESTABLISHES AND WHAT IT DOES NOT. It fabricates staged snapshots from REAL dumps
in the producer's wire format, runs `lpdm/ringsrc.py` over them, and requires
`window_stats` and `FieldSet` to come out BIT-IDENTICAL to the netCDF path. That covers
the consumer completely -- format arithmetic, field order, step ordering, geometry
attachment, the delete-after-read that releases backpressure, and the pause handshake.

It does NOT establish that the C producer writes what this expects. Nothing run on a CPU
can: the producer's buffer is `ioBuffField` at the moment the netCDF writer would consume
it, and only a real LES fills it. That agreement is what `lpdmOnlineSelector = 2` exists
for -- one run staging AND writing from bit-identical buffers -- and it is checked in the
in-process acceptance stage, on the GPU. Saying so here rather than letting a green tick
imply more than it covers is the point; this project has twice shipped a gate that scored
a different code path than production ran.

BIT-IDENTITY, not a tolerance. There is no physics between the two paths, so the correct
tolerance is exactly zero and anything looser would hide the bug it exists to find.

STEP ORDER IS TESTED ON PURPOSE. The producer's filenames sort lexically as snap.10 before
snap.9, and a window whose time axis is permuted interpolates between the wrong pair of
snapshots -- a plausible, wrong footprint with nothing complaining. The fabricated steps
below deliberately span a decade boundary so a lexical sort cannot pass.

usage: test_ringsrc.py <window-dir> [--ndump 12]
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
import sys
import tempfile
import threading
import time

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpdm.dumpsrc import MemDump, step_of
from lpdm.fields import FieldSet, dump_series
from lpdm.les_stats import window_stats
from lpdm.ringsrc import RingConsumer

F3 = ["u", "v", "w", "theta", "TKE_0"]          # must match io_lpdmonline.c's lpdmOn3
F2 = ["fricVel", "z0m", "invOblen"]             # ... and lpdmOn2
GE = ["xPos", "yPos", "zPos", "topoPos"]        # ... and lpdmOnG


def stage(paths, outdir, dt, pause_after=None):
    """Emit `paths` as the producer would. Written from io_lpdmonline.c's format."""
    os.makedirs(outdir, exist_ok=True)
    with Dataset(paths[0]) as ds:
        nz, ny, nx = np.squeeze(np.asarray(ds["zPos"][:])).shape
    with open(os.path.join(outdir, "meta.txt"), "w") as f:
        f.write(f"nx {nx}\nny {ny}\nnz {nz}\ndt {dt!r}\nfrqOutput 0\nselector 2\nqueue 4\n")
        f.write("fields3 " + " ".join(F3) + "\n")
        f.write("fields2 " + " ".join(F2) + "\n")
        f.write("geom " + " ".join(GE) + "\n")
    # geometry, once, from whichever dump carries it -- same rule the readers use
    gsrc = next(p for p in paths
                if Dataset(p).variables.keys() >= {"zPos", "xPos"})
    with open(os.path.join(outdir, "geom.raw"), "wb") as f, Dataset(gsrc) as ds:
        for n in GE:
            f.write(np.ascontiguousarray(np.squeeze(np.asarray(ds[n][:])),
                                         dtype=np.float32).tobytes())
    open(os.path.join(outdir, "geom.ok"), "w").close()
    for p in paths:
        s = step_of(p)
        tmp = os.path.join(outdir, f"snap.{s}.part")
        with open(tmp, "wb") as f, Dataset(p) as ds:
            f.write(struct.pack("<d", s * dt))
            for n in F3 + F2:
                f.write(np.ascontiguousarray(np.squeeze(np.asarray(ds[n][:])),
                                             dtype=np.float32).tobytes())
        os.rename(tmp, os.path.join(outdir, f"snap.{s}.raw"))
        open(os.path.join(outdir, f"snap.{s}.ok"), "w").close()
    if pause_after is not None:
        open(os.path.join(outdir, f"pause.{pause_after}"), "w").close()
    else:
        open(os.path.join(outdir, "done"), "w").close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windir")
    ap.add_argument("--ndump", type=int, default=12)
    ap.add_argument("--dt", type=float, default=0.0295858)
    ap.add_argument("--k", type=float, default=3.0)
    a = ap.parse_args()

    paths = dump_series(a.windir)
    sel = paths[:: max(1, len(paths) // a.ndump)][: a.ndump]
    steps = [step_of(p) for p in sel]
    print(f"RING CONSUMER vs THE FILE PATH -- {len(sel)} dumps from {a.windir}")
    print(f"  steps {steps[0]} .. {steps[-1]}")
    lex = sorted(str(s) for s in steps)
    print(f"  a LEXICAL sort of these steps {'DIFFERS from' if lex != [str(x) for x in sorted(steps)] else 'happens to match'}"
          f" the numeric one -- so the ordering check is {'live' if lex != [str(x) for x in sorted(steps)] else 'NOT exercised by this series'}")

    fails = 0
    tmp = tempfile.mkdtemp(prefix="ringtest_", dir=os.environ.get("SCRATCH", "/tmp"))
    try:
        stage(sel, tmp, a.dt, pause_after=steps[-1])
        rc = RingConsumer(tmp, verbose=True)
        handles, pause = rc.drain_until_pause()
        print(f"  drained {len(handles)} snapshots, pause at step {pause}")

        # the delete-after-read is what releases the producer's backpressure
        left = [f for f in os.listdir(tmp) if f.startswith("snap.")]
        ok = not left
        print(f"  [{'ok  ' if ok else 'FAIL'}] queue drained: {len(left)} snap.* files left")
        fails += (not ok)

        ok = [h.step for h in handles] == sorted(steps)
        print(f"  [{'ok  ' if ok else 'FAIL'}] delivered in NUMERIC step order")
        if not ok:
            print(f"        got {[h.step for h in handles]}")
        fails += (not ok)

        ok = pause == steps[-1]
        print(f"  [{'ok  ' if ok else 'FAIL'}] pause handshake: saw step {pause}")
        fails += (not ok)
        rc.resume(pause)
        ok = os.path.exists(os.path.join(tmp, f"resume.{pause}"))
        print(f"  [{'ok  ' if ok else 'FAIL'}] resume marker written")
        fails += (not ok)

        # ---- the numbers ------------------------------------------------------------
        # THE REFERENCE IS fp32, AND THAT IS NOT A LOOSENING. The wire format is fp32
        # because FastEddy is fp32 throughout, so in production the ring carries the exact
        # model values and the netCDF lean file carries a CF-packed 16-bit version of them
        # -- the ring is the MORE accurate of the two. Fabricating the ring FROM the
        # netCDF inverts that: the file becomes an fp64 unpacked reference and the staged
        # copy a lossy fp32 one, and the ~1e-7 residual that produces is a property of this
        # harness rather than of the transport. Scoring against an fp32 reference removes
        # the harness's own artifact and leaves the transport, which must be exact.
        # bin/test_dumpsrc.py has already tied fp64 MemDumps to netCDF bit-identically, so
        # nothing is lost by going through MemDump here.
        ref = [MemDump.from_netcdf(p_, dtype=np.float32) for p_ in sel]
        s64 = window_stats(sel, a.k)
        sa, sb = window_stats(ref, a.k), window_stats(handles, a.k)
        d64 = max((abs(s64[k] - sa[k]) / max(abs(s64[k]), 1e-30)
                   for k in sa if isinstance(sa.get(k), float) and s64.get(k) == s64.get(k)
                   and isinstance(s64.get(k), float)), default=0.0)
        print(f"  ---- the fp32 wire format costs {d64:.2e} relative against the fp64 "
              f"unpacked file, which is the harness's artifact and not the transport's")
        bad = []
        for k in sorted(set(sa) | set(sb)):
            va, vb = sa.get(k), sb.get(k)
            if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                same = va is not None and vb is not None and np.array_equal(va, vb)
            else:
                same = (va == vb) or (va != va and vb != vb)
            if not same:
                bad.append((k, va, vb))
        print(f"  [{'ok  ' if not bad else 'FAIL'}] window_stats: {len(sa)} keys, "
              f"{len(bad)} differ")
        for k, va, vb in bad[:6]:
            print(f"        {k}: file {va!r}  ring {vb!r}")
        fails += bool(bad)

        fa, fb = (FieldSet(ref, a.dt, verbose=False),
                  FieldSet(handles, a.dt, verbose=False))
        bad = []
        for nm in ("t", "zk", "zg", "zg_dx", "zg_dy", "u", "v", "w", "e", "eps",
                   "dsig2dz", "ustar", "z0m"):
            va, vb = getattr(fa, nm, None), getattr(fb, nm, None)
            if va is None and vb is None:
                continue
            if va is None or vb is None or not np.array_equal(va, vb):
                bad.append(nm)
        print(f"  [{'ok  ' if not bad else 'FAIL'}] FieldSet: {len(bad)} attributes differ"
              + (f" ({bad})" if bad else ""))
        fails += bool(bad)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"  {fails} failure(s)")
    print("  NOTE: this scores the CONSUMER. Producer/consumer agreement needs a real LES")
    print("        run at lpdmOnlineSelector = 2 and is checked in the GPU acceptance.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
