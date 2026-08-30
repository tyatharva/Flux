#!/usr/bin/env python3
"""TRANSPARENCY: does routing the readers through open_dump() change a single number?

WHY THIS TEST IS THE WHOLE POINT OF THE INDIRECTION. `lpdm/dumpsrc.py` lets the two
functions every gate depends on -- `FieldSet` and `window_stats` -- read a snapshot that is
in RAM instead of on disk. That is worth doing only if it is FREE: if the in-RAM path
returns anything different from the file path, every gate result in this project acquires a
new dependency on which one was used, and the in-process hook stops being plumbing and
becomes an estimator change.

So this asserts BIT-IDENTITY, not agreement within a tolerance. There is no physics between
the two paths -- the same bytes reach the same arithmetic -- so any difference at all is a
bug in the indirection and a tolerance would only hide it. This is the opposite situation
from a sampling comparison, where PROJECT_BRIEF.md requires a DERIVED tolerance: here the correct
tolerance is exactly zero and anything else is wrong.

Three things are checked:

  1. `window_stats` returns bit-identical values through both handle kinds.
  2. `FieldSet` builds bit-identical caches through both handle kinds.
  3. The readers ask for NO variable outside the enumerated set in `lpdm/dumpsrc.py`.
     A duck type that silently lacks an attribute the real object has is the same failure
     shape as a silently defaulted parameter -- it produces a plausible wrong number rather
     than an error -- so the set is asserted rather than assumed, and this test fails if a
     reader grows a new access.

usage: test_dumpsrc.py <window-dir> [--ndump 12]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpdm import dumpsrc
from lpdm.dumpsrc import DUMP_VARS_2D, DUMP_VARS_3D, DUMP_VARS_GEOM, MemDump
from lpdm.fields import FieldSet, dump_series
from lpdm.les_stats import window_stats


class Recording(MemDump):
    """A MemDump that records every name asked of it, so drift is caught rather than
    silently returning a KeyError from somewhere deep in a reader."""

    __slots__ = ("seen",)

    def __init__(self, arrays, step, seen):
        super().__init__(arrays, step)
        object.__setattr__(self, "seen", seen)

    def __getitem__(self, k):
        self.seen.add(k)
        return super().__getitem__(k)

    def __contains__(self, k):
        self.seen.add(k)
        return super().__contains__(k)

    @property
    def variables(self):
        return _RecordingMap(super().variables, self.seen)


class _RecordingMap(dict):
    def __init__(self, d, seen):
        super().__init__(d)
        self._seen = seen

    def __contains__(self, k):
        self._seen.add(k)
        return dict.__contains__(self, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windir")
    ap.add_argument("--ndump", type=int, default=12)
    ap.add_argument("--dt", type=float, default=0.0295858)
    ap.add_argument("--k", type=float, default=3.0)
    a = ap.parse_args()

    paths = dump_series(a.windir)
    if not paths:
        raise SystemExit(f"no dumps in {a.windir}")
    sel = paths[:: max(1, len(paths) // a.ndump)][: a.ndump]
    print(f"TRANSPARENCY OF lpdm/dumpsrc.py -- {len(sel)} dumps from {a.windir}")
    print()

    seen = set()
    mem = [Recording(MemDump.from_netcdf(p, dtype=np.float64).variables,
                     dumpsrc.step_of(p), seen) for p in sel]

    fails = 0

    # ---- 1. window_stats -----------------------------------------------------------
    sa = window_stats(sel, a.k)
    sb = window_stats(mem, a.k)
    bad = []
    for k in sorted(set(sa) | set(sb)):
        va, vb = sa.get(k), sb.get(k)
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            ok = (va is not None and vb is not None
                  and np.array_equal(np.asarray(va), np.asarray(vb)))
        else:
            ok = (va == vb) or (va != va and vb != vb)     # NaN == NaN counts as equal
        if not ok:
            bad.append((k, va, vb))
    print(f"  [{'ok  ' if not bad else 'FAIL'}] window_stats: {len(sa)} keys, "
          f"{len(bad)} differ")
    for k, va, vb in bad[:8]:
        print(f"        {k}: file {va!r}  ring {vb!r}")
    fails += bool(bad)

    # ---- 2. FieldSet ---------------------------------------------------------------
    fa = FieldSet(sel, a.dt, verbose=False)
    fb = FieldSet(mem, a.dt, verbose=False)
    bad = []
    for nm in ("t", "zk", "zg", "zg_dx", "zg_dy", "u", "v", "w", "e", "eps", "dsig2dz",
               "ustar", "z0m"):
        va, vb = getattr(fa, nm, None), getattr(fb, nm, None)
        if va is None and vb is None:
            continue
        if va is None or vb is None or not np.array_equal(np.asarray(va), np.asarray(vb)):
            d = (float(np.nanmax(np.abs(np.asarray(va, np.float64)
                                        - np.asarray(vb, np.float64))))
                 if va is not None and vb is not None
                 and np.shape(va) == np.shape(vb) else float("nan"))
            bad.append((nm, d))
    for nm in ("nx", "ny", "nz", "dx", "dy", "x0", "y0", "Lx", "Ly", "dt_dump"):
        if getattr(fa, nm) != getattr(fb, nm):
            bad.append((nm, float("nan")))
    print(f"  [{'ok  ' if not bad else 'FAIL'}] FieldSet: 23 attributes, {len(bad)} differ")
    for nm, d in bad[:8]:
        print(f"        {nm}: max |diff| {d:.3e}")
    fails += bool(bad)

    # ---- 3. no reader asks for anything outside the enumerated set -----------------
    known = set(DUMP_VARS_3D) | set(DUMP_VARS_2D) | set(DUMP_VARS_GEOM)
    extra = seen - known
    print(f"  [{'ok  ' if not extra else 'FAIL'}] the readers asked for "
          f"{len(seen)} distinct names, {len(extra)} outside lpdm/dumpsrc.py's set")
    if extra:
        print(f"        UNKNOWN: {sorted(extra)}")
        print( "        Add them to DUMP_VARS_* and make sure the ring supplies them --")
        print( "        a name the ring does not carry is a KeyError deep inside a reader.")
    fails += bool(extra)
    print(f"        asked: {sorted(seen)}")
    print()
    print(f"  {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
