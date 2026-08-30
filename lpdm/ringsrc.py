"""Consume the in-process LPDM hand-off: staged snapshots -> MemDumps -> the VRAM ring.

THE OTHER HALF OF SRC/IO/io_lpdmonline.c. FastEddy stages each output-cadence snapshot as
one raw file in a tmpfs directory and blocks when the queue is full; this reads them, hands
them to the readers as `MemDump`s (`lpdm/dumpsrc.py`), and deletes them. Nothing is encoded,
nothing reaches a disk, and the analysis stack above it is unchanged -- which is the whole
design constraint, because `window_stats` and `FieldSet` are what every gate runs through
and a second implementation of either would be an assumption rather than a fact.

THE FORMAT IS READ, NOT ASSUMED. `meta.txt` carries the extents, the timestep, and the
field ORDER. Hard-coding that order here would put the same constant on both sides of an
interface, which in this project is a constant that eventually disagrees silently; the
producer writes it and this parses it, so a field inserted on the C side is picked up or
raises, never mis-assigned.

WHAT THE PROTOCOL GUARANTEES, and why it is files rather than a lock-free ring:

  snap.<step>.raw     one whole snapshot, written to .part and renamed, so it is never
                      observable half-written
  snap.<step>.ok      completion marker, created only after the data file is closed --
                      this is the entire synchronisation protocol
  geom.raw/.ok        static geometry, once per run (xPos, yPos, zPos, topoPos)
  pause.<step>        the LES has staged a whole window and is blocked
  resume.<step>       written by this side to let it continue
  done                the run is over; stop polling

Deleting a snapshot after reading it is what releases the producer's backpressure, so a
consumer that crashes stalls the LES visibly instead of letting it fill tmpfs.
"""
from __future__ import annotations

import glob
import os
import re
import time

import numpy as np

from .dumpsrc import MemDump

POLL_S = 0.002
# How long to wait for the next snapshot before concluding the producer has died. A
# 122^3 step at the production dt is ~2.5 s of wall per 5 s of model time, so 300 s is
# two orders of magnitude of slack; the point is to fail with a message rather than to
# hang a corpus job overnight.
STALL_S = 300.0


class RingMeta:
    """The snapshot format, as published by the producer."""

    def __init__(self, path):
        kv, self.fields3, self.fields2, self.geom = {}, [], [], []
        for line in open(path):
            t = line.split()
            if not t:
                continue
            if t[0] == "fields3":
                self.fields3 = t[1:]
            elif t[0] == "fields2":
                self.fields2 = t[1:]
            elif t[0] == "geom":
                self.geom = t[1:]
            elif len(t) >= 2:
                kv[t[0]] = t[1]
        self.nx, self.ny, self.nz = (int(kv[k]) for k in ("nx", "ny", "nz"))
        self.dt = float(kv["dt"])
        self.frq_output = int(kv["frqOutput"])
        self.selector = int(kv["selector"])
        self.queue = int(kv["queue"])
        self.n3 = self.nx * self.ny * self.nz
        self.n2 = self.nx * self.ny
        # bytes: one f64 model time, then the 3-D fields, then the 2-D ones
        self.snap_bytes = 8 + 4 * (len(self.fields3) * self.n3 + len(self.fields2) * self.n2)

    def __repr__(self):
        return (f"RingMeta({self.nx}x{self.ny}x{self.nz}, dt={self.dt:g}, "
                f"3d={self.fields3}, 2d={self.fields2}, "
                f"{self.snap_bytes/1e6:.1f} MB/snapshot)")


def _wait_for(path, what, stall_s=STALL_S):
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > stall_s:
            raise TimeoutError(
                f"waited {stall_s:.0f} s for {what} ({path}). The LES side either died or "
                f"never reached this point; check its log rather than retrying, because "
                f"the staging directory is not restartable mid-window.")
        time.sleep(POLL_S)


class RingConsumer:
    """Reads staged snapshots out of `dirpath` in step order."""

    def __init__(self, dirpath, verbose=True):
        self.dir = dirpath
        self.verbose = verbose
        _wait_for(os.path.join(dirpath, "meta.txt"), "the producer's meta.txt")
        self.meta = RingMeta(os.path.join(dirpath, "meta.txt"))
        if verbose:
            print(f"  ring: {self.meta}")
        self._geom = None
        self._seen = set()

    # -- geometry -------------------------------------------------------------------
    def geometry(self):
        """The static coordinate fields, read once. Attached to the first MemDump.

        FieldSet and window_stats both SEARCH the series for a handle carrying zPos rather
        than assuming the first one has it, because under ioLPDMmode geometry lands only in
        the run's first file. Attaching it to the first snapshot reproduces that layout
        exactly, so neither reader needs to know which source it is reading.
        """
        if self._geom is not None:
            return self._geom
        _wait_for(os.path.join(self.dir, "geom.ok"), "the static geometry")
        m = self.meta
        raw = np.fromfile(os.path.join(self.dir, "geom.raw"), dtype=np.float32)
        want = sum(m.n2 if n == "topoPos" else m.n3 for n in m.geom)
        if raw.size != want:
            raise ValueError(f"geom.raw has {raw.size} floats, the format says {want}")
        out, off = {}, 0
        for n in m.geom:
            ne = m.n2 if n == "topoPos" else m.n3
            shp = (m.ny, m.nx) if n == "topoPos" else (m.nz, m.ny, m.nx)
            out[n] = raw[off:off + ne].reshape(shp)
            off += ne
        self._geom = out
        return out

    # -- snapshots ------------------------------------------------------------------
    def _read_one(self, step, keep=False):
        m = self.meta
        p = os.path.join(self.dir, f"snap.{step}.raw")
        with open(p, "rb") as f:
            buf = f.read()
        if len(buf) != m.snap_bytes:
            raise ValueError(f"{p} is {len(buf)} bytes, the format says {m.snap_bytes}")
        t_model = float(np.frombuffer(buf, dtype=np.float64, count=1)[0])
        a = np.frombuffer(buf, dtype=np.float32, offset=8)
        out, off = {}, 0
        for n in m.fields3:
            out[n] = a[off:off + m.n3].reshape(m.nz, m.ny, m.nx)
            off += m.n3
        for n in m.fields2:
            out[n] = a[off:off + m.n2].reshape(m.ny, m.nx)
            off += m.n2
        if not self._seen:
            out.update(self.geometry())
        # ASSERT ON THE ARTIFACT. A snapshot of NaN costs a whole window and the only other
        # symptom is a footprint that looks merely odd; the check is one pass over 36 MB.
        for n in m.fields3:
            if not np.isfinite(out[n]).all():
                raise ValueError(f"snapshot {step}: {n} is not finite -- the LES has "
                                 f"produced CORRUPTED state and the window is dead")
        self._seen.add(int(step))
        if not keep:
            os.remove(p)
            os.remove(os.path.join(self.dir, f"snap.{step}.ok"))
        return MemDump(out, step), t_model

    def iter_until_pause(self, keep=False):
        """YIELD each snapshot as it is staged, in step order, up to the next pause.

        This is the streaming form, and it is what keeps a window off the host. The
        producer stages at most `lpdmOnlineQueue` (default 4) snapshots before it BLOCKS
        (`io_lpdmonline.c:189`), so the tmpfs is bounded at ~146 MB whatever the consumer
        does -- but the CONSUMER's own memory is not bounded by anything except what it
        chooses to keep, and `drain_until_pause` chooses to keep all 541. At 36.5 MB each
        that is **19.7 GB**, on top of whatever the caller then builds from them.

        A caller that consumes each snapshot as it arrives and releases it (see
        `MemDump.release` and `lpdm/fields.py:FieldSet.load`) holds one or two instead.

        Reading in STEP order matters and is not automatic: the producer's filenames sort
        lexically as snap.10 < snap.9, and a window whose time axis is out of order
        interpolates between the wrong pair of snapshots and produces a plausible, wrong
        footprint.

        After the generator finishes, `self.last_pause_step` is the step of the pause that
        ended it, or None if the run finished instead. It is an attribute rather than a
        return value because a generator's return value is awkward to reach and this one is
        needed to resume the LES.
        """
        self.last_pause_step = None
        t0 = time.time()
        while True:
            # THE PAUSE MARKER IS SAMPLED BEFORE THE DIRECTORY LISTING, so a snapshot
            # staged in the same instant as the marker is still drained: the exit test
            # below requires the marker AND a freshly empty ready-list.
            pause = self._pause_step()
            for s in sorted(self._ready()):
                h, _ = self._read_one(s, keep=keep)
                yield h
                t0 = time.time()
            if pause is not None and not self._ready():
                self.last_pause_step = pause
                return
            if os.path.exists(os.path.join(self.dir, "done")) and not self._ready():
                self.last_pause_step = None
                return
            if time.time() - t0 > STALL_S:
                raise TimeoutError(
                    f"no new snapshot in {STALL_S:.0f} s and no pause marker in "
                    f"{self.dir}. The LES side has stopped producing.")
            time.sleep(POLL_S)

    def drain_until_pause(self, keep=False):
        """Every snapshot up to and including the next pause, in step order, AS A LIST.

        Returns (handles, pause_step). A thin `list()` over `iter_until_pause`, kept
        because the transparency tests (`bin/test_ringsrc.py`, `bin/test_lpdmonline.py`)
        compare whole windows and genuinely want them all at once. **Production streams
        instead** -- this form holds the entire window in host RAM by construction.
        """
        handles = list(self.iter_until_pause(keep=keep))
        return handles, self.last_pause_step

    def expected_snapshots(self, t_min, t_max, dt_model):
        """How many snapshots a window of [t_min, t_max] should deliver.

        The streamed consumer has to size its cache BEFORE the first snapshot arrives, and
        the count is not knowable from the ring until the pause lands. It is knowable from
        the schedule: the output cadence is `frq_output * dt` and the driver already
        asserts that lands on an integer step count. The delivered count is asserted
        against this rather than trusted -- a window one dump short is a real failure mode
        this project has already paid for.
        """
        cad = float(self.meta.frq_output) * float(dt_model)
        if cad <= 0:
            raise ValueError(f"the ring meta gives a non-positive output cadence {cad}")
        return int(round((float(t_max) - float(t_min)) / cad)) + 1

    def _ready(self):
        """Steps that are staged and NOT yet read.

        The `not yet read` half is load-bearing and was missing. Deleting a snapshot after
        reading it is what normally removes it from this list, so the omission was
        invisible in production -- but `keep=True` (the acceptance comparison, where the
        staged files ARE the artifact) turned the drain loop into an unbounded accumulator:
        the same 23 snapshots re-read every 2 ms until the container was OOM-killed. Exit
        137 and, because stdout was redirected and therefore buffered, not one line of
        output to say where. Tracking what has been read makes `keep` orthogonal to
        termination, which is what it always should have been.
        """
        out = []
        for p in glob.glob(os.path.join(self.dir, "snap.*.ok")):
            k = int(re.search(r"snap\.(\d+)\.ok$", p).group(1))
            if k not in self._seen:
                out.append(k)
        return out

    def _pause_step(self):
        g = glob.glob(os.path.join(self.dir, "pause.*"))
        return int(g[0].rsplit(".", 1)[1]) if g else None

    def resume(self, step):
        """Let the LES continue past its pause."""
        open(os.path.join(self.dir, f"resume.{step}"), "w").close()

    def done(self):
        return os.path.exists(os.path.join(self.dir, "done"))
