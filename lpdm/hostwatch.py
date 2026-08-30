"""Watch what the in-process hand-off actually costs the HOST, and refuse a /dev/shm too small.

Two small things that exist because the hand-off's whole argument is "this removes ~20 GB of
scratch per case", and an argument like that has to be MEASURED on the run that makes it
rather than asserted from the design.

**`ShmGuard`** -- the staging directory is a tmpfs, and Docker gives a container 64 MB of
`/dev/shm` by default. That is already recorded in this project (a 60-snapshot staging
attempt died with ENOSPC at 2.2 GB) and `docker/pyrun.sh` mounts a host tmpfs to work around
it -- but on a rented box nothing guarantees the mount, and the failure arrives 40 minutes
into a case as a write error rather than at second zero as a refusal. This checks the free
space against what the queue depth actually needs and says the required figure.

**`HostWatch`** -- a sampler thread reporting peak consumer RSS and peak staging-directory
size. Both are the numbers that say whether the stream is a stream: steady state should be
one or two snapshots resident, not a window's worth. It samples; it does not gate. A
threshold here would be a number picked rather than derived, and what is wanted is the
measurement.
"""
from __future__ import annotations

import os
import threading
import time

POLL_S = 0.5


def _rss_bytes():
    """This process's resident set size, from /proc. Returns None where unavailable."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return None


def dir_bytes(path):
    """Total size of the regular files directly in `path`. 0 if it is not there."""
    tot = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_file(follow_symlinks=False):
                        tot += e.stat(follow_symlinks=False).st_size
                except OSError:
                    pass
    except OSError:
        return 0
    return tot


class ShmGuard:
    """Refuse a staging directory whose filesystem cannot hold the queue. Loudly."""

    def __init__(self, path, snap_bytes, queue, margin=2):
        self.path = path
        self.snap_bytes = int(snap_bytes)
        self.queue = int(queue)
        # THE QUEUE DEPTH PLUS A MARGIN, because the producer is allowed to be exactly
        # `queue` snapshots ahead and is mid-write on one more -- and a tmpfs that is
        # exactly full is a write error, not a wait.
        self.need = self.snap_bytes * (self.queue + int(margin))

    def check(self, fatal=True):
        try:
            st = os.statvfs(self.path)
        except OSError as e:
            msg = f"cannot stat the staging filesystem at {self.path}: {e}"
            if fatal:
                raise RuntimeError(msg)
            return {"ok": False, "reason": msg}
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        ok = free >= self.need
        info = {"ok": bool(ok), "path": self.path, "total_bytes": int(total),
                "free_bytes": int(free), "need_bytes": int(self.need),
                "snap_bytes": self.snap_bytes, "queue": self.queue}
        if not ok and fatal:
            raise RuntimeError(
                f"the staging filesystem at {self.path} has {free/1e6:.0f} MB free of "
                f"{total/1e6:.0f} MB, and the hand-off needs at least "
                f"{self.need/1e6:.0f} MB ({self.queue} queued snapshots + 2 margin, at "
                f"{self.snap_bytes/1e6:.1f} MB each).\n"
                f"  Docker's default /dev/shm is 64 MB. Mount a larger tmpfs at this path "
                f"on BOTH sides -- docker/pyrun.sh and the FastEddy wrapper must agree on "
                f"it, because lpdmOnlineDir is written into one container's .in and polled "
                f"from another -- or run with --shm-size.")
        return info


class HostWatch:
    """Sample peak host RSS and peak staging-directory size until stopped."""

    def __init__(self, staging_dir=None, poll_s=POLL_S):
        self.dir = staging_dir
        self.poll_s = float(poll_s)
        self.peak_rss = 0
        self.peak_dir = 0
        self.peak_dir_files = 0
        self.n = 0
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            r = _rss_bytes()
            if r is not None and r > self.peak_rss:
                self.peak_rss = r
            if self.dir:
                b = dir_bytes(self.dir)
                if b > self.peak_dir:
                    self.peak_dir = b
                    try:
                        self.peak_dir_files = len(
                            [f for f in os.listdir(self.dir) if f.endswith(".raw")])
                    except OSError:
                        pass
            self.n += 1
            self._stop.wait(self.poll_s)

    def start(self):
        if self._t is None:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
            self._t = None
        return self.report()

    def report(self):
        return {"peak_rss_bytes": int(self.peak_rss),
                "peak_rss_gb": self.peak_rss / 1e9,
                "peak_staging_bytes": int(self.peak_dir),
                "peak_staging_mb": self.peak_dir / 1e6,
                "peak_staging_files": int(self.peak_dir_files),
                "samples": int(self.n), "poll_s": self.poll_s,
                "staging_dir": self.dir}

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
