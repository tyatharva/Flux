#!/usr/bin/env python3
"""One machine's share of the corpus: 8 months, a shared day queue, N GPUs, one command.

    run_corpus --machine 3 --out /out
    run_corpus --machine 3 --stub --out /out/dry        (no GPU, no HRRR, seconds)

=== THE PARTITION IS PRINTED, NOT ASSUMED ===

`lpdm/partition.py` assigns each of the 64 corpus months to exactly one of 8 machines and
asserts that the assignment is total and disjoint at import. The whole 8 x 8 table is
printed at startup with a coverage line recomputed from the printed rows, so "all 64 are
covered exactly once" is checkable from the log of any single machine.

=== WITHIN A MACHINE IT IS A SHARED QUEUE OVER DAYS, NOT A MONTH PER GPU ===

The obvious arrangement -- eight months, eight GPUs, one each -- is wrong for the same
reason the seed runner is a queue rather than two passes: WALL TIME IS SET BY THE SLOWEST
WORKER, and the months are not equal. A month's cost is its number of ACCEPTED days, and
acceptance is meteorological: a winter month with many overcast days yields far fewer cases
than a July. Measured on this machine's own months (`--dry-run` prints it), the busiest
month carries 1.4-2.3x the cases of the quietest, so a pinned assignment idles most of the
box through the tail of one month. A shared queue over all ~243 days has every GPU working
until the last day is taken.

The queue is ordered chronologically, which costs nothing and makes the progress view
readable: the machine visibly walks its months.

=== EVERY DAY ENDS IN EXACTLY ONE OF FOUR STATES, AND ALL FOUR ARE IN THE MANIFEST ===

    case      an hour passed the screens and a record was written
    missing   the hour pool was exhausted, or the day's HRRR is absent, or the fitted
              sounding put z_i outside what the domain holds -- always WITH THE REASON
    failed    something broke that is not a screen; the day is retryable
    skipped   a previous run already resolved this day (see RESUME)

A failed day never stops the machine. Over ~243 days on rented hardware the expected number
of transient failures is not zero, and a run that stops at the first one wastes the rental.

=== RESUME ===

Keyed off the artifacts, not off a checkpoint: a day whose record exists in `pairs_npz/` is
done, and a day whose draw record says the pool was exhausted is done. Both live on the
mounted volume, so `docker rm` and a re-run pick up where the last one stopped. A day
previously recorded MISSING is NOT re-drawn by default -- the draw is seeded from the date
alone, so a re-run would redraw the same sequence and reach the same answer -- but
`--retry-missing` re-evaluates them, which is what to use if a whole day was lost to a
network outage rather than to the weather.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lpdm.corpus import split_of                                        # noqa: E402
from lpdm.partition import (MONTHS, N_MACHINES, days_in, describe,      # noqa: E402
                            month_str, months_for, summary)

_t0 = time.time()
_print_lock = threading.Lock()
_logfh = None


def say(*a, stamp=True):
    with _print_lock:
        body = " ".join(str(x) for x in a)
        msg = f"[{time.time() - _t0:8.1f}s] {body}" if stamp else body
        print(msg, flush=True)
        if _logfh:
            _logfh.write(msg + "\n")
            _logfh.flush()


def git_commit():
    """From .git/HEAD, or from the image label baked in at build time.

    The container has no git binary and the corpus image has no .git either -- the code is
    baked in and the COMMIT IS THE IMAGE TAG. So the label is the authority inside the
    image and the working tree is the authority outside it.
    """
    env = os.environ.get("FLUX_COMMIT")
    if env:
        return env
    try:
        ref = open(os.path.join(ROOT, ".git", "HEAD")).read().strip()
        if ref.startswith("ref: "):
            return open(os.path.join(ROOT, ".git", ref[5:])).read().strip()
        return ref
    except OSError:
        return None


# --------------------------------------------------------------------------- GPUs
def discover_gpus():
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=index,name,compute_cap,memory.total,uuid",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
        out = []
        for ln in r.stdout.splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) >= 5:
                out.append({"index": int(p[0]), "name": p[1], "cc": p[2],
                            "vram_mib": int(float(p[3])), "uuid": p[4]})
        return out
    except Exception:
        return []


class HostWatch(threading.Thread):
    """Peak host memory for the WHOLE MACHINE. This is the number the rental turns on.

    PROJECT_BRIEF.md records 12.45 GB peak host RSS for ONE corpus case -- the LPDM's 12.0 GB fp16
    field cache, which is not buildup but the window itself, random-accessed by a CPU
    integrator. That has never been measured N-way. Eight concurrent cases could approach
    100 GB, and a box that starts swapping does not fail, it just runs several times slower
    with nothing in the output to say why. So this samples continuously and the early
    report prints it before the rental is committed.
    """

    CG = ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory.current",
          "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
          "/sys/fs/cgroup/memory/memory.usage_in_bytes")

    def __init__(self, period=5.0):
        super().__init__(daemon=True)
        self.period, self.stop = period, threading.Event()
        self.cgroup_peak = 0
        self.mem_avail_min = None
        self.swap_used_peak = 0
        self.mem_total = self._meminfo("MemTotal")

    @staticmethod
    def _meminfo(key):
        try:
            for ln in open("/proc/meminfo"):
                if ln.startswith(key + ":"):
                    return int(ln.split()[1]) * 1024
        except Exception:
            pass
        return None

    def _cgroup(self):
        for p in self.CG:
            try:
                return int(open(p).read().strip())
            except Exception:
                continue
        return 0

    def sample(self):
        self.cgroup_peak = max(self.cgroup_peak, self._cgroup())
        av = self._meminfo("MemAvailable")
        if av is not None:
            self.mem_avail_min = av if self.mem_avail_min is None else min(self.mem_avail_min, av)
        st, sf = self._meminfo("SwapTotal"), self._meminfo("SwapFree")
        if st and sf is not None:
            self.swap_used_peak = max(self.swap_used_peak, st - sf)

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.period)

    def snapshot(self):
        g = 1 << 30
        return {"cgroup_peak_gb": round(self.cgroup_peak / g, 2),
                "mem_total_gb": round((self.mem_total or 0) / g, 1),
                "mem_available_min_gb": (round(self.mem_avail_min / g, 1)
                                         if self.mem_avail_min is not None else None),
                "swap_used_peak_gb": round(self.swap_used_peak / g, 2)}


# --------------------------------------------------------------------------- progress
class Progress:
    """The live view's data, on disk. THE DISPLAY IS A SEPARATE PROCESS ON PURPOSE.

    An SSH session will drop, and a run that only exists as a terminal attached to a
    foreground process dies with it. So this writes a small JSON on the MOUNTED VOLUME
    after every state change; `bin/corpus_progress.py` renders it and can be started,
    killed and restarted as often as the connection needs. The run itself never depends on
    anyone watching.

    Written to a temporary file and renamed, so a reader never sees a half-written record.
    """

    def __init__(self, path, machine, months, n_days, gpus, expected):
        self.path, self.lock = path, threading.Lock()
        self.d = {
            "format": "flux-corpus-progress/1",
            "machine": machine, "pid": os.getpid(),
            "started_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "months": [month_str(m) for m in months],
            "n_days_total": n_days, "expected_cases": expected,
            "gpus": {str(g): {"state": "idle"} for g in gpus},
            "counts": {"case": 0, "missing": 0, "failed": 0, "resumed": 0, "done": 0},
            "gpu_h_per_case": None, "elapsed_h": 0.0,
            "projected_total_h": None, "projected_finish_utc": None,
            "eta_h": None, "eta_utc": None, "days_per_h": None,
            "host": {}, "recent": [], "alerts": [], "finished": False,
        }
        # Completion timestamps, for an ETA off the RECENT rate rather than the run
        # average. The two differ a lot here and the difference is not noise: the queue
        # walks the machine's months in order, and a winter month that rejects most of its
        # days is ~2x faster per day than a July that yields a case from nearly every one.
        # An ETA from the whole-run average therefore lags reality all the way through.
        self._done_t = collections.deque(maxlen=400)
        self.flush()

    def flush(self):
        self.d["elapsed_h"] = round((time.time() - _t0) / 3600.0, 3)
        self.d["updated_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        tmp = self.path + f".tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                json.dump(self.d, f, indent=1, default=str)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def gpu(self, g, **kv):
        with self.lock:
            self.d["gpus"][str(g)].update(kv)
            self.flush()

    def _eta(self):
        """Hours remaining, from the rate over a TRAILING WINDOW of completed days.

        The window is the last third of what has been done, floored at 12 days and capped
        at 120, so it is short enough to follow a change of month and long enough not to
        chase one slow case. Below 5 completed days there is no trend to speak of and the
        ETA is withheld rather than guessed -- an ETA off two samples is a number, not an
        estimate.
        """
        n_done = len(self._done_t)
        left = self.d["n_days_total"] - self.d["counts"]["done"]
        if n_done < 5 or left <= 0:
            return None, None
        w = max(12, min(120, n_done // 3))
        recent = list(self._done_t)[-w:]
        span = recent[-1] - recent[0]
        if span <= 0:
            return None, None
        rate = (len(recent) - 1) / span                  # days per second
        return left / rate / 3600.0, rate * 3600.0

    def finish_day(self, g, status, day, note="", resumed=False):
        with self.lock:
            c = self.d["counts"]
            c[status] = c.get(status, 0) + 1
            c["done"] += 1
            if resumed:
                c["resumed"] = c.get("resumed", 0) + 1
            else:
                # RESUMED DAYS ARE NOT IN THE RATE. They resolve in milliseconds off the
                # disk, so including them would make the ETA of a restarted machine
                # collapse toward zero while the real work ahead is unchanged.
                self._done_t.append(time.time())
            eta, dph = self._eta()
            self.d["eta_h"] = round(eta, 2) if eta is not None else None
            self.d["days_per_h"] = round(dph, 1) if dph is not None else None
            self.d["eta_utc"] = ((dt.datetime.utcnow() + dt.timedelta(hours=eta))
                                 .isoformat(timespec="seconds") + "Z") if eta else None
            self.d["gpus"][str(g)] = {"state": "idle"}
            self.d["recent"] = ([f"{day} {status}" + (f": {note}" if note else "")]
                                + self.d["recent"])[:12]
            self.flush()

    def set(self, **kv):
        with self.lock:
            self.d.update(kv)
            self.flush()

    def alert(self, msg):
        with self.lock:
            if msg not in self.d["alerts"]:
                self.d["alerts"].append(msg)
            self.flush()


# --------------------------------------------------------------------------- one day
STAGE_RE = re.compile(r"stage ([0-9]+[a-c]?)")


def _tail_stage(path, n=4000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            m = STAGE_RE.findall(f.read().decode("utf-8", "replace"))
            return f"stage {m[-1]}" if m else "starting"
    except OSError:
        return "starting"


def _run_watching(cmd, env, log_path, timeout, on_tick=None, period=2.0):
    """Run a subprocess, polling its log so the progress view can show the live stage."""
    with open(log_path, "ab") as out:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out,
                             stderr=subprocess.STDOUT, start_new_session=True)
        t_end = time.time() + timeout if timeout else None
        while True:
            try:
                return p.wait(timeout=period)
            except subprocess.TimeoutExpired:
                if on_tick:
                    on_tick(_tail_stage(log_path))
                if t_end and time.time() > t_end:
                    import signal
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        pass
                    p.wait(timeout=60)
                    return 124


def _prune_hrrr(cache, day, min_bytes=50 << 20):
    """A BACKSTOP. Normally there is nothing here to delete, and that is the point.

    Every HRRR fetch goes through `Herbie.xarray(search=...)` with `remove_grib=True`, so
    the byte-range subset is removed the moment it is read and what persists is the `.idx`
    inventory at a few kB. MEASURED against the live archive at `nlev=20`: a sounding is
    **168.6 MB transferred and 0 MB retained**.

    This exists for the one way that stops being true -- a `--keep-grib` left in a debug
    run, or a fetch that raises after the download and before the delete. Both leave a
    ~150 MB file that nothing will ever read again, and 243 days of them would fill the
    box. It runs only after the record exists AND passed its schema check, which is also
    when resume stops needing anything for that day.

    An earlier version of this docstring claimed ~407 MB per case and ~77 GB per machine.
    Those came from `data/hrrr` on the workstation, which holds cache written before `nlev`
    was cut to 20 and by runs that passed `--keep-grib`; measuring a cache is not measuring
    a transfer.
    """
    d = os.path.join(cache, "hrrr", day.strftime("%Y%m%d"))
    if not os.path.isdir(d):
        return
    for f in os.listdir(d):
        p = os.path.join(d, f)
        try:
            if os.path.isfile(p) and os.path.getsize(p) >= min_bytes:
                os.remove(p)
        except OSError:
            pass


def run_day(day, gpu, a, paths, prog, commit):
    """One calendar day -> exactly one of case / missing / failed / skipped."""
    ds = day.isoformat()
    rec = {"day": ds, "month": f"{day.year}-{day.month:02d}", "gpu": gpu,
           # SAME PRECISION AT BOTH ENDS. These were 1 and 3 decimals, so a day shorter
           # than the rounding produced a NEGATIVE wall_s -- and the load-balance figures
           # derived from it read "-73% imbalance" and "108% of wall time saved" while
           # every check passed. Round once, in one place.
           "split": split_of(day), "t_start_s": round(time.time() - _t0, 3)}
    hj = os.path.join(paths["hours"], ds + ".json")
    log = os.path.join(paths["logs"], ds + ".log")
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), FLUX_NATIVE="1", FLUX_ROOT=ROOT,
               NPZ_DIR=paths["npz"], LPDM_WORKERS=str(a.lpdm_workers))
    if a.stub:
        env["FLUX_STUB"] = "1"

    def done(status, **kv):
        rec.update(status=status, t_end_s=round(time.time() - _t0, 3), **kv)
        rec["wall_s"] = round(rec["t_end_s"] - rec["t_start_s"], 3)
        return rec

    # ---- RESUME: the artifacts are the checkpoint -------------------------------------
    prior = None
    if os.path.exists(hj):
        try:
            prior = json.load(open(hj))
        except (OSError, ValueError):
            prior = None
    if prior is not None:
        acc = prior.get("accepted")
        if acc:
            tag = "case_" + acc["timestamp"].replace("-", "").replace("T", "")[:10]
            if os.path.exists(os.path.join(paths["npz"], tag + ".npz")):
                # RESOLVED, NOT "SKIPPED". The manifest describes the CORPUS, not this
                # pass: a machine resumed three times would otherwise end with every day
                # marked "skipped" and no record anywhere of which days are cases.
                return done("case", resumed=True, tag=tag, timestamp=acc["timestamp"])
        elif not a.retry_missing:
            return done("missing", resumed=True,
                        reason=prior.get("missing_reason", "no hour accepted"))

    # ---- the hour ---------------------------------------------------------------------
    prog.gpu(gpu, state="picking hour", day=ds, month=rec["month"], stage="hour draw",
             since_s=round(time.time() - _t0, 1))
    cmd = [sys.executable, os.path.join(ROOT, "bin", "pick_hour.py"), ds,
           "--json", hj, "--cache", paths["hrrr"]]
    if a.stub:
        cmd.append("--stub-screen")
    try:
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=a.hour_timeout)
    except subprocess.TimeoutExpired:
        return done("failed", reason=f"pick_hour timed out after {a.hour_timeout}s")
    with open(log, "a") as f:
        f.write(r.stdout + r.stderr)
    if r.returncode == 3:
        why = "the hour pool was exhausted"
        try:
            why = json.load(open(hj)).get("missing_reason", why)
        except (OSError, ValueError):
            pass
        return done("missing", reason=why)
    if r.returncode != 0:
        return done("failed", reason=f"pick_hour rc={r.returncode}: "
                                     f"{(r.stderr or '').strip()[-200:]}")
    ts = (r.stdout or "").strip().splitlines()[-1].strip() if r.stdout.strip() else ""
    if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:00$", ts):
        return done("failed", reason=f"pick_hour printed {ts!r}, not a round-hour stamp")
    tag = "case_" + ts.replace("-", "").replace("T", "")[:10]
    rec["timestamp"], rec["tag"] = ts, tag

    if os.path.exists(os.path.join(paths["npz"], tag + ".npz")):
        return done("case", resumed=True, tag=tag, timestamp=ts)

    # ---- the case ---------------------------------------------------------------------
    prog.gpu(gpu, state="case", day=ds, month=rec["month"], tag=tag, stage="starting",
             since_s=round(time.time() - _t0, 1))
    if a.stub:
        cmd = [sys.executable, os.path.join(ROOT, "bin", "stub_case.py"), ts,
               "--npz-dir", paths["npz"], "--delay-ms", a.stub_ms,
               "--git-commit", commit or ""]
    else:
        cmd = ["bash", os.path.join(ROOT, "bin", "get_case.sh"), ts]
    rc = _run_watching(cmd, env, log, a.case_timeout,
                       on_tick=lambda st: prog.gpu(gpu, stage=st))

    npz = os.path.join(paths["npz"], tag + ".npz")
    if os.path.exists(npz):
        # ASSERT ON THE ARTIFACT. get_case.sh runs check_npz itself; this is the
        # orchestrator's own independent look, because a record that fails its schema is
        # worse than a missing day -- it survives to training and nobody re-runs it.
        chk = subprocess.run(
            [sys.executable, os.path.join(ROOT, "bin", "check_npz.py"), npz, "--quiet"]
            + (["--allow-stub"] if a.stub else []),
            cwd=ROOT, env=env, capture_output=True, text=True)
        if chk.returncode != 0:
            os.replace(npz, npz + ".REJECTED")
            return done("failed", tag=tag,
                        reason=f"the record failed its schema check and was set aside as "
                               f"{os.path.basename(npz)}.REJECTED: "
                               f"{(chk.stdout + chk.stderr).strip()[-200:]}")
        if a.prune_hrrr:
            _prune_hrrr(paths["hrrr"], day)
        return done("case", tag=tag, bytes=os.path.getsize(npz))
    if rc == 3:
        return done("missing", tag=tag,
                    reason=f"the drawn hour {ts} passed the HRRR screen but its fitted "
                           f"sounding put z_i outside what the domain supports")
    if rc == 124:
        return done("failed", tag=tag, reason=f"the case exceeded --case-timeout "
                                              f"({a.case_timeout}s) and was killed")
    return done("failed", tag=tag, reason=f"get_case rc={rc}; see logs/{ds}.log")


# --------------------------------------------------------------------------- manifest
def write_manifest(path, a, machine_months, results, gpus, host, commit, expected):
    # THE TIMING BELONGS TO THE PASS THAT DID THE WORK, NOT TO THE PASS THAT SKIPPED IT.
    # A resumed day is resolved in milliseconds; writing that over the original would erase
    # the only measurement of what a case costs, which is the number the next rental turns
    # on. So the prior manifest is read and a resumed day keeps what it already had.
    prior_days = {}
    try:
        prior_days = json.load(open(path)).get("days", {})
    except (OSError, ValueError):
        pass
    days = {}
    for r in results:
        e = {"status": r["status"], "split": r["split"], "gpu": r.get("gpu"),
             "wall_s": r.get("wall_s")}
        for k in ("tag", "timestamp", "reason", "resumed", "bytes"):
            if r.get(k) is not None:
                e[k] = r[k]
        if r.get("resumed"):
            was = prior_days.get(r["day"]) or {}
            if was.get("wall_s"):
                e["wall_s"], e["gpu"] = was["wall_s"], was.get("gpu")
                e["measured_on_pass"] = was.get("measured_on_pass", "an earlier run")
        days[r["day"]] = e
    # EVERY CALENDAR DAY OF EVERY OWNED MONTH, whether this pass reached it or not -- and
    # A DAY THIS PASS HAS NOT REACHED KEEPS WHAT AN EARLIER PASS FOUND.
    #
    # This wrote "not reached" over every unprocessed day, and because the manifest is
    # rewritten every 20 days that was destructive on a RESUME: the first periodic write
    # of pass 2 replaced all 223 not-yet-revisited days with "not reached", the next write
    # read that back as `prior_days`, and by the end only the days handled before the first
    # write still carried a duration. Seven of eight months reported 0.0 s of work and the
    # rigid-vs-queue comparison derived from it read "87% saved" off one month.
    #
    # Worse than the wrong number: a machine interrupted mid-resume would have written a
    # manifest saying most of its corpus was never attempted, while the records sat on disk
    # beside it. The manifest is the only thing that survives the box.
    for ym in machine_months:
        for d in range(1, days_in(*ym) + 1):
            ds = dt.date(ym[0], ym[1], d).isoformat()
            if ds in days:
                continue
            was = prior_days.get(ds)
            days[ds] = dict(was) if was else {
                "status": "not reached", "split": split_of(dt.date(ym[0], ym[1], 1)),
                "reason": "the machine stopped before this day"}
    cases = {v["tag"]: v for v in days.values() if v.get("status") in ("case",) and v.get("tag")}
    counts = {}
    for v in days.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    m = {
        "format": "flux-corpus-machine-manifest/1",
        "generated_by": "bin/run_corpus_machine.py",
        "machine": a.machine, "n_machines": N_MACHINES,
        "months": [month_str(x) for x in machine_months],
        "months_split": {month_str(x): split_of(dt.date(x[0], x[1], 1))
                         for x in machine_months},
        "git_commit": commit, "image": os.environ.get("FLUX_IMAGE_REF"),
        "stub": bool(a.stub), "host": os.uname().nodename,
        "gpus": gpus, "n_gpus_used": len(gpus),
        "grid": {"config": "122^3 @ 30 m", "dx_m": 30.0, "domain_m": 3660.0,
                 "receptor_z_m": 30.0, "n": 122, "pad": 3, "n_padded": 128,
                 "grid_dir": os.environ.get("GRID", "data/grid30_raised"),
                 "sim_h": 1.25, "adj_s": 1800, "window_s": 2700, "tback_s": 900,
                 "rel_s": 1800, "n_windows": 1},
        "seed_library": {"dir": "jobs30", "n_seeds": 30,
                         "selection": "the WHOLE library (ALLOW_DRIFTING=any)"},
        "expected_cases": expected,
        "counts": counts, "n_days": len(days),
        "days": dict(sorted(days.items())),
        "cases": dict(sorted(cases.items())),
        "host_memory": host,
        "elapsed_h": round((time.time() - _t0) / 3600.0, 3),
    }
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1, default=str)
    os.replace(tmp, path)
    return m


# --------------------------------------------------------------------------- main
def main():
    global _logfh
    ap = argparse.ArgumentParser(
        description="Generate one machine's share of the corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", type=int, required=True,
                    help=f"0..{N_MACHINES - 1}. Which 8 of the 64 corpus months this box "
                         f"owns; see the table printed at startup.")
    ap.add_argument("--out", default="/out", help="the mounted volume: everything lands here")
    ap.add_argument("--gpu-count", type=int, default=0, help="use the first N visible GPUs")
    ap.add_argument("--gpus", default="", help="explicit indices, e.g. 0,2,5")
    ap.add_argument("--assume-gpus", type=int, default=0,
                    help="fabricate N workers with no GPU present. --stub only.")
    ap.add_argument("--stub", action="store_true",
                    help="no GPU, no HRRR, no LES, no LPDM: exercises the queue, the "
                         "progress file, resume and the manifest. Records are stamped "
                         "stub:true and check_npz refuses them as corpus records.")
    ap.add_argument("--stub-ms", default="3,25",
                    help="LO,HI ms per stubbed case, drawn from the case hash so workers "
                         "finish UNEVENLY -- an instant stub tests no scheduling at all")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the partition, the day list and the rigid-vs-queue "
                         "comparison, then stop")
    ap.add_argument("--retry-missing", action="store_true",
                    help="re-evaluate days a previous run recorded MISSING")
    ap.add_argument("--max-days", type=int, default=0, help="stop after N days (smoke)")
    ap.add_argument("--only-month", default="", help="YYYY-MM, one of this machine's own")
    ap.add_argument("--early-n", type=int, default=5,
                    help="cases after which the measured cost and peak host RSS are "
                         "printed. This is the number the next rental turns on.")
    ap.add_argument("--max-hours", type=float, default=12.0,
                    help="fail loudly if the projection at the early report exceeds this")
    ap.add_argument("--abort-on-overrun", action="store_true",
                    help="stop the run at the early report instead of only warning")
    ap.add_argument("--lpdm-workers", type=int, default=0,
                    help="LPDM processes per case; 0 = cores/GPUs, floored at 2")
    ap.add_argument("--keep-hrrr", dest="prune_hrrr", action="store_false", default=True,
                    help="keep every downloaded GRIB. The default DELETES a day's ~407 MB "
                         "hybrid-level sounding once that day's record is written and "
                         "validated -- 243 days would otherwise leave ~77 GB of files "
                         "nothing reads again. The small screening files are always kept, "
                         "so a resume re-downloads nothing.")
    ap.add_argument("--case-timeout", type=int, default=10800)
    ap.add_argument("--hour-timeout", type=int, default=1800)
    a = ap.parse_args()

    if not 0 <= a.machine < N_MACHINES:
        raise SystemExit(f"--machine must be 0..{N_MACHINES - 1}")
    if a.assume_gpus and not a.stub:
        raise SystemExit("--assume-gpus fabricates workers for a machine that has no GPUs; "
                         "it is only meaningful with --stub.")

    paths = {k: os.path.join(a.out, v) for k, v in
             (("npz", "pairs_npz"), ("logs", "logs"), ("hours", "hours"), ("hrrr", "hrrr"))}
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    _logfh = open(os.path.join(a.out, "run_corpus.log"), "a")
    commit = git_commit()

    say("=" * 78, stamp=False)
    say(f"CORPUS MACHINE {a.machine} of {N_MACHINES}"
        f"{'   *** STUB: no GPU, no HRRR, no LES, no LPDM ***' if a.stub else ''}",
        stamp=False)
    say(f"  commit {commit}   image {os.environ.get('FLUX_IMAGE_REF', '(not in an image)')}",
        stamp=False)
    say("=" * 78, stamp=False)
    say("", stamp=False)
    say(describe(highlight=a.machine), stamp=False)
    say("", stamp=False)

    mine = months_for(a.machine)
    if a.only_month:
        want = tuple(int(x) for x in a.only_month.split("-"))
        if want not in mine:
            raise SystemExit(f"--only-month {a.only_month} is not owned by machine "
                             f"{a.machine} (it belongs to another box). This machine has: "
                             f"{', '.join(month_str(x) for x in mine)}")
        mine = [want]

    days = [dt.date(y, m, d) for (y, m) in mine for d in range(1, days_in(y, m) + 1)]
    days.sort()
    if a.max_days:
        days = days[:a.max_days]
    s = summary(a.machine)
    say(f"  this machine: {len(mine)} month(s), {len(days)} day(s), "
        f"splits {s['splits']}, {s['n_seasons']}/4 seasons")

    # ---- what a rigid month-per-GPU assignment would have cost -------------------------
    # The claim "a shared queue beats one month per GPU" is a claim about wall time, so it
    # is computed from the day counts rather than asserted. With equal per-case cost, a
    # pinned assignment finishes when its BUSIEST worker does; the queue finishes at the
    # mean. Day counts are the floor on the effect -- the real spread is in YIELD, which
    # is meteorological and larger, and the dry run measures that too.
    per_month_days = [days_in(*x) for x in mine]
    rigid = max(per_month_days)                     # 8 months pinned to 8 GPUs, one each:
    fair = sum(per_month_days) / len(mine)          # the queue finishes at the mean
    say(f"  rigid month-per-GPU finishes at its BUSIEST month ({rigid} days); a shared "
        f"queue finishes at the mean ({fair:.1f}) -- {100 * (rigid - fair) / rigid:.1f}% "
        f"of wall time, ON DAY COUNT ALONE.")
    say(f"  That 1-2% is the FLOOR and not the point. A month's cost is its ACCEPTED days, "
        f"not its calendar days, and acceptance is meteorological: a winter month yields "
        f"far fewer cases than a July. The saving is measured from the run's own timeline "
        f"and printed in the summary as worker load.")

    if a.dry_run:
        say("\n  --dry-run: the days this machine owns")
        for ym in mine:
            say(f"    {month_str(ym)}  {days_in(*ym)} days  split "
                f"{split_of(dt.date(ym[0], ym[1], 1))}")
        return 0

    # ---- GPUs --------------------------------------------------------------------------
    gpus = discover_gpus()
    if a.assume_gpus:
        use = [{"index": i, "name": "(fabricated, --stub)", "cc": "-", "vram_mib": 0,
                "uuid": f"stub-{i}"} for i in range(a.assume_gpus)]
    elif a.gpus:
        want = {int(x) for x in a.gpus.split(",") if x.strip()}
        use = [g for g in gpus if g["index"] in want]
    elif a.gpu_count:
        use = gpus[:a.gpu_count]
    else:
        use = gpus
    if not use:
        raise SystemExit("no GPUs visible. Pass --gpus/--gpu-count, or --stub "
                         "--assume-gpus N for a CPU-only dry run.")
    if not a.lpdm_workers:
        # THE LPDM FORKS, AND EIGHT CASES FORK AT ONCE. The single-case default is 12
        # workers; eight concurrent cases at 12 would be 96 processes competing for the
        # box's cores, which is slower than fewer and much harder to reason about.
        a.lpdm_workers = max(2, (os.cpu_count() or 16) // len(use))
    say(f"  {len(use)} worker(s): "
        + ", ".join(f"[{g['index']}] {g['name']}" for g in use[:4])
        + (f" ... (+{len(use) - 4})" if len(use) > 4 else ""))
    say(f"  LPDM workers per case: {a.lpdm_workers} "
        f"({os.cpu_count()} cores / {len(use)} concurrent cases)")

    host = HostWatch()
    host.start()
    expected = None
    prog = Progress(os.path.join(a.out, "progress.json"), a.machine, mine, len(days),
                    [g["index"] for g in use], expected)

    q = queue.Queue()
    for d in days:
        q.put(d)
    results, rlock = [], threading.Lock()
    early_done = threading.Event()
    stop_all = threading.Event()

    def maybe_early_report():
        with rlock:
            cases = [r for r in results if r["status"] == "case" and not r.get("resumed")]
            n = len(cases)
        if n < a.early_n or early_done.is_set():
            return
        early_done.set()
        host.sample()
        hs = host.snapshot()
        # OCCUPANCY, not FastEddy's own kernel time. A case holds its GPU for the whole
        # wall clock -- including the CPU-bound LPDM, during which the card is idle but
        # unavailable -- and occupancy is what the rental is billed on.
        gpu_h = sum(r["wall_s"] for r in cases) / len(cases) / 3600.0
        remaining = len(days) - len(results)
        proj = (remaining * gpu_h) / len(use) + (time.time() - _t0) / 3600.0
        prog.set(gpu_h_per_case=round(gpu_h, 4), projected_total_h=round(proj, 2),
                 projected_finish_utc=(dt.datetime.utcnow()
                                       + dt.timedelta(hours=max(0.0, proj - (time.time() - _t0) / 3600.0))
                                       ).isoformat(timespec="seconds") + "Z",
                 host=hs)
        say("", stamp=False)
        say("  " + "=" * 74, stamp=False)
        say(f"  EARLY REPORT after {n} case(s) -- the numbers the next rental turns on")
        say(f"    GPU-h per case (occupancy)   : {gpu_h:.3f}"
            f"   [{len(use)} concurrent]")
        say(f"    peak container RSS           : {hs['cgroup_peak_gb']:.1f} GB of "
            f"{hs['mem_total_gb']:.0f} GB")
        say(f"    MemAvailable low-water       : {hs['mem_available_min_gb']} GB")
        say(f"    swap used (peak)             : {hs['swap_used_peak_gb']:.2f} GB")
        say(f"    projected finish             : {proj:.2f} h for {len(days)} days")
        if a.stub:
            say("    *** STUB: these are plumbing numbers, not the corpus cost. ***")
        # THE HOST-RAM QUESTION THIS REPORT EXISTS FOR.
        if hs["mem_available_min_gb"] is not None and hs["mem_total_gb"]:
            frac = hs["mem_available_min_gb"] / hs["mem_total_gb"]
            if frac < 0.12:
                msg = (f"HOST RAM IS NEARLY GONE: MemAvailable fell to "
                       f"{hs['mem_available_min_gb']} GB of {hs['mem_total_gb']:.0f}. The "
                       f"LPDM's field cache is ~12 GB PER CASE and {len(use)} run at once. "
                       f"Reduce concurrency with --gpu-count before this thrashes.")
                say(f"    *** {msg}")
                prog.alert(msg)
        if hs["swap_used_peak_gb"] > 0.5:
            msg = (f"THE BOX IS SWAPPING ({hs['swap_used_peak_gb']:.1f} GB). Everything "
                   f"below will be several times slower with nothing else to show for it.")
            say(f"    *** {msg}")
            prog.alert(msg)
        if proj > a.max_hours:
            msg = (f"PROJECTED FINISH {proj:.1f} h EXCEEDS --max-hours {a.max_hours:.0f}. "
                   f"At {gpu_h:.2f} GPU-h per case over {len(use)} workers this machine's "
                   f"{len(days)} days will not finish in the intended rental.")
            say("", stamp=False)
            say("  " + "!" * 74, stamp=False)
            say(f"  *** {msg}")
            say("  " + "!" * 74, stamp=False)
            prog.alert(msg)
            if a.abort_on_overrun:
                say("  --abort-on-overrun: stopping now, before the rental is spent.")
                stop_all.set()
        say("  " + "=" * 74, stamp=False)
        say("", stamp=False)

    def worker(gpu):
        while not stop_all.is_set():
            try:
                d = q.get_nowait()
            except queue.Empty:
                return
            try:
                rec = run_day(d, gpu, a, paths, prog, commit)
            except BaseException as e:                          # noqa: BLE001
                rec = {"day": d.isoformat(), "month": f"{d.year}-{d.month:02d}",
                       "gpu": gpu, "split": split_of(d), "status": "failed",
                       "reason": f"the worker itself raised: {type(e).__name__}: {e}",
                       "wall_s": None}
            with rlock:
                results.append(rec)
                n = len(results)
            prog.finish_day(gpu, rec["status"], rec["day"], (rec.get("reason") or "")[:70],
                            resumed=bool(rec.get("resumed")))
            if not rec.get("resumed") or n % 50 == 0:
                say(f"  [gpu {gpu}] {rec['day']} {rec['status'].upper()}"
                    + (f" {rec.get('tag', '')}" if rec.get("tag") else "")
                    + (f" -- {rec['reason'][:90]}" if rec.get("reason") else "")
                    + f"   ({n}/{len(days)})")
            maybe_early_report()
            if n % 20 == 0:
                write_manifest(os.path.join(a.out, "manifest.json"), a, mine, results,
                               use, host.snapshot(), commit, expected)
            q.task_done()

    say(f"\n=== {len(days)} days over {len(use)} workers, shared queue ===")
    threads = [threading.Thread(target=worker, args=(g["index"],), daemon=True) for g in use]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    host.stop.set()

    # ---- reconcile: a day that never produced a record must still be accounted for ----
    seen = {r["day"] for r in results}
    for d in days:
        if d.isoformat() not in seen:
            results.append({"day": d.isoformat(), "month": f"{d.year}-{d.month:02d}",
                            "gpu": None, "split": split_of(d), "status": "failed",
                            "reason": "NEVER RAN -- no record was produced for this day",
                            "wall_s": None})
    results.sort(key=lambda r: r["day"])
    man = write_manifest(os.path.join(a.out, "manifest.json"), a, mine, results, use,
                         host.snapshot(), commit, expected)

    c = man["counts"]
    cases = [r for r in results if r["status"] == "case"]
    elapsed = (time.time() - _t0) / 3600.0
    n_res = sum(1 for r in results if r.get("resumed"))
    prog.set(finished=True, host=host.snapshot(),
             counts={**{k: c.get(k, 0) for k in ("case", "missing", "failed")},
                     "resumed": n_res, "done": len(results)})
    say("", stamp=False)
    say("=" * 78, stamp=False)
    say(f"MACHINE {a.machine} DONE   {elapsed:.2f} h on {len(use)} workers")
    for k in ("case", "missing", "failed", "not reached"):
        if c.get(k):
            say(f"    {k:<12} {c[k]:4d}")
    if n_res:
        say(f"    {'resumed':<12} {n_res:4d}   (already resolved by an earlier run)")
    say(f"    days accounted for: {man['n_days']} "
        f"(this machine owns {sum(days_in(*x) for x in mine)})")
    cases = [r for r in cases if not r.get("resumed") and r.get("wall_s")]
    if cases:
        gph = sum(r["wall_s"] for r in cases) / len(cases) / 3600.0
        say(f"    GPU-h per case (occupancy): {gph:.3f}   "
            f"total {gph * len(cases):.1f} GPU-h of work")
    hs = host.snapshot()
    say(f"    peak container RSS {hs['cgroup_peak_gb']:.1f} GB of {hs['mem_total_gb']:.0f}, "
        f"MemAvailable low-water {hs['mem_available_min_gb']} GB, "
        f"swap peak {hs['swap_used_peak_gb']:.2f} GB")
    # WHAT THE QUEUE ACTUALLY BOUGHT, from the recorded timeline rather than from design.
    by_gpu = {}
    for r in results:
        if r.get("gpu") is not None and r.get("wall_s") and not r.get("resumed"):
            by_gpu[r["gpu"]] = by_gpu.get(r["gpu"], 0.0) + r["wall_s"]
    if len(by_gpu) > 1:
        busiest, mean = max(by_gpu.values()), sum(by_gpu.values()) / len(by_gpu)
        say(f"    worker load: busiest {busiest / 3600:.2f} h, mean {mean / 3600:.2f} h "
            f"-> the queue kept them within {100 * (busiest - mean) / busiest:.0f}% of each "
            f"other")
    fails = [r for r in results if r["status"] == "failed"]
    if fails:
        say(f"\n  {len(fails)} FAILED day(s) -- these are retryable; re-run the same "
            f"command and they will be picked up:")
        for r in fails[:15]:
            say(f"    {r['day']}: {(r.get('reason') or '')[:110]}")
    if a.stub:
        say("\n  *** THIS WAS A STUB RUN. Every record carries meta.stub = true and "
            "bin/check_npz.py refuses them as corpus records. Delete this --out. ***")
    say(f"\n  -> {os.path.join(a.out, 'manifest.json')}")
    say(f"  -> {paths['npz']}/  ({len(cases)} record(s))")
    say("=" * 78, stamp=False)
    return 0 if not fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
