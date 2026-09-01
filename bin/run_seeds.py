#!/usr/bin/env python3
"""Generate the whole seed library on one multi-GPU machine, with ONE command.

    docker run --gpus all -v /out:/out <image> run_seeds --gpu-count 16

WHAT THIS IS FOR. The seed library is 30 flat spin-ups -- 5 rungs x 6 base angles -- whose
only purpose is to delete each corpus case's spin-up: ~29 GPU-h that buys back ~900. They
are embarrassingly parallel and share nothing, so they are exactly the thing to rent 16
GPUs for. What did not exist until now is anything that fans them out: jobs/run_seed.sh
runs ONE seed, on whatever GPU the driver happens to expose, and exits.

30 SEEDS OVER 16 GPUs IS HANDLED INTERNALLY, WITH A WORK QUEUE, AND NOT WITH TWO PASSES.
A pass model would idle 15 cards through the tail of pass 1 waiting for the slowest seed,
and the spread here is real: jobs/seed_watch.sh stops a seed the moment its
oscillation-immune limits enter band, and a convective rung turns over on z_i/w* ~ 350 s
against a neutral rung's h/u* ~ 1500 s. A queue starts the 17th seed on whichever card
finishes first. `--pass` is still accepted, and means something different and useful:
splitting the library across SEVERAL MACHINES.

A FAILED SEED NEVER ABORTS THE MACHINE. Each seed runs in its own subprocess with its own
working directory on the mounted volume; a crash, a gate failure and a timeout are all
recorded with a reason and the queue moves on. That is the same discipline
bin/run_month.sh already applies to corpus cases.

THE GPU IS CHOSEN WITH CUDA_VISIBLE_DEVICES AND NOTHING ELSE -- see docker/run_case.sh for
why that is sufficient, and for the measurement behind it.

WHAT LANDS IN THE MOUNTED OUTPUT
    <out>/seeds/<job>/            the seed's return/: restart, gate JSON, acceptance, logs
    <out>/work/<job>/             its working directory, dumps included, left for inspection
    <out>/machine_manifest.json   every seed, its verdict, its cost, its stop time
    <out>/threadblock_sweep.json  the block shape measured ON THIS GPU, and the runners-up
    <out>/run_seeds.log           this script's own output
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.environ.get("FLUX_ROOT", "/flux")
FE_BIN = os.path.join(ROOT, "FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy")
PROV = os.path.join(ROOT, "IMAGE_PROVENANCE.txt")
TOTAL_TIME = re.compile(r"^\s*([0-9.]+)\s*\|\s*(\d+)\s*\|", re.M)

_print_lock = threading.Lock()
_logfh = None


_t0 = time.time()


def say(*a, stamp=True):
    """Print and log. TIMESTAMPED, because the log is the evidence for the schedule.

    "a freed GPU picks up the next job" is a claim about WHEN things happened, and an
    unstamped log cannot support it. Seconds since this invocation started, so the
    timeline is readable without correlating wall-clock across a 16-way run.
    """
    with _print_lock:
        body = " ".join(str(x) for x in a)
        msg = f"[{time.time() - _t0:8.1f}s] {body}" if stamp else body
        print(msg, flush=True)
        if _logfh:
            _logfh.write(msg + "\n")
            _logfh.flush()


# --------------------------------------------------------------------------- GPUs
def nvsmi(query, extra=()):
    try:
        r = subprocess.run(["nvidia-smi", f"--query-{query[0]}={query[1]}",
                            "--format=csv,noheader,nounits", *extra],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return []
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def discover_gpus():
    idx = nvsmi(("gpu", "index,name,compute_cap,memory.total,uuid"))
    out = []
    for ln in idx:
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 5:
            continue
        out.append({"index": int(p[0]), "name": p[1], "cc": p[2],
                    "vram_mib": int(float(p[3])), "uuid": p[4]})
    return out


def binary_sass():
    """The architectures the FastEddy binary ACTUALLY carries. Asserted, never assumed."""
    try:
        r = subprocess.run(["cuobjdump", "--list-elf", FE_BIN],
                           capture_output=True, text=True, timeout=120)
        return sorted({m for m in re.findall(r"\.(sm_\d+)\.cubin", r.stdout)})
    except Exception:
        return []


class VramWatch(threading.Thread):
    """Peak per-GPU memory, sampled from the compute-apps table.

    memory.used on the DEVICE is not the right number: it includes anything else on the
    card (a display server holds ~900 MB on the development workstation) and would
    over-report the seed. The compute-apps table attributes bytes to processes, so what is
    reported is what FastEddy actually took.
    """

    def __init__(self, uuid_to_index, period=5.0):
        super().__init__(daemon=True)
        self.map, self.period, self.stop = uuid_to_index, period, threading.Event()
        self.peak = {i: 0 for i in uuid_to_index.values()}
        self.peak_dev = {i: 0 for i in uuid_to_index.values()}

    def run(self):
        while not self.stop.is_set():
            for ln in nvsmi(("compute-apps", "gpu_uuid,used_memory")):
                p = [x.strip() for x in ln.split(",")]
                if len(p) == 2 and p[0] in self.map:
                    i = self.map[p[0]]
                    try:
                        self.peak[i] = max(self.peak[i], int(float(p[1])))
                    except ValueError:
                        pass
            for ln in nvsmi(("gpu", "index,memory.used")):
                p = [x.strip() for x in ln.split(",")]
                if len(p) == 2 and int(p[0]) in self.peak_dev:
                    self.peak_dev[int(p[0])] = max(self.peak_dev[int(p[0])], int(float(p[1])))
            self.stop.wait(self.period)


class HostWatch(threading.Thread):
    """Peak HOST memory for the whole machine, and for FastEddy alone.

    WHY IT IS HERE. The corpus rental is the expensive one, and its sizing question is
    "how much system RAM does N-way need". PROJECT_BRIEF.md's 12.45 GB figure is the peak host
    RSS of a CORPUS CASE -- the LPDM's 12.0 GB fp16 field cache, which `compute_footprint`
    random-accesses -- and a SEED runs no LPDM at all. So the two numbers are different by
    construction and quoting the case number for a seed run would over-size the box by an
    order of magnitude. Both are reported, labelled, and neither is inferred from the other.

    Three quantities, because they answer different questions:
      cgroup_peak   what the CONTAINER used, which is what an instance has to be sized for
      fe_rss_sum    the sum over live FastEddy processes: how N-way scales
      mem_avail_min the low-water mark of MemAvailable: how close the box came to swapping
    """

    CG = ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory.current",
          "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
          "/sys/fs/cgroup/memory/memory.usage_in_bytes")

    def __init__(self, period=5.0):
        super().__init__(daemon=True)
        self.period, self.stop = period, threading.Event()
        self.cgroup_peak = 0
        self.fe_rss_peak = 0
        self.fe_rss_peak_n = 0
        self.mem_avail_min = None
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

    @staticmethod
    def _fe_rss():
        """Summed RSS of every live FastEddy, and how many there were."""
        tot = n = 0
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/cmdline", "rb") as f:
                    if b"FEMAIN/FastEddy" not in f.read():
                        continue
                for ln in open(f"/proc/{d}/status"):
                    if ln.startswith("VmRSS:"):
                        tot += int(ln.split()[1]) * 1024
                        n += 1
                        break
            except Exception:
                continue
        return tot, n

    def run(self):
        while not self.stop.is_set():
            self.cgroup_peak = max(self.cgroup_peak, self._cgroup())
            rss, n = self._fe_rss()
            if rss > self.fe_rss_peak:
                self.fe_rss_peak, self.fe_rss_peak_n = rss, n
            av = self._meminfo("MemAvailable")
            if av is not None:
                self.mem_avail_min = av if self.mem_avail_min is None else min(self.mem_avail_min, av)
            self.stop.wait(self.period)


# --------------------------------------------------------------------------- .in edits
def rewrite_in(path, **kv):
    seen = set()
    lines = []
    for ln in open(path):
        k = ln.split("=", 1)[0].strip() if "=" in ln else None
        if k in kv:
            # Keep the trailing comment: it is where this project records WHY a value is
            # what it is, and dropping it would quietly erase the reasoning.
            cm = ln.split("#", 1)[1].rstrip() if "#" in ln else ""
            lines.append(f"{k} = {kv[k]}" + (f"  #{cm}\n" if cm else "\n"))
            seen.add(k)
        else:
            lines.append(ln)
    missing = set(kv) - seen
    if missing:
        raise RuntimeError(f"{path} has no line for {sorted(missing)}")
    open(path, "w").write("".join(lines))


# --------------------------------------------------------------------------- one seed
def _kill_tree(pid, wd):
    """Kill a subprocess AND the setsid'd FastEddy it may have left holding the GPU."""
    import signal
    for target in (pid,):
        try:
            os.killpg(os.getpgid(target), signal.SIGKILL)
        except Exception:
            try:
                os.kill(target, signal.SIGKILL)
            except Exception:
                pass
    # docker/run_case.sh writes the LES's process-GROUP leader here, precisely so it can be
    # reached from outside. Without this the orphan keeps the card and the per-GPU mutex
    # refuses every later seed routed to it.
    try:
        fp = os.path.join(wd, ".fe.pid")
        if os.path.isfile(fp):
            fe = int(open(fp).read().strip())
            try:
                os.killpg(fe, signal.SIGKILL)
            except Exception:
                os.kill(fe, signal.SIGKILL)
            os.remove(fp)
    except Exception:
        pass


def _run_group(cmd, cwd, env, out, timeout, wd):
    """subprocess.run with a timeout that reaches the whole process group."""
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(p.pid, wd)
        try:
            p.wait(timeout=30)
        except Exception:
            pass
        raise


def fasteddy_gpu_seconds(log_path):
    """GPU-resident seconds, summed from FastEddy's own per-batch TIMESTEP PERFORMANCE.

    Not the wall clock of the job: that includes the stationarity gate, the acceptance
    battery and the restart copy, all of which are CPU. Quoting wall as GPU-h would
    over-price the library by whatever the analysis took.
    """
    # ZERO-STEP BLOCKS EXCLUDED. FastEddy emits a final TIMESTEP PERFORMANCE block with
    # `Batch Steps = 0` for the shutdown; it carries ~0.17 s of teardown that is not
    # integration. Small here, and excluded for the same reason it had to be excluded in
    # bin/threadblock_sweep.py, where reading it instead of the run made every block shape
    # score identically.
    try:
        return sum(float(m[0]) for m in TOTAL_TIME.findall(open(log_path).read())
                   if int(m[1]) > 0)
    except Exception:
        return None


def newest_step(d, base):
    best = None
    try:
        for f in os.listdir(d):
            if f.startswith(base + "."):
                try:
                    s = int(f.rsplit(".", 1)[1])
                except ValueError:
                    continue
                best = s if best is None else max(best, s)
    except OSError:
        pass
    return best


def gate_state(js):
    """DRIFTING / INDETERMINATE / PASS, from the gate's own arrays.

    `pass` alone is not the label to carry: docs/PLAN.md records that TKE_BL/u*^2 and z_i cannot
    be resolved against their thresholds at ANY window width in an affordable spin-up, so
    almost every seed in this library is legitimately INDETERMINATE and `pass` is False.
    INDETERMINATE and DRIFTING are different verdicts and pick_seed.py treats them
    differently -- it admits the first under a flag and refuses the second outright -- so
    collapsing them into "failed" would throw the library away.
    """
    if not js:
        return "NO-GATE", []
    drift = js.get("drifting") or [r["name"] for r in js.get("gated", []) if r.get("ok") is False]
    indet = js.get("indeterminate") or [r["name"] for r in js.get("gated", []) if r.get("ok") is None]
    if drift:
        return "DRIFTING", drift
    if indet:
        return "INDETERMINATE", indet
    return "PASS", []


def run_one(job, gpu, a, workroot, outroot, tb):
    """One seed, start to finish, on one GPU. Never raises: the caller must not be aborted."""
    name = job["job"]
    rec = {"job": name, "rung": job.get("rung"), "regime": job.get("regime"),
           "base_angle_deg": job.get("base_angle_deg"), "gpu": gpu,
           "status": "unknown", "reason": "", "gate_state": None, "gate_limits": [],
           "wall_s": None, "gpu_s": None, "sim_h": None, "stop_step": None,
           "ceiling_sim_h": a.ceiling_h, "accepted": False,
           # THE TIMELINE, IN SECONDS SINCE THE RUN STARTED. This is what makes "a freed
           # GPU picked up the next job" a checkable statement rather than a design
           # intention: bin/test_work_queue.sh reads these back and asserts that some
           # worker's second job STARTED after its first one ENDED, and that the number of
           # jobs overlapping in time never exceeded the worker count.
           "t_start_s": None, "t_end_s": None}
    t0 = time.time()
    rec["t_start_s"] = round(t0 - _t0, 2)
    wd = os.path.join(workroot, name)
    try:
        src = os.path.join(a.jobs_dir, name)
        # A STALE WORKING DIRECTORY IS THE ORCHESTRATOR'S OWN, AND IT HAS TO DECIDE.
        # jobs/run_seed.sh refuses a PARTIAL run outright rather than wiping it -- correct,
        # because on a workstation that directory may be hours of GPU a person staged and
        # wants to look at. Here it cannot be: <out>/work/<job> is scratch this script
        # created, so a partial run in it is an interrupted attempt of THIS command and
        # nothing else. Refusing would make a rented box that lost power unresumable, and
        # wiping unconditionally would throw away a finished seed on a re-run. So:
        #   finished (seed_restart.nc present)  -> skip, and say so. The machine is resumable.
        #   partial                              -> restart from step 0 with --restart-over.
        done_restart = os.path.join(wd, "return", "seed_restart.nc")
        done_accept = os.path.join(wd, "return", "acceptance.txt")
        restart_over = False
        # "COMPLETE" MEANS THE DELIVERABLE IS COMPLETE, NOT THAT THE LES FINISHED.
        # jobs/run_seed.sh writes seed_restart.nc and THEN this script runs the acceptance
        # battery and copies return/ to the mount -- so a box interrupted between those two
        # leaves a restart on disk with no battery, no Gate C2, no rotation check and
        # nothing in <out>/seeds. Skipping on the restart alone would report that seed
        # "already complete", count it as accepted whatever its gate said, and leave the
        # operator with no artifact at all. The skip therefore requires the battery's own
        # output too (unless --skip-accept asked for no battery).
        complete = (os.path.isfile(done_restart) and os.path.getsize(done_restart) > 0
                    and (a.skip_accept
                         or (os.path.isfile(done_accept) and os.path.getsize(done_accept) > 0)))
        if complete and not a.force:
            js = None
            stat = os.path.join(wd, "return", "stationarity.json")
            if os.path.isfile(stat):
                try:
                    js = json.load(open(stat))
                except Exception:
                    js = None
            rec["gate_state"], rec["gate_limits"] = gate_state(js)
            # A STUB ON DISK IS NOT A COMPLETE SEED, AND THE RESUME PATH HAD TO BE TOLD.
            # MEASURED: re-running the scheduler test over an /out that already held its
            # own stub output reported "18 accepted" -- the fresh path excludes a stub
            # explicitly, the skip path derived `accepted` from the gate state alone, and
            # the gate state of a stub is whatever its fabricated JSON says. That is
            # precisely the failure PROJECT_BRIEF.md forbids: a stubbed record masquerading as a
            # real one. The stub flag travels in the artifact, so read it from there.
            stub_on_disk = bool(js and js.get("stub"))
            if not stub_on_disk:
                try:
                    stub_on_disk = bool(json.load(
                        open(os.path.join(wd, "return", "manifest.json"))).get("stub"))
                except Exception:
                    pass
            rec["stub_on_disk"] = stub_on_disk
            rec.update(status="skipped", reason="already complete in the work directory",
                       accepted=(not a.stub and not stub_on_disk
                                 and rec["gate_state"] != "DRIFTING"))
            if stub_on_disk and not a.stub:
                # Refusing to SKIP it, not merely refusing to accept it: leaving a stub in
                # place would leave a 1 kB text file where the corpus expects a 73 MB
                # restart, and every later resume would skip it again.
                say(f"  [gpu {gpu}] {name}: the work directory holds a STUB, not a seed; "
                    f"re-running it for real")
                shutil.rmtree(os.path.join(wd, "output"), ignore_errors=True)
                shutil.rmtree(os.path.join(wd, "return"), ignore_errors=True)
                rec.update(status="unknown", reason="", accepted=False)
                stub_on_disk = False
            else:
                man = json.load(open(os.path.join(wd, "manifest.json")))
                step = newest_step(os.path.join(wd, "output"), man["run"]["outFileBase"])
                rec["stop_step"] = step
                rec["sim_h"] = round(step * float(man["run"]["dt"]) / 3600.0, 4) if step else None
                rec["gpu_s_run"] = fasteddy_gpu_seconds(os.path.join(wd, "return", "run.log"))
                rec["gpu_s_accel"] = fasteddy_gpu_seconds(os.path.join(wd, "return", "accel.log"))
                rec["gpu_s"] = (rec["gpu_s_run"] or 0.0) + (rec["gpu_s_accel"] or 0.0) or None
                rec["early_stopped"] = os.path.isfile(os.path.join(wd, "output", ".early_stop"))
                # AND THE COPY HAPPENS ANYWAY. The skip is about not re-spending GPU-hours,
                # not about withholding the artifact -- a resumed run must leave the same
                # <out>/seeds a first run would.
                dst = os.path.join(outroot, "seeds", name)
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(os.path.join(wd, "return")):
                    shutil.copy2(os.path.join(wd, "return", f), os.path.join(dst, f))
                rec["out_dir"] = dst
                say(f"  [gpu {gpu}] {name}: already complete in {wd}; skipping the GPU work "
                    f"(gate {rec['gate_state']}, --force to redo)")
                return rec
        if os.path.isfile(done_restart) and not a.force:
            say(f"  [gpu {gpu}] {name}: a restart is on disk but the acceptance battery is "
                f"not; re-running the seed from step 0")
        if os.path.isdir(os.path.join(wd, "output")) and os.listdir(os.path.join(wd, "output")):
            say(f"  [gpu {gpu}] {name}: a partial run is in {wd}; restarting it from step 0")
            restart_over = True
            # THE WHOLE OUTPUT DIRECTORY GOES, not just the FE_SEED.* family that
            # run_seed.sh's own --restart-over removes. A neutral rung's Steinfeld burn-in
            # writes FE_SEED_ACC.* and stages FE_ACC.0, and `rm -f "$OUTBASE".*` does not
            # match either -- while bin/seed_report.py globs `output/*.[0-9]*` with no
            # family filter and would fold the burn-in's dumps into the run's series.
            # ONE RUN PER DIRECTORY, OR IT IS NOT A SERIES (docs/FASTEDDY_TRAPS.md 18c).
            shutil.rmtree(os.path.join(wd, "output"), ignore_errors=True)
            shutil.rmtree(os.path.join(wd, "return"), ignore_errors=True)
            for stale in ("FE_ACC.0", "accel.in", "run.in"):
                try:
                    os.remove(os.path.join(wd, stale))
                except OSError:
                    pass
            # --restart-over is still passed below, as belt and braces: with the
            # directory already empty it is a no-op in run_seed.sh (its own discard is
            # guarded on `ls` finding something), and if the rmtree above only partly
            # succeeded it is what stops the run being REFUSED as partial.
        os.makedirs(os.path.join(wd, "output"), exist_ok=True)
        os.makedirs(os.path.join(wd, "return"), exist_ok=True)
        for f in ("manifest.json", "seed.in"):
            if not os.path.isfile(os.path.join(src, f)):
                rec.update(status="failed", reason=f"job template has no {f}")
                return rec
            shutil.copy2(os.path.join(src, f), os.path.join(wd, f))
        if tb:
            rewrite_in(os.path.join(wd, "seed.in"),
                       tBx=tb["tBx"], tBy=tb["tBy"], tBz=tb["tBz"])

        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            FLUX_NATIVE="1", FLUX_ROOT=ROOT, FE_BIN=FE_BIN,
            SEED_CEILING_H=str(a.ceiling_h),
            SEED_EARLY_STOP="1" if a.early_stop else "0",
            ALLOW_INDETERMINATE="1",
            # 16 concurrent jobs: uncapped, numpy alone opens one thread per core in each.
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1", LPDM_WORKERS=str(a.lpdm_workers),
            # Per-seed scratch for the two acceptance steps that would otherwise share a
            # fixed directory across all 16 workers. See bin/c2_restart_check.sh.
            C2_ROOT=os.path.join(workroot, "_scratch"),
            ROTCHK_ROOT=os.path.join(workroot, "_scratch"),
        )
        # THE STEINFELD ACCELERATOR IS A NEUTRAL-RUNG DEVICE, and it is applied by REGIME
        # rather than by rung name. A neutral boundary layer has no buoyant production to
        # organise the cold-start perturbation field, so it is the slowest regime to reach
        # resolved turbulence -- h/u* ~ 1500 s against T* ~ 350 s convectively -- and 3000 s
        # at surflayer_wth = +0.05 K m/s trips the transition. Convective rungs do not need
        # it and would have their own flux overwritten by it.
        if job.get("regime") in a.accel_regimes:
            env["SEED_ACCEL_S"] = str(a.accel_s)
            env["SEED_ACCEL_WTH"] = str(a.accel_wth)
        if a.stub:
            env["STUB_SEED"] = "1"
            env["STUB_SEED_S"] = str(a.stub_seconds)
            env["STUB_SEED_FAIL"] = "1" if name in a.stub_fail_set else "0"
            rec["stub"] = True

        say(f"  [gpu {gpu}] {name}: starting ({job.get('regime')}, rung {job.get('rung')}"
            f"{', accelerator ' + env['SEED_ACCEL_S'] + ' s' if 'SEED_ACCEL_S' in env else ''})")
        with open(os.path.join(wd, "run_seed.stdout"), "w") as f:
            # start_new_session, SO A TIMEOUT CAN ACTUALLY REACH THE RUN.
            # subprocess's own timeout kills only the direct child, jobs/run_seed.sh --
            # and docker/run_case.sh deliberately launches the LES under `setsid`, in its
            # OWN session, so mpirun and FastEddy survive. On a 16-GPU box that orphan then
            # holds the card: the next seed's per-GPU mutex sees a FastEddy with the same
            # CUDA_VISIBLE_DEVICES, refuses, and EVERY subsequent seed routed to that GPU
            # fails for a reason that has nothing to do with it. One wedged seed would take
            # a sixteenth of the machine out for the rest of the run.
            rc = _run_group([os.path.join(ROOT, "jobs/run_seed.sh"), wd]
                            + (["--restart-over"] if restart_over else []),
                            cwd=ROOT, env=env, out=f, timeout=a.job_timeout, wd=wd)
        rec["run_seed_rc"] = rc

        # ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS. run_seed.sh exits 1 when the
        # stationarity GATE fails, which is a verdict about the boundary layer and not a
        # failure of the machine -- so the exit code cannot be read as "did this work".
        restart = os.path.join(wd, "return", "seed_restart.nc")
        stat = os.path.join(wd, "return", "stationarity.json")
        js = None
        if os.path.isfile(stat):
            try:
                js = json.load(open(stat))
            except Exception:
                js = None
        rec["gate_state"], rec["gate_limits"] = gate_state(js)

        if not os.path.isfile(restart) or os.path.getsize(restart) == 0:
            tail = ""
            try:
                tail = open(os.path.join(wd, "run_seed.stdout")).read()[-600:]
            except Exception:
                pass
            rec.update(status="failed",
                       reason=f"no seed_restart.nc (rc={rc}) :: {tail.strip()[-400:]}")
            return rec

        man = json.load(open(os.path.join(wd, "manifest.json")))
        dt = float(man["run"]["dt"])
        base = man["run"]["outFileBase"]
        step = newest_step(os.path.join(wd, "output"), base)
        rec["stop_step"] = step
        rec["sim_h"] = round(step * dt / 3600.0, 4) if step else None
        # BOTH LOGS. A neutral rung runs the Steinfeld accelerator FIRST -- 3000 s of
        # burn-in at surflayer_wth = +0.05, which this script enables for every neutral
        # seed -- and jobs/run_seed.sh writes it to return/accel.log, a SEPARATE file.
        # MEASURED on jobs30/seed_nbl-deep_a015: accel.log carries 1451.7 GPU-seconds
        # (0.403 GPU-h) against run.log's 5042.7. Counting only run.log under-reports a
        # neutral seed by ~29% and the 30-seed library by ~4.8 GPU-h -- and "measured
        # GPU-h per seed" is one of the numbers this whole run exists to produce.
        rec["gpu_s_run"] = fasteddy_gpu_seconds(os.path.join(wd, "return", "run.log"))
        rec["gpu_s_accel"] = fasteddy_gpu_seconds(os.path.join(wd, "return", "accel.log"))
        rec["gpu_s"] = (rec["gpu_s_run"] or 0.0) + (rec["gpu_s_accel"] or 0.0) or None
        rec["early_stopped"] = os.path.isfile(os.path.join(wd, "output", ".early_stop"))

        # ---- the acceptance battery. GPU is still ours: step 6 (Gate C2) needs it. ----
        say(f"  [gpu {gpu}] {name}: LES done at {rec['sim_h']} sim-h"
            f"{' (early stop)' if rec['early_stopped'] else ''}, gate {rec['gate_state']}"
            f" -- running the acceptance battery")
        if not a.skip_accept:
            wall_arg = ["--wall-seconds", str(int(time.time() - t0))]
            with open(os.path.join(wd, "seed_accept.stdout"), "w") as f:
                rec["accept_rc"] = _run_group(
                    [os.path.join(ROOT, "bin/seed_accept.sh"), wd, *wall_arg],
                    cwd=ROOT, env=env, out=f, timeout=a.accept_timeout, wd=wd)
            acc = os.path.join(wd, "return", "acceptance.txt")
            rec["acceptance"] = os.path.isfile(acc) and os.path.getsize(acc) > 0
            if rec["acceptance"]:
                txt = open(acc, errors="replace").read()
                rec["turb_alive"] = ("VERDICT: OK" in txt) or ("turb-alive OK" in txt)
                rec["c2"] = "GATE C2: PASS" in txt
                rec["battery_fail_lines"] = [l.strip() for l in txt.splitlines()
                                             if "*** FAIL" in l or "NO VERDICT" in l][:6]
                # k0/k1 IS SURFACED, NOT ONLY ENFORCED. It already fails a run loudly --
                # docker/k0k1_check.py exits 1, check_run.sh sets fail, run_case.sh returns
                # nonzero and run_seed.sh dies -- so a bad dt cannot produce a seed. But
                # the standing question on a NEW architecture is whether the CFL accuracy
                # boundary carried, and "no seed failed" is a weaker answer than the
                # numbers. ~0.27 is correct here; ~9 means dt is past the boundary.
                m = re.search(r"k0/k1 (OK|FAIL|SKIP)[^\n(]*\(?([0-9.]+)?", txt)
                if m:
                    rec["k0k1_status"] = m.group(1)
                    rec["k0k1"] = float(m.group(2)) if m.group(2) else None
        # ACCEPTED means "this machine produced a usable seed artifact", NOT "every gated
        # limit resolved in band". The gate state travels with the seed and pick_seed.py
        # is what decides whether a given case may use it. A DRIFTING limit is the one
        # verdict that is not usable, because pick_seed refuses it outright.
        # A STUB IS NEVER ACCEPTED, whatever else is on disk.
        rec["accepted"] = (not a.stub
                           and rec["gate_state"] != "DRIFTING"
                           and os.path.getsize(restart) > 0)
        rec["status"] = "ok"
        if rec["gate_state"] == "DRIFTING":
            rec["reason"] = "gate DRIFTING on " + ", ".join(rec["gate_limits"])

        dst = os.path.join(outroot, "seeds", name)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(os.path.join(wd, "return")):
            shutil.copy2(os.path.join(wd, "return", f), os.path.join(dst, f))
        rec["out_dir"] = dst
    except subprocess.TimeoutExpired as e:
        t = getattr(e, "timeout", None)
        rec.update(status="failed",
                   reason=f"timeout after {t:.0f}s; the run and any FastEddy it left were "
                          f"killed so the GPU is free for the next seed"
                   if t else "timeout; the run and any FastEddy it left were killed")
    except Exception as e:  # never let one seed take the machine down
        rec.update(status="failed", reason=f"{type(e).__name__}: {e}")
    finally:
        rec["wall_s"] = round(time.time() - t0, 1)
        rec["t_end_s"] = round(time.time() - _t0, 2)
        # The dumps are the biggest thing a seed leaves and they are not a deliverable:
        # ~1.8 GB per seed at the 2.0 h ceiling, and the ONE artifact anything downstream
        # reads is return/seed_restart.nc. Kept by default so a failure can be examined.
        if a.prune_dumps and rec["status"] in ("ok", "skipped"):
            shutil.rmtree(os.path.join(wd, "output"), ignore_errors=True)
        shutil.rmtree(os.path.join(workroot, "_scratch", f"c2_check_{name}"), ignore_errors=True)
        shutil.rmtree(os.path.join(workroot, "_scratch", f"rotchk_{name}"), ignore_errors=True)
    return rec


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Generate the seed library across every visible GPU.")
    ap.add_argument("--gpu-count", type=int, default=0, help="0 = every visible GPU")
    ap.add_argument("--gpus", default="", help="explicit comma-separated indices, overrides --gpu-count")
    ap.add_argument("--pass", dest="pass_", default="all",
                    help="all (default) | N/M to take shard N of M -- for SPLITTING THE "
                         "LIBRARY ACROSS MACHINES. 30-over-16 on ONE machine needs no pass: "
                         "it is a work queue.")
    ap.add_argument("--out", default=os.environ.get("FLUX_OUT", "/out"))
    ap.add_argument("--jobs-dir", default=os.path.join(ROOT, "jobs30"))
    ap.add_argument("--only", default="", help="comma-separated job names, for a smoke test")
    ap.add_argument("--ceiling-h", type=float, default=2.0, help="simulated-hour hard ceiling")
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    ap.add_argument("--accel-regimes", default="neutral")
    ap.add_argument("--accel-s", type=int, default=3000)
    ap.add_argument("--accel-wth", type=float, default=0.05)
    ap.add_argument("--lpdm-workers", type=int, default=1)
    ap.add_argument("--sweep-steps", type=int, default=200)
    ap.add_argument("--no-sweep", action="store_true", help="keep the .in's block shape")
    ap.add_argument("--threadblock", default="", help="tBxXtByXtBz, skips the sweep")
    ap.add_argument("--skip-accept", action="store_true")
    ap.add_argument("--prune-dumps", action="store_true",
                    help="delete each seed's output/ once it succeeds (~1.8 GB per seed)")
    ap.add_argument("--job-timeout", type=int, default=6 * 3600)
    ap.add_argument("--accept-timeout", type=int, default=3600)
    # ---- SCHEDULER SELF-TEST ONLY. Both refuse to be useful for anything else. ----
    ap.add_argument("--stub", action="store_true",
                    help="SCHEDULER TEST: run jobs/run_seed.sh with STUB_SEED=1 -- no LES, "
                         "no gate, no battery. Every artifact is stamped stub:true and can "
                         "never be counted as an accepted seed.")
    ap.add_argument("--stub-seconds", type=float, default=2.0,
                    help="how long each stubbed job occupies its worker")
    ap.add_argument("--stub-fail", default="",
                    help="comma-separated job names the stub should FAIL, to show that a "
                         "failed seed frees its GPU instead of stranding it")
    ap.add_argument("--assume-gpus", type=int, default=0,
                    help="SCHEDULER TEST: pretend this many GPUs are visible. Refused "
                         "unless --stub, because it would otherwise hand real seeds to "
                         "devices that do not exist.")
    ap.add_argument("--force", action="store_true",
                    help="re-run seeds whose work directory already holds a finished restart")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.accel_regimes = {s.strip() for s in a.accel_regimes.split(",") if s.strip()}
    a.stub_fail_set = {s.strip() for s in a.stub_fail.split(",") if s.strip()}

    os.chdir(ROOT)
    os.makedirs(a.out, exist_ok=True)
    global _logfh
    _logfh = open(os.path.join(a.out, "run_seeds.log"), "a")

    # ---- provenance, first, because a rented machine has no other way to know ----
    say("=" * 78, stamp=False)
    say("Flux seed library -- FastEddy v5.0.1 (kegonsa fork)")
    if os.path.isfile(PROV):
        for ln in open(PROV):
            say("  " + ln.rstrip(), stamp=False)
    else:
        say("  WARNING: no IMAGE_PROVENANCE.txt; this is not the baked image")
    say("=" * 78)

    if a.assume_gpus and not a.stub:
        raise SystemExit(
            "FATAL: --assume-gpus is a scheduler self-test and requires --stub. Without it "
            "this would hand real seeds to devices that do not exist, and FastEddy would "
            "abort in gpuAssert on every one of them.")
    if a.stub:
        say("*" * 78, stamp=False)
        say("*** --stub: NO LES, NO GATE, NO ACCEPTANCE BATTERY. This is a test of the", stamp=False)
        say("*** WORK QUEUE and nothing else. Every artifact it writes is stamped", stamp=False)
        say("*** stub:true and is refused by bin/check_npz.py and by pick_seed.py.", stamp=False)
        say("*" * 78, stamp=False)

    gpus = discover_gpus()
    if a.assume_gpus:
        # Fabricated, and SAID SO. The uuid is a literal so VramWatch's map never matches
        # and no fake memory number can be reported as a measurement.
        gpus = [{"index": i, "name": "FAKE (--assume-gpus, scheduler test)", "cc": "0.0",
                 "vram_mib": 0, "uuid": f"FAKE-{i}"} for i in range(a.assume_gpus)]
        say(f"  --assume-gpus {a.assume_gpus}: the GPU list is FABRICATED for this test")
    if not gpus:
        raise SystemExit("FATAL: nvidia-smi reported no GPUs. Was --gpus all passed to docker run?")
    if a.gpus:
        want = [int(x) for x in a.gpus.split(",")]
        use = [g for g in gpus if g["index"] in want]
    else:
        use = gpus if a.gpu_count <= 0 else gpus[:a.gpu_count]
    if a.gpu_count > len(gpus):
        say(f"  NOTE: --gpu-count {a.gpu_count} but only {len(gpus)} GPUs are visible; using {len(gpus)}.")
    say(f"  GPUs visible: {len(gpus)}; using {len(use)}")
    for g in use:
        say(f"    [{g['index']}] {g['name']}  cc {g['cc']}  {g['vram_mib']} MiB")

    # ---- does the binary actually carry code for these cards? --------------------
    sass = binary_sass()
    say(f"  FastEddy SASS: {' '.join(sass) if sass else '(cuobjdump unavailable)'}")
    if sass and not a.assume_gpus:
        bad = [g for g in use if f"sm_{g['cc'].replace('.', '')}" not in sass]
        if bad:
            raise SystemExit(
                "FATAL: no compiled code for " +
                ", ".join(f"GPU {g['index']} (cc {g['cc']})" for g in bad) +
                f"; the binary carries [{' '.join(sass)}] and NO PTX, so it would fail at "
                "cuModuleLoad with 'no kernel image is available for execution on the "
                "device'. Rebuild the image with those architectures in FE_GENCODE.")
    # BITWISE REPRODUCIBILITY IS NOT CLAIMED AND IS NOT SOUGHT.
    ccs = sorted({g["cc"] for g in use})
    if len(ccs) > 1:
        say(f"  NOTE: mixed architectures on this machine ({', '.join(ccs)}).")
    say("  Seeds are turbulence REALISATIONS. FastEddy is not bitwise reproducible run to")
    say("  run on one GPU (~1e-4 relative in velocity after 200 steps), and it is not")
    say("  across architectures either. Do not diff two seeds; that is not a defect.")

    # ---- the job list -------------------------------------------------------------
    idx = os.path.join(a.jobs_dir, "index.json")
    if not os.path.isfile(idx):
        raise SystemExit(f"FATAL: no {idx}")
    jobs = json.load(open(idx))["jobs"]
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        jobs = [j for j in jobs if j["job"] in keep]
    if a.pass_ != "all":
        try:
            n, m = (int(x) for x in a.pass_.split("/"))
        except ValueError:
            raise SystemExit("FATAL: --pass must be 'all' or 'N/M'")
        jobs = [j for i, j in enumerate(jobs) if i % m == (n - 1) % m]
        say(f"  --pass {a.pass_}: this machine takes {len(jobs)} of the library")
    say(f"  seeds to run: {len(jobs)}  (ceiling {a.ceiling_h} sim-h, "
        f"early stop {'on' if a.early_stop else 'off'})")
    if a.dry_run:
        for j in jobs:
            say(f"    {j['job']:28s} {j['regime']:11s} zi {j['target']['zi_m']:>6.0f} m  "
                f"G {j['target']['G']:.1f}  {j['run']['steps_total']} steps")
        return 0

    if a.stub:
        # A stub writes a 1 kB text file where a 73 MB netCDF goes, so the battery would
        # fail on every step for reasons that say nothing about the scheduler; and the
        # sweep needs a real GPU. Forced rather than merely defaulted, so a stub run cannot
        # be talked into producing something that looks like an acceptance.
        a.skip_accept, a.no_sweep, a.threadblock = True, True, ""
    workroot = os.path.join(a.out, "work")
    os.makedirs(os.path.join(workroot, "_scratch"), exist_ok=True)

    # ---- the block shape, MEASURED HERE ------------------------------------------
    tb = None
    if a.threadblock:
        x, y, z = (int(v) for v in a.threadblock.lower().split("x"))
        tb = {"tBx": x, "tBy": y, "tBz": z, "shape": a.threadblock, "source": "--threadblock"}
        say(f"\n  thread block {a.threadblock} taken from the command line; no sweep")
    elif not a.no_sweep:
        say(f"\n=== thread-block sweep on GPU {use[0]['index']} "
            f"({use[0]['name']}), {a.sweep_steps} steps per shape ===")
        tmpl = os.path.join(a.jobs_dir, jobs[0]["job"], "seed.in")
        sj = os.path.join(a.out, "threadblock_sweep.json")
        env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1",
                   OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1", OMPI_MCA_plm="isolated")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "bin/threadblock_sweep.py"),
                            "--template", tmpl, "--steps", str(a.sweep_steps),
                            "--gpu", str(use[0]["index"]), "--fe-bin", FE_BIN,
                            "--work", os.path.join(workroot, "_tb"), "--json", sj],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        for ln in (r.stdout + r.stderr).splitlines():
            say("  " + ln)
        # ASSERT ON THE ARTIFACT. If the sweep did not write a winner, the .in's own shape
        # stands -- it is the Ada measurement, which is a real measurement on a real GPU,
        # just not this one. Falling back to it is honest; guessing is not.
        if os.path.isfile(sj):
            tb = json.load(open(sj))["winner"]
            tb["source"] = "measured on this machine"
        else:
            say("  WARNING: the sweep produced no winner; keeping the .in's own tBx/tBy/tBz")
    else:
        say("\n  --no-sweep: keeping each .in's own tBx/tBy/tBz (measured on Ada)")
    if tb:
        say(f"  block shape for every seed: {tb['tBx']}x{tb['tBy']}x{tb['tBz']} ({tb['source']})")

    # ---- the queue ---------------------------------------------------------------
    vram = VramWatch({g["uuid"]: g["index"] for g in use})
    vram.start()
    host = HostWatch()
    host.start()

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    results, rlock = [], threading.Lock()
    t_start = time.time()

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            # run_one is written not to raise, but that only covers the body of ITS try --
            # and say() below writes to a log file on the mounted volume, which is exactly
            # the thing that fails when /out fills up (the default keeps every seed's
            # dumps, ~53 GB for the library). An exception here would kill the worker
            # thread with the job already popped: never recorded, never re-queued, the GPU
            # silently retired, and every count in the summary derived from len(results)
            # so the loss would not appear anywhere.
            try:
                rec = run_one(j, gpu, a, workroot, a.out, tb)
            except BaseException as e:
                rec = {"job": j.get("job", "?"), "gpu": gpu, "status": "failed",
                       "reason": f"the worker itself raised: {type(e).__name__}: {e}",
                       "gate_state": None, "gate_limits": [], "accepted": False,
                       "sim_h": None, "gpu_s": None, "wall_s": None}
            with rlock:
                results.append(rec)
                done, tot = len(results), len(jobs)
            try:
                say(f"  [gpu {gpu}] {rec['job']}: {rec['status'].upper()}"
                    f" gate={rec['gate_state']}"
                    f" {rec['sim_h']} sim-h"
                    f" {(rec['gpu_s'] or 0) / 3600:.2f} GPU-h"
                    f"   ({done}/{tot} done)")
            except Exception:
                pass
            q.task_done()

    say(f"\n=== running {len(jobs)} seeds across {len(use)} GPUs ===")
    threads = [threading.Thread(target=worker, args=(g["index"],), daemon=True) for g in use]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    vram.stop.set()
    host.stop.set()
    elapsed = time.time() - t_start

    # ---- the summary --------------------------------------------------------------
    # RECONCILE. Every count below is derived from `results`, so a job that never produced
    # a record would simply not exist in the summary -- the machine would report success
    # over a library with a hole in it. Compare against the list that was queued.
    ran = {r["job"] for r in results}
    lost = [j["job"] for j in jobs if j["job"] not in ran]
    for name in lost:
        results.append({"job": name, "status": "failed", "gpu": None,
                        "reason": "NEVER RAN -- no record was produced for this seed. A "
                                  "worker died between taking the job and recording it.",
                        "gate_state": None, "gate_limits": [], "accepted": False,
                        "sim_h": None, "gpu_s": None, "wall_s": None})
    results.sort(key=lambda r: r["job"])
    ok = [r for r in results if r["status"] in ("ok", "skipped")]
    acc = [r for r in ok if r["accepted"]]
    bad = [r for r in results if r["status"] not in ("ok", "skipped")]
    say("\n" + "=" * 78)
    say(f"SUMMARY  {len(acc)} accepted / {len(ok)} completed / {len(results)} attempted"
        f"   wall {elapsed / 3600:.2f} h on {len(use)} GPUs")
    say("=" * 78)
    say(f"  {'seed':28s} {'gate':14s} {'sim-h':>6s} {'ceil':>5s} {'GPU-h':>6s} {'wall':>6s}  note")
    for r in results:
        note = ""
        if r["status"] == "skipped":
            # NOT "FAILED". `bad` already excludes skipped, so labelling the row FAILED
            # made the table contradict its own header: resuming a finished library
            # printed 30 rows of "FAILED: already complete" above "30 accepted / 30
            # completed" and an exit status of 0.
            note = "skipped: already complete, GPU work not repeated"
        elif r["status"] != "ok":
            note = "FAILED: " + r["reason"][:90]
        elif r.get("early_stopped"):
            note = "early stop"
        elif r["sim_h"] and r["sim_h"] < a.ceiling_h * 0.99:
            note = "short of ceiling"
        else:
            note = "ran to the ceiling"
        if r["gate_state"] == "DRIFTING":
            note += " | DRIFTING: " + ", ".join(r["gate_limits"])[:60]
        elif r["gate_state"] == "INDETERMINATE" and r["gate_limits"]:
            note += " | indet: " + ", ".join(r["gate_limits"])[:50]
        say(f"  {r['job']:28s} {str(r['gate_state']):14s} "
            f"{(r['sim_h'] if r['sim_h'] is not None else float('nan')):6.3f} "
            f"{a.ceiling_h:5.1f} "
            f"{((r['gpu_s'] or 0) / 3600):6.2f} {((r['wall_s'] or 0) / 3600):6.2f}  {note}")

    # ONE LIST OF PAIRS, NOT TWO INDEPENDENT LISTS. The first version filtered gpu_s and
    # sim_h separately and then divided sum by sum, so a seed missing either one moved the
    # ratio without moving the other side: one lost run.log out of two seeds turns 0.479
    # into 0.240, which reads as "Blackwell is twice as fast as Ada" -- a headline
    # architectural claim manufactured from a missing file. And fasteddy_gpu_seconds
    # returns 0.0, which is falsy, when a log exists but holds no performance block.
    pairs = [(r["gpu_s"] / 3600.0, r["sim_h"], r["job"]) for r in ok
             if r.get("gpu_s") and r.get("sim_h")]
    missing = [r["job"] for r in ok if not (r.get("gpu_s") and r.get("sim_h"))]
    gpuh = [p[0] for p in pairs]
    simh = [r["sim_h"] for r in ok if r.get("sim_h")]
    if gpuh:
        say(f"\n  measured GPU-h per seed (n={len(gpuh)}): min {min(gpuh):.3f}  median "
            f"{sorted(gpuh)[len(gpuh) // 2]:.3f}  max {max(gpuh):.3f}  total {sum(gpuh):.2f}")
        # NOT `acc`. That name holds the ACCEPTED SEEDS from further up this function, and
        # rebinding it here made `n_accepted` in the machine manifest the count of seeds
        # with an accelerator leg -- i.e. the number of NEUTRAL seeds. On the first real
        # library that printed "11 accepted" to the log and wrote 12 to the JSON, and 12 is
        # a plausible enough number that it was read and quoted before the log was checked.
        accel_h = [(r["gpu_s_accel"] or 0) / 3600.0 for r in ok if r.get("gpu_s_accel")]
        if accel_h:
            say(f"    of which the Steinfeld accelerator (neutral rungs): "
                f"{sum(accel_h):.2f} GPU-h over {len(accel_h)} seeds")
    if pairs:
        rate = sum(p[0] for p in pairs) / sum(p[1] for p in pairs)
        say(f"  measured GPU-h per SIMULATED hour: {rate:.3f}  over the {len(pairs)} seeds"
            f" that have BOTH numbers   (the 30 m Ada measurement is 0.479)")
    if missing:
        # NO SILENT EXCLUSIONS: a seed dropped from the rate is named, because the rate is
        # the number a reader will quote as an architecture comparison.
        say(f"  EXCLUDED from that rate for want of a GPU-time or a sim-time: "
            f"{', '.join(missing)}")
    if simh:
        at_ceiling = sum(1 for s in simh if s >= a.ceiling_h * 0.99)
        say(f"  stop times vs the {a.ceiling_h} sim-h ceiling: {at_ceiling} of {len(simh)} ran to "
            f"the ceiling; {len(simh) - at_ceiling} stopped early "
            f"(min {min(simh):.3f} h, median {sorted(simh)[len(simh) // 2]:.3f} h)")
    k0 = [(r["job"], r.get("k0k1"), r.get("k0k1_status")) for r in ok if r.get("k0k1_status")]
    if k0:
        vals = [v for _, v, st in k0 if v is not None and st == "OK"]
        say(f"\n  k0/k1 (the accuracy-CFL check; ~0.27 correct, ~9 means dt is past the "
            f"boundary): " + (f"{min(vals):.3f}-{max(vals):.3f} over {len(vals)} seeds"
                              if vals else "no numeric verdict"))
        for j, v, st in k0:
            if st != "OK":
                say(f"    {j}: k0/k1 {st} -- this established NOTHING about dt")
        say("  A run whose k0/k1 FAILS never gets this far: docker/k0k1_check.py exits 1,")
        say("  check_run.sh fails the run and jobs/run_seed.sh dies before a seed exists.")
    # ---- HOST MEMORY. The number the next rental is sized on. --------------------
    gb = lambda b: (b or 0) / 1024**3
    say(f"\n  HOST MEMORY under {len(use)}-way load:")
    say(f"    peak container RSS (cgroup)      : {gb(host.cgroup_peak):6.2f} GB")
    if host.fe_rss_peak:
        say(f"    peak summed FastEddy RSS         : {gb(host.fe_rss_peak):6.2f} GB "
            f"over {host.fe_rss_peak_n} concurrent process(es)"
            + (f"  = {gb(host.fe_rss_peak) / host.fe_rss_peak_n:.2f} GB each"
               if host.fe_rss_peak_n else ""))
    if host.mem_total:
        say(f"    machine RAM                      : {gb(host.mem_total):6.2f} GB"
            + (f", low-water MemAvailable {gb(host.mem_avail_min):.2f} GB"
               if host.mem_avail_min is not None else ""))
    say("    NOTE: a SEED runs no LPDM, so this is NOT the corpus-case figure. PROJECT_BRIEF.md's")
    say("    12.45 GB peak host RSS is a CASE -- the LPDM's 12.0 GB fp16 field cache, which")
    say("    a seed never allocates. Size a seed box on the number above; size a CORPUS box")
    say("    on ~12.5 GB per concurrent case.")

    # ---- THE QUEUE, AUDITED FROM THE TIMELINE ------------------------------------
    tl = [r for r in results if r.get("t_start_s") is not None and r.get("t_end_s") is not None]
    if tl:
        per_gpu = {}
        for r in tl:
            per_gpu.setdefault(r["gpu"], []).append(r)
        reused = {g: v for g, v in per_gpu.items() if len(v) > 1}
        # peak concurrency, from the interval overlaps rather than from the design
        edges = sorted([(r["t_start_s"], 1) for r in tl] + [(r["t_end_s"], -1) for r in tl])
        cur = peak = 0
        for _, d in edges:
            cur += d
            peak = max(peak, cur)
        say(f"\n  QUEUE, from the recorded timeline (not from the design):")
        say(f"    workers used            : {len(per_gpu)} of {len(use)}")
        say(f"    workers that took >1 job: {len(reused)}"
            + (f"  ({', '.join(f'gpu {g}: {len(v)}' for g, v in sorted(reused.items()))})"
               if reused else ""))
        say(f"    peak concurrent jobs    : {peak}  (worker count {len(use)})")
        if peak > len(use):
            say(f"    *** MORE JOBS RAN AT ONCE THAN THERE ARE WORKERS. The queue is not "
                f"bounding concurrency.")

    if max(vram.peak.values(), default=0) == 0:
        say("  peak VRAM: nothing ran on a GPU during this invocation (every seed was "
            "already complete), so there is nothing to report.")
    say(f"  peak VRAM per GPU (compute apps): " +
        "  ".join(f"[{i}] {m} MiB" for i, m in sorted(vram.peak.items())))
    say(f"  peak VRAM per GPU (whole device): " +
        "  ".join(f"[{i}] {m} MiB" for i, m in sorted(vram.peak_dev.items())))
    hdr = max((g["vram_mib"] for g in use), default=0)
    # ONLY IF SOMETHING ACTUALLY RAN. On a pure resume nothing touches a GPU, the sampler
    # sees no compute app, and the headroom line read "the largest seed took 0 MiB of
    # 16376 MiB (0.0%)" -- a measurement of nothing, printed as a measurement.
    if hdr and vram.peak and max(vram.peak.values()) > 0:
        say(f"  headroom: the largest seed took {max(vram.peak.values())} MiB of {hdr} MiB "
            f"({100 * max(vram.peak.values()) / hdr:.1f}%). The ring buffer was sized for a "
            f"16 GB card; nothing here assumes that budget.")
    if bad:
        say("\n  FAILED SEEDS:")
        for r in bad:
            say(f"    {r['job']}: {r['reason']}")
    drift = [r for r in ok if r["gate_state"] == "DRIFTING"]
    if drift:
        say("\n  DRIFTING (produced an artifact, but pick_seed.py refuses these outright "
            "unless ALLOW_DRIFTING admits them):")
        for r in drift:
            say(f"    {r['job']}: {', '.join(r['gate_limits'])}")

    # ---- the library-wide direction table, ONCE, over everything that finished --------
    # Step 8 of each seed's own battery runs bin/direction_drift.py over the library AS IT
    # STOOD WHEN THAT SEED FINISHED -- the first seed sees a library of one and the last
    # sees thirty. Every one of those tables is correct for its moment and none of them is
    # the library's. This is the one that is.
    dd = os.path.join(a.out, "direction_drift_library.txt")
    try:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "bin/direction_drift.py"),
                            "--library", workroot, "--out", dd],
                           cwd=ROOT, capture_output=True, text=True, timeout=900,
                           env=dict(os.environ, OMP_NUM_THREADS="1"))
        if os.path.isfile(dd):
            say(f"\n  library-wide direction drift over {len(ok)} completed seeds -> {dd}")
        else:
            say(f"\n  direction_drift produced no table: {(r.stderr or r.stdout)[-300:]}")
    except Exception as e:
        say(f"\n  direction_drift skipped: {type(e).__name__}: {e}")

    man = {"generated_by": "bin/run_seeds.py",
           "provenance": open(PROV).read() if os.path.isfile(PROV) else None,
           "gpus": use, "n_gpus_used": len(use),
           "fasteddy_sass": sass,
           "threadblock": tb, "ceiling_sim_h": a.ceiling_h,
           "early_stop": a.early_stop, "pass": a.pass_,
           "wall_h": round(elapsed / 3600, 3),
           "peak_vram_mib_compute": vram.peak, "peak_vram_mib_device": vram.peak_dev,
           "stub": bool(a.stub),
           "host_memory": {"cgroup_peak_bytes": host.cgroup_peak,
                           "fasteddy_rss_peak_bytes": host.fe_rss_peak,
                           "fasteddy_rss_peak_nproc": host.fe_rss_peak_n,
                           "mem_total_bytes": host.mem_total,
                           "mem_available_min_bytes": host.mem_avail_min,
                           "note": "A seed runs no LPDM. PROJECT_BRIEF.md's 12.45 GB is a CORPUS "
                                   "CASE (the LPDM field cache), not a seed."},
           # ASSERT ON THE VALUE, not on the variable still being in scope: this field and
           # the SUMMARY line above must be the same number, and for one library they were
           # not (see the accel_h rename above).
           "n_attempted": len(results), "n_completed": len(ok),
           "n_accepted": sum(1 for r in ok if r["accepted"]),
           "seeds": results}
    mp = os.path.join(a.out, "machine_manifest.json")
    json.dump(man, open(mp, "w"), indent=1)
    say(f"\n  -> {mp}")
    say(f"  -> {os.path.join(a.out, 'seeds')}/<job>/  (seed_restart.nc and the battery)")
    # The machine did its job if it attempted everything and recorded every outcome. A
    # DRIFTING seed is a result about the boundary layer, not a failure of the run.
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
