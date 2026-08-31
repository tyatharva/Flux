# Generating the seed library on Vast.ai

**Pick the image in Vast, SSH in, run one command.** The code is baked in and the tag names
the commit, so a rented machine clones nothing, builds nothing, and cannot pull the wrong
revision. Rationale: `PROJECT_BRIEF.md` (dated block at the top) and `FASTEDDY_TRAPS.md` §23.

---

## 0. ONE-TIME: make the package public

**The image is on GHCR and is currently PRIVATE. Vast pulls anonymously, so an instance will
fail to start until this is changed.** Verified rather than assumed — an anonymous manifest
fetch returns **HTTP 403** today, where a package known to be public returns 200:

```bash
T=$(curl -s "https://ghcr.io/token?scope=repository:tyatharva/flux-seeds:pull&service=ghcr.io" \
    | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $T" \
  -H "Accept: application/vnd.oci.image.manifest.v1+json" \
  https://ghcr.io/v2/tyatharva/flux-seeds/manifests/latest
```

`200` means Vast can pull it. `403` means it cannot.

GitHub exposes package visibility **only through the web UI** — the REST endpoint returns
404 even with a `write:packages` token, so this cannot be scripted:

> https://github.com/users/tyatharva/packages/container/package/flux-seeds
> → **Package settings** → **Danger Zone** → **Change visibility** → **Public**

**If you would rather keep it private**, Vast's template has *Docker registry credentials*
fields: username `tyatharva`, password a GitHub PAT with `read:packages`. That works, at the
cost of every instance carrying a credential.

---

## 1. Vast instance — the exact values to enter

| field | value |
|---|---|
| **Image path/tag** | `ghcr.io/tyatharva/flux-seeds:86f86d65ed2e-fe0ce48d5dff06` |
| *(or the stable tag)* | `ghcr.io/tyatharva/flux-seeds:latest` |
| **Launch mode** | **SSH** (Interactive shell server) |
| **On-start script** | **LEAVE EMPTY.** Nothing auto-runs — deliberate; see §3. |
| **Docker options** | **LEAVE EMPTY.** Everything needed is in the image ENV. |
| **Disk space** | **100 GB** |
| **GPU** | 16x RTX 5090 |
| **CUDA filter** | **≥ 13.0** — a hard requirement, see below |
| **RAM filter** | **≥ 64 GB** (floor 32 GB) |

### CUDA ≥ 13.0 is a hard filter, and getting it wrong wastes the rental

Vast filters on the **driver**, reported as a CUDA version. This image is a **CUDA 13.0.1**
binary, and NVIDIA's minimum Linux driver for CUDA 13.0 GA is **580.65.06**. A machine
reporting CUDA ≥ 13.0 has at least that driver.

**A 5090 box with a 570.x driver will NOT run this image**, even though a 5090 is `sm_120`
and the binary contains `sm_120` SASS. The *toolkit* major version is what the driver has to
satisfy, not the architecture, and the failure is at image load — before any physics.

The image does ship NVIDIA's forward-compatibility package
(`/usr/local/cuda/compat/libcuda.so.580.82.07`). **Do not plan around it**: forward
compatibility is supported on data-center GPUs, not GeForce. If only 570-driver boxes are
available, rebuild on CUDA 12.8 instead — `Dockerfile.blackwell`'s `FE_GENCODE` is unchanged
and 12.8 also targets `sm_120`.

### Disk: 100 GB

| | |
|---|---|
| image, uncompressed on disk | 13.1 GB |
| 30 seeds x 24 dumps x 73.3 MB in `/out/work` | 54 GB |
| `/out/seeds` deliverables | 2.2 GB |
| per-seed scratch (Gate C2 restart copy, 3 rotation restarts), 16 concurrent | ~6 GB peak |
| **total** | **~75 GB** |

100 GB gives headroom. With `--prune-dumps` the total is ~22 GB and **40 GB** suffices — but
the dumps are the only thing that lets a surprising seed be re-examined without re-running
it, and it deletes them only on success.

### RAM: 64 GB — and the number that matters LATER is a different one

**MEASURED on this image** (`memory.peak` of a container running only that phase):

| | |
|---|---|
| LES phase, peak container RSS | **0.82 GB** (FastEddy itself 0.49 GB over 2 processes) |
| acceptance battery, peak container RSS | **1.51 GB** |
| **16-way worst case** (all sixteen in their battery at once) | **~24 GB** |

64 GB is comfortable; 32 GB is the floor. The run reports its own peak, so the next rental
is sized on a measurement rather than on this estimate.

**THE 12.45 GB FIGURE IS NOT THIS, and the distinction is the whole point.** `PROJECT_BRIEF.md`'s
peak host RSS of 12.45 GB is a **corpus CASE** — the LPDM's 12.0 GB fp16 field cache, which
`compute_footprint` random-accesses on the CPU. **A seed runs no LPDM and never allocates
it.** So:

- **seed box (this run): 16 x ~1.5 GB → 64 GB is generous.**
- **corpus box (later, and much larger): 16 x 12.45 GB ≈ 200 GB → size it at 256 GB.**

Sizing the seed box on the case number over-rents by 4x; sizing the corpus box on the seed
number swaps it to death. The run reports both, labelled.

### What is NOT needed, and why it is worth knowing

- **No `-v` mount.** Vast's Docker options field "will only accept ports and environment
  variables" — there are no host bind mounts. `/out` lives in the container filesystem.
- **No `--shm-size`.** That is for the corpus cases' in-process LPDM ring staging. A seed
  never touches `/dev/shm`.
- **No data mount.** A seed is a flat, uniform, doubly-periodic spin-up: `topoFile` empty,
  `inFile` empty. It reads no sounding, no terrain and no land cover. HRRR and the surface
  grids matter only for corpus CASES.

---

## 2. Image reference and size

| | |
|---|---|
| registry | **GHCR** (`ghcr.io`) — the credential is already on this machine and the image is built from a GitHub-hosted commit, so the tag and the source live in one place |
| commit-pinned | `ghcr.io/tyatharva/flux-seeds:86f86d65ed2e-fe0ce48d5dff06` |
| stable | `ghcr.io/tyatharva/flux-seeds:latest` |
| **compressed (what Vast pulls, per instance)** | **4.05 GB**, 22 layers |
| largest layers | 2.14 GB (CUDA 13.0.1 base), 1.40 GB (pip: scipy/xarray/dask/herbie) |
| uncompressed on disk | 13.1 GB |

The tag is `<flux-sha>-fe<fasteddy-sha>` — **both** repositories, because they are two repos
and an image naming only one of them is not pinned. `docker/build_image.sh` refuses a dirty
tree in either, and prints the tag it just built.

**THE COMMIT-PINNED TAG WRITTEN ABOVE CAN GO STALE BY ONE COMMIT, AND THAT IS UNAVOIDABLE
RATHER THAN SLOPPY.** Editing this file changes the flux SHA, which changes the tag, which
would have to be written back into this file — a regress with no fixed point. So:
**`:latest` is the paste-able value**, and the authoritative answer to "what is actually
running" is `provenance` on the box, which prints the commit baked into the image. If the
tag above and `provenance` disagree, `provenance` is right and the difference is
documentation-only.

Rebuild and push:

```bash
docker/build_image.sh                                   # -> flux-seeds:<flux>-fe<fasteddy>
TAG=$(docker images --format '{{.Tag}}' flux-seeds | grep -v latest | head -1)
docker tag  flux-seeds:$TAG ghcr.io/tyatharva/flux-seeds:$TAG
docker tag  flux-seeds:$TAG ghcr.io/tyatharva/flux-seeds:latest
docker push ghcr.io/tyatharva/flux-seeds:$TAG
docker push ghcr.io/tyatharva/flux-seeds:latest
```

---

## 3. SSH workflow — nothing auto-runs

**The image has NO `ENTRYPOINT`.** Vast's own documentation: *"With the SSH launch option
your docker image entrypoint is not called as we must override it."* The previous entrypoint
printed help and **exited**, which on Vast is an instance that comes up and dies. Now:

- `CMD` is `sleep infinity` — the container stays up doing nothing.
- The subcommands are real executables on `PATH`
  (`/usr/local/bin/{run_seeds,verify,seed,accept,provenance}`).
- `FLUX_ROOT` / `FLUX_NATIVE` / `FLUX_OUT` are in the image ENV **and** in
  `/etc/environment`, because an SSH login shell inherits neither the image ENV nor
  `docker run -e`. Without the second, `run_seeds` would run with `FLUX_NATIVE` unset and
  shell out to a docker daemon that is not there.
- `openssh-server` is installed, because Vast requires an image "compatible with typical
  ssh daemon setup" without saying whether it supplies sshd. ~1 MB against a class of
  instance-will-not-come-up failure only discoverable after renting.
- Logging in prints the image provenance and the two commands.

The same image still works unchanged on a workstation — there is no entrypoint to prepend
and the names resolve on `PATH`:

```bash
docker run --gpus all -v /out:/out ghcr.io/tyatharva/flux-seeds:latest run_seeds --gpu-count 16
```

---

## 4. LAUNCH PROCEDURE — paste this

```bash
# --- on your machine: connect (Vast gives you the exact host and port) -------------
ssh -p <PORT> root@<HOST>

# --- on the instance ---------------------------------------------------------------
provenance                       # which commit, which CUDA, which architectures
verify                           # seconds, no GPU-hours. STOP if this does not say PASS.
nvidia-smi -L | wc -l            # confirm 16 cards are actually attached

run_seeds --gpu-count 16 2>&1 | tee /out/run_seeds.console.log
```

That is the whole thing. **There is no staged single-GPU step**: per-seed GPU-h is measured
from the running queue and reported in the summary, so a warm-up run would only produce a
number the real run measures anyway — and under the wrong contention.

`run_seeds` first sweeps the CUDA thread-block shape on the first GPU (**1 min 54 s
measured**, 14 of 106 legal shapes, the other 92 named rather than silently dropped), then
fans all 30 seeds across the 16 cards.

**Expected wall clock:** 0.94 GPU-h per seed measured on Ada; 30 seeds over 16 GPUs is two
rounds, so **~2 hours** if Blackwell matches Ada. The summary reports what it actually was.

### It is resumable, and a failed seed never stops the box

Re-running the same command skips seeds whose work directory already holds a finished
`seed_restart.nc` **and** its acceptance battery, and restarts partial ones from step 0.
A box that lost power costs only the seeds that were mid-flight. Each seed runs in its own
subprocess; a crash, a gate failure and a timeout are recorded with a reason and the queue
moves on.

---

## 5. Getting the seeds back

**~73 MB per seed, ~2.2 GB total.** From your machine:

```bash
# the deliverables only (2.2 GB) -- this is what the corpus needs
rsync -avP -e "ssh -p <PORT>" root@<HOST>:/out/seeds/ ./vast-seeds/

# the machine-level records: manifest, sweep, direction table, log (a few MB)
rsync -avP -e "ssh -p <PORT>" \
  root@<HOST>:/out/{machine_manifest.json,threadblock_sweep.json,direction_drift_library.txt,run_seeds.log} \
  ./vast-seeds/

# then, locally, put each seed where bin/pick_seed.py looks for it
for d in ./vast-seeds/seed_*; do
  n=$(basename "$d"); mkdir -p "jobs30/$n/return" && cp -a "$d"/* "jobs30/$n/return/"
done
```

`scp -P <PORT> -r root@<HOST>:/out/seeds ./vast-seeds` works too; `rsync -P` is preferred
because it resumes.

**WHERE `/out` LIVES, AND WHAT SURVIVES WHAT.** `/out` is a directory in the container
filesystem, created in the image. Vast **stops and starts** a container rather than
recreating it, so `/out` survives an instance stop/start and a container restart — which is
what makes the resume above useful. It does **not** survive **destroying** the instance.
**Pull the seeds down before you destroy the box.** Nothing in the run deletes them;
`--prune-dumps` only removes `/out/work/<seed>/output/`, never `/out/seeds/`.

---

## 6. What the run reports, and why each number is there

Printed at the end and written to `/out/machine_manifest.json`:

| | |
|---|---|
| **measured GPU-h per seed** | min / median / max / total, over the seeds that have both a GPU time and a sim time — **computed from paired values, never from two independently filtered sums** — and any seed excluded for want of one is named |
| **measured GPU-h per SIMULATED hour** | quoted against the **0.469** measured single-GPU on Ada. This is the 16-way contention number, and the reason not to rent the corpus box blind. |
| of which the Steinfeld accelerator | neutral rungs run a 3000 s burn-in in a *separate* log; counting only `run.log` under-reports a neutral seed by ~29% |
| **peak host RSS across the machine** | peak container RSS (cgroup), peak summed FastEddy RSS with its process count, and the `MemAvailable` low-water mark |
| stop times vs the 2.0 sim-h ceiling | how many ran to the ceiling, how many the watcher stopped early |
| peak VRAM per GPU | attributed to compute apps, and whole-device |
| k0/k1 per seed | the accuracy-CFL check, ~0.27 correct. A run whose k0/k1 FAILS never produces a seed: `k0k1_check.py` exits 1, `check_run.sh` fails the run, `run_seed.sh` dies. |
| queue audit | workers used, workers that took more than one job, peak concurrency — **from the recorded per-job timeline, not from the design** |

### `accepted` does not mean every gated limit resolved, and it must not

`PLAN.md` records that `TKE_BL/u*^2` and `z_i` cannot be resolved against their thresholds
at **any** scoring-window width in an affordable spin-up — they decorrelate on the eddy
turnover, not on the dump interval. **INDETERMINATE is the library's normal state** and
`bin/pick_seed.py` admits it under a flag. What it refuses outright is **DRIFTING**, and
that is what the summary calls out separately.

---

## 7. The scheduler is verified, not asserted

Both tests run in about a minute, on **no GPU**, against the shipped image.

### The work queue — `bin/test_work_queue.sh`

20 jobs over 6 fabricated workers with the LES stubbed (`STUB_SEED=1` replaces FastEddy, the
gate and the battery; every artifact is stamped `stub: true`; `--assume-gpus` refuses to run
without `--stub`). Every check reads the recorded timeline out of `machine_manifest.json`:

```
[PASS] every job recorded exactly once                20 records, 20 distinct
[PASS] a worker took a SECOND job after its first     14 such hand-offs
       gpu 4: seed_nbl-shallow_a060 ended 3.1s -> seed_nbl-deep_a000    started 3.1s
       gpu 4: seed_nbl-deep_a000    ended 6.2s -> seed_cbl-shallow_a000 started 6.2s
       gpu 4: seed_cbl-shallow_a000 ended 9.3s -> seed_cbl-mid_a000     started 9.3s
[PASS] the deliberately-failed jobs did fail          2 of 2
[PASS] a failed job RELEASED its worker               gpu 5 freed by seed_cbl-shallow_a015
                                                      gpu 2 freed by seed_nbl-shallow_a030
[PASS] peak concurrency never exceeded the workers    peak 6, workers 6
[PASS] the run ended only when the queue was empty    attempted 20 of 20
[PASS] no stub was counted as an accepted seed        n_accepted=0
```

The hand-offs land on the 3 s stub duration exactly — 3.1s, 6.2s, 9.3s — which is what a
queue does and what a two-pass scheduler cannot.

### The per-GPU mutex — `bin/test_gpu_mutex.sh`

Eight launches released from **one fifo barrier** at device 0, so they reach the guard
simultaneously. A `/proc` scan is check-then-act: it passes a sequential two-process test
while failing this one silently. The guard is therefore an `flock(2)` held for the life of
the run, with the scan kept only to name the holder.

```
[PASS] all racers returned a status                   8 of 8
[PASS] EXACTLY ONE launched FastEddy                  launched: [6]
[PASS] every other racer was REFUSED with exit 2      7 refused
[PASS] the refusals name the device                   7 of 7
```

The stub is a compiled ELF at a path ending in `FEMAIN/FastEddy`, because the guard resolves
`/proc/<pid>/exe` — a shell-script stub would be invisible to the mechanism under test, and
the test would pass without testing anything.

---

## Options worth knowing

| | |
|---|---|
| `--gpu-count N` / `--gpus 0,3,7` | 0 (default) uses every visible GPU |
| `--pass N/M` | **for splitting the library across SEVERAL machines.** 30 over 16 on ONE machine needs no pass — it is a queue |
| `--only <job,...>` | a smoke test on named seeds |
| `--ceiling-h 2.0` | simulated-hour hard ceiling per seed |
| `--no-sweep` | keep the `.in`'s measured `1x2x64` instead of re-measuring |
| `--prune-dumps` | delete each seed's `output/` on success (~1.8 GB each) |
| `--dry-run` | list what would run |
| `--force` | re-run seeds whose work directory already holds a finished restart |
| `--stub`, `--assume-gpus` | scheduler self-test only; cannot produce an accepted seed |

## What lands in `/out`

| | |
|---|---|
| `seeds/<job>/seed_restart.nc` | **the seed**, 73.3 MB |
| `seeds/<job>/` | `stationarity.json`, `manifest.json` (with `achieved` and `run.ceiling_steps`), `acceptance.txt`, `seed_report.json`, `turb_alive.json`, `rotation_check.json`, `direction_drift.txt`, logs |
| `machine_manifest.json` | every seed: verdict, GPU-h, stop time, per-job timeline, peak VRAM, host memory |
| `threadblock_sweep.json` | the block shape measured on this machine, runners-up, repeat noise |
| `direction_drift_library.txt` | Ekman backing over the whole library, computed once at the end |
| `work/<job>/` | working directory, dumps included |
| `run_seeds.log` | everything the run printed, timestamped |

## Things that are true and easy to be surprised by

**Everything in `/out` is written as root.** The image declares no `USER`, because OpenMPI
refuses to launch as root without `OMPI_ALLOW_RUN_AS_ROOT` and setting that is simpler than
matching a uid that differs on every rented box. On Vast you are root anyway.

**Bitwise reproducibility does not hold across architectures, and is not sought.** FastEddy
is not bitwise reproducible run-to-run on ONE GPU with ONE binary — ~1e-4 relative in
velocity after 200 steps, from the block-retirement order of an `atomicAdd` in the slab-mean
reduction. Seeds are turbulence realisations. Do not diff two of them.

**The image carries real SASS for `sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120` and no
PTX at all** (the analysis library `lib/liblpdm.so` does carry PTX — it is one translation
unit, compiled whole-program, and it is there so the contrast can be checked on the image
itself). That is not an omission: FastEddy is built with separate compilation, and
`nvcc -dlink` silently drops every PTX image from the fatbin. There is therefore no JIT
fallback by construction, and anything older than Turing will not run. `verify` says so at
startup rather than at the first kernel launch.

**An out-of-range `CUDA_VISIBLE_DEVICES` fails loudly, at exit 100.** `gpuErrchk` at
`fecuda_Device.cu:55` prints `GPUassert: no CUDA-capable device is detected` before the
no-device branch is ever reached. The orchestrator and the per-seed preflight assert the
device exists first anyway, because failing at the preflight costs nothing.

**The thread-block sweep runs before the seeds and takes about two minutes.** It is a pure
performance knob and cannot move the physics: the one reduction that accumulates is
templated on compile-time constants that do not follow `tBx/tBy/tBz`. It reports the shapes
it did NOT measure by name, so the winner is "best of what was tried" rather than an implied
"best that exists".
