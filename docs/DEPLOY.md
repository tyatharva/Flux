# Running Flux on Vast.ai — the seed library, then the corpus

**This file covers two separate campaigns and they are run months apart.**

| | what it makes | where | when |
|---|---|---|---|
| **Part 1 — §0-§7** | the **30-seed library**, 1 machine x 16 GPUs, ~1 h | `/out/seeds/` | **DONE** — see `docs/results/SEED_LIBRARY_RESULT.md`; the library is baked into the image |
| **Part 2 — §C0-§C9** | the **corpus**, ~1500 training pairs, **8 machines x 8 GPUs** | `/out/pairs_npz/` -> `corpus.h5` | **this is the one you are about to run** |

Part 1 is kept because the image, the driver, the Vast fields and the SSH workflow are
shared, and because a seed ever needing to be regenerated is a thing that happens. **If you
are generating the corpus, start at §C0.**

---

# Part 1 — Generating the seed library on Vast.ai

**Pick the image in Vast, SSH in, run one command.** The code is baked in and the tag names
the commit, so a rented machine clones nothing, builds nothing, and cannot pull the wrong
revision. Rationale: `PROJECT_BRIEF.md` (dated block at the top) and `docs/FASTEDDY_TRAPS.md` §23.

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
| **Image path/tag** | `ghcr.io/tyatharva/flux-seeds:bb961bcfc77d-fe0ce48d5dff06` |
| *(or the stable tag)* | `ghcr.io/tyatharva/flux-seeds:latest` |
| *(or the immutable digest)* | `ghcr.io/tyatharva/flux-seeds@sha256:329a4a2c21f8d16b83e13427f3444d7543c53cf551999a16a7a248589ba37afa` |
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
| commit-pinned | `ghcr.io/tyatharva/flux-seeds:bb961bcfc77d-fe0ce48d5dff06` |
| stable | `ghcr.io/tyatharva/flux-seeds:latest` |
| **immutable digest** | `ghcr.io/tyatharva/flux-seeds@sha256:329a4a2c21f8d16b83e13427f3444d7543c53cf551999a16a7a248589ba37afa` |
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

`docs/PLAN.md` records that `TKE_BL/u*^2` and `z_i` cannot be resolved against their thresholds
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

---

# Part 2 — Generating the CORPUS on Vast.ai — 8 machines x 8 RTX 5090

Same image, a different command. The seed library above is **baked into this image**, so a
corpus machine pulls one thing and needs no seed transfer at all.

| | |
|---|---|
| what one machine does | **8 of the 64 corpus months**, ~243 calendar days, as a SHARED QUEUE over its 8 GPUs |
| what you pass | `--machine N`, N = 0..7. Nothing else differs between the eight boxes. |
| what lands in `/out` | one `pairs_npz/case_YYYYMMDDHH.npz` per accepted day, plus `manifest.json` accounting for **every** day |
| how long | measured and reported after the first 5 cases, and again at the end. **Not assumed** — see §C4. |

## C0. The partition — verify it before renting anything

```bash
docker run --rm ghcr.io/tyatharva/flux-seeds:latest \
  python3 -c "import sys;sys.path.insert(0,'/flux');from lpdm.partition import describe;print(describe())"
```

Prints the whole 8 x 8 table with a coverage line **recomputed from the printed rows**, so
"all 64 months exactly once" is checkable without trusting the code. Every machine prints
the same table at startup with its own row marked, so a machine's log is self-describing.

The rule is `sort by (calendar month, year), then index % 8`. Chronological order was tried
first and rejected on its own output: `gcd(8,12) = 4` gave each machine only three distinct
calendar months and left seven of eight missing a season. Under the shipped rule **every
machine holds 8 distinct calendar months, all four seasons, and at most 25% of any one
split** — so losing a machine to a dead rental costs a slice of everything rather than a
season or a split.

## C1. Vast instance — the values to enter

**The exact reference, and it is the same image the seed library came out of:**

```
ghcr.io/tyatharva/flux-seeds:corpus                          <- put this in the template
ghcr.io/tyatharva/flux-seeds:6a225c0ea250-fe0ce48d5dff06     <- the immutable one
digest sha256:19d426e52df6f87275238f7fb435e9cea8702abc3d1e2513ed1d97572a458a79
```

`:corpus` is a moving tag, for convenience in the template. **The SHA tag is what a result
is attributable to** and it names both repositories — flux `6a225c0ea250`, FastEddy
`0ce48d5dff06`. `provenance` inside the container prints both, plus the CUDA version, the
gencode list, and the SASS actually present in the binary. Every machine manifest records
`git_commit` from the image's own `FLUX_COMMIT`, so a record says which code made it after
the box is gone.

**5.33 GB compressed** in the registry; **16.3 GB on disk**, of which 2.2 GB is the code and
the baked 30-seed library.

*(The tag names the commit the image was built from; this file was edited afterwards to
record the tag, so `git log` shows one later doc-only commit. Nothing executable differs.)*

Identical to §1 above with three changes:

| field | value |
|---|---|
| image | `ghcr.io/tyatharva/flux-seeds:corpus` |
| **GPUs** | **8x RTX 5090** (or any 8 of `sm_120`) |
| **Disk** | **60 GB** (100 GB for comfort) — see below |
| **RAM** | **≥ 256 GB.** See below; this is the one spec that is a genuine unknown. |
| CUDA driver | **≥ 580** (the `CUDA >= 13.0` filter), exactly as for the seed run |
| On-start script | **EMPTY.** Nothing auto-runs. |
| Launch mode | SSH |

### Disk: 60 GB is plenty. The HRRR figure in an earlier draft of this file was wrong.

| | |
|---|---|
| image on disk | **16.3 GB** measured (14.1 GB toolchain and CUDA + 2.2 GB code and baked seed library) |
| HRRR on disk | **~0.3 GB.** Herbie is called with `remove_grib=True`, so every subset is deleted the moment it is read — what persists is `.idx` inventories at a few kB each. |
| the corpus itself | **~40 kB per record**, ~190 records = **~8 MB per machine** |
| logs, manifest, progress | a few MB |

**60 GB is comfortable; 100 GB if you want room to not think about it.**

> **CORRECTION.** An earlier draft of this file said each sounding was 407 MB and that a
> machine would accumulate ~77 GB. Both numbers came from measuring `data/hrrr` on this
> workstation, which holds cache written **before** `nlev` was reduced to 20 **and** by runs
> that passed `--keep-grib`. Re-measured against the live archive at `nlev=20`, a sounding
> is **168.6 MB transferred and 0 MB retained**. The disk line was wrong; the transfer is
> the real cost and it is §C1b.

`--keep-hrrr` keeps the GRIBs instead of deleting them (for debugging a fetch). Then a
machine does accumulate ~40 GB and you want **200 GB**.

### C1b. Network transfer — measured, subsetted, and not on the critical path

**Byte-range subsetting IS active.** Every fetch goes through `Herbie.xarray(search=...)`,
which downloads only the `.idx` byte ranges the regex matches — the cache filenames are
`subset_<hash>__hrrr...grib2`, which is Herbie's own name for a range-fetched file.

**But GRIB subsetting is per MESSAGE, and a message is a full CONUS field** (1799 x 1059,
~1.5 MB compressed) whatever a caller wants from it. There is no spatial subsetting to
enable: NOMADS offers a `subregion` GRIB filter, but it holds only the last ~2 days and this
corpus spans 2021–2026, so the historical archive has full messages or nothing. **One grid
point costs a whole CONUS field, and that is inherent.**

Measured against the live archive (2023-07-15 19Z, `nlev=20`):

| fetch | messages | size | time |
|---|---|---|---|
| `nat` — 5 vars x 20 hybrid levels | 100 | **152.8 MB** | 17.7 s |
| `sfc` — 8 surface fields | 8 | 15.1 MB | 2.7 s |
| `prs` — HGT at 700 mb | 1 | 0.7 MB | 0.6 s |
| **one sounding (an accepted day)** | 109 | **168.6 MB** | **~22 s** |
| one screening candidate, **before** | 4 | 9.19 MB | 1.1 s |
| one screening candidate, **now** | 2 | **4.66 MB** | 0.7 s |

**`nat` is 91% of a case.** The only lever on it is `nlev` (levels 1–20 reach ~6.1 km AGL
against a 4 km `z_i` search ceiling), and that is a **sounding-fit input**, not a plumbing
knob — it is left alone rather than trimmed to save bandwidth.

**What did change: the screen no longer fetches the 10 m wind.** `lpdm/corpus.py:screen()`
reads `hpbl`, `shtfl` and `dz_i/dt` and nothing else; the wind is a *label* on the accepted
hour. Fetching it for every candidate cost 4.53 MB a time on days that reject 24 hours and
yield nothing — and **the screening term is dominated by exactly those days** (an accepted
day costs ~8 fetches, an exhausted one **26**, both measured). It is now fetched once, on
the hour that is accepted. Verified against the archive: **identical `hpbl` and `shtfl`,
49% fewer bytes per candidate.**

| | accepted day | missing day | **per machine (243 days, 80% yield)** |
|---|---|---|---|
| before | 242 MB | 239 MB | **58.7 GB** |
| **now** | **210 MB** | **121 MB** | **46.8 GB** |
| saved | 32 MB | 118 MB | **11.9 GB (20%)** |

**~47 GB per machine, ~374 GB across all eight.**

### Is download serial with compute? Yes, and it is ~2-3% of wall time.

A worker does `pick_hour` (network) then `get_case` (compute) for its day, so the fetch is
serial *within that day*. The ratio is what matters:

| | |
|---|---|
| network, serial per machine | **2.4 h** (accepted day 34 s, missing day 39 s) |
| compute, 194 cases over 8 GPUs at 0.4–0.61 GPU-h/case | 9.7–14.8 h |
| **network share of wall time** | **2.0–3.0%** |
| average bandwidth across the run | **~1.1 MB/s = 9 Mbit/s** per machine |
| peak, if all 8 workers fetch a sounding at once | ~61 MB/s = **491 Mbit/s**, briefly |

So it is negligible against compute and the sustained draw on a shared Vast link is
single-digit Mbit/s. The only thing that would sting is eight workers hitting `nat`
simultaneously, which is transient and self-desynchronises within the first few days.

**The compute side of that ratio is not yet measured** — 0.4–0.61 GPU-h/case brackets it
from the LES cost and the ninth pass's per-case figure. The early report (§C4) prints the
real one after 5 cases.

### RAM: ≥256 GB, and the run reports what it actually used

`PROJECT_BRIEF.md` records **12.45 GB peak host RSS for ONE corpus case** — the LPDM's 12.0 GB
fp16 field cache, which is not buildup but the window itself, random-accessed by a CPU
integrator. **That has never been measured 8-way.** Eight concurrent cases could approach
100 GB, and a box that starts swapping does not fail — it runs several times slower with
nothing in the output to say why.

So: **rent ≥ 256 GB for the first machine**, and read the early report (§C4) before renting
the other seven. If it comes in well under, size the rest down. If it swaps, drop to
`--gpu-count 4` and say so in the manifest.

## C2. Launch procedure — paste this

```bash
# --- on your machine ----------------------------------------------------------------
ssh -p <PORT> root@<HOST>

# --- on the instance ----------------------------------------------------------------
verify                                  # SASS vs the cards, then a 200-step run. Seconds.

# The partition, and the days THIS box owns. No work, no GPU.
run_corpus --machine 0 --dry-run

# The run itself. nohup so an SSH drop cannot take it with you.
nohup run_corpus --machine 0 --out /out > /out/nohup.log 2>&1 &

# Watch it. Separate process reading /out/progress.json -- kill and restart it freely.
corpus_progress
```

**`--machine` is the ONLY thing that differs between the eight boxes.** Machine 0 through
machine 7, one each.

`corpus_progress` shows, refreshed in place: a progress bar over the machine's days, the
case/missing/failed counts, **a live ETA**, running mean GPU-h per case, machine-wide peak
host RSS, and **per-GPU current month, day and pipeline stage**. Ctrl-C it whenever; the run
neither knows nor cares.

**The ETA is computed from the RECENT completion rate**, over a trailing window of the last
third of completed days (floored at 12, capped at 120), not from the run average — the queue
walks the machine's months in order and a winter month that rejects most of its days runs
about twice as fast per day as a July that yields a case from nearly every one, so a
run-average ETA lags the whole way through. It is withheld below 5 completed days rather
than guessed, and resumed days are excluded from the rate (they resolve in milliseconds off
the disk and would otherwise collapse a restarted machine's ETA toward zero). The one-shot
projection from the early report is shown beside it, labelled, because that is the number
the rental decision was taken on.

## C3. What it does per day

Drawn without replacement from the day's round hours, seeded from the date alone so a
re-run reproduces the same selection:

1. draw an hour uniformly from what is left of the pool
2. screen it: `z/L < 0`, `z_i` in 300–1250 m, `|dz_i/dt| < 15 %/h`
3. rejected -> the hour is spent anyway; draw again. Pool exhausted -> **DAY MISSING with a
   reason**, logged, queue continues.
4. accepted -> sounding from the HRRR analysis valid at **exactly that hour**, seed from the
   **whole 30-seed library**, 1.25 sim-h, footprint over the last 30 minutes, **one npz**.

2026-08-31 is capped at 12 UTC (`lpdm/corpus.py:HOUR_CAPS`) because the later analyses do
not exist.

**Why a shared queue and not one month per GPU.** Wall time is set by the slowest worker,
and the months are not equal — a month's cost is its ACCEPTED days, and acceptance is
meteorological. Measured on the stubbed dry run's own yields: a rigid month-per-GPU
assignment finishes at its busiest month, the queue at the mean, **16% of wall time**, and
the queue kept all eight workers within **1.4%** of each other. On calendar days alone the
figure would be 2%; the rest is yield, which is the part a pinned assignment cannot see.

## C4. The early report — read this before renting the other seven

After the first **5 cases** the run prints, and puts in `progress.json`:

```
  EARLY REPORT after 5 case(s) -- the numbers the next rental turns on
    GPU-h per case (occupancy)   : 0.xxx   [8 concurrent]
    peak container RSS           : xx.x GB of xxx GB
    MemAvailable low-water       : xxx GB
    swap used (peak)             : x.xx GB
    projected finish             : x.xx h for 243 days
```

- **GPU-h per case is OCCUPANCY**, wall clock x 1 GPU, not FastEddy kernel time. A case
  holds its card through the CPU-bound LPDM too, and occupancy is what the rental bills.
- **If MemAvailable falls below 12% of total, or swap is touched, it says so loudly** and
  puts an alert in `progress.json`. **Not a gate** — with ≥256 GB the 8-way field cache
  (~100 GB worst case) has room. It is recorded so the number exists.
- **If the projection exceeds `--max-hours` (default 12) it prints a `!!!!` block** naming
  the number. It keeps running — stopping wastes the rental too — and `--abort-on-overrun`
  stops instead if you would rather it did.

**Do not carry the seed run's 0.189 GPU-h/sim-h to this.** A seed runs FastEddy and nothing
else; a case also runs the LPDM and the ring. That is exactly what this report measures.

## C5. Resume — a dead machine costs one day, not eight months

The artifacts on the mounted volume ARE the checkpoint. Re-run the identical command:

```bash
nohup run_corpus --machine 0 --out /out > /out/nohup.log 2>&1 &
```

- a day whose record is in `pairs_npz/` is **skipped** and keeps its original timing in the
  manifest — a resumed pass never overwrites what the pass that did the work measured
- a day recorded MISSING is **not re-drawn** (the draw is seeded from the date, so it would
  reach the same answer). `--retry-missing` re-evaluates them — use it if a day was lost to
  a network outage rather than to the weather.
- a **failed** day is retried automatically on the next run, and the summary lists them

If the *instance* is destroyed, mount the same volume on a new one and run the same command.
If the volume is gone, pull `/out` first (§C6) and re-run against a directory holding it.

## C6. Getting the corpus back

```bash
# the corpus itself -- ~8 MB per machine
rsync -avP -e 'ssh -p <PORT>' root@<HOST>:/out/pairs_npz/ ./corpus/machine0/pairs_npz/

# the accounting: manifest, log, progress. A few MB.
rsync -avP -e 'ssh -p <PORT>' root@<HOST>:/out/manifest.json root@<HOST>:/out/run_corpus.log \
      ./corpus/machine0/

# all eight, into one place -- the filenames are globally unique (case_YYYYMMDDHH)
for m in 0 1 2 3 4 5 6 7; do
  rsync -avP -e "ssh -p ${PORT[$m]}" root@${HOST[$m]}:/out/pairs_npz/ ./corpus/pairs_npz/
  rsync -avP -e "ssh -p ${PORT[$m]}" root@${HOST[$m]}:/out/manifest.json ./manifests/machine$m.json
done
```

**Total expected size: ~60 MB for the whole corpus.** ~1500 records at ~40 kB. The eight
manifests add a few MB. That is the entire deliverable — everything else is scratch and is
deleted by `bin/get_case.sh` on its way out, including on failure.

Check what came back before destroying anything:

```bash
python3 bin/check_npz.py corpus/pairs_npz/*.npz --quiet     # every record against the schema
python3 -c "
import json,glob,collections
c=collections.Counter(); d=collections.Counter()
for p in sorted(glob.glob('manifests/*.json')):
    m=json.load(open(p))
    for v in m['days'].values(): c[v['status']]+=1
    d[m['machine']]=m['counts']
print(c); [print(' machine',k,v) for k,v in sorted(d.items())]"
```

## C7. It is verified without a GPU

```bash
bash bin/test_corpus_machine.sh
```

Runs in ~15 s and checks, from the artifacts rather than from the design: the 64-month
partition covers every month exactly once across `--machine 0..7`; all 8 of a machine's
months are walked and every calendar day is accounted for; **every worker takes work and
they take UNEVEN numbers of days** (the queue actually rebalances); worker busy time is a
sane measurement and they finish within 25% of each other; the progress file renders in the
separate viewer; a resume marks every day resumed while the manifest still says what each
day IS; one record per accepted day and no others; and every record passes the npz schema
check *and* is refused as a corpus record because it carries `meta.stub`.

The LES, the LPDM, the ring and the footprint do not run in it — those are validated
elsewhere and none of them is a scheduling question. **The stub deliberately does not return
instantly**: a few milliseconds, varied per case from the case's own hash. With an instant
stub every worker finishes at the same moment and a rigid month-per-GPU assignment would
produce an identical timeline, so the rebalancing claim would be untested while looking
tested.

## C8. Things that are true and easy to be surprised by

- **A stubbed record can never be mistaken for a corpus record.** `--stub` stamps
  `meta.stub = true` and `bin/check_npz.py` refuses it unless asked for one. The stub paths
  also require `FLUX_STUB=1` in the environment, so no ordinary corpus command can reach
  them.
- **HRRR is byte-range subsetted, and a subset is deleted as soon as it is read.** What
  costs is that a GRIB message is a full CONUS field however little of it you want — 168.6
  MB per case, of which 91% is the 100 hybrid-level messages. See §C1b.
- **The seed library is in the image, not on the volume.** All 30 seeds, because a case
  picks its seed per case. The image build asserts 30 restarts of identical size plus each
  seed's `manifest.json` and `stationarity.json` — a partial library would otherwise be an
  unexplained failure hours into a rental.
- **Seed selection uses the whole library** (`ALLOW_DRIFTING=any`, 2026-08-31). Every pair
  still carries `seed.gate_state`. See `docs/results/SEED_LIBRARY_RESULT.md`.
- **`--only-month` refuses a month this machine does not own**, by name, rather than running
  it — otherwise two boxes would generate the same days.
- **The manifest describes the corpus, not the pass.** A resumed day is recorded as
  `case`/`missing` with `resumed: true`, never as a status of its own, and a day this pass
  has not reached keeps what an earlier pass found.
- **`nohup`, not `tmux`.** The run writes `progress.json` and needs no terminal. `tmux`
  works too if you prefer it.

## C9. Consolidate the eight machines into one training file

§C6 brings the records down; nothing merges them. This does, and it is the last step before
any ML.

```bash
# from the repo root, with corpus/pairs_npz/ and corpus/manifests/ as §C6 left them
docker run --rm -v "$PWD":/w -w /w ghcr.io/tyatharva/flux-seeds:corpus \
    python3 bin/consolidate_corpus.py \
        --npz-dir corpus/pairs_npz --manifests corpus/manifests --out corpus.h5
```

(The host python has no h5py; the analysis stack lives in the image, as it does for
everything else in this project.)

### What comes out

```
corpus.h5
  scalars          (N, 6)         float32   h, ustar, sigma_v, L, sin_wdir, cos_wdir
  kljun            (N, 128, 128)  float32   chunked (32,128,128), gzip-4 + shuffle
  target           (N, 128, 128)  float32   signed and unclipped
  meta/            datetime, parent_case, run_id, split, split_index, gate_state,
                   integral, peak_x_m, centroid_dist_m, array_share, zi_achieved_m,
                   inv_L, wdir_deg, seed_job, seed_rot, git_commit, kljun_source,
                   h_estimator, scalar_names, valid_mask
  grid/            n=122, pad=3, n_padded=128, dx_m=30, domain_m=3660, receptor_z_m=30
  norm/            scalars_mean, scalars_std, kljun_scale, target_scale  (TRAIN ONLY)
  counts/          cases and missing days per split, and what the manifests expected
```

**Expected size: ~110 MB** for ~1469 records at `N_WINDOWS = 1`, ~215 MB at 2.
**Measured** on real records: **73 kB per record**, 1.8x compression against the raw
arrays (1.6x on stubs, whose rasters are smoother). Compression is gzip-4 with the byte
**shuffle** filter, which is worth 20% for nothing — measured 1.32x without it, 1.65x with,
and gzip-9 buys another 1% for 3x the write time.

### It refuses rather than warns, and each refusal is one that training would hide

| refusal | why it is fatal |
|---|---|
| **split disagreement** | the split is re-derived from each record's own datetime via `lpdm/corpus.py:split_of` and compared with the split it was generated under. A mismatch means the train/test boundary is not where anyone thinks, and **a good validation score is exactly what that looks like** |
| **duplicate record** | keyed on `run_id`. Each month belongs to one machine, so a duplicate means two boxes ran the same `--machine`. (Two windows of one case share a `parent_case` and are *not* this — they differ in `run_id`, and the count check compares distinct cases.) |
| **`meta.stub`** | a record whose LES and LPDM were an analytic blob. `--allow-stub` exists only to test this script and stamps `h.attrs["stub"]` on the output |
| **count mismatch** | records on disk vs what the eight manifests account for, reported **per machine** so it is obvious which `rsync` to repeat |
| non-finite cells, wrong shape, non-zero pad, wrong `dx` | a record off a retired grid is complete, plausible and wrong |

### Two things it decides, and both are decisions rather than defaults

**Normalisation is computed on TRAIN only.** Statistics over the whole corpus leak val and
test into the input scaling — a small leak, and the kind that never appears as a failure,
only as a uniformly better score with nothing pointing at why. `norm/` records
`computed_on = "train split only"` and `n_train`.

The rasters use `y = arcsinh(x / s)`, **signed, unclipped**, because a footprint spans
orders of magnitude and its negative lobes are physical — 5.8–11.1% of |flux| (`PROJECT_BRIEF.md`).
`s` is the median **over train records of each record's peak |x|** inside the valid frame.
Not the median over cells: that was the first version and it returned **s = 7.9e-23**,
because a footprint is a compact blob in a 122² frame and the median *cell* is deep in a
tail orders of magnitude below the peak.

**The pad extent is recorded so the loss can mask it.** 122 → 128 is a zero-pad of 3 cells,
not a resize; those 1,500 border cells are structural zero on both channels.
`meta.attrs["pad_cells"] = 3` and a ready-made `meta/valid_mask` (14,884 of 16,384 cells
true) are both stored, so a consumer cannot get them inconsistent. **A loss averaged over
the full frame reports a number 9.2% of which is the model learning to emit zero where it
was told to.**

### Check what it built

```bash
docker run --rm -v "$PWD":/w -w /w ghcr.io/tyatharva/flux-seeds:corpus python3 -c "
import h5py
with h5py.File('/w/corpus.h5') as h:
    print(h.attrs['format'], h.attrs['n'], 'records, stub =', h.attrs['stub'])
    print({k: int(v) for k, v in h['counts'].attrs.items() if k.startswith('cases_')})
    print('norm computed on:', h['norm'].attrs['computed_on'], h['norm'].attrs['n_train'])"
```
