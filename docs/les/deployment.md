# Deployment on rented GPUs

Two campaigns share one image, one driver and one SSH workflow, and were run weeks apart.

| | what it makes | machines | time | status |
|---|---|---|---|---|
| **Part 1** | the 30-seed library | 1 × 16 RTX 5090 | about 1 h | done; the library is baked into the corpus image |
| **Part 2** | the corpus, 1366 pairs | 8 × 8 RTX 5090 | about 12 h | done 2026-09-01 |

Both ran on [Vast.ai](https://vast.ai). Pick the image, SSH in, run one command. The code is
baked in and the tag names the commit, so a rented machine clones nothing, builds nothing, and
cannot pull the wrong revision.

## The image

Built by `docker/build_image.sh` from `Dockerfile.blackwell` (CUDA 13.0.1, real SASS for
`sm_75 … sm_120`, no PTX; see [environment](../getting-started/environment.md)). The tag is
`<flux-commit>-fe<fasteddy-id>`, and `provenance` inside a container prints both plus the
CUDA version, the gencode list and the SASS actually present in the binary. Every machine
manifest records `git_commit` from the image's own `FLUX_COMMIT`, so a record says which code
made it after the box is gone.

| purpose | reference |
|---|---|
| the seed library run | `ghcr.io/tyatharva/flux-seeds:bb961bcfc77d-fe0ce48d5dff06`, digest `sha256:329a4a2c21f8d16b83e13427f3444d7543c53cf551999a16a7a248589ba37afa` (13.1 GB on disk, 4.05 GB compressed, 22 layers) |
| **the corpus run** | `ghcr.io/tyatharva/flux-seeds:7de9dee2a01d-fe0ce48d5dff06`, digest `sha256:3f58d049d895178e9a9035e9317d6a11582f9002dc801be3e2dd7a20430e8404` (16.3 GB on disk, 5.33 GB compressed; 2.2 GB of it is the code and the baked seed library) |

!!! warning "Do not use the moving tags"
    `:latest` and `:corpus` on GHCR point at other builds (`:corpus` at an image built from
    an unmerged branch, see [unmerged work](../history/unmerged-producer-consumer.md)). Use
    the commit-pinned tag or the digest. The commit in the tag belongs to the pre-rewrite
    history; the same tree is tagged `pre-cleanup-2026-09-04` in this repository.

The GHCR package must be **public** for Vast to pull it anonymously (an anonymous manifest
fetch returns 403 while it is private). GitHub exposes package visibility only through the
web UI: package settings → Danger Zone → Change visibility → Public. Check:

```bash
T=$(curl -s "https://ghcr.io/token?scope=repository:tyatharva/flux-seeds:pull&service=ghcr.io" \
    | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $T" \
  -H "Accept: application/vnd.oci.image.manifest.v1+json" \
  https://ghcr.io/v2/tyatharva/flux-seeds/manifests/7de9dee2a01d-fe0ce48d5dff06     # 200 = pullable
```

Rebuild and push:

```bash
bin/fetch_assets.sh seeds          # the 30 restarts must be at seeds/*/return/seed_restart.nc first
docker/build_image.sh              # -> flux-seeds:<flux>-fe<fasteddy>
TAG=$(docker images --format '{{.Tag}}' flux-seeds | grep -v latest | head -1)
docker tag flux-seeds:$TAG ghcr.io/tyatharva/flux-seeds:$TAG && docker push ghcr.io/tyatharva/flux-seeds:$TAG
```

### Nothing auto-runs

The image has no `ENTRYPOINT` (explicitly cleared: the nvidia base image's is inherited
otherwise). Vast's SSH launch mode overrides entrypoints, and the previous one printed help
and exited, which on Vast is an instance that comes up and dies. `CMD` is `sleep infinity`.
The subcommands are real executables on `PATH`: `run_seeds`, `run_corpus`, `corpus_progress`,
`verify`, `seed`, `accept`, `provenance`. `FLUX_ROOT`, `FLUX_NATIVE=1` and `FLUX_OUT` are in
the image ENV **and** in `/etc/environment`, because an SSH login shell inherits neither the
image ENV nor `docker run -e`. `openssh-server` is installed because Vast requires it.
Logging in prints the provenance and the two commands.

The same image works on a workstation:

```bash
docker run --gpus all -v /out:/out ghcr.io/tyatharva/flux-seeds:<tag> run_seeds --gpu-count 16
```

Everything in `/out` is written as root: the image declares no `USER` because OpenMPI refuses
to launch as root without `OMPI_ALLOW_RUN_AS_ROOT`, and on Vast you are root anyway.

## Part 1: the seed library

### Instance

| field | value |
|---|---|
| image | the commit-pinned tag or digest above |
| launch mode | SSH |
| on-start script | empty |
| Docker options | empty (everything is in the image ENV; Vast accepts no bind mounts) |
| disk | 100 GB (image 13.1 GB + 30 seeds × 24 dumps × 73.3 MB = 54 GB in `/out/work` + 2.2 GB deliverables + about 6 GB scratch = about 75 GB; 40 GB suffices with `--prune-dumps`) |
| GPU | 16 × RTX 5090 |
| **CUDA filter** | **≥ 13.0**, a hard requirement |
| RAM | ≥ 64 GB (floor 32 GB) |

**CUDA ≥ 13.0 is a driver filter and getting it wrong wastes the rental.** The image is a
CUDA 13.0.1 binary; NVIDIA's minimum Linux driver for 13.0 is 580.65.06. A 5090 box with a
570 driver will not run it, even though the binary carries `sm_120` SASS: the toolkit major
version is what the driver must satisfy, and the failure is at image load. The image ships the
forward-compatibility package (`libcuda.so.580.82.07`), but that is supported on data-centre
GPUs, not GeForce. If only 570 boxes exist, rebuild on CUDA 12.8 (`FE_GENCODE` unchanged;
12.8 also targets `sm_120`).

**RAM, measured on the image**: the LES phase peaks at 0.82 GB container RSS (FastEddy itself
0.49 GB over 2 processes); the acceptance battery at 1.51 GB; 16-way worst case about 24 GB.
The 12.45 GB figure elsewhere is a *corpus case* (the LPDM's 12.0 GB field cache); a seed
runs no LPDM. Sizing the seed box on the case number over-rents by 4×; sizing the corpus box
on the seed number swaps it to death.

A seed needs no data mount: it is a flat, uniform, doubly-periodic spin-up with an empty
`topoFile` and an empty `inFile`, so no sounding, terrain or land cover is read.

### Launch

```bash
ssh -p <PORT> root@<HOST>
provenance                       # which commit, which CUDA, which architectures
verify                           # SASS vs the cards, then a 200-step run. Seconds. STOP if not PASS.
nvidia-smi -L | wc -l            # confirm 16 cards are attached
run_seeds --gpu-count 16 2>&1 | tee /out/run_seeds.console.log
```

There is no staged single-GPU step: per-seed GPU-h is measured from the running queue and
reported in the summary. `run_seeds` first sweeps the CUDA thread-block shape on the first GPU
(1 min 54 s measured, 14 of 106 legal shapes tried, the other 92 named rather than silently
dropped), then fans all 30 seeds across the cards. The sweep is a pure performance knob: the
one reduction that accumulates is templated on compile-time constants.

Measured: 0.936 h wall, 13.24 GPU-h for all 30 seeds on 16 cards, 0.189 GPU-h per simulated
hour at 16-way against 0.469 single-GPU in the same image. Contention cost nothing; it was
2.5× faster. **Do not carry 0.189 to a corpus estimate**: a seed runs FastEddy and nothing
else, a case also runs the LPDM and the ring.

It is resumable. Re-running the same command skips seeds whose work directory holds a
finished `seed_restart.nc` and its acceptance battery, and restarts partial ones from step 0.
Each seed runs in its own subprocess; a crash, a gate failure or a timeout is recorded with a
reason and the queue moves on.

| option | |
|---|---|
| `--gpu-count N` / `--gpus 0,3,7` | 0 (default) uses every visible GPU |
| `--pass N/M` | split the library across several machines; one machine needs no pass, it is a queue |
| `--only <job,…>` | a smoke test on named seeds |
| `--ceiling-h 2.0` | simulated-hour ceiling per seed |
| `--no-sweep` | keep the `.in`'s measured `1x2x64` |
| `--prune-dumps` | delete each seed's `output/` on success (about 1.8 GB each) |
| `--dry-run`, `--force`, `--stub`, `--assume-gpus` | list, re-run finished seeds, scheduler self-test only |

### What lands in `/out`, and getting it back

| | |
|---|---|
| `seeds/<job>/seed_restart.nc` | the seed, 73.3 MB, all 22 variables |
| `seeds/<job>/` | `stationarity.json`, `manifest.json` (with `achieved` and `run.ceiling_steps`), `acceptance.txt`, `seed_report.json`, `turb_alive.json`, `rotation_check.json`, `direction_drift.txt`, logs |
| `machine_manifest.json` | every seed: verdict, GPU-h, stop time, per-job timeline, peak VRAM, host memory |
| `threadblock_sweep.json`, `direction_drift_library.txt`, `run_seeds.log` | the sweep, the Ekman backing over the library, the timestamped log |
| `work/<job>/` | working directories, dumps included |

```bash
rsync -avP -e "ssh -p <PORT>" root@<HOST>:/out/seeds/ ./returned-seeds/
rsync -avP -e "ssh -p <PORT>" root@<HOST>:/out/{machine_manifest.json,threadblock_sweep.json,direction_drift_library.txt,run_seeds.log} ./results/seed_library/
for d in ./returned-seeds/seed_*; do n=$(basename "$d"); mkdir -p "seeds/$n/return" && cp -a "$d"/* "seeds/$n/return/"; done
```

`/out` is a directory in the container filesystem. Vast stops and starts a container rather
than recreating it, so `/out` survives a stop/start and a restart. It does not survive
destroying the instance. Pull the seeds down first.

The run reports, and writes to `machine_manifest.json`: measured GPU-h per seed (min, median,
max, total, from paired values only, naming any seed excluded for want of one); GPU-h per
simulated hour against the 0.469 single-GPU reference; the Steinfeld accelerator's separate
log for neutral rungs (counting only `run.log` under-reports a neutral seed by about 29%);
peak host RSS three ways; stop times against the 2.0 sim-h ceiling; peak VRAM per GPU;
`k0/k1` per seed (a run whose `k0/k1` fails never produces a seed); and a queue audit from the
recorded per-job timeline. `accepted` does not mean every gated limit resolved:
INDETERMINATE is the library's normal state (see [seed library](seed-library.md)).

### The scheduler is verified, not asserted

Both tests run in about a minute on no GPU against the shipped image.

`bin/test_work_queue.sh`: 20 jobs over 6 fabricated workers with the LES stubbed
(`STUB_SEED=1` replaces FastEddy, the gate and the battery; every artifact is stamped
`stub: true`; `--assume-gpus` refuses to run without `--stub`). Every check reads the recorded
timeline: every job recorded exactly once; a worker took a second job after its first (14
hand-offs, landing on the 3 s stub duration exactly: 3.1 s, 6.2 s, 9.3 s); the deliberately
failed jobs failed and released their workers; peak concurrency never exceeded the workers;
the run ended only when the queue was empty; no stub was counted as an accepted seed.

`bin/test_gpu_mutex.sh`: eight launches released from one fifo barrier at device 0 reach the
per-GPU guard simultaneously. A `/proc` scan is check-then-act and passes a sequential test
while failing this one silently, so the guard is an `flock(2)` held for the life of the run.
Exactly one launched FastEddy; seven were refused with exit 2 naming the device. The stub is a
compiled ELF at a path ending in `FEMAIN/FastEddy`, because the guard resolves
`/proc/<pid>/exe`.

Also true and easy to be surprised by: an out-of-range `CUDA_VISIBLE_DEVICES` fails loudly at
exit 100 (`gpuErrchk` prints `no CUDA-capable device is detected` before the no-device branch
is reached); bitwise reproducibility does not hold across architectures and is not sought.

## Part 2: the corpus

Same image, a different command. The seed library is baked in, so a corpus machine pulls one
thing and needs no seed transfer. One machine does 8 of the 64 corpus months (about 243
calendar days) as a shared queue over its 8 GPUs, and writes one `pairs_npz/case_YYYYMMDDHH.npz`
per accepted day plus a `manifest.json` accounting for every day.

### The partition

```bash
docker run --rm ghcr.io/tyatharva/flux-seeds:<tag> \
  python3 -c "import sys;sys.path.insert(0,'/flux');from lpdm.partition import describe;print(describe())"
```

Prints the whole 8 × 8 table with a coverage line recomputed from the printed rows. The rule
is `sort by (calendar month, year), then index % 8`. Chronological round-robin was tried first
and rejected on its own output: `gcd(8, 12) = 4` gave each machine only three distinct
calendar months and left seven of eight missing a season. Under the shipped rule every
machine holds 8 distinct calendar months, all four seasons, and at most 25% of any one split,
so losing a machine costs a slice of everything rather than a season or a split. Total and
disjoint are asserted at import in `lpdm/partition.py`.

### Instance

Identical to Part 1 with three changes: **8 × RTX 5090**, **disk 60 GB** (100 GB for
comfort), **RAM ≥ 256 GB**.

Disk: 16.3 GB image, about 0.3 GB of HRRR (Herbie is called with `remove_grib=True`, so every
subset is deleted the moment it is read; only `.idx` inventories persist), about 40 kB per
record, a few MB of logs. An earlier estimate of 77 GB per machine came from a workstation
cache written before `nlev` was cut to 20 and by runs that kept the GRIBs. `--keep-hrrr`
keeps them (then plan on 200 GB).

RAM: one corpus case peaks at 12.45 GB host RSS (the LPDM's 12.0 GB fp16 field cache,
random-accessed by a CPU integrator). Eight concurrent cases could approach 100 GB, and a box
that starts swapping does not fail; it runs several times slower with nothing in the output to
say why. Rent ≥ 256 GB for the first machine and read the early report before renting the
other seven.

### Network transfer, measured

Byte-range subsetting is active (every fetch goes through `Herbie.xarray(search=...)`), but
GRIB subsetting is per message, and a message is a full CONUS field (1799 × 1059, about
1.5 MB compressed). There is no spatial subsetting for a historical archive; one grid point
costs a whole field.

| fetch | messages | size | time |
|---|---|---|---|
| `nat`, 5 variables × 20 hybrid levels | 100 | 152.8 MB | 17.7 s |
| `sfc`, 8 surface fields | 8 | 15.1 MB | 2.7 s |
| `prs`, HGT at 700 mb | 1 | 0.7 MB | 0.6 s |
| **one sounding (an accepted hour)** | 109 | **168.6 MB** | about 22 s |
| one screening candidate | 2 | 4.66 MB | 0.7 s |

`nat` is 91% of a case; the only lever is `nlev`, a sounding-fit input, left alone. The screen
reads only `hpbl`, `shtfl` and `dz_i/dt`; the 10 m wind is fetched once, on the accepted hour
(before that change a candidate cost 9.19 MB). An accepted day costs about 8 fetches and an
exhausted one 26, so the screening term is dominated by days that yield nothing: 210 MB per
accepted day, 121 MB per missing day, **about 47 GB per machine**, 374 GB across eight. The
network is 2.4 h serial per machine against 9.7–14.8 h of compute: 2–3% of wall time, about
1.1 MB/s sustained, with brief 61 MB/s peaks if all eight workers fetch a sounding at once.

### Launch, watch, resume

```bash
ssh -p <PORT> root@<HOST>
verify                                  # SASS vs the cards, then a 200-step run
run_corpus --machine 0 --dry-run        # the partition and the days THIS box owns; no work
nohup run_corpus --machine 0 --out /out > /out/nohup.log 2>&1 &
corpus_progress                         # a separate process reading /out/progress.json
```

`--machine N` (0–7) is the only thing that differs between the eight boxes. `corpus_progress`
shows, refreshed in place, a progress bar over the machine's days, the case/missing/failed
counts, a live ETA, the running mean GPU-h per case, machine-wide peak host RSS, and per-GPU
current month, day and pipeline stage. The ETA is computed from the recent completion rate
over a trailing window of the last third of completed days (floored at 12, capped at 120),
because the queue walks the months in order and a winter month that rejects most of its days
runs about twice as fast per day as a July. It is withheld below 5 completed days, and resumed
days are excluded from the rate.

**Per day** (the draw and the screen are described in [case generation](case-generation.md)):
draw an hour without replacement from the day's 24, screen it (`z/L < 0`, `z_i` in 300–1250 m,
`|dz_i/dt| < 15 %/h`), spend it either way; an exhausted pool is a missing day with a reason;
an accepted hour runs the 1.25 sim-h case and writes one `.npz`. 2026-08-31 is capped at
12 UTC because the later analyses do not exist.

**Why a shared queue and not one month per GPU.** Wall time is set by the slowest worker, and
a month's cost is its accepted days, which is meteorological. Measured on the stubbed dry
run's own yields: a rigid month-per-GPU assignment finishes at its busiest month, the queue at
the mean, a 16% wall-time difference, with all eight workers within 1.4% of each other.

**The early report**, after 5 cases, printed and put in `progress.json`: GPU-h per case as
*occupancy* (wall clock × 1 GPU, which is what the rental bills), peak container RSS,
`MemAvailable` low-water mark, swap peak, projected finish. It says so loudly if
`MemAvailable` falls below 12% of total or swap is touched, and prints a `!!!!` block if the
projection exceeds `--max-hours` (default 12); `--abort-on-overrun` stops instead.

**Resume**: re-run the identical command. A day whose record is in `pairs_npz/` is skipped and
keeps its original timing in the manifest (a resumed pass never overwrites what the pass that
did the work measured). A day recorded missing is not re-drawn (`--retry-missing` re-evaluates
it, for days lost to a network outage rather than to the weather). A failed day is retried
automatically. If the instance is destroyed, mount the same volume on a new one; if the volume
is gone, pull `/out` first and re-run against a directory holding it.

### Getting the corpus back, and consolidating it

```bash
for m in 0 1 2 3 4 5 6 7; do
  rsync -avP -e "ssh -p ${PORT[$m]}" root@${HOST[$m]}:/out/pairs_npz/ ./corpus/pairs_npz/
  rsync -avP -e "ssh -p ${PORT[$m]}" root@${HOST[$m]}:/out/manifest.json ./corpus/provenance/manifests/machine$m.json
done
python3 bin/check_npz.py corpus/pairs_npz/*.npz --quiet      # every record against the schema
```

About 60 MB for the whole corpus. Then, in the analysis image (the host Python has no h5py):

```bash
docker run --rm -v "$PWD":/w -w /w ghcr.io/tyatharva/flux-seeds:<tag> \
    python3 bin/consolidate_corpus.py --npz-dir corpus/pairs_npz \
        --manifests corpus/provenance/manifests --out corpus/corpus_raw.h5
docker run --rm -v "$PWD":/w -w /w ghcr.io/tyatharva/flux-seeds:<tag> python3 bin/mask_cone.py   # -> corpus_cone.h5
```

`consolidate_corpus.py` writes `scalars`, `kljun` and `target` (chunked (32, 128, 128), gzip-4
with the byte-shuffle filter: 1.32× compression without shuffle, 1.65× with, and gzip-9 buys
1% for 3× the write time; measured 73 kB per record), `meta/`, `grid/`, `norm/` and `counts/`.
It **refuses** rather than warns on: a split disagreement (the split is re-derived from each
record's own datetime and compared with the one it was generated under; a mismatch means the
train/test boundary is not where anyone thinks, and a good validation score is exactly what
that looks like); a duplicate `run_id` (two boxes ran the same `--machine`); a `meta.stub`
record; a count mismatch between disk and the manifests, reported per machine; non-finite
cells, wrong shape, non-zero pad, wrong `dx`.

Two decisions inside it. Normalisation is computed on train only (`norm/` records
`computed_on = "train split only"` and `n_train`). The rasters use `y = arcsinh(x / s)`, signed
and unclipped, with `s` the median over train records of each record's peak |x|. Not the
median over cells: that was the first version and it returned `s = 7.9e-23`, because a
footprint is a compact blob in a 122² frame and the median cell is deep in the tail. The pad
extent is recorded (`pad_cells = 3`, `meta/valid_mask` with 14,884 of 16,384 cells true) so
the loss can mask it: a loss averaged over the full frame is 9.2% the model learning to emit
zero where it was told to.

### Verified without a GPU

`bash bin/test_corpus_machine.sh` runs in about 15 s and checks, from the artifacts: the
64-month partition covers every month exactly once across `--machine 0..7`; all 8 of a
machine's months are walked and every calendar day is accounted for; every worker takes work
and they take uneven numbers of days (the queue rebalances); worker busy times are sane and
within 25% of each other; the progress file renders in the viewer; a resume marks every day
resumed while the manifest still says what each day *is*; one record per accepted day and no
others; every record passes the schema check *and* is refused as a corpus record because it
carries `meta.stub`. The stub deliberately does not return instantly (a few ms, varied per case
from its hash), otherwise every worker would finish at once and the rebalancing claim would
be untested while looking tested.

That dry run stubs the screener and the case, so it opens no file the case path reads; it
passed while the image had no `.in` template. The check that closes the gap is one real case
with only the LES and LPDM stubbed, about 4 minutes on a CPU with HRRR access:

```bash
STUB_LES=1 bin/run_corpus_case.sh 2023-01-18T18:00 stubcheck_20230118
```

It fetches the real sounding, fits the base state, builds the per-case surface, picks a seed
from the library, prepares the restart, and writes a record that `check_npz.py` refuses as a
stub. The image build additionally asserts every case-path input by name (`runs/g30_base/base.in`
with `Nz = 122`, the three surface maps, `results/sigma_w_curve_30m.json`, the FFP, `liblpdm.so`),
30 restarts of identical size, and a 2400 MB ceiling on the whole `/flux` tree.

### Easy to be surprised by

- A stubbed record can never be mistaken for a corpus record: `meta.stub = true` is refused
  by `check_npz.py`, and the stub paths also require `FLUX_STUB=1` in the environment.
- The seed library is in the image, not on the volume, because a case picks its seed per case.
- Seed selection uses the whole library (`ALLOW_DRIFTING=any`); every pair carries
  `seed.gate_state`.
- `--only-month` refuses a month this machine does not own, otherwise two boxes would generate
  the same days.
- The manifest describes the corpus, not the pass: a resumed day is `case` or `missing` with
  `resumed: true`, never a status of its own.
- `nohup`, not `tmux`: the run writes `progress.json` and needs no terminal.
