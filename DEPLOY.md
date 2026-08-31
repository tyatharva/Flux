# Generating the seed library on a rented multi-GPU box

**One image, one command.** The code is baked in and the tag names the commit, so a rented
machine clones nothing, builds nothing, and cannot pull the wrong revision. Full rationale
in `PROJECT_BRIEF.md` (the dated block at the top) and `FASTEDDY_TRAPS.md` §23.

---

## The command

```bash
docker run --gpus all -v /out:/out ghcr.io/<you>/flux-seeds:<tag> run_seeds --gpu-count 16
```

Check the box first — it takes seconds and no GPU-hours, and it is the difference between
finding out now and finding out after the grid has been built:

```bash
docker run --gpus all -v /out:/out ghcr.io/<you>/flux-seeds:<tag> verify
```

**Mount the output directory and nothing else.** A seed is a flat, uniform, doubly-periodic
spin-up: `topoFile` is empty and `inFile` is empty, so it reads no sounding, no terrain and
no land cover. The HRRR and surface trees are only needed for corpus CASES, and then only
at `-v /data/hrrr:/flux/data/hrrr:ro`.

## Build and publish it

```bash
docker/build_image.sh                       # -> flux-seeds:<flux-sha>-fe<fasteddy-sha>
docker tag flux-seeds:<tag> ghcr.io/<you>/flux-seeds:<tag>
docker push ghcr.io/<you>/flux-seeds:<tag>
```

The build refuses a dirty working tree in either repository, because an image tagged with a
commit whose tree it does not contain is worse than an untagged one. `FLUX_ALLOW_DIRTY=1`
overrides and produces a `-dirty` tag.

## What lands in `/out`

| | |
|---|---|
| `seeds/<job>/seed_restart.nc` | **the seed**, 73.3 MB. Copy these back into `jobs30/<job>/return/`; `bin/pick_seed.py` reads the manifests from there |
| `seeds/<job>/` | also `stationarity.json`, `manifest.json` (with an `achieved` block), `acceptance.txt`, `seed_report.json`, `turb_alive.json`, `rotation_check.json`, the logs |
| `machine_manifest.json` | every seed, its verdict, its measured GPU-h, its stop time, peak VRAM per GPU |
| `threadblock_sweep.json` | the block shape measured **on this machine**, with the runners-up and the repeat noise |
| `direction_drift_library.txt` | the Ekman-backing table over the whole library, computed once at the end |
| `work/<job>/` | the working directory, dumps included, left for inspection (~1.8 GB per seed; `--prune-dumps` deletes them on success) |
| `run_seeds.log` | everything the run printed |

## Options worth knowing

| | |
|---|---|
| `--gpu-count N` / `--gpus 0,3,7` | 0 (default) uses every visible GPU |
| `--pass N/M` | **for splitting the library across SEVERAL machines.** 30 seeds over 16 GPUs on ONE machine needs no pass — it is a work queue, and a queue starts the 17th seed on whichever card finishes first instead of idling 15 cards through the tail of a pass |
| `--only <job,...>` | a smoke test on named seeds |
| `--ceiling-h 2.0` | the simulated-hour hard ceiling per seed. The watcher usually stops sooner |
| `--no-sweep` | keep the `.in`'s Ada-measured `1x2x64` instead of re-measuring |
| `--prune-dumps` | delete each seed's `output/` on success |
| `--dry-run` | list what would run |
| `--force` | re-run seeds whose work directory already holds a finished restart |

**It is resumable.** Re-running the same command skips seeds whose work directory already
holds a finished `seed_restart.nc` and restarts partial ones from step 0. A rented box that
lost power costs the seeds that were mid-flight, not the ones that finished.

**A failed seed never aborts the machine.** Each runs in its own subprocess with its own
working directory; a crash, a gate failure and a timeout are recorded with a reason and the
queue moves on.

## What the summary tells you, and how to read it

Seeds accepted, seeds failed with reason, measured GPU-h per seed, measured GPU-h per
*simulated* hour against the 0.479 measured on Ada, stop times against the ceiling, and peak
VRAM per GPU both as compute-app attribution and as whole-device.

**`accepted` does not mean every gated limit resolved in band, and it must not.** PLAN.md
records that `TKE_BL/u*^2` and `z_i` cannot be resolved against their thresholds at ANY
scoring-window width in an affordable spin-up — they decorrelate on the eddy turnover, not
on the dump interval — so **INDETERMINATE is the library's normal state** and
`bin/pick_seed.py` admits it under a flag. What it refuses outright is **DRIFTING**, and
that is the verdict the summary calls out separately.

## Two operational notes

**Everything in `/out` is written as root.** The image declares no `USER`, because OpenMPI
refuses to launch as root without `OMPI_ALLOW_RUN_AS_ROOT` and setting that is simpler and
more portable than matching a uid that differs on every rented box. On the box itself this
is invisible. If you mount `/out` somewhere you later want to clean up as a normal user,
either `chown -R` afterwards or add `--user "$(id -u):$(id -g)"` to the `docker run` — the
entrypoint works either way.

**Budget the disk.** Each seed keeps its dumps: 24 x 73.3 MB = **1.8 GB**, so the full
library is ~53 GB in `/out/work` plus ~2.2 GB of deliverables in `/out/seeds`.
`--prune-dumps` deletes a seed's `output/` once it succeeds and leaves it on failure, which
takes the total to ~2.5 GB.

## Things that are true and easy to be surprised by

**Bitwise reproducibility does not hold across architectures, and is not sought.** FastEddy
is not bitwise reproducible run-to-run on ONE GPU with ONE binary — ~1e-4 relative in
velocity after 200 steps, from the block-retirement order of an `atomicAdd` in the slab-mean
reduction. Seeds are turbulence realisations. Do not diff two of them.

**The image carries real SASS for `sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120` and
no PTX at all** (the analysis library `lib/liblpdm.so` does carry PTX — it is one
translation unit, compiled whole-program, and it is there so the contrast can be checked on
the image itself). That is not an omission: FastEddy is built with separate compilation, and
`nvcc -dlink` silently drops every PTX image from the fatbin. There is therefore no JIT
fallback by construction — which is why every supported architecture carries compiled code
instead, and why anything older than Turing will not run. `verify` says so at startup rather
than at the first kernel launch.

**An out-of-range `CUDA_VISIBLE_DEVICES` fails loudly, at exit 100.** `gpuErrchk` at
`fecuda_Device.cu:55` prints `GPUassert: no CUDA-capable device is detected` before the
no-device branch is ever reached. The orchestrator and the per-seed preflight assert the
device exists first anyway, because failing at the preflight costs nothing and failing
after an `mpirun` costs a confusing log on a box running fifteen other seeds.

**The thread-block sweep runs before the seeds and takes about a minute.** It is a pure
performance knob and cannot move the physics: the one reduction that accumulates is
templated on compile-time constants that do not follow `tBx/tBy/tBz`. It reports the shapes
it did NOT measure by name, so the winner is "best of what was tried" rather than an
implied "best that exists".
