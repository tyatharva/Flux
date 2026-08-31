# Seed jobs — how to run one on a rented GPU

> **THE WHOLE LIBRARY NOW HAS A ONE-COMMAND PATH, AND IT IS NOT THIS ONE — 2026-08-31.**
> `docker run --gpus all -v /out:/out <image> run_seeds --gpu-count 16` builds all 30 seeds
> across every visible GPU from a self-contained image with the code baked in: no clone, no
> build, no chance of the wrong commit. See `bin/run_seeds.py` and `Dockerfile.blackwell`.
> This file describes running ONE seed by hand from a checkout, which is still how the
> development workstation does it and is what the orchestrator drives underneath.
>
> **THIS DIRECTORY IS THE RETIRED 16 m LIBRARY. `jobs30/` IS PRODUCTION** — 5 rungs x 6 base
> angles = 30 seeds at 122^3 @ 30 m. The 18 here are a 16 m grid at a 10 m receptor.

Each directory here is one **self-contained spin-up job**: one `.in`, one manifest, one
entrypoint, no absolute paths, no shared state.

A seed is **pre-spun flat, uniform, doubly-periodic turbulence**. It is not a corpus point
and is never trained on. Its only job is to delete the 3 simulated hours of spin-up a
cold-started corpus case would need — 52 GPU-h that buys back about 5500 across ~1825
cases. See `../LIBRARY_PLAN.md`.

## What the machine needs

| | |
|---|---|
| GPU | **an architecture the binary actually carries SASS for**, ~0.65 GB VRAM (MEASURED at 122^3; the manifests' 1.6 GB is an unverified literal and nothing reads it). The old wording here said a newer architecture "JITs from PTX and works but is slower" — **that was false for every binary this project has ever built**, and would have been a `no kernel image` failure on a 5090. `-arch=sm_89` embeds no PTX, and `nvcc -dlink` drops PTX from a separately-compiled binary anyway (`FASTEDDY_TRAPS.md` §23). The entrypoint now compares the GPU's capability against `cuobjdump --list-elf` on the binary itself and refuses rather than warns. |
| runtime | Docker with `nvidia-container-toolkit`, and `--gpus all` available |
| image | `flux-seeds:<commit>` (`docker/build_image.sh`, CUDA 13.0, real SASS sm_75-sm_120, code baked in) — or `flux-fasteddy:cuda118` (`docker/build_fasteddy.sh`) for the toolchain-only workstation path |
| repo | a checkout of this repository, **anywhere** — the entrypoint discovers its own root |
| build | **FastEddy must be built inside the checkout** at `FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy`. It is gitignored, so a fresh clone does not have it. |
| disk | ~1.8 GB per job while running (24 dumps x 73 MB at the 2.0 sim-h ceiling); **~70 MB** comes home (measured) |
| time | **ONE invocation, ~58 min wall** at the 2.0 sim-h ceiling and 0.48 GPU-h/sim-h; usually less, because the watcher stops the run when the oscillation-immune limits enter band |
| network | none during the run; the job is entirely local once the repo and image are present |

## Run one

```bash
jobs/run_seed.sh jobs/seed_cbl-mid_a030 --dry-run   # preflight only, no GPU time
jobs/run_seed.sh jobs/seed_cbl-mid_a030
```

**IT IS NOT RESUMABLE, AND THIS FILE SAID THE OPPOSITE UNTIL 2026-08-31.** It claimed "the
chain restarts from the newest dump on disk, so a kill costs at most one segment" — the
chain was retired 2026-08-26 and a seed is now ONE continuous invocation. `run_seed.sh`
REFUSES a partial run rather than resuming or wiping it (`--restart-over` discards it
deliberately); re-invoking a COMPLETE job is a no-op, which is idempotence and not a
restart. **A killed job costs the whole run.** The correction matters because the old
sentence made a kill look cheap.

It is **serialised**, and the scope depends on where it runs: on the workstation
`docker/run_case.sh` refuses to start a second FastEddy container anywhere on the machine;
inside the portable image (`FLUX_NATIVE=1`) it refuses a second FastEddy **on the same
GPU**, keyed on `CUDA_VISIBLE_DEVICES`. Machine-wide would serialise all sixteen cards onto
one seed at a time. What the rule protects against — two runs writing one `output/` and
interleaving their dumps, which looks like a stall rather than an error — is per-device.

## What comes back

Everything under `<job>/return/`:

| file | |
|---|---|
| `seed_restart.nc` | the final dump, 73.3 MB, all 22 variables. **This is the seed.** |
| `stationarity.json` | the gate verdict and the achieved state |
| `stationarity.txt` | the same, human-readable |
| `seed.log` | the concatenated FastEddy logs |
| `manifest.json` | the job spec **plus an `achieved` block** |

**The ~36 intermediate 300 s dumps stay on the rented machine.** The gate runs *inside* the
job, so ~2.7 GB of turbulence never has to travel; the verdict comes home as a few kB.

Copy `return/` back into `jobs/<job>/return/` on the machine that will run the corpus.
`bin/pick_seed.py` reads the manifests from there.

## The gate

`bin/seed_stationarity.py`, seven limits scored on the last 1.5 h. **It gates on `U/u*`,
not on `u*`.** A doubly-periodic neutral Ekman layer forced by a constant geostrophic wind
does not settle to a fixed `u*` on any affordable timescale — the inertial period here is
**17.6 h**, and `u*` moved **-27%** over 6.26 simulated hours on `g16_spin` while `U/u*`
was within **0.31%** of its final value by 3.01 h. Gating on `u*` alone failed this
project's spin-ups twice for a reason that was never a modelling error.

A job whose gate FAILs still returns its artifacts and exits 1. `bin/pick_seed.py`
**excludes** such a seed rather than offering it, and does not silently fall back to its
target values either.

## Two things that are true and easy to be surprised by

**Bitwise reproducibility will not hold across different physical GPUs.** FastEddy is
already non-reproducible run-to-run on *one* GPU — ~1e-4 relative in velocity and ~7e-4 K
in theta after 200 steps. Seeds are turbulence realisations, so this costs nothing. Do not
diff two seeds expecting equality.

**An out-of-range parameter does not stop FastEddy.** It prints one line, leaves the
variable at its compiled-in default, and runs a different case than the `.in` describes —
`FASTEDDY_TRAPS.md` §13. The `.in` files here are generated with every value guaranteed in
range, but if you edit one by hand, grep the log for `outside limits` as well as
`CORRUPTED`.
