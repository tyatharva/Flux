# Seed jobs — how to run one on a rented GPU

Each directory here is one **self-contained spin-up job**: one `.in`, one manifest, one
entrypoint, no absolute paths, no shared state. 18 of them make the seed library.

A seed is **pre-spun flat, uniform, doubly-periodic turbulence**. It is not a corpus point
and is never trained on. Its only job is to delete the 3 simulated hours of spin-up a
cold-started corpus case would need — 52 GPU-h that buys back about 5500 across ~1825
cases. See `../LIBRARY_PLAN.md`.

## What the machine needs

| | |
|---|---|
| GPU | **sm_89** (compute capability 8.9), ~1.6 GB VRAM. The entrypoint checks and warns: a **newer** architecture JITs from PTX and works but is slower; an **older** one will not run at all. |
| runtime | Docker with `nvidia-container-toolkit`, and `--gpus all` available |
| image | `flux-fasteddy:cuda118`, built by `docker/build_fasteddy.sh` |
| repo | a checkout of this repository, **anywhere** — the entrypoint discovers its own root |
| build | **FastEddy must be built inside the checkout** at `FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy`. It is gitignored, so a fresh clone does not have it. |
| disk | ~2.7 GB per job while running (36 dumps x 73 MB); ~74 MB comes home |
| time | 4 segments of ~46 min wall = **~3.1 h per job** |

## Run one

```bash
jobs/run_seed.sh jobs/seed_cbl-mid_a030 --dry-run   # preflight only, no GPU time
jobs/run_seed.sh jobs/seed_cbl-mid_a030
```

It is **resumable**: the chain restarts from the newest dump on disk, so a kill costs at
most one segment. It is **serialised**: `docker/run_case.sh` refuses to start a second
FastEddy container, because two of them writing the same `output/` interleave their dumps
and corrupt both while looking merely stalled.

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
