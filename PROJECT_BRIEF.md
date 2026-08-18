# Flux Footprint Emulator — Kegonsa Solar Array

## Goal

Train a conditional normalizing flow (CNF) that predicts 2-D flux footprints for the
UW-Madison Kegonsa Solar Array eddy-covariance tower, using **only the scalar inputs the
Kljun et al. (2015) model uses**, and beats Kljun at that site.

Training targets come from FastEddy LES + a backward Lagrangian particle dispersion model
(LPDM) written for this project. LES and LPDM are offline target generators — they are
**never** part of inference.

Scope is deliberately narrow: this is a **site-calibrated emulator for one tower**. It has
zero transfer to other sites, and that is an accepted, stated limitation. Do not add scope.

## Hardware / environment

- Single NVIDIA RTX 4080 (16 GB), Ada, `sm_89`
- CUDA **11.8** (do not "upgrade" — other versions have failed to build)
- FastEddy **v5.0.1**
- FastEddy runs in **fp32**. **Confirmed (Stage 0b, 2026-08-17)** — see Conventions.

## Repository layout

```
Flux/                          <- working dir, main project repo root
├── PROJECT_BRIEF.md
├── PLAN.md
├── Dockerfile
├── inst.txt                   <- crude dependency/build notes, written for v4.0.1
├── <DaneCounty LiDAR DEM>.zip <- NOT in git. See Data below.
└── FastEddy-model-5.0.1/      <- separate repo. Gitignored by the main repo.
```

## Build and run: Docker only

Local dependency installation is not used. FastEddy runs containerized.

- `inst.txt` documents the v4.0.1 dependency and compile procedure. v5.0.1 is likely the
  same but **verify against the FastEddy docs rather than assuming**.
- Translate `inst.txt` into a `Dockerfile`. Expected shape:
  - Base `nvidia/cuda:11.8.0-devel-ubuntu22.04`
  - MPI (FastEddy is MPI-parallel), NetCDF-C, HDF5, build toolchain
  - Compile with `sm_89`
- Host requires `nvidia-container-toolkit`. Run with `--gpus all`.
- Bind-mount `Flux/` into the container so output lands on the host, not in the image.
- Keep the image build and the model build as separate layers so source edits don't
  trigger a full dependency reinstall.

**Gate:** the container must reproduce a known-good tutorial case before it's trusted.

## Data

- Dane County, WI LiDAR DEM, currently a zip in the project root.
- **Never commit the DEM.** Gitignore it. Record provenance (source, download URL, date,
  CRS, resolution) in `data/README.md` so the pipeline is reproducible without the binary.
- Same rule for CONUS404 extracts and any FastEddy output.

## Git structure — two repositories

**1. FastEddy modifications** → a **fork of `NCAR/FastEddy-model`**, working branch `kegonsa`.

Fork it; do not create a fresh repo from the source tree. A fork keeps the diff against
upstream visible and lets us pull NCAR bugfixes. A fresh repo throws both away permanently.
Preserve NCAR's license and attribution.

**2. Main project repo** → everything else at `Flux/` root: Dockerfile, LPDM, preprocessing,
run configs, analysis, ML code, PROJECT_BRIEF.md, PLAN.md.

- `.gitignore`: `FastEddy-model-5.0.1/`, `*.zip`, DEM/CONUS404 data, LES output
- Record the FastEddy fork URL and **pinned commit SHA** in `FASTEDDY_VERSION.txt`, updated
  whenever the fork changes. This is the reproducibility link between the two repos.
- Submodule instead of gitignore is acceptable if preferred, but the pinned-SHA file is the
  lower-friction option for a solo project and is what we're defaulting to.

**Rule:** any edit inside `FastEddy-model-5.0.1/` is a commit to the fork, on `kegonsa`,
with a message explaining the physical or numerical reason — not just what changed.
Never leave uncommitted modifications in that tree; an untracked LES source edit silently
invalidates every result produced after it.

## Site

- UW-Madison Kegonsa Solar Array, southern Wisconsin
- Solar array footprint ~100 m x 400 m
- ~30 m of elevation change across the area
- EC tower measurement height: **30 m AGL**

## Domain configuration (settled)

Wind-aligned elongated box, rotated per case so mean wind is along +x.

| | 10 m option | 8 m option |
|---|---|---|
| Grid `Nx x Ny x Nz` | 434 x 146 x 122 | 542 x 182 x 154 |
| Padded (`N + 2*Nh`, Nh=3) | 440 x 152 x 128 | 548 x 188 x 160 |
| Physical extent | 4.34 x 1.46 x 1.22 km | 4.34 x 1.46 x 1.23 km |
| Cells | ~7.7M | ~15.2M |

- Thread block: **4 x 4 x 16**. Larger blocks (512/1024 threads) fail with
  `too many resources requested for launch` — the hydro-core kernels are register-heavy.
- **Hard rule:** `(N + 6) % tB == 0` for each dimension. Any grid change must re-satisfy this.
- Tower sits at ~3/4 along x (cell ~324 of 434) and centered in y. Gives ~3.24 km upwind
  fetch, ~1.08 km downwind, and guarantees a >1 km square of surface is always in-domain
  under any rotation angle.
- Vertical needs **stretching** — 1.22 km flat is too shallow for a daytime CBL. Fine (10 m)
  near surface, growing above ~500 m, top ~2.5 km, with a Rayleigh damping layer.

## Boundary and initial conditions (settled)

- **Fully doubly periodic** in x and y. No Dirichlet inflow, no cell perturbation.
- Soundings from **CONUS404** (4 km, WY1980-2024) set: initial theta/u/v profiles,
  geostrophic forcing tuned to observed mean wind, and surface heat flux.
- Prefer **tower-observed surface flux** over CONUS404 where available.
- Terrain and land cover must be **tapered to a constant over the outer few hundred meters
  in both x and y**, or the periodic wrap creates a surface discontinuity at the seam.
- Each run is one quasi-stationary state, matching one 30-min EC averaging period.

## Rotation

Done entirely in preprocessing. Resample DEM and land cover into a wind-aligned frame and
supply the correspondingly rotated `lat(y,x)` / `lon(y,x)` to `GeoSpec.py`. FastEddy needs
no code change — its lat/lon are 2-D arrays independent of the x/y grid, and v5.0 uses them
for Coriolis and virtual tower placement.

## Solar panels

Represented as a **bulk surface patch**: reduced albedo, elevated z0 (~0.1-0.3 m vs
~0.01-0.05 for grass), displacement height d ~1-1.5 m. Do NOT use explicit geometry
(URBAN/IBFM or GAD) — panel row spacing is ~5-7 m and the grid cannot resolve it, so
explicit geometry would look resolved while being meaningless.

Known omissions, accepted: elevated heat source, directional roughness anisotropy from row
alignment, diurnal tilt, ~20% shortwave leaving as electricity.

## Footprint computation

Backward LPDM, run offline on saved FastEddy output.

- Save 3 velocity components + SGS TKE, ~5 s cadence (2 s if validation demands)
- **Measured (Stage 0a): FastEddy writes 19 3-D fields = 76 B/cell, and exposes NO way to
  select output fields or a vertical subset.** Only 5 IO parameters exist (`ioOutputMode`,
  `inFile`, `outFileBase`, `frqOutput`, `towerIOSelector`); binary mode walks the same
  variable list as NetCDF, so it is the same data in a different container.
- Real numbers for the production grid (7.73 M cells): **0.59 GB per dump**, so a 30-min
  window at 5 s cadence is **213 GB** — not the 11-33 GB originally budgeted, which assumed
  a `z < 400 m` subset that does not exist.
- The LPDM needs only u, v, w, SGS TKE = 16 B/cell. Getting there is a Stage 3 problem:
  field selection + fp16 on write targets <30 GB per window. Note 12 of the 76 B/cell are
  `xPos`/`yPos`/`zPos`, rewritten identically every dump.
- Pipeline per run: LES -> scratch -> LPDM -> 2-D footprint (~1 MB) -> **delete fields**
- Peak storage is one run. Never accumulate.
- SGS component is a Langevin model driven by FastEddy's output SGS TKE (Weil et al. 2004)
- **Well-mixed condition is the critical correctness test.** Release a uniform particle
  distribution in a flat neutral case; it must stay uniform. Failure means artificial
  near-surface accumulation, which corrupts exactly the near-field signal we care about.

## ML model

- **Conditional normalizing flow.** Inputs are Kljun's scalars only.
- **Residual formulation**: predict Kljun + learned correction, not the raw footprint.
- Wind direction is the dominant skill axis — it indexes site geometry (array at 270 deg,
  hill at 90 deg). Stratify the corpus heavily on direction, weighted by the site wind rose.
- Pretrain on cheaply-generated analytical (Kljun/Kormann-Meixner) footprints, then
  fine-tune **all weights** on the LES corpus.
- Loss: not raw MSE. Use log-space or per-sample normalization, plus physical terms
  (centroid displacement, 80% source-area overlap, integral = 1). Consider Sinkhorn/W2.
- **Split by LES run, never by sample.** Samples from one run share a turbulence
  realization; shuffling leaks and produces a meaningless validation curve.
- Effective sample size for generalization is the number of *runs*, not samples.

## Conventions

- **Particle state in fp64** even though velocity fields are fp32. Trajectory integration
  accumulates roundoff over thousands of steps.
- **Precision — Stage 0b PASSED, 2026-08-17.** FastEddy is hardwired fp32 with no build
  switch and no typedef to change: bare `float` on every prognostic field, `MPI_FLOAT` in
  the halo exchange (`SRC/FECUDA/fecuda_Utils.cu`), `NC_FLOAT` in the writer
  (`SRC/IO/io_netcdf.c:697,927`). Confirmed in output — all 29 variables of a tutorial dump
  are `float`. Nothing to change; this is recorded so it is never re-litigated.
- **FastEddy is NOT bitwise reproducible.** Identical binary, identical container, two runs
  of the same case differ by ~1e-4 relative in velocity and ~7e-4 K in theta after only 200
  steps. Consequences: never diff two runs expecting equality; a corpus run can be
  reproduced statistically but not exactly; and any "did my change matter?" test must
  compare against this nondeterminism floor, not against zero.
- Any grid change re-checks the `(N + 6) % tB == 0` rule before running.
- Commit at every verification gate in PLAN.md. Do not proceed past a failed gate.

## Ruled out — do not propose these

These were evaluated and rejected. Re-proposing them wastes time.

- **STILT** — replaced by the project's own backward LPDM.
- **Mesoscale coupling** (`hydroBCs=1`, GenICBCs, cell perturbation) — the CP fetch
  requirement would consume most of the domain. Periodic instead.
- **LES-to-LES nesting** — schedule does not permit it.
- **NSCBC** — unnecessary; FastEddy's Dirichlet+CP path already exists, and we're not
  using lateral forcing at all.
- **Running FastEddy backwards in time** — mathematically impossible, not a code
  limitation. Reversing t and u flips the sign of the SGS stress term, giving negative eddy
  viscosity and the backward heat equation. At 10 m spacing noise e-folds every ~20 s.
  Backward LPDM steps *particles* backward through *forward-stored* fields.
- **Multiple virtual tower locations** — would inject unexplained variance, since surface
  fields are not model inputs. One fixed tower.
- **Surface fields as ML inputs** ("Experiment 2") — out of scope for this project.
- **512^3 domains** — computationally infeasible for the required corpus size.
- FNO / U-Net may be *benchmarked* against the CNF, but CNF is the primary architecture.

## Status

**Stage 0a PASSED (2026-08-17)** — see `STAGE0A_RESULTS.md` for full evidence.
Container builds; Example03_SBL and Example01_NBL both run to completion on `sm_89` with
no `too many resources requested for launch`. The NBL failure that previously blocked
everything is resolved. Initial states reproduce the published GABLS1 and NBL specs.

**Stage 0b PASSED** — precision documented in Conventions above.

Measured, and load-bearing for everything downstream:

- **Compute cost is 9.37 ns/cell/step**, with two cases 12x apart in size agreeing to 1.7%.
  The production grid (434 x 146 x 122 = 7.73 M cells) therefore costs **0.0725 s/step**.
- **Memory is not a constraint.** NBL at 23.5 M cells peaked at 6.4 GiB of 16 GiB; the
  production grid is 3x smaller.
- **`dt` is acoustically limited, not inherited.** FastEddy is fully compressible with RK3
  (Wicker-Skamarock 2002) and no acoustic sub-stepping, and contains **no CFL machinery at
  all** — `dt` is a mandatory user constant bounded only by `FLT_MIN..FLT_MAX`, never
  computed or checked. Sound speed sets the limit, so `dt` scales with grid spacing and
  cannot be raised by fiat.
- **CUDA-aware MPI is NOT required** (tested, not assumed). **MPI Fortran bindings ARE**,
  despite FastEddy having no Fortran source — its C code broadcasts with `MPI_INTEGER` /
  `MPI_CHARACTER` in 60 places. Stock Ubuntu `libopenmpi-dev` satisfies both.
- **Thread blocks:** every shipped tutorial uses 256 threads/block (`1x4x64` or `1x8x32`).
  `4x4x16` is also 256 and equally valid. The divisibility rule is enforced on **per-rank,
  halo-inclusive** extents (`SRC/GRID/grid.c:222-240`); Nz is never decomposed.

See @PLAN.md for the staged path.
