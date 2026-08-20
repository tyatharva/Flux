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
- **Tower coordinate, SURVEYED: `42.957160, -89.292362`** (EPSG:3071 577719.1, 276299.5).
  This is the single source of truth; it lives in `TOWER_LON/TOWER_LAT` in
  `bin/prep_stage6.py` and `docker/prep_dem30.py` imports it from there. Any result produced
  before 2026-08-19 at a surrogate coordinate is void.
- **Solar array — THE TOWER IS INSIDE IT.** Corrected 2026-08-19. It extends **60 m east
  and west, 250 m north, 100 m south** of the tower: 120 x 350 m, 4.20 ha (4.32 ha once
  discretised at 24 m). It is a rectangle in EPSG:3071 and nothing about it depends on the
  wind. Earlier specifications as a patch at bearing 270 deg / 200 m, or worse as an
  "upwind distance", are void — the latter silently rotated the array with the wind and
  made it upwind in every case by construction.

  Because the tower is inside it, the array's UPWIND REACH is a strong function of
  direction — 250 m for a northerly, 100 m for a southerly, 60 m for an easterly or
  westerly — and the fraction of the crosswind-integrated footprint inside that reach swings
  by ~300x:

  | stability | E/W (60 m) | S (100 m) | **N (250 m)** |
  |---|---|---|---|
  | very unstable | 0.1% | 4.2% | **34.1%** |
  | neutral | 0.0% | 1.2% | **24.2%** |
  | stable | 0.0% | 0.1% | **13.6%** |

  Site consequence, not just a test: **this tower measures the array on northerlies and
  measures the neighbours on easterlies and westerlies.**
- ~30 m of elevation change across the area
- EC tower measurement height: **30 m AGL**
- **Land cover comes from ESA WorldCover v200 (2021), 10 m.** Replaces the LiDAR-flatness
  water mask 2026-08-19, and with it the Dane County bare-earth DEM (withdrawn) and
  `docker/prep_dem30.*` (deleted). Terrain now comes from a **USGS 3DEP 1/3-arcsecond**
  raster. Both live in `data/raw/` and are gitignored; `bin/prep_surface.py` builds the
  model grid from them.

  On the 186 x 186 @ 24 m box: 37.4% cropland, 28.5% tree cover, **16.1% permanent water**,
  15.7% grassland, 2.2% built. Roughness is assigned per class (water 1e-4, grass 0.03,
  cropland 0.10, built 0.5, tree 1.0), then the array rectangle overrides it — WorldCover
  labels the array as cropland, because it does not see photovoltaics.

  **The water is strongly directional, which is the point.** Within 4 km, by octant:
  N 0%, NE 58%, **E 72%**, SE 2%, S 0.4%, SW 0%, W 0%, NW 0%. Nothing inside 1 km.

  *The old LiDAR mask was not actually wrong* — it agrees with WorldCover to ~1 point per
  annulus. What made the lake look wrong was the rotated 4380 x 1500 m strip slicing a
  ribbon through it. The static domain fixes that; WorldCover is adopted because it is
  authoritative and carries every roughness class, not because the old mask failed.
- **Terrain is tapered at the wrap seams; land cover is NOT.** Terrain height enters the
  coordinate transform and its metric tensor, so a seam step is a numerical cliff. Roughness
  and surface heat flux are local boundary conditions, where a seam is just a coastline —
  and tapering them would erase the water from exactly the easterly cases meant to sample it.

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

**Albedo has no pathway, and that is not an omission.** FastEddy in this configuration has
no radiation scheme at all — `surflayerSelector = 1` prescribes the kinematic surface heat
flux directly — so what albedo would have controlled is subsumed by `htFlux`, which IS
per-cell (`cuda_surfaceLayerDevice.cu:191` reuses the array when `surflayer_idealsine = 0`).
`htFlux`, `z0m`, `z0t` and `tskin` are all IO-registered, so they survive the restart read
and `bin/prep_stage6.py` writes them there. The built-in `surflayer_offshore` wave-roughness
parameterisations are a **global** switch and cannot be applied to water cells only, so
per-cell `z0` is used for the water instead.

**Measured 2026-08-19 (Stage 6, 30 m grid):** the array takes **10.3x its area share** of
the footprint when upwind (7.32% of the footprint from 0.71% of the domain) and **0.06x**
when downwind. Same patch, same tower, same spun-up state, rotated 180 deg.

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

**Stages 2-6, THIRD pass, 2026-08-20** — see `THIRD_PASS_RESULTS.md`. Static
`186 x 186 x 122` domain at **24 m** (4464 m box), geography built once from USGS 3DEP +
ESA WorldCover, direction set by rotating the geostrophic wind. Supersedes the second pass.

| stage | gate | result |
|---|---|---|
| 2 | TKE stationarity | ✅ **PASS** — TKE -0.22 sigma, `u*` +1.40 sigma at t = 5 h |
| 3 | 30-min window under 30 GB | ✅ **PASS** — 15 GB via `ioLPDMmode` on the fork |
| 5 | sub-grid fraction < 40% | ❌ ~80%, but the error it proxies for is now diagnosed |
| 5 | Kljun (secondary) | peak +29% (was +86%), 80% area exact, integral **0.984** |
| 6 | explicable difference | ✅ **PASS** — array 15.9x its area share on a northerly, 0.00x on a westerly |

**The near-field error is a `sigma_w` deficit, not diffuse resolution loss.** At the receptor
the LES gives `sigma_w/u* = 1.09` against the surface-layer 1.25; low `sigma_w` makes backward
particles descend too slowly and travel too far. The physically motivated fix -- an
anisotropic sub-grid split -- made it FOUR TIMES worse (peak 390 -> 1170 m), which is what
confirmed the diagnosis. Supplying the missing variance instead moves the peak to 270 m.
Adopted as `--sgs-most`, a height-dependent MOST-anchored floor, never as a tuned scalar.

**Stage 2's earlier failure was a sampling artifact.** `u*` overshoots to 0.41 near t = 1 h,
decays through -7 %/h at 3.1 h, and settles by 5 h. The second pass sampled at 6.4 h on a
coarser grid, caught the flow mid-decay, and read a transient as an unreachable trend.

---

**Stages 2-6, second pass, 2026-08-19** — see `STAGE2-6_RESULTS_V2.md`. At the 30 m
pipeline-development grid with `dz_sfc = 8.56 m`:

| stage | gate | result |
|---|---|---|
| 2 | bitwise restart | ✅ PASS — byte-identical dumps at the new grid |
| 2 | TKE stationarity | ❌ NO — `u*` still drifting -2.25 +/- 0.27 %/h (-8.4 sigma) at 6.4 h |
| 3 | 30-min window under 30 GB | ✅ PASS by configuration — 13.0 GB, IO ~3% of compute |
| 4 | well-mixed | ✅ PASS — backward rms 4.91% vs a 4.48% counting-noise floor |
| 4 | plausible transit time | ✅ PASS — median 287 s |
| 5 Gate 1 (revised) | sub-grid fraction of `sigma_w^2` < 40% | ❌ **FAIL twice**, 96.4% -> 88.3%; **unreachable at `dx = 30 m`** |
| 5 Gate 2 | error floor | ✅ MEASURED — floor separated from signal (59.2% vs 36.9%) |
| 6 | explicable difference | ✅ PASS — array 10.3x its area share upwind, 0.06x downwind |

**The one blocking result is Stage 5 Gate 1.** The sub-grid fraction collapses onto
`z/Delta`; the 40% gate needs `Delta <~ 8.6 m`, which is a statement about `dx`, not `dz`.
The grids that pass cost **20-23 GPU-hours for the spin-up alone**. That is a project-level
decision about corpus cost, not a configuration change, and nothing at `dx = 30 m` reaches it.

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
- **Thread blocks — `tBx` MUST BE 1, and `4x4x16` was costing 17%.** Corrected by
  measurement 2026-08-19. `i <- threadIdx.x` (`cuda_hydroCoreDevice.cu:648`) while
  `kStride = 1` and `iStride = (Ny+6)(Nz+6)` (`grid.c:621-623`). CUDA linearises a warp
  with `threadIdx.x` fastest, so any `tBx > 1` makes adjacent threads in a warp read
  addresses `iStride` floats apart — one 128-byte transaction becomes four 32-byte ones.
  Every shipped tutorial uses `tBx = 1` (`1x4x64`, `1x8x32`) for exactly this reason.

  Swept nine legal shapes over 200 steps at `186x186x122`:

  | block | threads | s/step |
  |---|---|---|
  | **1x2x64** | 128 | **0.0359** |
  | 1x6x32 | 192 | 0.0364 |
  | 1x3x64 | 192 | 0.0367 |
  | 1x8x32 | 256 | 0.0380 |
  | 2x4x32 / 2x2x64 / 1x1x64 / 1x4x64 | 256/256/64/256 | 0.0383-0.0386 |
  | `4x4x16` (the old default) | 256 | **0.0421** |

  **`tBz = 128` is rejected by the device** — CUDA caps `blockDim.z` at 64, and FastEddy
  reports it as `tBz = 128 > max allowed on device = 64`. The divisibility rule is enforced
  on **per-rank, halo-inclusive** extents (`SRC/GRID/grid.c:222-240`); Nz is never decomposed.

- **Cost is 8.51 ns/cell/step with `1x2x64`**, not the 9.37 measured with `4x4x16`. Use the
  new figure for projections; the old one is 10% pessimistic.

See @PLAN.md for the staged path.

## Settled by measurement — do not re-derive

**Stage 1 PASSED 2026-08-18** (`STAGE1_RESULTS.md`). These are measured facts, not estimates.

- **`dt = 0.0250 s` at 434 x 146 x 122 @ 10 m.** (Supersedes an earlier 0.0275 s, which was
  wrong — see below.) The limit is the **3-D acoustic CFL**,
  `CFL = dt * c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)`, with `c = 347.2 m/s` at 300 K.
  FastEddy has **no CFL machinery at all**, so both thresholds below were bisected
  empirically on the real grid. Tutorial `dt` values are hand-picked and mutually
  inconsistent — never copy them.

- **STABILITY AND ACCURACY ARE DIFFERENT LIMITS, and the accuracy one is lower.**

  | | CFL_3d | dt @ 10 m | behaviour |
  |---|---|---|---|
  | stability limit | ~1.79 | 0.0300 | above this: NaN, `CORRUPTED` |
  | **accuracy limit** | **~1.64** | **0.0273** | above this: silent grid-scale noise |
  | production | 1.503 | **0.0250** | 8% margin below the accuracy limit |

  Between the two the model runs to completion, exits 0, prints no warning, and
  produces **resolved `w` at the lowest ~3 levels that is grid-scale acoustic noise
  rather than turbulence** — `sigma_w^2` at the first level inflated ~9x, i.e. 33,000x
  in absolute terms. The transition is sharp: `dt = 0.0272` is clean, `0.0275` is not.
  Everything else (u, v, theta, u*, and the profile above ~45 m) looks perfectly fine,
  which is exactly what makes it dangerous.

  **The accuracy boundary is a property of CFL_3d, not of the spacing** (confirmed
  2026-08-18 at 30 m: `dx` 3x coarser and `dz_sfc` 2x coarser left the threshold between
  CFL_3d 1.60 and 1.70 — clean at 1.599, `k0/k1 = 7.44` at 1.701). The *stability*
  boundary does move with the grid: at 30 m, CFL_3d = 1.80 produces garbage but still
  completes without NaN, so the silent window is **wider** at coarser resolution.

  **TERRAIN AMPLIFIES THE EFFECTIVE CFL, and the amplification scales with GRID
  ANISOTROPY.** In a terrain-following coordinate the horizontal derivative picks up
  `J31 d/dzeta`, so the effective acoustic Courant number is roughly

      CFL_eff  ~  CFL_3d * sqrt(1 + (slope * dx/dz)^2)

  Measured on the Kegonsa terrain at `dx = 30 m`, `dz_sfc = 8.56 m` (`dx/dz = 3.50`):

  | slope | amplification | CFL_eff at the flat-run `dt` |
  |---|---|---|
  | p50 0.039 | 1.009 | 1.502 |
  | p90 0.099 | 1.058 | 1.575 |
  | p99 0.182 | 1.187 | **1.766** |
  | max 0.259 | 1.350 | **2.009** |

  The flat run at the same `dt` is clean (`k0/k1 = 0.128`); the terrain run trips the
  accuracy limit at the steep cells (`k0/k1 = 3.85`). Confirmed to be grid noise and not
  flow-following motion: subtracting the terrain-following component `u.grad(zg)` leaves
  the ratio unchanged (3.845 -> 3.915) and the two correlate at only +0.16.

  **The earlier `dz_sfc = 20 m` grid did not show this** because `dx/dz` was 1.50 rather
  than 3.50 — the same terrain cost 2.3x less amplification. So refining `dz` alone makes
  a grid MORE sensitive to terrain, not less. Terrain runs here need `dt = 5/199 s`
  against the flat runs' `5/147 s`.

  **Never set `dt` from the stability boundary. Use the accuracy boundary with margin,
  re-derive it whenever the grid changes, and multiply the margin by the terrain
  amplification before any run with topography.** Verify with `docker/diag_near_surface.py`:
  the first-level `w` variance ratio `k0/k1` must be **< 1** (~0.27 when correct, matching
  NCAR's NBL at 0.25). A value near 9 means dt is too large.

- **`hydroSubGridWrite = 0`** drops the 9 SGS stress fields: 19 -> 10 3-D fields,
  76.3 -> 40.3 B/cell, **212 GB -> 112 GB** per 30-min window at 5 s cadence. Free, via
  config. Reaching Stage 3's ~30 GB gate needs the 4 LPDM fields only, which is *not*
  config-reachable.

- **TRAP: FastEddy prints `****CORRUPTED***` on NaN/Inf but still exits 0.** A fully NaN
  field returns exit status 0. **Every run script must grep output for `CORRUPTED`/NaN and
  must never trust the exit code alone.** Use `docker/check_run.sh`.

- **TRAP: a MISSING RESTART FILE does not abort.** FastEddy prints
  `Error: No such file or directory`, then carries on with x,y,z dimensions of 0 and
  produces a run in which **every cell of every field is NaN** — while still exiting 0.
  It cost a 30-minute spin-up segment before `check_run.sh` caught it at the end.
  `docker/run_case.sh` now verifies `inPath`+`inFile` exists before spending GPU time.

- **TRAP: `frqOutput` must divide the ABSOLUTE step number, not merely equal `NtBatch`.**
  The output test is `it % frqOutput == 0` where `it` is the absolute timestep, which on a
  restart starts at the restart step. With `frqOutput = NtBatch = 199` and a restart at step
  418200 (`418200 % 199 = 71`), the test never fires and the run writes exactly ONE dump --
  the unconditional final one after the loop. It exits 0 and reports `RUN OK`. Choose `dt`
  so that the sampling interval is an integer number of steps AND the restart step is a
  multiple of it.

- **TRAP: `frqOutput` finer than `NtBatch` is SILENTLY IGNORED.** FastEddy's time loop is
  `for(it = simTime_it; it < Nt; it += NtBatch)` with the output test `if(it % frqOutput == 0)`
  *inside* it (`SRC/FEMAIN/FastEddy.c:400,423`) — a batch is a GPU-resident launch and the
  host cannot write partway through one. So `NtBatch = 28800, frqOutput = 80` does not give
  5 s output; it gives **two dumps**, at the start and the end, with no warning and a normal
  exit. **For a sampling window, set `NtBatch = frqOutput`.** FastEddy's own parameter help
  says so in passing ("should be an even multiple of NtBatch") and it is easy to read past.
  Cost of the fine batching is small: a 1800 s window in 360 batches of 80 steps runs in
  about the same time as one batch of 28800, plus ~0.05 s of IO per dump.

- **TRAP: `Nt` is an ABSOLUTE target timestep, not a number of steps to run.** A restart
  from step 500 with `Nt = 500` performs **zero** timesteps, writes one dump, and exits 0
  looking like a successful run. To advance 500 steps from step 500, set `Nt = 1000`.
  `simTime_it` resumes from the restart file, and output files are named by absolute step.

- **Restart is a true bit-for-bit state resume** (verified 2026-08-18 at 10 m,
  **re-verified 2026-08-18 at the 30 m grid** — `cmp` reports the two 25.4 MB dumps
  byte-identical; a restarted trajectory then diverges from a continuous one at the
  expected ~1e-4 floor after 500 steps). Restarting from a
  dump and re-dumping reproduces that dump **byte-for-byte**; every prognostic field differs
  by exactly 0. It is not a silent reinitialization. A restarted trajectory then tracks a
  continuous one to within the ~1e-4 nondeterminism floor. This is what makes the
  spinup-once / adjust-per-direction corpus structure valid. Restart requires netCDF —
  `ioOutputMode = 1` binary output is **not** restartable.

- **Vertical stretching is not a speed lever.** With `dx = dy = 10 m` fixed, even an
  infinitely coarse vertical only relaxes the 3-D CFL by `sqrt(3/2)` — **at most ~22% more
  `dt`**. Stretch for domain depth, never for speed.

### 30 m pipeline-development grid (settled 2026-08-18)

The 10 m grid above remains the production target. Everything from Stage 2 onward was
**developed and validated at 30 m**, deliberately: this configuration validates the
pipeline, not the science, and the corpus is regenerated at finer resolution later.

**Superseded 2026-08-19 by the second-pass grid below.** The first-pass grid was
`146 x 50 x 90`, `d_zeta = 30.167598`, `verticalDeformFactor = 0.662868`
(`dz_sfc = 19.997 m`), `dt = 0.0625 s`, 657,000 cells, measured 0.0066 s/step. It is kept
here because Items 1 and 2 of the second pass (reference frame, wrap-around) were measured
on it. The receptor sat at `k=1` with **one** cell below the tower.

**SECOND-PASS GRID (current):**

| | value | note |
|---|---|---|
| `Nx x Ny x Nz` | 146 x 50 x **122** | padded 152/56/**128**, all divisible by 4/4/16 |
| `dx = dy` | 30.0 m | domain 4380 x 1500 m, same physical extent as the 10 m grid |
| `d_zeta` | **22.222222** | `zCeiling = d_zeta*(Nz-0.5) = 2700 m` |
| `verticalDeformFactor` | **0.385204** | **`dz_sfc = 8.5601 m`**, receptor `k=3` at exactly 30.000000000 m |
| **`dt`, flat** | **5/147 = 0.0340136 s** | `CFL_3d = 1.488`, 9% below the 1.64 accuracy limit |
| **`dt`, terrain** | **5/200 = 0.0250 s** | slope-amplified `CFL_eff = 1.48` at the steepest cell |
| `dampingLayerDepth` | 540.0 | 20% of 2700 m |
| cells | 890,600 | |
| levels below 400 m / below 30 m | **40 / 3** | was 20 / 1 |

**Two `dt` values, and they are not interchangeable.** The flat value trips the accuracy
limit over terrain — see the slope-amplification section above. `dt = 0.025` was also
chosen so `5 s / dt = 200` divides the restart step 418200; `5/199` does not, and
`frqOutput` is tested against the ABSOLUTE step (see the trap below).

**`dz_sfc` and a 30 m receptor are not independent.** The near-surface cubic correction in
`zDeform` is under 0.05 m, so cell centres sit at `(k+0.5)*dz_sfc` and a centre at exactly
30 m requires `dz_sfc = 30/(k+0.5)`: 20 m (k=1), **12 m (k=2)**, **8.571 m (k=3)**, 6.67 m
(k=4). A request for "dz_sfc = 10 m with a cell centre at 30.000 m" cannot be honoured; the
bracketing choices are 12 m and 8.571 m.

**The receptor lands on a cell centre at exactly 30.000 m.** `verticalDeformFactor` is
solved for that, not rounded to it: `dz_sfc = 8.5601 m` puts the `k=3` centre at
30.000000000 m once the cubic term's few mm are absorbed by tuning the factor.

**Refining `dz` from 20 m to 8.56 m bought 8 points of sub-grid fraction and cost a `dt`
recalibration plus a terrain-CFL surprise.** It moved the sub-grid fraction of `sigma_w^2`
at the receptor from 96.4% to 88.3% — real, in the right direction, and nowhere near the
40% gate. See the `z/Delta` collapse section: that gate is a statement about `dx`.

The measured 0.0066 s/step at 657 k cells was 7% ABOVE the 9.37 ns/cell/step model — launch
overhead at a size 3x smaller than the smallest case the model was fitted to. Use measured
values, not the model, below ~1 M cells.

### Backward-LPDM traps, settled by measurement (2026-08-18)

- **Periodic wrap-around double-counts the footprint.** A backward trajectory that travels
  more than one domain length re-enters the turbulence it already sampled; its later
  touchdowns are the same eddies counted again. Left uncapped, the flux-footprint integral
  climbs past 1 exactly as wrapping sets in. On the flat 30 m window:

  | t_back (s) | wrapped | integral, uncapped | integral, capped at one domain length |
  |---|---|---|---|
  | 300 | 0.0% | 0.643 | 0.643 |
  | 600 | 0.0% | 0.800 | 0.800 |
  | 900 | 8.2% | 0.791 | 0.896 |
  | 1500 | 31.8% | **1.064** | **0.961** |

  **Always cap trajectory displacement at one streamwise domain length.** Capped, the flat
  integral converges from below rather than sailing past 1.

- **THE CAP MAKES DOMAIN LENGTH A CORRECTNESS CONSTRAINT, not just a fetch one.** The cap
  retires a trajectory once it has travelled one streamwise domain length, so the estimator
  recovers only the influence inside that distance. Re-measured 2026-08-19 on the finer
  grid, where the footprint is broader: the flat 80% source area reaches **3810 m** in a
  4380 m domain, and the capped integral saturates at **0.821**, ~18% short. Ruled out as
  causes: the weight floor (a 4x reduction, `w_floor` 0.02 -> 0.005, moves it only
  0.781 -> 0.815) and `t_back` truncation (flat by 900 s).

  **The streamwise domain must be long enough to contain the footprint.** Uncapped
  double-counts, capped truncates; there is no setting that rescues a domain shorter than
  the footprint. This is partly self-limiting — the footprint is artificially broad at 88%
  sub-grid (Kljun's 80% area ends at 1410 m against the LES's 3810 m) — but it must be
  re-checked at whatever grid the corpus finally uses.

- **The flux weight is frame-dependent, but the frame is a ~2% effect.** A real EC tower is
  double-rotated (Wilczak et al. 2001) so the mean vertical velocity vanishes. Rotating the
  LPDM release velocities into that streamline frame agrees with simply removing the mean to
  within 2% at every `t_back`. The rotation is used because it is the frame the instrument
  reports in and it makes the mean vanish by construction — not because it changes answers.

- **Over a slope the footprint integral is not 1, and that is physical.** With wrap-around
  capped, the terrain case saturates at 1.19-1.35 while the flat case converges to 1. The
  residual is `w_bar` times the concentration integral, both converged. At a receptor
  sitting in 1.5 sigma of mean subsidence the turbulent flux genuinely is not the surface
  flux — the advection non-closure that makes eddy covariance hard in complex terrain.

- **The touchdown weight uses the surface-normal approach rate**, `|d(z-z_ground)/dt|`, not
  `|w|`. Over sloping ground a particle loses height-above-surface because the ground rises
  under it, with `w` near zero, and the `2/|w|` weight explodes — one touchdown reached a
  weight of 208,220 and drove the integral negative. Flat ground hides this completely.

### Sub-grid fraction: what actually sets near-field footprint fidelity

The resolved fraction of `sigma_w^2` collapses onto **`z / Delta`**, with
`Delta = (dx dy dz)^(1/3)`. **Measured on two grids** at `dx = dy = 30 m`:

| `dz_sfc` | `Delta` | `z/Delta` at 30 m | sub-grid fraction | 40% crossing |
|---|---|---|---|---|
| 20.0 m | 26.21 m | 1.14 | **96.4%** | `z/Delta` = 3.74 |
| 8.56 m | 19.78 m | 1.52 | **88.3%** | `z/Delta` = 3.49 |

The two crossings agree, so the collapse variable is right and the requirement is
**`Delta <~ 8.6 m`** for a 30 m receptor. Refining `dz` alone moved the fraction 96.4 -> 88.3%
— real, and nowhere near enough.

**With `dx = dy = 30 m` the 40% target is unreachable at any `dz`**: it would need
`dz <= 0.71 m`, and at that anisotropy the horizontal filter (`2 dx = 60 m`) still cannot
resolve the 30 m eddies, so the `Delta` collapse itself stops describing the physics.

What does reach it, and what it costs on this GPU (9.37 ns/cell/step):

| `dx = dy` | `dz_sfc` | `Delta` | sub-grid | GPU h per simulated h | 3.5 h spin-up |
|---|---|---|---|---|---|
| 15 m | 8.56 m | 12.44 m | 65% | 1.4 | 5 h |
| 10 m | 8.56 m | 9.49 m | 47% | 3.9 | 14 h |
| **10 m** | **6.0 m** | **8.43 m** | **39%** | 6.6 | **23 h** |
| **8.6 m** | **8.6 m** | **8.60 m** | **40%** | 5.8 | **20 h** |

So the gate is a **20-23 GPU-hour spin-up**, i.e. 30-35 chained 40-minute segments. That is
a project-level decision, not something a configuration tweak reaches.

### Stage 2 vertical grid (settled)

`zDeform` (`SRC/GRID/grid.c:1117`) is a cubic map with **`dz_surface = verticalDeformFactor
* d_zeta`** and `zCeiling = d_zeta*(Nz - 0.5)`. Verified: NBL's `0.75 * 20` predicts its
measured 15.005 m.

Settled values — `Nz = 122`, `d_zeta = 20.576`, `verticalDeformFactor = 0.4860`,
`verticalDeformQuadCoeff = 0.0`, giving 10.0 m at the surface, 14.4 m at 500 m, 41.5 m at
top, domain 2500 m, **37 levels below 400 m**, 3 below 30 m. Cell count and cost are
**identical to Stage 1** (0.0735 s/step) — the depth doubles for free.

Nz = 154 was evaluated and rejected: 26% more cost for **one** extra level below 400 m.
Stretching redistributes resolution, it does not add it where the footprint lives.

## Restart overwrites grid and surface fields — the Stage 6 trap

`hydro_coreInit()` runs at `FEMAIN/FastEddy.c:157`; the restart read
`ioReadNetCDFinFileSingleTime()` runs at line 221, i.e. **after**, and walks the entire
registered variable list. That list includes **`xPos`, `yPos`, `zPos`, `topoPos` and
`z0m`**. Consequences, both load-bearing:

- **Trap.** Restarting a FLAT spin-up with a `topoFile` set leaves the solver with correct
  terrain-following metrics (`J31/J33/D_Jac`, built in `gridInit` before the read) but
  silently overwrites the *diagnostic* `zPos`/`topoPos` in every later dump with the flat
  values from the spin-up file. The LES is right and the output coordinates are wrong — so
  the LPDM places every particle at the wrong height with nothing to indicate it.
- **Lever.** The same mechanism is the ONLY way to give FastEddy v5.0.1 a spatially varying
  roughness. `z0m` is a 2-D field but is initialised uniformly from the scalar
  `surflayer_z0` (`hydro_core.c:1379`) and no input path exists for it. Writing `z0m` into
  the restart file works, and **no source change is needed** for the solar-array bulk patch.

`bin/prep_stage6.py` writes terrain, terrain-following `zPos`, and the roughness map into
the restart file so the read becomes a no-op and grid, output and LPDM stay consistent.

## Corpus structure (settled)

One spun-up **flat-terrain** state per `(stability, wind speed)` bin, **shared across all
wind directions in that bin**. Turbulence over a homogeneous surface has no preferred
horizontal direction, so the expensive part is paid once per bin rather than once per
direction — and direction is the axis needing the most samples.

```
  once per bin:        flat-terrain spinup to stationarity        ~5 h wall
  once per direction:  restart -> real rotated surface
                       -> ~20 min adjustment + 30 min sampling    ~2.11 h wall
```

At `dt = 0.0250 s` the cost is **2.94 s wall per simulated second**. A monolithic 3 h run
would be 8.82 h — above the ~4 h threshold. The split puts each production run at
**2.45 h**, under it.

**A sampling window is `t_back` + sampling time, not sampling time.** Measured 2026-08-19:
the first `t_back` seconds of a window produce no releases, because a backward trajectory
needs that much history behind it. With `t_back = 900 s`, a 30-min window yields only 21 min
of releases. **37.5 min is the floor**, not a converged choice — it buys the peak (12.5 min)
comfortably and leaves the centroid at ~336 m of p90 scatter. See the convergence table
below before sizing the corpus.

### Ensemble convergence — measured, and it is the corpus design parameter

From 18 sub-windows of 150 s within one integration (not from separate runs). The
sub-windows are **independent**: lag-1 autocorrelation +0.19 (peak) and -0.10 (centroid),
both below `2/sqrt(18) = 0.47`. So ensembles are bought with sampling *time inside one run*,
not with extra runs.

Measured two ways. The **randomised** held-out reference is the one to use: shuffle the 18,
take a random 9 as reference and `n` of the remaining 9 as the sample, 400 draws, so both
sides carry uncertainty. A *fixed* reference has only one available subset at `n = 9`, and
its "p90" there is a single draw — that artefact is why an earlier table read 120 m.

| n sub-windows | sampling time | peak p90 | centroid p90 |
|---|---|---|---|
| 3 | 7.5 min | 120 m | 615 m |
| **5** | **12.5 min** | **60 m (1 cell)** | 446 m |
| 9 | 22.5 min | 60 m | **336 m** |

**The peak converges at 12.5 min. The centroid never reaches 100 m in the measurable
range** — 336 m at 22.5 min and still improving. The centroid is tail-dominated and is the
expensive metric; 22.5 min is a floor, not a sufficient sampling time. The residual 60 m
peak offset between window halves is *systematic* — it tracks the residual spin-up drift —
so more averaging will not remove it; only a stationary spin-up will.

The curve also predicts terrain-run scatter: the Stage 6 windows leave 900 s of releases,
i.e. ~3 sub-windows per half, for which the fixed-reference table gave a centroid p90 of
893 m against a measured half-vs-half difference of 906 m. Under the randomised estimate
(615 m at `n = 3`) that single measurement sits between the median and the p90, which is
where one draw should sit.


