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

  **Never set `dt` from the stability boundary. Use the accuracy boundary with margin,
  and re-derive it whenever the grid changes.** Verify with `docker/diag_near_surface.py`:
  the first-level `w` variance ratio `k0/k1` must be **< 1** (~0.27 when correct, matching
  NCAR's NBL at 0.25). A value near 9 means dt is too large.

- **`hydroSubGridWrite = 0`** drops the 9 SGS stress fields: 19 -> 10 3-D fields,
  76.3 -> 40.3 B/cell, **212 GB -> 112 GB** per 30-min window at 5 s cadence. Free, via
  config. Reaching Stage 3's ~30 GB gate needs the 4 LPDM fields only, which is *not*
  config-reachable.

- **TRAP: FastEddy prints `****CORRUPTED***` on NaN/Inf but still exits 0.** A fully NaN
  field returns exit status 0. **Every run script must grep output for `CORRUPTED`/NaN and
  must never trust the exit code alone.** Use `docker/check_run.sh`.

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

| | value | note |
|---|---|---|
| `Nx x Ny x Nz` | 146 x 50 x 90 | padded 152/56/96, all divisible by 4/4/16 |
| `dx = dy` | 30.0 m | domain 4380 x 1500 m, same physical extent as the 10 m grid |
| `d_zeta` | 30.167598 | `zCeiling = d_zeta*(Nz-0.5) = 2700 m` |
| `verticalDeformFactor` | 0.662868 | `dz_sfc = 19.997 m` |
| **`dt`** | **0.0625 s** | `CFL_3d = 1.491`, 9% below the 1.64 accuracy limit |
| `dampingLayerDepth` | 540.0 | 20% of 2700 m |
| cells | 657,000 | |
| **measured cost** | **0.0066 s/step** | vs 0.00616 predicted from 9.37 ns/cell/step |

**The receptor lands on a cell centre at exactly 30.000 m.** `verticalDeformFactor` was
solved for that, not rounded to it: with `dz_sfc = 20 m` the k=1 centre is at half a
spacing above the k=0 centre, i.e. 1.5 x 20 = 30 m, and the cubic term's 4 mm was absorbed
by tuning the factor. There are 20 levels below 400 m and 1 below 30 m.

The measured 0.0066 s/step is 7% ABOVE the 9.37 ns/cell/step model — launch overhead at
657 k cells, which is 3x smaller than the smallest case the model was fitted to. Use the
measured value, not the model, for projections at this size.

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


