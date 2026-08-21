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
- FastEddy **v5.0.1**, on our fork, branch `kegonsa`
- FastEddy runs in **fp32**. **Confirmed (Stage 0b, 2026-08-17)** — see Conventions.

## Repository layout

```
Flux/                          <- working dir, main project repo root
├── PROJECT_BRIEF.md
├── PLAN.md
├── FASTEDDY_TRAPS.md          <- every trap that has cost GPU time. Read before running.
├── Dockerfile
├── inst.txt                   <- crude dependency/build notes, written for v4.0.1
├── data/raw/                  <- NOT in git. USGS 3DEP + ESA WorldCover.
└── FastEddy-model-5.0.1/      <- separate repo (the fork). Gitignored by the main repo.
```

## Build and run: Docker only

Local dependency installation is not used. FastEddy runs containerized (`flux-fasteddy:cuda118`),
and so does the analysis stack — **the host python has no scipy**, so every analysis script runs
inside the same image.

- Base `nvidia/cuda:11.8.0-devel-ubuntu22.04`, MPI + NetCDF-C + HDF5, compiled `sm_89`
- Host requires `nvidia-container-toolkit`. Run with `--gpus all`.
- Bind-mount `Flux/` so output lands on the host, not in the image.

---

# THE RECEPTOR IS AT 10 m

**Corrected 2026-08-21.** The instrument height is **~10 m AGL**, not 30 m. Everything
downstream of that number changes, and four of the changes are large enough to restate the
project:

**1. The footprint shrinks by roughly a factor of three, so the domain shrinks with it.**
Kljun `x90` at `z_m = 10 m`, `z0 = 0.03`:

| stability | `x_peak` | `x50` | `x80` | **`x90`** |
|---|---|---|---|---|
| very unstable | 45 | 116 | 338 | **680** |
| neutral | 51 | 132 | 384 | **766** |
| stable | 58 | 149 | 430 | **855** |
| very stable | 72 | 184 | 526 | **1033** |

At 30 m the same four rows were 1936 / 2253 / 2804 / 3751 m. **Every stability class now fits
inside a 1860 m box**, including the very-stable class that a 4464 m box could not hold — so
the wrap-cap truncation that capped the flat integral ~18% low is structurally resolved rather
than merely reduced.

**2. This tower measures the solar array, in every wind direction.** Fraction of the
crosswind-integrated footprint inside the array's upwind reach (E/W 60 m, S 100 m, N 250 m):

| stability | E/W (60 m) | S (100 m) | **N (250 m)** | *(same at 30 m)* |
|---|---|---|---|---|
| very unstable | **24.1%** | **44.4%** | **73.6%** | *0.1 / 3.3 / 32.4%* |
| neutral | **19.1%** | **39.3%** | **70.4%** | *0.0 / 1.1 / 24.3%* |
| stable | **14.8%** | **34.4%** | **67.2%** | *0.0 / 0.1 / 12.9%* |

The site question changes shape. At 30 m it was *"when does this tower see the array?"* —
answer, on northerlies only, a ~300x directional swing. At 10 m it is *"the tower always sees
the array; how much, and what mixes in?"* — the N-vs-E/W ratio falls from ~370x measured to
about **3.7x**. Stage 6's directional test is correspondingly weaker, and near-field fidelity
goes from important to being the whole game.

**3. The lake leaves the science.** Water is 1-1.5 km from the tower at the nearest. At 30 m
the 1000-1500 m band carried 7.5-11.4% of the crosswind-integrated footprint and the LES
measured 35.2% of a neutral easterly footprint as water. At 10 m the same band carries
**2.5-3.1%**, and only **7-9% of the footprint lies beyond 930 m at all**. An 1860 m domain
excludes the lake entirely, and that costs a few percent rather than a third. **Verify this
against the real WorldCover map before committing the domain size** — it is a Kljun estimate,
not a measurement.

**4. The receptor may be inside the roughness sublayer.** See the next section. This is the
serious one.

---

## The roughness sublayer, and what it invalidates

Solar panels are **2-3 m** tall. A roughness sublayer extends to roughly **2-5 canopy heights**,
i.e. **5-15 m**. A 10 m sensor standing inside the array therefore sits **at or inside the RSL**,
where Monin-Obukhov similarity does not hold. Three things depend on MOST and are weakened:

- **Kljun over the array is no longer a reference.** It is a MOST-based model evaluated where
  MOST fails. Over the array, report it as context and never as a target or an error score.
- **The MOST-anchored `sigma_w` floor loses its justification over the array.** The floor is
  the adopted sub-grid closure correction (`--sgs-most`), anchored to `sigma_w/u* = 1.25 phi_w`.
  Anchoring to surface-layer similarity inside the RSL is an extrapolation.
- **Displacement height enters everything.** With `d ~ 1-1.5 m`, similarity functions want
  `z - d = 8.5-9 m`, a 10-15% correction to every argument. FastEddy's surface layer has no
  `d`, so this is an LPDM-side correction only.

**The flat/neutral control is unaffected** — uniform `z0 = 0.03 m`, RSL top ~0.3-0.5 m, receptor
at 10 m, well clear. It stays the only place Kljun is diagnostic, and it is now the *only* place,
where at 30 m the whole domain qualified.

**A harder version of the same problem is numerical.** With `dz_sfc = 4.0 m` the first model
level sits at **z = 2.0 m** — at or below panel top, and inside the displacement layer. The
bulk-patch representation of the array (`z0 ~ 0.1-0.3 m`, `d ~ 1-1.5 m`) was chosen when the
first level was at 4.3 m. At 2.0 m with `z0 = 0.3 m` the surface-layer scheme is solving a log
law across `ln(z/z0) = 1.9`, which is not enough room for it to mean anything.

**Decision: set the array `z0 = 0.10 m`, the low end of the range**, so the first level keeps
`z/z0 = 20`. Record that the array's surface exchange is parameterised rather than resolved,
and treat it as the dominant known modelling uncertainty at this receptor height. `dz_sfc`
alternatives and their costs are in the grid section.

---

## Site

- UW-Madison Kegonsa Solar Array, southern Wisconsin
- **Tower coordinate, SURVEYED: `42.957160, -89.292362`** (EPSG:3071 577719.1, 276299.5).
  Single source of truth; lives in `TOWER_LON/TOWER_LAT` in `bin/prep_stage6.py`.
- **EC tower measurement height: ~10 m AGL** (corrected 2026-08-21; every result produced
  before that date used 30 m and its absolute distances do not carry over).
- **Solar array — THE TOWER IS INSIDE IT.** It extends **60 m east and west, 250 m north,
  100 m south** of the tower: 120 x 350 m, 4.20 ha. A rectangle in EPSG:3071; nothing about it
  depends on the wind.
- ~30 m of elevation change across the area.
- **Land cover comes from ESA WorldCover v200 (2021), 10 m.** Terrain from **USGS 3DEP
  1/3-arcsecond**. Both in `data/raw/`, gitignored; `bin/prep_surface.py` builds the model grid.
  Roughness per class (water 1e-4, grass 0.03, cropland 0.10, built 0.5, tree 1.0), then the
  array rectangle overrides it — WorldCover labels the array as cropland, because it does not
  see photovoltaics.
- **At `dx = 10 m` the 3DEP raster is at its native resolution**, so warping applies no
  smoothing and terrain slopes will be materially steeper than the 24 m map. Re-measure the
  slope distribution before setting the terrain `dt`, and consider a light smooth — the DEM's
  own noise now enters the metric tensor directly.
- **Terrain is tapered at the wrap seams; land cover is NOT.** Terrain height enters the
  coordinate transform and its metric tensor, so a seam step is a numerical cliff. Roughness
  and surface heat flux are local boundary conditions, where a seam is just a coastline.

---

## Domain configuration (10 m receptor)

| | value | note |
|---|---|---|
| `Nx x Ny x Nz` | **186 x 186 x 122** | padded 192/192/128 |
| `dx = dy` | **10.0 m** | domain **1860 x 1860 m** |
| `d_zeta` | **20.576132** | `zCeiling = d_zeta*(Nz-0.5) = 2500.0 m` |
| `verticalDeformFactor` | **0.194059** | `verticalDeformQuadCoeff = 0` |
| `dz_sfc` | **3.9933 m** | **`k = 2` centre at exactly 10.000000 m** |
| first four centres | 1.9966, 5.9933, **10.000000**, 14.0236 m | 2 cells below the receptor |
| `dz` at 400 m / at top | 14.2 m / 53.3 m | **55 levels below 400 m** |
| `dampingLayerDepth` | **500.0** | clean domain to 2000 m |
| thread block | **1 x 2 x 64** | `(N+6)` = 192/192/128, so 1/2/64 all divide |
| cells | **4.22 M** | same tensor shape as the 24 m campaign |
| **`dt`, flat** | **1/68 = 0.0147059 s** *(starting point)* | `CFL_3d = 1.468`. **Bisect it.** |
| **`dt`, terrain** | **to be bisected** | `dx/dz = 2.504`, milder than 24 m's 2.80 |
| cost | **2.44 GPU-h per simulated hour** | **2.24x** the 24 m grid's 1.09 |
| `Delta` / `z/Delta` | **7.36 m / 1.36** | at 24 m it was 17.02 / 1.76 |
| storage (`ioLPDMmode`) | 33.8 MB/dump | 35-min window at 5 s = **14.2 GB** |

**`(N + 6) % tB == 0` is satisfied**: `186 + 6 = 192 = 2^6 * 3` and `122 + 6 = 128 = 2^7`, so
`1 x 2 x 64` is legal and so is every other shape the block sweep found competitive. Keeping
`N = 186` also keeps the tensor shape identical to the fourth-pass campaign, so every script,
figure and the CNF raster shape carry over unchanged.

**`dz_sfc` and a 10 m receptor are not independent.** Cell centres sit at `(k+0.5)*dz_sfc`, so a
centre at exactly 10.000 m requires `dz_sfc = 10/(k+0.5)`:

| `k` | `dz_sfc` | first level | cells below receptor | `z/Delta` | cost, GPU-h/sim-h |
|---|---|---|---|---|---|
| 1 | 6.667 m | 3.33 m | 1 | 1.14 | **1.75** |
| **2** | **3.993 m** | **2.00 m** | **2** | **1.36** | **2.44** |
| 3 | 2.857 m | 1.43 m | 3 | 1.52 | 3.21 |

`k = 2` is the adopted choice. `k = 3` buys `z/Delta` 1.36 -> 1.52 for **+32%** cost and pushes
the first level to 1.43 m, deeper inside the panel displacement layer. `k = 1` is cheaper but
leaves one cell below the receptor, which the LPDM interpolation cannot work with.

**Fallback if the campaign budget binds:** `N = 162` at `dx = 10` (`L = 1620 m`, `N+6 = 168`,
`tB` to 56) costs **1.85 GPU-h/sim-h**, 24% less. It still contains `x90` for every class.

**Convective boundary layers are the binding constraint on `z_i`, not the footprint.** A
doubly-periodic CBL wants `L >= 4 z_i` or the largest thermals lock to the domain. At
`L = 1860 m` that caps `z_i` at **465 m** (620 m if you accept `L >= 3 z_i`). CONUS404 puts the
site's all-hours `z_i` median at 493 m and the **convective-midday median at 859 m** — so the
box supports the lower half of the distribution and **deep midday CBLs are out of reach at this
domain size.** State it wherever the corpus is described.

This is less damaging than it looks, for a reason specific to the new receptor height: Kljun's
only `z_i` channel is the factor `1/(1 - z_m/h)`, which at `z_m = 10 m` spans just **4.4%** over
`h = 200-1200 m`, against 14.7% at 30 m. **`z_i` is a nearly inert input to Kljun at 10 m.** It
still must be swept — the LES's real `z_i` dependence runs through `w*` and thermal structure,
which is exactly the residual the CNF is being asked to learn — but not in order to match Kljun.

---

## Boundary and initial conditions

- **Fully doubly periodic** in x and y. No Dirichlet inflow, no cell perturbation.
- Soundings from **CONUS404** (4 km, WY1980-2024) set initial theta/u/v profiles, geostrophic
  forcing, and surface heat flux ranges. **CONUS404 is a climatology, never a forcing.**
- Terrain tapered to a constant over the outer ring in both x and y; land cover is not.
- Each run is one quasi-stationary state, matching one 30-min EC averaging period.

### FastEddy capabilities confirmed in source 2026-08-21 — use these

Read out of `SRC/HYDRO_CORE/hydro_core.c` and the CUDA device files, not from documentation.

- **Geostrophic forcing supports a linear vertical gradient.** `z_Ug`, `z_Vg`, `Ug_grad`,
  `Vg_grad` (`hydro_core.c:655-662`, applied at `:1837-1845` and in
  `cuda_BCsDevice.cu:307-314`): below `z_Ug` the forcing is `U_g`; above it,
  `U_g + Ug_grad*(z - z_Ug)`. All four are **`PARAM_MANDATORY`** — they must appear in the
  `.in` file even when zero. Defaults `z_Ug = z_Vg = 10000 m`, gradients 0, which reproduces
  height-constant forcing exactly.
- **`stabilityScheme = 2` gives a 4-segment piecewise-linear base-state theta profile** via
  `zStableBottom{,2,3}` and `stableGradient{,2,3}` (`hydro_core.c:1776-1810`), hydrostatically
  integrated. All six are `PARAM_MANDATORY`. **Fit the CONUS404 sounding to these six numbers**
  — least squares on the mean profile, weighted toward the lowest 1.5 km. Fall back to injecting
  a 3-D theta field through the restart file only if the fit is demonstrably inadequate, and
  demonstrate it with a number before doing so.
- **`lsfSelector = 1` gives large-scale forcing**, and subsidence needs **both**
  `lsfSelector = 1` and **`lsf_horMnSubTerms = 1`** (`hydro_core.c:509`;
  `cuda_largeScaleForcingsDevice.cu`). Theta, qv and w forcings are each a two-level
  piecewise-linear profile (`lsf_*_surf`, `lsf_*_lev1`, `lsf_*_lev2` at `lsf_*_zlev1/2`).
  **Input values are per hour** — the kernel divides by 3600. Subsidence is applied against the
  **slab-mean** profile gradient, which is the correct formulation, and FastEddy warns that the
  slab mean is computed **per GPU**; single-rank, so it is exact for us.
  This is the physical fix for the Stage 2 stationarity failure: an idealised neutral BL with no
  capping inversion has no equilibrium depth and deepens forever. Note that arresting a CBL
  outright would need `w_sub(z_i) ~ -0.04 m/s`, well above realistic fair-weather subsidence
  (0.005-0.01 m/s), so expect subsidence to **slow** entrainment growth, not stop it. `z_i` is
  still measured and reported per window.
- **`surflayer_idealsine` gives a diurnal heat-flux cycle — and it is INCOMPATIBLE with the
  per-cell `htFlux` map.** `cuda_surfaceLayerDevice.cu:185-192`: when `surflayer_idealsine = 1`,
  **both branches assign a scalar** to `*htFlux`, overwriting the per-cell land-cover map. Only
  `surflayer_idealsine = 0` reaches the `// reuse *htFlux array values` branch. The per-cell map
  is load-bearing — it is what gives the array its 1.6x heat enhancement — so
  **`surflayer_idealsine` is rejected.** Diurnal variation is sampled instead as separate
  quasi-stationary states drawn from the CONUS404 distribution, which is what the one-state-per-
  averaging-period design already implies.
- **`moistureSelector = 1` requires `surflayer_wq`** (`hydro_core.c:534-545`) and adds a
  prognostic `qv`. **Decision: run DRY.** The footprint estimator transports a passive scalar and
  does not need moisture; the CNF's inputs are Kljun's dry scalars; and the cost is a second
  prognostic field in every dump. The one thing moisture would change is buoyancy, through
  `w'theta_v' = w'theta' + 0.61 T w'q'`. **That is absorbed for free by prescribing `htFlux` as
  the VIRTUAL heat flux rather than the sensible one** — do this, and record it, or `z_i` and
  `w*` come out ~5-10% low at this site's summer Bowen ratio.

---

## Rotation

Wind direction is set by **rotating the geostrophic vector**, `(U_g, V_g) = G(sin th, cos th)`,
not by rotating the map. The surface is built once and is bit-identical for every direction, so
any directional difference in the footprint is flow and cannot be a resampling artifact.

A square periodic domain with `dx = dy` over a flat uniform surface is exactly equivariant under
90-degree rotation, so one spun-up flat state re-indexes into four directions.

**Achieved direction is not forcing direction.** Ekman turning is 22-25 deg in the neutral cases
and 7-13 deg in the CBL. Either compensate the forcing angle or label cases by achieved
direction — do not silently mix the two.

## Solar panels

Represented as a **bulk surface patch**: elevated `z0`, displacement height `d ~1-1.5 m`, and a
raised surface heat flux. Do NOT use explicit geometry (URBAN/IBFM or GAD) — panel row spacing
is ~5-7 m and the grid cannot resolve it.

**`z0` for the array is 0.10 m at this receptor height**, not 0.1-0.3 — see the RSL section: the
first model level is at 2.0 m and a larger `z0` leaves the surface-layer scheme no room.

**Albedo has no pathway, and that is not an omission.** FastEddy in this configuration has no
radiation scheme — `surflayerSelector = 1` prescribes the kinematic surface heat flux directly —
so what albedo would have controlled is subsumed by `htFlux`, which IS per-cell
(`cuda_surfaceLayerDevice.cu:191` reuses the array when `surflayer_idealsine = 0`). `htFlux`,
`z0m`, `z0t` and `tskin` are all IO-registered, so they survive the restart read and
`bin/prep_stage6.py` writes them there. The built-in `surflayer_offshore` wave-roughness
parameterisations are a **global** switch and cannot be applied to water cells only.

Known omissions, accepted: directional roughness anisotropy from row alignment, diurnal tilt,
~20% shortwave leaving as electricity, and — new at this receptor height — **the entire
roughness sublayer**, which is now inside the measurement height rather than safely below it.

---

## Footprint computation

Backward LPDM, run offline on saved FastEddy output.

- **THE FOOTPRINT RASTER IS THE LES GRID.** Touchdowns are binned by their LES column index,
  folded modulo the periodic domain, so a footprint cell IS an LES column — the same indexing the
  land-cover masks use, and the array the CNF will consume. Cloud-in-cell deposition, which is
  exactly conservative and cuts per-cell noise to 0.67x nearest-grid-point.
- **The CNF raster is `Nx x Ny = 186 x 186`. Confirmed by `ncdump` 2026-08-21: FastEddy output
  contains NO halos** — `xIndex = 186, yIndex = 186, zIndex = 122` on a `186 x 186 x 122` run.
  The `2*Nh` padding is internal to the solver and never reaches the file.
- Kljun is evaluated at the static cells' own coordinates rather than rotated onto them
  (`lpdm.kljun.footprint_on_static`, 8x8 sub-sampling per cell). **Note the return signature:
  `crosswind_integrated` returns `(fy, xs)`, a tuple, not an array.**
- **A WINDOW IS (30 min + `t_back`), NOT 30 min.** The first `t_back` seconds of any window
  produce no releases, because a backward trajectory needs that much history behind it. The
  averaging period stays 30 minutes — that is how eddy covariance is defined. `--rel-seconds 1800`
  holds the release period to exactly 30 min however long the window is.
- **`t_back` must be re-measured at 10 m.** At 30 m the shape converged by 450-600 s and
  production used 900 s. Descent time scales roughly with `z/sigma_w`, so expect the median
  transit to fall from ~180-290 s to **~60-95 s** and the convergence point to ~150-250 s, giving
  **35-minute windows**. That is an estimate: measure it the same way, by masking one release
  ensemble on touchdown age. No new LES is needed once the first window exists.
- Save 3 velocity components + SGS TKE at ~5 s cadence.
- The analysis cache is **float16**. `scipy.ndimage.map_coordinates` refuses float16, so the 4-D
  linear interpolation in `lpdm/fields.py` is written out by hand — verified against
  `map_coordinates` to float32 roundoff, and marginally faster.
- Pipeline per run: LES -> scratch -> LPDM -> 2-D footprint (~1 MB) -> **delete fields**.
  Peak storage is one run. Never accumulate.
- SGS component is a Langevin model driven by FastEddy's output SGS TKE (Weil et al. 2004).
- **Well-mixed condition is the critical correctness test, and it MUST be run in the
  configuration footprints are actually computed in** (`stage4_wellmixed.py --sgs-most`).

## ML model

- **Conditional normalizing flow.** Inputs are Kljun's scalars only.
- **Residual formulation**: predict Kljun + learned correction, not the raw footprint.
- Wind direction is still the dominant skill axis, but **it is a weaker axis at 10 m** — the
  array's N-vs-E/W footprint share ratio is ~3.7x rather than ~370x. Near-field structure is
  where the site-specific signal now lives.
- Pretrain on cheaply-generated analytical (Kljun/Kormann-Meixner) footprints, then fine-tune
  **all weights** on the LES corpus.
- Loss: not raw MSE. Log-space or per-sample normalization, plus physical terms (centroid
  displacement, 80% source-area overlap, integral = 1). Consider Sinkhorn/W2.
- **Split by LES run, never by sample.** Effective sample size for generalization is the number
  of *runs*, not samples.

---

## Conventions

- **Particle state in fp64** even though velocity fields are fp32.
- **Precision — Stage 0b PASSED, 2026-08-17.** FastEddy is hardwired fp32 with no build switch:
  bare `float` on every prognostic field, `MPI_FLOAT` in the halo exchange, `NC_FLOAT` in the
  writer. Nothing to change; recorded so it is never re-litigated.
- **FastEddy is NOT bitwise reproducible.** Two runs of the same case differ by ~1e-4 relative in
  velocity and ~7e-4 K in theta after 200 steps. Never diff two runs expecting equality; any
  "did my change matter?" test compares against this floor, not against zero.
- **Restart is a true bit-for-bit state resume** (verified twice, at two grids). Restarting from a
  dump and re-dumping reproduces that dump byte-for-byte. Requires netCDF; `ioOutputMode = 1`
  binary output is **not** restartable.
- Any grid change re-checks the `(N + 6) % tB == 0` rule before running.
- Commit at every verification gate in PLAN.md. Do not proceed past a failed gate.
- **The analysis stack lives in the container.** The host python has no scipy; run analysis as
  `docker run --rm -v /home/atyagi/Flux:/w -w /w flux-fasteddy:cuda118 python3 ...`.

## Ruled out — do not propose these

Evaluated and rejected. Re-proposing them wastes time.

- **STILT** — replaced by the project's own backward LPDM.
- **Mesoscale coupling** (`hydroBCs=1`, GenICBCs, cell perturbation) — the CP fetch requirement
  would consume most of the domain. Periodic instead.
- **LES-to-LES nesting**, **NSCBC**, **512^3 domains** — schedule / unnecessary / infeasible.
- **Running FastEddy backwards in time** — mathematically impossible, not a code limitation.
  Reversing t and u flips the sign of the SGS stress term, giving negative eddy viscosity and the
  backward heat equation. Backward LPDM steps *particles* backward through *forward-stored* fields.
- **Multiple virtual tower locations** — would inject unexplained variance. One fixed tower.
- **Surface fields as ML inputs** ("Experiment 2") — out of scope.
- **`surflayer_idealsine` / diurnal cycle within a run** — overwrites the per-cell `htFlux` map.
- **Moisture (`moistureSelector = 1`)** — run dry, prescribe the virtual heat flux instead.
- **Sub-grid-fraction < 40% as a gate** — retired. See below.
- **24 m vs 12 m convergence test** — dropped; the grid is changing anyway.
- **Online footprint calculation inside FastEddy** — rejected 2026-08-21. Two reasons, both
  measured. **(i) It solves a problem we do not have**: IO is ~3% of compute and a window is
  14-22 GB against a 30 GB budget, so there is nothing to buy. **(ii) It would be a worse
  estimator.** FastEddy's auxiliary scalars advance forward in time and resolve source *tiles*,
  so an online footprint means one forward tracer per source region — the footprint's resolution
  becomes the number of tracers you can afford, and the near field, which is where the whole
  signal is at a 10 m receptor, would be the coarsest part. The backward LPDM gets the full raster
  from one release ensemble. It would also lock the footprint definition into the LES run, so any
  change to `t_back`, the weighting, or the closure would require re-running the LES rather than
  re-running 10 minutes of LPDM.
- FNO / U-Net may be *benchmarked* against the CNF, but CNF is the primary architecture.

---

## Settled by measurement — do not re-derive

### `dt` is set by the acoustic CFL, and the accuracy limit is below the stability limit

FastEddy is fully compressible with RK3 and **no acoustic sub-stepping and no CFL machinery at
all** — `dt` is a mandatory user constant, never computed or checked. Tutorial values are
hand-picked and mutually inconsistent; never copy them.

| | CFL_3d | behaviour |
|---|---|---|
| stability limit | ~1.79 | above this: NaN, `CORRUPTED` |
| **accuracy limit** | **~1.64** | above this: **silent** grid-scale acoustic noise |
| production | ~1.47-1.50 | ~10% margin |

Between the two the model runs to completion, exits 0, prints no warning, and produces resolved
`w` at the lowest few levels that is grid-scale acoustic noise rather than turbulence. Everything
else looks perfectly fine, which is what makes it dangerous. **The accuracy boundary is a property
of `CFL_3d`, not of the spacing** (confirmed at 10 m and again at 30 m).

**Verify every run with `docker/diag_near_surface.py`: the first-level `w` variance ratio `k0/k1`
must be `< 1`** (~0.27 when correct, matching NCAR's NBL at 0.25). A value near 9 means `dt` is
too large.

**TERRAIN AMPLIFIES THE EFFECTIVE CFL, and the amplification scales with GRID ANISOTROPY:**

    CFL_eff  ~  CFL_3d * sqrt(1 + (slope * dx/dz)^2)

At the new grid `dx/dz = 2.504`, milder than the 24 m grid's 2.80 and the 30 m grid's 3.50. But
the 3DEP raster is at native resolution at `dx = 10 m`, so the **slope distribution itself will be
steeper** and must be re-measured rather than carried over. **Never set `dt` from the stability
boundary. Use the accuracy boundary with margin, re-derive it whenever the grid changes, and
multiply the margin by the terrain amplification before any run with topography.**

**Vertical stretching is not a speed lever.** With `dx` fixed, even an infinitely coarse vertical
relaxes the 3-D CFL by at most `sqrt(3/2)`. Stretch for domain depth, never for speed.

### Cost and thread blocks

- **8.51 ns/cell/step** with the `1 x 2 x 64` block. Two cases 12x apart in size agreed to 1.7%.
- **`tBx` MUST BE 1.** `i <- threadIdx.x` while `kStride = 1` and `iStride = (Ny+6)(Nz+6)`, so any
  `tBx > 1` makes adjacent threads in a warp read addresses `iStride` floats apart — one 128-byte
  transaction becomes four 32-byte ones. Every shipped tutorial uses `tBx = 1`. The old `4x4x16`
  default was costing **17%**.
- Best measured shapes: `1x2x64` (0.0359 s/step at 186x186x122), `1x6x32`, `1x3x64`.
  **`tBz = 128` is rejected by the device** — CUDA caps `blockDim.z` at 64.
- The divisibility rule is enforced on **per-rank, halo-inclusive** extents (`grid.c:222-240`);
  Nz is never decomposed.
- Below ~1 M cells the ns/cell/step model is 7% optimistic — launch overhead. Use measured values.

### Output configuration

- **`ioLPDMmode` (fork)** writes only the fields the LPDM reads and CF-packs the 3-D prognostics
  to 16 bit: **8 B/cell**. Verified harmless on real fields — fp16 vs fp32 footprints differ by
  0 m in peak and 19 m in centroid, against a 59.2% source-area error floor.
- **`ioLPDMfullFrq` (fork) makes a sampling window chainable.** Lean output is deliberately not
  restartable (`rho` and `pressure` are absent), so a whole window had to be one invocation.
  `ioLPDMfullFrq = N` writes any output whose ABSOLUTE step is a multiple of `N` in full upstream
  form while every other dump stays lean. `bin/run_window.sh` is the driver.
- **`hydroSubGridWrite = 0`** drops the 9 SGS stress fields when running in upstream mode.

### Sub-grid fraction — the gate is retired, not merely failing

The resolved fraction of `sigma_w^2` collapses onto `z / Delta`, `Delta = (dx dy dz)^(1/3)`,
crossing 40% at `z/Delta ~ 3.5` (neutral) and `~2.1` (convective — thermals draw on `z_i`-scale
motions, so convection nearly halves the requirement).

**Lowering the receptor to 10 m makes this strictly harder, because the energy-containing eddy
scale shrinks with height while `Delta` does not.** At the new grid `z/Delta = 1.36`, against 1.76
on the 24 m grid at a 30 m receptor. Reaching `z/Delta = 3.5` at `z = 10 m` needs `Delta <= 2.9 m`,
i.e. `dx ~ 3-4 m` — roughly **55 GPU-h per simulated hour** (`465 x 465 x 170` at `dx = 4 m`), **22x** this configuration and out of reach.

**So the 40% gate is retired.** It is not a bar this project can clear at any affordable grid, and
carrying it forward as a permanent FAIL communicates nothing. What replaces it:

1. **Measure and report** the sub-grid fraction every pass. It is a real number and it belongs in
   the results; it is just not a gate.
2. **The well-mixed test, run in the production closure configuration**, is the correctness gate
   for the closure. It has already caught one real violation that the 40% number never would have.
3. **Bound the closure's influence** with the anchor-sensitivity band already measured: the choice
   of `sigma_w` anchor is worth **46-66% shape L1**, against a 38% sampling floor. Quote that band
   wherever a near-field number is quoted.
4. **State it as the dominant known uncertainty**, alongside the RSL caveat, which now compounds it.

### Backward-LPDM traps, settled by measurement

- **RESCALING THE SUB-GRID VARIANCE BREAKS THE WELL-MIXED CONDITION UNLESS THE DRIFT IS RESCALED
  WITH IT.** Thomson's reverse-time drift contains `d(sigma^2)/dz`. `--sgs-most` multiplies the
  variance by a height-dependent `sc(z)`, so the gradient the drift needs is
  `sc*dsig2dz + (2/3)*e*dsc/dz`, and `dsc/dz` is the larger of the two. Using the unscaled
  gradient made the flux-footprint integral climb past 1 and keep going. Fixed by the product rule,
  which reduces to the unscaled field exactly when `sc = 1`.
  Two lessons beyond the arithmetic. **An integral that crosses 1 and keeps climbing cannot be
  truncation** — a finite backward time can only lose influence — so it is always a model
  inconsistency. And **the well-mixed gate must be run in the configuration the footprints are
  actually computed in.**
- **The footprint integral is a statement about the DOMAIN, not about `t_back`.** At `t_back = 900 s`
  the flat/neutral integral was 0.888 — and Kljun, evaluated on the IDENTICAL box, integrates to
  **0.875**. Never report the shortfall as an estimator error without quoting Kljun on the same
  cells. **At a 10 m receptor this should improve sharply**: only 7-9% of the footprint lies beyond
  930 m, against 23-36% at 30 m.
- **Periodic wrap-around double-counts the footprint. Always cap trajectory displacement at one
  streamwise domain length.** Uncapped, the integral climbs past 1 exactly as wrapping sets in.
  Capped, it converges from below.
- **The cap makes domain length a correctness constraint, not just a fetch one.** At 30 m the
  capped flat integral saturated ~18% short because the 80% source area (3810 m) approached the
  domain length (4380 m). At 10 m, `x90 <= 1033 m` in a 1860 m box, so this is structurally fixed.
- **The flux weight is frame-dependent, but the frame is a ~2% effect.** The double rotation
  (Wilczak et al. 2001) is used because it is the frame the instrument reports in.
- **Over a slope the footprint integral is not 1, and that is physical.** The residual is `w_bar`
  times the concentration integral. At a receptor in mean subsidence the turbulent flux genuinely
  is not the surface flux — the advection non-closure that makes EC hard in complex terrain.
- **The touchdown weight uses the surface-normal approach rate**, `|d(z-z_ground)/dt|`, not `|w|`.
  Over sloping ground a particle loses height-above-surface because the ground rises under it, and
  the `2/|w|` weight explodes. Flat ground hides this completely.

### Ensemble convergence — the corpus design parameter

From 18 independent 150 s sub-windows within one integration (lag-1 autocorrelation +0.19 peak,
-0.10 centroid, both below `2/sqrt(18) = 0.47`). Ensembles are bought with sampling *time inside
one run*, not with extra runs. Randomised held-out reference, 400 draws:

| n sub-windows | sampling time | peak p90 | centroid p90 |
|---|---|---|---|
| 3 | 7.5 min | 120 m | 615 m |
| **5** | **12.5 min** | **60 m (1 cell)** | 446 m |
| 9 | 22.5 min | 60 m | **336 m** |

**The peak converges at 12.5 min. The centroid never reaches 100 m in the measurable range.**
Measured at a 30 m receptor on a 24 m grid; the absolute metres will shrink with the footprint at
10 m, but the shape of the curve is the transferable part. **Re-measure once.**

---

## Restart overwrites grid and surface fields — the Stage 6 trap and the Stage 6 lever

`hydro_coreInit()` runs at `FEMAIN/FastEddy.c:157`; the restart read runs at line 221, i.e.
**after**, and walks the entire registered variable list — which includes **`xPos`, `yPos`,
`zPos`, `topoPos` and `z0m`**.

- **Trap.** Restarting a FLAT spin-up with a `topoFile` set leaves correct terrain-following
  metrics but silently overwrites the *diagnostic* `zPos`/`topoPos` in every later dump with flat
  values. The LES is right and the output coordinates are wrong — so the LPDM places every particle
  at the wrong height with nothing to indicate it.
- **Lever.** The same mechanism is the ONLY way to give FastEddy v5.0.1 a spatially varying
  roughness or heat flux. `z0m` is a 2-D field initialised uniformly from the scalar
  `surflayer_z0` with no input path. Writing it into the restart file works, **no source change
  needed**.

`bin/prep_stage6.py` writes terrain, terrain-following `zPos`, and the surface maps into the
restart file so the read becomes a no-op and grid, output and LPDM stay consistent.

Every other trap that has cost GPU time lives in **`FASTEDDY_TRAPS.md`**. Read it before running.

---

## Site climatology — CONUS404, and what it is for

`bin/conus404_site.py` streams a stratified 45-year hourly sample at the tower cell off the USGS
Open Storage Network pod over plain HTTPS. `bin/conus404_dist.py` summarises it.

**It sets sweep ranges and sampling density. It never forces a run.** No per-case sounding, no
projection matching, no time-varying boundary conditions. A 1.9 km doubly-periodic box cannot
sustain mesoscale forcing anyway.

Measured at the tower (39,456 hourly records, 1979-2024), QC'd at `u* >= 0.15 m/s` (65.2%):

| | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| `z_i` | 80 m | 267 m | 493 m | 835 m | 1475 m |
| `w'theta'` | -0.027 | -0.006 | +0.015 | +0.076 | +0.164 K m/s |
| `u*` | 0.17 | 0.24 | 0.32 | 0.44 | 0.65 m/s |
| `U(30 m)` | 2.4 | 3.9 | 5.2 | 6.8 | 10.0 m/s |

**The site is unstable more than half the time**: 27.2% very unstable (`z/L < -0.5`), 30.3%
unstable, 13.3% near-neutral, 20.4% stable, 8.8% very stable. A neutral-only corpus misses the
modal daytime state.

**`z_i` must still be swept**, but see the domain section: the box supports `z_i <~ 465 m`, and
Kljun's `z_i` channel is nearly inert at a 10 m receptor anyway. Sweep it for the LES residual,
and state the deep-CBL exclusion.

**The wind rose and the array signal point in different directions.** Rose: S 16.0%, W 14.4%,
NW 14.5%, SW 14.3% against N 10.6%, NE 10.2%, E 10.4%, SE 9.8%. Direction sampling needs a
**floor**, not pure rose weighting — though the case for it is weaker at 10 m, where the array is
in the footprint from every direction.

Convective midday reference (local 10-16 h, `w'theta' > 0.05`, n = 7,461): `z_i` p50 **859 m**,
`w'theta'` p50 **0.109 K m/s**, `u*` p50 0.40, `U(30)` p50 5.4 m/s, `z_i/L` p50 **-19.8**.

## Convective configuration

Dry CBL, `surflayer_wth` prescribed as the **virtual** kinematic heat flux, mixed layer under a
capping inversion via `stabilityScheme = 2`, and **`lsfSelector = 1` with `lsf_horMnSubTerms = 1`
for subsidence** to hold `z_i` inside what the domain supports.

**`z_i` grows by entrainment and that is not drift** — a convective boundary layer has no
stationary depth. Subsidence slows it; it does not stop it. The achieved `z_i` is measured and
reported per window.

**Surface heat flux is per-cell, from the land cover** (`prep_surface.py --wth`). Water 0.12 of
the land value, built 1.5, tree/grass 1.1, cropland 1.0, **array 1.6** — PV modules are darker
than the crop they replaced and do not transpire, and field studies of utility-scale arrays report
a daytime sensible enhancement of order 1.5-2. With no radiation scheme, `htFlux` is the channel
albedo would have acted through.

## Corpus structure

One spun-up **flat-terrain** state per `(stability, wind speed)` bin, **shared across all wind
directions in that bin** by 90-degree re-indexing.

```
  once per bin:        flat-terrain spinup to stationarity     ~5 h simulated, 12.2 GPU-h, 26 segments
  once per direction:  restart -> real rotated surface
                       -> ~20 min adjustment + (30 min + t_back) sampling   2.24 GPU-h
```

At 2.44 GPU-h per simulated hour, **nothing is a single run** — every spin-up is ~26 chained
segments and every window is 3-4. The 45-minute wall cap is enforced by the drivers
(`bin/run_window.sh`, `bin/run_directions.sh`), which project before launching and refuse.

An 8-case campaign (2 stability x 4 directions from 2 base states) costs about **42 GPU-h**,
against 20 GPU-h for the same campaign at the 24 m grid.

---

## Status

**FIFTH PASS — PLANNED, NOT RUN.** See `PLAN.md`. The receptor moved from 30 m to 10 m on
2026-08-21 and the grid is being rebuilt around it. Everything below is the state of the
*previous* configuration and its absolute distances do not carry over; the methodology,
the traps, and the closure findings do.

**FOURTH PASS COMPLETE, 2026-08-21** — `FOURTH_PASS_RESULTS.md`. Eight production cases on the
static `186 x 186 x 122` @ 24 m domain, **at a 30 m receptor**: four wind directions in each of
two stability regimes, all from two spun-up states by 90-degree re-indexing, 30 minutes of
releases each.

| stage | gate | neutral | convective |
|---|---|---|---|
| 2 | stationarity | PASS | PASS — `w*/u*` 2.86, entrainment ratio 0.149 |
| 3 | window < 30 GB | PASS 22 GB, chainable | PASS |
| 4 | well-mixed | PASS backward rms **3.61%** vs a 5.48% counting floor | inherited |
| 5 | sub-grid < 40% | FAIL 85.5% | FAIL 52.3% — **gate now retired** |
| 5 | error floor | 37-54% overlap, centroid 152-436 m | 43-51%, centroid 15-90 m |
| 6 | explicable difference | PASS array swing **368x** | PASS **528x** |

**Convection changes what a 30 m tower measures.** On a convective northerly the solar array
supplied 48% of the flux from 0.22% of the domain — 222x its area share, against 3.01% neutrally.
The lake ran the other way: 15.3% of the neutral easterly footprint, 5.3% convectively. Both
follow from one per-cell `htFlux` map and the flow.

**Two real bugs were found by the standing flat/neutral control on its first run:** the `sigma_w`
floor was breaking the well-mixed condition, and `run_case.sh` was scoring the wrong dump for
every window run since the third pass. Both are in `FOURTH_PASS_RESULTS.md` §5 and
`FASTEDDY_TRAPS.md`.

Earlier passes: `STAGE0A_RESULTS.md`, `STAGE1_RESULTS.md`, `STAGE2-6_RESULTS.md`,
`STAGE2-6_RESULTS_V2.md`, `THIRD_PASS_RESULTS.md`. All superseded on absolute numbers.

See @PLAN.md for the staged path.
