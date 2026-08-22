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
Kljun evaluated on the REAL WorldCover/3DEP map at `z_m = 10 m` (`bin/phaseA_geometry.py`,
`results/phaseA_geometry.txt`), not on idealised distances:

| stability | `x_peak` | `x90` | cells to the peak at `dx = 16 m` |
|---|---|---|---|
| very unstable | 27 m | 397 m | 1.7 |
| unstable | 33 m | 478 m | 2.1 |
| neutral | 38 m | 538 m | 2.4 |
| stable | 48 m | 662 m | 3.0 |
| very stable | 91 m | 1089 m | 5.7 |

At 30 m the same classes ran 1936-3751 m. **Every class fits inside a 1952 m box**, so the
wrap-cap truncation that capped the flat integral ~18% low is structurally resolved. Note
the last column: the peak sits 1.7-5.7 cells from the tower, which bounds how sharply the
CNF target can represent it. Recorded, not gated -- the grid is set by corpus economics and
the near field is closure-dominated at `z/Delta ~ 1` regardless.

**2. This tower measures the solar array, in every wind direction -- and by MORE than the
idealised table said.** Footprint-weighted array share on the real map, `z_m = 10 m`:

| stability | E/W | S | **N** | N/E ratio | *(idealised estimate)* |
|---|---|---|---|---|---|
| very unstable | **43.6%** | 58.1% | **70.4%** | 1.61x | *24.1 / 44.4 / 73.6%* |
| unstable | **35.4%** | 58.5% | **78.1%** | 2.21x | |
| neutral | **29.9%** | 55.5% | **80.6%** | **2.69x** | *19.1 / 39.3 / 70.4%* |
| stable | **20.3%** | 45.9% | **72.9%** | 3.59x | *14.8 / 34.4 / 67.2%* |
| very stable | 3.0% | 21.2% | 59.3% | 19.5x | |

The idealised numbers were crosswind-INTEGRATED fractions inside the array's upwind reach
along a line from the tower. **The tower is inside a 2-D rectangle**, so flux arriving from
crosswind angles still lands on the array and the real share is 1.4-1.6x larger --
which makes the **N-vs-E/W RATIO smaller**, 3.7x -> **2.69x** neutral. Gate F must lean on
**absolute share by direction**, not on the ratio.

**3. The lake has left the science entirely.** Measured, not estimated: the 1952 m box
contains **8 water cells of 14,884 (0.05%)**, and the worst-case footprint water share over
every direction and stability is **0.01%**. At 30 m the LES measured 35.2% of a neutral
easterly footprint as water. Land cover in the new box is crop 50.8%, tree 23.5%, grass
20.5%, built 4.9%, array 1.03%.

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
`z/z0 = 20` (`ln(z/z0) = 2.99` at the 1.997 m first level). Record that the array's surface
exchange is parameterised rather than resolved, and treat it as the dominant known modelling
uncertainty at this receptor height.

**AND RECORD WHAT IT COSTS, because it is not free and it is easy to miss.** WorldCover
labels the array as cropland, whose `z0` is ALSO 0.10 m. At `z0_array = 0.10` the override
therefore changes nothing at all: **the array is aerodynamically identical to the surface it
replaced, and its entire NEUTRAL signal is zero**, leaving only the convective heat-flux
contrast. `bin/prep_surface.py` now prints a warning when the two coincide rather than
letting it pass silently.

The way out is `--raise-topo`: put the displacement height into `topoPos` over the array, so
the first model level sits 2.0 m above the RAISED surface (3.5 m above bare ground, clear of
panel top) and a larger `z0` has room. **Which of the two is right is a measured
sensitivity, not a decision** -- see the displacement-height treatments in the plan.

**Displacement height is first-order here, and it was absent.** Kljun at `z_m = 10.0 -> 8.5 m`
moves the array's E/W share **29.9% -> 38.2% (1.28x)** and `x90` 701 -> 596 m. `d` now enters
the LPDM sub-layer log law, the MOST-anchored `sigma_w` floor (at the RECEPTOR column, not
the domain mean -- 23.5% of the box is tree cover whose `d ~ 0.7 h_c` is metres the LES never
resolves), and Kljun's `z_m`.

**The receptor datum is 10 m above BARE GROUND (AGL).** So over the array the effective
aerodynamic height is `z - d ~ 8.5 m`, and if `topoPos` is raised by `d` the receptor must be
released at a FRACTIONAL level (`stage5_footprint.py --exact-agl`) to stay 10 m above true
ground. Snapping to the nearest level there would put it 10 m above the PANELS -- an 11.5 m
receptor, a 15% error in exactly the quantity this pass exists to get right.

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
- **Land cover of the 1952 m box, measured**: cropland 50.8%, tree 23.5%, grass 20.5%,
  built 4.9%, bare 0.3%, **water 0.05% (8 cells)**, array override 1.03% (154 cells, 3.94 ha).
  Terrain relief 34.6 m raw; after mean removal and the taper, -18.2 to +13.9 m.
- **At `dx = 16 m` the 3DEP raster (native 10 m) is only mildly coarsened**, so warping does
  smooth a little. Measured slopes after 2 x (1-2-1): p50 0.041, p90 0.096, p99 0.146,
  max 0.188. With `dx/dz_sfc = 4.007` that is a CFL amplification of 1.072 at p90 and
  **1.252 at the steepest cell** — the projection that brackets the terrain `dt` search.
- **Tree cells are the grid's worst case, and it is inherent.** `z0 = 1.0 m` with a first
  model level at 1.997 m leaves `ln(z/z0) = 0.69` — the surface-layer scheme has almost no
  room there. 23.5% of the box is tree. Recorded as a limitation of `dx = 16 m` with a 10 m
  receptor, not fixable at this grid.
- The domain's **geometric-mean `z0` is 0.1435 m** (used for `surflayer_z0` in the flat
  spin-up); the drag-weighted effective value at 10 m is 0.2902 m, twice that. The spin-up
  uses the geometric mean, consistent with prior passes, and the ~20 min adjustment on the
  real surface absorbs the difference.
- **Terrain is tapered at the wrap seams; land cover is NOT.** Terrain height enters the
  coordinate transform and its metric tensor, so a seam step is a numerical cliff. Roughness
  and surface heat flux are local boundary conditions, where a seam is just a coastline.

---

## Domain configuration (10 m receptor, 122^3 @ 16 m)

**Grid decision, made 2026-08-22 and not to be reopened.** Chosen for corpus economics:
targets are needed in quantity and per-target precision is secondary.

| | value | note |
|---|---|---|
| `Nx x Ny x Nz` | **122 x 122 x 122** | `(N+6) = 128 = 2^7` in ALL THREE |
| `dx = dy` | **16.0 m** | domain **1952 x 1952 m** |
| `d_zeta` | **20.576132** | `zCeiling = d_zeta*(Nz-0.5) = 2500.0 m` |
| `verticalDeformFactor` | **0.194059** | `verticalDeformQuadCoeff = 0` |
| `dz_sfc` | **3.9933 m** | **`k = 2` centre at exactly 10.000000000 m** |
| first four centres | 1.9966, 5.9933, **10.000000**, 14.0236 m | 2 cells below the receptor |
| `dz` at 400 m / at top | 14.4 m / 53.3 m | **55 levels below 400 m**, 84 below 1000 m |
| `dampingLayerDepth` | **500.0** | clean domain to 2000 m; supports `z_i` to ~1000 m |
| thread block | **1 x 2 x 64** | **measured fastest**, see below |
| cells | **1.816 M** | |
| **`dt`, flat** | **0.0146417 s** | `CFL_3d = 1.35`, **measured**, see below |
| **`dt`, terrain** | *to be bisected* | `dx/dz = 4.007`, worse than any previous grid |
| cost | **0.0149 s/step measured** -> **0.94-0.99 GPU-h per simulated hour** | |
| `Delta` / `z/Delta` | **10.09 m / 0.99** | at 24 m it was 17.02 / 1.76 |
| storage (`ioLPDMmode`) | 14.5 MB/dump | a 2400 s window at 5 s = **7.0 GB** |
| wall cap | **1 hour** | = **1.02 simulated hours per segment** |

**`bin/vgrid.py` solves this grid from FastEddy's own `zDeform`** (`grid.c:1114-1127`), so it
is never hand arithmetic in a comment again. With `verticalDeformQuadCoeff = 0`,

    z(zeta) = ((1 - c1)/zC^2) zeta^3 + c1 zeta,  zeta_k = (k+1/2) d_zeta,  zC = (Nz-1/2) d_zeta

which is LINEAR in `c1`, so pinning a cell centre to an exact height is a division, not a
root-find. It reproduces the retired 24 m grid exactly (`d_zeta` 24.691358, factor 0.346601,
k=3 at 30.000000 m).

**`dt` is set by `CFL_3d = c dt sqrt(2/dx^2 + 1/dz_sfc^2)`, c = 347.2 m/s.** That form
reproduces the 24 m grid's stated 1.4946 at its `dt` to four digits. Here `1/CFL` = 92.202,
so `dt = CFL/92.202`. A 5 s output cadence needs an integer step count; 0.0146417 gives
341.5 steps, so **spin-up runs use a 300 s cadence and sampling windows re-derive `dt` to
land the cadence on an integer** (`bin/run_window.sh` asserts it).

**Convective boundary layers are the binding constraint on `z_i`.** `L >= 4 z_i` caps `z_i`
at **488 m**; `L >= 2 z_i` at **976 m**. Measured coverage of this site's convective-midday
hours (`bin/zi_coverage.py`, `results/zi_coverage.txt`):

| rule | `z_i` cap | all QC | unstable | **convective midday** |
|---|---|---|---|---|
| `L >= 4 z_i` | 488 m | 49.5% | 45.7% | **19.3%** |
| `L >= 3 z_i` | 651 m | 63.4% | 57.7% | **33.6%** |
| `L >= 2 z_i` | 976 m | 81.3% | 76.1% | **60.9%** |

**And the cap is BIASED, not merely restrictive.** `z_i` and surface heat flux are
positively correlated (rank correlation **+0.43** over convective midday), so the excluded
deep-CBL hours carry **1.51x the heat flux** and **1.58x the `w*`** of the representable
ones. A `z_i`-capped corpus is thinnest exactly where the array's flux enhancement is
largest. **Whether the 4 `z_i` rule is binding for a 10 m FOOTPRINT is a separate and
measurable question** -- it was written for `w*` scaling and entrainment -- and
`bin/domain_adequacy.py` answers it. The fallback if it is binding is `218^2 @ 16 m`
(`L = 3488 m`, 3.2x cost, 53.0% convective-midday coverage at `L >= 4 z_i`).

This matters less than it looks for Kljun: its only `z_i` channel is `1/(1 - z_m/h)`, which
at `z_m = 10 m` moves the array share by **1.0 percentage point** over `h = 200-1200 m` and
`x90` by 3.5%. **`z_i` is a nearly inert input to Kljun at 10 m.** It still must be swept --
the LES's real `z_i` dependence runs through `w*` and thermal structure, which is the
residual the CNF is being asked to learn -- but not in order to match Kljun.

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
- **Fitting `stabilityScheme = 2` to a CONUS404 mean sounding** — impossible, and it was
  the wrong idea anyway. Checked in the store's own `.zmetadata` 2026-08-22: `conus404_hourly`
  carries **no time-varying atmospheric profiles**. Its only 4-D variables are soil and snow
  (`SH2O`, `SMOIS`, `TSLB`, `SNICE`, `SNLIQ`, `TSNO`, `ZSNSO`); `PB` and `PHB` are static
  base-state fields with no time axis. And the fit would have been self-defeating: the fitted
  mean `z_i` (~860 m) is a state a 1952 m box cannot hold. **The capping inversion is a
  CONTROL on `z_i` here, not a target to match**, and it is set to hold the case's `z_i`.
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
| accuracy limit, 24-30 m grids | ~1.64 | above this: **silent** grid-scale acoustic noise |
| **accuracy limit, 122^3 @ 16 m** | **~1.51** | measured 2026-08-22: 1.50 clean, 1.53 gives k0/k1 = 8.7 |
| **production, this grid** | **1.35** | `dt = 0.0146417 s`, ~10% margin |

**The accuracy boundary is NOT the same number at every grid.** PROJECT_BRIEF.md previously said it
was "a property of `CFL_3d`, not of the spacing (confirmed at 10 m and again at 30 m)". On
this grid it is **~1.51, not ~1.64**, and the transition is sharp: `CFL_3d = 1.50` gives
`k0/k1 = 0.610` and `1.53` gives `8.681`. What changed is the grid ANISOTROPY --
`dx/dz_sfc = 4.007` here against 2.80 at 24 m -- so treat the boundary as something to
**re-measure at every grid**, never to carry over. The 10% margin rule survives; the number
it is applied to does not.

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
- Best measured shapes at 186x186x122: `1x2x64` (0.0359 s/step), `1x6x32`, `1x3x64`.
  **`tBz = 128` is rejected by the device** — CUDA caps `blockDim.z` at 64.
- **At 122^3 the legal set is different**: `(N+6) = 128 = 2^7` in all three, so `tBy`/`tBz`
  must be POWERS OF TWO and the 24 m grid's runners-up `1x6x32` and `1x3x64` are illegal.
  Swept 2026-08-22, 300 steps each: **`1x2x64` fastest at 0.01475 s/step compute**, then
  `1x2x32` (0.01485) and `1x8x16` (0.01490); `1x16x16` 12% slower, `1x32x8` **46% slower**.
- **Measured 0.0149 s/step at 122^3 with a spin-up IO cadence**, 0.0155 with a 5 s window
  cadence. The 8.51 ns/cell/step model predicts 0.01545 — 3.5% pessimistic here.
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

### The terrain taper width, measured

The terrain is tapered to a constant over an outer ring so the periodic seam is not a
numerical cliff. The ring costs real geography, and at a 1952 m box that is no longer
cheap. Swept 2026-08-22:

| `pad` | ring | real terrain reaches | slope p90 | slope max | CFL amplification |
|---|---|---|---|---|---|
| 20 | 320 m | 656 m | 0.0789 | 0.1838 | 1.242 |
| **12** | **192 m** | **784 m** | 0.0964 | **0.1880** | **1.252** |
| 10 | 160 m | 816 m | 0.1011 | 0.2102 | 1.307 |
| 8 | 128 m | 848 m | 0.1059 | 0.2217 | 1.338 |

**`pad = 12` is the knee.** Going 20 -> 12 buys 128 m of real terrain — which is what covers
Kljun's `x90` here — for **+0.8%** CFL amplification. At 10 and 8 the TAPER becomes the
steepest cell in the domain and starts setting the terrain `dt` itself, which is the wrong
thing to be paying for.

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

**Surface heat flux is per-cell, from the land cover, and it is the VIRTUAL flux**
(`prep_surface.py --wth-sensible`). The per-class numbers in the literature are SENSIBLE-flux
ratios — water 0.12, built 1.5, bare 1.4, tree/grass 1.1, cropland 1.0, **array 1.6** (PV
modules are darker than the crop they replaced and do not transpire; field studies of
utility-scale arrays report a daytime sensible enhancement of order 1.5-2). But the field
FastEddy is given must be the VIRTUAL flux, because the run is dry and buoyancy is what
`htFlux` is for. **The conversion is Bowen-ratio dependent and therefore class-dependent:**

    w'th_v' = w'th' + 0.61 th w'q' = w'th' (1 + 0.61 th c_p / (B L_v)) = w'th' (1 + 0.0735/B)

| class | B | s->v factor | sensible ratio | **virtual ratio** |
|---|---|---|---|---|
| cropland (reference) | 0.4 | 1.184 | 1.00 | **1.000** |
| **solar array** | 4 | 1.018 | 1.60 | **1.376** |
| built | 2 | 1.037 | 1.50 | 1.314 |
| bare | 2 | 1.037 | 1.40 | 1.226 |
| tree | 0.4 | 1.184 | 1.10 | 1.100 |
| grassland | 0.5 | 1.147 | 1.10 | 1.066 |
| wetland | 0.2 | 1.368 | 0.60 | 0.693 |
| water | 0.15 | 1.490 | 0.12 | **0.151** |

**Working in virtual flux COMPRESSES the wet-dry buoyancy contrast**, because the wetter
surface's larger latent flux buys buoyancy back. That is physically correct and it is exactly
what the decision to run dry is trading for. The fourth pass prescribed CONUS404's SENSIBLE
flux directly and applied sensible ratios to it; PROJECT_BRIEF.md predicted that would cost 5-10% in
`z_i` and `w*`.

At the convective-midday reference (sensible p50 **0.109 K m/s**): cropland virtual
**0.1290 K m/s (149 W/m2)**, **array 0.1776 (205 W/m2)**, water 0.0195. The array's own value
barely moves from the fourth pass (0.176 -> 0.178, the two corrections nearly cancel) but the
**array-to-water contrast falls ~32%** — and that contrast is what the directional signal is
made of. The array multiplier is insensitive to its OWN Bowen ratio (1.37-1.40 over B = 2-6)
and sensitive to cropland's (1.31-1.45 over B = 0.3-0.6); sweep it with `--bowen-crop`.

**The `.in` scalar `surflayer_wth` must be the DOMAIN MEAN of that map, not the cropland
reference.** A flat spin-up has no restart injection, so the scalar IS its flux; using the
reference instead spins up a boundary layer at the wrong `z_i` for the run it feeds. On this
domain the mean is **0.1363 K m/s** against a 0.1290 reference — and note the sign flipped
from the 24 m domain (0.1006 vs 0.11), where 16% water pulled the mean DOWN. The water is
gone; tree and built now pull it up. `prep_surface.py` prints it.

## Corpus structure

One spun-up **flat-terrain** state per `(stability, wind speed)` bin, **shared across all wind
directions in that bin** by 90-degree re-indexing.

```
  once per bin:        flat-terrain spinup to stationarity     ~5 h simulated, ~4.9 GPU-h, 5 segments
  once per direction:  restart -> real static surface
                       -> ~20 min adjustment + (30 min + t_back) sampling   ~0.9 GPU-h, 1 segment
```

At ~0.97 GPU-h per simulated hour and a **1-hour wall cap**, one segment is **1.02 simulated
hours** — so a spin-up is 5-6 chained segments and **a whole sampling window fits in ONE**.
The cap is enforced by the drivers (`bin/run_window.sh`, `bin/spin_cbl.sh`), which project
before launching and refuse. `run_window.sh` derives both the planner and the refusal from a
single `WALLCAP`; they used to be two independent constants that happened to agree.

An 8-case campaign (2 stability x 4 directions from 2 base states) costs about **12 GPU-h**,
against 42 at the 186^2 @ 10 m grid and 20 at 24 m. That is the corpus economics the grid was
chosen for.

---

## Status

**FIFTH PASS IN PROGRESS, 2026-08-22.** Rebuilt around a **10 m receptor on a
`122 x 122 x 122` @ 16 m grid** (1952 m domain), chosen for corpus economics. Phase A
complete and committed; Phase B smoke batch mostly complete.

| gate | result |
|---|---|
| **A1 water share** | **PASS** — worst case **0.01%** over every direction and stability, against a 10% threshold. The lake is outside the box and it costs nothing. |
| **A** geometry + `z_i` coverage | **PASS**, committed. Array share on the real map is 1.4-1.6x the idealised estimate; the N-vs-E/W ratio falls to 2.69x neutral. |
| **B1** grid launch | **PASS** — 122^3 launches clean, 0.0149 s/step |
| **B2** thread blocks | **PASS** — `1x2x64` fastest of 9 legal shapes at 0.01475 s/step |
| **B3** flat `dt` | **PASS** — accuracy boundary **~1.51**, production `CFL_3d = 1.35` (`dt = 0.0146417`) |
| **B4** terrain `dt` | pending — needs a developed terrain state |
| **B5** restart injection | **PASS** — `topoPos` 9.5e-7, `z0m` 1.5e-9, `htFlux` exact, `zPos` 1.2e-4 against the terrain-following formula |
| **B6** 90-deg equivariance | **PASS** — rotation exact to 1.2e-14; after 200 steps mean wind agrees to 1.7e-5, column TKE to 1.2e-3 |
| **B7** subsidence | **PASS**, after a FastEddy source fix — theta warms at 1.10x the prescribed `-w_sub dtheta/dz` |
| **B8** halo check | **PASS** — `xIndex = yIndex = zIndex = 122`, interior only. **The CNF raster is 122 x 122.** |

**A real FastEddy bug, found and fixed on the fork.** `lsf_horMnSubTerms = 1` with
`moistureSelector = 0` dies instantly with an illegal memory access: `cuda_lsfSlabMeans()`
launches the qv slab-mean over `moistScalars_d`, and `cudaDevice_lsfRHS` writes `Frhs_qv`,
both unconditionally — while `cuda_moistureDeviceSetup()` allocates them only when
`moistureSelector > 0`. **Upstream v5.0.1 subsidence is only usable with moisture on.** Both
are now guarded on `kegonsa`; see `FASTEDDY_TRAPS.md` §10. Same class as the `NORHO` bug,
differing only in whether the bad pointer trapped or produced `inf`.

**And a plan error it exposed:** PLAN.md asked the smoke test to confirm that `w` acquires
the prescribed subsidence. It never will. The scheme adds the tendency to `U`, `V`, `THETA`
and `qv` — there is no `W_INDX` term — so subsidence is a large-scale ADVECTION tendency
against the slab-mean gradient, not a resolved vertical motion. The check would have failed a
correct implementation. The real test is differential on theta, and it passes.

**A live bug in our own analysis path**, found before it cost anything:
`stage5_footprint.py` never passed `z_target` to `compute_footprint`, so it fell through to
the 30.0 default and every footprint would have been computed on the level nearest 30 m with
nothing in the output to say so. Fixed, along with four other hard-coded 30 m receptors.

**FOURTH PASS COMPLETE, 2026-08-21** — `FOURTH_PASS_RESULTS.md`. Eight production cases on a
static `186 x 186 x 122` @ 24 m domain, **at a 30 m receptor**. Its absolute distances do not
carry over; the methodology, the traps and the closure findings do.

| stage | gate | neutral | convective |
|---|---|---|---|
| 2 | stationarity | PASS | PASS — `w*/u*` 2.86, entrainment ratio 0.149 |
| 3 | window < 30 GB | PASS 22 GB, chainable | PASS |
| 4 | well-mixed | PASS backward rms **3.61%** vs a 5.48% counting floor | inherited |
| 5 | sub-grid < 40% | FAIL 85.5% | FAIL 52.3% — **gate now retired** |
| 5 | error floor | 37-54% overlap, centroid 152-436 m | 43-51%, centroid 15-90 m |
| 6 | explicable difference | PASS array swing **368x** | PASS **528x** |

**Two real bugs were found by the standing flat/neutral control on its first run:** the
`sigma_w` floor was breaking the well-mixed condition, and `run_case.sh` was scoring the
wrong dump for every window run since the third pass. Both are in `FOURTH_PASS_RESULTS.md` §5
and `FASTEDDY_TRAPS.md`.

Earlier passes: `STAGE0A_RESULTS.md`, `STAGE1_RESULTS.md`, `STAGE2-6_RESULTS.md`,
`STAGE2-6_RESULTS_V2.md`, `THIRD_PASS_RESULTS.md`. All superseded on absolute numbers.

See @PLAN.md for the staged path.
