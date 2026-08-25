# The seed library and the sounding-forced corpus

> **Status, 2026-08-25.** The pipeline is BUILT and validated offline on branch
> `library-states`. Stages 1, 2, 4 and 8 run end to end; stage 3 generates 18 portable
> jobs. What remains is GPU time: spin the 18 seeds, then run the corpus.
> `main` is untouched.

---

## What this is for

The emulator is trained on **real calculated footprints**, not on analytical ones. Each
training record is one LES run:

| | |
|---|---|
| **input** | Kljun's scalars, every one read off the LES window itself |
| **target** | the 122 x 122 LPDM flux footprint computed on that same window |

The corpus is **~1825 cases**, one per day over five years, each forced by a real HRRR
pseudo-sounding at the tower. That is what makes the corpus represent the weather rather
than a sweep: an SBL on 1 February at 04 UTC is a different state from an SBL on
2 September at 11 UTC, and both are in it.

**Seed states exist only to delete the spin-up.** They are pre-spun flat, uniform,
doubly-periodic turbulence fields. A case restarts from the nearest seed, adjusts for
30 minutes under its *own* sounding's forcing, then samples for 30 minutes. Seeds are never
training data, and 18 of them are not a corpus.

Costed at the **measured** `t_back = 600 s` (`results/tback_production.txt`), not the
150-250 s PROJECT_BRIEF.md estimated before it was measured, so a case is
`1800 + 1800 + 600 = 4200 s = 1.167` simulated hours at ~0.97 GPU-h per simulated hour:

| | GPU-h |
|---|---|
| 1825 cases x (30 min adjust + 30 min window + 600 s `t_back`) = 1.167 sim-h | **2065** |
| the seed library, 18 x 3.0 sim-h | **52** |
| **total** | **2117** |
| the same corpus, cold-started at 3 h of spin-up each | **7376** |

**52 GPU-h buys back about 5250.** The corpus is the real cost; the library is rounding
error beside it.

A case's window is 42 min of wall clock and its adjustment 31, so **both fit inside the
1-hour cap as single segments** -- which is what makes 1825 of them schedulable at all.

---

## The library: 6 rungs x 3 base angles = 18 states

### Why these axes, and no others

Sized by **what 30 minutes cannot adjust**, which is the only criterion that matters for a
state whose entire purpose is to be adjusted away. Every number below is measured on this
project's own runs:

| quantity | closed in 30 min? | evidence | axis? |
|---|---|---|---|
| **direction** | no, ~2.7 deg | -5.4 deg/h backing, `g16_spin` | **yes -- 3 base angles -> 12 headings** |
| **z_i** | no, ~+40 m | +79 m/h entrainment, `g16_cbl_shallow` | **yes -- 6 real depths** |
| **stability regime** | no | a CBL needs ~8 `T*` ~ 1.2 h to turn over | **yes -- in the rungs** |
| u\* / wind speed | partly | the surface layer is ~0.1 `z_i` deep, so it re-equilibrates in ~2 min at a 10 m receptor | no |
| fine `z/L` | yes | the surface flux is prescribed and the surface layer follows | no |

### The rungs are coupled, not a product

A 150 m stable boundary layer cannot carry a 12 m/s geostrophic wind -- shear that strong
destroys the stratification that defines it. So `G` belongs to the rung. Six rungs walk the
site's real joint `(z_i, flux, wind)` distribution as CONUS404 measures it at this tower
(`z_i` p25/p50/p75 = 267/493/835 m; `w'th'` p25/p50/p75 = -0.006/+0.015/+0.076 K m/s;
`U(30 m)` p25/p50/p75 = 3.9/5.2/6.8 m/s):

| rung | regime | `z_i` target | `w'th_v'` | `G` | how `z_i` is held |
|---|---|---|---|---|---|
| `sbl` | stable | 150 m | **-0.020** | 6 m/s | surface cooling; no neutral layer, inversion from the ground |
| `nbl-shallow` | neutral | 300 m | 0.000 | 8 m/s | capping inversion alone |
| `nbl-deep` | neutral | 550 m | 0.000 | 12 m/s | capping inversion alone |
| `cbl-shallow` | convective | 450 m | +0.060 | 7 m/s | cap + subsidence |
| `cbl-mid` | convective | 700 m | +0.110 | 9 m/s | cap + subsidence |
| `cbl-deep` | convective | 950 m | +0.160 | 11 m/s | cap + subsidence |

The capping inversion is **+8 K across 100 m** (`stableGradient = 0.08`), then a
free-atmosphere lapse of 0.004 K/m. That is the `z_i` **control**, exactly as PROJECT_BRIEF.md
already says it must be -- not a profile to be matched.

`cbl-deep` stops at 950 m because **the 1952 m box supports `z_i <= 976 m`** at `L >= 2 z_i`,
the rule Phase E validated (the stricter `L >= 4 z_i` is not binding for a 10 m footprint,
p ~ 0.54). `bin/sounding_to_forcing.py` flags any sounding above that cap as
`representable: false` rather than running it and mis-labelling it.

### Three base angles, not four

A square doubly-periodic flat uniform domain with `dx = dy` is exactly equivariant under
90-degree rotation -- Gate B6 measured the rotation exact to **1.2e-14** -- so each base
angle re-indexes into four headings and `{0, 30, 60}` covers the compass on a clean 30 deg
grid: **0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330**.

Under a **smooth** reconstruction -- which is what a CNF learns; it does not
piecewise-linearly interpolate -- `{0,30,60}` reconstructs the Kljun
array-share-vs-bearing curve to **0.80 points** maximum error, about 4x below the
**3.03-point** LES sampling standard error. An uneven four-angle set scores **worse**
(1.36) because it makes the spline ring. Four evenly-spaced angles would leave the same
15 deg worst case as three, for 6 more spin-ups.

### Per-state spin-up spec

3.0 simulated hours, chained as **4 segments of 45.9 min wall** each, under the 1-hour cap.
`dt = 0.0146199 s` (`CFL_3d = 1.3480`), a stationarity dump every **300 s**.

**The gate is on `U/u*`, not on `u*`.** A doubly-periodic neutral Ekman layer forced by a
constant geostrophic wind does not settle to a fixed `u*` on any affordable timescale:
`f = 9.94e-5` here, so the inertial period is **17.6 h**, and `u*` falls for a quarter of it
and then rises. Measured on `g16_spin`, `u*` moved **-27%** over 6.26 simulated hours while
`U/u*` was within **0.31%** of its final value by 3.01 h. **Gating on `u*` alone failed this
project's spin-ups twice for a reason that was never a modelling error.**

Kljun's `Pi_4 = U(z_m)/u*` is the only channel through which the wind enters the streamwise
footprint shape, and both of its terms ride the oscillation together. The seven limits,
scored on the last 1.5 h (`bin/seed_stationarity.py`):

| quantity | limit |
|---|---|
| `U/u*` (Kljun `Pi_4`) | **1.0 %/h** |
| `sigma_v/u*` | 3.0 %/h |
| `sigma_w/u*` at the receptor | 2.0 %/h |
| `TKE/u*^2` | 5.0 %/h |
| `z_i` | 3.0 %/h |
| Kljun `x_peak` | 1.0 %/h |
| Kljun `x90` | 1.0 %/h |

3.0 h is the first duration where all seven pass; 2.0 h fails on `TKE/u*^2` at +15.6 %/h.
That is where the flat 3.0 h budget comes from.

**The gate runs inside the job**, so the ~36 x 73 MB stationarity dumps never leave the
machine that made them -- the verdict travels home as a few kB of JSON.

**Seeds are labelled by what they ACHIEVED, not by what they were asked for.**
`jobs/run_seed.sh` writes the measured `z_i`, `u*`, `U` and direction into
`manifest["achieved"]`, and `bin/pick_seed.py` matches on those. PROJECT_BRIEF.md already requires
this for direction; the same argument applies to depth, and a seed that entrained past its
target simply *is* a different rung than the one it was aimed at.

### Job structure, for rented GPUs

Each job is one directory: **one `.in`, one manifest, one entrypoint, no absolute paths, no
shared state.** The repo root is discovered from the entrypoint's own location and exported
as `FLUX_ROOT`, which is why `docker/run_case.sh`, `docker/pyrun.sh`, `bin/spin_cbl.sh`,
`bin/run_directions.sh` and `bin/preflight.sh` now read that variable instead of a literal
`/home/atyagi/Flux`.

- **requires**: an sm_89 GPU (checked, and a mismatch warns with the consequence spelled
  out: newer JITs from PTX and is slower, older will not run at all), ~1.6 GB VRAM, the
  `flux-fasteddy:cuda118` image, and a checkout of this repo.
- **returns**: `seed_restart.nc` (73.3 MB, all 22 variables), `stationarity.json`,
  `stationarity.txt`, `seed.log`, and the manifest with `achieved` filled in.
- **resumable**: the chain restarts from the newest dump on disk, so a kill costs at most
  one segment.

> **Bitwise reproducibility will not hold across different physical GPUs.** Stated, not
> fixed. FastEddy is already non-reproducible run-to-run on *one* GPU -- ~1e-4 relative in
> velocity and ~7e-4 K in theta after 200 steps. Seeds are turbulence realisations, so this
> costs nothing; it is recorded so nobody later diffs two seeds expecting equality.

---

## The pipeline, stage by stage

| # | stage | file | status |
|---|---|---|---|
| 1 | HRRR pseudo-sounding at the tower | `bin/hrrr_sounding.py` | **built, validated** |
| 2 | sounding -> FastEddy `.in` parameters | `bin/sounding_to_forcing.py` | **built, validated** |
| 3 | the 18 portable seed jobs | `bin/make_seed_jobs.py`, `jobs/run_seed.sh` | **built**, awaiting GPU |
| 3 | this case's surface heat-flux map | `bin/case_surface.py` | **built, validated** |
| 4 | which seed does this case restart from | `bin/pick_seed.py` | **built, validated** |
| 5 | restart + surface injection + adjust | `bin/prep_restart.py`, `bin/prep_stage6.py` | existing |
| 6 | the sampling window | `bin/run_window.sh` | existing |
| 7 | LPDM -> footprint | `bin/stage5_footprint.py`, `lpdm/` | existing |
| 8 | assemble one (input, target) record | `bin/make_pair.py` | **built, validated** |
| — | the stationarity gate, portable | `bin/seed_stationarity.py` | **built** |
| — | offline validation of 1-2 across regimes | `bin/test_sounding.py` | **built** |
| — | one case, end to end | `bin/run_corpus_case.sh` | **built** |
| — | a short cold start per regime config | `bin/smoke_check.py` | **built, PASS x3** |
| — | Gate B6 re-run convectively | `bin/b6_convective.sh` | **built** |

### 1. `bin/hrrr_sounding.py`

Herbie, HRRR `nat` product (**hybrid levels**, against `prs`'s 40 pressure levels which put
3-4 levels in the whole boundary layer), `fxx=0` analysis, plus `sfc` for `HPBL`, `SHTFL`,
`LHTFL`, `PRES` and the 10 m wind.

**The download is the corpus's largest data cost, and it was going to be 561 GB.** GRIB
byte-range subsetting works per **message**, not per area, so 6 variables x 50 hybrid
levels is 300 full CONUS fields — **measured at 315 MB per timestamp** on the first six
cached subsets. Three changes, each verified rather than assumed:

| | | per case |
|---|---|---|
| as first written | 6 vars x 50 levels | 315 MB, cached |
| drop `SPFH` | nothing reads it: the run is dry and buoyancy comes from the virtual `htFlux` | 228 MB |
| **lowest 20 hybrid levels only** | verified against the file's own inventory: HRRR numbers level 1 at the **model bottom** (level 1 at 289 m ASL here, level 20 at 6413 m, level 50 at 27176 m), so 1-20 reaches ~6.1 km AGL | **~91 MB** |
| delete the GRIB after extraction | the durable artifact is the 8 kB sounding | **8 kB at rest** |

~6.1 km contains everything downstream needs: the 2500 m LES column, the 4 km ceiling on
the `z_i` searches, and the above-BL geostrophic layer, which tops out at `z_i + 550
<= 1526 m` for the deepest representable case. **Checked, not asserted:** fetching one
timestamp at 20 and at 50 levels gives bit-identical `z_i` (all three diagnostics),
surface fluxes, Bowen ratio, 10 m wind, `U_g`, `V_g`, geostrophic speed and direction, and
a shared profile matching to **0.000e+00** in both `z` and `theta`. The restriction is
exactly free.

Together: ~416 GB of transfer becomes ~166 GB, and cache-at-rest goes from 561 GB to
15 MB.

**Two traps, both of which produce plausible wrong numbers rather than errors:**

1. **HRRR GRIB winds are GRID-relative.** On the Lambert grid at this longitude that is a
   **5.11 deg** rotation -- most of a direction bin in a 12-direction library, and
   *invisible in the wind speed*, which is rotation-invariant. Rotated with pyproj's own
   meridian convergence rather than a hand-rolled Lambert formula. `bin/conus404_dist.py`
   hits the identical issue and quotes 5.5 deg for CONUS404's grid; the agreement is the
   cross-check. **This was found because the first implementation caught the failure in a
   bare `except` and left the angle at exactly 0.0 with no warning.**
2. **HRRR longitudes run 0-360.** Left unnormalised, the geostrophic box test
   `|lon - lon0| <= box` matched **zero** points and the code fell back to the above-BL
   proxy silently. Both the empty-box path and the no-projection path now warn.

**Four `z_i` diagnostics, one of them primary.** `HPBL` is primary because it is HRRR's own
PBL-scheme depth, so it is the *same* diagnostic in all ~1825 cases -- consistency across
the corpus beats any per-case improvement. Reported alongside it:

| diagnostic | 2023-07-15 19Z | what it is |
|---|---|---|
| `HPBL` | **1648 m** | HRRR's MYNN TKE-threshold depth |
| bulk Richardson (`Rb = 0.25`) | 1106 m | regime-independent; works in a stable layer too |
| parcel (`theta_ml + 0.5 K`) | 1244 m | the classic convective mixed-layer top |
| max theta gradient | *2041 m* | **recorded, not used -- see below** |

> The max-gradient pick was the original estimator and it is quietly wrong. On a summer
> profile with **no capping inversion** the free troposphere runs 3.8-4.9 K/km all the way
> up, so the maximum lands at 2041 m on a boundary layer HRRR puts at 1648 m and whose
> mixed layer ends near 1250 m. It is kept in the output, named and labelled, because
> deleting it would only invite it back.

**Geostrophic wind: `above_bl` is primary, and that is a decision, not a default.** FastEddy
runs doubly periodic on a 1952 m box, so it can represent neither synoptic curvature nor a
horizontal height gradient; forcing it with the height-gradient geostrophic wind would
drive a boundary layer far stronger than the one HRRR has. Measured on 2023-07-15 19Z: the
actual wind is 6.2 m/s where the 850 mb gradient says 10.7, and the profile shows exactly
why -- **850 mb (~1230 m AGL) sits *inside* a 1648 m boundary layer**, so its wind is
sub-geostrophic and backed 28 deg, which is Ekman balance behaving correctly. The
height-gradient estimate is kept as a recorded diagnostic and the disagreement is reported
rather than hidden.

The above-BL wind is a **height** average over `[z_i+50, z_i+550]` on a uniform 25-point
grid, not a level average: above a 1648 m boundary layer that 600 m slab held **exactly one
hybrid level**, so a level-mean was a single sample taken wherever that level happened to
fall, in a layer where the direction wobbles by ~10 deg.

### 2. `bin/sounding_to_forcing.py`

**The base-state fit, and the three constraints FastEddy puts on it.** Read out of
`SRC/HYDRO_CORE/hydro_core.c:1776-1822`. The base state is continuous piecewise-linear in
theta with four segments:

```
z <= b1        theta = theta_grnd                    (FORCED NEUTRAL -- no free gradient)
b1 < z <= b2   theta = theta_grnd + g1 (z - b1)
b2 < z <= b3   theta = ...       + g2 (z - b2)
z  > b3        theta = ...       + g3 (z - b3)
```

1. **The lowest segment has no free gradient.** For a CBL that is what you want and `b1` is
   the mixed-layer top. A **stable** case has no neutral layer to give it, so the fit drives
   `b1` to 0 and the first gradient segment starts at the ground. Nothing special is needed;
   it falls out of leaving `b1` free.
2. **All three gradients must be strictly positive** -- queried over `[FLT_MIN, FLT_MAX]`
   (`hydro_core.c:642,646,650`), so zero and negative are both rejected. Floored at
   **1e-4 K/m** (0.1 K/km), physically negligible over any segment the LES column holds.
3. **AND A REJECTED VALUE DOES NOT STOP THE RUN.** `parameters.c:309-315` prints
   `ERROR: parameter '<name>' value <v> is outside limits`, increments `numErrors`, and
   **leaves the variable at its compiled-in default** -- and `FastEddy.c:96` never checks
   the return code of `hydro_coreGetParams()`. So an out-of-range `stableGradient` silently
   runs the case with **0.1 K/m**: a 10 K capping inversion where the sounding wanted 0.4.
   The only trace is one line in a log otherwise grepped for `CORRUPTED`. **New trap;
   recorded in `FASTEDDY_TRAPS.md` §13.** This stage guarantees the ranges rather than
   hoping, and `bin/test_sounding.py` re-checks every one against the source's own limits.

The pressure integral carries `(1/g) log(1 + g dz/theta)`, which looks like it would lose
the neutral limit as `g -> 0`. It does not: the literal `1.0` promotes the expression to
double, so the term is accurate to ~1e-13 relative at the floor. **The positivity
constraint is a parameter-range rule, not a numerical one.**

**The fit is done on the LES's own cell centres, weighted by layer thickness.** Fitting on
HRRR's ~13 levels below the ceiling under-resolves the inversion; fitting on the LES levels
*unweighted* over-resolves the surface layer, where 55 of 122 levels sit below 400 m and
the LES's own dynamics -- not the base state -- decide the answer anyway. Thickness
weighting makes the residual an integral over height, so it is invariant to how the grid is
stretched, and one deliberate 3x tier weight below 1.5 km is then the only thumb on it.

Also emitted: `(U_g, V_g)`; `surflayer_wth` as the **domain mean** of the per-cell virtual
map, with the cropland reference backed out by dividing by `mean(f)` (getting this backwards
spins the seed up at the wrong `z_i` -- PROJECT_BRIEF.md, 0.1363 vs 0.1290); the ground state; a
`dt` that lands the 5 s cadence on an integer step count; and subsidence with its knee at
the case's own `z_i`.

**The per-case Bowen ratio is a real improvement over the class table.** `SHTFL`/`LHTFL`
give the actual ratio for that hour, so the sensible-to-virtual conversion is exact rather
than derived from a land-cover Bowen assumption. On 2023-07-15 19Z: B = 0.44,
`w'th'` 0.1204 -> `w'th_v'` **0.1406 K m/s**.

**Direction is recorded, not corrected.** The forcing is the real above-BL wind and the LES
finds its own Ekman turning over the real Kegonsa roughness, which is the more faithful of
the two available choices. `dir10_residual_deg` carries HRRR's own 10 m direction minus the
Ekman prediction per case. It is **+19.3 deg** on 2023-07-15 19Z -- the profile *backs*
9 deg with height through a `z_i/L = -38` boundary layer, so **at this site the thermal wind
can exceed the 10 deg convective Ekman angle outright.** Recorded per case so a corpus-wide
bias is visible rather than assumed away. `--match-10m` rotates the forcing instead.

### 3. `bin/case_surface.py` — and the trap that made it necessary

`bin/prep_restart.py` injects `htFlux` into the restart file from the grid directory, and
**the restart read overwrites the `.in`'s `surflayer_wth`** — the same Stage 6 mechanism
PROJECT_BRIEF.md documents for terrain, pointed at the flux. `data/grid16` ships with `htFlux.npy`
**all zeros** because it is a neutral build. The retired per-bin campaign never hit this
because it built one fixed grid per regime (`data/grid16_cbl`, `data/grid16r_nbl`, …); a
sounding-forced corpus has ~1825 different fluxes and cannot. Point a convective case at
`data/grid16` and **it runs neutral, exits 0, and says nothing.**

The per-cell map is `wth_reference x f`, where `f` is a class-ratio field that does not
depend on the case at all — so the static geography is **hardlinked** and only
`htFlux.npy` is written fresh. A case directory is ~116 kB and no copy.

**Validated bit-for-bit against the campaign's own grid.** Given `data/grid16_cbl`'s
cropland reference (0.12903676), the output is **bit-identical** to it. Given a 4-decimal
rounding of that reference the difference is 5.060e-05, which the rounding times the
array's 1.3764 ratio predicts as 5.0598e-05 — so the ratio field itself is exact, and the
tables are read out of `prep_surface.py`'s **own source** rather than copied.

**The three regimes are not the same problem, and the third one is a decision:**

| flux | map | |
|---|---|---|
| `> 0` | per-class daytime ratios | array 1.376x the cropland reference (virtual) |
| `~ 0` | zero everywhere | which is what neutral means |
| `< 0` | **uniform** | the class table is a DAYTIME table; there is no nocturnal equivalent |

> **A stable corpus case carries no array signal at all.** Not thermally — the map is
> uniform, because applying daytime enhancement ratios to a negative flux would invent a
> nocturnal contrast nothing measured. And not aerodynamically — `z0_array = 0.10 m` is
> exactly WorldCover's cropland value, so the override changes nothing in any regime.
> **The array signal this project exists to resolve is a DAYTIME signal.** Stable cases
> are still real corpus points — they teach the flow, the terrain and the stability
> dependence of the footprint — but the corpus must be described that way rather than as
> uniformly array-sensitive.

### 4. `bin/pick_seed.py`

**The metric is "what will 30 minutes fail to close", and nothing else.**

- **regime** (stable / neutral / convective, from the *prescribed* virtual heat flux) is a
  **hard constraint**, not a cost -- a CBL turns over in ~1.2 h, so 30 min does not convert
  one regime into another.
- **`z_i`** costs `|ln(z_i_seed / z_i_case)| / ln 2`.
- **direction** costs `d_dir / 30 deg`, one library bin.
- **`z/L` and `u*` are reported and never costed** -- they re-equilibrate in ~2 min.

> An earlier version standardized every axis by the library's own sample spread. That is
> wrong in a way worth recording: the spread is a property of the library, not of the
> physics, so the **narrowest** axis gets the **largest** weight -- and with an unspun
> library whose `z/L` values were all placeholder estimates within 0.01 of each other,
> `z/L` ended up weighted **5x more heavily than `z_i`**, exactly inverting the table
> above. The scales are now fixed and physical.

Regime comes from the prescribed surface flux, so **no `u*` estimate enters the choice at
all** -- and a `u*` estimate is precisely what there is no honest way to get before the LES
has run.

**A mismatch does not corrupt a case.** Inputs are read off the LES window, so an
imperfectly-closed gap moves where a case *lands* in input space without making it wrong.
Seed spacing is a **coverage** question, not a correctness one.

### 8. `bin/make_pair.py`

`run_id` and `split_key` are written into every record. **Split by run, never by sample**:
the effective sample size for generalisation is the number of LES runs. `L = inf` is a
legitimate value (exactly neutral) and is carried as `1/L`, which is finite everywhere and
is the form the similarity functions use. `z_0` and the receptor height are recorded as
provenance, not offered as features -- a constant column is not a predictor, and this is a
single-tower emulator by design. Achieved-minus-requested deltas are carried per case, so
whether 30 minutes of adjustment is actually enough becomes measurable across 1825 cases
instead of assumed.

---

## Every file this added or changed

**New — pipeline**

| file | |
|---|---|
| `bin/hrrr_sounding.py` | stage 1: the pseudo-sounding at the tower |
| `bin/sounding_to_forcing.py` | stage 2: sounding -> the FastEddy `.in` parameters |
| `bin/case_surface.py` | stage 3: this case's per-cell surface heat flux |
| `bin/make_seed_jobs.py` | generates the 18 seed jobs and `jobs/index.json` |
| `bin/pick_seed.py` | stage 4: which seed, and which 90-degree rotation |
| `bin/make_pair.py` | stage 8: assemble one (input, target) record |
| `bin/run_corpus_case.sh` | one case end to end, timestamp -> training pair |
| `bin/run_corpus.sh` | the corpus: one case per day, resumable, with a skip ledger |
| `jobs/run_seed.sh` | the portable seed-job entrypoint |
| `jobs/README.md` | what a rented machine needs, and what comes back |

**New — gates and validation**

| file | |
|---|---|
| `bin/seed_stationarity.py` | the portable Gate C1; **the single definition of the seven limits** |
| `bin/smoke_check.py` | a short cold start per regime config, including the base-state closure |
| `bin/b6_convective.sh` | Gate B6 re-run convectively, scored against block sampling spread |
| `bin/test_sounding.py` | stages 1-2 across four regimes, offline |

**Changed**

| file | why |
|---|---|
| `docker/run_case.sh`, `docker/pyrun.sh` | repo root from `$FLUX_ROOT`, defaulting to the current value so nothing that already worked changes |
| `bin/spin_cbl.sh`, `bin/run_directions.sh`, `bin/run_window.sh` | same |
| `bin/preflight.sh` | discovers its own root; covers `jobs/*.sh`; every new entry point must answer `--help` |
| `bin/run_pass5.sh` | imports the seven limits from `seed_stationarity.py` instead of restating them |
| `Dockerfile` | `eccodes`, `cfgrib`, `herbie-data`, `pyproj`, `s3fs`, `scikit-learn` |
| `.gitignore` | `data/hrrr/`, `results/soundings/`, `results/forcing/`, `pairs/`, `data/case_grids/`, `data/smokelib/` |
| `PROJECT_BRIEF.md` | the forcing-source reversal and the four rules it contradicts; the stable-case array limitation; the sampling-spread tolerance rule; the convective B6 result |
| `PLAN.md` | points the corpus phase here |
| `FASTEDDY_TRAPS.md` | §13, an out-of-range parameter does not stop FastEddy |

**Deviation from the original plan:** it listed `runs/seed_base/*.in`. Each job's `.in` is
generated into `jobs/seed_*/seed.in` instead, so a job directory is self-contained and can
be shipped to a rented machine on its own. There is no shared template to fall out of sync.

---

## Two conventions, settled

**Averaging is period-ENDING.** `data/raw/H_and_sigma_w.csv` runs `2025-05-01 00:30` ->
`2026-05-01 00:00`, exactly 365 x 48 = 17,520 rows, so a record stamped `00:30` covers
`00:00-00:30`. Matching the tower:

> **A footprint stamped 01:00 UTC is the average over 00:30-01:00 UTC.** Adjustment runs
> 00:00-00:30, so integration begins at 00:00 UTC. But the LES has no absolute clock --
> forcing is constant and each run is one quasi-stationary state -- so what matters is
> which analysis sets the forcing, and the window midpoint 00:45 is nearest the 01:00
> analysis. **Use the HRRR analysis whose valid time equals the footprint timestamp.**

**Those timestamps are UTC, not local.** Three independent checks: solar noon at
lon -89.292 is 17.95 UTC and the median-H peak sits at 18:00 in the file clock; H crosses
zero at 13h and 23h file-clock (= 08:00 / 18:00 CDT); and reproducing PROJECT_BRIEF.md's own
CONUS404 numbers needs UTC-6 (`w'theta'` p50 0.111 against 0.109 published, `z_i` 830 m
against 859 m). Corrected to local midday, median H is **110 W/m2** and 85% of hours exceed
25 W/m2 -- not 0 W/m2 and 26.5%. **That file is a sanity check, not training data**, but the
constant belongs written down so it is not mis-read again.

---

## Why HRRR displaced CONUS404 as the forcing source

CONUS404 appears throughout PROJECT_BRIEF.md, and it keeps its role: it sets sweep ranges and
sampling density, and it is the 45-year climatology this site is characterised by. What it
cannot do is force a run.

| | CONUS404 | HRRR |
|---|---|---|
| horizontal | 4 km | **3 km** |
| atmospheric profiles | **none** -- `conus404_hourly`'s only 4-D variables are soil and snow; `PB`/`PHB` are static | **~50 hybrid levels** |
| surface fluxes | — | `SHTFL`/`LHTFL`, giving a **per-case Bowen ratio** |
| per-timestamp subsetting | — | Herbie |
| record | one configuration throughout WY1980-2024 | **v4 from 2020-12-02**, minor upgrades within the window |

**The trade-off, stated:** the corpus trades configuration homogeneity for resolution and
per-case realism. Pick a five-year span inside v4 to keep that trade small.

---

## Validation

**Offline, no GPU -- done.** `bin/test_sounding.py` runs stages 1-2 on four timestamps
spanning summer convective midday, a summer nocturnal stable layer, winter midday, and an
autumn morning transition, asserting: monotone `z`; physical theta; stratification the right
way up; the meridian convergence actually applied; the `z_i` diagnostics bracketing HPBL;
every one of the ten `.in` parameters inside FastEddy's *own* declared range; the stable
layer bases ordered; the base-state fit within 0.5 K rms **and** reproducing the sounding
below 1.5 km; the 5 s cadence landing on an integer step count; and `CFL_3d <= 1.35`.

**`bin/preflight.sh` -- extended.** It now covers `jobs/*.sh`, and every new entry point
must additionally answer `--help`: a clean parse says nothing about a `NameError` at module
scope or an argparse definition that raises, and both look exactly like a working script
until a campaign calls one.

**A 5-minute cold start per regime config — `bin/smoke_check.py`, three of four PASS**
(the fourth is the non-zero base angle, below). It cannot tell you a seed is converged;
nothing at 5 minutes can. It tells you the configuration is not broken, which is the only
question worth asking before committing 3.1 h of GPU per job:

| | `sbl` | `nbl-shallow` | `cbl-mid` |
|---|---|---|---|
| every field finite | ok | ok | ok |
| `k0/k1` (must be < 1; ~9 = `dt` past the accuracy boundary) | **0.124** | **0.132** | **0.144** |
| receptor on cell centre `k = 2` | 10.000011 m | 10.000011 m | 10.000011 m |
| **`z_i` vs the rung target** | **154 / 150 m** | **299 / 300 m** | 310 / 700 m (still growing) |
| log clean of `CORRUPTED` / `outside limits` | ok | ok | ok |

> **The strongest check is the base state, because it closes a loop nothing offline can.**
> `bin/test_sounding.py` verifies the fit arithmetically, but "my formula reproduces my
> formula" is not evidence that FastEddy read the six numbers, inverted `temp_grnd` into
> `theta_grnd` with **its** gas constants, and integrated the hydrostatic profile the way
> `hydro_core.c:1776-1810` says. The dump is: **max |theta_LES − theta_base| = 0.0001 K**
> over 50-60 levels, on both rungs without subsidence.
>
> The convective rung scored 0.1596 K and that is subsidence *working*: 25 m/h for 300 s is
> 2.08 m of descent, and through the 0.08 K/m capping inversion that is **0.167 K
> predicted** — 4% from observed. The tolerance is now the predicted warming, so every
> convective smoke run re-confirms Gate B7 rather than tripping over it.

The receptor sits at 10.000011 m rather than 10.000000: `bin/vgrid.py` solves it in fp64,
but FastEddy is hardwired fp32 and writes `zPos` as `NC_FLOAT`, so 1.1e-6 relative **is the
file's own precision**. A tolerance tighter than that fails a correct grid.

**Still to run (~30 min GPU, nothing over 1 h wall):**

1. ~~The non-zero base angle (30 deg)~~ **DONE -- PASS.** `seed_cbl-mid_a000` forces the
   geostrophic wind FROM **270.00 deg** aloft and `seed_cbl-mid_a030` FROM **240.00 deg**:
   exactly 30.00 deg apart aloft and 30.03 at the receptor. The rotated forcing reaches
   the solver. (The 10 m wind is backed only 0.6 deg after five simulated minutes -- the
   Ekman spiral needs the full spin-up, so this validates the FORCING, not the turning.)
2. One job-bundle round trip from an unrelated checkout, then Gate C2 on the returned
   artifact: restart from it with `Nt` = the restart step (trap 6 => zero timesteps, one
   dump) and diff byte-for-byte. **The path-discovery half is already done** --
   `jobs/run_seed.sh` runs correctly from a checkout at `/tmp/.../altroot/Flux` that knows
   nothing about `/home/atyagi/Flux`.
3. ~~One short **convective** B6~~ **DONE -- PASS** (`bin/b6_convective.sh`). PROJECT_BRIEF.md
   forbids inferring a regime from a gate that ran in another, and the seed library leans
   on the same rotation for convective rungs as for neutral ones.

   | | rot0 vs rot1 | its own block SE | ratio |
   |---|---|---|---|
   | resolved TKE | 0.447% | 5.758% | **0.08** |
   | `sigma_w^2` | 3.587% | 9.538% | **0.38** |
   | buoyancy flux `w'theta'` | 1.566% | 16.729% | **0.09** |
   | SGS TKE | 0.118% | 4.641% | **0.03** |

   `z_i` identical at 428 m; mean wind 2.35e-05, mean theta 1.05e-07.

   > The first version used a **fixed** 3e-2 on `sigma_w^2` and reported DIFFERS at
   > 3.587e-2. Loosening it would have been exactly the mistake PROJECT_BRIEF.md records. Scoring
   > against **how well one run agrees with itself** — 4x4 sub-blocks of 30 cells, near
   > independent because the domain is 1952 m and the convective integral scale is
   > `~z_i ~ 430 m` — showed the only offending level was `z = 2 m`, where `ww = 0.0013`
   > and the field's own block SE is 8.1%. Everywhere else the two rotations agree to
   > **0.001-0.015%**.
4. One end-to-end case: sounding -> forcing -> surface -> seed -> adjust -> window -> LPDM
   -> pair. Short window; the product is a well-formed pair, not a converged footprint.

Acceptance throughout: **assert on the artifact, never the exit status**
(`FASTEDDY_TRAPS.md` §12 -- analyses are piped into `grep`, so bash reports grep's status),
and `np.isfinite(...).all()` never `isnan().any()` (§1 -- `inf` is not `CORRUPTED`, and NaN
passes every `>` comparison).

---

## Deferred, with reasons

- **The 30-minute adjustment study.** How far adjustment carries each axis, which is what
  sets the allowed seed spacing. Deferred because it is a **coverage** question, not a
  correctness one -- inputs come from the LES window, so a wide gap costs diversity, not
  validity. `make_pair.py` already records achieved-minus-requested per case, so the study
  is mostly a matter of reading 1825 records once the corpus exists.
- **Direction-resolution recheck against the LES, not Kljun.** The three-base-angle choice
  was validated by reconstructing **Kljun's** array-share-vs-bearing curve -- and this
  project's entire premise is that the LES curve departs from Kljun's. If the real curve is
  sharper near N, 12 directions may under-resolve it. Once 12 directions of real cases
  exist, re-run the same spline reconstruction against the **measured LES** curve and
  confirm the error still sits under the sampling SE. Equally cheap once the pipeline is
  live.
- **Splitting `cbl-strong`.** If a 7th rung is ever wanted, the very-unstable class is the
  gap (`z/L` spans two decades there). Note `u*` is unidentifiable from `sigma_w` alone for
  19.7% of midday hours, so that tail is data-limited too.
- **Deep boundary layers.** `z_i > 976 m` is outside what a 1952 m box supports. Those
  hours are flagged `representable: false` and skipped. The exclusion is **biased, not
  neutral**: `z_i` and surface heat flux correlate at +0.43, so the excluded hours carry
  **1.51x** the heat flux and **1.58x** the `w*` of the representable ones. The fallback is
  `218^2 @ 16 m` (`L = 3488 m`, 3.2x cost), and that is a grid decision.
