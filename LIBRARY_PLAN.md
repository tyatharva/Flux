# The seed library and the sounding-forced corpus

> **THE LIBRARY MOVED TO 30 m / 24 m, 2026-08-29 — `jobs24/`, not `jobs/`.** The 15 (later
> 30) seeds in `jobs/` were built for a 10 m receptor on a 1952 m box and none of them
> transfers: different `d_zeta`, different `dt`, different domain, different surface `z0`.
> `jobs24/` holds the 30 replacements (5 rungs x 6 base angles), generated with
>
>     bin/make_seed_jobs.py --outdir jobs24 --template runs/g24_base/base.in \
>       --dx 24 --nz 122 --zceiling 3000 --deform 0.346601 \
>       --grid data/grid24_raised --receptor 30 --receptor-k 3
>
> **Four things changed with it, and each was a latent trap in the generator.**
>
> 1. **`surflayer_z0` is now read off the grid** (`--grid`), not a constant. It was
>    hardcoded at 0.1435 m — the 16 m map's geometric mean. At 24 m over a 2928 m box the
>    same WorldCover map gives **0.0832 m**, 42% smaller, because the box reaches the lake
>    and the coarser cells average tree and crop together. Carrying the constant would have
>    spun every seed up over the wrong surface, silently.
> 2. **The gate's receptor is in the manifest** (`gate: {zm, k}`) and `jobs/run_seed.sh`
>    passes it. `bin/seed_stationarity.py` defaults to `--zm 10 --k 2`; a 24 m library
>    scored with the defaults would evaluate Kljun's `x_peak` and `x90` at the wrong height
>    AND read `sigma_w` off level k=2 (21.4 m) instead of k=3, and every number would still
>    print.
> 3. **The budget is measured, not assumed** — see below.
> 4. **Cost per seed falls from ~2.9 h wall to ~87 min** (0.481 GPU-h/sim-h measured), so
>    30 seeds is **~43 GPU-h** rather than ~87.
>
> ## The 3.0-hour class is now a CEILING, and the stop is measured
>
> `jobs/seed_watch.sh` scores the trailing window every 30 simulated minutes and stops the
> run as soon as the **oscillation-immune** limits are in band — `U/u*`, `sigma_v/u*`,
> `sigma_w/u*`, Kljun `x_peak`, Kljun `x90`. `TKE_BL/u*^2` and `z_i` are deliberately NOT
> in the criterion: they decorrelate on the eddy turnover rather than on the 300 s dump
> interval, `n_eff` saturates at 3-5 at every window width from 1.0 h to 2.5 h, and
> requiring them would mean never stopping early while misreporting why. A DRIFTING verdict
> on any limit still blocks the stop — unestablished stationarity is not stationarity, but
> neither is it drift. **A seed that has not entered band by 3.0 simulated hours stops
> there and that IS the result: no extension, no respec.**
>
> **Neutral rungs get Steinfeld's accelerator.** 3000 s at `surflayer_wth = +0.05 K m/s`,
> then the open-ended run at the rung's own flux, restarting from the burn-in dump with
> `htFlux` **zeroed in the FILE** (`bin/zero_htflux.py`, which re-reads to confirm). That
> restart is the only one in a seed and it is the dangerous kind: `htFlux` is IO-registered,
> so the main invocation would otherwise inherit +0.05 whatever its `.in` says
> (`FASTEDDY_TRAPS.md` 17). The existing per-run read-back assertion is the second lock.
> Neutral is the regime the accelerator is for: `h/u*` is ~1500 s there against `T* ~ 350 s`
> convectively, so it is the slowest to organise a perturbation field into turbulence.
> **The no-accelerator control was cut** — if the accelerated seed passes, it is not needed.
>
> ## The corpus band moved with the box
>
> `z_i` **300-1250 m** (was 100-976): the floor is `10 z_m` and tracks the receptor, the
> ceiling is the LOWER of the width constraint `L >= 2 z_i` (1464 m) and the domain-height
> constraint (~1250 m, half the clean column under the 500 m sponge). Measured on the same
> code: day coverage **75.0% -> 80.4%**, **1370 -> 1469 cases** over five years. The deep
> exclusion is barely less biased — rejected unstable hours carry **2.33x** the mean surface
> heat flux of accepted ones against 2.44x before — so the widening buys cases, not fairness.
>
> `bin/sounding_to_forcing.py` now reads the receptor height off `<grid>/meta.npy` and the
> domain `z0` off `<grid>/z0m.npy` instead of hardcoding 10.0 and 0.1435, and takes
> `--zi-max-abs` for the height ceiling.

---


> **Status, 2026-08-25.** The pipeline is **built and validated end to end** on branch
> `library-states`; `main` is untouched. All eight stages run, `bin/make_seed_jobs.py`
> generates the 15 portable jobs, and every gate that could be run without spinning the
> library has been run and passed — stages 1-2 across four regimes (70/70), a cold start
> per regime config, the non-zero base angle, **Gate B6 convectively**, a job round trip
> from an unrelated checkout, and **Gate C2 bit-for-bit on the returned artifact**.
>
> **What remains is GPU time**: spin the 15 seeds (~43 GPU-h, shippable to rented
> machines — see `jobs/README.md`), then run the corpus.

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

**RE-COSTED 2026-08-26 on the measured rate and the current case count.** Two things moved
since the table below was first written: the stable regime is excluded, so 75.0% of days
carry a case rather than 100%, and the s/step is measured rather than assumed
(13.94 ms/step at a 300 s cadence, `results/sblweak_seed_report.txt`; ~15.5 ms/step with a
5 s window cadence, which is what a target case runs at).

| | sim-h each | wall each | GPU-h |
|---|---|---|---|
| **1370 cases** (75.0% of 1826 days) x 4200 s | 1.167 | 74.2 min | **1695** |
| **the seed library, 15 x 3.0 sim-h** | 3.000 | 2.86 h | **43** |
| **total** | | | **1738** |
| the same corpus cold-started (3 h spin-up per case) | 4.167 | | **~5420** |

**43 GPU-h buys back about 3700.** The corpus is the real cost; the library is rounding
error beside it. Each case is now ONE FastEddy invocation of 287,280 steps — 841 dumps at a 5 s cadence
(measured 18.4 MB lean, 73.3 MB full), **15.3 GB peak**, of which the 360 adjustment dumps
(6.5 GB) are deleted and the 481 window dumps (8.7 GB) survive. Three dumps are full-form
(steps 0, 123120, 246240) because `ioLPDMfullFrq = SKIP_NT`: under `ioLPDMmode` the static
geometry is written to the first file of the run ONLY, and that file is an adjustment dump
that gets deleted — so the first SURVIVING dump has to be full-form or nothing downstream
can build a field cache.

> **1825 is the number of DAYS. Superseded twice: `z_i` outside `100-976 m` was already
> refused, and stable hours (`z/L > 0`) are now refused too — 75.0% of days carry a case,
> about 1370. The 70.6% / 1289 / 1459 GPU-h figures below predate the stability screen.** Days whose `z_i` falls outside `100-976 m` are
> refused rather than run and mis-labelled. `bin/corpus_coverage.py`, on a 51-case weekly
> sample across 2023 walking the full diurnal cycle (`results/corpus_coverage.txt`):

| | |
|---|---|
| accepted | **36/51 = 70.6%** |
| rejected, **too deep** (`z_i > 976 m`) | 13.7% |
| rejected, **too shallow** (`z_i < 100 m`) | 15.7% |

Both bounds bite, at opposite ends of the year and of the day. Acceptance by month runs
**100%** in January, March, September and October against **25%** in June and **20%** in
August; by hour it is **100%** at 02-03 UTC (20-21 local) and **50%** at 06-07 UTC (dawn,
where the boundary layer is too shallow) and at 16-17 UTC (mid-afternoon, where it is too
deep).

> **And the deep exclusion is not a neutral trim — now measured on HRRR rather than
> inferred from CONUS404.** The days rejected as too deep carry **3.56x** the virtual heat
> flux of the accepted set (0.0859 against 0.0241 K m/s), and the rank correlation of `z_i`
> with flux over the sample is **+0.492** — independently reproducing the **+0.43**
> PROJECT_BRIEF.md measured from CONUS404, on a different dataset and a different diagnostic. **The
> corpus is thinnest exactly where the array's flux enhancement is largest.** State it
> wherever the corpus is described.
>
> If ~1289 is not enough, the levers are a longer span (7 years gives ~1800) or more than
> one hour per accepted day — the latter at the cost of within-day correlation, which is
> the reason one-per-day was chosen. That is a design call, not a pipeline one.

A case's window is 42 min of wall clock and its adjustment 31, so **both fit inside the
1-hour cap as single segments** -- which is what makes 1825 of them schedulable at all.

---

## The library: 5 rungs x 3 base angles = 15 states

> **The `sbl` rung is DELETED.** A cold-started stable boundary layer does not survive
> `Delta = 16 m` at this site -- two seeds were built and both collapsed, and the cause is
> resolution, measured (`L_O/Delta = 3.57` at the receptor against a decade requirement;
> GABLS1 runs the regime at `dx = 6.25 m`). Full evidence: `STABLE_REGIME_RESULT.md`. The
> table below keeps the row so the reasoning is legible; it is not built and not selectable.

### Why these axes, and no others

Sized by **what 30 minutes cannot adjust**, which is the only criterion that matters for a
state whose entire purpose is to be adjusted away. Every number below is measured on this
project's own runs:

| quantity | closed in 30 min? | evidence | axis? |
|---|---|---|---|
| **direction** | no, ~2.7 deg | -5.4 deg/h backing, `g16_spin` | **yes -- 3 base angles -> 12 headings** |
| **z_i** | no, ~+40 m | +79 m/h entrainment, `g16_cbl_shallow` | **yes -- 5 real depths** |
| **stability regime** | no | a CBL needs ~8 `T*` ~ 1.2 h to turn over | **yes -- in the rungs** |
| u\* / wind speed | partly | the surface layer is ~0.1 `z_i` deep, so it re-equilibrates in ~2 min at a 10 m receptor | no |
| fine `z/L` | yes | the surface flux is prescribed and the surface layer follows | no |

### The rungs are coupled, not a product

A 150 m stable boundary layer cannot carry a 12 m/s geostrophic wind -- shear that strong
destroys the stratification that defines it. So `G` belongs to the rung. The rungs walk the
site's real joint `(z_i, flux, wind)` distribution as CONUS404 measures it at this tower
(`z_i` p25/p50/p75 = 267/493/835 m; `w'th'` p25/p50/p75 = -0.006/+0.015/+0.076 K m/s;
`U(30 m)` p25/p50/p75 = 3.9/5.2/6.8 m/s):

| rung | regime | `z_i` target | `w'th_v'` | `G` | how `z_i` is held |
|---|---|---|---|---|---|
| ~~`sbl`~~ | ~~stable~~ | ~~150 m~~ | ~~**-0.020**~~ | ~~6 m/s~~ | **DELETED -- collapses at this grid, see above** |
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

> **THE 30-DEGREE SPACING DOES NOT DELIVER +/-15 DEG, MEASURED 2026-08-27.** The section
> below reasons from where the seeds are PLACED. Both corpus cases that have run show the
> direction gap WIDENING through the adjustment rather than closing -- `case_2023031014`
> 11.3 -> 21.8 deg, `e2e_20230118` 14.1 -> 36.0 -- so the worst-case gap on the DOMINANT
> SKILL AXIS is 25-35 deg, not 15. `bin/pick_seed.py` now projects the seed's own
> freeze-time drift forward, which removes the MEAN of that excursion and leaves its
> SCATTER; it is a partial fix and cannot be the whole one.
>
> **Nothing yet predicts the drift rate.** n = 2 seeds with a measured rate (-5.63 and
> -7.79 deg/h) and n = 2 cases with a measured widening -- with that sample any predictor
> fits exactly and its correlation is an artifact of the sample size, so no fit is
> reported. `bin/direction_drift.py` lays out `u*`, `z_i`, `h/u*` and the drift-per-turnover
> beside the rate so the answer arrives on its own as seeds accumulate.
>
> **PROPOSED, NOT APPLIED: 6 base angles at 15 deg** = 24 library headings, worst-case
> 7.5 deg before drift and ~15 after -- which is what 3 angles were believed to give. Cost
> 30 seeds instead of 15, ~86 GPU-h against ~43, i.e. **2.5% of the ~1700 GPU-h corpus** to
> fix the axis the emulator is judged on. Denser angles are the honest fix; smarter
> projection is not.

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

3.0 simulated hours as **ONE continuous invocation** of 738,720 steps, ~2.9 h wall.
Chaining was retired 2026-08-26 (`bin/test_unchained.py`: chained vs unchained is
0.89-1.08x the run-to-run reproducibility floor, so nothing is lost by removing it), and
with it the 1-hour cap. The only restart left in the project is seed -> target.
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
- **idempotent, NOT resumable** — a resume is a restart, and restarts within a run are
  what was removed. Re-invoking a COMPLETE job is a no-op; a kill costs the whole run.
  (retired) the chain used to restart from the newest dump on disk, so a kill cost at most
  one segment.

> **Bitwise reproducibility will not hold across different physical GPUs.** Stated, not
> fixed. FastEddy is already non-reproducible run-to-run on *one* GPU -- ~1e-4 relative in
> velocity and ~7e-4 K in theta after 200 steps. Seeds are turbulence realisations, so this
> costs nothing; it is recorded so nobody later diffs two seeds expecting equality.

---

## The pipeline, stage by stage

**Per case**, driven end to end by `bin/run_corpus_case.sh` (and per day by
`bin/run_corpus.sh`):

| # | stage | file | status |
|---|---|---|---|
| 1 | HRRR pseudo-sounding at the tower | `bin/hrrr_sounding.py` | **built, validated** |
| 2 | sounding -> FastEddy `.in` parameters | `bin/sounding_to_forcing.py` | **built, validated** |
| 3 | this case's surface heat-flux map | `bin/case_surface.py` | **built, validated** |
| 4 | which seed, and which 90-degree rotation | `bin/pick_seed.py` | **built, validated** |
| 5 | rotate the flow, inject the static surface | `bin/prep_restart.py` | existing |
| 6a | ~30 min adjustment under this case's forcing | — | existing |
| 6b | the (30 min + `t_back`) sampling window | `bin/run_window.sh` | existing |
| 7 | backward LPDM -> the 122 x 122 footprint | `bin/stage5_footprint.py`, `lpdm/` | existing |
| 8 | assemble one (input, target) record | `bin/make_pair.py` | **built, validated** |

**The library**, built once and shipped to rented GPUs:

| | file | status |
|---|---|---|
| generate the 18 portable jobs | `bin/make_seed_jobs.py` | **built** |
| run one seed, gate it, return it | `jobs/run_seed.sh` | **built, round trip PASS** |
| the stationarity gate, portable | `bin/seed_stationarity.py` | **built** |

**Gates and validation**

| | file | status |
|---|---|---|
| stages 1-2 across four regimes, offline | `bin/test_sounding.py` | **PASS 70/70** |
| a short cold start per regime config | `bin/smoke_check.py` | **PASS x4** |
| Gate B6, re-run convectively | `bin/b6_convective.sh` | **PASS** |
| Gate C2, on a returned artifact | `bin/c2_restart_check.sh` | **PASS bit-for-bit** |
| how many days the domain accepts | `bin/corpus_coverage.py` | **built** |

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

**`--available-only`, and why `bin/run_corpus_case.sh` passes it by default.** The driver
is about to RESTART from the chosen seed, so ranking a seed with no returned artifact is
never right there: the pick names a file that does not exist and the case stops. The
second reason matters more while the library is being built -- an unbuilt seed's heading is
an **estimate** (its geostrophic angle minus a nominal Ekman backing) while a spun one
reports what it **achieved**, so ranking the two together compares a guess against a
measurement. With a complete library the flag is a no-op; `SEED_ANY=1` restores the
full-library ranking for planning.

**The geostrophic SPEED is reported and never costed, and the report now says what that
claim rests on.** Everything Kljun sees is a ratio: `U(z_m)` and `u*` scale together, so
`Pi_4 = U/u*` is nearly invariant under a speed mismatch -- measured on `g16_spin`, `u*`
moved 18% across five windows while `U/u*` moved 0.6%. What 30 minutes does **not** do is
close the gap: the mean flow accelerates at `f (G_case - G_seed)`, so it closes
`f dt = 9.94e-5 x 1800 = 17.9%` of it whatever its size, and the case samples a flow
somewhere between the two forcings. That is sound for a modest gap and an extrapolation for
a large one, so `pick_seed.py` prints the ratio and **warns past a factor of two**.

> Measured while choosing the first target case: over 118 screened neutral-regime candidate
> hours the implied `G = U(10)/0.55` runs p25 2.9, **median 6.3**, p75 9.3 m/s, so the
> `nbl-shallow` rung's `G = 8.0` sits between the median and p75 and is well specified. The
> array-loaded near-neutral hours are a WINDY subset of that population -- filtering on
> `u* >= 0.30` and an N/S sector returned cases at `G = 12-22 m/s` -- which is a property of
> the selection, not of the rung.

**A seed that failed its gate is not a seed, and the verdict is not in the manifest.**
`jobs/run_seed.sh` stamps `achieved` into the return manifest as its LAST step, so a job
that died after the gate and before the stamp leaves a manifest with no verdict at all.
`pick_seed.py` reads `return/stationarity.json` directly; a return directory with neither a
verdict nor a restart is reported as an **unfinished job**, not as an unbuilt one. See
`FASTEDDY_TRAPS.md` §18d for the live instance.

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

**A 5-minute cold start per regime config — `bin/smoke_check.py`, all four PASS.** It
cannot tell you a seed is converged; nothing at 5 minutes can. It tells you the
configuration is not broken, which is the only question worth asking before committing
3.1 h of GPU per job:

| | `sbl` | `nbl-shallow` | `cbl-mid` | `cbl-mid_a030` |
|---|---|---|---|---|
| every field finite | ok | ok | ok | ok |
| `k0/k1` (must be < 1; ~9 = `dt` past the accuracy boundary) | **0.124** | **0.132** | **0.144** | **0.150** |
| receptor on cell centre `k = 2` | 10.000011 m | 10.000011 m | 10.000011 m | 10.000011 m |
| **`z_i` vs the rung target** | **154 / 150 m** | **299 / 300 m** | 310 / 700 m (still growing) | 322 / 700 m |
| log clean of `CORRUPTED` / `outside limits` | ok | ok | ok | ok |

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

**The GPU validations — all four run, all four pass (~25 min of GPU in total):**

1. ~~The non-zero base angle (30 deg)~~ **DONE -- PASS.** `seed_cbl-mid_a000` forces the
   geostrophic wind FROM **270.00 deg** aloft and `seed_cbl-mid_a030` FROM **240.00 deg**:
   exactly 30.00 deg apart aloft and 30.03 at the receptor. The rotated forcing reaches
   the solver. (The 10 m wind is backed only 0.6 deg after five simulated minutes -- the
   Ekman spiral needs the full spin-up, so this validates the FORCING, not the turning.)
2. ~~One job-bundle round trip from an unrelated checkout~~ **DONE -- PASS.**
   `jobs/run_seed.sh` ran a seed from a checkout at `/tmp/.../altroot/Flux` that knows
   nothing about `/home/atyagi/Flux`, returned all six artifacts (70 MB), and correctly
   exited 1 because a five-minute cold start fails the stationarity gate -- which is the
   gate working, not the job failing. **Gate C2 on the returned file: PASS, bit-for-bit,
   0 of 23 variables differ.**

   > The first run of that gate reported **10 of 23 differing, `u` by 2.65 m/s** -- not
   > roundoff, a full integration. The pipeline was fine; the *test* combined two traps
   > backwards. **Trap 4**: the restart step is parsed from the FILENAME
   > (`time_integration.c:104`), so naming the returned dump `FE_RST.0` reset the counter
   > to zero. **Trap 6**: `Nt` is an ABSOLUTE target step, so `Nt = 20520` from a counter
   > at 0 ran 20520 real steps rather than the intended zero. `bin/c2_restart_check.sh`
   > now takes the step as an explicit argument and names the file for it.
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

4. ~~One end-to-end case~~ **DONE -- PASS, exit 0.** 2023-01-18 18Z, all eight stages,
   13 min of GPU: sounding -> forcing -> per-case surface -> seed -> adjust -> window ->
   LPDM -> pair. `k0/k1` 0.515 on the adjustment and 0.483 on the window; the window wrote
   121 dumps (2.2 GB) and was deleted afterwards, as designed.

   The record it produced:

   | | |
   |---|---|
   | `run_id` / `split_key` | `e2e_20230118` |
   | target | 122 x 122, integral **0.858** (Kljun on the same box: 0.955) |
   | inputs | `u_mean` 5.462, `u*` 0.5328, `sigma_v` 0.9446, `h` 399.5 m, `L` -71.04, `wdir` 59.67, `1/L` -0.0141 |
   | receptor | 8.500 m above the raised surface = 10.000 m above bare ground |
   | closure | `sgs_most`, weighted, `eps`-consistent -- the sixth pass's production closure |
   | source area | **solar array 55.4%**, cropland 35.3%, tree 7.1%, water 0.0% |
   | seed | `seed_cbl-mid_a030` rot 2, 14.1 deg away |

   A north-easterly footprint putting more than half its flux on the array is what the
   geometry says it should be — the array runs 250 m north of the tower.

   **Deliberately not converged, and the numbers say so.** The seed was a five-minute cold
   start and the adjustment 150 s rather than 1800. Achieved minus requested: `z_i`
   **-184 m**, direction **+36.0 deg** — the achieved direction is essentially the *seed's*,
   because 150 s closes about 0.2 deg of a -5.4 deg/h backing. That is the axis argument
   arriving as a measurement: **direction is a seed axis precisely because adjustment does
   not close it**, which is why there are 12 of them.

Acceptance throughout: **assert on the artifact, never the exit status**
(`FASTEDDY_TRAPS.md` §12 -- analyses are piped into `grep`, so bash reports grep's status),
and `np.isfinite(...).all()` never `isnan().any()` (§1 -- `inf` is not `CORRUPTED`, and NaN
passes every `>` comparison).

---

## Time selection: enumerate, then fill the thinnest cell

**MEASURED, and it overturns the premise it was built on.** The corpus took one midday
per day and 70.6% of days survived. Two changes were proposed: raise the `z_i` cap
976 -> 1200 m, and enumerate all hours instead of retrying. Ablated on 92 days of 2023
(2208 candidate hours, `results/candidates.tsv`):

| | `z_i <= 976 m` | `z_i <= 1200 m` |
|---|---|---|
| pick 18Z only | 65.2% | 70.7% |
| **enumerate + stationarity screen** | **91.3%** | **92.4%** |
| enumerate, no stationarity screen | 100% | 100% |

> **The enumeration is worth +26 points. The cap raise is worth +1.1.** Once every hour is
> a candidate, *every one of the 92 days had some hour with an acceptable `z_i`* — the
> binding constraint stops being the domain and becomes stationarity. So the cap raise
> buys almost nothing while stepping outside what Phase E measured, and **the corpus keeps
> `R = 2` (976 m), inside licensed territory, and still gets 91.3%.**
>
> That is the opposite of the expected answer: the "clean lever" was supposed to be the
> cap and the compromise was supposed to be time-shifting. Measured, time-shifting does
> essentially all the work and the cap does essentially none.

**dz_i/dt is screened separately from `z_i`, and that is the whole design.** Widening the
acceptable band pushes selection towards morning and evening, because those are the hours
when a day whose midday is too deep still has an acceptable depth — and they are exactly
when `z_i` changes fastest. A naive widening trades a domain violation for a stationarity
violation and reports neither. Measured median `|dz_i/dt|`: **15.9 %/h at 09 CST**
(morning growth) and **14.8 %/h at 18 CST** (evening collapse) against **6-7 %/h**
overnight. The threshold is 15 %/h — comparable to the LES's own entrainment drift over a
window (~8 %/h at 500 m), so it does not demand more stationarity of the forcing than the
simulation itself delivers. The result is insensitive to it: 89.1% at 8 %/h, 91.3% at 15,
93.5% at 30.

**All seven days with no case failed the stationarity screen, not the depth screen** —
"`z_i` acceptable somewhere, but never stationary enough". Not one of the 92 days lacked an
acceptable depth entirely.

**Scaled: ~1670 cases from 1826 days**, against ~1289 for one-midday-per-day.

At most one case per day: two times from one day share the synoptic state, the soil
moisture and the morning's history, so they are not independent units and `run_id` would
stop being the right split key. **Selection is deterministic** — days in date order, ties
broken on (fewest in cell, most stationary, earliest hour) — so the same candidate table
always yields the same corpus.

**Coverage is what the surplus is spent on.** With ~24x more candidates than days, validity
stops binding and the greedy fills the thinnest `direction x stability x z_i` cell. The
rose is S/SW/W-heavy (S 16.0%, W 14.4% against N 10.6%, NE 10.2%), and direction is the
dominant skill axis, so a northerly hour is worth more than a southerly one even though
both are equally valid. On the 85-case sample the selected direction counts run N 7,
NNE 5, ENE 8, E 5, ESE 5, SSE 7, S 11, SSW 10, WSW 6, W 8, WNW 5, NNW 8 — still
S/SSW-leaning but far flatter than the rose, and the time-of-day histogram spans all 24
hours rather than clustering at midday.

**HRRR analyses are hourly, so a day holds 24 candidates and not 48.** The tower's
averaging periods are half-hourly, but the forcing comes from the analysis whose valid time
equals the footprint timestamp, and a :30 timestamp has no analysis behind it. HRRR's
`subh` product is 15-minute FORECAST output; the `nat` hybrid-level profile the sounding
needs is hourly-only regardless. A property of the data source, not a choice.

**Cost of the full enumeration**: 4 screening fields x 24 h x 1826 days at ~7.5 MB per hour
is **~330 GB over ~20 h**, run once, GRIB deleted as it goes. `--stride` samples it. The
92-day sample here took 69 minutes.

---

## The stable regime does not survive this grid — measured, 2026-08-25

**Gate D1 in stable conditions cannot be run at `dx = 16 m`, because there is no stationary
stable state to run it on.** That is a stronger and more useful result than the gate verdict
would have been, and it is a property of the grid rather than of the closure.

A stable seed was built at GABLS1's own regime — `G = 8 m/s`, `w'th' = -0.012 K m/s`,
solved so that `z_i = 0.4 sqrt(u* L/f)` lands on the 150 m target — and run for 3.0
simulated hours with a neutral warm-up first. It was **healthy for 1.75 h**:

| | t = 1.50 h (healthy) | t = 3.00 h (collapsed) |
|---|---|---|
| `u*` | 0.2361 m/s | **0.0984** |
| `z/L` at the receptor | +0.123 | **+2.67** (peak) |
| `Ri_g` through the layer | 0.03-0.05 | — |
| receptor direction | 228.8 deg (**11.2 deg** of Ekman backing from a 240 deg forcing) | 200.2 deg (39.8 deg) |

Then it collapsed. All seven stationarity limits drift; **`x_peak` binds last at 6989% of
its limit**, `U/u*` at +68 %/h, `u*` at -75 %/h.

**The cause is resolution, and it is measured rather than inferred.** At the *healthy*
dump — an hour before anything looked wrong — the Ozmidov scale `L_O = sqrt(eps/N^3)`, the
largest eddy stratification permits to overturn. Measured with `bin/ozmidov.py` using
FastEddy's own `eps` and mixing length (`results/ozmidov_regimes.txt`):

| regime | `L_O/Delta` at the 10 m receptor | surface layer (min-median) | resolved `sigma_w^2` at the receptor |
|---|---|---|---|
| **stable** (GABLS1 regime) | **3.57** | 2.41 - 5.21 | **0.2%** |
| neutral | 318.07 | 43.2 - 92.7 | 2.7% |
| convective | *unstratified — no Ozmidov constraint at all* | — | 12.1% |

A factor of **89** between stable and neutral at the same receptor on the same grid. The
model is not simulating stable turbulence at the receptor; it is running a sub-grid
closure. **GABLS1 uses `dx = 6.25 m`** — 2.6x finer, and **17x the cells** for this domain.

> **CORRECTED 2026-08-25.** This paragraph previously said "1.0-3.2 x `Delta` anywhere in
> the layer" and "0.6% at 6 m and 4.0% at 14 m". Both were remembered from an ad-hoc
> calculation; the script gives 2.41-5.21 through the surface layer, 3.57 at the receptor,
> and 0.2% resolved there. **And the column median would have hidden the result** —
> `L_O/Delta` rises steeply with height as `N` falls, so the median over the whole
> stratified column reads **8.97**, far healthier than the surface layer actually is.
> Score at the receptor, where the footprint is made.

> **`k0/k1` was 0.442 for the whole run**, including after the collapse. The standing
> accuracy check passes on a boundary layer that has died, because it is a `dt` check and
> not a physics check. `u*`, `z/L` and the mean wind profile are what caught it.

**This is a corpus-scope decision, not something to tune around.** Stable hours are ~29% of
this site's QC'd record and **44% of a coverage-balanced selection** (35 stable + 2 very
stable of 85 in the sample). Three options, none of them free:

1. **Exclude stable cases.** The corpus becomes convective-and-neutral, which is also where
   the array's signal lives (a stable case has no thermal array contrast at all). Coverage
   falls to ~56% of a balanced selection, and the emulator is undefined at night.
2. **A finer grid for stable cases only.** `dx = 6.25 m` over the same 1952 m box is
   `312^2`, about **16x the cells** — and it breaks the one-grid design the whole seed
   library rests on.
3. **Restrict to weakly stable cases only.** The site's typical stable night is
   `z/L ~ 0.03-0.10` (`U(10) ~ 3.3 m/s`, `w'th' ~ -0.006`), which is 2-5x weaker than what
   collapsed here. That may be resolvable, and it is one run to find out — but it narrows
   "stable" to "near-neutral" and should be called that.

**Not attempted: weakening the cooling until it passes.** Three spec changes were already
made to this rung on physical grounds, and a fourth chosen to obtain a pass would be
tuning, not measurement.

### DECIDED 2026-08-25: option 3, and the bound is measured rather than asserted

**How much of the site is weakly stable was a number, not a judgement**, and it was
measured over three independent sources with three different determinations of `u*`
(`bin/stable_fraction.py`, `results/stable_fraction.txt`):

| source | median stable `z/L` | runnable (`z/L <= 0.10`) share of stable hours | cost, share of whole QC'd record |
|---|---|---|---|
| the tower's own `H` + `sigma_w`, 1 y | 0.056 | **65.8%** | 15.2% |
| HRRR, the corpus's own forcing | 0.063 | **64.9%** | 15.4% |
| CONUS404, 45 y, `u*` direct | 0.071 | **60.7%** | 14.0% |

So **about two thirds of the site's stable hours survive the restriction**, and the
exclusion costs 14-15% of the record. That is a bounded limitation, not a lost regime —
and the "44% of a coverage-balanced selection" figure above was measured BEFORE the screen
existed, on a selector free to pick hours the grid cannot run.

**The knob is more wind, not less cooling**, and the reason is the failure mode: `z/L`
falls as `u*^-3` while `eps` rises as `u*^3`, so more wind buys weaker stratification AND a
larger Ozmidov scale at once, where a weaker flux buys only the first. It also deepens the
layer (`z_i ~ u*^2`), and **depth relative to the filter width is what the grid has to
buy**:

| | `z_i` | `Delta` | `z_i/Delta` |
|---|---|---|---|
| GABLS1 | ~180 m | 6.25 m | **28.8** |
| the collapsed rung | 150 m | 10.09 m | **14.9** |
| **`sbl-weak`** | **~280 m** | 10.09 m | **27.7** |

`sbl` therefore became **`sbl-weak`**: `G = 10 m/s`, `w'th' = -0.012` (unchanged), `z_i`
target 280 m, one neutral warm-up segment — placed at the site's MEDIAN stable hour rather
than at the edge of the band, so a pass would license the band and a failure would be
unambiguous.

### It failed. Option 1 by elimination: stable is EXCLUDED — measured 2026-08-26

`seed_sbl-weak_a030`, 3.0 simulated hours, fresh, at `z/L = 0.044`:

| t (h) | `u*` | `z/L` | backing | `\|U-G\|` aloft | `Ri_g`@20 m | `dθ/dz`@2 m | `zTKE95` |
|---|---|---|---|---|---|---|---|
| 0.75 | 0.2794 | 0.074 | 8° | 0.0001 | −0.000 | −0.0 | **92 m** |
| 1.50 | 0.3334 | **0.044** | 7° | 0.344 | 0.012 | 7.1 | 559 m |
| 3.00 | **0.1848** | 0.253 | 21° | 0.467 | 0.043 | 12.4 | **1825 m** |

`u*` at **40% of its own peak** and still falling at −40 %/h; resolved TKE at **5%** of
its peak; all seven limits fail. **Halving `z/L` bought a slightly slower death.**

**Not the cold-start fault.** `Ri_g` peaked at 0.043 against a critical 0.25, Ekman backing
was normal and increasing, the inversion was an ordinary 12 K/km, and the flow aloft
*departed* from geostrophic. The surface layer was healthy; the resolved energy drained
upward. `bin/sbl_diagnose.py` scores both signatures and its control on the retired seed
reports *starved **and** decoupled*, so it discriminates rather than labelling everything
the same.

**Consequences, all live in the code:**

- `bin/select_times.py --max-zol` defaults to **0.0**. No stable cases.
- The `sbl` rung is deleted: **5 rungs × 3 angles = 15 seeds, ~43 GPU-h** measured.
- Corpus: **1370 cases from 1826 days**, 75.0% of days (was 80.4% with weak stable).
- Convective share of selected cases rises 40.5% → **65.2%**.

**State the reach limitation, not the count.** 44% of QC'd hours are stable and the
emulator is undefined in all of them. Day coverage barely moves (−5.4 points) because a
case is drawn from any acceptable hour of the day, and 26% of retained cases are still
outside 06–18 LST — but those are near-neutral or weakly unstable nights and **must not be
quoted as stable coverage**.

Full evidence: **`STABLE_REGIME_RESULT.md`**.

---

## The one gate the corpus needs that has never been run

> **SUPERSEDED 2026-08-26 by the stable exclusion, and kept because the ARGUMENT is not
> superseded.** `bin/select_times.py --max-zol` defaults to 0.0 and the `sbl` rung is
> deleted, so **the corpus contains no stable cases** and there is no stable closure left
> to validate. The gate below is therefore moot *for this corpus*. What survives is the
> reasoning: a regime the gate has not run in is no evidence at all, and if stable is ever
> re-admitted -- at a finer grid, which is what `STABLE_REGIME_RESULT.md` says it would
> take -- this gate is the price of admission and must be run before any stable case is
> trusted.

**Gate D1 (well-mixed) has never been run in STABLE conditions.** Checked against
`bin/run_pass6.sh`: the battery ran on `g16_flat` (neutral) and `g16_flatcbl` (convective),
and on nothing else. The sixth pass's table has a neutral row and a convective row and no
third one.

That was fine for a corpus of eight convective and neutral directions. It is not fine for a
corpus that walks the whole diurnal cycle — this site is **stable in ~29% of QC'd hours**
(PROJECT_BRIEF.md's climatology: 20.4% stable, 8.8% very stable), the `sbl` rung exists precisely
to serve them, and PROJECT_BRIEF.md's own convention says to **treat a regime the gate has not run
in as no evidence at all**. It is the same mistake the fifth pass made when it recorded the
convective closure as "inherited" from a neutral PASS.

Stable is also the hardest of the three, for three separate reasons:

- **`z/Delta` is worst there.** The energy-containing eddy scale shrinks with stability
  while `Delta` does not, and the receptor already sits at `z/Delta = 0.99`.
- **The MOST anchor is a different function.** The floor is anchored to
  `sigma_w/u* = 1.25 phi_w`, and `phi_w` in stable stratification is not the neutral or
  convective branch — so the floor's factor at the receptor is a third number, not the
  1.000 (neutral) or 1.59 (convective) already measured.
- **A stable LES can laminarise.** At `z_i ~ 150 m` the whole boundary layer is ~38 model
  levels and the turbulence is weak and intermittent; there is no guarantee the state stays
  turbulent at all, which is a question the stationarity gate can answer but has not been
  asked.

**This is cheap to close** — one `sbl` window and one `stage4_wellmixed.py --sgs-most` run,
the same battery `bin/run_pass6.sh` already applies to the other two regimes — and it
should be closed before any stable corpus case is trusted. It is listed here rather than in
Deferred because it is a **gate**, not a refinement.

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
- **Which `z_i` the representability filter should use.** The corpus filters on HRRR's
  **HPBL**, a TKE-threshold depth from HRRR's own PBL scheme. It is the right choice for
  consistency — the same diagnostic in all ~1825 cases — and it is the closest analogue of
  what the LES itself reports (`window_stats` takes the highest level with resolved TKE
  above 5% of its maximum). But it runs systematically **higher** than a theta-based depth:
  1648 m against a 1244 m parcel top on 2023-07-15 19Z, a factor of 1.33. Since the filter
  is a hard cut at 976 m, that factor decides how many summer days survive. CONUS404's
  `PBLH` is the same class of diagnostic, so PROJECT_BRIEF.md's 60.9% and this are comparable —
  but the sensitivity is worth measuring once rather than assuming.

- **Splitting `cbl-strong`.** If a 7th rung is ever wanted, the very-unstable class is the
  gap (`z/L` spans two decades there). Note `u*` is unidentifiable from `sigma_w` alone for
  19.7% of midday hours, so that tail is data-limited too.
- **Deep boundary layers.** `z_i > 976 m` is outside what a 1952 m box supports. Those
  hours are flagged `representable: false` and skipped. The exclusion is **biased, not
  neutral**: `z_i` and surface heat flux correlate at +0.43, so the excluded hours carry
  **1.51x** the heat flux and **1.58x** the `w*` of the representable ones. The fallback is
  `218^2 @ 16 m` (`L = 3488 m`, 3.2x cost), and that is a grid decision.
