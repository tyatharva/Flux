# Case generation

One corpus case turns a timestamp into an (input, target) training pair. `bin/run_corpus_case.sh`
drives the eight stages. `bin/run_corpus_machine.py` runs one case per day over a machine's
share of the calendar ([deployment](deployment.md)).

```
bin/run_corpus_case.sh 2023-01-18T18:00 [tag]
```

| # | stage | file |
|---|---|---|
| 1 | HRRR pseudo-sounding at the tower | `bin/hrrr_sounding.py` |
| 2 | sounding → FastEddy `.in` parameters | `bin/sounding_to_forcing.py` |
| 3 | this case's surface heat-flux map | `bin/case_surface.py` |
| 4 | which seed, and which 90° rotation | `bin/pick_seed.py` |
| 5 | rotate the seed's flow, inject the static surface | `bin/prep_restart.py` |
| 6 | 30 min adjustment + (30 min + `t_back`) window, one invocation | `bin/run_window.sh` |
| 7 | backward LPDM → the 122 × 122 footprint | `bin/stage5_footprint.py`, `lpdm/` |
| 8 | assemble the training record | `bin/make_pair.py` |

Defaults are the 30 m production geometry (`GRID=data/grid30_raised`, `ZTARGET=28.5`,
`EXACT_AGL=1`, `SEED_LIB=seeds`, `COVER_GROUPS=10`, `WINDOW_S=2700`, `ADJ_S=1800`,
`TBACK=900`). They were the retired 16 m values until 2026-08-31. That meant every caller had
to export the whole block, and a caller that forgot got a complete, plausible case on the wrong
grid. Forgetting now gives production. `STUB_LES=1` stops after stage 5 and replaces the LES
and LPDM with an analytic stub, for CPU-only checks of everything else.

The adjustment is the reason the seed library exists. A cold start would need 3 simulated
hours. Restarting from a seed of the right regime, depth and heading needs 30 minutes.
Across the corpus that is the difference between about 2000 and 7600 GPU-h.

## How a case is chosen

Per day, draw a round hour without replacement from the 24, seeded from the date alone, so a
re-run reproduces the sequence. Screen it: HRRR analysis present, `z/L < 0`, `z_i` in
300–1250 m, `|dz_i/dt| < 15 %/h`. An hour is spent the moment it is drawn, so a day ends
at an accepted hour or as a **missing day with a reason**. There is no rose weighting and no
direction stratification. The weather supplies the rose. At most one case per day, because
two hours of one day share the synoptic state, the soil moisture and the morning's history.

The screen reads `hpbl`, `shtfl` and `dz_i/dt` and nothing else (`lpdm/corpus.py:screen`).
The wind is a label fetched once on the accepted hour. `HPBL` is the depth diagnostic because
it is HRRR's own PBL-scheme depth, so it is the same diagnostic in every case.

**Why enumerate hours instead of taking midday.** Measured on 92 days of 2023 (2208 candidate
hours, `results/candidates.tsv`): picking 18Z only accepted 65.2% of days at `z_i ≤ 976 m`.
Enumerating hours with the stationarity screen accepted 91.3%. Raising the depth cap to
1200 m added 1.1 points. Once every hour is a candidate, every one of the 92 days had some
hour with an acceptable depth, and the binding constraint became stationarity. All seven
days with no case failed the stationarity screen.

**`dz_i/dt` is screened separately from `z_i`, and that is the whole design.** Widening the
depth band pushes selection toward morning and evening, exactly when `z_i` changes fastest
(median `|dz_i/dt|` 15.9 %/h at 09 CST, 14.8 %/h at 18 CST, 6–7 %/h overnight). The 15 %/h
threshold is comparable to the LES's own entrainment drift over a window (about 8 %/h at
500 m). The result is insensitive to it: 89.1% at 8 %/h, 91.3% at 15, 93.5% at 30.

The `z_i` band is 300–1250 m. The floor is `10 z_m` and follows the receptor. The ceiling is
the lower of the width constraint `L ≥ 2 z_i` (1464 m) and the domain-height constraint (about
1250 m, half the clean column under the 500 m sponge). The exclusion of deep boundary layers
is biased, not neutral. Rejected unstable hours have 2.33× the mean surface heat flux of
accepted ones, and `z_i` correlates with surface flux at +0.43 to +0.49. The corpus is
thinnest exactly where the array's flux enhancement is largest.

Day yield follows the weather: 78% overall but 38% in June and 57% in July. A summer convective
boundary layer is inside the band only while growing at 17–45 %/h.

**Conventions.** Timestamps are UTC. Averaging is period-ending, matching the tower. A
footprint stamped 01:00 UTC is the average over 00:30–01:00. The forcing is the HRRR analysis
valid at exactly T, not T−1, because forcing is constant through the run. The LES is
initialised from the sounding and integrates 1.25 h under fixed geostrophic wind and surface
flux, so it never evolves from a T−1 state toward a T state. HRRR analyses are hourly, so a
day has 24 candidates, not 48.

## Stage 1: the HRRR pseudo-sounding

Herbie, HRRR `nat` product (hybrid levels. `prs` puts 3–4 pressure levels in the whole
boundary layer), `fxx = 0` analysis, plus `sfc` for `HPBL`, `SHTFL`, `LHTFL`, `PRES` and the
10 m wind. Only the lowest 20 hybrid levels are fetched. HRRR numbers level 1 at the model
bottom (level 1 at 289 m ASL here, level 20 at 6413 m), so 1–20 reaches about 6.1 km AGL,
which covers the LES column, the 4 km ceiling of the `z_i` searches and the above-BL
geostrophic layer. Checked: 20 and 50 levels give bit-identical `z_i`, fluxes, Bowen ratio,
winds and geostrophic wind, and a shared profile matching to 0.000e+00. `SPFH` is dropped
(the run is dry). The GRIB is deleted after extraction. The durable artifact is an 8 kB
sounding JSON.

Two traps, both of which produce plausible wrong numbers:

1. **HRRR GRIB winds are grid-relative.** On the Lambert grid at this longitude that is a
   5.11° rotation, invisible in the wind speed. Rotated with pyproj's own meridian convergence.
   The first implementation caught the failure in a bare `except` and left the angle at
   exactly 0.0 with no warning.
2. **HRRR longitudes run 0–360.** Left unnormalised, the geostrophic box test matched zero
   points and the code fell back to the above-BL proxy silently. Both paths now warn.

Four `z_i` diagnostics are reported (on 2023-07-15 19Z: `HPBL` 1648 m, bulk Richardson
`Rb = 0.25` 1106 m, parcel `θ_ml + 0.5 K` 1244 m, maximum θ gradient 2041 m). The
maximum-gradient pick was the original estimator and is wrong on a summer profile with no
capping inversion. It is kept in the output, labelled, because deleting it would only invite
it back.

**The geostrophic wind is the above-BL wind**, a height average over `[z_i + 50, z_i + 550]`
on a uniform 25-point grid (above a 1648 m boundary layer that slab held exactly one hybrid
level, so a level mean was a single sample). It is not the height-gradient wind. FastEddy runs
doubly periodic, so it can represent neither synoptic curvature nor a horizontal height
gradient, and on 2023-07-15 19Z the actual wind was 6.2 m/s where the 850 mb gradient said
10.7, because 850 mb was inside the boundary layer. The gradient estimate is recorded as a
diagnostic and the disagreement reported.

## Stage 2: sounding to forcing

FastEddy's base state (`stabilityScheme = 2`) is continuous piecewise-linear in θ with four
segments, `hydro_core.c:1776-1822`:

```
z <= b1        theta = theta_grnd                     (forced neutral, no free gradient)
b1 < z <= b2   theta = theta_grnd + g1 (z - b1)
b2 < z <= b3   theta = ...        + g2 (z - b2)
z  > b3        theta = ...        + g3 (z - b3)
```

Three constraints. The lowest segment has no free gradient, so for a stable case the fit
drives `b1` to 0. All three gradients must be strictly positive (queried over
`[FLT_MIN, FLT_MAX]`), so they are floored at 1e-4 K/m. **A rejected value does not stop the
run.** `parameters.c:309-315` prints `outside limits`, increments `numErrors`, leaves the
variable at its compiled-in default, and `FastEddy.c:96` never checks the return code. An
out-of-range `stableGradient` silently runs with 0.1 K/m, a 10 K capping inversion where the
sounding wanted 0.4. This stage guarantees the ranges, and `bin/test_sounding.py` re-checks
every one against the source's own limits.

The fit is done on the LES's own cell centres, weighted by layer thickness, so the residual is
an integral over height and invariant to how the grid is stretched (0.04–0.27 K rms over the
column). The stage also emits `(U_g, V_g)`, `surflayer_wth` as the domain mean of the per-cell
virtual map (getting this backwards spins the seed up at the wrong `z_i`), the ground state, a
`dt` that puts the 5 s cadence on an integer step count, subsidence with its knee at the
case's own `z_i`, and the per-case Bowen ratio from `SHTFL`/`LHTFL` (on 2023-07-15 19Z
B = 0.44, `w'θ'` 0.1204 → `w'θ_v'` 0.1406 K m/s). The Bowen ratio makes the
sensible-to-virtual conversion exact rather than a class assumption.

Direction is recorded, not corrected. The forcing is the real above-BL wind and the LES finds
its own Ekman turning over the real roughness. `dir10_residual_deg` stores HRRR's own 10 m
direction minus the Ekman prediction (+19.3° on 2023-07-15 19Z. The thermal wind can exceed
the 10° convective Ekman angle at this site). `--match-10m` rotates the forcing instead.

The receptor height is read from `<grid>/meta.npy` and the domain `z0` from `<grid>/z0m.npy`,
not hardcoded. Soundings above the depth cap are flagged `representable: false` rather than
run and mislabelled.

## Stage 3: the per-case surface

`bin/prep_restart.py` injects `htFlux` into the restart file, and the restart read overwrites
the `.in`'s `surflayer_wth`. A neutral grid ships with `htFlux.npy` all zeros. Point a
convective case at it and it runs neutral, exits 0 and says nothing. So each case gets its
own map: `wth_reference × f`, where `f` is the class-ratio field from
[the site](../problem/site.md), which does not depend on the case. The static geography is
hardlinked and only `htFlux.npy` is written fresh (a case directory is about 116 kB).
Validated bit-for-bit against the campaign's own grid, with the tables read from
`prep_surface.py`'s own source rather than copied.

| flux | map |
|---|---|
| `> 0` | per-class daytime ratios. The array is 1.376× the cropland reference (virtual) |
| `≈ 0` | zero everywhere |
| `< 0` | uniform. The class table is a daytime table with no nocturnal equivalent |

## Stage 4: which seed

The metric is "what will 30 minutes fail to close", and nothing else. Regime (from the
prescribed virtual heat flux) is a hard constraint. A convective boundary layer turns over in
about 1.2 h, so 30 min does not convert one regime into another. `z_i` costs
`|ln(z_i,seed / z_i,case)| / ln 2`. Direction costs `d_dir / 30°`. `z/L`, `u*` and the
geostrophic speed are reported and never costed (they re-equilibrate in about 2 min. `U/u*`
moved 0.6% while `u*` moved 18% across five windows), and the speed ratio warns past a factor
of two. An earlier version standardised every axis by the library's own spread, which
weighted the narrowest axis the most and inverted the table. The scales are now fixed and
physical.

Seeds are matched on what they **achieved** (`manifest["achieved"]`: measured `z_i`, `u*`,
`U`, direction), not on what they were asked for. The seed's own direction drift at freeze
time is projected forward over `ADJ_S + WINDOW_S/2` = 0.875 h, because measured on two cases
the adjustment widened the direction gap rather than closing it (11.3 → 21.8° and
14.1 → 36.0°). The depth converges faster than designed: +233 m/h measured against an
assumed +79 m/h, because entrainment is set by the case's flux against the case's inversion.

**Selection uses the whole library** (`ALLOW_DRIFTING=any`, the default since 2026-08-31).
Seeds with a DRIFTING limit and INDETERMINATE ones are ranked alongside any that passed. A
seed is an initial condition, not a corpus point. The case adjusts under its own forcing and
every ML input is measured over the footprint's own window, so the pair is self-consistent
whatever the seed's drift state. `gate_state` is stamped on every pair. `--strict-gate`
restores the refusal. Ekman backing per rung is measured from the library and used for unspun
seeds only (convective 5.2°, neutral 16.9°, against a nominal 22–23.5°).

A mismatch does not corrupt a case. Inputs are read from the LES window, so an imperfectly
closed gap moves where a case lands in input space without making it wrong. Seed spacing is a
coverage question, not a correctness one. Rotation by 90° multiples is exact (measured to
1.2e-14), so 6 base angles at 15° give 24 headings, worst case 7.5° before drift.

## Stages 5 and 6: restart and run

`prep_restart.py` rotates the seed's flow by the chosen multiple of 90° and writes the static
surface (terrain, `z0`, `htFlux`) into the restart file. That is the only way FastEddy
v5.0.1 takes spatially varying roughness or heat flux
([configuration](configuration.md)). The receptor is released at a fractional level exactly
28.5 m above the raised surface (`EXACT_AGL=1`). Snapping to the nearest level would put it
elsewhere by up to a cell.

`run_window.sh` runs 1800 s of adjustment and one 2700 s window as one invocation of 145,800
steps. Under `lpdmOnlineSelector = 1` the LES passes each snapshot to the LPDM in RAM and
writes no window netCDF ([LPDM and the footprint](lpdm-and-footprint.md)). It deletes the
adjustment's dumps and refuses unless the earliest survivor is the adjustment-end step.

## Stages 7 and 8: footprint and record

`stage5_footprint.py` runs the backward LPDM over the window and writes the 122 × 122 signed
footprint with its metrics, with `COVER_GROUPS=10` independent release groups for the
array-share standard error and `--t-min` refusing any field before the adjustment end.
`make_pair.py` evaluates the official FFP on the raster's own cell centres, assembles the
self-contained `.npz` (`scalars`, `kljun`, `target`, `meta`), checks the cell edges against the
centres to 1e-9 m and warns when `L` is non-finite. `run_id` and `split_key` are written into
every record. `bin/check_npz.py` validates the record against the format at the end of every
case.

## Why HRRR forces the runs and CONUS404 does not

| | CONUS404 | HRRR |
|---|---|---|
| horizontal | 4 km | 3 km |
| atmospheric profiles | none. The only 4-D variables are soil and snow | about 50 hybrid levels |
| surface fluxes | none | `SHTFL`, `LHTFL`: a per-case Bowen ratio |
| per-timestamp subsetting | none | Herbie |
| record | one configuration, WY1980–2024 | v4 from 2020-12-02 |

CONUS404 keeps its role as the 45-year climatology that characterises the site. The corpus
trades configuration homogeneity for resolution and per-case realism. The five-year span is
inside HRRR v4 to keep that trade small.

## Validation

`bin/test_sounding.py` runs stages 1–2 on four timestamps (summer convective midday, a
summer nocturnal stable layer, winter midday, an autumn morning transition) and asserts:
monotone `z`, physical θ, stratification the right way up, the meridian convergence applied,
the `z_i` diagnostics bracketing HPBL, every `.in` parameter inside FastEddy's own declared
range, the base-state fit within 0.5 K rms and reproducing the sounding below 1.5 km, the 5 s
cadence on an integer step count, `CFL_3d ≤ 1.35`. `bin/smoke_check.py` ran a 5-minute cold
start per regime configuration and confirmed the base state in the dump (max
|θ_LES − θ_base| = 0.0001 K. The convective rung's 0.16 K is subsidence working, predicted
0.167 K). The end-to-end case `e2e_20230118` (2023-01-18 18Z) ran all eight stages in 13 min
of GPU and put 55.4% of its north-easterly footprint on the array, which is what the geometry
says it should. The ninth pass validated the in-RAM hand-off on real production cases
([ninth pass](../history/pass-9.md)).
