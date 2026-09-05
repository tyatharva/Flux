# The seed library

Thirty pre-spun turbulence states, one per rung and base angle, that every corpus case
restarts from. A seed exists only to remove the spin-up. A case restarts from the nearest
seed, adjusts for 30 minutes under its own sounding's forcing, then samples for 30 minutes.
Seeds are never training data. Specs and verdicts are in `seeds/`. The 30 restart files
(73.3 MB each) are on Hugging Face ([data](../getting-started/data.md)). The machine-level
records of the run are in `results/seed_library/`.

## Why a library, and why these axes

A cold start needs about 3 simulated hours here. Restarting from a seed of the right regime,
depth and heading needs 30 minutes. Across the corpus that is roughly 2000 GPU-h against 7600.

The axes are chosen by what 30 minutes cannot adjust. That is the only criterion for a state
whose purpose is to be adjusted away. Every number was measured on this project's own runs:

| quantity | closed in 30 min? | evidence | axis? |
|---|---|---|---|
| direction | no, about 2.7° | −5.4 °/h backing on `g16_spin` | yes: base angles |
| `z_i` | no, about +40 m | +79 m/h entrainment on `g16_cbl_shallow` (later measured at +233 m/h on a real case) | yes: five depths |
| stability regime | no | a convective layer needs about 8 `T*` ≈ 1.2 h to turn over | yes: in the rungs |
| `u*`, wind speed | partly | the surface layer is about 0.1 `z_i` deep and re-equilibrates in minutes | no |
| fine `z/L` | yes | the surface flux is prescribed and the surface layer follows | no |

The rungs are coupled, not a product. A 150 m stable layer cannot support a 12 m/s geostrophic
wind, so `G` belongs to the rung. The rungs follow the site's real joint distribution as
CONUS404 measures it at this tower (`z_i` p25/p50/p75 = 267/493/835 m, `w'θ'`
−0.006/+0.015/+0.076 K m/s, `U(30 m)` 3.9/5.2/6.8 m/s):

| rung | regime | `z_i` target | `w'θ_v'` [K m/s] | `G` [m/s] | how `z_i` is held |
|---|---|---|---|---|---|
| `nbl-shallow` | neutral | 300 m | 0.000 | 8 | capping inversion alone |
| `nbl-deep` | neutral | 550 m | 0.000 | 12 | capping inversion alone |
| `cbl-shallow` | convective | 450 m | +0.060 | 7 | cap + subsidence |
| `cbl-mid` | convective | 700 m | +0.110 | 9 | cap + subsidence |
| `cbl-deep` | convective | 950 m | +0.160 | 11 | cap + subsidence |

The capping inversion is +8 K across 100 m (`stableGradient = 0.08`), then a free-atmosphere
lapse of 0.004 K/m. It is the `z_i` control, not a profile to be matched. A stable rung (`sbl`)
was built twice and collapsed twice. The cause is resolution ([stable regime](../history/stable-regime.md)).

**Six base angles at 15°.** A square doubly-periodic flat uniform domain with `dx = dy` is
exactly equivariant under 90° rotation (measured to 1.2e-14), so each base angle re-indexes
into four headings: 6 × 4 = 24 library headings, worst case 7.5° before drift. Three angles
at 30° were used first and measured insufficient. The direction gap widens through the
adjustment rather than closing (11.3 → 21.8° and 14.1 → 36.0° on the two cases that tested
it), so the worst case on the dominant skill axis was 25–35°. Denser angles were the correct
fix. Smarter projection was not. The cost is 30 seeds instead of 15, about 2.5% of the corpus.

The seed's own drift at freeze time is projected forward by `bin/pick_seed.py`, which removes
the mean of the excursion and leaves its scatter. Ekman backing per rung, measured from the
library: cbl-deep 4.2°, cbl-mid 5.0°, cbl-shallow 6.4°, nbl-deep 14.3°, nbl-shallow 19.6°
(n = 6 each). By regime it is convective 5.2° and neutral 16.9°, against a nominal 23.5°
neutral and 22.1° convective.

## The spin-up

Each seed is one continuous FastEddy invocation from a flat, uniform, doubly-periodic
perturbation field: empty `topoFile`, empty `inFile`, no sounding, terrain or land cover.
Chaining segments was retired on 2026-08-26 after `bin/test_unchained.py` measured chained
against unchained at 0.89–1.08× the run-to-run reproducibility floor. Nothing is lost by
removing it, and the 1-hour segment cap went with it. There is a stationarity dump every 300 s.

**The ceiling is 2.0 simulated hours, every rung and every regime**, and the stop is
measured. `bin/seed_watch.sh` scores the trailing window every 30 simulated minutes and stops
the run as soon as the oscillation-immune limits are in band (`U/u*`, `σ_v/u*`, `σ_w/u*`,
Kljun `x_peak`, Kljun `x90`). `TKE_BL/u*²` and `z_i` are excluded from the stop criterion on
purpose. They decorrelate on the eddy turnover rather than on the dump interval, `n_eff`
saturates at 3–5 at every window width from 1.0 to 2.5 h, and requiring them would mean never
stopping early while misreporting why. A DRIFTING verdict on any limit still blocks the stop.
A seed that has not entered band by the ceiling stops there and that is the result. No
extension, no respec. The gate scores `min(2.0, sim_h − 0.5)` = 1.5 h, and refuses a window
that reaches the first dump.

The first design was 3.0 h flat, the first duration where all seven limits passed on the 16 m
prototype (2.0 h failed on `TKE/u*²` at +15.6 %/h). An early `SEED_CEILING_H = 2.0` produced
1.9167 h, not 2.0, because a 1e-6 s tolerance on a whole-dump decision fired on a 0.6 ms
floating-point artifact and removed a 300 s dump ([standing rule 4](../reference/standing-rules.md)).

**Neutral rungs get Steinfeld's accelerator**: 3000 s at `surflayer_wth = +0.05 K m/s`, then
the open-ended run at the rung's own flux, restarting from the burn-in dump with `htFlux`
zeroed *in the file* (`bin/zero_htflux.py`, which re-reads to confirm). That restart is the
only one inside a seed and it is the dangerous kind. `htFlux` is IO-registered, so the main
invocation would otherwise inherit +0.05 whatever its `.in` says. Neutral is the regime the
accelerator is for. `h/u*` is about 1500 s there against `T* ≈ 350 s` convectively, so it is
the slowest to organise a perturbation field into turbulence. The accelerator leg is logged
separately (`accel.log`). Counting only `run.log` under-reports a neutral seed by about 29%.

`surflayer_z0` is read from the grid (`data/grid30_raised/z0m.npy`, geometric mean 0.0615 m),
not from a constant. An earlier hardcoded 0.1435 m was the 16 m map's value and would have
spun every seed up over the wrong surface. The gate's receptor (`zm`, `k`) is in the manifest
and passed explicitly. Scored with defaults, a 30 m library would evaluate Kljun's `x_peak` at
the wrong height and read `σ_w` from the wrong level, and every number would still print.

## The gate is on `U/u*`, not on `u*`

A doubly-periodic neutral Ekman layer forced by a constant geostrophic wind does not settle
to a fixed `u*` on any affordable timescale. `f = 9.94e-5 s⁻¹` here, so the inertial period
is **17.6 h**, and `u*` falls for a quarter of it and then rises. Measured on `g16_spin`, `u*`
moved −27% over 6.26 simulated hours while `U/u*` was within 0.31% of its final value by
3.01 h. Gating on `u*` alone failed this project's spin-ups twice for a reason that was never
a modelling error. Kljun's `Π_4 = U(z_m)/u*` is the only channel through which the wind enters
the streamwise footprint shape, and both of its terms follow the oscillation together. So do
`x_peak` and `x90`.

`z_i` is the one exception. It is a length, with no `u*` to cancel. It is made immune by how it
is measured. The gate uses a fixed 0.01 m²/s² threshold, while `window_stats` produces the
corpus input `h` as a peak fraction, and the two differ by 7–21%. The gate measures a trend
and needs a threshold that does not move. The matcher compares a value and needs the
definition the corpus inputs use. A linear trend through a staircase reports the staircase
(`z_i` only falls on model levels), so the gate prints the distinct-level count and span
beside the trend.

The seven limits, scored on the last 1.5 h by `bin/seed_stationarity.py` (the single
definition, and nothing restates them):

| quantity | limit |
|---|---|
| `U/u*` (Kljun `Π_4`) | 1.0 %/h |
| `σ_v/u*` | 3.0 %/h |
| `σ_w/u*` at the receptor | 2.0 %/h |
| `TKE_BL/u*²` | 5.0 %/h |
| `z_i` (fixed-threshold) | 3.0 %/h |
| Kljun `x_peak` | 1.0 %/h |
| Kljun `x90` | 1.0 %/h |

Each trend is reported with an AR(1)-corrected standard error and `n_eff`. The verdict is
INDETERMINATE rather than PASS or FAIL when the threshold is within 3 SE of the measurement.

## The run: 30 seeds, 16 RTX 5090, 0.936 h

Delivered on 2026-08-31 from the deployment image ([deployment](deployment.md)), and checked
on the artifacts rather than on the exit status:

| check | result |
|---|---|
| seeds returned | 30 of 30, 376 files, 2.1 GB, zero empty files |
| `seed_restart.nc` | all present, all exactly 73,271,565 B, 30 distinct md5 sums |
| finiteness | `np.isfinite` on `u`, `v`, `w`, `theta` in all 30: no NaN, no Inf |
| `CORRUPTED`, `#NaN`, `#Inf` | none in any run log |
| `achieved` block | present in 30 of 30 manifests |
| `k0/k1` | 0.124–0.144, OK on all 30 (the Blackwell numerics check) |
| `turb_alive` | real OK on all 30, not a SKIP |
| Gate C2 restart | bit-for-bit on all 30 |
| static rotation check | PASS on all 30 |
| ceiling arithmetic | `run.ceiling_steps` = 233,280 = 2.000 sim-h in every manifest |

The physics is ordered as expected. `w_rms` is 0.28–0.74 m/s on the convective rungs against
0.07–0.20 neutral, and within a rung the six base angles cluster tightly (`U` spreads under
0.05 m/s). The machine was 16 × RTX 5090 (`sm_120`, 32,607 MiB each). Peak VRAM was 904 MiB
per seed (2.8%). Peak container RSS was 58.9 GiB, and summed FastEddy RSS 8.98 GiB over 32
processes. The work queue behaved as a work queue: 16 of 16 workers used, 14 took a second
job, peak concurrency exactly 16, zero failures.

**0.189 GPU-h per simulated hour under full 16-way load**, against 0.469 measured single-GPU
on the RTX 4080 from inside the same image (0.479 at Ada bring-up). Contention cost nothing.
The number is 2.5× better. The LES leg is the same in both regimes (1365.8 s convective,
1361.9 s neutral). The neutral rungs' all-in 0.268 is entirely the accelerator (+567 s).
**This is a seed number and must not be carried to the corpus.** A case additionally runs the
LPDM and the ring, whose 12.0 GB host cache has never been exercised 16-way.

The thread-block sweep picked `1x8x16` at 0.00580 s/step where Ada picks `1x2x64` (0.00590
here, 1.017×). Do not read that as a Blackwell result. FastEddy prints its timing to five
decimals, so one quantum is 0.00010 s = 1.7%, three shapes tie exactly, and the reported
"repeat noise 0.00%" is quantisation. The worst case if the choice is wrong is 1.7%.

## The gate accepted 11 of 30, and that was never a quality statement

All 19 refusals are DRIFTING verdicts. Every one of the nineteen produced a complete, finite,
battery-passing seed. Drifting limits per seed: `TKE_BL/u*²` 11, `σ_w/u*` 9, `σ_v/u*` 4, `z_i` 3,
Kljun `x_peak` 2, `U/u*` 1. The drift is a rung-wide property. The mean gated `TKE_BL/u*²`
trend is cbl-deep +3.5 %/h (5/6 accepted under the strict gate), cbl-mid +12.5 (2/6),
cbl-shallow +22.5 (0/6), nbl-deep +19.4 (3/6), nbl-shallow +35.9 (1/6).

**The accept/refuse split followed the standard error, not the trend.** INDETERMINATE within
3 SE is correct and intended, but at these magnitudes the better-measured seed is the one
refused. `cbl-shallow_a000` at +23.5 %/h with SE 7.37 was admitted as INDETERMINATE, while
`cbl-shallow_a030` at +22.0 %/h with SE 2.41 was refused as DRIFTING. The eleven are the seeds
whose drift could not be resolved.

**And `TKE_BL/u*²` was measuring its own references**, the fourth instance of
[standing rule 3](../reference/standing-rules.md). The absolute `TKE_BL` trend can be recovered
from what was returned by two routes with disjoint inputs, `trend(TKE_BL) = trend(TKE_BL/u*²)
+ 2·trend(u*)` and `trend(TKE_BL) = trend(domain TKE) − trend(z_i)`. Their agreement is the
evidence (`bin/seed_tke_rescore.py`, retired. The table is the record):

| rung | gated | `u*` | `z_i` | absolute | \|A − B\| | reading |
|---|---|---|---|---|---|---|
| cbl-deep | +3.5 | −7.3 | +16.7 | −13.5 | 4.9 | unresolved |
| cbl-mid | +12.5 | −8.8 | +6.4 | −5.0 | 0.2 | steady |
| cbl-shallow | +22.5 | −11.1 | −0.3 | **+0.5** | 0.5 | **steady** |
| nbl-deep | +19.4 | −2.4 | +4.0 | +12.7 | 3.8 | unresolved |
| nbl-shallow | +35.9 | −3.2 | +0.5 | **+29.4** | 0.1 | **rising** |

The gated ratio has three moving parts and only one is the turbulence. `u*²` falls through
the first quarter of the inertial period (about 10 %/h), and the averaging depth `z_i`
entrains upward. Every other gated limit is a ratio whose numerator follows the oscillation
with `u*` and cancels it. `TKE_BL` is an energy and nothing cancels. The two conclusions point
opposite ways. The convective rungs were not drifting (cbl-shallow's absolute BL TKE is flat
while the gate reports +22.5), and the neutral rungs are still spinning up (nbl-shallow rising
at +29.4 %/h with both routes agreeing). A 2.0 sim-h ceiling is short for the neutral half,
and that is a real limitation of the library.

## Decision: selection uses the whole library

Since 2026-08-31 `bin/pick_seed.py` ranks all 30. `--allow-drifting` defaults to `any`, and
`--strict-gate` restores the refusal. A seed is an initial condition, not a corpus point. The
case restarts from it, integrates 1800 s under its own sounding's forcing, and every ML input
is measured over exactly the same window as the footprint, so the pair is self-consistent
whatever the seed's drift state. Refusing a seed removes a restart point without removing any
error. `gate_state` is still stamped on every pair.

What the strict gate cost, measured: 11 of 30 seeds available, cbl-shallow 0 of 6 (the
weakly convective rung had no restart point at all), neutral 4 of 12 (four base angles),
Ekman calibration n = 5/2/3/1 with one rung absent. With the whole library the convective pick
for `case_2023052519` improved from cost 0.346 (`z_i` 766 vs 970 m) to 0.268 (1011 vs 970).
The neutral pick for `case_2023112120` improved from cost 0.983 with a 14.5° direction gap and
the half-spacing warning firing, to 0.216 with a 1.3° gap and no warning. That is 4.6× in cost
and 11× in direction gap.

Two defects were found and fixed along the way. `run_seed.sh` returned exit 1 for all thirty
seeds because no seed had a clean PASS. It now returns exit 0 when the run produced a seed,
and `SEED_STRICT_EXIT=1` restores the old signal. `n_accepted` in the machine manifest was the
count of neutral seeds (a reused variable name. The log said 11, the JSON 12).

## Running one seed by hand

```bash
bin/run_seed.sh seeds/seed_cbl-mid_a030 --dry-run   # preflight only, no GPU
bin/run_seed.sh seeds/seed_cbl-mid_a030
```

A job is one directory: `seed.in`, `manifest.json`, no absolute paths, no shared state. The
repo root is found from the script's own location. It needs a GPU the FastEddy binary
has SASS for (about 0.65 GB of VRAM measured at 122³. The manifests' 1.6 GB is an
unverified literal), Docker with the NVIDIA toolkit and about 1.8 GB of disk while running.
About 58 min of wall at the ceiling. **Not resumable**: a killed job costs the whole run, and
re-invoking a complete job is a no-op. It is serialised per device (`flock(2)` keyed on
`CUDA_VISIBLE_DEVICES`), because two runs writing one `output/` interleave their dumps and
look like a stall. The files returned under `return/` are `seed_restart.nc`,
`stationarity.json` and `.txt`, `manifest.json` with `achieved`, `acceptance.txt`,
`seed_report.json`, `turb_alive.json`, `rotation_check.json`, `direction_drift.txt` and
`budget.json`. A job whose gate fails still returns its artifacts.

Bitwise reproducibility does not hold across GPUs and is not sought. An out-of-range `.in`
parameter does not stop FastEddy (it prints one line and runs a different case), so a
hand-edited `.in` must be grepped for `outside limits` as well as `CORRUPTED`.

## What is outstanding

Seed runs should return the scored series, not only the verdicts and trends fitted to it. It
is a few kB per seed, and it would make the rescore above a measurement rather than a
reconstruction. The gated form of `TKE_BL/u*²` should be reconsidered now that it is known to
be driven by `u*` and the averaging depth (it was not changed on the same pass that
reinterpreted its output). The neutral rungs would benefit from +1.0 sim-h (about +2.3 GPU-h
for 12 seeds). Nothing has run the ring or the LPDM on Blackwell. The seeds exercise FastEddy
only.
