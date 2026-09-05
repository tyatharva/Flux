# The stable regime does not survive this grid

Measured 2026-08-25 and 2026-08-26 on the 16 m grid with a 10 m receptor; the resolution
argument carries to the 30 m production grid, whose `Δ` is larger. A stable boundary layer at
the canonical benchmark regime cannot be sustained, the reason is resolution rather than
forcing, numerics or a bug, and every standing check passed while it died.

## Two collapses from two causes

**The cold start (fixed, traps §15).** `sbl` at `G = 6 m/s`, `w'θ' = −0.020 K m/s`,
cold-started: `u*` 0.219 → 0.043 m/s in an hour, `z_i` 209 → 61 m, `z/L` +34.8, 2551 K/km at
the first level, the mean wind at exactly the geostrophic value above 66 m with `Ri_g` about
1e8. Runaway surface cooling under a prescribed-flux boundary condition with no turbulence to
mix it away; GABLS1 prescribes a cooling rate for this reason. Fixed with a neutral warm-up
segment so turbulence exists before cooling starts, which is also how a stable layer forms in
nature.

**The warm-started run at GABLS1's own regime (not fixed).** `G = 8 m/s`, `w'θ' = −0.012 K m/s`,
one neutral warm-up, 3.0 sim-h, base angle 30°. Healthy for 1.75 h: `u*` 0.236, `z/L` +0.123,
`Ri_g` 0.03–0.05, 11.2° of Ekman backing, a coupled, sheared, turbulent layer. Then it
collapsed: `u*` 0.098, `z/L` +2.67, resolved TKE at 8% of its own peak, all seven stationarity
limits failing, `x_peak` binding last at 6989% of its limit.

## The cause is resolution, measured at the healthy dump

The Ozmidov scale `L_O = sqrt(eps/N³)` is the largest eddy stratification permits to overturn.
An LES represents turbulence in the band `Δ < l < L_O`; `L_O/Δ ≈ 1` means there is no band and
the model is running a sub-grid closure and calling it a boundary layer. `eps` and the mixing
length are FastEddy's own (`bin/ozmidov.py`, `results/ozmidov_regimes.txt`):

| regime | `L_O/Δ` at the 10 m receptor | surface layer (min–median) | resolved `σ_w²` at the receptor |
|---|---|---|---|
| **stable** (GABLS1 regime), healthy dump at 0.92 h | **3.57** | 2.41–5.21 | **0.2%** |
| neutral | 318 | 43–93 | 2.7% |
| convective | unstratified, no constraint | | 12.1% |

A factor of 89 between stable and neutral at the same receptor on the same grid, and the
verdict does not turn on where a threshold is drawn. The column median of `L_O/Δ` reads 8.97
and would have hidden it: score where the answer is needed. Convection has no Ozmidov
constraint at all because below 50 m `N² ≤ 0`, which is why convective cases are the easy case
for this grid and a convective pass says nothing about stable.

## Why every check passed while it died

| check | reading | why it missed |
|---|---|---|
| exit status | 0 | never trusted alone |
| `CORRUPTED` grep, `np.isfinite` | clean | nothing was numerically wrong |
| **`k0/k1`** | **0.442, a comfortable pass** | a ratio between two levels, and both went quiet together |
| column-integrated TKE | rising | gravity-wave variance aloft, which grows as turbulence dies |
| `z_i` as 5% of peak TKE | falling | a diagnostic artifact in its own right (traps §16) |

`k0/k1` is a `dt` check, not a physics check. `docker/turb_alive.py` now asks whether a boundary
layer still exists, everywhere `k0/k1` runs. Its obvious metric was also wrong: `e_res/u*²`
reads 11.72 at the healthy peak and 4.71 after the collapse, inside the healthy band, because
`u*` dies with the turbulence. The check scales against the forcing instead, `max_k e_res/U_ref²`
with `U_ref` the geostrophic wind: healthy 4.6–9.1e-3, dead 6.3–7.1e-4, a factor of 6.5 with
nothing between.

## What it would cost, and the weakly stable attempt

GABLS1 uses `dx = 6.25 m`: 2.6× finer and about 17× the cells for this domain. The framing that
generalises is depth relative to the filter width: GABLS1 `z_i/Δ` 28.8, the collapsed rung 14.9.
So a deeper, weakly stratified layer might be resolved at 16 m in the same relative sense, and
the knob is more wind, not less cooling: `z/L` falls as `u*⁻³` while `eps` rises as `u*³`, and
the layer deepens as `u*²`.

How much of the site is weakly stable was measured over three sources (`bin/stable_fraction.py`,
`results/stable_fraction.txt`): median stable `z/L` 0.056 (tower), 0.063 (HRRR), 0.071
(CONUS404); the share of stable hours with `z/L ≤ 0.10` 65.8%, 64.9%, 60.7%; excluding the rest
costs 14–15% of the QC'd record. Without the `u* ≥ 0.15` QC the runnable share falls to
45.2 / 43.8 / 27.6%.

**`sbl-weak`**: `G = 10 m/s`, `w'θ' = −0.012`, `z_i` target 280 m (`z_i/Δ` 27.7), at the site's
median stable hour so a pass would license the band. 3.0 sim-h, base angle 30°:

| t [h] | `u*` | `z/L` | backing | `Ri_g` at 20 m | `dθ/dz` at 2 m | `zTKE95` |
|---|---|---|---|---|---|---|
| 0.75 (cooling starts) | 0.2794 | 0.074 | 8° | −0.000 | −0.0 | 92 m |
| 1.50 | 0.3334 | 0.044 | 7° | 0.012 | 7.1 | 559 m |
| 3.00 | 0.1848 | 0.253 | 21° | 0.043 | 12.4 | 1825 m |

`u*` at 40% of its own peak and falling at −40 %/h, resolved TKE at 5% of its peak, all seven
limits failing. **Halving the stratification bought a slightly slower death.** And this was not
the cold-start failure: `Ri_g` peaked at 0.043 against a critical 0.25, Ekman backing was normal
and increasing, the inversion an ordinary 12 K/km, the flow aloft departing from geostrophic.
The surface layer stayed healthy while the height holding 95% of the column TKE ran 92 → 1825 m:
the turbulence failed to be resolved at the surface and what the integral still counted was
wave energy aloft. `bin/sbl_diagnose.py` (retired) scored both signatures; on the retired GABLS1
seed it reported starved *and* decoupled, so it discriminated. One alternative not excluded: a
stratified periodic box with a 500 m sponge can also accumulate upward-propagating wave energy.
The load-bearing evidence is the Ozmidov measurement at the healthy dump, independent of what
happened later and above.

## The decision

**Stable is excluded from the corpus.** `bin/select_times.py --max-zol` defaults to 0.0; the
`sbl` rung was deleted; a third respec chosen to obtain a pass would be tuning, not measurement.
The cost is small in cases and large in coverage: day coverage 80.4% → 75.0% at the time (a
case is drawn from any acceptable hour of the day), the convective share of selected cases
40.5% → 65.2%, and **the emulator is undefined in stable conditions, about 44% of the site's
QC'd hours**. 26% of retained cases still fall outside 06–18 local time; those are near-neutral
or weakly unstable nights and must not be quoted as stable coverage.

## What this changed about how we work

`k0/k1` is a `dt` check and was being read as a health check. A ratio between two quantities
that die together cannot detect the death. Score a profile where the answer is needed, not where
it averages well. Diagnose the grid at the healthy dump: every number above was available an
hour before the collapse. A regime a gate has never run in is unknown, not fine.

Artifacts kept: `results/ozmidov_regimes.txt`, `results/stable_fraction.txt`, `results/time_selection.txt`,
`bin/ozmidov.py`, `bin/stable_fraction.py`, `docker/turb_alive.py`. Removed on 2026-09-04 (at the
`pre-cleanup-2026-09-04` tag): `results/retired_sbl_gabls1/`, `results/retired_sbl_weak/`,
`results/sbl_seed_report.txt`, `results/sblweak_seed_report.txt`, `bin/sbl_diagnose.py`.
