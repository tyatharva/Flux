# Fourth pass: true 30-minute footprints, a static raster and the convective regime

August 2026, on the third pass's 24 m grid. Four things the third pass left open were closed
and one regime was added. Every absolute distance is a 30 m-receptor number on the 24 m grid.

## A window is 30 min plus `t_back`, and FastEddy had to change to allow it

The third pass's footprints were 15-minute footprints with a 30-minute label. A backward
trajectory cannot be released until `t_back` seconds of field lie behind it. The averaging
period stays 30 minutes (that is what eddy covariance means). The LES window grows. That ran
into the 45-minute per-run ceiling of the time (a 2700 s window at 1.28 s of wall per
simulated second is 57.5 min) and could not be split, because lean `ioLPDMmode` output is not
restartable. `ioLPDMfullFrq` (patch 0003) writes any dump whose absolute step is a multiple of
it in full upstream form, so a restartable dump is at each end of a chain link. Verified on
400 steps: identical variable set and dtypes to a mode-0 dump. Values against an independent
mode-0 run at the nondeterminism floor (`u` 3.4e-5, `w` 3.4e-4 relative). The second segment
restarts cleanly. The cost is one 170 MB dump per link against 42.7 MB lean.

## The footprint raster is the LES grid

The third pass accumulated on a 60 m wind-aligned raster and interpolated onto the map. The
resample fell hardest on the near field, where the peak and the array are, and the emulator
will consume a fixed north-up raster. Touchdowns are now binned by their LES column index,
folded modulo the periodic domain, so a footprint cell *is* an LES column. The wind frame was
made exact rather than abandoned. The crosswind-integrated footprint is a histogram of the
touchdowns' own upwind coordinate, and Kljun is evaluated at the static cells' own
coordinates (8×8 sub-sampled per cell near the receptor). A side effect: the integral now
includes the wrapped touchdowns. A 541-dump window is 55 GB of field cache at fp32 and 28 GB at
fp16 on a 62 GB machine, so the cache became float16 and the 4-D linear interpolation was
written by hand in `lpdm/fields.py` (`scipy.ndimage.map_coordinates` refuses float16).

## CONUS404 as a climatology, never as forcing

`bin/conus404_site.py` streams a stratified 45-year hourly sample at the tower cell from the
USGS Open Storage Network pod (274 of 2740 time chunks, 39,456 hourly records, about 30 GB
streamed and discarded in 8 minutes). Quality-controlled at `u* ≥ 0.15 m/s` (65.2% of hours):

| | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| `z_i` [m] | 80 | 267 | 493 | 835 | 1475 |
| `w'θ'` [K m/s] | −0.027 | −0.006 | +0.015 | +0.076 | +0.164 |
| `u*` [m/s] | 0.17 | 0.24 | 0.32 | 0.44 | 0.65 |
| `U(30 m)` [m/s] | 2.4 | 3.9 | 5.2 | 6.8 | 10.0 |

Three things followed. The site is unstable more than half the time (27.2% very unstable,
30.3% unstable, 13.3% near-neutral, 20.4% stable, 8.8% very stable), so a neutral-only corpus
misses the modal daytime state. `z_i` spans a factor of 18 and must be swept because Kljun
takes it as an input. The wind rose (S 16.0%, NW 14.5%, W 14.4%, SW 14.3% against N 10.6%)
points away from the array signal. The grid-relative wind rotation (−5.55° here) is a whole
direction bin and is applied.

## The convective regime

`runs/g24_base/base_cbl.in`: an idealised dry CBL from the CONUS404 midday distribution.
`surflayer_wth = 0.11 K m/s` (127 W/m², the p50 of local 10–16 h hours), mixed layer 0–800 m at
300 K, capping inversion 800–900 m at 0.08 K/m, free atmosphere 0.004 K/m, `U_g = 10 m/s`
unchanged from neutral so the only difference is thermal. `w* = 1.42 m/s`, `T* = 562 s`,
`z_i/L = −18` designed. Spin-up 5400 s = 9.6 `T*`.

**Per-cell surface heat flux from the land cover** (`prep_surface.py --wth`): cropland 1.00,
tree/grass/shrub 1.10, built 1.50, water 0.12, array 1.60 (sensible-flux ratios. The virtual
conversion came later). With `surflayer_idealsine = 0` FastEddy's surface kernel reuses the
`htFlux` array the restart injected, so no source change was needed. The water multiplier is
directional and the array multiplier retires an accepted omission. PV modules are darker than
the crop they replaced and do not transpire, and field studies report a daytime enhancement of
1.5–2. Convectively the array is a roughness patch and a heat source. Neutrally, roughness only.

**The `σ_w` floor became stability-aware**: anchored to `σ_w/u* = 1.25 φ_w` with
`φ_w = (1 − 3z/L)^{1/3}` unstable (Panofsky et al. 1977) and `1 + 0.2 z/L` stable. `φ_w(0) = 1`,
so every neutral result is unchanged bit for bit. At the planned CBL the receptor target rises
from 1.25 to 1.81 `u*`.

The spin-up validated against convective similarity (`bin/cbl_check.py`, against Lenschow et
al. 1980, not NCAR's neutral case): `w*` 1.434 m/s (1.42 predicted), `T*` 574 s, `z_i` 823 m,
`w*/u*` 2.86, entrainment ratio 0.149 (0.10–0.35 expected), `σ_w/w*` within 10% of Lenschow
through the bulk. `z_i/L` came out −9.3 rather than −18 because the CBL mixed momentum down and
raised `u*` to 0.502. Kept, because holding `U_g` fixed is what makes the difference
attributable to thermal forcing alone. A diagnostic trap: reading the surface buoyancy flux as
the resolved covariance at `k = 0` gives 0.0088 against a prescribed 0.11, because at the lowest
level almost all the heat flux is sub-grid.

## How much backward time a window needs, and the bug the question exposed

Measured on the flat/neutral control by masking one release ensemble on touchdown age. **The
first measurement said something impossible.** The integral crossed 1 near `t_back = 670 s` and
kept climbing (0.747, 0.882, 0.968, 1.031, 1.089 at 300–900 s) while the peak stayed at 264 m
throughout. A finite backward time can only lose influence. Wraparound was ruled out (nothing
arrived beyond one domain length). **The cause was the `σ_w` floor's drift.** Thomson's reverse
drift contains `dσ²/dz`, the floor rescales `σ²` by a height-dependent `sc(z)`, and the drift
kept using the unscaled gradient. Correct is `sc·dσ²/dz + (2/3)e·dsc/dz`, and the second term
is the larger. The third pass never saw it because its wind-frame raster dropped everything
beyond 4500 m. Folding onto the LES columns removed the truncation and exposed the bias. The
third pass's reading of the floor moving the integral "toward 1" had been reading a bias as an
improvement. The standing flat control found it on its first run.

After the fix: monotone from below (0.671 at 300 s, 0.808 at 600, 0.888 at 900), peak 312 m
at every `t_back`, shape L1 against 900 s 12.0% at 600 s and 2.5% at 800 s, against a
half-vs-half floor of 29.4%. Well-mixed re-run with the floor active for the first time:
backward rms 3.61% against a 5.48% counting-noise floor. Magnitude does not converge and cannot.
Kljun on the identical 4464 m domain integrates to 0.875, so LES 0.888 against Kljun 0.875 is
agreement to 1.5%, and the shortfall below 1 is the tail outside the domain. **Decision:
`t_back = 900 s`, production windows 45 minutes.** Recorded rather than passed over: the
forward well-mixed control showed a 12–18% bulge over 30–70 m, and the transit-time check's
">50% reach the surface in 900 s" sub-criterion failed at 46.2% while the median transit was
3.0 min, inside the plan's range.

## Stage 6, neutral: PASS on the number the plan predicted

Four directions, one spun-up state, 30 min of releases, `t_back = 900 s`:

| case | wind from | array | × area | water | peak | integral |
|---|---|---|---|---|---|---|
| wN | 335° | **3.01%** | 13.9× | 0.00% | 264 m | 0.838 |
| wS | 158° | 0.85% | 3.9× | 0.47% | 300 / **1080 m** | 1.065 |
| wE | 66° | 0.01% | 0.1× | **15.33%** | 192 m | 0.919 |
| wW | 247° | 0.01% | 0.0× | 0.00% | 288 m | 0.731 |

Array swing wN/wE 215×, wN/wW 368×, against a prediction of about 300× from geometry alone.
The southerly is bimodal, and `bin/upwind_transect.py` said why. Tree cover (`z0 = 1.00 m`) in
a hollow at 300–600 m lifts trajectories over it, and the far lobe at 1080 m is on rising
open cropland beyond. It is a difference from Kljun you can point at on the map. **The floor
switched itself off over real neutral terrain** (factor 1.000–1.002. The LES already delivered
`σ_w/u*` 1.20–1.24) while it was 1.45× on flat ground and 3.1–3.4× convectively. The floor is
driven by stability, not terrain. The integral straddled 1 with the sign of the mean vertical
motion (wS 1.065 at `w̄ = +0.15`, wN 0.838 at −0.15): the advective non-closure, a result and
not an estimator error.

## Stage 6, convective: PASS, and convection changes the answer

| case | wind from | array | × area | water | peak | `x50` | integral |
|---|---|---|---|---|---|---|---|
| wN | 353° | **48.02%** | 222× | 0.00% | 144 m | 202 m | 1.250 |
| wS | 177° | 8.67% | 40× | −0.02% | 168 m | 272 m | 0.693 |
| wE | 83° | 0.21% | 1.0× | **5.32%** | 168 m | 191 m | 1.047 |
| wW | 263° | 0.09% | 0.4× | 0.00% | 240 m | 339 m | 0.759 |

**On a convective northerly the tower is measuring itself: 48% of the flux from 0.22% of the
domain.** Three effects compound. `x50` halves (468 → 202 m). The achieved wind is closer to
north (353° against 335°, Ekman turning 7–13° in a well-mixed CBL against 22–25° neutral), so
the chord is 252 m against 143. And the array has 1.6× the heat flux. The lake runs the
opposite way (15.33% → 5.32%). The compact convective footprint does not reach it, and water
generates almost none of the thermals a convective footprint is made of. Predicting the share
from the chord and the LES's own `f_y` reproduced the measured attribution to 0.6% and 1.4%
(47.74 vs 48.02%, 8.79 vs 8.67%). Kljun predicts 28.31% for the northerly. It under-predicts
by 1.7× because its footprint is broader than the convective LES's.

Convective windows converge faster: half-vs-half centroid difference 15–90 m against
152–436 m neutral, peak identical between halves in three of four. The sampling time a case
needs is stability-dependent and the neutral cases set it.

## The sub-grid gate, and what the floor contributes

| | neutral flat | convective flat |
|---|---|---|
| sub-grid fraction of `σ_w²` at 30 m | 85.5% | **52.3%** |
| `Δ` needed for the 40% gate | ≲ 8.6 m | **≲ 14.4 m** |
| floor factor at the receptor | 1.45× | **3.37×** |
| `σ_w` deficit vs similarity | 1.02 vs 1.25 `u*` (18% low) | 1.06 vs 1.56 `u*` (32% low) |

Convection nearly halves the resolution requirement, and the floor does more, not less,
because similarity asks for much more. That exposed a modelling freedom. Lenschow's mixed-layer
relation asks 1.26 `u*` where Panofsky asks 1.56. The two agree to 1% in the free-convection
limit and differ in the transition where a 30 m receptor under a 900 m CBL is.
`--sgs-most-mode {surface, blend, mixed}` made the choice explicit. `blend` must never be the
default because as `w* → 0` the mixed-layer target vanishes and would switch the floor off
just short of neutral. One convective flat window scored four ways on identical fields:

| anchor | peak | `x50` | 80% area | overlap 80/50 | integral |
|---|---|---|---|---|---|
| none | 264 m | 611 m | 54.3 ha | 29% / 31% | 0.887 |
| mixed-layer (Lenschow) | 216 m | 460 m | 33.4 ha | 44% / 50% | 0.906 |
| **surface-layer (adopted)** | **192 m** | **292 m** | **14.8 ha** | **59% / 62%** | 0.934 |
| Kljun on the same cells | 192 m | | 26.0 ha | | 0.881 |

The choice is not within noise (shape L1 66.1% with no floor, 46.5% with the mixed-layer anchor,
against a 38% overlap floor). The adopted anchor matches Kljun's peak exactly on the one
configuration where Kljun is diagnostic. Without the floor the footprint is 3.7× too large.
Stated as a limitation: the near-field convective footprint is anchored to surface-layer
similarity and the tail is free. The "46–66% anchor-sensitivity band against a 38% sampling
floor" quoted with every near-field number comes from here.

## What was left

The 24 m vs 12 m convergence test. Sweep `z_i`. Compensate Ekman turning or label by achieved
direction. A stability-dependent sampling time. The forward bulge. The fifth pass moved instead
to a 10 m receptor on a 16 m grid. The [seventh](pass-7.md) came back to a 30 m receptor.
