# Fourth pass — true 30-minute footprints, a static raster, and the convective regime

Supersedes `THIRD_PASS_RESULTS.md` for everything about the estimator, the window length
and the corpus design. The grid, the surface and the sub-grid closure are unchanged from
the third pass and are not re-litigated here.

Four things the third pass left open are closed, and one new regime is added.

---

## 1. A window is (30 min + `t_back`), and the fork had to change to allow it

**The third pass's footprints were 15-minute footprints wearing a 30-minute label.** A
backward trajectory cannot be released until it has `t_back` seconds of stored field
behind it, so the first `t_back` of any window produces nothing. With `t_back = 900 s` a
30-minute window yields 15 minutes of releases — and 15 minutes is below what the
ensemble-convergence curve says the centroid needs, which is why the terrain cases showed
27-43% half-vs-half overlap against 53.6% flat.

The averaging period stays 30 minutes. That is what eddy covariance means and it is not
ours to move; the **LES window** is what has to grow.

**That ran straight into the 45-minute per-run ceiling.** At `186 x 186 x 122` the terrain
`dt` gives 1.28 s of wall clock per simulated second, so a 2700 s window is 57.5 minutes —
and it could not be split, because lean `ioLPDMmode` output is deliberately not restartable
(`rho` and `pressure` are absent by construction).

`ioLPDMfullFrq` on the `kegonsa` fork fixes exactly that: any output whose **absolute**
timestep is a multiple of it is written in full upstream form — every registered variable,
fp32, coordinate geometry included — while every other dump stays lean and 16-bit packed.
Setting it to the segment length puts a restartable dump at each end of a chain link.

Verified before use, on 400 steps:

| | result |
|---|---|
| variable set and dtypes vs a mode-0 dump | identical |
| values vs an independent mode-0 run | `u` 3.4e-5, `w` 3.4e-4 relative — the nondeterminism floor |
| second segment restarts from it | yes, clean, no missing-variable errors |
| cost | one 170 MB dump per chain link against 42.7 MB lean, i.e. ~0.5% of a window |

`bin/run_window.sh` is the driver: it takes a window length, works out the segmentation
that keeps every segment under the ceiling, checks the projection before launching, and
preserves each boundary dump outside `window/` so a failed segment does not take the chain
point with it.

---

## 2. The footprint raster is now the LES grid

The third pass accumulated on a 60 m wind-aligned raster and then interpolated onto the map
for figures. Both halves of that were wrong for what comes next.

- The resample fell hardest on the **near field**, which is where the footprint peak sits
  and where the solar array is. It blurred exactly the part the result depends on.
- The emulator will consume a fixed north-up raster. Training it on a rotated-then-
  resampled array means training it on the resample.

Touchdowns are now binned by their **LES column index**, folded modulo the periodic domain,
so a footprint cell IS an LES column — the same indexing the land-cover masks use. Nothing
is rotated and nothing is interpolated.

The wind frame has not been abandoned; it has been made exact. The crosswind-integrated
footprint is a 1-D histogram of the touchdowns' own upwind coordinate at 24 m, and Kljun is
evaluated at the static cells' own coordinates (`lpdm.kljun.footprint_on_static`, 8x8
sub-sampled per cell because near the receptor `sigma_y` is smaller than one cell) rather
than rotated onto them. FFP is a closed-form function; interpolating it was gratuitous.

**A side effect worth stating: the integral now includes the wrapped touchdowns.** They
fold onto the same surface they came from, so they belong in it — where the old
4500 x 3000 m wind-frame grid simply dropped everything beyond its edge.

One implementation consequence. A 541-dump window is 55 GB of field cache at fp32 and
28 GB at fp16, on a 62 GB machine, so the cache had to become float16 — and
`scipy.ndimage.map_coordinates` refuses float16. The 4-D linear interpolation is therefore
written out by hand in `lpdm/fields.py`, gathering the 16 corners once and reusing the
index set across all six fields. It matches `map_coordinates` to float32 roundoff and is
marginally faster per field.

---

## 3. CONUS404 as a climatology, never as forcing

`bin/conus404_site.py` streams a stratified 45-year hourly sample at the tower cell
directly off the USGS Open Storage Network pod — anonymous S3 over plain HTTPS, no cloud
SDK, no credentials, no egress charge. 274 of 2740 time-chunks (6 contiguous days out of
every 60, so every month of every year is represented and the diurnal cycle is complete
within each block), 39,456 hourly records, ~30 GB streamed and discarded, 8 minutes.

**Nothing here forces a run.** No per-case sounding, no projection matching, no
time-varying boundary conditions. Each LES case stays one idealised quasi-stationary state.
CONUS404 only decides which states are worth GPU time, and how many of each.

Quality-controlled at `u* >= 0.15 m/s` (65.2% of hours):

| | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| `z_i` | 80 m | 267 m | 493 m | 835 m | 1475 m |
| `w'theta'` | -0.027 | -0.006 | +0.015 | +0.076 | +0.164 K m/s |
| `u*` | 0.17 | 0.24 | 0.32 | 0.44 | 0.65 m/s |
| `U(30 m)` | 2.4 | 3.9 | 5.2 | 6.8 | 10.0 m/s |

**Three things follow directly.**

1. **The site is unstable more than half the time** — 27.2% very unstable (`z/L < -0.5`),
   30.3% unstable, 13.3% near-neutral, 20.4% stable, 8.8% very stable. A neutral-only
   corpus misses the modal daytime state entirely, which is why this pass adds one.
2. **`z_i` must be swept, over a factor of 18.** Kljun takes `z_i` as an input; a corpus at
   one `z_i` leaves that input channel untrained and the emulator cannot learn what it does.
3. **The wind rose and the array signal point in different directions.** The rose is
   S 16.0%, NW 14.5%, W 14.4%, SW 14.3% against N 10.6%, E 10.4%, NE 10.2%, SE 9.8%. But
   the array is upwind only on northerlies, so N/NE/NW is where the site-specific skill
   has to come from. Direction sampling needs a **floor**, not pure rose weighting.

Derivations are all standard surface-layer relations and are written out in the script:
`ACSHFLSM` (kJ m^-2 accumulated over the prior hour) to `H` to `w'theta'` through
`rho = PSFC/(R_d T)`; `u*` and `L` solved jointly from `U10` with `z0 = 0.05 m`. The
grid-relative to earth-relative wind rotation (`COSALPHA`/`SINALPHA`, -5.55 deg here) is a
whole direction bin and is applied.

---

## 4. The convective regime

The third pass ran neutral only. CONUS404 says that is the wrong single regime to have: at
this site 27.2% of quality-controlled hours are very unstable and another 30.3% unstable,
against 13.3% near-neutral. So the convective boundary layer is the modal daytime state,
and the corpus has to contain it.

`runs/g24_base/base_cbl.in` — an idealised dry CBL, cold start, every number taken from the
CONUS404 midday distribution rather than guessed:

| | value | source |
|---|---|---|
| `surflayer_wth` | **0.11 K m/s** (127 W/m2) | CONUS404 p50, local 10-16 h, `w'theta' > 0.05` |
| mixed layer | 0-800 m at 300 K | CONUS404 midday `z_i` p50 = 859 m |
| capping inversion | 800-900 m at 0.08 K/m (+8 K) | |
| free atmosphere | 0.004 K/m | |
| `U_g` | 10 m/s, unchanged from neutral | so the ONLY difference is thermal |
| derived | `w* = 1.42 m/s`, `T* = 562 s`, `z_i/L = -18` | CONUS404 midday p50 `z_i/L` = -19.8 |

Spin-up is 5400 s = **9.6 `T*`**, chained as three 33-minute segments.

**Holding `U_g` fixed at 10 m/s is a design choice, not laziness.** The neutral corpus was
forced the same way, so the neutral-to-convective difference is attributable to the thermal
forcing alone rather than confounded with a wind-speed change. The achieved `u*` and
`U(30 m)` are measured and reported per case.

**`z_i` grows at about 148 m/h by entrainment and that is not drift.** A convective boundary
layer has no stationary depth; `bin/cbl_check.py` reports the achieved `z_i`, `w*`, the
entrainment ratio (expected ~0.2) and `sigma_w/w*` against Lenschow et al. (1980) rather
than against NCAR's neutral validation case, which says nothing about a CBL.

### Surface heat flux is per-cell, from the land cover

`prep_surface.py --wth` gives `htFlux` a per-class map. With `surflayer_idealsine = 0`
FastEddy's surface-layer kernel comments "reuse *htFlux array values" and never overwrites
what the restart injected, so this needs no source change — the same restart-injection
lever that already carries `z0m`.

| class | multiplier | `w'theta'` (K m/s) |
|---|---|---|
| cropland (reference) | 1.00 | 0.110 |
| tree / grass / shrub | 1.10 | 0.121 |
| built-up | 1.50 | 0.165 |
| **permanent water** | **0.12** | **0.013** |
| **solar array** | **1.60** | **0.176** |

**The water multiplier is directional, and that is the point.** Within 4 km the water is
almost entirely E and NE, so an easterly fetch is over a surface with a tenth of the land's
sensible heat flux — a real, strong, direction-dependent surface heterogeneity that the
neutral cases cannot express at all.

**The array multiplier retires an accepted omission.** CLAUDE.md lists the elevated heat
source as a known omission; with a per-cell `htFlux` it no longer has to be one. PV modules
are darker than the crop they replaced and do not transpire, so nearly all absorbed
shortwave that is not exported as electricity leaves as sensible heat, and field studies of
utility-scale arrays report a daytime enhancement of order 1.5-2. This is also the pathway
CLAUDE.md identifies for albedo: with no radiation scheme, `htFlux` is what albedo would
have controlled. So in the convective cases the array is a roughness patch **and** a heat
source; in the neutral cases it is roughness only. The contrast is deliberate.

### The sigma_w floor had to become stability-aware

The adopted `--sgs-most` floor was anchored to `sigma_w/u* = 1.25`, the **neutral**
surface-layer value. Under free convection the surface-layer value is larger and grows as
`(-z/L)^(1/3)`, so the neutral constant would anchor the correction to the wrong target in
exactly the regime that matters most.

    phi_w = (1 - 3 z/L)^(1/3)   z/L < 0     (Panofsky et al. 1977)
          = 1 + 0.2 z/L         z/L > 0     (Kaimal & Finnigan 1994)

`phi_w(0) = 1` exactly, so **every neutral result is unchanged bit for bit** — this is a
strict generalisation of the relation the floor was calibrated on, not a retune. At the
planned CBL it takes the receptor target from `1.25 u*` to `1.81 u*`.

The taper across `0.1h - 0.2h` is unchanged and matters more convectively than neutrally:
above it, CBL `sigma_w` scales with `w*`, not `u*`, and a surface-layer relation has nothing
to say there.

---

## 5. How much backward time a window really needs — and a bug the question exposed

The plan was to measure the capture curve on the flat/neutral control and size every
production window as (30 min + `t_back`). The measurement is exact rather than five
separate runs: one release ensemble, truncated at each `t_back` by masking on the
touchdowns' recorded age, so nothing but the backward integration limit differs.

**The first measurement said something impossible.** The flux-footprint integral over a
horizontally homogeneous surface must be 1, and it must approach 1 **from below** — a
finite backward time can only lose influence, never invent it. It did not:

| `t_back` | 300 | 450 | 600 | 750 | 900 |
|---|---|---|---|---|---|
| integral, **before the fix** | 0.747 | 0.882 | 0.968 | **1.031** | **1.089** |

It crossed 1 near 670 s and kept climbing. The peak, meanwhile, sat at 264 m for every
`t_back` from 150 s up — so whatever was growing lived entirely in the far tail.

**Ruled out first: wrap-around.** The integral decomposed by trajectory displacement
showed nothing at all arriving beyond one domain length (1.089 within 1.00 L, 1.089 within
2.00 L), and it already exceeded 1 within 0.75 L. The wrap cap was doing exactly its job.

**The cause was the `sigma_w` floor itself.** Thomson's (1987) reverse-time drift contains
`d(sigma^2)/dz`. `--sgs-most` rescales the sub-grid variance by a **height-dependent**
factor `sc(z)` running from 2.5 near the surface to 1 above `0.2h` — but the drift kept
using the unscaled gradient:

        used:      dsig2dz
        correct:   sc * dsig2dz  +  (2/3) e * dsc/dz

Both terms were missing and `dsc/dz` is the larger, comparable in magnitude to the LES's
own gradient. A drift that weak cannot balance the extra mixing the factor introduces, so
backward particles get unopposed vertical motion, touch down too often, and the integral
inflates **with integration time** — exactly the observed signature.

The third pass never saw it because its wind-frame raster dropped everything beyond
4500 m upwind and +/-1500 m crosswind; that truncation was cancelling the bias. Folding onto
the LES columns removed the truncation and left the bias exposed. It also means the third
pass's reading of the floor moving the integral 0.805 -> 0.882 "toward 1" was reading a bias
as an improvement.

**This is what the standing flat/neutral control is for**, and it found it on the first run.

### After the fix

| `t_back` | 300 | 450 | 550 | 600 | 700 | 800 | 900 |
|---|---|---|---|---|---|---|---|
| integral | 0.671 | 0.759 | 0.791 | 0.808 | 0.836 | 0.867 | **0.888** |
| peak | 312 | 312 | 312 | 312 | 312 | 312 | 312 m |
| `x50` | 496 | 566 | 591 | 603 | 622 | 636 | 645 m |
| shape L1 vs 900 s | 43.4% | 23.3% | 15.7% | 12.0% | 6.4% | 2.5% | — |

Monotone from below, as it must be. And the well-mixed gate, re-run **with the floor
active** — which it never had been — passes cleanly backward, the direction footprints
use: max deviation 5.83%, **rms 3.61% against a 5.48% counting-noise floor**.

### The answer, in two parts

**Shape converges by 450-600 s.** The half-vs-half difference of the same window is a
**29.4%** shape L1, so `t_back = 450 s` is already inside the sampling floor and 600 s is
comfortably inside it (12.0%). The peak does not move at all across the whole range.

**Magnitude does not converge, and cannot — it is a statement about the domain, not about
`t_back`.** At 900 s the integral is 0.888 and still gaining ~0.02 per 100 s. But Kljun,
evaluated on the *identical* 4464 m box, integrates to **0.875**: even an exact analytic
footprint only puts 87.5% of its mass inside this domain. **LES 0.888 against Kljun 0.875
is agreement to 1.5%** — so at 900 s the estimator has recovered essentially everything the
domain can offer, and the shortfall below 1 is the tail that lies outside the box.

That comparison is only available because Kljun is now evaluated on the same cells. It
replaces "the integral is 12% short" with "the integral is where a correct footprint
truncated to this domain has to be", which is a different and much more useful statement.

**Decision: `t_back = 900 s`, production windows 45 minutes.** The shape argument alone
would justify 40 minutes, but the integral is still gaining at 900 s and the difference
between the two is ~7 minutes of wall clock per case — about one hour across the whole
campaign, against an integral 9% closer to the domain-limited truth and directly
comparable to Kljun on the same box. The measurement is reported either way, so shortening
later needs no re-derivation.

### One residual, reported rather than waved through

The **forward** well-mixed control shows a systematic 12-18% bulge over 30-70 m
(rms 8.10%, max 17.87%). It passes its own criterion — 4 sigma is 21.9% — but the pattern is
four consecutive high bins, not scatter. Backward is clean. Footprints use backward, so
this does not invalidate them, but a genuinely well-mixed model should be well mixed in
both directions and this is not yet fully explained.

The transit-time check also prints FAIL, on a sub-criterion that is not in PLAN.md: the
script requires that more than half of the backward particles reach the surface within
`t_limit`, and 46.2% do. PLAN.md asks for a *plausible transit time* — 1-5 min unstable,
10-15 min stable, neutral between — and the median is **179 s = 3.0 min**, inside that
range. The median halving while the >900 s fraction fell is a coherent consequence of the
floor being surface-layer-only: below 121 m particles get extra mixing both downward (fast
touchdowns) and upward (escape into the LES's own deficient variance aloft, where they
linger). Recorded as a threshold that the fix tightened, not as a passing grade.

---

## 6. Stage 6, neutral — PASS, on the number the plan predicted

Four directions from ONE spun-up state by 90-degree re-indexing. Terrain, roughness and the
array are **bit-identical** in all four, so every difference is flow. 30 minutes of releases
each, `t_back = 900 s`, accumulated on the LES columns.

| case | wind from | **array** | x area | **water** | x area | peak | integral | wrapped |
|---|---|---|---|---|---|---|---|---|
| **wN** | 335 deg | **3.01%** | **13.9x** | 0.00% | 0.00 | 264 m | 0.838 | 2.7% |
| wS | 158 deg | 0.85% | 3.9x | 0.47% | 0.03 | 300 / **1080 m** | 1.065 | 2.8% |
| wE | 66 deg | 0.01% | 0.1x | **15.33%** | **0.95** | 192 m | 0.919 | 1.1% |
| wW | 247 deg | 0.01% | 0.0x | 0.00% | 0.00 | 288 m | 0.731 | -2.0% |

Array area share 0.22%, water 16.09%. Shares exclude periodically folded touchdowns.

**Array swing wN/wE = 215x, wN/wW = 368x.** PLAN.md predicted **~300x** from the geometry
alone, before any of these runs existed. The ordering is exact and follows the upwind reach:
north 250 m, south 100 m, east and west 60 m.

**The water is the mirror image on a different axis.** 15.33% of the footprint on an
easterly against 0.00% on a westerly and a northerly, tracking the lake's real E/NE
position. Two independent surface features, two independent directional signatures, one
fixed map.

### Made quantitative

`bin/stage6_predict.py` predicts each share from the array rectangle's upwind chord and
Kljun's cumulative footprint:

| pair | predicted | measured |
|---|---|---|
| wN / wE | 89.5x | **215x** |
| wN / wS | 2.8x | **3.5x** |
| wS / wE | 32.2x | **60.8x** |

The measured swing **exceeds** the predicted one in every pair, in the same direction each
time. That is a consistency check rather than a discrepancy: the prediction is built on
Kljun's near field, whose peak is at 192 m against the LES's 264 m, so a patch reaching only
65-143 m upwind loses proportionally more in the LES than a Kljun-based estimate says.

### The southerly is bimodal, and the surface says why

`f_y` for the southerly has a near lobe at 300 m and a **larger** far lobe at **1080 m**,
reproduced independently by both halves of the window, so it is structure and not sampling
noise. `bin/upwind_transect.py` puts `f_y` directly above the terrain, roughness and
land-cover class along the same ray, and the answer is one line:

- 300-600 m: **tree cover, `z0` = 1.00 m** against 0.10 for the cropland around it, sitting
  in the deepest part of a hollow (terrain -18 m)
- 800-1000 m: grass, then trees again
- 1080-1600 m: open cropland on ground that **climbs about 20 m**

Tall roughness in a hollow lifts backward trajectories over it instead of letting them
touch down; the rising ground beyond comes up to meet them. The dips in `f_y` sit on the
forest bands and the far peak sits on the rising open ground. The westerly, whose transect
is near-uniform cropland for 1200 m, is single-peaked exactly as Kljun expects.

**This is what the Stage 6 gate asks for** — a difference from Kljun you can point at on
the map — and it is a stronger form of it than the third pass reached, because the feature
is named rather than merely present.

### The sigma_w floor switches itself off over real terrain

Measured at the receptor in all four terrain cases: floor factor **1.000-1.002**, because
the LES already delivers `sigma_w/u*` of **1.20-1.24** against the surface-layer 1.25. On
flat uniform ground the same grid gives 1.02 and the floor lifts it to 1.20.

Real terrain generates resolved-scale vertical motion (resolved `sigma_w` 0.186-0.195
against 0.136 flat) that flat, uniform ground cannot, so in NEUTRAL conditions the floor
has nothing left to supply and stands aside. The four neutral terrain footprints are
therefore free of it entirely, and the "constrained vs free" caveat applies only to the
neutral flat control.

**That does not generalise to the convective cases, and it would have been easy to claim it
did.** Convectively the floor is 3.1x over the same terrain (section 7): the similarity
target rises with `phi_w` to 1.74 u* while the LES only reaches 1.16 u*, so the gap widens
even though the terrain is identical. The floor is driven by STABILITY, not by terrain:

| | flat | over terrain |
|---|---|---|
| neutral | 1.45x | **1.00x** |
| convective | 3.37x | **3.11x** |

### The integral straddles 1 with the sign of the mean vertical motion

wS 1.065 at `w_bar = +0.15 m/s`, wE 0.919 at -0.08, wN 0.838 at -0.15, wW 0.731 at +0.08.
The streamline rotation removes `w_bar` from the *weight* -- verified, 99-100% of it -- but
it cannot remove it from the *transport*. This is the advective non-closure that makes eddy
covariance hard over complex terrain, and it is a result rather than an estimator error.

---

## 7. The convective regime, and what it did to the sub-grid gate

### The spin-up validates against convective similarity

90 minutes = 9.6 `T*`, cold start, flat and uniform. `bin/cbl_check.py` compares against a
CBL's own structure rather than against NCAR's neutral case, which says nothing about one.

| | LES | expected |
|---|---|---|
| `w*` | 1.434 m/s | 1.42 predicted from `w'theta'` and `z_i` |
| `T* = z_i/w*` | 574 s | 562 |
| `z_i` | 823 m | 800 initial, entraining |
| `w*/u*` | **2.86** | > 2 means free convection dominates |
| **entrainment ratio** | **0.149** | 0.10-0.35 (Deardorff 1972) — **OK** |
| `sigma_w/w*` at `z/z_i` = 0.1 / 0.2 / 0.35 / 0.5 | 0.491 / 0.607 / 0.683 / 0.703 | Lenschow 0.572 / 0.659 / 0.679 / 0.641 |

Within 10% of Lenschow through the bulk and within 21% everywhere. Domain TKE flat over the
last three dumps.

**One deviation, stated rather than tuned away.** `z_i/L` came out **-9.3**, not the -18
designed for, because `u*` reached 0.502 rather than the assumed 0.40: a CBL with the same
geostrophic forcing mixes momentum down and raises `u*`. So the case lands in the site's
*unstable* class (30.3% of quality-controlled hours) rather than *very unstable* (27.2%).
Kept, because holding `U_g` fixed at 10 m/s is what makes the neutral-to-convective
difference attributable to the thermal forcing alone. Achieved values are reported
throughout, never requested ones.

**A diagnostic trap worth recording.** `cbl_check.py` first read the surface buoyancy flux
as the RESOLVED covariance at `k = 0`. At the lowest LES level almost all of the heat flux
is sub-grid — it enters as a surface-layer boundary condition — so that reads 0.0088 K m/s
against a prescribed 0.11. Every derived quantity then said "this is not a convective
boundary layer" about a boundary layer that is: `w*` three times too small, `L` twelve times
too long, `z_i/L` = -0.6 instead of -9.3, and `sigma_w/w*` at 2-2.7x Lenschow.

### The sub-grid gate: the predicted improvement, measured

| | neutral flat | convective flat |
|---|---|---|
| sub-grid fraction of `sigma_w^2` at 30 m | **85.5%** | **52.3%** |
| `Delta` needed to reach the 40% gate | <~ 8.6 m | **<~ 14.4 m** |
| resolved `w` variance at `k = 1` | 3.2e-3 | **2.75e-2** (8.6x) |

It falls, substantially, for the predicted reason: CBL `sigma_w` draws on resolved
`z_i`-scale thermals rather than `z`-scale eddies, and 24 m resolves a 900 m thermal far
better than it resolves a 30 m surface-layer eddy. **Convection nearly halves the resolution
requirement** — the gate needs `Delta <~ 14.4 m` instead of 8.6 m, i.e. about 1.2x finer
spacing rather than 2x. Still a FAIL at `Delta = 17.05 m`, but a much closer one.

### The floor does MORE convectively, not less — and why that is not a contradiction

Floor factor at the receptor: **3.37x** convectively, 1.45x on flat neutral ground, and
**1.00-1.002x over real terrain in all four neutral directions**. The expectation going in
was that better-resolved convection would need less. It needs more, and the reason is that
two different quantities were being conflated:

| | sub-grid FRACTION | `sigma_w` DEFICIT vs similarity |
|---|---|---|
| neutral flat | 85.5% | 1.02 against 1.25 u*, **18% low** |
| convective flat | **52.3%** | 1.06 against 1.56 u*, **32% low** |

Convection resolves a far larger *share* of the variance and still sits further below what
similarity asks for, because similarity asks for much more: the target rises from 1.25 u* to
1.56 u* while the LES only moves from 1.02 to 1.06.

**This also exposed a modelling freedom that had been implicit.** The floor is anchored to
Panofsky et al. (1977). Lenschow et al. (1980)'s mixed-layer relation asks for 1.26 u* where
Panofsky asks 1.56 u*. The two **agree to 1% in the free-convection limit** —
`1.803 kappa^(1/3) / 1.34 = 0.991` — and differ only in the transition, where Panofsky
retains a neutral 1.25 u* term carrying shear production and Lenschow has none. A 30 m
receptor under a 900 m CBL sits exactly in that transition.

`--sgs-most-mode {surface,blend,mixed}` makes the choice explicit and measurable. The
default stays `surface`, on the grounds that it is the more complete of the two where shear
is not negligible; **`blend` must never be the default**, because as `w* -> 0` the
mixed-layer target goes to zero and the minimum would switch the floor off just short of
neutral, where it is needed most.

### The convective flat control is the best Kljun agreement of the pass

| | LES | Kljun |
|---|---|---|
| peak | **192 m** | **192 m** — exact |
| 80% source area | 14.8 ha | 26.0 ha |
| 80% / 50% overlap | **64% / 68%** | — |
| integral | 0.902 | 0.875 |

Against 50% / 44% overlap for the neutral flat control. The LES footprint is *tighter* than
Kljun's, which is what a convective boundary layer should do: strong vertical mixing brings
influence down closer to the tower.

---

## 8. Stage 6, convective — PASS, and convection changes the answer completely

Same four directions, same fixed geography, same 90-degree re-indexing, from the convective
spin-up. Achieved winds land closer to the cardinal points than the neutral set because the
Ekman turning angle is smaller in a well-mixed CBL (7-13 deg against 22-25 deg).

| case | wind from | **array** | x area | **water** | x area | peak | `x50` | integral |
|---|---|---|---|---|---|---|---|---|
| **wN** | 353 deg | **48.02%** | **222x** | 0.00% | 0.00 | 144 m | **202 m** | 1.250 |
| wS | 177 deg | 8.67% | 40x | -0.02% | 0.00 | 168 m | 272 m | 0.693 |
| wE | 83 deg | 0.21% | 1.0x | **5.32%** | 0.33 | 168 m | 191 m | 1.047 |
| wW | 263 deg | 0.09% | 0.4x | 0.00% | 0.00 | 240 m | 339 m | 0.759 |

**Array swing 528x**, against 368x neutral.

### On a convective northerly the tower is measuring itself

**48% of the flux comes from the solar array** — from 0.22% of the domain. Neutrally the same
geometry gives 3.01%. Three effects compound and all three are physical:

1. **The footprint is far more compact.** `x50` halves, 468 m -> 202 m, because convective
   mixing brings influence down close to the tower.
2. **The achieved wind is closer to due north** (353 deg against 335 deg), so the array's
   upwind chord is 252 m against 143 m — nearly the full 250 m the rectangle offers.
3. **The array carries 1.6x the surface heat flux** of the cropland around it. That is the
   albedo pathway CLAUDE.md lists as an accepted omission, and it is no longer omitted.

### The lake runs the opposite way

15.33% of the neutral easterly footprint, **5.32% convectively** — a 2.9x reduction for the
same lake and nearly the same direction. Again two compounding effects: the convective
footprint does not reach as far into water that starts more than 1 km out, and water carries
**0.12x** the land's sensible heat flux, so it generates almost none of the thermals a
convective footprint is made of.

**Two surface features, two opposite responses to the same change in stability.** The array
gains an order of magnitude; the lake loses two thirds. Neither is put in by hand — both
follow from one per-cell `htFlux` map and the flow.

### The gate, made quantitative — and it closes to 1%

| case | chord | PRED from the LES's own `f_y` | **MEASURED** |
|---|---|---|---|
| wN | 252 m | 47.74% | **48.02%** |
| wS | 100 m | 8.79% | **8.67%** |
| wE | 60 m | 0.93% | 0.21% |
| wW | 60 m | 0.60% | 0.09% |

For the two directions where the array registers at all, predicting its share from the
rectangle's upwind chord and the LES's **own** crosswind-integrated footprint reproduces the
measured land-cover attribution to **0.6% and 1.4%**. Those are independent calculations —
one geometric, one accumulated from ~236,000 touchdowns in LES index space — so the
agreement says the attribution is real and not an artifact of how touchdowns are binned. The
60 m chords are 2.5 grid cells and the geometric estimate is not meaningful there.

Kljun predicts **28.31%** for the northerly against 48.02% measured: it under-predicts by
1.7x because its footprint is broader than the convective LES's. That is the gate — a
difference from Kljun, in a direction that is explained.

### Convective windows are cheaper to converge, which the corpus should exploit

Half-vs-half sampling floor, all eight production cases:

| | 80% overlap | peak difference | **centroid difference** |
|---|---|---|---|
| neutral, four directions | 37-54% | 0-168 m | **152-436 m** |
| convective, four directions | 43-51% | 0-24 m | **15-90 m** |

**The convective centroid is four to ten times better determined by the same 30 minutes of
releases.** The third pass established the centroid as the expensive metric — 336 m at p90
after 22.5 min — but that was measured on a *neutral flat* case. A compact footprint with a
short tail converges far faster, and the peak is identical between halves in three of the
four convective cases.

So the sampling time a corpus case needs is **stability-dependent**, and the neutral cases
set the requirement. That is worth knowing before buying GPU hours uniformly across the
sweep.

---
