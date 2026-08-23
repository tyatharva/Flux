# Fifth pass — 10 m receptor on a 122 x 122 x 122 @ 16 m grid

Rebuilt around the corrected instrument height. The grid was chosen for **corpus
economics**: an 8-case campaign costs ~12 GPU-h here against 42 at `186^2 @ 10 m`, because
targets are needed in quantity and per-target precision is secondary. What that buys and
what it costs are both recorded below.

Absolute distances from earlier passes do not carry over. Methodology, traps and closure
findings do.

---

## 1. The configuration

| | value | how it was fixed |
|---|---|---|
| `Nx x Ny x Nz` | 122 x 122 x 122 | `(N+6) = 128 = 2^7` in all three |
| `dx = dy` | 16.0 m -> 1952 m domain | given |
| `d_zeta` / `verticalDeformFactor` | 20.576132 / 0.194059 | `bin/vgrid.py`, solved from FastEddy's `zDeform` |
| `dz_sfc` | 3.9933 m | receptor exact at 10.000000000 m on `k = 2` |
| block | `1 x 2 x 64` | fastest of 9 legal shapes, 0.01475 s/step |
| `dt` flat | 0.0146417 s (`CFL_3d` 1.35) | measured accuracy boundary 1.51, 10% margin |
| `dt` terrain | **the same** | measured; terrain did not lower it here |
| cost | 0.0149 s/step -> **0.94-0.99 GPU-h / sim-h** | measured |
| `z/Delta` at the receptor | **0.99** | sub-grid gate stays retired |
| taper `pad` | 12 cells (192 m) | measured knee; real terrain reaches 784 m |

`bin/vgrid.py` reproduces the retired 24 m grid exactly (`d_zeta` 24.691358, factor
0.346601, `k = 3` at 30.000000 m), which is what licenses trusting it here.

---

## 2. Gate results

| gate | verdict | number |
|---|---|---|
| **A1** water share | **PASS** | worst case **0.01%** over every direction x stability, vs 10% |
| **B1** grid launch | **PASS** | clean, 0.0149 s/step |
| **B2** thread blocks | **PASS** | `1x2x64` at 0.01475 s/step; `1x32x8` 46% slower |
| **B3** flat `dt` | **PASS** | boundary **~1.51**, not the ~1.64 recorded at 24-30 m |
| **B4** terrain `dt` | **PASS** | clean to CFL 1.40; no slope bin rings |
| **B5** restart injection | **PASS** | `topoPos` 9.5e-7, `z0m` 1.5e-9, `htFlux` exact |
| **B6** 90-deg equivariance | **PASS** | rotation exact to 1.2e-14; mean wind agrees to 1.7e-5 |
| **B7** subsidence | **PASS** | theta warms at 1.10x the prescribed rate (after a source fix) |
| **B8** halo check | **PASS** | 122 x 122 x 122 interior. **The CNF raster is 122 x 122.** |
| **C1** stationarity | **PASS** | `U/u*` +0.03 %/h; Kljun `x_peak` +0.06 %/h |
| **C2** restart resume | **PASS** | the read is bit-for-bit on every field |
| **D1** well-mixed | **PASS** | backward rms **4.26%**, forward 4.12%, vs a 5.48% counting floor |
| **D2** the integral | **PASS** | LES **0.914** vs Kljun **0.956** on the identical box |
| **D3** error floor | **PASS**, narrowly | half-vs-half 80% overlap **43%** vs LES-vs-Kljun **33%** |

---

## 2b. Phase D — the flat/neutral control

One 2400 s window, 481 dumps, 8.4 GB, 315,700 particles released over exactly 30 minutes.

**Gate D1.** Backward and forward both pass, and they agree with each other (rms 4.26% and
4.12%, max 11.0% and 11.2%, lowest three bins 1.025 and 1.032). Agreement between the two
directions is the real content: a sign error in the reverse-time drift shows up as a
backward-only failure, which is exactly the bug the fourth pass found. Run with `--sgs-most`
and the displacement correction active, i.e. in the configuration footprints are actually
computed in.

**Gate D2.** The integral converges **from below** with the wrap cap on -- 0.775 within
0.25 L, 0.868 within 0.5 L, 0.901 within 0.75 L, **0.914** overall -- and Kljun evaluated on
the identical cells gives **0.956**. So the LES captures 95.6% of what the analytic model
captures on the same box. At a 30 m receptor the comparable pair was 0.888 vs 0.875, i.e.
the LES exceeded Kljun; here it sits just below, which is the direction finite `t_back`
should push it.

**Gate D3, and a caveat about which metrics to trust.** The half-vs-half 80% overlap is
**43%** against an LES-vs-Kljun overlap of **33%** -- above it, so the metric is not at its
own noise floor, but by 10 points where the 24 m grid separated by 22. Related and worth
stating: **`x80` is not a reliable statistic at this grid.** Its half-vs-half floor is
**203 m**, 54% of `x80` itself, because the tail is carried by rare large-weight touchdowns
(the largest single weight is 170, and the top 0.1% carry 9.6% of the total). Quote the peak
and the 80% source AREA; treat `x80` as indicative only.

**`t_back` = 600 s, measured not assumed.** The capture curve, free from masking the same
touchdowns on age:

| `t_back` | integral | % of the 600 s value | peak | `x80` |
|---|---|---|---|---|
| 100 s | 0.654 | 71.5% | 64 m | 155 m |
| 200 s | 0.778 | 85.1% | 64 m | 240 m |
| 300 s | 0.832 | 91.0% | 64 m | 302 m |
| 400 s | 0.873 | 95.5% | 64 m | 344 m |
| **500 s** | **0.904** | **98.9%** | 64 m | 369 m |
| 600 s | 0.914 | 100% | 64 m | 378 m |

**The peak is converged at every `t_back` down to 60 s.** Convergence is set by the integral,
which reaches 98.9% at 500 s; production takes 500 x 1.25 = **600 s**, so a production window
is 2400 s and fits in ONE sub-1-hour segment.

**Backward transit from the 10 m receptor: median 59 s**, p25 28 s, p75 133 s, p95 443 s.
At 30 m the median was 180-290 s. The `z/sigma_w` scaling predicted 60-95 s; measured 59 s.

**The OBSERVABLE converges well before the integral does.** The integral keeps collecting
far-field tail; the array sits within 250 m of the tower. Tracking the land-cover share on
the same age mask:

| `t_back` | array share | vs its 600 s value |
|---|---|---|
| 60 s | 48.14% | +19.3 pts |
| 100 s | 38.96% | +10.1 pts |
| 200 s | 32.70% | +3.9 pts |
| 250 s | 31.31% | +2.5 pts |
| **400 s** | **28.94%** | **+0.11 pts** |
| 600 s | 28.83% | -- |

The array share is inside its own **5.07-point** half-vs-half sampling floor from `t_back`
~200 s and converged to 0.1 points by 400 s, while the integral needed 500 s to reach 98.9%.
Production keeps 600 s because the window fits in one segment either way and it is the
measured value with margin -- but the sizing is now known to be set by the tail, not by the
signal.



**Ensemble convergence at the 10 m receptor -- the corpus design parameter.** From 12
independent 150 s sub-windows of the one integration (lag autocorrelations |r| <= 0.28
against a 2/sqrt(12) = 0.58 independence threshold, so the sub-windows are independent as
assumed), scored against a held-out reference half over 400 random draws:

| n sub-windows | sampling time | peak p90 | centroid p90 | 80% overlap |
|---|---|---|---|---|
| 1 | 2.5 min | **0 m** | 525 m | 32.5% |
| 2 | 5.0 min | 0 m | 256 m | 38.2% |
| 3 | 7.5 min | 0 m | 154 m | 41.7% |
| 4 | 10.0 min | 0 m | 116 m | 44.9% |
| **5** | **12.5 min** | **0 m** | **80 m** | 47.2% |
| 6 | 15.0 min | 0 m | 43 m | 47.8% |

**The peak is converged in a single 2.5-minute sub-window** and the centroid reaches 100 m
by 12.5 minutes. Compare the 30 m receptor on the 24 m grid: the peak needed 12.5 min to
reach one cell, and **the centroid never reached 100 m at all** in the measurable range
(336 m at 22.5 min). The footprint is ~3x smaller here and transit times are 3-5x shorter,
so each sub-window holds far more independent realisations. **The 30-minute averaging period
eddy covariance requires is already well past what convergence needs**, so windows are sized
by `t_back` and the EC definition, never by sampling noise.

---

## 2c. Phase C — the convective base states

**Gate C3, shallow (`L = 4 z_i` target): PASS.**

| | value | expected |
|---|---|---|
| `z_i` | **428 m** | -> `L/z_i` = **4.56**, so the rule genuinely holds |
| `w*` | 1.236 m/s | |
| `u*` | 0.482 m/s | |
| `w*/u*` | **2.56** | above ~2 = free convection dominates |
| entrainment ratio | **0.223** | ~0.2 |
| `sigma_w/w*` vs Lenschow | 0.96-1.02 through `z/z_i` = 0.2-0.5 | ~1 |
| `z_i/L` | -6.7 | CONUS404 midday p50 -19.8 |
| `T* = z_i/w*` | 346 s | |

`z_i` is still growing at +79 m/h, which is entrainment, not drift -- a CBL has no
stationary depth. Over a 40-minute window that is +53 m, taking `L/z_i` from 4.56 to 4.06,
still inside the rule. The achieved value is what the case is labelled with.

Note the resolved surface heat flux at the first level is only 0.0077 of the prescribed
0.1363 K m/s -- the rest is sub-grid. That is `z/Delta ~ 1` again, and it is why
`bin/cbl_check.py` must read the PRESCRIBED `htFlux` rather than the resolved covariance:
using the resolved value would make this real CBL look like it was not one.

**Gate C3, deep (`L = 2 z_i` target): PASS.** `z_i` = **857 m** -> `L/z_i` = **2.28**,
`w*` = 1.557 m/s, `u*` = 0.500, **`w*/u*` = 3.11**, entrainment ratio **0.172**.

So the pair the adequacy test actually runs at is **`L/z_i` = 4.56 and 2.28** -- a clean
bracket of the rule, with identical surface heat flux (0.1363 K m/s virtual) and `z_i`
separated by the capping inversion and subsidence alone.

### The deep case IS locked in, and it is not a marginal call

Measured directly, without reference to the footprint -- the 2-D spectrum of `w` at
mid-depth:

| case | `z_i` | `L/z_i` | peak wavelength | **mode-1 share of the variance** |
|---|---|---|---|---|
| shallow | 458 m | 4.26 | 976 m (**mode 2**) | **2.2%** |
| deep | 907 m | 2.15 | **1952 m = L exactly (mode 1)** | **54.5%** |

Over half the mid-depth `w` variance in the deep case sits in the single largest mode the
box can hold, against 2% in the compliant one -- a **25x** difference. The box has no room
for the thermals' natural scale, so the energy collects in the one mode available.

A second, independent signature agrees: `sigma_w/w*` runs **1.13-1.29x Lenschow at
`z/z_i` = 0.5-0.7** in the deep case against 1.02-1.14 in the shallow one. The locked
circulation concentrates variance at mid-depth, which is exactly where mixed-layer
similarity is supposed to hold.

**This makes the footprint comparison MORE informative, not less.** The 4 `z_i` rule was
written for mixed-layer similarity, and lock-in is precisely its failure. The separate
question -- whether that reaches a footprint at 10 m, where surface-layer scaling governs
and Kljun's own `z_i` channel spans one percentage point of array share -- is now being
asked with the artifact demonstrably PRESENT, so agreement would be evidence rather than a
vacuous pass.

---

## 2d. Phase E — domain adequacy. THE DECISION EXPERIMENT

Two convective windows in the SAME 1952 m box, identical in `dx`, `dz`, `t_back`, surface,
closure and **surface heat flux** (0.1363 K m/s virtual). `z_i` separated by the capping
inversion and subsidence alone, so `u*` and `L` are close and the box is what differs.
Achieved: **`L/z_i` = 4.56 (shallow) and 2.28 (deep)** -- a clean bracket of the rule.

**The artifact is present, and it is not subtle.** In the adequacy windows themselves:

| case | `z_i` | `L/z_i` | peak wavelength | **mode-1 share of mid-depth `w` variance** |
|---|---|---|---|---|
| shallow | 458 m | 4.26 | 976 m (mode 2) | **4.8%** |
| deep | 907 m | 2.15 | **1952 m = L exactly** | **50.2%** |

**And the footprint does not notice.**

| observable | shallow | deep | difference | tolerance |
|---|---|---|---|---|
| peak | 64 m | 64 m | **0 m** | 16 m (one cell) |
| centroid | 116.4 m | 114.2 m | -2.2 m | 39 m |
| `x80` | 164.2 m | 177.0 m | +12.9 m | 44.6 m |
| **array share** | 38.63 +/- 5.96% | 36.76 +/- 7.52% | **-1.88 pt** | 6.07 pt |

Over 10 independent release groups per case: difference **-1.878 points, SE 3.034,
t = -0.62, p ~ 0.54**. The difference is **0.25x one window's own sampling sd**. A bias
smaller than the noise on each individual corpus target cannot be learned as a systematic
error by anything trained on those targets.

**GATE E: PASS.** `L >= 2 z_i` is not binding for a 10 m footprint. Convective-midday corpus
coverage goes **19.3% -> 60.9%** and `122^3` covers the corpus -- the `218^2` box (3.2x cost)
is not needed.

### The first answer was wrong, and why

Run with two halves per case, this gate said DIFFERS: a 1.215-point difference against a
1.19-point "floor", i.e. missing by 2% of tolerance. **That floor was an underestimate by a
factor of five.** It came from a single half-vs-half difference -- one degree of freedom --
while the array share's actual sampling sd over 10 release groups is **6.0-7.5 points**.
Splitting the release period into 10 groups instead of 2 costs nothing (the touchdowns are
already labelled by release time) and replaces one difference with a distribution.

The lesson is the project's own, in a new place: **a tolerance has to be measured the same
way the quantity is, and with enough degrees of freedom to mean something.** A gate that
compares a difference against another single difference is a coin flip dressed as a test.

---

## 2e. Phase F — production directions, neutral regime

Four directions from one spin-up by 90-degree re-index, each restarted onto the real static
surface with 20 min of adjustment before a 2400 s window. Labelled by ACHIEVED direction,
which is backed from the geostrophic forcing by Ekman turning and carried further by the
inertial oscillation -- none is a due N/S/E/W case.

| case | achieved dir | `u*` | `U` | `h` | **array share** | integral | vs Kljun (80% ovl) |
|---|---|---|---|---|---|---|---|
| wW | 239 deg | 0.379 | 3.86 | 443 m | **20.08%** | 0.926 | 27% |
| wS | 147 deg | 0.327 | 3.35 | 428 m | **45.79%** | 0.899 | 39% |
| wE | 47 deg | 0.338 | 3.49 | 559 m | **37.68%** | 0.803 | 33% |
| wN | 320 deg | 0.388 | 3.96 | 414 m | **42.13%** | 1.005 | 42% |

**Measured swing 20.1% to 45.8% = 2.3x**, against an array occupying **1.03%** of the
domain -- an enrichment of **19x at worst and 44x at best**. At a 30 m receptor the swing
was ~370x, because the array was then in the footprint only on northerlies. It is 2.3x here
because the tower is inside the array and sees it from every direction, which is the whole
change at this receptor height.

**Gate F -- the difference from Kljun is explicable, in the same direction every time.**
The share predicted from the array chord and the LES's own crosswind-integrated `f_y`:

| case | chord | PRED from Kljun `f_y` | PRED from the LES's own `f_y` | MEASURED |
|---|---|---|---|---|
| wN | 92 m | 55.68% | 47.49% | 42.13% |
| wS | 110 m | 61.49% | 52.98% | 45.79% |
| wE | 82 m | 51.44% | 40.74% | 37.68% |
| wW | 70 m | 45.98% | 26.63% | 20.08% |

Substituting the LES's own `f_y` for Kljun's moves the prediction toward the measurement in
**every** case, which is what it must do if the attribution is sound. And the **ratios agree
exactly** -- wN/wE 1.1x predicted vs 1.1x measured, wN/wS 0.9 vs 0.9, wS/wE 1.2 vs 1.2 --
which is the robust content, because the chords are only 4-7 cells at 16 m.

**Why the LES share is lower than Kljun's, quantified:**

| | LES | Kljun |
|---|---|---|
| peak of `f_y` | 64 m (all four) | 48 m |
| 80% source area | 2.2-4.3 ha | 1.6-1.7 ha |

The LES footprint is **40-150% broader in 80% source area** and its peak is one cell farther
out, so less of its mass lands on a patch that only reaches 70-110 m upwind. That is the
sign and size of the array-share gap, it is the same in all four directions, and it is the
near-field deficit the MOST-anchored `sigma_w` floor exists to bound -- at `z/Delta` = 0.99
the closure supplies most of `sigma_w`, so quote the **46-66% anchor-sensitivity band**
with any of these near-field numbers.

**A geometric point that matters for the corpus.** The array chord is capped by the array's
**120 m WIDTH**, not its 350 m length: a ray from the tower toward anything off due north
leaves through an east or west edge within ~110 m. That is why the directional swing is
modest at this receptor, and why **absolute share by direction, not the N-vs-E/W ratio, is
the discriminator.**

### Convective regime, and the residual the CNF exists to learn

| case | achieved dir | `u*` | `z_i/L` | **array share** | 80% area | integral |
|---|---|---|---|---|---|---|
| wN | 339 deg | 0.466 | -27.3 | **81.36%** | 1.2 ha | 1.180 |
| wS | 170 deg | 0.427 | -35.9 | **74.13%** | 1.0 ha | 0.997 |
| wE | 70 deg | 0.416 | -64.7 | **63.02%** | 0.7 ha | 0.777 |
| wW | 257 deg | 0.475 | -24.8 | **29.14%** | 1.5 ha | 0.768 |

**Convection roughly doubles what this tower measures of the array**: 1.45x to 1.93x by
direction, **mean 1.67x**. Swing across direction 2.8x, enrichment **28x to 79x** over the
array's 1.03% area share. On a convective northerly the array supplies **81% of the flux
from 1% of the domain.**

**And the sign of the disagreement with Kljun FLIPS between regimes, for one reason:**

| regime | LES 80% area / Kljun's | measured array share vs predicted |
|---|---|---|
| neutral | **1.29-2.69** (LES broader) | BELOW |
| convective | **0.31-0.77** (LES more compact) | ABOVE |

Kljun's own 80% source area barely moves between the two regimes -- 1.6-1.7 ha neutral,
1.9-2.2 ha convective -- and it moves the WRONG WAY. The LES says convection makes the
footprint **2 to 3 times more compact**; Kljun says slightly broader. **That is the residual
the emulator exists to learn**, and it is a structural failure of the analytic model at a
10 m receptor, not a tuning offset: no rescaling of Kljun's parameters changes the sign of
its regime dependence.

Convectively the chord prediction is also much sharper -- PRED-LES lands within 1-2 points
of measured on wN (83.4 vs 81.4) and wS (75.2 vs 74.1) -- because the compact convective
footprint sits well inside the array chord, so the 1-D chord approximation stops mattering.

**Water is 0.00-0.05% of every one of the eight production footprints**, confirming Gate A1
on real LES fields rather than on a Kljun estimate.

### 2f. The displacement-height sensitivity -- the paper result

Three treatments of the same convective northerly, **with the instrument held fixed at
10.0 m above BARE GROUND in all three** and the effective aerodynamic height matched
(8.50 vs 8.53 m), so the only thing that differs is how the array's surface is represented:

| treatment | array `z0` | `topoPos` | receptor | **array share** | vs baseline |
|---|---|---|---|---|---|
| baseline | 0.10 m | flat | `k = 2` | 81.36% | -- |
| bracket | 0.25 m | flat | `k = 2` | 82.26% | **+0.90 pt** |
| raised | 0.25 m | **+1.5 m over the array** | fractional, `z_agl` = 8.5 m | **84.12%** | **+2.76 pt** |

**The array's surface representation is worth 2.76 points of array share, 3.4% relative.**
Of that, +0.90 is the roughness contrast alone (`z0` 0.10 -> 0.25, i.e. 1.0x -> 2.5x against
cropland) and the remaining **+1.86 is putting the displacement height into the terrain**,
which is what lifts the first model level from 2.0 m above bare ground to 3.5 m -- clear of
panel top instead of inside it.

**And separately, the receptor's own effective height is worth more.** A fourth run placed
the instrument 11.5 m above bare ground (`z_eff` 8.53 -> 10.00 m) and the share fell
**1.85 points**. So a 1.5 m error in either the surface representation or the receptor
datum moves this number by 2-3 points in opposite directions.

**This is a LOWER BOUND on the sensitivity, and deliberately so.** It was measured on the
northerly, the direction with the LARGEST array share (81%), where the share is closest to
saturation and least sensitive. Kljun on the real map predicts the same 1.5 m of effective
height is worth **+8.3 points** on a neutral crosswise (E/W) direction, where the share is
29.9% and far from saturated. Quote 2.8 points as what was measured where it matters least,
and expect several times that crosswise.

**What this settles.** The array's footprint share was partly a modelling choice, and the
choice is now bounded rather than hidden. The `--raise-topo` treatment is the physically
better one -- it is the only configuration in which the first model level sits above panel
top and the array carries a real roughness contrast -- and it costs 2.76 points relative to
the `z0 = 0.10` workaround that PROJECT_BRIEF.md adopted when the first level was thought to be the
binding constraint.

---

## 3. What was found that changes the science

### 3.1 The lake has left the study, and it costs nothing

Measured on the real WorldCover map, not estimated: the 1952 m box contains **8 water cells
of 14,884 (0.05%)**, and the worst-case footprint water share over every direction and
stability class is **0.01%**. At a 30 m receptor the LES measured 35.2% of a neutral
easterly footprint as water. PROJECT_BRIEF.md's estimate for the 10 m receptor was 2.5-3%; the
truth is two orders below that, because the box shrank as well as the footprint.

### 3.2 The array share is larger, and the directional ratio smaller, than the idealised table said

PROJECT_BRIEF.md's array shares were crosswind-INTEGRATED fractions along a line from the tower.
**The tower sits inside a 2-D rectangle**, so flux arriving from crosswind angles still
lands on the array. On the real map at `z_m = 10 m`:

| stability | E/W | S | N | **N/E ratio** |
|---|---|---|---|---|
| very unstable | 43.6% | 58.1% | 70.4% | 1.61x |
| neutral | 29.9% | 55.5% | 80.6% | **2.69x** |
| stable | 20.3% | 45.9% | 72.9% | 3.59x |

Shares are 1.4-1.6x the idealised estimate, so the **ratio Gate F leans on falls further** --
3.7x -> 2.69x neutral, against ~370x measured at 30 m. **Gate F must use absolute share by
direction, not the ratio.**

### 3.3 Displacement height was absent and is first-order

Kljun at `z_m = 10.0 -> 8.5 m` moves the array's E/W share **29.9% -> 38.2% (1.28x)** and
`x90` 701 -> 596 m; the effect reaches **+8.4 percentage points** in stable conditions. `d`
now enters the LPDM sub-layer log law, the MOST-anchored `sigma_w` floor, and Kljun's `z_m`.
The receptor datum is **10 m above bare ground**, so over the array the effective height is
`z - d ~ 8.5 m` and a raised-terrain treatment must release at a fractional level.

### 3.4 The z_i cap is expensive AND biased

`L >= 4 z_i` caps `z_i` at 488 m and covers only **19.3%** of convective-midday hours;
`L >= 2 z_i` covers **60.9%**. The excluded deep-CBL hours carry **1.51x the surface heat
flux** and **1.58x the `w*`** (rank correlation `z_i` vs `w'th'` = **+0.43**), so a capped
corpus is thinnest exactly where the array's flux enhancement is strongest. Whether the cap
binds for a 10 m FOOTPRINT is measured separately (Phase E).

### 3.5 The array's heat flux was the wrong quantity

PROJECT_BRIEF.md requires `htFlux` be the **virtual** flux because the run is dry. The fourth pass
prescribed CONUS404's **sensible** flux and applied sensible-flux ratios to it. The
conversion `w'th_v' = w'th'(1 + 0.0735/B)` is Bowen-ratio dependent and therefore
class-dependent: renormalised to cropland, the **array multiplier falls 1.60 -> 1.376** and
**water rises 0.12 -> 0.151**. The array's own value barely moves (0.176 -> 0.178 K m/s --
the two corrections nearly cancel) but the **array-to-water contrast falls ~32%**, and that
contrast is what the directional signal is made of.

### 3.6 At `z0_array = 0.10 m` the array is aerodynamically invisible

WorldCover labels the array as cropland, whose `z0` is **also 0.10 m**. The override
therefore changes nothing, and the array's entire NEUTRAL signal is zero -- only the
convective heat-flux contrast distinguishes it. `prep_surface.py` now warns when the two
coincide. The `--raise-topo` treatment restores the contrast (2.5x) while keeping the first
model level above panel top; which is right is a measured sensitivity, not a decision.

---

## 4. Bugs found

### 4.1 In FastEddy: subsidence is unusable dry (fixed on the fork)

`lsf_horMnSubTerms = 1` with `moistureSelector = 0` dies on the first timestep with an
illegal memory access. `cuda_lsfSlabMeans()` launches the qv slab-mean over
`moistScalars_d` and `cudaDevice_lsfRHS` writes `Frhs_qv`, both unconditionally, while
`cuda_moistureDeviceSetup()` allocates them only inside `if (moistureSelector > 0)`.
**Upstream v5.0.1 subsidence works only with moisture on.** Both guarded on `kegonsa`.
Same class as the `NORHO` bug, differing only in whether the bad pointer trapped or
produced `inf`. `FASTEDDY_TRAPS.md` section 10.

### 4.2 In our analysis: every footprint would have been computed at 30 m

`stage5_footprint.py` never passed `z_target` to `compute_footprint`, so it fell through to
the 30.0 default and `receptor_indices` picked the level nearest 30 m -- with nothing in the
output to say so. Four other hard-coded 30 m receptors went with it.

### 4.3 In the driver: Gate D1 was never run

`regression_flat.sh` deleted the window fields before the well-mixed test could see them,
so the gate the plan calls non-negotiable was silently skipped.

### 4.4 In the driver: the convective stages used the neutral base file

The deep-CBL adequacy window would have run the shallow case's subsidence profile (peak at
500 m instead of 1000 m) -- in a pair that exists precisely to differ in nothing but `z_i`.

---

## 5. Three gates that were specified wrong

Worth keeping, because each would have rejected a correct configuration.

**C1 gated on `u*`.** A doubly-periodic neutral Ekman layer forced by a constant
geostrophic wind does not settle to a fixed `u*` on any affordable timescale: `f` gives an
**inertial period of 17.6 h**, `u*` fell for ~4.4 h (a quarter period) and then rose,
reaching +6.3 %/h at 6.26 simulated hours. Damping it needs several periods -- 35-50
simulated hours for ONE base state -- and a real boundary layer does not do it either.
Kljun's `Pi_4 = u(z_m)/u*` is a **ratio**, and both terms ride the oscillation together:
`U/u*` moves **+0.03 %/h** while its numerator and denominator each move +6.3 %/h. The
derived `x_peak` spans 38.0-38.3 m against a **16 m** raster cell. The gate now tests the
footprint's own controlling parameters; the mean-flow drift is carried as a per-case label,
which is what `window_stats` already does.

**C2 demanded bit equality over a whole segment.** FastEddy is chaotic in fp32, so two
IDENTICAL re-runs diverge too. Measured: chain-vs-re-run and re-run-vs-re-run agree to
8-14% on the max and 3% on the rms over 20,000 steps -- the restart adds nothing above the
solver's own nondeterminism. The testable claim is the restart READ, and it is **bit-for-bit
on every prognostic and surface field**.

**B4's check was structurally blind.** `docker/k0k1_check.py` averages over the whole plane,
but terrain amplification is local and only 1.7% of this domain exceeds slope 0.14 -- a few
ringing columns cannot move a 14,884-cell mean. `bin/k0k1_by_slope.py` conditions on slope,
and the result stands up: `k0/k1` rises monotonically 0.42 -> 0.68 with slope, which is
resolved motion over topography. Acoustic noise is a ratio near 9, and it is nowhere.

### 2g. The remaining Gate F discriminators, and the sub-grid report

**Sub-grid fraction of `sigma_w^2` at the receptor -- REPORTED, NOT GATED** (the 40% gate is
retired; it is unreachable by ~2 orders of magnitude at any affordable grid):

| state | sub-grid fraction at `z/Delta` = 0.99 | 40% crossing |
|---|---|---|
| neutral | **96.5%** | `z/Delta` ~ 3.58 |
| convective shallow | **91.4%** | `z/Delta` ~ 2.42 |
| convective deep | **90.7%** | `z/Delta` ~ 2.43 |

At the 24 m grid and a 30 m receptor these were 85.5% and 52.3%. **The closure now supplies
over 90% of `sigma_w^2` at the receptor**, which is the context for §5b: when the model
supplies nine tenths of the quantity, a defect in how it is applied is not a detail.
Reaching 40% would need `Delta <~ 2.8 m` neutrally, i.e. `dx ~ 3 m`.

**Gate F discriminator 2 -- the upwind roughness transect.** Produced for both regimes:
`figures/g16_nbl_transect_wN_wS_wE_wW.png` and `figures/g16_cbl_transect_wN_wS_wE_wW.png`,
the footprint against the surface it came from, out to 900 m.

**Gate F discriminator 3 -- terrain response, array cells excluded.** Kljun has no terrain,
so any systematic relation between the non-array footprint mass and the ground under it is
structure the analytic model cannot produce:

| case | dir | footprint-mean terrain | along-wind slope | r(f, slope) |
|---|---|---|---|---|
| nbl_wN | 319 | -1.03 m | **+0.0129** | +0.177 |
| nbl_wS | 147 | **-12.11 m** | **-0.0120** | -0.187 |
| nbl_wE | 47 | -1.20 m | +0.0198 | +0.259 |
| nbl_wW | 239 | -9.55 m | -0.0082 | -0.184 |
| cbl_wN | 339 | +0.41 m | +0.0170 | +0.105 |
| cbl_wS | 170 | -10.29 m | -0.0168 | -0.102 |
| cbl_wE | 70 | -3.98 m | +0.0251 | +0.080 |
| cbl_wW | 257 | -10.26 m | -0.0206 | -0.127 |

Domain-mean terrain is -3.54 m. **The footprint-mean terrain spans 12.5 m across
direction** -- northerlies and easterlies sample ground ~2.5 m above the domain mean,
southerlies and westerlies 6-8 m below it. The along-wind slope **flips sign consistently
between opposing directions**, as it must for a fixed terrain field, and `r(f, slope)`
follows it. The correlations are modest (|r| 0.04-0.26) but their SIGN tracks the wind in
all eight cases, which is the content: the footprint is sitting on real topography and
responding to it.

---

## 5b. THE PRINCIPAL OPEN FINDING: the sigma_w floor is not well-mixed convectively

Found at the end of this pass, by asking a question the plan did not ask.

**The neutral well-mixed gate says nothing about the convective closure, because the floor
is a different closure in each.** Measured floor factor at the receptor:

| case | factor at the receptor | factor over the column |
|---|---|---|
| flat/neutral control | **1.000** (INACTIVE) | 1.00-2.45 |
| convective, all cases | **1.57-1.68** | 1.00 to **12-26** |

The fourth pass recorded convective well-mixedness as "inherited" from the neutral test.
That inheritance is invalid: neutrally the floor does nothing, so the neutral PASS is a
test of the unmodified model.

**Run in the convective closure, Gate D1 fails one direction:**

| direction | max abs(ratio-1) | rms | lowest 3 bins | verdict |
|---|---|---|---|---|
| BACKWARD (what footprints use) | 13.67% | 7.51% | 1.059 | PASS |
| FORWARD (control) | 31.27% | 13.28% | **1.258** | **FAIL** |

A correct Lagrangian stochastic model in a stationary field is well mixed in BOTH
directions. This asymmetry is not the sign error PROJECT_BRIEF.md warns about (that would fail
backward and pass forward); it is the taper.

**The mechanism, measured.** The floor factor is not monotone in height:

| z (m) | resolved `ww` | sgs (2/3)e | **floor factor** | `sigma_w^2` | d/dz |
|---|---|---|---|---|---|
| 2.0 | 0.0012 | 0.3662 | 1.05 | 0.3865 | +0.0095 |
| 26.3 | 0.1694 | 0.0820 | 5.00 | 0.5795 | +0.0064 |
| **52.1** | 0.3758 | 0.0360 | **9.45** | **0.7159** | +0.0028 |
| 61.3 | 0.4333 | 0.0321 | 8.77 | 0.7145 | **-0.0014** |
| 91.5 | 0.5708 | 0.0268 | 3.79 | 0.6725 | **-0.0015** |
| 114.3 | 0.6341 | 0.0256 | 1.00 | 0.6597 | +0.0021 |

The factor peaks at **9.45 at 52 m -- exactly the taper's inner edge** (`0.1h` = 54 m) --
then falls to 1 as the taper switches it off by `0.2h`. That manufactures a **spurious
`sigma_w^2` MAXIMUM at the taper edge**, with `d(sigma_w^2)/dz < 0` at 10 of the 26 levels
below 120 m. `sigma_w^2` must increase away from an impermeable wall; where the floor makes
it decrease, Thomson's drift points inward from both sides and particles converge on the
artificial maximum. The scored "lowest 3 bins" span 2-62 m and CONTAIN that maximum, which
is what the 1.258 excess is.

**The observable consequence.** Convective footprint integrals saturate ABOVE 1 -- 1.022 and
1.040 on flat ground, where a correct estimator converges to ~1 from below as the neutral
control does (0.914). They do SATURATE (flat beyond 1.0 L), so this is not the periodic
wrap-around double counting PROJECT_BRIEF.md describes; it is the closure. **Read the convective
array shares with a systematic uncertainty of order the integral overshoot, 2-4%.**

**The fix is a closure change and is NOT made here.** The floor must be applied so that the
transported `sigma_w^2` stays monotone in the surface layer -- e.g. by tapering the TARGET
rather than the factor, or by clipping the factor so the product never turns over. That
needs its own well-mixed validation in both directions, and making it silently at the end of
a campaign would invalidate the eight production footprints already computed. It is the
first thing the next pass should do.

**Neutral results are unaffected** -- the floor is inactive there, and the neutral control
passed both directions (4.26% and 4.12% rms).

---

## 6. Known limitations

1. The receptor may be inside the **roughness sublayer** over the array; MOST does not hold
   there, so Kljun is not a reference over the array and the `sigma_w` floor is extrapolated.
2. The **first model level is 1.997 m**, at or below panel top. The array's surface exchange
   is parameterised, not resolved.
3. **`z/Delta` = 0.99** at the receptor, worse than 1.76 at 24 m. The near field is
   closure-dominated; quote the 46-66% anchor-sensitivity band with any near-field number.
4. **Tree cells have `ln(z_first/z0) = 0.69`** -- 23.5% of the box, where the surface-layer
   scheme has almost no room. Inherent to `dx = 16 m` with a 10 m receptor.
5. The footprint **peak sits 1.7-5.7 cells** from the tower, bounding how sharply the CNF
   target can represent it.
6. **Real terrain reaches ~784 m**; land cover is real to the seam.
7. **Deep convective boundary layers are constrained** -- see 3.4 and Phase E.
