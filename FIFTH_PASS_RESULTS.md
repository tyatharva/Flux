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
