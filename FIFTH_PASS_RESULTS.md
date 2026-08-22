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
