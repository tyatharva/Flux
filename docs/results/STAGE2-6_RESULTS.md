# Stages 2-6 at 30 m — pipeline validation (FIRST PASS)

> **Superseded 2026-08-19 by `docs/results/STAGE2-6_RESULTS_V2.md`.** Everything site-specific in this
> document was produced at a *surrogate* tower coordinate and is **void**. The estimator
> results (well-mixed, Kljun comparison, error floor) were produced at the coarser
> `dz_sfc = 20 m` grid and are superseded. Kept for the diagnostic trail — in particular the
> first-pass measurements that motivated the regrid, and Items 1 and 2 of the second pass,
> which were deliberately measured on this grid because they are questions about the
> estimator rather than about resolution.


**One FastEddy run producing one backward-LPDM flux footprint.** That was docs/PLAN.md's goal;
this document records whether it happened and what each gate returned.

Everything here is at the **30 m pipeline-development grid**, not the 10 m production grid.
That is deliberate: this configuration validates the pipeline, not the science, and the
corpus is regenerated at finer resolution later. Where a number is resolution-dependent it
is flagged as such.

Supersedes the earlier `docs/results/STAGE2_RESULTS.md`, which documented a 10 m spin-up at
`dt = 0.0275` — a timestep since shown to be above the accuracy-CFL limit.

## Gate summary

| gate | result |
|---|---|
| SETUP-1 bitwise restart at 30 m | ✅ byte-identical re-dump |
| SETUP-2 grid divisibility, receptor on a cell centre | ✅ 152/56/96 by 4/4/16; k=1 at 30.000 m |
| SETUP-3 `dt` recalibrated, k0/k1 < 1 | ✅ CFL_3d 1.491, k0/k1 = 0.17 |
| SETUP-3b accuracy limit is grid-independent | ✅ same threshold at 10 m and 30 m |
| SETUP-4 k0/k1 as a standing check | ✅ in `check_run.sh` |
| SETUP-5 DEM resampled to 30 m | ✅ (tower coordinate is a documented surrogate) |
| SETUP-6 upstream issue | ✅ NCAR/FastEddy-model#134 |
| Stage 2 stationarity + restart | ✅ trends 0.6 and 0.9 sigma |
| Stage 3 storage under ~30 GB | ✅ 9.6 GB, by configuration alone |
| Stage 4 well-mixed + transit time | ✅ rms 3.4% vs 4.5% counting noise; median 3.2 min |
| Stage 5 Gate 1, Kljun agreement | ❌ **peak 310 m vs 198 m**; diagnosed as resolution |
| Stage 5 Gate 2, error floor | ✅ measured, and large enough to change the corpus plan |
| Stage 6 explicable difference | ⚠️ array share 9.7% -> 14.8%, but integral 1.64 |

---

## Configuration

| | |
|---|---|
| grid | 146 x 50 x 90, `dx = dy = 30 m`, `dz_sfc = 20.00 m`, top 2700 m |
| padded | 152 / 56 / 96 — divisible by the 4 / 4 / 16 thread block |
| `dt` | **0.0625 s**, `CFL_3d = 1.491` |
| forcing | neutral, `U_g = 10 m/s`, `z0 = 0.03 m`, doubly periodic, flat |
| receptor | **z = 30.000 m exactly**, cell centre k = 1, at (i, j) = (109, 25) |
| measured cost | **0.0066 s/step** |

`verticalDeformFactor = 0.662868` was solved for, not rounded: FastEddy's `zDeform` puts
the first cell centre at half a spacing, so `dz_sfc = 20 m` puts the *second* centre at
1.5 x 20 = 30 m, and the cubic term's 4 mm was absorbed into the factor. The receptor
therefore sits on a grid point rather than being interpolated to one.

`dt` was **recalibrated at this grid, not scaled from the 10 m value**, per the
accuracy-CFL rule in PROJECT_BRIEF.md.

---

## SETUP-1 — Bitwise restart, re-verified at 30 m ✅ PASS

The GPU overclock that killed the previous spin-up was reverted before this session; the
restart mechanism was therefore re-tested from scratch at the new grid, in 1000 timesteps
(8 s of wall clock), not by re-running any spin-up.

```
run A:  0 -> 1000 steps, dumps at 0 / 500 / 1000
run B:  restart from A's step-500 dump with Nt = 500 (ABSOLUTE, so zero timesteps), re-dump
        cmp A/FE_R.500  B/FE_R.500   ->  IDENTICAL   (26,644,621 bytes)
run C:  restart from A's step-500 dump with Nt = 1000, i.e. 500 further steps
```

| field | max abs diff, C vs A at step 1000 | relative |
|---|---|---|
| u | 2.651e-04 | 2.6e-05 |
| v | 2.731e-04 | 1.8e-03 |
| w | 3.989e-04 | 1.6e-03 |
| theta | 1.282e-03 | 4.0e-06 |
| rho | 4.768e-06 | 4.1e-06 |

State resume is **byte-exact**; the subsequent trajectory then separates at the known ~1e-4
nondeterminism floor. It is a true state resume, not a reinitialisation, and the
spinup-once / adjust-per-direction corpus structure remains valid at this grid.

---

## SETUP-3 — `dt` recalibration ✅ PASS

`CFL_3d = dt * c * sqrt(1/dx^2 + 1/dy^2 + 1/dz_sfc^2)`, `c = 347.2 m/s`:
`sqrt(sum) = 0.0687236`, so `c * sqrt(sum) = 23.861 /s` and `dt = 0.0625 s` gives
**`CFL_3d = 1.4913`** — 9% below the 1.64 accuracy limit and 17% below the 1.79 stability
limit.

Verified with the k0/k1 test on the spun-up state, where turbulence is developed and the
ratio is meaningful:

```
FE_S30.259200   k0/k1 = 0.174    ww[0]=1.00e-03  ww[1]=5.77e-03
FE_S30.288000   k0/k1 = 0.176    ww[0]=8.85e-04  ww[1]=5.02e-03
```

Physically `w` variance must GROW away from an impermeable wall, so the ratio must be < 1;
a value near 9 is the silent grid-scale-noise failure. 0.17 is clean, and matches NCAR's
NBL at 0.25.

---

## SETUP-4 — k0/k1 as a standing check ✅ DONE

`docker/check_run.sh` now scores the newest dump as well as the log, so every run tests the
**silent** failure mode alongside the loud ones (`CORRUPTED`, NaN, missing completion
banner). It reports **SKIP**, not FAIL, when `<w'w'>` at the second level is below 1e-5:
early in a spin-up the ratio is two near-zero variances and carries no information, and a
check that fails on undeveloped turbulence would be trained away rather than trusted.

`docker/run_case.sh` also now **refuses to start while another FastEddy container is
running**. Two runs sharing an `output/` directory interleave their dumps and corrupt both,
and it presents as a mysteriously stalled run rather than an error — which is exactly how it
presented once during this session.

---

## SETUP-5 — DEM ✅ DONE

`data/dem/kegonsa_30m_wtm.tif`, 267 x 267 at 30 m, EPSG:3071, from the 2024 Dane County
0.4572 m bare-earth LiDAR by `gdalwarp -r average` (**not** nearest: this is a 65.6x linear
reduction and nearest would alias individual LiDAR posts into LES surface elevations). No
SRTM anywhere in the pipeline.

**The tower coordinate in this repository is a documented surrogate, not the surveyed
position.** The first estimate, `-89.2450 / 42.9686`, reads 256.64 m in the DEM and lies 6 m
from open water — it is on **Lake Kegonsa**. 25% of the original tile is flat at
256.6 +/- 0.6 m and that flat surface's centroid, `-89.2504 / 42.9652`, matches the published
lake position, which is what identifies it. The coordinate now used, `-89.2539 / 42.9419`,
is the nearest position whose 4380 x 1500 m westerly domain contains **no water at all** —
an explicit rule rather than a guess. It sits 810 m from the shore with 35 m of relief
across the domain, matching PROJECT_BRIEF.md's "~30 m of elevation change across the area".
Full provenance in `data/README.md`; it is one constant in `bin/prep_stage6.py`.

---

## SETUP-3b — The accuracy limit is a property of CFL_3d, not of resolution

The 10 m bisection left one question open: is ~1.64 a threshold in CFL_3d, or an artifact of
that particular spacing? Three 300 s runs restarted from the spun-up state answer it.

| grid | `dt` (s) | CFL_3d | k0/k1 | `<w'w'>` level 0 | verdict |
|---|---|---|---|---|---|
| 10 m | 0.0272 | 1.636 | 0.27 | 6e-5 | clean |
| 10 m | 0.0275 | 1.654 | **8.94** | **1.97** | silent corruption |
| **30 m** | 0.0625 | 1.491 | 0.16 | 7e-4 | clean (production) |
| **30 m** | 0.0670 | 1.599 | 0.17 | 9e-4 | clean |
| **30 m** | 0.0713 | 1.701 | **7.44** | **577** | silent corruption |
| **30 m** | 0.0755 | 1.802 | **5.97** | — | silent corruption, still exits 0 |

`dx` changed by 3x and `dz_sfc` by 2x, and **the threshold stayed between CFL_3d 1.60 and
1.70**. It is the 3-D acoustic Courant number that matters, not the spacing — which is why
a user cannot escape this by choosing a "safe-looking" grid, and why `dt` has to be
re-derived from CFL_3d whenever the grid changes.

Note the last row: at 30 m, CFL_3d = 1.80 produces garbage but does **not** go NaN. The
*stability* boundary moves with the grid; the *accuracy* boundary does not. The gap between
them — the silent window — is therefore wider at coarser resolution, not narrower.

---

## SETUP-6 — Upstream issue ✅ FILED

[NCAR/FastEddy-model#134](https://github.com/NCAR/FastEddy-model/issues/134) — "dt has an
accuracy limit BELOW its stability limit; between them near-surface resolved w is silently
replaced by grid-scale noise", with the 10 m bisection table, the `Example01_NBL` margin
(CFL_3d 1.603 against an onset near 1.64), the 30 m confirmation table above, and a
suggested startup diagnostic plus warning. Offered a PR for the diagnostic.

---

## Stage 2 — Vertical stretching and spin-up ✅ PASS

Flat, uniform, neutral, doubly periodic, stretched vertical with a 540 m Rayleigh damping
layer. Run to **6 h of simulated time** in two restart-chained segments (90 min, then
90 min -> 6 h) so that no single run exceeded the 45-minute wall-clock limit: **9.6 min** and
**28.5 min**, against projections of 9.6 and 29.0 from the measured 0.0066 s/step.

### Gate 1 — statistical stationarity ✅

The transition overshoots: domain TKE climbs to 0.100 at 2 h, falls back, and settles near
0.045 from about 3.5 h onward.

```
  --- stationarity over the last 6 dumps (t = 210-360 min) ---
  domain TKE  mean   0.04551  scatter   7.4%   trend  +2.12 +/- 3.80 %/h   (+0.56 sigma)
  u*          mean   0.33260  scatter   1.6%   trend  +0.71 +/- 0.77 %/h   (+0.92 sigma)
  STATIONARY: YES  (both trends within 2 sigma of zero)
```

The criterion is a **trend test, not a difference of two dumps**. The original
growth-per-dump rule returned "NO" on this same converged series, because domain TKE in a
converged neutral boundary layer wanders by ~7% between dumps and a two-point rule measures
that scatter rather than drift — it also flips sign as the window slides. Fitting a line
over the tail and comparing its slope with its own standard error separates the two.
`docker/stage2_gate.py` was changed accordingly.

### Gate 2 — profile shape against NCAR's NBL validation case ✅ (with one expected offset)

| quantity | ours | NBL reference |
|---|---|---|
| `sigma_w^2` peak / `u*^2` | **0.526** | 0.730 |
| height of `sigma_w^2` peak (m) | 171 | 130 |
| `u*` (m/s) | 0.338 | 0.410 |
| wind speed, first level (m/s) | 4.90 | 4.30 |
| wind veering, surface -> free (deg) | **-20.2** | -25.0 |
| `sigma_w^2` -> 0 by (m) | **693** | 650 |

The **shape is right**: zero at the surface, a single peak at ~0.25 of the boundary-layer
depth, monotone decay to zero at the top. Veering and boundary-layer depth match closely.

The peak `sigma_w^2/u*^2` is **28% below** the reference, and this is expected rather than a
failure: it is the *resolved* variance, and at `dx = 30 m` with `dz_sfc = 20 m` a larger
fraction of the near-surface variance lives in the sub-grid than on NCAR's finer NBL grid.
It is the resolution being traded away, and it is one of the things that must be re-checked
at 10 m before any corpus is generated. It does not affect the LPDM, which is driven by
resolved **plus** modelled sub-grid motion, with the sub-grid part taken from FastEddy's own
`TKE_0`.

### Gate 3 — the restart is real ✅

Covered above under SETUP-1: byte-identical re-dump, then divergence at the nondeterminism
floor. The spin-up itself is the demonstration — it is two chained restarts, and its TKE
series is continuous across both joins with no transient.

---

## The LPDM

`lpdm/` is a backward Lagrangian particle dispersion model driven by interpolated FastEddy
fields plus a sub-grid Langevin model (Weil, Sullivan & Moeng 2004). Three decisions in it
are load-bearing enough to record.

**1. The reverse-time drift.** Reversing a Langevin model by substituting `(u,t) -> (-u,-t)`
gives an *anti-damped* velocity equation that diverges. That substitution is wrong: Thomson
(1987, section 5) shows the reversed diffusion picks up a term from the stationary density.
For `dX = A dt + B dW` with stationary density `p`, the reverse drift is
`A_hat = -A + (B B^T) grad_X ln p`. With `p` Gaussian in `u`,

```
A_hat_i = -(C0 eps / 2 sigma^2) u_i  -  (1/2)[ dsigma^2/dx_i + (u_i u_j / sigma^2) dsigma^2/dx_j ]
```

— the damping keeps its sign, and **only the sigma^2-gradient term flips**. Getting this
wrong either diverges (loud) or drops the gradient term and accumulates particles at the
surface (silent, and looks like a plausible footprint).

**2. `eps` is FastEddy's own, not a textbook constant.** `eps = c_e e^(3/2) / l` with
`l = min(0.76 sqrt(e)/N, Delta)` when `N^2 > 0` else `Delta`, `Delta = (dx dy dz J)^(1/3)`,
`c_e = 0.93` — read out of `SRC/HYDRO_CORE/CUDA/cuda_sgstkeDevice.cu` and recomputed at load
time. A Langevin model driven by an inconsistent `eps` fails the well-mixed test in a way
that looks like an integrator bug.

**3. Below the lowest LES level** (10 m) the LES carries no information. The horizontal wind
is continued by the neutral log law anchored at that level, resolved `w` goes to zero
linearly at the ground, `eps` follows surface-layer `1/z` scaling, and `sigma_s^2` is HELD
CONSTANT — so its gradient is zero there and the sub-layer is trivially well mixed and
cannot manufacture the accumulation the Stage 4 gate exists to detect. Touchdown is at 2 m.
This is a 30 m-grid approximation and is the first thing to revisit at 10 m.

FastEddy's output was verified to be **collocated and primitive** before any of this: every
field sits at the cell centre on the same index space as `zPos`, and `theta` reads 300.000 K
at the surface where `temp_grnd = 300.0`, `u` reads 10.0000 m/s where `U_g = 10.0`. No
de-densifying, no staggered offsets. `TKE_0` is the SGS TKE the LPDM needs.

### Estimator, and its unit test

Flux footprint (Flesch, Wilson & Yee 1995; Flesch 1996):

```
F/Q = (1/N) sum_i  w_release,i  *  sum_{touchdowns of i} 2/|w_td|
```

`w_release` is the vertical velocity the trajectory had **at the receptor** — resolved LES
value plus a sub-grid draw, i.e. the actual joint distribution, not an assumed Gaussian.
Trajectories arriving with `w > 0` came from below and carry surface-influenced air; those
with `w < 0` came from aloft. Their touchdown densities differ, and the signed difference is
the flux footprint — so the estimator must NOT use `|w_release|`.

`bin/test_estimator.py` checks the constant independently of any LES, by replacing the
fields with homogeneous turbulence (constant `U`, `sigma_s^2`, `eps`, zero gradients) where
the answer is known.

The **half-space** form of the test converges to 1 only slowly, because a finite backward
time truncates the far-field tail and biases the result low:

```
   t_back (s)   touchdowns  int f_flux   peak_x (m)
          300       23,598       0.713          150
          600       37,715       0.791          190
          900       48,477       0.718          170
         1800       72,781       0.917          150
  8 independent seeds at 1800 s:  0.914 +/- 0.035
```

The **closed-slab** form is the decisive one. Cap the slab with a reflecting lid and the
answer is no longer 1: with a surface source and no flux through the lid, the whole slab
accumulates uniformly, so the flux surviving to height `z` is `Q(1 - (z-z_td)/H)` — a
straight line from `Q` at the floor to zero at the lid. That is a **lid-dependent target**,
so a wrong constant cannot hide behind slow convergence.

```
   lid (m)  mix time (s)  predicted   measured    s.e.   sigmas
        60           313      0.600      0.530   0.068    -1.03
       100          1013      0.778      0.753   0.050    -0.49
```

Both agree, and the residual low bias tracks the mixing time (the 100 m slab is not fully
mixed within `t_back`). **The estimator constant is right**, verified without the LES,
without Kljun, and against a target that moves when the geometry moves.

---

## Stage 3 — Output configuration ✅ PASS, by configuration alone

docs/PLAN.md's gate: *"field selection + fp16 on write puts a 30-min window under ~30 GB."*
At 30 m the gate is met **without either**, and therefore without touching FastEddy source.

A dump carries 10 3-D fields (`xPos yPos zPos rho u v w theta pressure TKE_0`) at fp32 with
`hydroSubGridWrite = 0`, which drops the 9 SGS stress fields. That is **40 B/cell**:

```
  657,000 cells x 40 B/cell            =  26.6 MB per dump      (measured: 26,644,621 B)
  30-min window at 5 s cadence, 360    =   9.6 GB per window
```

against the ~30 GB gate — a factor of 3 of headroom. The 213 GB figure in docs/PLAN.md was the
10 m grid with all 19 fields; the grid is 11.8x smaller in cell count and the field list is
halved.

**The fp16-on-write work and the `io_binary.c` k-range contingency are therefore not
needed at this resolution and were not written.** They come back at 10 m, where the same
window is 113 GB. The arithmetic is unchanged; only the grid moved.

`TKE_0` is present and is the SGS TKE the LPDM's Langevin term needs, so no field the
pipeline requires is missing.

---

## Reproducing this

Everything runs through the container; the host needs only `rasterio`/`pyproj`/`netCDF4`
for the Stage 6 preprocessing.

```bash
# 30 m DEM tile from the county LiDAR (reads the tower coordinate from bin/prep_stage6.py)
./docker/prep_dem30.sh

# bitwise restart verification, 1000 steps, ~8 s
./docker/run_case.sh runs/s30_restart/a restart_a.in
./docker/run_case.sh runs/s30_restart/b restart_b.in
cmp runs/s30_restart/{a,b}/output/FE_R.500        # must be identical

# spin-up: segment 1 to 90 min, segment 2 to 6 h (chained by restart)
./docker/run_case.sh runs/s30_spinup seg1.in
./docker/run_case.sh runs/s30_spinup seg2.in
FE_DT=0.0625 ./docker/pyrun.sh docker/stage2_gate.py $(ls -v runs/s30_spinup/output/FE_S30.*)

# stages 3-6: sampling windows, decorrelation gap, terrain prep, adjustment, terrain window
./bin/run_pipeline.sh

# gates
./docker/pyrun.sh bin/test_estimator.py runs/s30_spinup/output      # estimator constant
./docker/pyrun.sh bin/stage4_wellmixed.py runs/s30_w1/output        # Stage 4
./docker/pyrun.sh bin/stage5_footprint.py runs/s30_w1/output runs/s30_w2/output --tag stage5
./docker/pyrun.sh bin/stage5_footprint.py runs/s30_stage6_smp/output --tag stage6_raw
./docker/pyrun.sh bin/stage6_compare.py results/stage5.npz results/stage6_raw.npz
```

Every FastEddy invocation goes through `run_case.sh`, which scores the log for
`CORRUPTED`/NaN/errors/completion **and** the newest dump for k0/k1, and refuses to start
beside another running FastEddy container.

---

## Stage 4 — LPDM, well-mixed first ✅ PASS

Released 60,000 particles uniformly over 2-1200 m in the flat/neutral window and integrated
900 s, scoring 2-400 m — the layer the footprint actually lives in.

| | backward (the mode footprints use) | forward (control) |
|---|---|---|
| max \|ratio-1\| | **8.64%** | 9.53% |
| rms | **3.39%** | 3.97% |
| lowest 3 bins | **1.001** | 1.016 |
| 1-sigma counting noise | 4.48% | 4.48% |

The rms departure from uniform is **below the 1-sigma counting noise**, and the near-surface
bins — the ones the gate exists for — are flat to 0.1%. Forward and backward agree, which is
what confirms the reverse-time drift sign: a sign error there shows up as backward-only
accumulation.

### The lid that wasn't the model's fault

The first version of this test closed the layer with a reflecting lid at 400 m. It gave a
clean profile everywhere **except a 2x pile-up in the single bin touching the lid — in both
time directions**, with the near-surface bins uniform to 0.5%:

```
       350.2            0.875
       370.1            0.924
       390.0            1.952   <-- the lid
```

That was the test's fault. Reflection flips the particle's sub-grid velocity but not the
**resolved** `w` interpolated from the LES, so a particle reflected at an arbitrary height
keeps its resolved upward motion and gets pinned against the boundary. At the real surface
this never arises, because impermeability takes resolved `w` to zero there — which is
exactly why the surface bins were clean while the lid bin was 95% off. The fix was to remove
the artificial boundary: release through a column reaching well above the boundary layer,
where FastEddy's SGS TKE is ~0 and particles sit still as a reservoir, and score only the
interior.

### Second gate — backward transit time ✅

```
  reached the surface within 900 s: 54.3% of 20,000 particles
  transit time (s): p5=43  p25=93  p50=191  p75=385  p95=743
```

Median **3.2 min** from the 30 m receptor to the surface. docs/PLAN.md expects 1-5 min unstable
and 10-15 min stable; neutral sits between, and it does.

---

## Stage 5 — First footprint, flat and neutral ❌ GATE 1 NOT MET at 30 m

The pipeline runs end to end and produces a footprint. It does not reproduce Kljun.

158,200 backward particles from the 30 m receptor, released at 226 times across the last
900 s of an 1800 s window at 5 s field cadence, giving 237,905 touchdowns.

### Gate 1 — agreement with Kljun ❌

| | LES + LPDM | Kljun FFP |
|---|---|---|
| peak upwind distance | **310 m** | 190 m (analytic 198 m) |
| centroid, upwind | 1053 m | 788 m |
| 80% source area | 18.6 ha | 24.1 ha |
| 80% area reaches | 4210 m | 1510 m |
| integral over the grid | 0.76 | 0.93 |
| 80% source-area overlap | **39.0%** | |

LES scalars driving both: `U = 6.15 m/s`, `u* = 0.341`, `h = 623 m`, `sigma_w = 0.392`
(= 1.15 u*), `sigma_v = 0.463`, neutral.

The crosswind-integrated profiles say what is actually wrong, and it is not the overall
shape — **the LES footprint is missing its near field**:

```
     x (m)    LES f_y  Kljun f_y   LES cum  Kljun cum
        90  3.590e-06  3.331e-04     0.000      0.008
       190  5.920e-04  1.606e-03     0.043      0.148
       290  9.194e-04  1.364e-03     0.154      0.310
       490  7.817e-04  7.645e-04     0.374      0.526
       790  4.210e-04  3.741e-04     0.555      0.696
      1990  9.989e-05  7.408e-05     0.843      0.908
```

Inside 190 m the LES has 4.3% of its influence against Kljun's 14.8%; the two cross near
490 m and beyond that the LES tail is consistently heavier. Too little influence close in,
compensated far out.

### What it is not

Two attempts at resolving it, each a controlled sweep from one field-cache load:

| case | peak_x | centroid_x | int f | top 0.1% of weight |
|---|---|---|---|---|
| baseline, C0 = 3 | 330 m | 947 m | 0.752 | 14.1% |
| **C0 = 6** (halves T_L) | 330 m | 1281 m | 0.638 | 17.5% |
| **C0 = 1.5** (doubles T_L) | 270 m | 829 m | 0.839 | 10.6% |
| **\|w_td\| floor 0.05** | 270 m | 953 m | 0.722 | 6.9% |
| different random seed | 330 m | 983 m | 0.772 | 12.6% |

- **Not the Langevin constant.** A **4x** change in `C0`, and therefore in the Lagrangian
  timescale, moves the peak by one 60 m grid cell. If the sub-grid closure's timescale were
  setting the disagreement, this sweep would have found it.
- **Not the estimator.** `bin/test_estimator.py` verifies the constant analytically against
  a lid-dependent target, with no LES involved.
- **Not the well-mixed condition.** Stage 4 passes below the counting noise.
- **Not Monte-Carlo noise.** A seed change moves the peak by zero cells.
- **Partly the touchdown-weight tail.** Flooring `|w_td|` at 0.05 m/s cuts the largest
  0.1% of weights from 14% of the total to 7% and moves the peak 330 -> 270 m. Worth about
  a fifth of the gap, and worth carrying as a reported diagnostic.

### What it is

At 30 m spacing the LES does not resolve the eddies that make the near-field footprint:

```
    z (m)   resolved   sub-grid  resolved %
     10.0    0.00090    0.14275        0.6%
     30.0    0.00558    0.14805        3.6%   <-- the receptor
     50.0    0.01556    0.09066       14.6%
     70.0    0.02893    0.05621       34.0%
    110.2    0.05536    0.02653       67.6%
    170.8    0.06761    0.01529       81.6%
```

**96.4% of the vertical velocity variance at the receptor is sub-grid.** The near-field
footprint is therefore almost entirely a product of the Langevin model rather than of the
LES — precisely the regime where a hybrid LES/sub-grid dispersion model has least to say,
and exactly what the `C0` sweep shows cannot be tuned away. A particle released at 30 m
must descend 28 m within ~30 s to land inside 190 m; with `sigma_w = 0.39 m/s` and
`T_L ~ 27 s` its typical vertical excursion in that time is ~16 m, so only a few percent
manage it. That is the 4.3%.

This is a resolution limit, not a pipeline defect, and it is the one gate that genuinely
requires the finer grid. At `dz_sfc = 10 m` the receptor sits at the third level rather than
the second and the resolved fraction there is far higher.

### Gate 2 — the irreducible error floor ✅ MEASURED (and it is large)

Two realisations of the same configuration, separated by 1200 s of decorrelation:

| metric | half vs half | realisation vs realisation | LES vs Kljun |
|---|---|---|---|
| peak difference | 40 m | **40 m** | +120 m |
| centroid difference | -4 m | **+53 m** | +265 m |
| 80% source-area overlap | 30.0% | **32.4%** | 39.0% |
| normalised L1 difference | | **91.9%** | |

This is the gate's whole point, and it pays off immediately: **the 80% source-area overlap
metric is close to its noise floor.** Two realisations of one configuration overlap 32%,
while LES-vs-Kljun overlap 39% — barely distinguishable. Per-cell footprint values differ
by 92% in L1 between two realisations of the SAME configuration.

Peak and centroid, by contrast, ARE resolved above the floor: 40 m and ~50 m of
realisation-to-realisation scatter against 120 and 265 m of difference from Kljun.

**Consequences for the corpus and for scoring the emulator, both load-bearing:**

1. **Do not score the emulator on per-cell footprint error or on 80% source-area overlap**
   at this sampling. Those metrics are noise. Score on peak location, centroid, and
   crosswind-integrated shape, which are stable.
2. **A single 30-min window does not converge a per-cell footprint.** The two 15-min halves
   of one window overlap each other no better than two independent realisations do, so the
   window is already long enough that lengthening it is not the fix — the fix is averaging
   over realisations, or accepting a smoothed footprint.

docs/PLAN.md asked for this measurement **before** committing to a corpus size. It changes the
answer: the corpus must be sized by the number of *runs* needed to average the footprint
down, not by the number of samples.

---

## Stage 6 method — how the real surface gets in

Two products, both from `bin/prep_stage6.py`, and one non-obvious mechanism.

**Rotation is entirely in preprocessing.** The DEM is resampled bilinearly onto a frame
whose `+x` is the direction the wind blows *toward*, so the LES x axis is the mean-wind axis
by construction and FastEddy needs no code change. Terrain has its domain mean removed
(elevations here are ~277 m ASL, which would otherwise consume 10% of the domain depth) and
is tapered to that mean plane over 12 cells in x and 8 in y with a raised cosine — without
the taper the periodic wrap is a cliff.

**The restart file carries the surface, not just the state.** `hydro_coreInit()` runs at
`FEMAIN/FastEddy.c:157` and the restart read at line 221 — *after* — and the read walks the
entire registered variable list, which includes `xPos`, `yPos`, `zPos`, `topoPos` and
`z0m`. That has one trap and one lever:

- **Trap.** Restarting a flat spin-up with a `topoFile` set leaves the solver with correct
  terrain-following metrics (built in `gridInit`, before the read) but silently overwrites
  the diagnostic `zPos`/`topoPos` in every later dump with the *flat* values from the
  spin-up file. The LES is right and its output coordinates are wrong — and the LPDM reads
  `zPos`, so it would place every particle at the wrong height with nothing to show for it.
- **Lever.** The same path is the only way to give FastEddy v5.0.1 a spatially varying
  roughness: `z0m` is a 2-D field but is initialised uniformly from the scalar
  `surflayer_z0` (`hydro_core.c:1379`) and has no input of its own. Writing it into the
  restart file works.

So the preprocessing writes terrain, terrain-following `zPos`, and the roughness map into
the restart file, making the read a no-op and keeping grid metrics, output coordinates and
the LPDM consistent. **No FastEddy source change was needed for any of Stages 2-6**, and the
`kegonsa` fork branch still has an empty diff against upstream v5.0.1.

Verified in the output rather than assumed. In the adjustment run's dumps:

```
  topoPos   -11.83 .. 20.79 m        (vs prep's array, max |diff| 9.3e-07 m)
  z0m         0.030 .. 0.200         (vs prep's array, max |diff| 3.0e-09 m)
  zPos[0]    -1.79 .. 30.71 m ASL    (terrain-following, top flat at 2700.0 m)
  receptor cell (109,25): ground 5.81 m, level k=1 at 35.746 m ASL = 29.935 m AGL
```

That last line is why the LPDM had to be corrected to release at
`fs.height(k, i, j)` rather than at the flat-column height: over this terrain the
receptor level is at **35.7 m ASL**, and releasing at 30 m ASL would have put it 6 m too low
— and 20 m underground at the top of the hill. Kljun's `z_m` is likewise the height **above
ground**, 29.935 m, not 30.

**Solar array**, per PROJECT_BRIEF.md's bulk-patch specification: `z0` 0.03 -> 0.20 m over a
100 m (along-wind) x 400 m (crosswind) rectangle centred 200 m upwind of the tower.
FastEddy has no displacement-height input, so `d = 1.2 m` is represented the only way the
model can feel it — by raising the effective surface over the patch. No explicit panel
geometry, per PROJECT_BRIEF.md: row spacing is ~5-7 m and a 30 m grid cannot resolve it.

---

## Stage 6 — Real surface ⚠️ DIFFERENCE EXPLICABLE, ABSOLUTE NORMALISATION NOT TRUSTED

Spun-up flat state restarted onto the wind-aligned terrain and roughness map, 1200 s of
adjustment, then an 1800 s sampling window at 5 s cadence. Wind from 270 deg. All runs
clean: no `CORRUPTED`, no NaN, `k0/k1 < 1` throughout.

The mechanism is verified in the model's own output, not assumed — terrain, terrain-
following `zPos` and the roughness patch all survive the restart read and round-trip to the
preprocessing arrays to float32 precision (numbers in the method section above).

### The gate — is the difference explicable?

| quantity | flat | real surface | Kljun |
|---|---|---|---|
| peak upwind distance (m) | 310 | **550** | 190 |
| centroid, upwind (m) | 1053 | **1457** | 788 |
| centroid, crosswind (m) | +3.8 | **-39.4** | 0 |
| 80% source area (ha) | 18.6 | 18.2 | 24.1 |
| **integral over the grid** | 0.76 | **1.64** | 0.93 |
| **solar array's share of the footprint** | **9.7%** | **14.8%** | 18.2% |

**Yes, and it points at what was put in.** The array patch — `z0` 0.03 -> 0.20 m plus a
1.2 m displacement, 180-270 m upwind — takes **1.52x** the share of the footprint it does
over flat uniform ground (9.7% -> 14.8%). That is the right sign and the right place:
rougher surface, more surface exchange per unit area, larger footprint weight, localised
exactly where the patch is. The crosswind centroid also shifts -39 m, consistent with the
terrain being asymmetric about the tower line.

The peak moves 310 -> 550 m upwind. That is a larger shift than roughness alone explains,
and the terrain is the rest of it: the receptor sits at 5.1 m of local elevation in a domain
with 32 m of relief, with the ground falling away upwind.

### Why the absolute footprint is not trusted, and it is not a subtlety

The terrain footprint integrates to **1.64**, and a flux footprint cannot exceed 1.

This is the residue of a real problem that terrain exposed and flat terrain hid completely.
An eddy-covariance tower measures `<w' c'>`, the departure from the mean — and over this
terrain the receptor sits in persistent subsidence:

```
  mean w at the receptor over the sampling window
    flat      -0.0038 m/s   =  0.07 of its own standard deviation   (negligible)
    terrain   -0.0963 m/s   =  1.5  standard deviations              (dominant)
```

Weighting by raw `w` there mixed the mean advective flux into the turbulent one, and the
estimator's two halves cancelled: `+3.484` and `-3.506`, netting `-0.022` instead of ~1, with
a footprint whose centroid came out at **-5412 m**. Subtracting the window mean (Reynolds
decomposition, which is what the instrument does) fixes the sign and the shape — centroid
-5412 -> +1457 m, 80% area 0.64 -> 18.2 ha — but leaves the normalisation at 1.64.

That is not yet a settled question, and both candidate explanations are live:

1. **Physical.** Over sloping terrain the turbulent flux at the measurement height genuinely
   is not the surface flux; advective flux divergence is the well-known non-closure problem
   of eddy covariance in complex terrain. An integral away from 1 may be the correct answer.
2. **Estimator.** Subtracting `w_bar` adds a term `-w_bar` times the CONCENTRATION integral,
   and the concentration integral is not bounded — it grows with backward integration time.
   That makes the correction's magnitude depend on `t_back`, which a physical answer must not.

Test 2 first: it is decisive and cheap. Recompute with `t_back` at 600 and 1200 s. If the
integral moves with it, the correction is the artifact.

Until that is settled, **the Stage 6 footprint is usable as a difference against the flat
case and not as an absolute footprint.** The array's 1.52x enrichment is a ratio of two
quantities computed identically, so it survives a common normalisation error; the 1.64 does
not.

---

## Where this leaves the plan

**Stages 2, 3 and 4 pass.** The spin-up is stationary and NBL-shaped, the storage gate is met
by configuration alone, and the LPDM is well mixed to below counting noise in both time
directions with a physically plausible transit time.

**Stage 5 Gate 1 does not pass at 30 m, and the reason is the grid.** 96.4% of the vertical
velocity variance at the receptor is sub-grid, so the near-field footprint is manufactured
by the Langevin closure rather than resolved by the LES — and a 4x sweep in `C0` moves the
peak by one grid cell, which is what rules out tuning as a fix. docs/PLAN.md is right that this
gate must pass before Stage 6 means anything; it needs `dz_sfc = 10 m`, where the receptor
sits at the third level instead of the second.

**Stage 5 Gate 2 pays for itself immediately** and changes the corpus arithmetic:

- Per-cell footprint error and 80% source-area overlap are **at the noise floor** and must
  not be used to score the emulator. Peak location and centroid are resolved and must be.
- Two 15-min halves of one window disagree as much as two independent realisations, so
  lengthening the window is not the fix — averaging over realisations is.

**Stage 6's machinery is complete and verified**, including the two things that were not
obvious: that a spatially varying `z0` is reachable through the restart file with no source
change, and that the restart read silently overwrites `zPos`/`topoPos` if you do not.
