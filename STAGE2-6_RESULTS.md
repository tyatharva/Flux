# Stages 2-6 at 30 m — pipeline validation

**One FastEddy run producing one backward-LPDM flux footprint.** That was PLAN.md's goal;
this document records whether it happened and what each gate returned.

Everything here is at the **30 m pipeline-development grid**, not the 10 m production grid.
That is deliberate: this configuration validates the pipeline, not the science, and the
corpus is regenerated at finer resolution later. Where a number is resolution-dependent it
is flagged as such.

Supersedes the earlier `STAGE2_RESULTS.md`, which documented a 10 m spin-up at
`dt = 0.0275` — a timestep since shown to be above the accuracy-CFL limit.

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
accuracy-CFL rule in CLAUDE.md.

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
across the domain, matching CLAUDE.md's "~30 m of elevation change across the area".
Full provenance in `data/README.md`; it is one constant in `bin/prep_stage6.py`.

---

## SETUP-6 — Upstream issue ✅ FILED

[NCAR/FastEddy-model#134](https://github.com/NCAR/FastEddy-model/issues/134) — "dt has an
accuracy limit BELOW its stability limit; between them near-surface resolved w is silently
replaced by grid-scale noise", with the 10 m bisection table, the `Example01_NBL` margin
(CFL_3d 1.603 against an onset near 1.64), and a suggested startup diagnostic plus warning.

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
the answer is known: every bit of surface flux crosses the measurement height, so
`int f_flux dA -> 1`.

---

## Stage 3 — Output configuration ✅ PASS, by configuration alone

PLAN.md's gate: *"field selection + fp16 on write puts a 30-min window under ~30 GB."*
At 30 m the gate is met **without either**, and therefore without touching FastEddy source.

A dump carries 10 3-D fields (`xPos yPos zPos rho u v w theta pressure TKE_0`) at fp32 with
`hydroSubGridWrite = 0`, which drops the 9 SGS stress fields. That is **40 B/cell**:

```
  657,000 cells x 40 B/cell            =  26.6 MB per dump      (measured: 26,644,621 B)
  30-min window at 5 s cadence, 360    =   9.6 GB per window
```

against the ~30 GB gate — a factor of 3 of headroom. The 213 GB figure in PLAN.md was the
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

