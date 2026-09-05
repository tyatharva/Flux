# Stages 0 to 2: toolchain, timestep and the first pipeline

August 17–19, 2026. The project began as a staged plan: build FastEddy (stage 0a), run the
production grid once (stage 1), spin up and validate a flat neutral boundary layer (stage 2),
bring the output under a storage budget (3), verify the backward LPDM is well mixed (4),
compare a first footprint with Kljun (5) and put the real surface in (6). Everything here
was on a 10 m or 30 m pipeline-development grid with a **10 m or 30 m receptor at
`dz_sfc = 20 m`, then 8.56 m**, at a surrogate tower coordinate for the first pass. None of
the absolute numbers carry to production. The mechanisms and the traps do.

## Stage 0a (2026-08-17): the toolchain, and four corrections to assumptions

FastEddy `kegonsa` at `e0cd2f3` (upstream v5.0.1, zero divergence) built in the CUDA 11.8
image in about 2 minutes and ran NCAR's `Example03_SBL` (GABLS1) and `Example01_NBL` to
completion with the published stratification reproduced (SBL: 267.80 K at 378 m against a
spec of 267.78. NBL: 300.00 K through the mixed layer, inversion aloft). Per-cell compute
cost agreed to 1.7% across a 12× range in grid size (9.45 vs 9.30 ns per cell per step), so
extrapolation was sound. NBL at 23.5 M cells used 6.4 GiB of 16.

Four assumptions were corrected by measurement rather than reading:

1. **CUDA-aware MPI is not required.** FastEddy passes raw device pointers to `MPI_Isend` /
   `Irecv` under `hydroBCs = 2`, which looks as if it demands it. A from-source CUDA-aware
   OpenMPI 4.1.6 was built and compared against stock 4.1.2 on the SBL case. Differences at
   step 200 were the model's own nondeterminism (rho 3.70e-6 vs 3.81e-6, theta 7.0e-4 vs
   6.4e-4).
2. **MPI Fortran bindings are required** despite no Fortran source: 60 `MPI_INTEGER` and
   `MPI_CHARACTER` broadcasts across 8 files. `--disable-mpi-fortran` aborts with
   `MPI_ERR_TYPE`.
3. **FastEddy is not bitwise reproducible**: about 1e-4 relative in velocity after 200 steps
   from one binary in one image.
4. **Precision is hardwired fp32**: bare `float` on every prognostic field, `MPI_FLOAT` in the
   halo exchange, `NC_FLOAT` in the writer.

Two flags were raised. A 3 h run on the intended 434 × 146 × 122 grid at 10 m projected to
8.15 h wall (over the plan's 4 h threshold). The output volume, with no vertical-subset
option in FastEddy (`io.c` reads only `ioOutputMode`, `inFile`, `outFileBase`, `frqOutput`,
`towerIOSelector`), was 76.6 B per cell: 213 GB for a 30-minute window at 5 s. The
`nvidia-cuda-toolkit` package in the host notes (CUDA 11.5) predates `sm_89`. The 11.8 image
is the floor. `gcc-13` in the notes belongs to MPAS work. nvcc 11.8 needs gcc ≤ 11.

## Stage 1 (2026-08-18): the production grid runs, and `dt` has no headroom

434 × 146 × 122 at 10 m, block `4 × 4 × 16`, flat, neutral, one GPU: clean exit, 2,757 MiB,
2.53 s of wall per simulated second, and the stage 0a cost model held to 1.1%.

**FastEddy contains no CFL machinery at all.** `dt` is a mandatory user constant bounded only
by `FLT_MIN … FLT_MAX`. The solver is fully compressible with RK3 and no acoustic
sub-stepping, so the limit is acoustic. The tutorials do not share a CFL (NBL 1.603, SBL
0.904) and are no guide. Bisected on the production grid over 500-step trials scored by
FastEddy's own `CORRUPTED` detector (which logs and does not change the exit code):

| dt [s] | CFL_3d | result |
|---|---|---|
| 0.0267 | 1.606 | stable |
| 0.0290 | 1.744 | stable, the maximum |
| 0.0300 | 1.804 | unstable |

The boundary brackets RK3's theoretical imaginary-axis limit √3 = 1.732. The 1-D
minimum-spacing theory (0.050 s) was decisively wrong. The recommended `dt = 0.0275 s` was
**superseded the same day**. There is an accuracy boundary near CFL 1.64, below the
stability boundary, above which the run exits 0 while resolved `w` in the lowest levels
degenerates into grid-scale acoustic noise. That is what produced stage 2's first
near-surface artifact, and it was reported upstream as
[NCAR/FastEddy-model#134](https://github.com/NCAR/FastEddy-model/issues/134).

Also measured: `hydroSubGridWrite = 0` drops nine SGS stress fields (76 → 40 B per cell).
fp16 on write halves it again. Only the four LPDM fields with fp16 would reach 22 GB per
window, and that needed a source change (which became patch 0001).

## Stage 2 and the first pipeline pass (first-pass grid, `dz_sfc = 20 m`)

A 30 m pipeline-development grid, 146 × 50 × 90, `dt = 0.0625 s` (CFL 1.491), receptor at
exactly 30.000 m on cell centre `k = 1`. It was never meant to be the science grid.

| gate | result |
|---|---|
| bitwise restart at 30 m | PASS: byte-identical re-dump, then divergence at the 1e-4 floor |
| `dt` recalibrated, `k0/k1 < 1` | PASS: 0.17 |
| accuracy limit is grid-independent | PASS: threshold between CFL 1.60 and 1.70 at both 10 m and 30 m (`dx` 3×, `dz_sfc` 2×) |
| stationarity, 6 h spin-up | PASS: trends +0.56 σ (TKE), +0.92 σ (`u*`) |
| storage under 30 GB | PASS: 9.6 GB by configuration alone |
| well-mixed + transit time | PASS: rms 3.4% against 4.5% counting noise. Median transit 3.2 min |
| Kljun agreement | **FAIL**: peak 310 m vs 198 m, diagnosed as resolution |
| error floor | measured, and large enough to change the corpus plan |
| real surface | difference explicable (array share 9.7% → 14.8%), but integral 1.64 |

**What was learned that survived:**

- The stationarity gate is a **trend test**, not a difference of two dumps. Converged neutral
  TKE wanders about 7% between dumps, and a two-point rule measures that scatter.
- The accuracy boundary is a property of `CFL_3d`, not of spacing, and the gap between it and
  the stability boundary is wider at coarser resolution (CFL 1.80 at 30 m produces garbage and
  does not go NaN).
- **The reverse-time drift.** Reversing a Langevin model by `(u, t) → (−u, −t)` gives an
  anti-damped equation that diverges. Thomson (1987): the reverse drift is
  `Â = −A + (BBᵀ)∇ ln p`, so the damping keeps its sign and only the `σ²`-gradient term flips.
  Getting this wrong either diverges (loud) or drops the gradient term and accumulates
  particles at the surface (silent, plausible footprint).
- **`eps` is FastEddy's own**, `c_e e^{3/2}/l` with `l = min(0.76√e/N, Δ)`, read from
  `cuda_sgstkeDevice.cu` and recomputed at load time.
- **The estimator constant was verified without the LES** (`bin/test_estimator.py`). In
  homogeneous turbulence with a reflecting lid the surviving flux is `Q(1 − (z − z_td)/H)`, a
  lid-dependent target, and the measured 0.530 and 0.753 matched 0.600 and 0.778 within their
  standard errors. The half-space form converges slowly because a finite backward time
  truncates the tail (0.914 ± 0.035 at 1800 s over 8 seeds).
- **The well-mixed lid artifact.** Reflecting at an artificial lid flips the sub-grid velocity
  but not the resolved `w`, giving a 2× pile-up in the lid bin in both time directions. The
  fix was to release through a deep column and score only the interior.
- **The near field was missing.** Inside 190 m the LES had 4.3% of its influence against
  Kljun's 14.8%, because 96.4% of the vertical velocity variance at the receptor was sub-grid.
  A 4× sweep in `C0` moved the peak by one cell. A seed change moved it by zero. Not tunable.
- **The 80% source-area overlap was at its noise floor** (two realisations 32%, LES vs Kljun
  39%), and per-cell L1 between realisations was 92%. Peak and centroid were resolved.
- **The restart file carries the surface, and that is both a trap and the mechanism**
  ([configuration](../les/configuration.md)). No FastEddy source change was needed for stages
  2–6.
- **The 1.64 integral** over terrain came from mixing the mean advective flux into the
  turbulent one. Reynolds decomposition fixed the sign and shape (centroid −5412 → +1457 m)
  but not the normalisation.

## The second pass (2026-08-19): surveyed coordinate, water, finer vertical grid

Everything site-specific in the first pass was void. The tower coordinate was a surrogate
chosen by a water-avoidance rule (the first estimate had landed on Lake Kegonsa). The surveyed
position `42.957160, −89.292362` replaced it. Water became a land-cover class detected from
the sub-cell elevation spread of the LiDAR DEM (specular over water. A bimodal histogram with
the threshold in the empty gap at 0.01–0.02 m), later replaced by the WorldCover class when
the surface builder moved to 3DEP + WorldCover. Land cover is not tapered at the seams.
Terrain is. The array was redefined as a **geographic object** rather than an upwind distance,
which had made it follow the wind.

**Why the integral was not 1: periodic wraparound.** Sweeping `t_back` with the wrapped
fraction measured alongside, on the flat window: uncapped, the integral went past 1 exactly
as wrapping set in (0.791 → 1.064 from 900 to 1500 s, 8% → 32% wrapped). Capped at one domain
length it converged to 1 from below (0.896 → 0.961). A trajectory that travels more than one
domain length re-enters the turbulence it already sampled. `max_disp` defaults to one
streamwise domain length from here on. The streamline-frame rotation (Wilczak double
rotation) was implemented and found to change the result by 2%, not 64%. It was kept because
it is the frame the instrument reports in.

**The finer vertical grid**, 146 × 50 × 122 with `dz_sfc = 8.56 m` (a cell centre at exactly
30 m needs `dz_sfc = 30/(k + 0.5)`. 8.571 m puts it at `k = 3`), broke two things.
`Error: No such file or directory` on a missing restart is not fatal to FastEddy (890,600
NaN cells, exit 0. `run_case.sh` now refuses first). And **terrain amplifies the effective
CFL** as `CFL_3d · sqrt(1 + (slope · dx/dz)²)`. The westerly adjustment tripped `k0/k1 = 3.85`
where the flat run at the same `dt` was 0.128, because `dx/dz` had gone from 1.50 to 3.50.
Refining `dz` alone makes a grid more sensitive to terrain. Terrain runs took `dt = 5/199 s`.

Rerun at the finer grid: restart bitwise PASS. Stationarity not reached at 6.4 h (`u*` at
−2.25 ± 0.27 %/h, a systematic 60 m peak offset that no averaging removes). Well-mixed PASS
(rms 4.9% at a 4.5% floor). **Sub-grid fraction 88.3% at the receptor, FAIL against a 40%
gate that is unreachable at `dx = 30 m`.** The fraction collapses onto `z/Δ` with
`Δ = (dx dy dz)^{1/3}`, the 40% crossing is at `z/Δ ≈ 3.5–3.7`, so the gate needs
`Δ ≲ 8.6 m`, and with `dx = 30 m` that would take `dz ≤ 0.71 m`. The options that pass
(`dx = 10 m`, `dz_sfc = 6 m`, 23 GPU-h per spin-up. Isotropic 8.6 m, 20 GPU-h) were priced as
a project decision.

**Ensemble convergence**, from 18 independent 150 s sub-windows of one run (lag
autocorrelations below `2/√18`): the peak is stable to one cell at the 90th percentile with
5 sub-windows (12.5 min). The centroid is still 336 m at 22.5 min and improving. A fixed
held-out reference had understated this (120 m), an artifact of one degenerate subset that
randomising the reference removed. So a 30-minute window is sufficient for the peak and
marginal for the centroid. The way to improve it is sampling time within a run rather than
number of runs, and a production window needs `t_back` plus the sampling time.

**Stage 6, both directions.** The array took 10.3× its area share when upwind (westerly) and
0.06× when downwind (easterly). Water took 35.2% of the easterly footprint against a
predicted 35.5% from the band it occupies. The integrals saturated (the wrap cap held). The
westerly settled 30–40% high where the receptor is in mean ascent (advective
non-closure, `w̄ = +0.064 m/s`, concentration integral 18.5 against 8–10) and the other two
about 18% low because the footprint barely fit the 4380 m domain (80% source area to 3810–3870 m).
Terrain footprints were noisier for a measured reason. `t_back` used half of an 1800 s window,
leaving three sub-windows, and the ensemble curve predicted their centroid scatter to 1.5%.

What this pass left as the decision: the near field at 30 m is closure output, and the grid
had to change. The third pass took the static 24 m configuration. The fifth moved to 16 m
with a 10 m receptor. The seventh found that the 10 m receptor could not work either and moved
the receptor to 30 m on a 24 m grid. The eighth settled on 122³ at 30 m with the receptor at
30 m, three levels up. Every one of those is a chapter of this history.

Scripts and runs of these stages were removed from the tree on 2026-09-04 and remain in the
offline pre-cleanup archive of 2026-09-04: `runs/stage0a_smoke_*`, `runs/stage1_*`, `runs/stage2_*`,
`runs/s30_*` (except the 30 m CFL ladder, which is kept), `bin/run_pipeline.sh`,
`bin/stage6_compare.py`, `bin/stage6_predict.py`, `bin/ensemble_convergence.py`,
`results/stage4.txt`, `results/stage5*.{txt,npz}`, `results/stage6*`, `results/fv_*`.
