# Third pass: the static 24 m domain, and the `σ_w` deficit

August 2026. Grid 186 × 186 × 122 at `dx = 24 m` (a 4464 m box), `dz_sfc = 8.558 m`, receptor
at `k = 3` on a cell centre at exactly 30.000 m, top 3000 m, damping 600 m. Supersedes the
second pass for everything grid- and site-specific. The headline was not the new grid: the
near-field footprint error the second pass attributed to resolution turned out to be a
specific, measurable, correctable deficit in `σ_w`, and the "obvious" fix for it made things
four times worse.

## The grid, and a 17% speedup that had been given away

`Nz = 122` is the power-of-two choice once the halo is counted: FastEddy pads every
dimension by 6 and the divisibility rule applies to the padded extent, so `122 + 6 = 128 = 2⁷`.

**The thread block had been the wrong shape since stage 1.** `i ← threadIdx.x` while
`kStride = 1`, so any `tBx > 1` makes adjacent threads read addresses `iStride` floats apart:
one 128-byte transaction becomes four. Nine legal shapes, 200 steps each:

| block | threads | s/step |
|---|---|---|
| **1×2×64** | 128 | **0.0359** |
| 1×6×32 | 192 | 0.0364 |
| 1×3×64 | 192 | 0.0367 |
| 1×8×32 | 256 | 0.0380 |
| 4×4×16 (the old default) | 256 | 0.0421 |

17% free from a config line; `tBz = 128` is rejected by CUDA's 64 cap. The cost model became
8.51 ns per cell per step (the 9.37 had been measured with the bad block). Cost 1.09 GPU-h per
simulated hour at `dt = 5/152 s` (CFL 1.4946), `k0/k1 = 0.114`.

## The surface: static, authoritative, built once

No rotated GIS anywhere: `bin/prep_surface.py` builds terrain and land cover once on a fixed
north-up EPSG:3071 grid, and direction is changed by rotating the flow. The class of bug that
put the array at a fixed upwind distance became structurally impossible.

- Terrain: USGS 3DEP 1/3 arc-second, average-resampled to 24 m; 256.7–317.4 m, 60.7 m of
  relief, 269.1 m at the tower. Smoothed twice with a 1-2-1 filter: the raw field reaches
  `|∇z| = 0.37`, a 9 m drop across one cell that the grid can only alias, which through the
  terrain-following metric would force a 1.44× smaller `dt` on every terrain run. Two passes
  take the maximum to 0.245 for an rms change of 0.44 m against 61 m of relief.
- Land cover: ESA WorldCover v200 (2021), 10 m, mode-resampled (a class map is never averaged):
  37.4% cropland, 28.5% tree, 16.1% permanent water, 15.7% grassland, 2.2% built. Water is
  strongly directional within 4 km: N 0%, NE 58%, E 72%, SE 2%, S 0.4%, SW 0%, W 0%, NW 0%,
  nothing inside 1 km. The earlier LiDAR water mask agreed with WorldCover to about one point
  per annulus; WorldCover was adopted because it is authoritative and carries every class.
- **The array, corrected**: the tower is *inside* it, 60 m east and west, 250 m north, 100 m
  south, 120 × 350 m, 4.20 ha (75 cells at 24 m). WorldCover labels the rectangle cropland, so
  it is an override. Because the tower sits inside it, the upwind reach depends on direction
  (250 m northerly, 100 m southerly, 60 m easterly or westerly) and the fraction of the
  crosswind-integrated footprint inside that reach swings about 300×: neutral 24.2% for N,
  1.2% S, 0.0% E/W. This tower measures the array on northerlies and the neighbours otherwise.

## The sub-grid hypothesis, falsified and replaced

The plan's hypothesis was that the isotropic `σ_s² = (2/3) e_sgs` split is wrong near a wall,
where the surface layer runs `σ_u : σ_v : σ_w ≈ 2.5 : 1.9 : 1.25`. Implemented with those ratios
(`r_w = 0.410`), it made the footprint four times worse. Same fields, seed and releases, Kljun
fixed:

| variant | peak | vs Kljun | centroid | 80% area | overlap | integral |
|---|---|---|---|---|---|---|
| Kljun | 210 m | | 788 m | 26.6 ha | | 0.927 |
| isotropic `(2/3)e` (baseline) | 390 m | +86% | 1263 m | 45.4 ha | 36.9% | 0.805 |
| surface-layer anisotropic | **1170 m** | **+457%** | 2003 m | 71.3 ha | 18.5% | 0.595 |
| isotropic × 1.349 (scalar) | 270 m | +29% | 1021 m | 36.4 ha | **47.6%** | 0.812 |
| **MOST floor, surface layer (adopted)** | **270 m** | **+29%** | 1159 m | 39.2 ha | 40.0% | 0.882 |

The isotropic split was not the error; it was compensating for one, directly measurable at the
receptor: `σ_w/u* = 1.09` against the neutral surface-layer value of about 1.25, because at
`z/Δ ≈ 1.5` the eddies that carry `w` sit at or below the filter scale. Low `σ_w` makes
backward particles descend too slowly, so they travel further before touching down. Supplying
the missing variance recovers most of the gap (+86% → +29%) and pulls the 80% source area in
from 3810 m to 2730 m.

**What was adopted**: `--sgs-most`, a height-dependent MOST-anchored floor that supplies only
what similarity says is missing, never reduces the LES's own variance, and tapers off across
`0.1h–0.2h`. Factor 1.242 at the receptor. The tuned scalar scored better on overlap and was
not adopted: a constant fitted to one case at one height with no rule for transferring it.
The two fix the peak identically and differ only on the tail, so **the peak error is a
surface-layer `σ_w` deficit fixable at the closure level, and the residual tail error is a
boundary-layer-wide resolution limit**. (The floor's *magnitude* was later found wrong and
fixed in the [sixth pass](pass-6.md).)

## Stage 3: 16-bit output, and it is free

`ioLPDMmode` (patch 0001): only the fields a backward LPDM reads are written, the five 3-D
prognostics are CF-packed to 16 bit, and the static geometry goes into the first file of a run
only. 77 GB per window became about 19 GB. CF packing rather than IEEE half because it is
self-describing and its uniform absolute resolution suits a field that crosses zero. Verified
on real fields (`fp16_test.py`): peak identical, centroid +19.4 m, integral +0.015, 80%
overlap 75.7% against a 59.2% half-vs-half floor. The same result justified holding the
analysis cache in float16 (49 → 24 GB for a 480-dump window).

## Gates

| stage | gate | result |
|---|---|---|
| 2 | TKE stationarity | **PASS**: TKE −0.23 ± 1.03 %/h, `u*` +0.80 ± 0.57 %/h. The second pass had failed at −8.40 σ after 6.4 h; the chained output at 12.5-min resolution showed `u*` overshooting to 0.41 near 1 h, decaying, and settling by 5 h: an initial adjustment, not a slow inertial drift |
| 2 | profile vs NCAR NBL | `σ_w²`peak/`u*²` 0.780 (ref 0.730) at 137 m (ref 130), veering −21° (ref −25°) |
| 3 | window under 30 GB | PASS, 15 GB with `ioLPDMmode` |
| 5 | sub-grid fraction < 40% | FAIL at about 80%, but the quantity it proxies for was diagnosed and largely corrected |
| 5 | Kljun agreement | peak +29%, 80% area exact, overlap 48.6% against a 53.6% floor |
| 6 | explicable difference | **PASS, quantitatively** |

**Stage 6**: four directions from one spun-up state by 90° re-indexing, the surface bit-identical
in all four. Achieved winds backed 22–24° from the forcing by Ekman turning (wN 336°, wS 158°,
wE 67°, wW 247°). Array share 3.50% N, 0.53% S, 0.02% E, 0.00% W against an area share of
0.22% (15.9× northerly); water 13.79% on the easterly against 0 westerly. Predicted share
ratios from the chord and Kljun's cumulative footprint (`stage6_predict.py`): wN/wE 97× predicted
vs 175× measured, wN/wS 3.0× vs 6.6×, wS/wE 32× vs 26.5×. Ordering exact, magnitudes within a
factor of two, and the excess in the direction the LES's own near-field deficit requires.
Wraparound was ruled out as the source (the far field carries at most 2.3%, negative for wE
and wW).

## Traps found

The restart timestep is parsed from the filename (traps §4); `tBz` cannot exceed 64 (§8); a
24 m cell cannot carry a slope of 0.37.

## What was left

The sub-grid gate at about 80% was now a narrower statement. A production window should be
longer than 30 min: with `t_back = 900 s` a 30-min window yields 15 min of releases, visibly
not enough over terrain (half-vs-half overlap 27–43% against 53.6% flat). Ekman backing should
be compensated or the achieved direction used as the label; the runs report what they
achieved.

Removed from the tree on 2026-09-04 (in the offline pre-cleanup archive of 2026-09-04): `runs/g24_*`,
`results/g24_*`, `bin/g24_bringup.sh`, `bin/fp16_test.py`'s outputs.
