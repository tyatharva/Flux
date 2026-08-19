# Third pass — static 24 m domain, USGS + WorldCover, and the sigma_w deficit

Supersedes `STAGE2-6_RESULTS_V2.md` for everything grid- and site-specific. Plan:
`~/.claude/plans/fluttering-baking-stonebraker.md`.

The headline is not the new grid. It is that the near-field footprint error, which the
second pass attributed to resolution and left as a blocking gate, turns out to be a
**specific, measurable, correctable deficit in `sigma_w`** — and that the "obvious" fix for
it makes things four times worse.

---

## 1. The grid, and a 17% speedup we had been giving away

`186 x 186 x 122` at `dx = dy = 24 m` (4464 m box), `dz_sfc = 8.5580 m`, receptor at `k = 3`
on a cell centre at exactly 30.000000000 m, top 3000 m, damping 600 m.

**`Nz = 122` is the power-of-two choice, once you count the halo.** FastEddy pads every
dimension by `2*Nh = 6` and the divisibility rule applies to the padded extent, so what has
to factor is `Nz + 6`. `122 + 6 = 128 = 2^7`; `128 + 6 = 134 = 2 x 67` would cap `tBz` at 2.

**The thread block had been the wrong shape since Stage 1.** `i <- threadIdx.x`
(`cuda_hydroCoreDevice.cu:648`) while `kStride = 1` and `iStride = (Ny+6)(Nz+6)`
(`grid.c:621-623`). CUDA linearises a warp with `threadIdx.x` fastest, so any `tBx > 1`
makes adjacent threads read addresses `iStride` floats apart — one 128-byte transaction
becomes four 32-byte ones. Every shipped tutorial uses `tBx = 1`; ours used 4.

Nine legal shapes, 200 steps each, ~2 minutes total:

| block | threads | s/step | vs the 9.37 ns/cell/step model |
|---|---|---|---|
| **1x2x64** | 128 | **0.0359** | 0.91x |
| 1x6x32 | 192 | 0.0364 | 0.92x |
| 1x3x64 | 192 | 0.0367 | 0.93x |
| 1x8x32 | 256 | 0.0380 | 0.96x |
| 2x4x32 / 2x2x64 / 1x1x64 / 1x4x64 | | 0.0383-0.0386 | 0.97-0.98x |
| `4x4x16` — the old default | 256 | **0.0421** | 1.06x |

**17% free, from a config line.** `tBz = 128` is rejected by the device: CUDA caps
`blockDim.z` at 64, and FastEddy reports it as such. The cost model becomes **8.51
ns/cell/step**, not 9.37 — the old figure was measured with the bad block and is 10%
pessimistic.

Resulting cost: **1.09 GPU-h per simulated hour**, `dt = 5/152 s` flat (`CFL_3d = 1.4946`),
`k0/k1 = 0.114` — comfortably inside the accuracy limit.

---

## 2. The surface: static, authoritative, and built once

No rotated GIS anywhere. `bin/prep_surface.py` builds terrain and land cover ONCE on a
fixed, north-up EPSG:3071 grid; direction is changed by rotating the *flow*. The class of
bug that put the solar array at a fixed *upwind distance* is now structurally impossible.

- **Terrain**: USGS 3DEP 1/3 arcsec, `average`-resampled to 24 m. 256.7-317.4 m, 60.7 m of
  relief, 269.1 m at the tower.
- **Land cover**: ESA WorldCover v200 (2021), 10 m, `mode`-resampled (a class map must never
  be averaged). 37.4% cropland, 28.5% tree, **16.1% permanent water**, 15.7% grassland,
  2.2% built.
- **Water is strongly directional** — within 4 km: N 0%, NE 58%, **E 72%**, SE 2%, S 0.4%,
  SW 0%, W 0%, NW 0%. Nothing inside 1 km.

**The old LiDAR water mask was not wrong.** Checked against WorldCover it agrees to about
one percentage point per annulus (1-1.5 km: 12.2 vs 12.9%; 1.5-2 km: 21.1 vs 21.4%; 2-3 km:
19.7 vs 20.8%). What made the lake *look* wrong was the rotated 4380 x 1500 m strip slicing
a ribbon through it, plus a figure drawing the shoreline as a bare contour. WorldCover is
adopted because it is authoritative and carries every roughness class — not as a fix.

**Terrain smoothing, 2 x (1-2-1), and why it is not a fudge.** The raw 24 m field reaches
`|grad z| = 0.37` — a 9 m drop across one cell, which the grid can only alias. Through the
terrain-following metric that one cell alone would force a 1.44x smaller `dt` on every
terrain run. Two passes take the maximum to 0.245 (amplification 1.213) for an rms change of
**0.44 m against 61 m of relief**. This is filtering topography to the resolvable scale,
which is standard for terrain-following LES; it happens also to buy 16%.

### The array, corrected — and it makes Stage 6 a much sharper test

The tower is **inside** the array: 60 m east and west, 250 m north, 100 m south. 120 x 350 m,
4.20 ha nominal, 4.32 ha discretised (75 cells), and the tower cell is inside it. WorldCover
labels the whole rectangle "cropland" — it does not see photovoltaics — so the patch is an
override.

Because the tower sits inside it, the array's **upwind reach depends on direction**: 250 m
for a northerly, 100 m for a southerly, 60 m for an easterly or westerly. The fraction of
the crosswind-integrated footprint inside that reach therefore swings by ~300x:

| stability | E/W (60 m) | S (100 m) | **N (250 m)** |
|---|---|---|---|
| very unstable | 0.1% | 4.2% | **34.1%** |
| neutral | 0.0% | 1.2% | **24.2%** |
| stable | 0.0% | 0.1% | **13.6%** |

That is a site result, not just a test design: **this tower measures the array on
northerlies and measures the neighbours on easterlies and westerlies.**

---

## 3. The sub-grid hypothesis: falsified, then replaced

The plan's item F was that the isotropic `sigma_s^2 = (2/3) e_sgs` split is wrong near a
wall, where blocking suppresses `w` and the surface layer runs
`sigma_u : sigma_v : sigma_w ~ 2.5 : 1.9 : 1.25`. Implemented it with those ratios,
normalised to conserve energy (`r_u, r_v, r_w = 1.642, 0.948, 0.410`).

**It made the footprint four times worse.** All rows below use the same fields, the same
seed and the same releases; only the sub-grid variance differs. Kljun is held fixed.

| variant | peak `x` | vs Kljun | centroid | 80% area | overlap | integral |
|---|---|---|---|---|---|---|
| Kljun et al. (2015) | 210 m | — | 788 m | 26.6 ha | — | 0.927 |
| isotropic `(2/3)e` (baseline, WSM04) | 390 m | +86% | 1263 m | 45.4 ha | 36.9% | 0.805 |
| **surface-layer anisotropic** | **1170 m** | **+457%** | 2003 m | 71.3 ha | **18.5%** | 0.595 |
| isotropic, variance x1.349 (scalar) | 270 m | +29% | 1021 m | 36.4 ha | **47.6%** | 0.812 |
| **MOST floor, surface layer [adopted]** | **270 m** | **+29%** | 1159 m | 39.2 ha | 40.0% | ~0.81 |

**The isotropic split was not the error — it was compensating for one.** The thing it
compensates for is directly measurable at the receptor:

> `sigma_w / u* = 1.09`, against the neutral surface-layer value of **~1.25**

because at `z/Delta ~ 1.5` the eddies that carry `w` sit at or below the filter scale. A low
`sigma_w` makes backward particles descend too slowly, so they travel further before
touching down — peak too far upwind, distribution too broad. Both are what the baseline
shows, and the anisotropic split cut `sigma_w` further (`r_w = 0.41`), moving the error in
the predicted direction by roughly the predicted amount. The hypothesis is cleanly killed.

**Supplying the missing variance instead recovers most of the gap**: peak error +86% ->
+29%, 80% overlap 36.9% -> 47.6%, 80% area 45.4 -> 36.4 ha, and the 80% source area pulls in
from 3810 m to 2730 m — which also relieves the domain-truncation problem from the second
pass. The integral is unchanged at 0.81, so this is not a rescaling of the answer.

### What is adopted, and why not the better-scoring one

Implemented as `--sgs-most`: a **height-dependent, MOST-anchored floor**. It supplies only
what similarity says is missing, never reduces the LES's own sub-grid variance, and tapers
off across `0.1h - 0.2h` because MOST is a surface-layer relation. At the receptor it is a
factor of 1.242 and takes `sigma_w/u*` from 1.09 to 1.20.

The tuned scalar scores better on overlap (47.6% vs 40.0%, against a 63.0% half-vs-half
floor) and is **not** adopted: it is a constant fitted to one case at one height with no
rule for transferring it, and adopting it would bake a tuned number into the corpus targets.

The gap between them is itself informative. Both fix the **peak** identically. They differ
only on the tail, and the scalar's advantage comes entirely from adding variance *above* the
surface layer, which the MOST floor deliberately will not do. Read straight:

> the **peak** error is a surface-layer `sigma_w` deficit, and is fixable at the closure
> level; the residual **tail** error is boundary-layer-wide, and *is* a resolution limit.

That is a sharper statement than "82% of `sigma_w^2` is sub-grid", and it is actionable.

---

## 4. Stage 3: 16-bit output, and it is free

`ioLPDMmode` on the `kegonsa` fork (commit `59f0472`), one optional parameter defaulting to
upstream behaviour. With it on: only the fields a backward LPDM reads are written, the five
3-D prognostics are CF-packed to 16-bit (`scale_factor`/`add_offset`, so any CF-aware reader
unpacks them unchanged), and the static coordinate geometry goes into the first file of a run
only. **77 GB per 30-min window becomes about 19 GB.**

CF packing rather than IEEE half deliberately: it is self-describing, needs no reader change,
and its uniform absolute resolution suits a velocity field that crosses zero better than a
floating-point one. The files are **not restartable** by construction — `rho` and `pressure`
are absent — and the parameter help says so.

**Verified on real fields before adoption** (`bin/fp16_test.py`, no LES required):

| | fp32 | fp16 | difference |
|---|---|---|---|
| peak | 330 m | 330 m | **0 m** |
| centroid | 1243.8 m | 1263.2 m | +19.4 m |
| integral | 0.7919 | 0.8071 | +0.015 |
| 80% overlap | — | — | **75.7%** |

Against an error floor of 59.2% overlap / 60 m peak / 99 m centroid between two halves of
the same window, the quantisation is comfortably inside the estimator's own noise. The same
result justifies holding the analysis cache in float16, which is what makes a 480-dump
window at this grid fit in RAM at all (49 GB -> 24 GB).

---

## 5. Traps found this pass

**The restart timestep is parsed from the FILENAME.** `time_integration.c:104` does
`sscanf` on the characters after the first `.` in `inFile`. A name like `restart.nc` leaves
`simTime_itRestart` **uninitialised**. This is also a lever: naming the restart `FE_RST.0`
resets the step counter, which is the clean way to keep `frqOutput` dividing the absolute
step across a restart — otherwise the run silently writes exactly one dump (the trap already
in CLAUDE.md).

**`tBz` cannot exceed 64.** CUDA's `maxThreadsDim[2]` is 64 on every current device, so the
otherwise-ideal `1x1x128` is rejected. FastEddy reports it cleanly.

**A 24 m cell cannot carry a slope of 0.37.** Terrain must be filtered to the resolvable
scale, or one aliased cell sets `dt` for the whole domain.
