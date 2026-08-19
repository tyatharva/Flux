# Second pass — surveyed coordinate, water land cover, finer vertical grid

Supersedes `STAGE2-6_RESULTS.md` for everything site-specific. That document's Stage 6
results were produced at a surrogate tower coordinate and are void.

---

## Item 1 — Reference frame ❌ NOT THE CAUSE (but implemented anyway)

A real EC system over a slope is double-rotated (Wilczak et al. 2001): first about `z` so
the mean crosswind vanishes, then about the new `y` so the mean vertical vanishes. The LPDM
now records all three velocity components at release and weights by the streamline-frame
vertical velocity

```
  w_sf = w cos(phi) - (u cos(theta) + v sin(theta)) sin(phi)
  theta = atan2(V, U),   phi = atan2(W, sqrt(U^2+V^2))
```

**It does not explain the 1.64.** Streamline-frame and simple mean-removal agree to within
2% at every backward time tested:

| `t_back` (s) | model, mean removed | STREAMLINE | difference |
|---|---|---|---|
| 300 | 0.861 | 0.855 | 0.7% |
| 600 | 1.282 | 1.261 | 1.6% |
| 900 | 1.210 | 1.194 | 1.3% |
| 1500 | 1.561 | 1.523 | 2.4% |

The pitch angle is only ~0.8 deg, so `sin(phi) ~ 0.014`; it multiplies the horizontal
fluctuation, whose flux is much larger than the vertical one, which is why the effect is
not entirely negligible — but it is a 2% correction, not a 64% one. The rotation is kept
because it is the frame the instrument reports in and it makes the mean vanish by
construction rather than by subtraction.

---

## Item 2 — Why the integral was not 1 ✅ RESOLVED: periodic wrap-around

The three hypotheses were separated by sweeping `t_back` with the wrap-around fraction
measured alongside. **The integral grows without saturating, and the growth tracks the
fraction of touchdowns beyond one domain length.** The FLAT window settles it, because
there the mean-subtraction term is ~0 and cannot be blamed:

| `t_back` (s) | wrapped | uncapped | capped at one domain length |
|---|---|---|---|
| 300 | 0.0% | 0.643 | 0.643 |
| 600 | 0.0% | 0.800 | 0.800 |
| 900 | 8.2% | 0.791 | **0.896** |
| 1500 | 31.8% | **1.064** | **0.961** |

Uncapped, the flat integral sails past 1 exactly as wrapping sets in. **Capped, it
converges to 1 from below** — which is what a flux footprint must do, and which is the
check that the estimator itself is sound on real LES fields rather than only on the
synthetic test.

A backward trajectory that travels more than one domain length re-enters the turbulence it
already sampled; its later touchdowns are the same eddies counted again. `max_disp` now
defaults to one streamwise domain length in `compute_footprint`.

**With the cap the terrain case stops diverging and saturates** at 1.19-1.35 instead of
climbing to 1.64:

| `t_back` (s) | terrain, capped | `w_bar * C` | wrapped |
|---|---|---|---|
| 300 | 0.861 | 0.519 | 0.0% |
| 600 | 1.270 | 1.191 | 0.0% |
| 900 | 1.354 | 1.241 | 0.0% |
| 1500 | 1.191 | 1.333 | 0.0% |

The residual is `w_bar` times the concentration integral, both now converged. At a receptor
sitting in 1.5 sigma of mean subsidence the turbulent flux genuinely is not the surface
flux — the advection non-closure of eddy covariance in complex terrain. It is now a
measured number rather than a runaway.

---

## Site: surveyed coordinate and water as a land-cover class

**Tower `42.957160, -89.292362`** (EPSG:3071 577719.1, 276299.5). Sanity checks at that
position: elevation 268.61 m, sub-cell elevation spread 0.287 m (land, not water), nearest
open water 346 m, 60 m of relief across the +/-4 km tile.

**Water is detected by measurement, not by a flatness guess.** `docker/prep_dem30.py`
aggregates the 0.4572 m LiDAR to 30 m and emits, alongside the mean, the standard deviation
of the source elevations *within* each cell. A bare-earth surface over open water is
specular and interpolated to a level plane, so that spread collapses to millimetres; land
keeps centimetres. The histogram is bimodal with a clean gap:

```
   sub-cell std (m)    cells
   0.000 - 0.010      12214     <- open water
   0.010 - 0.020        351     <- the gap; the threshold sits here
   0.020 - 0.050       1252
   0.100 - 0.200      11539     <- land
   0.200 - 0.500      23360
```

Classifying on "flat at 30 m" alone would sweep in ploughed fields and road corridors.

**The rotation now behaves correctly, which it did not before.** The array had been defined
by an *upwind distance*, so it silently followed the wind. It is now a geographic object
(bearing 270 deg, 200 m from the tower):

| wind from | water | solar array |
|---|---|---|
| 270 deg (westerly) | 840-1050 m **downwind** | 150-240 m **upwind** |
| 90 deg (easterly) | 840-3300 m **upwind**, **62% of the upwind half** | **downwind** |

**Land cover is deliberately NOT tapered**, unlike terrain. Terrain height enters the
coordinate transform and its metric tensor, so a seam step is a numerical cliff; roughness
and surface heat flux are local boundary conditions, where a seam is a coastline. Tapering
them would erase the water from the upwind edge of exactly the easterly cases meant to
sample it. Verified: all 3,396 water cells retain `z0 = 1e-4 m`.

**Albedo has no pathway, and that is not an omission.** FastEddy in this configuration has
no radiation scheme at all — `surflayerSelector = 1` prescribes the kinematic surface heat
flux directly — so what albedo would have controlled is subsumed by `htFlux`, which IS
per-cell: `cuda_surfaceLayerDevice.cu:191` reuses the `htFlux` array when
`surflayer_idealsine = 0`, and `htFlux`/`z0m`/`z0t`/`tskin` are all IO-registered so they
survive the restart read. The built-in `surflayer_offshore` wave-roughness parameterisations
are a **global** switch and cannot be applied to water cells only, so per-cell `z0` is used.

---

## New grid, and the two things it broke

`146 x 50 x 122`, `dx = dy = 30 m`, **`dz_sfc = 8.5601 m`**, top 2700 m, `dt = 5/147 s`
(`CFL_3d = 1.488`), receptor at **exactly 30.000000000 m**, three levels below the tower
(was one), 40 levels below 400 m (was 20). `k0/k1 = 0.128` on the flat runs.

**`dz_sfc = 10 m` and a cell centre at 30.000 m are mutually exclusive.** The near-surface
cubic correction in `zDeform` is under 0.05 m, so centres sit at `(k+0.5)*dz_sfc` and a
centre at 30 m requires `dz_sfc = 30/(k+0.5)`: 20 m (k=1), 12 m (k=2), **8.571 m (k=3)**,
6.67 m (k=4). 8.56 m was chosen as the option that goes further in the intended direction
than 10 m rather than less far.

### Break 1 — a missing restart file is not fatal to FastEddy

A path typo cost a 30-minute segment. FastEddy printed `Error: No such file or directory`,
**carried on with x,y,z dimensions of 0**, and produced a run in which all 890,600 cells of
every field were NaN — exiting 0. `check_run.sh` caught it at the end; `run_case.sh` now
resolves `inPath`+`inFile` and refuses before spending GPU time.

### Break 2 — terrain amplifies the effective CFL, and the amplification scales with grid anisotropy

The Stage 6 westerly adjustment tripped `k0/k1 = 3.85` while the flat run at the identical
grid and `dt` was clean at 0.128.

**It is not flow-following motion**, which was the obvious alternative: subtracting the
terrain-following component `u.grad(zg)` leaves the ratio unchanged (3.845 -> 3.915), and
the actual near-surface `w` correlates with that prediction at only **+0.16** while being
2.7x larger in rms. It is grid-scale noise.

In a terrain-following coordinate the horizontal derivative picks up `J31 d/dzeta`, so

```
  CFL_eff  ~  CFL_3d * sqrt(1 + (slope * dx/dz)^2)
```

| slope | amplification | `CFL_eff` at the flat-run `dt` |
|---|---|---|
| p50 0.039 | 1.009 | 1.502 |
| p90 0.099 | 1.058 | 1.575 |
| p99 0.182 | 1.187 | **1.766** |
| max 0.259 | 1.350 | **2.009** |

At the steepest cells that is past both the 1.64 accuracy limit and the 1.79 stability
limit. **The earlier `dz_sfc = 20 m` grid never showed this** because `dx/dz` was 1.50
rather than 3.50 and the same terrain cost 2.3x less amplification — so **refining `dz`
alone makes a grid MORE sensitive to terrain, not less**, which is the opposite of the
intuition that motivated the refinement. Terrain runs now use `dt = 5/199 s`, giving
`CFL_eff = 1.48` at the steepest cell; `k0/k1` returned to **0.825**.

