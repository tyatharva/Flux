# Second pass — surveyed coordinate, water land cover, finer vertical grid

Supersedes `STAGE2-6_RESULTS.md` for everything site-specific. That document's Stage 6
results were produced at a surrogate tower coordinate and are void.

---

**Items 1 and 2 were measured on the first-pass grid** (`dz_sfc = 20 m`, receptor at `k=1`),
because they are questions about the *estimator*, not about resolution, and answering them
first is what made the regrid worth paying for. Items 3 and 4 are on the new grid.

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


---

## Item 3 — Stages 2-6 rerun at the finer vertical grid

### Stage 2 — spin-up.  Restart ✅ PASS.  Stationarity ❌ NO

`146 x 50 x 122` flat, uniform `z0 = 0.1 m`, neutral, doubly periodic, spun up in chained
segments of under 45 min each to **6.43 h of simulated time** (386 min shown below; the
sampling window runs on past it).

**Bitwise restart re-verified at this grid.** Run to a dump, restart from it, re-dump: `cmp`
reports the two 25.4 MB files byte-identical, every prognostic field differing by exactly 0.
A restarted trajectory then separates from a continuous one at the expected ~1e-4 relative
floor after 500 steps. This is what makes the chained-segment structure — and the whole
spin-up-once / adjust-per-direction corpus design — valid.

**Stationarity is not reached.** Over the last 6 dumps (t = 276-386 min):

| quantity | mean | scatter | trend | significance |
|---|---|---|---|---|
| domain TKE | 0.0788 | 2.3% | **-2.10 +/- 1.13 %/h** | -1.85 sigma |
| `u*` | 0.3343 | 1.7% | **-2.25 +/- 0.27 %/h** | **-8.40 sigma** |

TKE alone would pass at 2 sigma; `u*` fails it by a factor of four. The flow is still
relaxing after 6.4 h. The profile shape is right — `sigma_w^2/u*^2` peaks at 0.844 against
NCAR's published NBL 0.730, and goes to zero by 668 m against their 650 m — but the peak
sits at 174 m rather than 130 m and `u*` is 0.328 against 0.410, both consistent with a
boundary layer that is still deepening and decelerating.

This is a **known, quantified, non-blocking** shortfall: a -2%/h drift is small against the
+86% Stage 5 discrepancy below, and it does not change any conclusion in this document. It
is reported rather than fixed because fixing it means more spin-up hours at a grid that
Stage 5's gate has already ruled out.

### Stage 4 — well-mixed ✅ PASS

| direction | max deviation | rms | lowest 3 bins |
|---|---|---|---|
| **backward** (the mode footprints use) | 10.07% | **4.91%** | **0.978** |
| forward (control) | 15.85% | 4.67% | 1.045 |

~995 particles per bin, so 1-sigma counting noise is 4.48% — the rms is at the noise floor
in both directions, and the lowest three bins, where a broken sub-grid closure piles
particles up, are flat to 2%.

The **artificial lid** in the earlier test was itself the artifact: reflecting at a lid
flips the sub-grid velocity but not the resolved `w`, producing a 2x pile-up in the lid bin
in *both* time directions. Particles are now released through a deep column and only the
interior is scored.

**Second gate:** backward transit from the 30 m receptor — p5 = 73 s, median **287 s**
(4.8 min), p95 = 765 s, with 62% reaching the surface inside 900 s. PLAN.md expects 1-5 min
unstable and 10-15 min stable; neutral sitting at the fast end of that range is correct.

### Stage 5 Gate 1 (revised) — sub-grid fraction ❌ FAIL, and unreachable at `dx = 30 m`

The revised gate is *sub-grid fraction of `sigma_w^2` at the receptor < 40%*. Measured:

| grid | `dz_sfc` | `Delta` | `z/Delta` at 30 m | sub-grid | gate |
|---|---|---|---|---|---|
| first pass | 20.0 m | 26.21 m | 1.14 | 96.4% | FAIL |
| **this pass** | **8.56 m** | **19.78 m** | **1.52** | **88.3%** | **FAIL** |

Refining `dz` by 2.3x moved the fraction by 8 points. The two grids agree that the fraction
collapses onto **`z/Delta`** with `Delta = (dx dy dz)^(1/3)`, and both put the 40% crossing
at `z/Delta` = 3.5-3.7 — so the gate needs **`Delta <~ 8.6 m`**.

**With `dx = dy = 30 m` that is unreachable at any `dz`.** It would require `dz <= 0.71 m`,
and at that anisotropy the horizontal filter (`2 dx = 60 m`) still cannot resolve the 30 m
eddies, so the `Delta` collapse stops describing the physics before the arithmetic gets
there. The gate is a statement about `dx`, not about `dz`.

What passes, and what it costs on this GPU at the measured 9.37 ns/cell/step:

| `dx = dy` | `dz_sfc` | `Delta` | sub-grid | cells | GPU h per sim h | 3.5 h spin-up |
|---|---|---|---|---|---|---|
| 15 m | 8.56 m | 12.44 m | 65% | 4.03 M | 1.4 | 5 h |
| 10 m | 8.56 m | 9.49 m | 47% | 9.07 M | 3.9 | 14 h |
| **10 m** | **6.0 m** | **8.43 m** | **39%** | 12.88 M | 6.6 | **23 h** |
| **8.6 m** | **8.6 m** | **8.60 m** | **40%** | 12.23 M | 5.8 | **20 h** |

**This is the report-and-stop condition, not a fix to attempt.** A 20-23 GPU-hour spin-up is
30-35 chained 40-minute segments — a project-level decision about corpus cost, not a
configuration tweak, and nothing at `dx = 30 m` reaches it.

### Stage 5 Gate 1 (secondary) — Kljun agreement

Reported as a check, not a tuning target, per the revised gate.

| | peak `x` | centroid `x` | 80% source area |
|---|---|---|---|
| LES + LPDM | 390.0 m | 1263.1 m | 45.36 ha |
| Kljun FFP | 210.0 m | 787.7 m | 26.64 ha |
| difference | **+180 m (+86%)** | +475 m (+60%) | 80% overlap **36.9%** |

The footprint is too far out and too broad, in the direction the sub-grid failure predicts:
88.3% of the vertical velocity variance at the receptor is manufactured by the closure
rather than resolved, and the closure's velocities decorrelate faster than real eddies, so
particles travel further before touching down. The finer `dz` moved the peak from 310 m
(first pass, 96.4% sub-grid) the *wrong* way in absolute terms while the reference moved
too — the informative number is that the discrepancy is unchanged in character.

Integral 0.805 with the wrap cap and `t_back = 900 s`; the shortfall is influence truncated
by the backward time, and it approaches 1 from below as `t_back` grows (Item 2).

### Stage 5 Gate 2 — irreducible error floor, and it improved

| | value | first pass |
|---|---|---|
| half-vs-half 80% overlap | **59.2%** | 30.0% |
| half-vs-half peak difference | -60.0 m (one cell) | 40 m |
| half-vs-half centroid difference | -98.9 m | ~50 m |

The first pass measured LES-vs-Kljun overlap at 39% against a half-vs-half floor of 30% —
i.e. the metric was at its noise floor and could not be used to score anything. **At this
grid the floor separates from the signal**: 59.2% between two halves of one window against
36.9% between LES and Kljun. The 80% source-area overlap is now a usable metric, which it
was not before.

---

## Item 4 — Ensemble convergence ✅ MEASURED

Built from **sub-windows of one long integration**, not from separate runs, as instructed:
the 2700 s release period was split into **18 sub-windows of 150 s**, each producing its own
footprint from the same field cache.

**The sub-windows are independent.** Lag autocorrelation of the per-sub-window metrics:

| metric | lag1 | lag2 | lag3 | lag4 |
|---|---|---|---|---|
| peak `x` | +0.19 | -0.07 | -0.28 | -0.27 |
| centroid `x` | -0.10 | -0.12 | -0.16 | +0.23 |

All below `2/sqrt(18) = 0.47`. So a 150 s sub-window already exceeds the integral time scale
of the footprint metrics, and 18 of them are 18 samples rather than one smeared one.

Per-sub-window scatter: peak `x` = 347.6 +/- 70.6 m; the centroid is far noisier
(sd 3206 m) because a single sub-window's tail can be dominated by one long trajectory.

Convergence against a held-out 9-sub-window reference (peak 330 m, centroid 1307 m):

| n sub-windows | sampling time | \|d peak\| p90 | \|d centroid\| p90 | 80% overlap |
|---|---|---|---|---|
| 1 | 2.5 min | - | - | 33.5% |
| 2 | 5.0 min | - | - | 39.8% |
| **3** | **7.5 min** | **60 m (1 cell)** | 893 m | 43.5% |
| 5 | 12.5 min | 60 m | 452 m | 48.3% |
| 7 | 17.5 min | 60 m | 292 m | 52.0% |
| 9 | 22.5 min | 60 m | 120 m | 54.7% |

**CORPUS DESIGN PARAMETER**

- **Peak location** is stable to one grid cell (60 m) at the 90th percentile with
  **n = 3 sub-windows = 7.5 min** of sampling. It does not improve past that — the residual
  60 m offset between window halves is *systematic*, tracking the residual spin-up drift,
  not sampling noise. More averaging will not remove it; a stationary spin-up would.
- **Centroid** needs **n > 9, i.e. > 22.5 min**, to hold 100 m at the 90th percentile, and
  is still improving at n = 9. The centroid is tail-dominated and is the expensive metric.

So a 30-min sampling window is **comfortably sufficient for the peak and marginal for the
centroid**, and the lever that matters is sampling *time* within one run, not the number of
runs — which reverses the first pass's conclusion. The first pass reached "size the corpus
by runs, not samples" from a half-vs-half comparison at a grid where the overlap metric was
sitting on its own noise floor. With the floor now separated from the signal, sub-windows of
one run are demonstrably independent samples and buy convergence directly.

---

## Stage 6 — Real surface, both directions ✅ PASS

Two production windows from the same spun-up flat state, each restarted onto the real
rotated surface, given a 20-min adjustment, then sampled for 30 min at 5 s. Both at
`dt = 0.025 s` (the slope-amplified value), both clean: **`k0/k1` = 0.780 westerly and
0.771 easterly**, `RUN OK`, 361 dumps each, 13.0 GB per window.

Projection before launching, per the hard rule: 72,000 steps at the measured 0.0066 s/step
plus 361 dumps of IO = **10.3 min**. Measured 11.5 min. Every run in this pass was under
12 min; nothing was launched that projected past 45.

### The land-cover attribution — this is the gate

Footprint-weighted share, accumulated from the touchdowns themselves in LES index space
(unblurred, so no grid-resolution smearing), against each class's share of domain area:

| | westerly (from 270 deg) | easterly (from 90 deg) |
|---|---|---|
| **solar array** | **7.32%** of footprint / 0.71% of area = **10.3x** | **0.04%** / 0.71% = **0.06x** |
| **open water** | 0.03% / 0.64% = 0.05x | **35.16%** / 46.52% = 0.76x |
| grass | 92.65% / 98.64% | 64.80% / 52.77% |
| where the array is | 150-240 m **upwind** | 150-240 m **downwind** |
| where the water is | 840-1050 m **downwind** | 840-3300 m **upwind** |

**The array takes 10.3x its area share when it is upwind and 0.06x when it is downwind.**
That is the Stage 6 gate, and it is a mirror image rather than a single-sided result: the
same patch, the same tower, the same spun-up state, flipped by 180 degrees of rotation. The
independent `stage6_compare` window metric agrees for the westerly (13.3% of the footprint
in the array window against 7.1% over flat ground, **1.86x**) and reports exactly **0.00%**
for the easterly.

This is what the first pass could not test. The array had been specified by an *upwind
distance*, so it silently followed the wind and was upwind in every case by construction.
Defining it geographically is what makes the 0.06x number possible, and the 0.06x number is
what proves the rotation is right.

**The water share is quantitatively predicted, not merely present.** Water occupies the
840-3300 m upwind band; that band carries **39.5%** of the LES footprint (Kljun would give
it 30.8%). Within the band the water fraction is 81.8% by area but ~0.90 weighted toward
the domain centreline where the footprint is strongest. So the expectation is
`0.395 x 0.90 = 35.5%`, and the measurement is **35.2%**. The water is under-represented
relative to its raw area share (0.76x) for the obvious reason: it begins 840 m upwind, well
past the footprint peak at 210 m, on the falling limb.

### Kljun comparison (secondary)

| | flat | westerly | easterly |
|---|---|---|---|
| peak `x`, LES | 390 m | 510 m | **210 m** |
| peak `x`, Kljun | 210 m | 210 m | **270 m** |
| centroid `x`, LES / Kljun | 1263 / 788 | 969 / 778 | 1084 / 891 |
| 80% area, LES / Kljun (ha) | 45.4 / 26.6 | 23.4 / 27.4 | 31.7 / 33.5 |
| 80% overlap vs Kljun | 36.9% | **48.1%** | 34.3% |
| integral | 0.805 | **1.452** | 0.858 |
| receptor mean `w` (model frame) | -0.038 m/s | **+0.064 m/s** | **-0.109 m/s** |
| `u*` | 0.327 | 0.346 | 0.276 |
| `h` | 616 m | 1094 m | 723 m |

Both terrain cases agree with Kljun **better** than the flat case does — the easterly peak
lands within one grid cell of it. That is not evidence the terrain runs are more correct; it
is a reminder that at 88% sub-grid the peak location is set by the closure, and the closure
is being pushed around by whatever the local `u*` and `h` happen to be.

**The integral no longer runs away, and the residual tracks the receptor's mean vertical
velocity.** The `t_back` sweep was repeated on both terrain windows with the wrap cap on
(streamline frame, `max_disp` = one domain length; wrapped fraction 0% throughout):

All three runs below are on the **new** grid, so they are directly comparable:

| `t_back` (s) | westerly, `w_bar = +0.064` | easterly, `w_bar = -0.109` | flat, `w_bar = -0.038` |
|---|---|---|---|
| 300 | 0.909 | 0.773 | 0.516 |
| 600 | 1.305 | 0.981 | 0.702 |
| 900 | 1.403 | 0.804 | 0.781 |
| 1500 | 1.202 | 1.022 | **0.821** |
| concentration integral at 900 s | **18.5** | 10.5 | 8.3 |
| 80% source area extends to | 2370 m | 3870 m | 3810 m |

**All three saturate; none climbs.** That is the wrap-around fix (Item 2) holding on the new
grid and over real terrain. The **westerly settles ~30-40% high**, and it is the case with
mean *ascent* at the receptor, nearly twice the concentration integral of the other two, and
by far the **narrowest** footprint.

That combination is the signature of advective non-closure rather than an estimator bug.
The streamline rotation removes 98-99% of `w_bar` from the **weight** — that was Item 1, and
the raw-vs-mean-removed columns show the term it removes is worth 0.5-2.9 here, so the
rotation is doing real work. But nothing can remove `w_bar` from the **transport**: at a
receptor in persistent mean ascent the sampled air arrived from below, spent longer near the
surface (hence the 18.5), and genuinely carries more surface influence than a
horizontally-homogeneous flux footprint assumes. The turbulent flux at such a receptor is
not the surface flux — which is precisely why eddy covariance is hard over complex terrain.

It is now bounded, signed, reproduced in two directions and separated from the flat control,
rather than being a single unexplained 1.64.

**The other two saturate ~18% LOW, and that is the domain being too short for the
footprint.** The wrap cap retires a trajectory once it has travelled one streamwise domain
length (4380 m), so the estimator can only ever recover the influence lying inside that
distance. At this grid the flat 80% source area already extends to **3810 m** and the
easterly to **3870 m** — they barely fit — so a fifth of the influence is beyond the cap by
construction. The narrow westerly case (2370 m) is the one that does not under-integrate.

The weight floor was tested as the alternative and is **not** the cause: a 4x reduction
(`w_floor` 0.02 -> 0.005) moves the flat integral only 0.781 -> 0.815, about 4 points of the
18. Nor is it truncation by `t_back`, which has flattened by 900 s.

This is a **design constraint the corpus has to respect**, not a bug: *the streamwise domain
must be long enough to contain the footprint, or the wrap cap truncates it and the
alternative — no cap — double-counts.* It is also self-limiting: the footprint is
artificially broad here precisely because 88% of `sigma_w^2` is sub-grid (Kljun's 80% area
ends at 1410 m against the LES's 3810 m). A grid that passes the sub-grid gate should pull
the footprint back inside the existing 4380 m and close this on its own.

### The terrain footprints are noisier, and the ensemble result says exactly why

Half-vs-half 80% overlap is 30.6% (westerly) and 41.2% (easterly), against 59.2% flat. That
is **not** a terrain effect. The terrain windows are 1800 s long and `t_back = 900 s` eats
the first half, leaving 900 s of releases — one third of the flat window's 2700 s. Each
half is therefore ~3 sub-windows, and Item 4's convergence table puts the centroid p90 at
**893 m** for n = 3. The measured half-vs-half centroid difference is **906 m**. The
ensemble curve predicts the terrain scatter to within 1.5%.

**Consequence for the corpus:** a production window needs `t_back` plus the sampling time,
not just the sampling time. At `t_back = 900 s` and the 22.5 min the centroid needs, that is
a **37.5 min window**, not 30.

---

## Where this leaves the project

**Everything in the pipeline works.** Restart is bitwise; output volume is met by
configuration; the LPDM is well-mixed in both time directions at the counting-noise floor;
the wrap-around double-count is found and fixed; the reference frame is the one an EC tower
reports in; the footprint responds to a roughness patch by 10.3x when the patch is upwind
and 0.06x when it is downwind; open water is detected by measurement and carries the share
the geometry predicts to within 0.3 points. The estimator, the surface ingestion, the
rotation, the restart chain, and the analysis are all doing what they should.

**One thing blocks the science, and it is resolution.** At `dx = dy = 30 m`, 88.3% of the
vertical velocity variance at a 30 m receptor is sub-grid. The near-field footprint is
therefore manufactured by the Langevin closure rather than resolved by the LES, and no
amount of vertical refinement fixes it — the sub-grid fraction collapses onto
`Delta = (dx dy dz)^(1/3)`, so the 40% gate is a statement about `dx`.

That is the decision to make next, and it is a project-level one:

| option | `Delta` | sub-grid | spin-up cost | consequence |
|---|---|---|---|---|
| stay at 30 m | 19.8 m | 88% | 0.9 h | near-field footprint is closure output, not LES output |
| `dx = 15 m` | 12.4 m | 65% | 5 h | halves the closure's share; still dominant |
| **`dx = 10 m`, `dz_sfc = 6 m`** | **8.4 m** | **39%** | **23 h** | passes the gate |
| **isotropic 8.6 m** | **8.6 m** | **40%** | **20 h** | passes the gate, fewer cells |

The 20-23 h figures are spin-up only, chained in 30-35 segments of under 45 minutes each,
and they are per `(stability, wind speed)` bin — the per-direction production runs stay
cheap because the corpus design shares the spun-up state. That is the trade to weigh: the
spin-up is the whole cost, and it is paid once per bin, not once per sample.

**Two smaller items follow from the measurements rather than from the plan:**

- **A production window is `t_back` + sampling time.** With `t_back = 900 s` and the 22.5 min
  the centroid needs, that is 37.5 min, not 30. A 30-min window yields 21 min of releases.
- **Stage 2 stationarity is not reached at 6.4 h** (`u*` still at -2.25 +/- 0.27 %/h). It
  contributes a systematic 60 m peak offset that no amount of averaging removes. Worth
  fixing at whatever grid is chosen, since it is the one error that ensemble size cannot
  touch.
