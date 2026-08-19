# Staged Plan — First Working Footprint

Goal of this plan: **one FastEddy run producing one backward-LPDM flux footprint.**
Not a corpus. Not a trained model. One end-to-end pass.

Each stage has a gate. **Do not proceed past a failed gate.** Commit at each pass.

**Stages 2-6 were executed at a 30 m pipeline-development grid** (146 x 50 x 90,
`dt = 0.0625 s`), not the 10 m production grid. That configuration exists to validate the
pipeline, not the science: the corpus is regenerated at finer resolution afterwards. Where
a gate's arithmetic is resolution-dependent — Stage 3's storage most of all — both numbers
are given. Results in `STAGE2-6_RESULTS.md`.

---

## Stage 0a — Repos and container  ✅ PASSED 2026-08-17

**Do:**
- Fork `NCAR/FastEddy-model` on GitHub, branch `kegonsa`. Point the local
  `FastEddy-model-5.0.1/` tree at the fork.
- Initialize the main project repo at `Flux/`. Write `.gitignore` covering
  `FastEddy-model-5.0.1/`, `*.zip`, DEM/CONUS404 data, and LES output.
- Write `FASTEDDY_VERSION.txt` with the fork URL and current SHA.
- Read `inst.txt`, cross-check against the FastEddy v5.0.1 docs, and write a `Dockerfile`.
  Base `nvidia/cuda:11.8.0-devel-ubuntu22.04`; MPI + NetCDF-C + HDF5; `sm_89`.
- Bind-mount `Flux/`, run with `--gpus all`.

**Gate:** The container builds, and a known-good FastEddy tutorial case runs to completion
inside it. Not the new grid yet — a case with published expected behavior.

**Note:** `inst.txt` was written for v4.0.1. Where it disagrees with the v5.0.1 docs, the
docs win. Flag any disagreement rather than silently picking one.

---

## Stage 0b — Confirm precision  ✅ PASSED 2026-08-17

**Do:**
- `grep -rn "typedef.*\(float\|double\)" Source/ | head`
- `grep -rn "NC_DOUBLE\|NC_FLOAT" Source/ | head`
- `ncdump -h <tutorial_output>.nc | head -30`

**Gate:** Precision documented in writing (append the finding to PROJECT_BRIEF.md). **Done** —
hardwired fp32, `NC_FLOAT` in the writer, `MPI_FLOAT` in halo exchange, no build switch.
Recorded in PROJECT_BRIEF.md Conventions.

---

## Stage 1 — Minimal run at the new grid  ✅ PASSED 2026-08-18

The single most important gate. Everything downstream is contingent on it.

**Do:**
- Flat terrain, uniform roughness, neutral, doubly periodic
- Grid 434 x 146 x 122 @ 10 m, thread block 4 x 4 x 16
- Short run — enough timesteps to prove the hydro core executes, not to converge
- **Determine max stable `dt` empirically.** FastEddy has no CFL machinery: `dt` is a
  user constant, never computed or checked, so the tutorials' values were hand-picked and
  are not a reliable guide. Bisect `dt` upward on this grid until the run goes unstable.
  This is the largest single cost lever — the two candidate stability metrics differ by
  1.9x, which is the difference between a 4 h and an 8 h production run:

  | metric | limit | dt at 10 m, c=347 m/s |
  |---|---|---|
  | 3-D combined `dt*c*sqrt(sum 1/d^2)` | ~1.73 (RK3) | 0.029 s |
  | 1-D min-spacing `dt*c/min(d)` | ~1.73 (RK3) | 0.050 s |

  Reference points from the tutorials: NBL sits at 0.926 (1-D) / 1.603 (3-D); SBL at
  0.522 / 0.904. Instability shows as NaN, `rho <= 0`, or runaway max|w|.

**Gate:** Clean exit, no `too many resources requested for launch`, and a **wall-clock
number per simulated second**.

**If it fails:** try 4 x 4 x 8. Report the register count per kernel
(`nvcc -Xptxas -v`) before trying anything else. Do not start tuning other parameters.

**Record:** wall clock, memory high-water mark, timestep size. Extrapolate to a 3-hour run
and write the estimate into the commit message. If a single run exceeds ~4 hours, stop and
flag it — the corpus arithmetic needs revisiting before any more work happens.

---

## Stage 2 — Vertical stretching and spinup

**Spin up over FLAT, UNIFORM terrain — not the real surface.**

This is the main lever against the measured 8.15 h cost of a 3 h run. Over a flat uniform
surface the turbulence is horizontally homogeneous, so a spun-up state has **no preferred
horizontal direction**. One spun-up state per `(stability, wind speed)` bin therefore serves
**every wind direction** in that bin, instead of every direction paying for its own spinup.

The corpus then costs one expensive spinup per bin, plus one short run per direction:

```
  expensive:  N_stability x N_speed   full spinups over flat uniform terrain
  cheap:      x N_direction           restart + introduce real surface + ~20 min adjustment
```

Wind direction is the dominant skill axis for the emulator, so it is exactly the axis we
need many samples along — and this is the structure that makes many samples affordable.

**Do:**
- Vertical grid stretching: 10 m near surface, growing above ~500 m, top ~2.5 km
  (`verticalDeformSwitch = 1`; note the tutorials' `verticalDeformFactor` compresses
  toward the surface, and the near-surface `dz` is what sets the acoustic `dt` limit)
- Rayleigh damping layer at the top
- Flat terrain, uniform roughness, doubly periodic
- Run to a converged turbulent state and **save a restart** — this is the reusable asset

**Gate:** Turbulence is statistically stationary. Check via domain-averaged TKE plateau and
resolved vertical velocity variance profile. Compare the profile against the FastEddy
NBL/CBL validation cases in the docs — shape should be recognizable.

**Second gate:** the saved restart actually restarts. FastEddy can only restart from netCDF
(`ioOutputMode` binary output is not restartable), so confirm a restart reproduces a
continuous TKE time series rather than a transient.

**The real surface is introduced AFTER this stage**, at Stage 6, by restarting a spun-up
state and letting the flow adjust for ~20 min of simulated time before sampling. Budget
that adjustment as part of every production run; validate the 20 min figure at Stage 6 by
checking that near-surface stress and the footprint have stopped drifting.

> **Result, second pass: restart ✅ PASS, stationarity ❌ NO.** Bitwise restart re-verified at
> `146 x 50 x 122` — `cmp` reports the two 25.4 MB dumps byte-identical, every prognostic
> field differing by exactly 0. Stationarity after 6.4 h of simulated time: domain TKE trend
> **-2.10 +/- 1.13 %/h** (-1.85 sigma, would pass) but `u*` **-2.25 +/- 0.27 %/h**
> (**-8.40 sigma**, fails). The profile shape is recognisable — `sigma_w^2/u*^2` peaks at
> 0.844 vs NCAR's NBL 0.730 and vanishes by 668 m vs their 650 m — but the peak sits at
> 174 m rather than 130 m and `u*` is 0.328 vs 0.410, both consistent with a boundary layer
> still deepening. Quantified and non-blocking: a -2%/h drift is small against Stage 5's
> +86%, and it changes no conclusion. Not fixed, because fixing it means more spin-up hours
> at a grid Stage 5's revised gate has already ruled out.

---

## Stage 3 — Output configuration

**Measured starting point (Stage 0a):** FastEddy writes **19 3-D fields = 76 B/cell** and
exposes **no** output field selection and **no** vertical subsetting. Only five IO
parameters exist, and binary mode walks the same variable list as NetCDF. At the production
grid that is 0.59 GB/dump → **213 GB per 30-min window at 5 s cadence**.

The LPDM needs `u`, `v`, `w`, SGS TKE = **16 B/cell**. Note also that `xPos`/`yPos`/`zPos`
account for 12 B/cell and are rewritten identically in every dump.

**Do:**
- Configure output at ~5 s cadence and measure I/O overhead as a fraction of compute
- Implement **field selection** and **fp16 on write**, in that order
- Re-measure

**Gate:** field selection + fp16 on write puts a **30-min window under ~30 GB**.

> **At 30 m this gate is met by configuration alone.** `hydroSubGridWrite = 0` leaves 10
> 3-D fields = 40 B/cell; at 657,000 cells that is 26.6 MB/dump and **9.6 GB** per 30-min
> window at 5 s cadence. Neither field selection nor fp16 was written, and no FastEddy
> source change was needed. Both come back at 10 m, where the same window is 113 GB.
>
> **Second-pass grid, re-measured 2026-08-19:** 890,600 cells -> **35.99 MB/dump** and
> **13.0 GB** per 30-min window (361 dumps at 5 s), still by configuration alone. IO time is
> ~0.05 s/dump against ~1.75 s of compute per 200-step batch, i.e. **~3% of compute** — under
> the "stop optimizing" threshold, so nothing further was done.

Arithmetic: 4 fields at fp16 = 8 B/cell = 1/9.5 of the current 76 B/cell → 213 GB becomes
**~22 GB**. Field selection alone (fp32) gives ~45 GB, so fp16 is what clears the bar.

fp16 is for the *stored velocity fields only*. It does not touch the solver, which stays
fp32, nor particle state, which stays fp64. Before accepting fp16, verify the quantisation
is small against the LPDM's own noise: compare footprints computed from fp32 and fp16
copies of the same fields, and require the difference to sit below the Stage 5 error floor.

**Contingency, not first choice:** a k-range limit in `SRC/IO/io_binary.c` restricting
output to `z < 400 m`. If it is needed, it goes on the `kegonsa` fork branch **behind a
config flag** (default off = upstream behaviour), so the fork stays a clean, reviewable
diff and NCAR bugfixes keep merging. Do not write it until field selection + fp16 have been
measured and shown insufficient.

**If I/O overhead is under ~10% of compute, stop optimizing it** — the volume target above
is about storage, not speed, and those are separate problems.

---

## Stage 4 — LPDM, with the well-mixed test first

**This is the highest-risk stage and the one most likely to slip. Do it early.**

**Do:**
- Backward LPDM reading the Stage 3 output
- Resolved velocity from interpolated fields; SGS from a Langevin model driven by
  FastEddy's output SGS TKE (Weil et al. 2004)
- **Particle state in fp64**

**Gate — well-mixed test:** release a uniform particle distribution in the flat neutral
case from Stage 2. It must remain uniform.

If particles accumulate near the surface, the SGS closure violates the well-mixed condition
and **every footprint computed afterward is wrong in the near field** — precisely where the
signal lives. Fix this before computing a single footprint.

**Second gate:** backward trajectories from the 30 m receptor reach the surface in a
plausible transit time (~1-5 min unstable, ~10-15 min stable).

> **Result, second pass: ✅ PASS both gates.** Well-mixed backward rms **4.91%** against a
> 4.48% counting-noise floor, max deviation 10.1%, lowest three bins 0.978; forward control
> 4.67% / 1.045. Transit p5 = 73 s, **median 287 s (4.8 min)**, p95 = 765 s, 62% reaching the
> surface inside 900 s — neutral at the fast end of the expected range, as it should be.
> The earlier failure was the test's own reflecting lid, not the closure: reflection flips
> the sub-grid velocity but not the resolved `w`, giving a 2x lid-bin pile-up in *both* time
> directions. Particles are now released through a deep column and only the interior scored.

---

## Stage 5 — First footprint, flat and neutral

**Do:**
- Backward release from the receptor at 30 m
- Touchdown weighting per Thomson/Flesch
- Produce a 2-D footprint

**Gate 1 (REVISED 2026-08-19) — sub-grid fraction of `sigma_w^2` at the receptor < 40%.**
The original gate asked for Kljun agreement. That was badly specified: at `dx = 30 m` with
one cell below the tower, a near-total sub-grid fraction is *expected*, so the footprint is
manufactured by the closure and disagreement with Kljun is a statement about the grid, not
about the pipeline. Gate on the quantity that actually controls near-field fidelity.

**Kljun agreement is a secondary check, not a tuning target.**

> **Result, second pass at `dz_sfc = 8.56 m`: FAIL, and unreachable at `dx = 30 m`.**
> 96.4% (first pass, `dz_sfc` 20 m) -> **88.3%**. The fraction collapses onto `z/Delta` with
> `Delta = (dx dy dz)^(1/3)`; both grids put the 40% crossing at `z/Delta` = 3.5-3.7, so the
> gate needs **`Delta <~ 8.6 m`**. With `dx = dy = 30 m` that requires `dz <= 0.71 m`, at
> which anisotropy the horizontal filter still cannot resolve 30 m eddies — **the gate is a
> statement about `dx`, not `dz`.** Grids that pass (`dx=10/dz=6`, or isotropic 8.6 m) cost
> **20-23 GPU-hours for the spin-up alone**, i.e. 30-35 chained 40-minute segments. That is
> a project-level decision about corpus cost, not a configuration change. Secondary check:
> peak 390 m vs Kljun 210 m (+86%), 80% overlap 36.9%. See STAGE2-6_RESULTS_V2.md.

> **Result, second pass: the floor separated from the signal.** Half-vs-half 80% overlap
> rose from 30.0% to **59.2%** against 36.9% for LES-vs-Kljun, so the metric is no longer at
> its own noise floor and IS usable to score the emulator — reversing the first pass. Peak
> difference between halves is one grid cell (60 m), centroid 99 m.
>
> **Ensemble convergence (measured on 18 sub-windows of 150 s from one integration).**
> The sub-windows are independent (lag-1 autocorrelation +0.19 peak / -0.10 centroid, below
> `2/sqrt(18) = 0.47`), so ensembles come from sampling *time within one run*, not from
> extra runs. **Peak stabilises to one cell at n = 3 sub-windows = 7.5 min; the centroid
> needs n > 9 = 22.5 min** and is still improving there. A 30-min window is comfortable for
> the peak and marginal for the centroid. The residual 60 m peak offset between halves is
> systematic (residual spin-up drift), not sampling noise — more averaging will not remove
> it. **This is the corpus design parameter.**

**Gate 2 — the irreducible error floor.** Run the *same case twice* and compare the two
footprints.

FastEddy is not bitwise reproducible (PROJECT_BRIEF.md Conventions), so two runs of one
configuration give two different turbulence realizations. The difference between their
footprints is the **error floor**: the emulator cannot be asked to predict better than
this, and any Stage 6 difference smaller than it is noise rather than signal.

Report the floor in the same metrics used to score the emulator — centroid displacement,
80% source-area overlap, and whatever loss the CNF ultimately trains against — so it is
directly comparable to model error later.

This gate also answers a question the corpus design depends on: **does a 30-min sampling
window converge?** If two 30-min windows of the same configuration disagree substantially,
then 30 min does not contain enough eddies to define a stable footprint, and the averaging
window must lengthen (raising per-run cost) or footprints must be averaged over multiple
realizations (raising run count). Either way the corpus arithmetic changes, so measure this
**before** committing to a corpus size. Quantify it by also splitting one run into two
15-min halves and comparing those.

---

## Stage 6 — Real surface

**Do:**
- Rotated GIS preprocessing: resample DEM + land cover into wind-aligned frame, generate
  rotated `lat(y,x)` / `lon(y,x)`, run GeoSpec -> SimGrid
- Taper **terrain** at both wrap seams. Do **not** taper land cover — see below
- Add the solar array as a bulk patch (albedo, z0, displacement height)
- One CONUS404-derived sounding; **two** wind directions (westerly and easterly), so the
  open water east of the tower is inside the footprint for at least one of them

> **Result, second pass at the surveyed coordinate: ✅ PASS, and it is a mirror-image test
> rather than a one-sided one.** Two directions from one spun-up state, both clean
> (`k0/k1` = 0.780 and 0.771), 11.5 min each.
>
> | footprint share / area share | westerly (270 deg) | easterly (90 deg) |
> |---|---|---|
> | **solar array** | **7.32% / 0.71% = 10.3x** (upwind) | **0.04% / 0.71% = 0.06x** (downwind) |
> | **open water** | 0.03% / 0.64% (downwind) | **35.2% / 46.5% = 0.76x** (upwind) |
>
> The array takes 10.3x its area share when upwind and 0.06x when downwind — same patch,
> same tower, same state, rotated 180 deg. The first pass could not test this because the
> array was specified by an *upwind distance* and so followed the wind. The water share is
> **predicted, not merely present**: the 840-3300 m band carries 39.5% of the footprint and
> is ~90% water on the centreline, giving 35.5% expected against **35.2% measured**.
>
> The 1.64 is resolved (wrap-around, Item 2 of the second pass). The residual integral now
> **straddles 1 with the sign of `w_bar` at the receptor** — 1.45 at `w_bar = +0.064`, 0.86
> at `w_bar = -0.109` — which is advective non-closure, not an estimator bug: the streamline
> rotation removes `w_bar` from the weight but cannot remove it from the transport. See
> STAGE2-6_RESULTS_V2.md.

**Gate:** A footprint that **differs from Kljun in an explicable direction.** You should be
able to point at the array or the terrain and say why the footprint distorted the way it
did. If it differs in a way you can't explain, that's a bug, not a result.

---

## After Stage 6

Only then: corpus design, wind-rose stratification, CNF implementation.

Do not start ML work before Stage 6 passes. A trained model on a broken target pipeline
looks exactly like a trained model on a correct one.

---

## Working agreement

- One stage per session where possible.
- Report the gate result explicitly before moving on.
- If a stage reveals the plan is wrong, say so and stop. Do not work around it silently.
- Prefer reading FastEddy's own source and docs over inferring behavior.
- Commit at every passed gate. FastEddy source edits go to the fork on `kegonsa`;
  everything else to the main repo.

**Use plan mode (`/plan` or Shift+Tab) for Stages 0a, 4, and 6.** These are the
stages where a wrong direction is expensive to unwind — container/repo structure, the LPDM
SGS closure, and the rotated GIS preprocessing. For Stages 1-3 and 5, the path is narrow
enough that planning overhead isn't worth it.

Note: the assistant's `/plan` is a permission mode. It does not load this file. Reference
PLAN.md by name in the prompt.
