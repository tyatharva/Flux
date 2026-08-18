# Staged Plan — First Working Footprint

Goal of this plan: **one FastEddy run producing one backward-LPDM flux footprint.**
Not a corpus. Not a trained model. One end-to-end pass.

Each stage has a gate. **Do not proceed past a failed gate.** Commit at each pass.

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

**Gate:** Precision documented in writing (append the finding to CLAUDE.md). **Done** —
hardwired fp32, `NC_FLOAT` in the writer, `MPI_FLOAT` in halo exchange, no build switch.
Recorded in CLAUDE.md Conventions.

---

## Stage 1 — Minimal run at the new grid

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

---

## Stage 5 — First footprint, flat and neutral

**Do:**
- Backward release from the receptor at 30 m
- Touchdown weighting per Thomson/Flesch
- Produce a 2-D footprint

**Gate 1 — agreement with Kljun.** Over flat uniform terrain in neutral conditions the
result should be **close to Kljun**. That is the whole point of this stage — a homogeneous
surface is where the analytical model is valid, so agreement validates the pipeline.

Disagreement here is a pipeline bug, not a scientific finding. Do not proceed until the
flat/neutral case reproduces Kljun to within a sensible tolerance on peak location and
upwind extent.

**Gate 2 — the irreducible error floor.** Run the *same case twice* and compare the two
footprints.

FastEddy is not bitwise reproducible (CLAUDE.md Conventions), so two runs of one
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
- Taper terrain and land cover at both wrap seams
- Add the solar array as a bulk patch (albedo, z0, displacement height)
- One CONUS404-derived sounding, one wind direction

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

**Use Claude Code plan mode (`/plan` or Shift+Tab) for Stages 0a, 4, and 6.** These are the
stages where a wrong direction is expensive to unwind — container/repo structure, the LPDM
SGS closure, and the rotated GIS preprocessing. For Stages 1-3 and 5, the path is narrow
enough that planning overhead isn't worth it.

Note: Claude Code's `/plan` is a permission mode. It does not load this file. Reference
PLAN.md by name in the prompt.
