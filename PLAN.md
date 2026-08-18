# Staged Plan — First Working Footprint

Goal of this plan: **one FastEddy run producing one backward-LPDM flux footprint.**
Not a corpus. Not a trained model. One end-to-end pass.

Each stage has a gate. **Do not proceed past a failed gate.** Commit at each pass.

---

## Stage 0a — Repos and container

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

## Stage 0b — Confirm precision

**Do:**
- `grep -rn "typedef.*\(float\|double\)" Source/ | head`
- `grep -rn "NC_DOUBLE\|NC_FLOAT" Source/ | head`
- `ncdump -h <tutorial_output>.nc | head -30`

**Gate:** Precision documented in writing (append the finding to PROJECT_BRIEF.md).

**Note:** fp32 is expected and fine. This is to know, not to change.

---

## Stage 1 — Minimal run at the new grid

The single most important gate. Everything downstream is contingent on it.

**Do:**
- Flat terrain, uniform roughness, neutral, doubly periodic
- Grid 434 x 146 x 122 @ 10 m, thread block 4 x 4 x 16
- Short run — enough timesteps to prove the hydro core executes, not to converge

**Gate:** Clean exit, no `too many resources requested for launch`, and a **wall-clock
number per simulated second**.

**If it fails:** try 4 x 4 x 8. Report the register count per kernel
(`nvcc -Xptxas -v`) before trying anything else. Do not start tuning other parameters.

**Record:** wall clock, memory high-water mark, timestep size. Extrapolate to a 3-hour run
and write the estimate into the commit message. If a single run exceeds ~4 hours, stop and
flag it — the corpus arithmetic needs revisiting before any more work happens.

---

## Stage 2 — Vertical stretching and spinup

**Do:**
- Add vertical grid stretching: 10 m near surface, growing above ~500 m, top ~2.5 km
- Add Rayleigh damping layer at the top
- Run long enough to reach a converged turbulent state

**Gate:** Turbulence is statistically stationary. Check via domain-averaged TKE plateau and
resolved vertical velocity variance profile. Compare the profile against the FastEddy
NBL/CBL validation cases in the docs — shape should be recognizable.

---

## Stage 3 — Output configuration

**Do:**
- Configure 3-velocity-component output, **z < 400 m only**, ~5 s cadence
- Use raw binary output mode, not NetCDF, if it's meaningfully faster
- Measure actual I/O overhead as a fraction of compute

**Gate:** Overhead measured and recorded. If it's under ~10%, no optimization work happens —
move on. Do not engineer around a bottleneck that hasn't been demonstrated.

**If overhead is large:** in order — fp16 velocities, then lossy compression (zfp/blosc,
~10:1 on smooth LES fields), then reduce cadence. Not before measuring.

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

**Gate:** Over flat uniform terrain in neutral conditions, the result should be
**close to Kljun**. That is the whole point of this stage — a homogeneous surface is where
the analytical model is valid, so agreement validates the pipeline.

Disagreement here is a pipeline bug, not a scientific finding. Do not proceed until the
flat/neutral case reproduces Kljun to within a sensible tolerance on peak location and
upwind extent.

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

**Use plan mode (`/plan` or Shift+Tab) for Stages 0a, 4, and 6.** These are the
stages where a wrong direction is expensive to unwind — container/repo structure, the LPDM
SGS closure, and the rotated GIS preprocessing. For Stages 1-3 and 5, the path is narrow
enough that planning overhead isn't worth it.

Note: the assistant's `/plan` is a permission mode. It does not load this file. Reference
PLAN.md by name in the prompt.
