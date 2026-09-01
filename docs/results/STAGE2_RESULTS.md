# Stage 2 results — SUPERSEDED

This documented a 10 m spin-up at `dt = 0.0275 s`, a timestep later shown to sit **above
the accuracy-CFL limit** (~1.64) though below the stability limit (~1.79). Runs in that
window complete, exit 0, print no warning, and replace the resolved near-surface vertical
velocity with grid-scale acoustic noise. The `CONDITIONAL PASS` recorded here was therefore
conditioned on an artifact that turned out to be the timestep.

The finding itself is preserved in PROJECT_BRIEF.md ("STABILITY AND ACCURACY ARE DIFFERENT
LIMITS") and was reported upstream as
[NCAR/FastEddy-model#134](https://github.com/NCAR/FastEddy-model/issues/134).

**Current results: `docs/results/STAGE2-6_RESULTS.md`** — Stages 2 through 6 at the 30 m
pipeline-development grid, `dt = 0.0625 s`, `CFL_3d = 1.491`.
