# Development history

Nineteen days from the first FastEddy build to the test-split result, in nine passes on four
grids. Each pass had pre-registered gates; each page in this section keeps every measured
number and every decision of its source write-up, including what was tried and why it failed.
Numbers in these pages are superseded on absolutes by the current configuration pages; they
are the record of how each conclusion was reached.

## Timeline

| dates (2026) | pass | configuration | what it settled |
|---|---|---|---|
| 08-17 to 08-19 | [Stages 0–2](stages-0-2.md) | 10 m and 30 m development grids, receptor 30 m, surrogate tower | FastEddy builds and validates; `dt` has an accuracy limit below the stability limit; the LPDM's reverse drift, estimator constant and well-mixed test; the wraparound double-count and its cap; the surface goes in through the restart file; the near field at `Δ ≈ 20 m` is closure output |
| 08-20 to 08-21 | [Third pass](pass-3.md) | 186² at 24 m, receptor 30 m, surveyed tower, USGS 3DEP + WorldCover | the thread block (17%); the static north-up surface; the array is a 120 × 350 m rectangle with the tower inside; the `σ_w` deficit and the MOST floor; `ioLPDMmode` |
| 08-21 to 08-22 | [Fourth pass](pass-4.md) | same | a window is 30 min + `t_back`; the raster is the LES grid; CONUS404 as climatology; the convective regime and per-cell heat flux; the floor's drift bug; `t_back = 900 s`; the array at 48% on a convective northerly |
| 08-22 | [Fifth pass](pass-5.md) | 122³ at 16 m, receptor 10 m | all gates pass; `L ≥ 2 z_i` not binding at 10 m; three gates specified wrong (`u*`, bit equality, domain-mean `k0/k1`); the virtual-flux correction; the floor not well mixed convectively |
| 08-24 | [Sixth pass](pass-6.md) | same | the closure fixed by sub-grid-fraction weighting after two wrong diagnoses; the retired closure inflated convective shares by up to 18 points; the compaction ratio's sign depends on the closure |
| 08-25 to 08-27 | [Stable regime](stable-regime.md), [seed rungs](seed-rungs.md), [target case](target-case.md) | same | stable collapses at this grid (`L_O/Δ` 3.57); the first seeds; the `z_i` estimator rides the inertial oscillation; INDETERMINATE is normal; the first end-to-end case; the seed library and HRRR forcing designed |
| 08-29 | [Seventh pass](pass-7.md) | 122³ at 24 m, receptor 30 m | the 10 m receptor retired (the peak did not move); the peak moves 144 m at 30 m and it is not the closure; the GPU LPDM accepted; the asymptote `1 − z_m/z_i`; the lake is back |
| 08-30 | [Containment](containment.md), [eighth pass](pass-8.md) | 122³ at 30 m, receptor 30 m | neutral is not contained at 2928 m; 3660 m for 3%; the in-process hand-off, bit-identical; the `dt` boundary re-measured at 1.50–1.55 |
| 08-30 | [Ninth pass](pass-9.md) | same | the official FFP; the streamed hand-off at 12.4 GB; D1 in both regimes; containment PASS; the parity with Kljun lost; `h` measured in the wrong fluid |
| 08-30 to 08-31 | decisions | same | `bl_depth` on the surface-attached layer; `N_WINDOWS = 1`; a 2.0 sim-h ceiling; six base angles; the whole library admitted; splits by month; the Blackwell image |
| 08-31 | [seed library](../les/seed-library.md) | 16 × RTX 5090 | 30 seeds in 0.936 h; `TKE_BL/u*²` was measuring its own references |
| 09-01 | [deployment](../les/deployment.md), [cone mask](cone-mask.md) | 8 × 8 RTX 5090 | 1366 pairs; the wraparound cone with `k = 8` from an empty valley |
| 09-02 to 09-04 | [emulator timeline](emulator-timeline.md) | RTX 4080 | the FNO, the CFM, calibration, the frozen recipe, the test split |
| 09-01 | [unmerged work](unmerged-producer-consumer.md) | | the producer/consumer split and the HRRR-Zarr screen that never merged |

## The decisions that shaped the configuration

1. **Grid and receptor.** 10 m development grid → 30 m development grid (stages 0–2) → 24 m at a
   30 m receptor (passes 3–4) → 16 m at a 10 m receptor for corpus economics (passes 5–6) →
   back to a 30 m receptor on 24 m because the 10 m peak did not move (pass 7) → 30 m spacing
   for containment (pass 8). The receptor height is a resolution decision, not a correction.
2. **The closure.** A MOST-anchored `σ_w` floor supplies what the LES does not resolve at the
   receptor. Its magnitude, not its shape, was the defect; it is weighted by the sub-grid
   fraction and `eps` is scaled with it. Its worth (+8.40 points of convective array share) is
   quoted with every near-field number, and the compaction ratio with its closure.
3. **Forcing.** From a CONUS404 sweep to real HRRR analyses, one case per day drawn from the
   weather, because CONUS404 carries no profiles and a sweep has no rose.
4. **Seeds.** A library of pre-spun states, matched on what they achieved, gated on ratios that
   ride the inertial oscillation, admitted whatever their drift state because a seed is an
   initial condition.
5. **The target.** The raster on the LES's own cells, signed and unclipped, zero-padded to 128²,
   cropped by a cone whose parameter was measured; touchdowns not saved; a residual on the
   official Kljun FFP as the model's anchor.
6. **What was excluded.** Stable conditions (resolution), a second window per case (a
   near-duplicate), static surface channels as inputs (a positional basis), the CNF on the point
   process (reversed), and every item in [ruled out](../reference/ruled-out.md).

## Where the numbers now live

The current configuration is in [configuration](../les/configuration.md), the current
limitations in [limitations](../limitations-and-future-work.md), and the lessons in the
[standing rules](../reference/standing-rules.md) and [FastEddy traps](../reference/fasteddy-traps.md).
The retired scripts, runs and results named at the foot of each history page were removed from
the tree on 2026-09-04 and are kept in the author's offline pre-cleanup archive of 2026-09-04.
