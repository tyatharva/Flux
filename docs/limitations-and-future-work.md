# Limitations and future work

State these wherever the corpus or the emulator is described.

## Limitations

1. **The model receptor is 30 m. The instrument is at 10 m.** The emulator predicts a
   footprint the physical tower does not measure. This was a resolution choice, not a
   correction. At 10 m the footprint peak did not respond to meteorology (48 m in all
   three targets, max/min 1.00×) because the near field was closure output, not LES output.
   Tower comparisons use a Monin-Obukhov (MOST) translation whose stable branch is an
   upper bound.
2. **Only about 15% of the corpus has the array signal**, and the wind rose is skewed
   away from the directions that do. The array is in the footprint almost only for
   northerly flow. 202 of 1366 records (14.8%) have an array share above 5%. The median is
   0.49%. An aggregate metric over all records is dominated by cases where Kljun and the LES
   agree by construction. Weight the loss, or report the northerly subset separately.
3. **Gate A1 (water share) fails.** 11.58% worst case over corpus regimes against a 10%
   threshold. The cause is the site, not the domain. Kljun's `x90` barely moved between
   domains (1665 vs 1615 m), so the physical footprint is the same. What changed is that a
   3660 m domain contains it, where a 2928 m domain was replacing the lake with a periodic
   copy of its own land. The old PASS was truncation. E and NE are about 20% of the rose and
   have 6–12% water. The rest have about 0%.
4. **The LES loses tail that Kljun does not at 3660 m.** LES retains 0.756 of its asymptote
   against Kljun's 0.929, where at 2928 m the two were at parity (0.874 vs 0.867). An
   LES-vs-Kljun comparison on this domain is no longer the fair one that parity allowed.
5. **There are no stable cases, and the emulator is undefined there** (about 44% of QC'd
   hours). A stable seed was healthy for 1.75 simulated hours and then collapsed. The cause is
   resolution, measured at the healthy dump an hour before anything looked wrong:
   `L_O/Δ = 3.57` at the receptor against 318 neutral, a factor of 89. GABLS1 runs that regime
   at `dx = 6.25 m`, 17× the cells. Weakening the stratification was tried and failed the same
   way. See [stable regime](history/stable-regime.md).
6. **The near field is closure-dominated**: 52.5% sub-grid convective, 86.4% neutral. Quote
   the anchor-sensitivity band (46–66% shape L1 against a 38% sampling floor) with any
   near-field number.
7. **Every seed is DRIFTING or INDETERMINATE on at least one limit, and that is the normal
   state.** `TKE_BL/u*²` and `z_i` decorrelate on the eddy turnover, not on the dump interval,
   so `n_eff` saturates at 3–5 whatever the scoring window. Dumping more often cannot help,
   because the run is short. The neutral rungs are short at 2.0 simulated hours.
8. **The GPU LPDM is validated but is not the production integrator.** Host memory therefore
   has a floor at the 12.0 GB fp16 field cache rather than at one or two snapshots.
9. **The training target is `corpus_cone.h5`, and the wraparound it crops is a boundary
   artifact, not signal.** `corpus_raw.h5` is retained byte-identical to what the pipeline
   produced. The cone is an operational cleanup and not an integral correction. The median
   |error| against the `1 − z_m/z_i` asymptote goes 0.1443 → 0.1467, and 720 of 1366 records
   end up below it. What inflates the integral is still open. The advection non-closure is the
   candidate that fits (the departure tracks `w̄` at the receptor with the right sign:
   subsidence 1.497×, updraft 0.916×). Testing it needs `w̄` per record, which the corpus does
   not store.
10. **Seed grouping in the split is not settled.** `bin/seed_leakage.py` found no fingerprint
    at the un-confounded receptor (sharing a seed made two cases *less* alike, 2.47 vs 1.55
    floors). With n = 1 same-seed pair there it is weak evidence, not a reason to drop
    grouping. Splits are by calendar year, which keeps whole synoptic systems on one side.
11. **The emulator's spread is under-dispersed where it matters most.** After the global
    temperature fit (τ = 1.19) the array-in-view group still shows about 30% residual
    under-dispersion, and a group-specific τ is unstable across folds (1.02 vs 1.56 on n ≈ 20).
    The frozen recipe ships τ = 1. See [calibration](emulator/calibration.md).
12. **The corpus has gaps.** 166 days failed. Six months are empty: 2021-12, 2023-07, 2023-10,
    2024-01 (val), 2024-04 (val), 2026-08. 2021-06 and 2022-04 are partial. Machine 3 lost all
    8 GPUs to one fault 42% into its run (stage 7 timed out waiting for the LES to stage into
    `/dev/shm`. The root cause is unresolved and the LES logs are gone). This was checked and
    is not an input-space hole: 84–93% of a missing month's cases fall inside the retained
    months' p5–p95 on every scalar.

## Future work

- **Top up the corpus.** The failed days are named in the machine manifests
  (`corpus/provenance/manifests/`), and the hour draw is seeded from the date, so a re-run
  reproduces exactly the cases that would have been there. It is one command per machine. See
  [deployment](les/deployment.md).
- **Deposit the unfolded displacement at generation time.** This removes the wraparound at
  its source instead of masking it afterwards. It needs a corpus regeneration, which is why it
  is recorded here rather than done.
- **Store `w̄` at the receptor in every record**, so the advection non-closure can be tested
  as the cause of the inflated integral.
- **Move the GPU LPDM to the production path.** It is validated against the CPU integrator
  (`bin/test_gpu_lpdm.py`). What remains is the ring buffer in device memory, which would
  drop host residency from the 12 GB cache to a few snapshots.
- **Return the scored series from seed runs, not only the verdicts fitted to it.** This has
  been outstanding since the seed library was spun.
- **A second window per case** (`N_WINDOWS = 2` is still supported) would give a
  spread-estimating model an in-distribution target for the realisation floor.
- **Stable cases need a finer grid.** About `dx = 6 m` at this site, 17× the cells. Not
  affordable today. The emulator should report "undefined" rather than extrapolate there.
- **Compare against the tower.** The MOST translation from 30 m to 10 m and its stable-branch
  upper bound are described but have not been tested against measured fluxes.
- **Send two patches upstream.** The moisture-off subsidence guard and the CUDA 13
  `cudaDeviceProp` guard fix real FastEddy defects. See
  [FastEddy and the patches](les/fasteddy-and-patches.md).
