# Glossary

| term | meaning here |
|---|---|
| **array share** | the fraction of the footprint's (positive) mass that is on the solar array's 44 cells. The site-specific quantity the emulator exists to get right. Quoted in percentage points (pp) |
| **asinh space** | `y = arcsinh(x / s)`, the transform both emulators work in. `s` is the median over training records of each record's peak. Signed, so negative cells survive |
| **asymptote** | the value the footprint integral tends to over an infinite domain: `1 − z_m/z_i` (Steinfeld et al. 2008), not 1, because the fraction `z_m/z_i` of the column below the receptor never crosses it. 3.75% at 30 m in an 800 m layer |
| **adjustment** | the first 1800 s of a case, in which the seed's turbulence adapts to the case's own forcing before the sampling window opens |
| **base angle** | one of six geostrophic headings (0°, 15°, …, 75°) a seed is spun at. 90° rotations give the other three quadrants exactly |
| **CFL_3d** | `c·dt·sqrt(2/dx² + 1/dz_sfc²)` with `c = 347.2 m/s`. Production 1.3502. FastEddy has no CFL machinery, so `dt` is chosen against a measured accuracy boundary |
| **CFM** | conditional flow matching: the generative emulator, a U-Net velocity field from the noised Kljun prior to the LES target |
| **closure** | the sub-grid `σ_w` model in the LPDM: a MOST-anchored floor weighted by the sub-grid fraction, with `eps` scaled to preserve `T_L` |
| **composite** | geometric mean over five production-metric ratios of median|error_model| / median|error_Kljun|. Below 1 beats Kljun |
| **cone** | the wind-aligned mask `x' ≥ 0`, `|y'| ≤ max(8·σ_y(x'), 90 m)` that removes periodic wraparound from the target |
| **corpus** | the 1366 (input, target) pairs. `corpus_cone.h5` is the training set |
| **DRIFTING / INDETERMINATE** | stationarity-gate verdicts: a trend outside its limit by more than 3 SE, or a threshold within 3 SE of the measurement. INDETERMINATE is the library's normal state |
| **FFP** | Kljun et al. (2015) Flux Footprint Prediction, v1.42, vendored unmodified |
| **FiLM** | feature-wise linear modulation: the six scalars produce a per-channel scale and shift after each block |
| **FNO** | Fourier neural operator: the deterministic emulator, predicting a residual on Kljun |
| **floor** (realisation floor) | how much two footprints of the same conditions differ by turbulence alone: two windows of one run, two re-runs or half-vs-half release ensembles. Any metric is read against it |
| **G2b, G3b** | per-case sanity checks: integral in [0.6, 1.5]. LES peak distance over Kljun peak distance in [0.4, 2.5]. Flags, never exclusions |
| **Gate** | a pass/fail check on an artifact. Named A1 (water share), B6 (rotation equivariance), C1 (stationarity), C2 (bit-for-bit restart), D1 (well-mixed), etc. |
| **h**, **z_i** | boundary-layer depth. The corpus input `h` is the TKE peak-fraction estimator. The stationarity gate uses a fixed 0.01 m²/s² threshold. They differ by 7–21% |
| **hand-off** (in-process, the ring) | patch 0005: the LES stages each snapshot of the window to host RAM and the LPDM consumes it, so the window is never written to disk |
| **HRRR** | NOAA's 3 km High-Resolution Rapid Refresh. Its hourly analyses force every case |
| **k0/k1** | the ratio of first-level to second-level `w` variance. Below 1 (about 0.27) when `dt` is inside the accuracy boundary, near 9 when it is not |
| **Kljun** | the FFP. Also shorthand for its footprint raster, the emulators' input and baseline |
| **L**, **inv_L** | Obukhov length and its inverse. `L` is ±inf at neutral, so loaders use `inv_L` |
| **LPDM** | the backward Lagrangian particle dispersion model (`lpdm/`), this project's own |
| **negative lobe** | cells where the signed footprint is negative (downward flux from a source that the sensor sees as a sink). Physical, unclipped, 1.6% of |f| after the cone |
| **N_WINDOWS** | footprints per case. 1 in the corpus |
| **pass** | one development campaign (third through ninth) on a fixed configuration, with pre-registered gates |
| **receptor** | the release point of the backward LPDM: 30 m above bare ground, 28.5 m above the raised array surface |
| **rung** | a row of the seed library: a regime and a depth (`nbl-shallow`, `nbl-deep`, `cbl-shallow`, `cbl-mid`, `cbl-deep`) |
| **seed** | a pre-spun, flat, doubly-periodic turbulence state a case restarts from. Never training data |
| **σ_w floor** | the closure's lower bound on vertical velocity variance near the surface, weighted by the sub-grid fraction |
| **split** | train (2021–2023 and four 2026 months), val (2024), test (2025), by calendar month |
| **stub** | a run with the LES and LPDM replaced by an analytic footprint, stamped `meta.stub` and refused as a corpus record. For CPU checks of the orchestration |
| **t_back** | how far back in time a particle is followed: 900 s |
| **touchdown** | a particle's contact with the surface. Binned by LES column with cloud-in-cell deposition and weighted by the surface-normal approach rate |
| **virtual heat flux** | `w'θ_v'`, what buoyancy responds to in a dry run. The class table converts sensible-flux ratios with the Bowen ratio |
| **wraparound** | periodic-boundary artifact: touchdowns folded modulo the domain per axis. Removed by the cone. Not signal |
| **x80**, **overlap80** | the along-wind distance containing 80% of the crosswind-integrated footprint. The Jaccard overlap of two footprints' 80% source areas |
