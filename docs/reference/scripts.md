# Scripts

One line per entry point, grouped by where it belongs in the pipeline. Every Python script
prints usage with `--help`. Every one runs in the analysis image (`docker/pyrun.sh <script>`)
except the `ml/` and `ml_cfm/` modules, which run in the `LESNet` conda environment
(`python -m ml.train`).

## The corpus path (`bin/`)

| script | what |
|---|---|
| `run_corpus_machine.py` | one machine's share of the corpus: 8 months, a shared day queue, N GPUs, one command. Resumable. The early report |
| `run_corpus.sh`, `run_month.sh` | the single-machine drivers: one case per day, resumable, a failed day never stops the month |
| `get_case.sh` | one datetime in, one training record out, scratch deleted |
| `run_corpus_case.sh` | one case end to end, the eight stages. `STUB_LES=1` for a CPU check |
| `pick_hour.py` | draw the day's hour without replacement from the 24, seeded from the date, screened |
| `hrrr_sounding.py` | stage 1: one HRRR pseudo-sounding at the tower for one valid time |
| `sounding_to_forcing.py` | stage 2: the sounding → the FastEddy `.in` parameters (base state, geostrophic wind, flux, `dt`, subsidence) |
| `case_surface.py` | stage 3: the same static geography, this case's per-cell heat-flux map |
| `pick_seed.py` | stage 4: which seed and which 90° rotation, based on what the seeds achieved |
| `prep_restart.py` | stage 5: rotate the seed's flow, inject terrain, `z0` and `htFlux` into the restart |
| `run_window.sh` | stage 6: adjustment plus window as one continuous FastEddy invocation |
| `stage5_footprint.py` | stage 7: the backward LPDM footprint on the static north-up raster, its metrics and health |
| `make_pair.py` | stage 8: the self-contained `.npz` record, with the official Kljun channel on the same cells |
| `check_npz.py` | validate a record against the format |
| `corpus_monitor.py` | the per-case health gate (G1–G3b) |
| `corpus_progress.py` | the live view of a corpus machine |
| `consolidate_corpus.py` | eight machines' records → `corpus_raw.h5`, with the refusals and the train-only normalisation |
| `mask_cone.py` | `corpus_raw.h5` → `corpus_cone.h5`, the wind-aligned cone |
| `corpus_coverage.py`, `enumerate_times.py`, `select_times.py` | how many days the domain accepts. Every candidate hour screened. Deterministic coverage-filling selection (the enumeration study) |
| `stub_case.py`, `stub_footprint.py` | schema-valid records with no LES, LPDM or HRRR. Plumbing only, stamped `stub` |
| `fetch_assets.sh` | the Hugging Face assets, verified against `assets/SHA256SUMS` |

## The seed library (`bin/`)

| script | what |
|---|---|
| `make_seed_jobs.py` | generate the seed jobs and `seeds/index.json` |
| `run_seeds.py` | the whole library on one multi-GPU machine, a work queue, one command |
| `run_seed.sh` | one seed spin-up end to end on whatever machine the checkout is on |
| `seed_watch.sh` | watch a running seed and stop it when it is stationary |
| `seed_stationarity.py` | the seven limits, the single definition |
| `seed_accept.sh` | the full acceptance battery for one seed |
| `c2_restart_check.sh` | Gate C2: does a returned seed restart bit-for-bit |
| `rotation_check.py` | is the 90° re-index of this artifact exact |
| `seed_report.py`, `seed_budget.py` | what a seed cost and achieved. When it would have stopped |
| `direction_drift.py` | Ekman backing and direction drift over the library |
| `threadblock_sweep.py` | the fastest CUDA thread-block shape on this GPU |
| `zero_htflux.py` | zero `htFlux` inside a restart file, and read it back |

## Surface, grid, site (`bin/`)

| script | what |
|---|---|
| `prep_surface.py` | the static surface from 3DEP and WorldCover: terrain, roughness, heat-flux ratios, the array override |
| `prep_stage6.py` | the retired 10 m-era surface builder. The surveyed tower coordinate is defined here |
| `vgrid.py` | solve FastEddy's vertical grid from the receptor height up |
| `g30_bringup.sh` | the 30 m bring-up: cost and the flat `dt` accuracy boundary |
| `k0k1_by_slope.py` | the accuracy check conditioned on terrain slope |
| `phaseA_geometry.py` | footprint geometry on the real map at the real receptor height (Gate A1) |
| `conus404_site.py`, `stable_fraction.py`, `zi_coverage.py` | the CONUS404 site climatology. How much of the stable record is weakly stable. How much of convective midday the domain holds |
| `sigma_w_tower.py` | translate the tower's `σ_w` from 10 m to 30 m |
| `kljun_parity.py` | LES-vs-Kljun asymptote parity with the official FFP on identical cells |
| `seed_leakage.py` | do two cases sharing a seed have more similar footprints |
| `window_independence.py` | two footprints of one case: two samples or one written twice |
| `domain_adequacy.py`, `cbl_check.py`, `smoke_check.py`, `stage4_wellmixed.py`, `ozmidov.py`, `fp16_test.py`, `case_compare.py`, `case_surface.py` | the LES diagnostics: lock-in spectra, convective similarity, a short cold start, the well-mixed gate, the Ozmidov scale, the fp16 check, case comparison |
| `fig_corpus_pairs.py`, `fig_cone_mask.py` | the corpus figures |

## Tests (`bin/test_*`)

Self-contained: `test_bl_depth`, `test_displacement`, `test_floor_health`, `test_kljun_adapter`,
`test_negative_lobes`, `test_sgs_floor`, `test_sounding` (in the image). `test_ml_data`,
`test_ml_model`, `test_cfm` (LESNet). `test_corpus_machine.sh`, `test_work_queue.sh`,
`test_gpu_mutex.sh` (orchestration). Needing LES fields: `test_dumpsrc`, `test_estimator`,
`test_gpu_lpdm`, `test_lpdmonline`, `test_parallel_lpdm`, `test_ringsrc`, `test_streaming`,
`test_toolkit_parity`, `test_unchained`. See [gates and diagnostics](../les/gates-and-diagnostics.md).

## Containers (`docker/`)

| script | what |
|---|---|
| `run.sh`, `pyrun.sh`, `pyrun_gpu.sh` | run a command or a Python script in the analysis image with the repository mounted |
| `run_case.sh` | run one FastEddy case and score it: the concurrency guard, the `.in` check, the log, `k0/k1` |
| `check_run.sh`, `check_output.py` | score a run log and a dump |
| `k0k1_check.py`, `diag_near_surface.py`, `turb_alive.py` | the `dt` accuracy check and the is-there-turbulence check |
| `stage2_gate.py`, `analyze_spinup.py` | the spin-up stationarity gate and profiles |
| `build_fasteddy.sh`, `build_lpdm.sh`, `build_image.sh` | compile FastEddy in the checkout. Build `liblpdm.so`. Build the deployment image |
| `entrypoint.sh`, `verify_image.sh` | the deployment image's commands and its self-check |
| `expected_warnings.txt` | the nine-warning compile baseline |

## FastEddy (`fasteddy/`)

`fetch.sh` fetches NCAR v5.0.1, applies `patches/0001–0006` and verifies `MANIFEST.sha256`.
`UPSTREAM` is the pin. `README.md` explains each patch.

## The LPDM (`lpdm/`)

| module | what |
|---|---|
| `model.py` | the backward Lagrangian model: the Langevin closure, the reverse drift, the forked chunks |
| `driver.py` | release, rotate, accumulate: a `FieldSet` → a footprint |
| `footprint.py` | touchdown weighting, cloud-in-cell deposition, the 2-D metrics |
| `fields.py` | the float16 field cache and the hand-written 4-D interpolation |
| `les_stats.py` | `bl_depth`, `window_stats`, `WindowAccumulator`: the corpus inputs from the LES |
| `sgs_floor.py` | the MOST-anchored `σ_w` floor, in one place |
| `wellmixed.py` | the well-mixed test |
| `kljun_ffp.py`, `kljun.py` | the official FFP evaluated on the raster. The retired reimplementation kept for its gates |
| `dumpsrc.py`, `ringsrc.py`, `hostwatch.py` | where a dump comes from. The ring consumer. Host memory and `/dev/shm` guards |
| `corpus.py`, `partition.py` | the split by month, the hour draw and screen. The 8-machine partition |
| `gpu.py`, `cuda/` | the GPU LPDM front end and its CUDA source |

## The emulators (`ml/`, `ml_cfm/`)

| module | what |
|---|---|
| `ml/data.py`, `features.py` | the loader with its test guard and audit log. The asinh transforms and channels |
| `ml/model.py`, `losses.py`, `metrics.py` | the FNO. The masked losses. The production metrics |
| `ml/train.py`, `phase1.py`, `phase2_optuna.py`, `final.py`, `evaluate.py` | one run. The exploration matrix. The Optuna study. The final seeds. The evaluator with floors |
| `ml_cfm/flow.py`, `model.py`, `train.py`, `infer.py` | the prior-anchored flow, the U-Net, one run, sampling |
| `ml_cfm/campaign.py`, `run_all.sh`, `run_ext.sh`, `run_calib.sh`, `run_calib2.sh` | the unattended study drivers |
| `ml_cfm/evaluate.py`, `calibrate.py`, `crps.py`, `tailthresh.py`, `ccfilter.py`, `posthoc_estimators.py`, `solver_study.py`, `sample_count.py`, `sample_saturation.py`, `cut_sweep.py`, `cut_and_seeds.py` | the CFM evaluation, calibration, CRPS, the tail, the filter, estimators, the solver and sample-count studies, the cut |
| `ml_cfm/final_recipe.py`, `report_metrics.py`, `test_predictions.py` | the frozen recipe. The reporting metrics. The one audited test read |
| `ml_cfm/fig_showcase.py`, `fig_generative.py`, `fig_sectors.py`, `fig_distributions.py`, `fig_domain.py`, `fig_models_val.py`, `fig_uncertainty.py`, `figs_extra.py`, `figstyle.py` | the figures |
