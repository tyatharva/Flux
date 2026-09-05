# Gates and diagnostics

Every stage of the pipeline checks the artifact it produced. It checks the quantity, not the
presence of a file, never an exit code and never the configuration passed in. The
[standing rules](../reference/standing-rules.md) are the lessons in general form. This page is
the list of checks, what each asks and what it cannot see.

## On every FastEddy run

| check | file | asks | cannot see |
|---|---|---|---|
| `CORRUPTED` grep, completion banner | `docker/check_run.sh` | did FastEddy detect NaN, did it finish | `inf` (not NaN). A run that finished a different case |
| `np.isfinite(...).all()` | every script | any non-finite cell | a plausible wrong number |
| **`k0/k1`** | `docker/k0k1_check.py`, `docker/diag_near_surface.py` | first-level over second-level `w` variance. Below 1 (about 0.27) means `dt` is inside the accuracy boundary. Near 9 means grid-scale acoustic noise | a dead boundary layer (both levels quiet together, 0.442 on a collapsed stable run). Terrain-local noise (it is a domain mean. `bin/k0k1_by_slope.py` conditions on slope) |
| **`turb_alive`** | `docker/turb_alive.py` | is there any turbulence at all: `max_k e_res/U_ref²` against the geostrophic wind (healthy 4.6–9.1e-3, dead 6–7e-4), `u*` final/peak | the acoustic failure (OK at every CFL rung). A SKIP is not a PASS |
| `.in` line length | `docker/run_case.sh` | any line at or over 255 characters | (it segfaults the parser before it starts) |
| restart file exists | `docker/run_case.sh` | `inPath + inFile` | (a missing file gives a run of NaN at exit 0) |
| one FastEddy per device | `docker/run_case.sh`, `flock(2)` | a second run writing the same `output/` | (interleaved dumps look like a stall) |
| `outside limits` grep | drivers | an out-of-range parameter that FastEddy silently replaced with its default | |
| IO-registered fields read back | `bin/run_seed.sh`, `bin/prep_restart.py` | the `htFlux`, `z0m`, `topoPos` the dump contains, against what was asked | |

## On a seed

`bin/seed_stationarity.py` is the single definition of the seven stationarity limits (`U/u*`
1.0 %/h, `σ_v/u*` 3.0, `σ_w/u*` 2.0, `TKE_BL/u*²` 5.0, `z_i` at a fixed 0.01 m²/s² threshold 3.0,
Kljun `x_peak` 1.0, `x90` 1.0). They are scored on the last 1.5 h with an AR(1)-corrected
standard error and `n_eff` per trend, and the result is INDETERMINATE when the threshold is
within 3 SE. Every limit but `z_i` is a ratio that follows the inertial oscillation in both
terms. The gate prints the distinct-level count and span beside the `z_i` trend because a line
through a staircase reports the staircase. `bin/seed_watch.sh` scores the trailing window every
30 simulated minutes and stops the run when the oscillation-immune limits are in band.

`bin/seed_accept.sh` is the full battery, the same for every seed: the log checks above,
`k0/k1`, `turb_alive`, the Ozmidov scale at the receptor (`bin/ozmidov.py`, `L_O/Δ` against a
requirement of 10. 3.57 on the stable seed that collapsed, 318–1818 neutral), the resolved
fraction of `σ_w²`, **Gate C2** (`bin/c2_restart_check.sh`: restart the returned dump at its own
step and re-dump, bit-for-bit on 23 variables), the **static rotation check**
(`bin/rotation_check.py`: four 90° turns are the identity bit-for-bit and the FROM bearing moves
exactly −90° per turn), `bin/seed_report.py` (cost and achieved state) and `bin/seed_budget.py`
(when the seed would have stopped). Its exit status is 0 when the run produced a seed. The
verdict is in `stationarity.json`.

## On a corpus case

`bin/run_corpus_case.sh` checks the artifact of each of its eight stages. Stage 7c refuses a
case whose window `σ_w` falls outside the tower's IQR for its own heat flux
(`bin/sigma_w_tower.py` translates the 10 m eddy-covariance record to 30 m through `φ_w`).
`bin/corpus_monitor.py` is the per-case health gate for an unattended run:

| gate | asks |
|---|---|
| G1 floor health | where the `σ_w` floor adds its most variance, and how much. A floor working hardest where the LES already resolves 95% of the variance means `h` is wrong (it caught 2372 m) |
| G2a cap binds | `|I(2L)/I(1L) − 1| ≤ 1e-3`. Containment itself needs `--max-disp` raised |
| G2b integral | in [0.6, 1.5], reported beside Kljun on identical cells and the `1 − z_m/z_i` asymptote |
| G3b peak | LES peak distance over Kljun's in [0.4, 2.5] |

G2b and G3b are flags, not exclusions. 231 records have one ([dataset](../corpus/dataset.md)).
`bin/check_npz.py` validates every record against the format at the end of the case (shapes,
dtypes, finite, zero pad, `dx` = 30 m, split agreeing with the record's own datetime, `stub`
refused). `bin/consolidate_corpus.py` refuses a split disagreement, a duplicate `run_id`, a stub,
a count mismatch per machine or a record from a retired grid.

## The LPDM gates

- **D1 well-mixed** (`bin/stage4_wellmixed.py`, `lpdm/wellmixed.py`): particles released
  uniformly through a deep column, integrated backward and forward, with the concentration
  profile scored against uniform below `z_i` (rms against the 5.48% counting-noise floor, and the
  lowest three bins, where a broken closure piles particles up, within tolerance). Both
  directions, both regimes, with the production closure imported and the no-floor control beside
  it. Passes with 0 turnovers ([sixth pass](../history/pass-6.md), [ninth pass](../history/pass-9.md)).
  What it cannot see: whether `σ_w` has the right magnitude.
- **D2 integral** converges from below with the wrap cap and is quoted against Kljun on identical
  cells, never against 1.
- **D3 error floor**: half-vs-half and independent-realisation differences beside every metric.
  `--cover-groups 10` for the array-share standard error (a 2-group floor was wrong by a factor
  of 5).
- **Containment**: the by-displacement ladder with the cap raised to 3 L. The neutral integral
  must saturate by 2.5 L ([containment](../history/containment.md)).
- **Window independence** (`bin/window_independence.py`): two footprints of one case against the
  within-window floor and the release groups' own decorrelation ladder.
- **The estimator constant** (`bin/test_estimator.py`) against a lid-dependent analytic target.
- **Negative lobes** (`bin/test_negative_lobes.py`): nothing clips them.
- **The floor's structure** (`bin/test_sgs_floor.py`): never lowers the variance, never a
  turnover, inert when nothing needs repair, the `eps` clip pinned.
- **Bit-identity** where there is no physics between two paths: `test_parallel_lpdm` (1 vs 12
  workers), `test_dumpsrc` and `test_ringsrc` (reader indirection, the ring), `test_lpdmonline`
  (producer vs consumer on a real LES), `test_streaming` (streamed vs batched), `test_unchained`
  and `test_toolkit_parity` (against the run-to-run floor, not zero), `test_gpu_lpdm` (the GPU
  integrator against the CPU one within the half-vs-half floor), `test_kljun_adapter` (9.4e-16),
  `test_bl_depth` (47 of 47 exact), `test_displacement`, `test_sounding` (stages 1–2 across
  regimes, every `.in` parameter inside FastEddy's own declared range), `test_floor_health` (G1
  fires on the defect and is silent on every production record).

## The orchestration gates

`bin/preflight.sh` parses every Python entry point and shell driver on the host and in the
container before a campaign. `bin/test_corpus_machine.sh` runs a machine's share with the LES,
LPDM and HRRR stubbed and checks the partition, the queue, the resume and the manifest from the
artifacts. `bin/test_work_queue.sh` and `bin/test_gpu_mutex.sh` check the seed scheduler. The
deployment image asserts its own SASS, the warning baseline, the seed library and the case
path's inputs at build time, and `docker/verify_image.sh` re-checks the SASS against the attached
cards before spending GPU hours. One real case with only the LES stubbed (`STUB_LES=1`) closes
the gap the stubbed dry run leaves ([deployment](deployment.md)).

## The emulator gates

`bin/test_ml_data.py` checks that the loader refuses the test split, that every test-split read
in the audit log has `allow_test`, and that the cone masks, the normalisation and the split are
the file's own. `bin/test_ml_model.py` checks that a zero residual reproduces Kljun to 7.5e-8,
that every parameter has a gradient and that the metrics reproduce the production functions.
`bin/test_cfm.py` runs 26 checks: the interpolant, the CRPS estimator, the test guard and that
`ml/`, `results/ml/final/` and the first CFM run are unchanged by hash.

## Diagnostics that are reported and never gated

The sub-grid fraction of `σ_w²` at the receptor (the 40% gate is retired as unreachable), the
compaction ratio with its closure, the anchor-sensitivity band (46–66% shape L1), the wrapped
fraction, `w̄` at the receptor, the Ekman backing, achieved-minus-requested seed gaps and the
integral against its asymptote. A result quoted without its floor and its no-op control says only
that a number came out.
