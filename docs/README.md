# Documents

`PROJECT_BRIEF.md` stays at the repository root: it is the working brief and the only document
that is always current on absolute numbers. Everything here is either a live reference or
a record of how a conclusion was reached.

## Live — read these before running anything

| file | what it is |
|---|---|
| `FASTEDDY_TRAPS.md` | every trap that has cost GPU time. **Read before running FastEddy.** |
| `DEPLOY.md` | running the corpus on rented boxes: image, staging, §C2 stubbed-LES smoke test |
| `PLAN.md` | the staged path and the per-pass verdicts |
| `ML_TARGETS.md` | the FNO target design — the raster, the residual on Kljun, the FiLM conditioning |
| `LIBRARY_PLAN.md` | seed-library and corpus design: how a case is chosen, how a seed is spun |

`host_toolchain_install.txt` is the host-side toolchain install log (CUDA, OpenMPI, gcc).
It is provenance for the workstation environment, not a setup script — the reproducible
environments are `Dockerfile` and `Dockerfile.blackwell`.

## `results/` — the record

Twenty per-pass and per-experiment write-ups. **They are superseded on absolute numbers**
by `PROJECT_BRIEF.md`; they are kept for the methodology and for how each conclusion was reached.
When one of them and `PROJECT_BRIEF.md` disagree on a number, `PROJECT_BRIEF.md` is right and the older
document is the record of what was believed at the time.

| group | files | what they cover |
|---|---|---|
| passes | `THIRD` … `NINTH_PASS_RESULTS.md` | the seven development passes, in order |
| stages | `STAGE0A`, `STAGE1`, `STAGE2`, `STAGE2-6`, `STAGE2-6_RESULTS_V2` | the staged LES → LPDM → footprint pipeline |
| seeds | `SEED_LIBRARY_RESULT.md`, `SEED_{CBL_DEEP,CBL_DEEP_24M,NBL_DEEP,NBL_SHALLOW}_RESULT.md` | the 30-seed library and the individual seed rungs |
| regimes | `STABLE_REGIME_RESULT.md` | why there are no stable cases (resolution, `L_O/Delta` = 3.57) |
| box | `CONTAINMENT_RESULT.md` | domain size, water share, tail retention |
| cases | `TARGET_CASE_RESULT.md` | the validation case the estimator was built against |
| corpus | `CONE_MASK_RESULT.md` | the wind-aligned cone: how `target_cone`, the training target, was derived |

The most load-bearing of them, cited from `PROJECT_BRIEF.md`:

* `SIXTH_PASS_RESULTS.md` — the `sigma_w` closure. Two earlier diagnoses were wrong and
  each cost a rebuild; the cause was the MAGNITUDE of the inflation, not its shape.
* `NINTH_PASS_RESULTS.md` — the in-RAM LES→LPDM handoff, and the bit-identity assertion
  between the staged and the netCDF path.
* `SEED_LIBRARY_RESULT.md` — the 30 seeds, and the 0.189 GPU-h/sim-h at 16-way that must
  **not** be carried to a corpus estimate.
* `STABLE_REGIME_RESULT.md` — limitation 5 in `PROJECT_BRIEF.md`, with the measurement behind it.
* `CONE_MASK_RESULT.md` — 2026-09-01, the only write-up here that is CURRENT rather than
  superseded: it documents `corpus_cone.h5`, the shipped training set, and how its one free parameter was measured rather than picked.

## Documents that live elsewhere, on purpose

| path | why it is not here |
|---|---|
| `PROJECT_BRIEF.md` | the working brief; the harness reads it from the root |
| `corpus/README.md` | the dataset documents itself, beside the data (two .h5: raw and cone) |
| `figures/README.md` | the figures document themselves, beside the figures |
| `results/CLEANUP_INVENTORY.txt` | a scored artifact, and `results/` is where those live |
| `jobs/README.md`, `validation_pairs_30m/README.md` | describe the directory they sit in |
