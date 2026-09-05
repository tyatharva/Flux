# Repository layout

```
Flux/
├── README.md, LICENSE (Apache-2.0), CITATION.cff
├── mkdocs.yml, .readthedocs.yaml        this documentation
├── Dockerfile                           CUDA 11.8 toolchain image (workstation); every LES result came out of it
├── Dockerfile.blackwell                 CUDA 13.0 deployment image: code + FastEddy + the seed library baked in
├── .dockerignore, .gitignore
├── assets/SHA256SUMS                    checksums and Hugging Face paths of every large asset
├── fasteddy/                            NCAR v5.0.1 pin, the six patches, fetch.sh, MANIFEST.sha256
├── bin/                                 entry points, gates and tests (see Scripts)
├── docker/                              container wrappers, run scoring, image build
├── lpdm/                                the backward LPDM, the footprint estimator, the LES statistics, the hand-off consumer
├── ml/                                  the FNO emulator
├── ml_cfm/                              the CFM emulator, the frozen recipe, the figures
├── third_party/FFP/                     Kljun's official FFP v1.42, unmodified, with its licence and provenance
├── runs/                                the FastEddy .in templates: g30_base (production), g16_base and g24_base
│                                        (the retired grids' templates), and the 30 m CFL ladder (s30_*, g30_*)
├── data/                                README; grid30_raised/ (the production surface, tracked); raw tiles and caches (not in git)
├── seeds/                               the 30-seed library: specs, manifests and verdicts (restarts on Hugging Face)
├── corpus/                              README, INDEX.json, FLAGGED.tsv, provenance/ (the .h5 and .npz on Hugging Face)
├── figures/                             cone/ (the nine pair figures), cone_mask_effect.png, poster/ (the five final figures)
├── results/                             every scored artifact behind a number in these pages (below)
├── validation_pairs_30m/                the two-window pair of case_2023111718: the realisation floor
├── docs/                                these pages
└── FastEddy-model-5.0.1/                not in git: what fasteddy/fetch.sh produces
```

## `results/`

| path | what |
|---|---|
| `g30_bringup.txt`, `g30_flat*.{txt,json,npz}`, `regression_baseline_g30.json` | the 30 m bring-up: the `dt` ladder, the flat control, the regression baseline |
| `sigma_w_curve_30m.json`, `subgrid_fraction_30m.txt`, `subgrid_apriori_30m.txt`, `negative_lobe_30m.txt`, `phaseA_geometry_30m.txt` | the tower `σ_w` curve the stage-7c gate reads; the resolution split; the negative lobe; Gate A1 |
| `pass9/`, `pass10/`, `streaming*.txt`, `lpdmonline_acceptance.txt`, `gpu_lpdm_acceptance.json`, `kljun_adapter.json`, `kljun_parity.json`, `toolkit_parity*.json` | the ninth pass's acceptance evidence and the refused neutral case |
| `containment_gate*.txt`, `integral_decomposition.txt`, `compaction_check.txt`, `ekman_backing.txt`, `les_realisation_spread.txt`, `deciding_test_preregistration.txt`, `window_independence.txt` | the containment, closure and floor measurements |
| `corpus_coverage.txt`, `candidates.tsv`, `selected_times.tsv`, `time_selection.txt`, `zi_coverage.txt`, `conus404_site.{txt,npz}`, `stable_fraction.txt`, `ozmidov_regimes.txt`, `direction_drift.txt`, `hours/` | the corpus design measurements and the site climatology |
| `cone_mask_validation.txt`, `cone_mask_per_record.tsv` | the cone |
| `seed_library/`, `seed_*.{txt,json}`, `threadblock_sweep.json` | the production seed library's machine-level records |
| `cbl_*.npz`, `g16_cbl_*.npz`, `g16r_nbl_wN.json`, `_prof.npz` | inputs of `test_negative_lobes`, `test_floor_health` and `test_bl_depth` |
| `ml/` | `README.md`; `loader_audit.jsonl` (every corpus read); `phase1/summary.*`, `DECISIONS.md`; `phase2/trials.tsv`, `study_summary.*`; `haze/summary.*`; `final/` (the five seeds' `run.json`, `final.json`); `eval/final_ensemble`, `eval/final_seed*`, `eval/floor` |
| `ml_cfm/` | `phase1/`, `final/`, `eval/final/`, `calib/`, `tail/` summaries; `final_recipe/` (the recipe and the val and test metrics); `test/SHA256SUMS.txt`; `ml_final_sha256_before.txt` |

## What was removed on 2026-09-04, and where its information went

The repository was cut from 2059 tracked files to about 830 for public release. Everything
removed is one checkout away at the tag `pre-cleanup-2026-09-04`, and the information each item
carried is in these pages.

| removed | count | information now in |
|---|---|---|
| the project brief that served as the working notes (the previous README-equivalent) | 1 | every page of this site; the [standing rules](standing-rules.md), [ruled out](ruled-out.md), [limitations](../limitations-and-future-work.md) verbatim in substance |
| `FASTEDDY_VERSION.txt`, the fork checkout | 1 | `fasteddy/`, [FastEddy and the patches](../les/fasteddy-and-patches.md) |
| `lib/liblpdm.so` (a compiled sm_89 binary) | 1 | rebuilt by `docker/build_lpdm.sh` |
| `vast-seeds/`: 342 files byte-identical to `seeds/*/return/` | 342 | the 4 unique files moved to `results/seed_library/` |
| `jobs/` (the 16 m seed library), `jobs24/` (the 24 m rung), `jobs30/` renamed `seeds/` | 147 | [seed rungs](../history/seed-rungs.md), [seed library](../les/seed-library.md) |
| `seeds/*/return/{run,seed}.log` (raw FastEddy stdout) | 60 | the verdicts beside them |
| `runs/`: 58 retired-grid directories, 133 `.in` files | 133 | [stages 0–2](../history/stages-0-2.md), [passes 3–7](../history/pass-3.md) |
| `results/`: the 16 m and 24 m record (`g16*`, `g16p6*`, `g16r*`, `g24*`, `fv_*`, `frame_sweep_*`, `cbl_*.txt`, `stage*`, `pass6*`, `retired_sbl_*`, `.done*`, `keep/`, `CLEANUP_INVENTORY.txt`, …) | 345 | the history pages, each of which names what it condensed |
| `results/ml/phase1/*/`, `phase2/trials/`, `haze/*/` per-run files, `eval/early_b0x` | 168 | `summary.*` and `trials.tsv` beside them; [training](../emulator/training.md); [emulator timeline](../history/emulator-timeline.md) |
| 51 retired campaign drivers and one-off analyses in `bin/` | 51 | named at the foot of each history page; the dependency closure of the production path, the tests and the ML imports kept 80 scripts |
| `figures/raw/` (the nine pair figures on `corpus_raw.h5`) | 9 | one documented command regenerates them ([figures](../corpus/figures.md)) |
| `validation_pairs_retired/` (16 m records) | 7 | [target case](../history/target-case.md) |
| `docs/` (31 files: five design documents and 25 pass write-ups) | 31 | this site |
| `docs/host_toolchain_install.txt` | 1 | [environment](../getting-started/environment.md) |
| the local branch `wip/producer-consumer-zarr` (324 commits, disjoint history) | | [unmerged work](../history/unmerged-producer-consumer.md) |

What moved: `jobs30/` → `seeds/`; `jobs/run_seed.sh`, `jobs/seed_watch.sh` → `bin/`; the five
poster PNGs → `figures/poster/`; `SRC/LPDM/CUDA/` from the fork → `lpdm/cuda/`. What became
tracked: `data/grid30_raised/`, `corpus/{README.md,INDEX.json,FLAGGED.tsv,provenance/}`,
`LICENSE`, `CITATION.cff`, `assets/SHA256SUMS`, `bin/fetch_assets.sh`, `fasteddy/`.

## Conventions

- Every FastEddy invocation goes through `docker/run_case.sh`. Every analysis script runs in the
  image through `docker/pyrun.sh`; the host Python has no scipy or h5py on purpose.
- One run per directory. A dump is `<outFileBase>.<step>`; every glob filters on one base name.
- Paths in scripts are repository-relative and the root is discovered from the script's own
  location (`FLUX_ROOT`), so a checkout works anywhere.
- `results/<thing>.txt` is the human-readable record and `<thing>.json` the machine-readable one
  of the same measurement.
- Shell drivers are thin sequencing; logic lives in `bin/*.py`, because bash reads a running
  script by byte offset and Python does not (traps §19d).
