# flux-kegonsa

A site-calibrated flux-footprint emulator for the UW-Madison Kegonsa Solar Array
eddy-covariance tower. It takes the six scalars the Kljun et al. (2015) footprint model takes
and returns, in milliseconds, the two-dimensional footprint that a large-eddy simulation of
this site would produce. The training targets are 1366 FastEddy large-eddy simulations with a
backward Lagrangian particle model, one per day over five years, forced by real HRRR analyses.

**Documentation: https://flux-kegonsa.readthedocs.io** (source under `docs/`, built with MkDocs).

## The result

On the untouched 2025 test split (294 cases), both emulators beat Kljun on every metric:

| model | peak distance RMSE [m] | centroid RMSE [m] | integral RMSE | overlap80 | rel. L2 | sliced W1 [m] |
|---|---|---|---|---|---|---|
| Kljun | 104.0 | 129.3 | 0.240 | 0.548 | 0.565 | 75.0 |
| FNO | 33.1 | 92.8 | 0.184 | 0.604 | 0.365 | 53.5 |
| **CFM** | **30.6** | **69.3** | 0.190 | 0.604 | **0.359** | **40.9** |

![Four test cases](docs/assets/showcase_test.png)

## Quick start

```bash
git clone https://github.com/tyatharva/flux-kegonsa.git && cd flux-kegonsa
bin/fetch_assets.sh corpus weights          # from https://huggingface.co/datasets/tyatharva/flux-kegonsa
conda env create -f ml/environment.yml && conda activate LESNet
python -m ml_cfm.report_metrics --split val  # both emulators under the frozen recipe, on validation
```

To run the LES pipeline itself: `fasteddy/fetch.sh` fetches NCAR FastEddy v5.0.1 and applies the
six patches under `fasteddy/patches/`; `docker build -t flux-fasteddy:cuda118 .` builds the
toolchain image; `bin/run_corpus_case.sh 2023-01-18T18:00` runs one case. The docs' quick start
has the full paths, including deployment on rented GPUs.

## What is here

| | |
|---|---|
| `fasteddy/` | the FastEddy pin and patch series (no fork) |
| `bin/`, `docker/` | the pipeline's entry points, gates and container wrappers |
| `lpdm/` | the backward Lagrangian particle model and the footprint estimator |
| `ml/`, `ml_cfm/` | the FNO and the flow-matching emulators |
| `seeds/`, `corpus/`, `results/`, `figures/` | the seed library's records, the corpus index, every scored artifact, the figures |
| `docs/` | the documentation source |

Large assets (the corpus, the 30 seed restarts, the model weights; about 3 GB) live on Hugging
Face and are fetched and checksummed by `bin/fetch_assets.sh`.

## Scope

One tower, one grid, no transfer. The model receptor is at 30 m while the instrument is at
10 m; there are no stable cases; only about 15% of the corpus has the array in the footprint.
The documentation's limitations page states all twelve.

## License and citation

Apache-2.0 (`LICENSE`). Kljun's FFP is vendored under its own ISC-style licence
(`third_party/FFP/license.txt`). Cite with `CITATION.cff`.
