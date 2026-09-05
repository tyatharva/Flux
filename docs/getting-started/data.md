# Data and assets

Everything larger than a few hundred kilobytes lives outside git, on one Hugging Face dataset
repository: **`tyatharva/flux-kegonsa`** (`https://huggingface.co/datasets/tyatharva/flux-kegonsa`).
`bin/fetch_assets.sh` downloads a group, verifies every file against `assets/SHA256SUMS`, and
puts it at the path the code expects. `assets/SHA256SUMS` lists, for each file, its sha256, its
path on Hugging Face and its path in this repository.

```bash
bin/fetch_assets.sh corpus        # the two HDF5 files -> corpus/
bin/fetch_assets.sh pairs         # the 1366 source records -> corpus/pairs_npz/
bin/fetch_assets.sh seeds         # the 30 restarts -> seeds/*/return/seed_restart.nc
bin/fetch_assets.sh weights       # FNO and CFM checkpoints -> results/ml*/final/*/best.pt
bin/fetch_assets.sh predictions   # the audited test-split outputs and the FNO val predictions
bin/fetch_assets.sh all
```

## What is where

| Hugging Face path | repository path | size | what |
|---|---|---|---|
| `corpus/corpus_cone.h5` | `corpus/corpus_cone.h5` | 32.2 MB | **the training set**: 1366 records, `target` cropped to the wind-aligned cone |
| `corpus/corpus_raw.h5` | `corpus/corpus_raw.h5` | 44.2 MB | the same records with the wraparound still in `target` |
| `corpus/pairs_npz.tar` | `corpus/pairs_npz/` | 54 MB | the 1366 self-contained source `.npz`, one per case, plus the machines' `manifest.*.json` |
| `corpus/logs.tar` | `corpus/logs/` | 0.2 MB | the 8 orchestrator run logs |
| `seeds/<seed>/seed_restart.nc` × 30 | `seeds/<seed>/return/seed_restart.nc` | 30 × 73.3 MB = 2.1 GB | the seed library: the final dump of each spin-up, all 22 variables |
| `weights/fno/seed{0..4}/best.pt` | `results/ml/final/seed*/best.pt` | 5 × 108.3 MB | the FNO's five seeds |
| `weights/cfm/seed{0..4}/best.pt`, `long1000_seed0/best.pt` | `results/ml_cfm/final/*/best.pt` | 6 × 11.3 MB | the CFM's five seeds and the 1000-epoch run |
| `predictions/test/fno_seed{0..3}_pred_test.npz`, `seed{0..3}_samples_test.npz` | `results/ml_cfm/test/` | 181 MB | the one audited read of the test split: FNO predictions and 20 CFM samples per seed (`SHA256SUMS.txt` beside them is in git) |
| `predictions/val/fno_seed{0..4}_pred_val.npz` | `results/ml/final/seed*/pred_val.npz` | 19 MB | the FNO's val predictions |

In git beside them: `corpus/README.md`, `corpus/INDEX.json` (every record's tag, datetime,
split, machine, day and path), `corpus/FLAGGED.tsv`, `corpus/provenance/` (the 8 machine
manifests accounting for every day, the two rejected records, a partial manifest kept for
provenance), `seeds/*/manifest.json`, `seeds/*/seed.in` and each seed's verdicts under
`seeds/*/return/`, `results/seed_library/` (the library run's machine manifest, thread-block
sweep, direction-drift table and log), and every scored summary under `results/`.

## What is regenerable and not hosted

| | size | how to regenerate |
|---|---|---|
| `results/ml_cfm/final/seed*/samples_val*.npz` | about 1.3 GB | `ml_cfm/infer.py` from the checkpoints (Euler 16, 32 samples per seed; the 128 extra samples of the sample-count study likewise) |
| `results/ml/cache/`, `results/ml/*.db` | small | rebuilt by the loader; the Optuna study is not shipped |
| `data/hrrr/` | 2.4 GB | the HRRR cache; every case re-fetches its sounding from the archive |
| `data/raw/output_USGS10m.tif`, `rasters_USGS10m.tar.gz` | 321 MB + 321 MB | USGS 3DEP 1/3 arc-second, the tile covering the tower |
| `data/raw/ESA_WorldCover_10m_2021_v200_N42W090_Map.tif` | 92 MB | ESA WorldCover v200 (2021), tile N42W090 |
| `data/raw/conus404_site.npz`, `H_and_sigma_w.csv` | 1 MB each | `bin/conus404_site.py` (anonymous S3 over HTTPS); the tower's own eddy-covariance record |
| `data/grid16*`, `grid24*`, `grid_cbl`, `grid`, `case_grids/`, `smokelib/` | up to 280 MB | retired grids and the stubbed-LES smoke library; regenerable with `bin/prep_surface.py` |
| LES windows, dumps, `runs/*/output` | tens of GB per case | never kept; a case is one FastEddy run |

## The production surface

`data/grid30_raised/` is tracked (184 kB): `topo.npy`, `z0m.npy`, `htFlux.npy`, `lcclass.npy`,
`array.npy`, `water.npy`, `dmap.npy`, `meta.npy`, `topo.bin`, and the two clipped rasters
`dem24.tif`, `lc24.tif`. It is built by `bin/prep_surface.py` from the two raw tiles above (see
[the site](../problem/site.md)), and every corpus case reads it. The deployment image asserts
the three `.npy` maps are present.

## Sources

| | |
|---|---|
| HRRR | NOAA's 3 km High-Resolution Rapid Refresh analyses, hourly, v4 from 2020-12-02, fetched by Herbie from the public archive with byte-range subsetting; about 170 MB per case, nothing retained |
| CONUS404 | the 45-year 4 km hourly reanalysis, streamed from the USGS Open Storage Network; climatology only, never forcing |
| USGS 3DEP | 1/3 arc-second (about 10 m) bare-earth elevation, EPSG:4269 |
| ESA WorldCover | v200, 2021, 10 m land cover, EPSG:4326; it labels the array as cropland |
| Kljun FFP | v1.42, `footprint.kljun.net/downloads/v1.42/FFP_Python.zip`, retrieved 2026-08-30, vendored in `third_party/FFP/` with its ISC licence; file hashes in `third_party/FFP/PROVENANCE.md` |
| FastEddy | NCAR v5.0.1 plus `fasteddy/patches/`, fetched by `fasteddy/fetch.sh` |
| the tower | UW-Madison Kegonsa Solar Array eddy-covariance tower, `42.957160, −89.292362`, instrument at about 10 m |

## Sizes at a glance

| | |
|---|---|
| this repository | about 830 tracked files, about 60 MB (44 MB of it the five 600 dpi poster figures) |
| Hugging Face | about 3.1 GB |
| a full local working tree with caches and samples | about 11 GB |
