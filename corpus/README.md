# Flux footprint corpus: 122² @ 30 m, 28.5 m receptor

1366 (input, target) pairs for the Kegonsa Solar Array footprint emulator, generated 2026-09-01
on 8 machines × 8 RTX 5090. Documentation: `docs/corpus/dataset.md`.

| in git | |
|---|---|
| `INDEX.json` | every record: tag, datetime, split, machine, day, path |
| `FLAGGED.tsv` | the 231 records failing G2b or G3b (flags, not exclusions) |
| `provenance/` | the 8 machine manifests (every one of 1945 days accounted for), the two rejected records |

| on Hugging Face (`bin/fetch_assets.sh corpus`, `... pairs`) | |
|---|---|
| `corpus_cone.h5` | **train on this**: `target` cropped to the wind-aligned cone, 32 MB |
| `corpus_raw.h5` | the same records with the wraparound still in `target`, 44 MB |
| `pairs_npz/` | the 1366 source `.npz`, one per case |
| `logs/` | the 8 orchestrator run logs |

Both files carry `scalars` (N,6: `h, ustar, sigma_v, L, sin_wdir, cos_wdir`), `kljun` (N,128,128),
`target` (N,128,128), `meta` and `norm` (train split only); they differ only in `target`, and the
root attribute `variant` says which file is open. Splits: train 837 (2021–2023, 2026), val 235
(2024), test 294 (2025). `ml/data.py` refuses the test split unless told otherwise and logs every
read to `results/ml/loader_audit.jsonl`.
