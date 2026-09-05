# The dataset

1366 (input, target) pairs for the Kegonsa Solar Array footprint emulator, generated on
2026-09-01 on 8 machines × 8 RTX 5090 in about 12 hours. The files live on Hugging Face
([data and assets](../getting-started/data.md)); the index, flags and provenance are in this
repository under `corpus/`.

## Two files, one layout

| file | size | `variant` | what `target` is |
|---|---|---|---|
| **`corpus_cone.h5`** | 32 MB | `cone` | **Train on this.** The LES footprint cropped to the wind-aligned cone. |
| `corpus_raw.h5` | 44 MB | `raw` | The LES footprint as the pipeline produced it, periodic wraparound and all. |

Both carry the same `scalars`, `kljun`, `meta` and `norm`. They differ only in `target`, and
it is called `target` in both, so a loader points at one file or the other and nothing else
changes. The root attribute `variant` says which file you opened.

```
scalars   (1366, 6)        float32   h, ustar, sigma_v, L, sin_wdir, cos_wdir
kljun     (1366, 128, 128) float32   official FFP v1.42 on the target's own cell edges
target    (1366, 128, 128) float32   LES + backward LPDM, SIGNED and UNCLIPPED
meta/                                datetime, run_id, split, split_index, gate_state,
                                     integral, peak_x_m, array_share, zi_achieved_m, ...
norm/                                computed from the TRAIN split alone (837 records)
```

122 → 128 is a zero-pad of 3 cells per side, not a resize. The frame is north-up, with the
receptor at the centre cell (64, 64). Negative target values are physical and nothing clips
them. `L` is stored raw and is ±inf at exactly neutral; loaders use `meta/inv_L`, which is
finite everywhere.

Beside the two files: `INDEX.json` (tag, datetime, split, machine, day, path for every
record), `FLAGGED.tsv`, the 1366 source `.npz` (`pairs_npz/`, one per case, self-contained),
the 8 orchestrator logs, and `provenance/` with the 8 per-machine manifests. Every one of the
1945 days in the corpus window is accounted for in those manifests, with a reason where no
case was produced.

## Splits: by calendar month, assigned at generation

Whole years go to validation and test, so a split boundary never falls inside a synoptic
system and seasonal coverage is complete on each side.

| split | months | records |
|---|---|---|
| train | 2021, 2022, 2023, and 2026-02/04/06/08 (40 months) | 837 |
| val | 2024 (12 months) | 235 |
| test | 2025 (12 months) | 294 |

`lpdm/corpus.py:SPLITS` is the definition. A month not named is not in the corpus, and
`split_of` refuses it. Disjointness is asserted at import. The split is checked against the
case's own timestamp, and disagreement is fatal. Normalisation constants come from the
training split alone; computing them over the whole corpus would leak the validation
distribution into the inputs.

The test split was read exactly once, for the frozen evaluation; every read of the corpus is
logged in `results/ml/loader_audit.jsonl`, and `bin/test_ml_data.py` fails if any test-split
read lacks the `allow_test` flag.

## Only about 15% of the corpus carries the site signal

The array is in the footprint essentially only for northerly flow, and convective afternoons
here favour SW and W.

| direction | corpus | site rose | mean array share |
|---|---|---|---|
| **N** | 6.9% | 10.6% | **30.3%** |
| NW | 14.6% | 14.5% | 6.9% |
| NE | 6.4% | 10.2% | 2.7% |
| W | 21.4% | 14.4% | 0.3% |
| SW | 19.4% | 14.3% | 0.9% |
| E | 4.5% | 10.4% | 0.2% |

202 of 1366 records (14.8%) have an array share above 5%; the median is 0.49%. Per split:
train 14.7%, val 17.9%, test 12.6%, so every split sees it. An aggregate metric over all
1366 records is dominated by cases with no array in view, where Kljun and the LES agree by
construction. Weight the loss, or report the northerly subset separately. (The measured
N-wind array share of 30.28% matches the 30.7% Kljun predicts for north at `z_m = 30 m` to
1.4%, independently.)

## The cone

`corpus_cone.h5` zeroes everything outside a wind-aligned cone: `x' ≥ 0` and
`|y'| ≤ max(8·σ_y(x'), 90 m)`, where `σ_y` is Kljun's own, taken from the official FFP.

The raw target carries periodic wraparound. Touchdowns are folded modulo the domain per axis
and independently, so material that runs more than one domain length reappears through a
seam, downwind or upwind-but-far-off-axis. It is a boundary-condition artifact; no tower
measures it, and the emulator must not be asked to predict it.

All three cone parameters were measured, not picked:

- `k = 8`: the LES mass distribution against `q = |y'|/σ_y(x')` is bimodal with an empty
  valley. 0.0110% of |mass| lies in `q ∈ [5, 11)`, rising again past `q ≈ 11`. Any `k` in
  [5, 11] gives the same answer; removed mass moves 0.4 percentage points across a factor of
  four in `k`.
- `x_min = 0`: a genuine footprint puts nothing downwind. Diagonal winds scatter their wrap
  off-axis, and their retained profile is exactly 0.0000% at every bin with `x' < 0`.
- Wrap does not reach positive `x'`: the predicted reach grows from 2 m to 56 m across
  0–10° off-axis while the measured mass just upwind of the receptor is flat.

Removed mass is a median 12.46% of |f|; Kljun loses 0.00000%. Removed mass within 200 m of
the receptor, upwind where the near-field peak is, is a median 0.000% and at most 0.127%.

A downwind half-plane cut is not enough, because the fold is per axis: single-axis wrap lands
back upwind as a thin off-axis streak whenever the displacement exceeds
`3660 · max(|sin|, |cos|)`. The cone removes a median 1.18% of |f| (max 9.39%) that sits
upwind, on 1360 of 1366 records, all of it invisible to a half-plane. No particle can wrap
twice: `lpdm/driver.py` caps displacement at one domain length (`max_disp = fs.Lx` = 3660 m),
so every wrapped particle lands off-axis or downwind and the cone catches all of it.

**The cone is an operational cleanup, not an integral correction.** Median |error| against
the `1 − z_m/z_i` asymptote goes 0.1443 → 0.1467, and `r(|mass| removed, raw error) = −0.490`:
the records losing the most wrap were the ones already below the asymptote. Whatever
inflates the integral is not the wraparound. Full derivation: [cone mask](../history/cone-mask.md).

The negative lobe is a median 4.80% of |f| raw and 1.59% after the cone. It does not carry
the Steinfeld wind-turning signature (negatives to the right of the upstream direction): the
right-hand fraction is a median 0.534 with a right-hand majority in 53.8% of records, no side
preference at all.

## Flags, gaps and rejected records

- **`FLAGGED.tsv` lists 231 records (16.9%)** failing G2b (integral outside [0.6, 1.5]: 65
  records) or G3b (peak distance / Kljun peak distance outside [0.4, 2.5]: 187 records).
  Neither is an exclusion rule. Both are per-case pipeline sanity checks calibrated on a
  handful of validation cases, and an LES peak far from Kljun's may be the signal. Ablate; do
  not filter by default.
- **Two records were rejected** and are in neither file: `case_2022010915` and
  `case_2022122416` had `h = 2371.979 m`, which is `bl_depth`'s `DAMP_FRAC` search ceiling
  and not a measured depth. Their days are `missing` with that reason; the `.npz` are kept
  under `provenance/rejected/`. Dropping them cut the train split's `h` standard deviation by
  5% (237.9 → 226.5 m), so they were skewing the normalisation too.
- **166 days failed; six months are empty**: 2021-12, 2023-07, 2023-10, 2024-01 (val),
  2024-04 (val), 2026-08. 2021-06 and 2022-04 are partial. Machine 3 lost all 8 GPUs to one
  fault 42% into its run. Verified not to be an input-space hole: 84–93% of a missing month's
  cases fall inside the retained months' p5–p95 on every scalar.
- `provenance/PARTIAL_record_manifest_machine7_only.json` was found as
  `pairs_npz/manifest.json`; it indexes 196 of the 1368 records present at the time and is
  kept for provenance only.
- Day yield is meteorological: 78% overall, but 38% in June and 57% in July. A summer
  convective boundary layer is inside the 300–1250 m band only while growing at 17–45 %/h,
  and it is past the band by the time growth falls under the 15 %/h stationarity screen.

## Provenance and reproduction

| | |
|---|---|
| image | `ghcr.io/tyatharva/flux-seeds:7de9dee2a01d-fe0ce48d5dff06` |
| image digest | `sha256:3f58d049d895178e9a9035e9317d6a11582f9002dc801be3e2dd7a20430e8404` |
| code | the tag names a commit of the pre-cleanup history (`7de9dee`); the same tree is commit `7cdd65e` of the pre-rewrite `main`, kept in the author's offline pre-cleanup archive of 2026-09-04 |
| FastEddy | `flux.fasteddy.revision = 0ce48d5dff06`, which is upstream v5.0.1 plus `fasteddy/patches/` |

The 30-seed library is baked into that image. The hour draw is seeded from the date, so
re-running a failed day reproduces exactly the case that would have been there. Procedure:
[deployment](../les/deployment.md). Consolidation from the per-machine `.npz` into the two
HDF5 files: `bin/consolidate_corpus.py`, then `bin/mask_cone.py` for the cone variant.
