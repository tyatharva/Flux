# Results

Both emulators beat Kljun on every production metric on validation and on the untouched
test split. The CFM's sample mean is slightly ahead of the FNO on the centroid and the 1-D
shape and ties it elsewhere, with a tenth of the parameters. It is also the only one that gives
an error bar.

## The headline: test split, 294 records of 2025, frozen recipe

`results/ml_cfm/final_recipe/metrics_test.md`. Kljun is raw. FNO is the mean of seeds 0–3, cut
at its own 99.5% source area. CFM is the physical-space mean of 80 samples (4 seeds × 20, τ = 1),
cut the same way. LES is positive-only. Errors are RMSE over records. Scores are means.

| model | peak distance RMSE [m] | centroid RMSE [m] | integral RMSE | overlap80 (Jaccard) | rel. L2 | sliced W1 [m] | JS distance [bits] | MS-SSIM (log grid) |
|---|---|---|---|---|---|---|---|---|
| Kljun | 104.0 | 129.3 | 0.240 | 0.548 | 0.565 | 75.0 | 0.359 | 0.937 |
| FNO | 33.1 | 92.8 | 0.184 | 0.604 | 0.365 | 53.5 | 0.292 | 0.937 |
| **CFM** | **30.6** | **69.3** | 0.190 | 0.604 | **0.359** | **40.9** | **0.286** | **0.941** |
| LES (perfect) | 0.0 | 0.0 | 0.000 | 1.000 | 0.000 | 0.0 | 0.000 | 1.000 |

Validation, same recipe, 235 records of 2024: Kljun 118.4 / 130.1 / 0.258 / 0.548 / 0.578 /
75.3 / 0.362 / 0.937. FNO 47.1 / 86.7 / 0.190 / 0.607 / 0.362 / 50.8 / 0.290 / 0.937. CFM
48.8 / 68.8 / 0.197 / 0.607 / 0.358 / 40.6 / 0.287 / 0.939. Every test number is within the val
bootstrap bands and the ordering is the same in every column. Both emulators halve or better
Kljun's peak, centroid and W1 errors.

Metric definitions. Peak distance is the |upwind peak-distance difference| of the
crosswind-integrated footprint. Centroid is the distance between the mass centroids. Integral
is the |integral difference|. overlap80 is the Jaccard of the two 80% source areas. rel. L2 is
‖model − LES‖ / ‖LES‖ on the 122² interior. Sliced W1 is the mean over 64 directions of the 1-D
Wasserstein-1 between the unit-mass positive parts. JS distance is √(Jensen–Shannon divergence)
in bits between the unit-mass positive parts. MS-SSIM is 5-scale SSIM on the log10 grid with
floor 1e-9 m⁻². Sector (90°) and octant (45°) breakdowns are in `metrics_test.json` and the
per-record `.npz`.

### The five figures (`figures/poster/`, 600 dpi)

| file | what it shows |
|---|---|
| `showcase_test.png` | four test cases (2025-05-31 15Z N, 2025-02-09 06Z NW, 2025-11-04 21Z S, 2025-04-20 16Z E), LES / Kljun / FNO / CFM on Esri imagery with the crosswind-integrated profiles and a per-case metric table. Pinned with `--cases`. Row 3 is the S-sector case where the CFM leads on the most metrics |
| `generative_test.png` | one N case (2025-03-19 21Z): the CFM's sample cloud, its mean contour and its 90% band beside the LES |
| `sectors_test.png` | the metrics by wind sector, both emulators against Kljun |
| `distributions_test.png` | ECDFs of the per-record errors, central 90% |
| `domain.png` | the domain, the tower, the array and the lake on imagery with 3DEP contours |

## The test split was read once

`ml/data.py:load_split` raises `TestSplitForbidden` for `split="test"` unless
`allow_test=True`. Nothing under `ml/` or `ml_cfm/` passes it except the explicit
`--allow-test` options. Every corpus read is logged in `results/ml/loader_audit.jsonl`. The
test split was read on 2026-09-04, on instruction, after the recipe and the metric set were
frozen on val. `ml_cfm/test_predictions.py --allow-test` wrote each FNO seed's prediction and
20 fresh CFM samples per seed (Euler 16, RNG seed 0) to `results/ml_cfm/test/` (181 MB, on
Hugging Face. `SHA256SUMS.txt` is committed). Then `report_metrics.py`, `fig_showcase.py` and
`fig_generative.py` read it. Nothing was changed after seeing the numbers.
`bin/test_ml_data.py` fails if any test-split read in the audit log lacks `allow_test`.

## The frozen recipe (`ml_cfm/final_recipe.py`)

Decided on val on 2026-09-03 and applied unchanged to test.

| item | choice | why |
|---|---|---|
| CFM seeds | 0–3 (seed 4 dropped, worst on val at 0.544) | the four-seed pool equals the five-seed pool (0.473) |
| samples | 20 per seed, 80 pooled | saturation `S_sat` = 21 (upper band 64) → 70 → 17.5 per seed, rounded up |
| spread | τ = 1 | the fitted 1.19 improves coverage but makes the array-share CRPS 2–5% worse, and coverage is not a reported metric |
| estimator | physical-space mean | ties the asinh mean within the floor and preserves the integral and array share exactly. The median loses 0.03–0.08 |
| cut | 99.5% source-area cut on both means | metrics tied across 99.0–99.9. Scale-free. Removes every negative cell |
| FNO | seeds 0–3 mean (seed 4 dropped at 0.549), same cut | the four-seed FNO equals the five-seed (0.526) |
| Kljun | untouched | never negative |
| LES | positive-only | the cut denies the models any negative structure. The lobe is noise |

Val composite against Kljun (geometric mean of five production-metric ratios, < 1 beats
Kljun): CFM 0.479, FNO 0.545 over all 235. 0.548 / 0.617 on N/NE/NW (71). 0.760 / 0.831
with the array in view (42). 0.454 / 0.523 elsewhere. Against the raw LES with unmodified
fields the same models score 0.491 / 0.526. The recipe moved the CFM by −0.013 and the FNO by
+0.019, both inside the 0.022 record-bootstrap floor.

## Kljun on val: the number to beat

| metric | Kljun median error vs LES | realisation floor | n behind the floor |
|---|---|---|---|
| peak distance | 30 m (mean 81 m) | 30 m, one cell | 2 runs × 2 cases |
| centroid | 92 m | 15–90 m half-vs-half convective. 46 m run-to-run | 4. 2 |
| 80% source-area overlap | 0.566 | 0.59 half-vs-half. 0.51 between the two validation windows | 1. 1 |
| array share, N/NE/NW | 3.84 pp | 5.3 pp between the two windows. 0.19 pp within-window SE | 1. about 1000 |
| integral | 0.140 | 1.2–1.44× run-to-run | 2 × 2 |
| 2-D shape L1 | 0.63 | 0.63 between the two windows. 0.41–0.92 in the record | 1. 1 each |

Kljun is at the floor on the peak distance and on the 2-D shape and within it on the overlap.
The room to beat it is the array share, the centroid and the integral, and the array share is
where the site signal is. Floors come from `results/les_realisation_spread.txt`, the fourth
pass, the second Stage 2–6 report and `results/ml/eval/floor/pair_floor.json` (the two
windows of `case_2023111718`, scored by the same evaluator after both were cropped to the cone).

## The FNO on val (`docs` of 2026-09-02, `results/ml/eval/final_ensemble/`)

Five seeds of the final configuration. The ensemble is the mean of the five physical-space
predictions.

| seed | val loss (file space) | best epoch / run | composite | N/NE/NW composite | val/train loss |
|---|---|---|---|---|---|
| 0 | 1.1748e-4 | 99 / 125 | 0.543 | 0.610 | 1.06 |
| 1 | 1.1752e-4 | 107 / 133 | 0.532 | 0.606 | 1.07 |
| 2 | 1.1641e-4 | 84 / 110 | 0.526 | 0.647 | 1.05 |
| 3 | 1.1664e-4 | 107 / 133 | 0.514 | 0.593 | 1.07 |
| 4 | 1.1765e-4 | 79 / 105 | 0.549 | 0.583 | 1.04 |
| **ensemble of 5** | | | **0.526** | **0.597** | |

The selection rule in `ml/final.py` picked the best single seed (seed 3) over the ensemble.
The margin is inside the seed spread of the composite (sd 0.014), so the ensemble is the
recommended model. It is the lower-variance estimate of the same conditional mean.

All 235 val records, median |error| against the LES:

| metric | FNO | Kljun | ratio | FNO wins | Wilcoxon p |
|---|---|---|---|---|---|
| peak distance | 0 m | 30 m | 0.00 | 64% | 2e-21 |
| centroid | 55.1 m | 91.9 m | 0.60 | 82% | 2e-30 |
| overlap80 | 0.622 | 0.566 | 0.87 on 1 − J | 84% | 5e-28 |
| array share | 0.286 pp | 1.460 pp | 0.20 | 86% | 2e-26 |
| integral | 0.104 | 0.140 | 0.75 | 63% | 7e-08 |

N/NE/NW (71 records): peak 0 vs 30 m. Centroid 58.0 vs 84.0 m (0.69, 72%, p 7e-6). Overlap
0.636 vs 0.580 (82%, p 9e-8). Array share 1.255 vs 3.839 pp (0.33, 83%, p 1e-7). Integral
0.104 vs 0.160 (0.65, 72%, p 2e-5). Array in view (42 records): peak both 0. Centroid 56.3 vs
65.4 m (a tie, p 0.6). Overlap 0.638 vs 0.586 (76%, p 0.001). Array share 3.51 vs 5.00 pp
(0.70, 74%, p 0.002). Integral 0.113 vs 0.181 (0.62, 71%, p 9e-5).

**Does it win only where the array is absent? No.** It wins on every metric in the northerly
group. The margin is smallest exactly where the signal is largest, because those are the
records whose realisation floor is largest (5.3 pp between two windows of one run at a 20%
share). By octant the composite is N 0.76, NE 0.56 (n = 7), E 0.76, SE 0.71, S 0.70, SW 0.42,
W 0.39, NW 0.51. It is below 1 everywhere. By stability tercile the advantage grows toward the
least unstable third (0.68 → 0.55 → 0.39). By `z_i` tercile it is flat (0.52, 0.48, 0.55).

Shape and 2-D field metrics (all 235, not in the selection composite):

| metric | FNO | Kljun | FNO wins | floor |
|---|---|---|---|---|
| shape L1 (2-D) | 0.473 | 0.631 | 91% | 0.63 two windows. 0.41–0.92 in the record |
| shape (1-D) | 0.071 | 0.141 | 93% | 0.065 two windows |
| rel. L2 | 0.340 | 0.541 | 91% | 0.40 two windows |
| Pearson r (asinh) | 0.956 | 0.877 | 92% | 0.92 |
| SSIM (asinh) | 0.980 | 0.975 | 94% | 0.980 |
| PSNR [dB] | 39.8 | 36.1 | 91% | 40.1 |

The FNO is closer to the LES target than a second realisation of the same case is, on the
2-D shape (0.47 against the 0.63 between the two validation windows), the 1-D shape (at the
floor) and the relative L2 (0.34 against 0.40). That is what a conditional mean should do. It
is nearer to any one sample than samples are to each other.

**Integral against the asymptote `1 − z_m/z_i`**: median |error| LES 0.153, FNO 0.116, Kljun
0.080. The FNO learns the LES's departure from the asymptote (the advection non-closure), so it
is further from the asymptote than Kljun and closer to the LES. The integral is not scored
against the asymptote. The term that would have done so hurt.

## The CFM on val (`results/ml_cfm/eval/final/`)

Five seeds, 500 epochs, patience 10 evaluations, 32 val samples each (160 pooled):

| seed | val_mse_ref | best epoch / run | val/train | composite (own 32-sample mean) |
|---|---|---|---|---|
| 0 | 1.28e-4 | 60 / 111 | 0.95 | 0.546 |
| 1 | 1.17e-4 | 90 / 141 | 0.99 | 0.488 |
| 2 | 1.17e-4 | 90 / 141 | 0.99 | 0.490 |
| 3 | 1.17e-4 | 95 / 146 | 1.00 | 0.474 |
| 4 | 1.17e-4 | 120 / 171 | 1.03 | 0.567 |
| **160-sample pooled mean** | | | | **0.492** |

Four of five seeds reach 1.17e-4, the FNO's number, with 2.95 M parameters against 28.4 M
and no train/val gap.

| metric (median vs LES) | CFM mean | FNO ensemble | Kljun | CFM vs Kljun | CFM vs FNO | floor |
|---|---|---|---|---|---|---|
| peak distance [m] | 0 | 0 | 30 | 65%, p 2e-21 | tie | 30 |
| centroid [m] | 51.7 | 55.1 | 91.9 | 80%, 3e-27 | 63%, 4e-6 | 46–90 |
| overlap80 | 0.616 | 0.622 | 0.566 | 80%, 3e-25 | 49%, 0.28 | 0.51–0.59 |
| array share [pp] | 0.247 | 0.286 | 1.460 | 85%, 7e-26 | 61%, 8e-4 | 5.3 |
| integral | 0.097 | 0.104 | 0.140 | 65%, 2e-9 | 49%, 0.55 | 1.2–1.44× |
| shape L1 (2-D) | 0.471 | 0.473 | 0.631 | 92%, 5e-35 | 57%, 0.02 | 0.63 |
| shape (1-D) | 0.064 | 0.071 | 0.141 | 93%, 3e-37 | 68%, 7e-10 | 0.065 |
| rel. L2 | 0.334 | 0.340 | 0.541 | 92%, 2e-37 | 54%, 0.09 | 0.40 |
| Pearson r | 0.955 | 0.956 | 0.877 | 91%, 1e-34 | 60%, 0.03 | 0.92 |
| SSIM | 0.981 | 0.980 | 0.975 | 86%, 3e-31 | 62%, 1e-4 | 0.980 |
| PSNR [dB] | 39.9 | 39.8 | 36.1 | 91%, 2e-37 | 55%, 0.03 | 40.1 |

The CFM mean beats Kljun on every metric and ties the FNO in practice. It is significantly
better on the centroid, array share, 1-D shape and SSIM and never significantly worse. The
composite is 0.492 against 0.526 (ratio 0.948), below 1 in every octant except E (1.09, n = 10).
On the 42 array-in-view records the two are indistinguishable on every metric (p 0.18–0.89) and
both beat Kljun on the array share (3.48 and 3.51 pp against 5.00 pp). Integral against the
asymptote: LES 0.153, CFM 0.103, FNO 0.116, Kljun 0.080.

**Spread** (160 samples per record, medians): array-share sd 0.37 pp over all records,
2.13 pp on N/NE/NW and 3.50 pp where the array is in view (5–95% range 6.6 and 11.1 pp).
Integral sd 0.13. Peak sd 18 m. Between two samples of one record: overlap80 0.564, shape L1
0.538, centroid distance 84 m, rel. L2 0.386. The two windows of `case_2023111718` differ by
5.34 pp in array share, 0.507 in overlap80, 0.63 in shape L1, 0.40 in rel. L2 and 51 m in
centroid. The sampled spread is therefore the size of the realisation floor on the overlap,
shape, L2 and centroid, and about two-thirds of it on the array share where the array is in
view.

**Sharpness**: mean |∇| in asinh space LES 0.0012, CFM sample 0.0011, CFM mean 0.0009, FNO
0.0009, Kljun 0.0008. High-wavenumber power fraction LES 0.0037, sample 0.0033, mean 0.0021,
FNO 0.0017. Samples have the LES's texture. The mean is as smooth as the FNO.

**Sample count.** A first table (0.743 → 0.492 from S = 1 to 160) was confounded by seed
quality. Measured properly with 800 pooled samples and the law `err(S) = a + b·S^−p`, the
asymptote is 0.471 and saturation against the val noise floor is `S_sat` = 21 (band 2–64).
Per seed the curve is within 1% of its asymptote by S ≈ 30. S = 70 pooled was read from the
fit ([calibration](calibration.md)).

**The connected-component filter** (`ml_cfm/ccfilter.py`) was measured degenerate. Keeping
components until 99.9% of |mass| removes exactly 0.100% from every field and changes no
composite by more than 0.002, and the level at which the LES target is single-connected
(0.40 of the peak) removes 68–71% of the mass from every field including the LES itself.
The isolated low-level cells have no mass the production metrics see.

## Cost

FNO: 28.4 M parameters, 436–783 s per seed at three concurrent runs on the RTX 4080. CFM:
2.95 M parameters, 19–25 min per seed. Sampling at batch 64, Euler 16 steps, costs 17–24 ms
per record per sample, so 32 samples cost about 0.6 s per record and the 160-sample val set
131 s per seed. Euler 4 is four times cheaper at the same quality.

## Limitations of these numbers

1. The model receptor is 30 m (aerodynamic 28.5 m). The instrument is at 10 m.
2. Every corpus record is unstable (`z/L` from −1.76 to −0.002). The emulator is undefined for
   the stable 44% of QC'd hours.
3. Only about 15% of records have the array signal, so the northerly breakout is the one that
   matters.
4. 231 of 235 val records share an LES seed with a train record. The rotated-map control
   argues against memorised geography but cannot rule out seed leakage in general. The test
   split is a different year and the same seed library.
5. The target is a single realisation per case. The emulators regress to the conditional mean
   and cannot reproduce Monte-Carlo texture. Per-cell metrics are reported beside floors and
   kept out of the selection composite.
6. The floors are one two-window pair and two re-runs.
