# Evaluation `early_b0x` on val (235 records, 1 member(s))

## fno

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 20.170 | 80.936 | 0.00 | 65% | 1.4e-22 | [-60.000, -30.000] |
| centroid | m | 235 | 62.357 | 91.855 | 68.678 | 102.180 | 0.68 | 75% | 1.6e-20 | [-38.808, -24.602] |
| overlap80 | Jaccard | 235 | 0.597 | 0.434 | 0.590 | 0.452 | 1.38 | 16% | 2.8e-31 | [+0.136, +0.179] |
| array_share | pp | 235 | 0.256 | 1.460 | 1.115 | 2.197 | 0.18 | 80% | 7.8e-22 | [-1.443, -1.007] |
| integral |  | 235 | 0.153 | 0.140 | 0.165 | 0.182 | 1.10 | 48% | 0.59 | [-0.016, +0.037] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.403 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 8.451 | 24.085 | 0.00 | 39% | 0.00029 | [-30.000, +0.000] |
| centroid | m | 71 | 58.551 | 83.988 | 65.490 | 85.742 | 0.70 | 72% | 0.0001 | [-37.647, -13.283] |
| overlap80 | Jaccard | 71 | 0.602 | 0.420 | 0.599 | 0.444 | 1.43 | 15% | 5.4e-11 | [+0.147, +0.220] |
| array_share | pp | 71 | 1.200 | 3.839 | 2.697 | 4.402 | 0.31 | 76% | 1.8e-05 | [-3.426, -1.356] |
| integral |  | 71 | 0.133 | 0.160 | 0.150 | 0.188 | 0.83 | 62% | 0.08 | [-0.065, +0.035] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.398 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 8.571 | 13.571 | nan | 24% | 0.052 | [-30.000, +0.000] |
| centroid | m | 42 | 59.591 | 65.412 | 61.383 | 60.700 | 0.91 | 50% | 0.79 | [-21.298, +17.747] |
| overlap80 | Jaccard | 42 | 0.602 | 0.414 | 0.579 | 0.427 | 1.45 | 19% | 2.1e-08 | [+0.115, +0.222] |
| array_share | pp | 42 | 4.021 | 5.004 | 4.635 | 5.594 | 0.80 | 60% | 0.082 | [-2.349, +0.373] |
| integral |  | 42 | 0.136 | 0.181 | 0.183 | 0.245 | 0.75 | 67% | 0.042 | [-0.130, +0.014] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.398 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.579 | 0.631 | 0.610 | 0.662 | 0.92 | 65% | 4.4e-08 | [-0.069, -0.026] |
| shape_1d |  | 235 | 0.092 | 0.141 | 0.099 | 0.154 | 0.65 | 72% | 2.9e-19 | [-0.064, -0.039] |
| rel_l2 |  | 235 | 0.355 | 0.541 | 0.374 | 0.580 | 0.65 | 91% | 1.2e-36 | [-0.218, -0.163] |
| rel_l2_T |  | 235 | 0.338 | 0.515 | 0.363 | 0.555 | 0.66 | 91% | 1.2e-36 | [-0.211, -0.155] |
| mae_T | asinh | 235 | 0.002 | 0.002 | 0.002 | 0.002 | 1.07 | 42% | 0.01 | [+0.000, +0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.66 | 91% | 3e-36 | [-0.006, -0.004] |
| pearson_T | r | 235 | 0.047 | 0.123 | 0.062 | 0.142 | 0.38 | 91% | 7e-35 | [-0.086, -0.063] |
| ssim_T |  | 235 | 0.025 | 0.025 | 0.028 | 0.027 | 0.96 | 53% | 0.38 | [-0.002, +0.001] |
| psnr_T | dB | 235 | -39.556 | -36.139 | -39.567 | -35.752 | nan | 91% | 1e-36 | [-4.214, -2.681] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.568 | 0.607 | 0.598 | 0.639 | 0.94 | 68% | 0.0012 | [-0.076, +0.002] |
| shape_1d |  | 71 | 0.090 | 0.099 | 0.092 | 0.106 | 0.90 | 56% | 0.012 | [-0.024, +0.002] |
| rel_l2 |  | 71 | 0.327 | 0.466 | 0.344 | 0.491 | 0.70 | 92% | 3.4e-12 | [-0.196, -0.102] |
| rel_l2_T |  | 71 | 0.306 | 0.441 | 0.329 | 0.461 | 0.69 | 92% | 4.6e-12 | [-0.168, -0.099] |
| mae_T | asinh | 71 | 0.002 | 0.002 | 0.002 | 0.002 | 1.14 | 32% | 0.00045 | [+0.000, +0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.72 | 92% | 5.5e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.040 | 0.083 | 0.053 | 0.098 | 0.48 | 89% | 8.5e-11 | [-0.056, -0.032] |
| ssim_T |  | 71 | 0.019 | 0.020 | 0.021 | 0.022 | 0.98 | 55% | 0.29 | [-0.002, +0.002] |
| psnr_T | dB | 71 | -41.836 | -38.622 | -41.502 | -38.269 | nan | 92% | 3.8e-12 | [-3.935, -2.609] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.651 |
| oct_N | 20 | 0.934 |
| oct_NE | 7 | 0.900 |
| oct_E | 10 | 1.291 |
| oct_SE | 24 | 1.059 |
| oct_S | 42 | 0.797 |
| oct_SW | 48 | 0.458 |
| oct_W | 40 | 0.510 |
| oct_NW | 44 | 0.605 |
| north_N_NE_NW | 71 | 0.714 |
| not_north | 164 | 0.658 |
| array_in_view_gt5pct | 42 | 0.956 |
| array_absent_le5pct | 193 | 0.640 |
| zL_tercile_most_unstable | 79 | 0.792 |
| zL_tercile_middle | 78 | 0.581 |
| zL_tercile_least_unstable | 78 | 0.598 |
| zi_tercile_shallow | 86 | 0.667 |
| zi_tercile_middle | 77 | 0.613 |
| zi_tercile_deep | 72 | 0.630 |
| seed_shared_with_train | 231 | 0.652 |
| seed_not_in_train | 4 | 0.945 |

## fno_cone

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 20.809 | 80.936 | 0.00 | 64% | 1.7e-22 | [-60.000, -30.000] |
| centroid | m | 235 | 49.859 | 91.855 | 55.970 | 102.180 | 0.54 | 78% | 3.8e-21 | [-50.293, -36.457] |
| overlap80 | Jaccard | 235 | 0.436 | 0.434 | 0.448 | 0.452 | 1.00 | 53% | 0.32 | [-0.014, +0.015] |
| array_share | pp | 235 | 0.247 | 1.460 | 1.007 | 2.197 | 0.17 | 82% | 1.8e-25 | [-1.454, -1.024] |
| integral |  | 235 | 0.114 | 0.140 | 0.140 | 0.182 | 0.82 | 57% | 0.00017 | [-0.048, -0.001] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.564 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 8.451 | 24.085 | 0.00 | 39% | 0.00029 | [-30.000, +0.000] |
| centroid | m | 71 | 46.134 | 83.988 | 53.310 | 85.742 | 0.55 | 75% | 6e-06 | [-48.223, -25.159] |
| overlap80 | Jaccard | 71 | 0.425 | 0.420 | 0.438 | 0.444 | 1.01 | 51% | 0.49 | [-0.025, +0.037] |
| array_share | pp | 71 | 1.300 | 3.839 | 2.386 | 4.402 | 0.34 | 82% | 3.5e-07 | [-3.331, -1.563] |
| integral |  | 71 | 0.090 | 0.160 | 0.121 | 0.188 | 0.56 | 68% | 0.0002 | [-0.098, -0.008] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.575 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 8.571 | 13.571 | nan | 24% | 0.052 | [-30.000, +0.000] |
| centroid | m | 42 | 48.623 | 65.412 | 55.430 | 60.700 | 0.74 | 60% | 0.37 | [-32.523, +7.926] |
| overlap80 | Jaccard | 42 | 0.409 | 0.414 | 0.421 | 0.427 | 0.99 | 55% | 0.61 | [-0.031, +0.032] |
| array_share | pp | 42 | 3.469 | 5.004 | 4.107 | 5.594 | 0.69 | 69% | 0.0065 | [-2.877, -0.342] |
| integral |  | 42 | 0.137 | 0.181 | 0.160 | 0.245 | 0.76 | 69% | 0.0008 | [-0.141, -0.001] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.591 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.506 | 0.631 | 0.535 | 0.662 | 0.80 | 88% | 1.6e-31 | [-0.143, -0.100] |
| shape_1d |  | 235 | 0.069 | 0.141 | 0.076 | 0.154 | 0.49 | 88% | 7.9e-34 | [-0.085, -0.062] |
| rel_l2 |  | 235 | 0.354 | 0.541 | 0.374 | 0.580 | 0.65 | 91% | 1.1e-36 | [-0.219, -0.163] |
| rel_l2_T |  | 235 | 0.338 | 0.515 | 0.362 | 0.555 | 0.66 | 91% | 1.1e-36 | [-0.211, -0.156] |
| mae_T | asinh | 235 | 0.002 | 0.002 | 0.002 | 0.002 | 0.88 | 84% | 6e-25 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.65 | 91% | 2.7e-36 | [-0.006, -0.004] |
| pearson_T | r | 235 | 0.047 | 0.123 | 0.062 | 0.142 | 0.38 | 91% | 6.6e-35 | [-0.086, -0.063] |
| ssim_T |  | 235 | 0.022 | 0.025 | 0.025 | 0.027 | 0.87 | 73% | 1.1e-14 | [-0.004, -0.002] |
| psnr_T | dB | 235 | -39.574 | -36.139 | -39.583 | -35.752 | nan | 91% | 9.5e-37 | [-4.226, -2.696] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.485 | 0.607 | 0.518 | 0.639 | 0.80 | 93% | 7.3e-11 | [-0.158, -0.079] |
| shape_1d |  | 71 | 0.065 | 0.099 | 0.070 | 0.106 | 0.65 | 83% | 1.3e-09 | [-0.051, -0.022] |
| rel_l2 |  | 71 | 0.326 | 0.466 | 0.344 | 0.491 | 0.70 | 93% | 3.2e-12 | [-0.197, -0.102] |
| rel_l2_T |  | 71 | 0.306 | 0.441 | 0.328 | 0.461 | 0.69 | 92% | 4.3e-12 | [-0.169, -0.099] |
| mae_T | asinh | 71 | 0.002 | 0.002 | 0.002 | 0.002 | 0.91 | 83% | 2.3e-07 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.72 | 92% | 5.5e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.040 | 0.083 | 0.053 | 0.098 | 0.48 | 89% | 8.2e-11 | [-0.056, -0.032] |
| ssim_T |  | 71 | 0.017 | 0.020 | 0.019 | 0.022 | 0.87 | 66% | 0.00043 | [-0.004, -0.001] |
| psnr_T | dB | 71 | -41.861 | -38.622 | -41.519 | -38.269 | nan | 92% | 3.8e-12 | [-3.949, -2.625] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.524 |
| oct_N | 20 | 0.679 |
| oct_NE | 7 | 0.623 |
| oct_E | 10 | 1.033 |
| oct_SE | 24 | 0.811 |
| oct_S | 42 | 0.640 |
| oct_SW | 48 | 0.399 |
| oct_W | 40 | 0.428 |
| oct_NW | 44 | 0.514 |
| north_N_NE_NW | 71 | 0.570 |
| not_north | 164 | 0.517 |
| array_in_view_gt5pct | 42 | 0.826 |
| array_absent_le5pct | 193 | 0.506 |
| zL_tercile_most_unstable | 79 | 0.698 |
| zL_tercile_middle | 78 | 0.520 |
| zL_tercile_least_unstable | 78 | 0.414 |
| zi_tercile_shallow | 86 | 0.498 |
| zi_tercile_middle | 77 | 0.501 |
| zi_tercile_deep | 72 | 0.533 |
| seed_shared_with_train | 231 | 0.525 |
| seed_not_in_train | 4 | 0.495 |

## Realisation floors beside each metric

| metric | floor | independent realisations | source |
|---|---|---|---|
| peak_x | 1 cell (30 m) run-to-run, both cases | 2 runs x 2 cases | `results/les_realisation_spread.txt` |
| peak_x | 0-24 m half-vs-half, convective | 4 cases | `docs/results/FOURTH_PASS_RESULTS.md:547-560` |
| centroid | 46 m run-to-run (334 vs 380 m), convective | 2 runs x 1 case | `results/les_realisation_spread.txt` |
| centroid | 15-90 m half-vs-half, convective | 4 cases | `docs/results/FOURTH_PASS_RESULTS.md:547-560` |
| centroid | 336 m p90 at 22.5 min of sub-windows | 18 sub-windows x 1 window | `docs/results/STAGE2-6_RESULTS_V2.md:310-380` |
| overlap80 | 0.592 half-vs-half at this grid | 1 window | `docs/results/STAGE2-6_RESULTS_V2.md:296-305` |
| overlap80 | 0.43-0.51 half-vs-half, convective | 4 cases | `docs/results/FOURTH_PASS_RESULTS.md:547-560` |
| overlap80 | 0.56 two LPDM seeds on the same fields | 1 case | `results/les_realisation_spread.txt:30` |
| array_share | 5.65 -> 1.07 pp and 1.14 -> 0.47 pp run-to-run | 2 runs x 2 cases | `results/les_realisation_spread.txt` |
| array_share | 0.19 pp median within-window SE (release groups) | ~1000 records | `corpus/pairs_npz meta array_share_se, train+val` |
| integral | 1.44x and 1.20x run-to-run | 2 runs x 2 cases | `results/les_realisation_spread.txt` |
| integral | 5.5% two LPDM seeds on the same fields | 1 case | `results/les_realisation_spread.txt:30` |
| shape_l1_2d | 0.41 two LPDM seeds on the same fields | 1 case | `results/les_realisation_spread.txt:30` |
| shape_l1_2d | 0.92 two release ensembles, retired 60 m grid | 1 case | `results/stage5.txt:36; docs/results/STAGE2-6_RESULTS.md:520-545` |
| shape_1d | the two-window pair scored by this evaluator | 1 pair | `results/ml/eval/floor/pair_floor.json` |
| rel_l2 | per-cell L1 0.41 (two LPDM seeds) to 0.92 (two release ensembles, retired grid) | 1 case each | `results/les_realisation_spread.txt:30; results/stage5.txt:36` |
| rel_l2 | the two-window pair scored by this evaluator | 1 pair | `results/ml/eval/floor/pair_floor.json` |
| rel_l2_T | the two-window pair scored by this evaluator | 1 pair | `results/ml/eval/floor/pair_floor.json` |
| mae_T | the two-window pair scored by this evaluator | 1 pair | `results/ml/eval/floor/pair_floor.json` |
| rmse_T | the two-window pair scored by this evaluator | 1 pair | `results/ml/eval/floor/pair_floor.json` |
| pearson_T | the two-window pair scored by this evaluator | 1 pair | `results/ml/eval/floor/pair_floor.json` |
| ssim_T | the two-window pair scored by this evaluator | 1 pair | `results/ml/eval/floor/pair_floor.json` |
| psnr_T | the two-window pair scored by this evaluator | 1 pair | `results/ml/eval/floor/pair_floor.json` |

## Integral against the asymptote 1 - z_m/z_i (median |error|)

| field | median abs error |
|---|---|
| fno | 0.0980 |
| fno_cone | 0.1038 |
| kljun | 0.0801 |
| les | 0.1527 |
