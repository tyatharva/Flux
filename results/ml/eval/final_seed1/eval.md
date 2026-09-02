# Evaluation `final_seed1` on val (235 records, 1 member(s))

## fno

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 20.809 | 80.936 | 0.00 | 65% | 7.2e-23 | [-60.000, -30.000] |
| centroid | m | 235 | 56.201 | 91.855 | 62.586 | 102.180 | 0.61 | 83% | 4.4e-30 | [-42.796, -30.444] |
| overlap80 | Jaccard | 235 | 0.383 | 0.434 | 0.396 | 0.452 | 0.88 | 86% | 1.1e-27 | [-0.063, -0.036] |
| array_share | pp | 235 | 0.292 | 1.460 | 1.004 | 2.197 | 0.20 | 86% | 4.2e-26 | [-1.388, -0.993] |
| integral |  | 235 | 0.104 | 0.140 | 0.138 | 0.182 | 0.74 | 63% | 2.1e-07 | [-0.054, -0.013] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.617 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 10.141 | 24.085 | 0.00 | 38% | 0.00049 | [-30.000, +0.000] |
| centroid | m | 71 | 57.856 | 83.988 | 63.699 | 85.742 | 0.69 | 70% | 8.1e-06 | [-37.504, -15.634] |
| overlap80 | Jaccard | 71 | 0.373 | 0.420 | 0.388 | 0.444 | 0.89 | 80% | 9.3e-08 | [-0.075, -0.024] |
| array_share | pp | 71 | 1.312 | 3.839 | 2.328 | 4.402 | 0.34 | 83% | 1.5e-07 | [-3.202, -1.624] |
| integral |  | 71 | 0.103 | 0.160 | 0.130 | 0.188 | 0.64 | 72% | 7.4e-06 | [-0.077, -0.014] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.627 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 8.571 | 13.571 | nan | 21% | 0.035 | [-30.000, +0.000] |
| centroid | m | 42 | 58.006 | 65.412 | 57.011 | 60.700 | 0.89 | 52% | 0.47 | [-20.451, +14.925] |
| overlap80 | Jaccard | 42 | 0.363 | 0.414 | 0.388 | 0.427 | 0.88 | 74% | 0.002 | [-0.072, -0.013] |
| array_share | pp | 42 | 3.523 | 5.004 | 3.871 | 5.594 | 0.70 | 74% | 0.002 | [-2.426, -0.468] |
| integral |  | 42 | 0.123 | 0.181 | 0.167 | 0.245 | 0.68 | 74% | 6.5e-05 | [-0.134, -0.018] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.637 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.480 | 0.631 | 0.503 | 0.662 | 0.76 | 91% | 6.5e-35 | [-0.175, -0.131] |
| shape_1d |  | 235 | 0.071 | 0.141 | 0.074 | 0.154 | 0.50 | 92% | 3.5e-37 | [-0.084, -0.061] |
| rel_l2 |  | 235 | 0.342 | 0.541 | 0.366 | 0.580 | 0.63 | 92% | 8.2e-37 | [-0.229, -0.175] |
| rel_l2_T |  | 235 | 0.329 | 0.515 | 0.354 | 0.555 | 0.64 | 92% | 7.6e-37 | [-0.221, -0.165] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.78 | 92% | 5.8e-35 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.64 | 92% | 1.5e-36 | [-0.006, -0.004] |
| pearson_T | r | 235 | 0.045 | 0.123 | 0.059 | 0.142 | 0.36 | 92% | 5.3e-35 | [-0.088, -0.065] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.77 | 94% | 9.3e-35 | [-0.007, -0.005] |
| psnr_T | dB | 235 | -39.877 | -36.139 | -39.801 | -35.752 | nan | 92% | 4.9e-37 | [-4.443, -2.967] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.452 | 0.607 | 0.486 | 0.639 | 0.74 | 93% | 1.4e-11 | [-0.177, -0.101] |
| shape_1d |  | 71 | 0.063 | 0.099 | 0.067 | 0.106 | 0.64 | 86% | 2.3e-10 | [-0.050, -0.024] |
| rel_l2 |  | 71 | 0.327 | 0.466 | 0.338 | 0.491 | 0.70 | 93% | 4.1e-12 | [-0.186, -0.111] |
| rel_l2_T |  | 71 | 0.306 | 0.441 | 0.322 | 0.461 | 0.69 | 94% | 5.5e-12 | [-0.167, -0.107] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.83 | 90% | 2.4e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.70 | 94% | 6.2e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.038 | 0.083 | 0.050 | 0.098 | 0.45 | 90% | 1.1e-10 | [-0.056, -0.033] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.017 | 0.022 | 0.81 | 90% | 8.4e-10 | [-0.006, -0.002] |
| psnr_T | dB | 71 | -41.990 | -38.622 | -41.704 | -38.269 | nan | 94% | 3.2e-12 | [-4.070, -2.527] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.532 |
| oct_N | 20 | 0.789 |
| oct_NE | 7 | 0.635 |
| oct_E | 10 | 0.934 |
| oct_SE | 24 | 0.731 |
| oct_S | 42 | 0.636 |
| oct_SW | 48 | 0.425 |
| oct_W | 40 | 0.416 |
| oct_NW | 44 | 0.529 |
| north_N_NE_NW | 71 | 0.606 |
| not_north | 164 | 0.526 |
| array_in_view_gt5pct | 42 | 0.821 |
| array_absent_le5pct | 193 | 0.521 |
| zL_tercile_most_unstable | 79 | 0.704 |
| zL_tercile_middle | 78 | 0.564 |
| zL_tercile_least_unstable | 78 | 0.429 |
| zi_tercile_shallow | 86 | 0.549 |
| zi_tercile_middle | 77 | 0.493 |
| zi_tercile_deep | 72 | 0.545 |
| seed_shared_with_train | 231 | 0.540 |
| seed_not_in_train | 4 | 0.452 |

## fno_cone

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 20.809 | 80.936 | 0.00 | 65% | 7.2e-23 | [-60.000, -30.000] |
| centroid | m | 235 | 56.201 | 91.855 | 62.586 | 102.180 | 0.61 | 83% | 4.4e-30 | [-42.796, -30.444] |
| overlap80 | Jaccard | 235 | 0.383 | 0.434 | 0.396 | 0.452 | 0.88 | 86% | 1.1e-27 | [-0.063, -0.036] |
| array_share | pp | 235 | 0.292 | 1.460 | 1.004 | 2.197 | 0.20 | 86% | 4.2e-26 | [-1.388, -0.993] |
| integral |  | 235 | 0.104 | 0.140 | 0.138 | 0.182 | 0.74 | 63% | 2.1e-07 | [-0.054, -0.013] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.617 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 10.141 | 24.085 | 0.00 | 38% | 0.00049 | [-30.000, +0.000] |
| centroid | m | 71 | 57.856 | 83.988 | 63.699 | 85.742 | 0.69 | 70% | 8.1e-06 | [-37.504, -15.634] |
| overlap80 | Jaccard | 71 | 0.373 | 0.420 | 0.388 | 0.444 | 0.89 | 80% | 9.3e-08 | [-0.075, -0.024] |
| array_share | pp | 71 | 1.312 | 3.839 | 2.328 | 4.402 | 0.34 | 83% | 1.5e-07 | [-3.202, -1.624] |
| integral |  | 71 | 0.103 | 0.160 | 0.130 | 0.188 | 0.64 | 72% | 7.4e-06 | [-0.077, -0.014] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.627 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 8.571 | 13.571 | nan | 21% | 0.035 | [-30.000, +0.000] |
| centroid | m | 42 | 58.006 | 65.412 | 57.011 | 60.700 | 0.89 | 52% | 0.47 | [-20.451, +14.925] |
| overlap80 | Jaccard | 42 | 0.363 | 0.414 | 0.388 | 0.427 | 0.88 | 74% | 0.002 | [-0.072, -0.013] |
| array_share | pp | 42 | 3.523 | 5.004 | 3.871 | 5.594 | 0.70 | 74% | 0.002 | [-2.426, -0.468] |
| integral |  | 42 | 0.123 | 0.181 | 0.167 | 0.245 | 0.68 | 74% | 6.5e-05 | [-0.134, -0.018] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.637 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.480 | 0.631 | 0.503 | 0.662 | 0.76 | 91% | 6.5e-35 | [-0.175, -0.131] |
| shape_1d |  | 235 | 0.071 | 0.141 | 0.074 | 0.154 | 0.50 | 92% | 3.5e-37 | [-0.084, -0.061] |
| rel_l2 |  | 235 | 0.342 | 0.541 | 0.366 | 0.580 | 0.63 | 92% | 8.2e-37 | [-0.229, -0.175] |
| rel_l2_T |  | 235 | 0.329 | 0.515 | 0.354 | 0.555 | 0.64 | 92% | 7.6e-37 | [-0.221, -0.165] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.78 | 92% | 5.8e-35 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.64 | 92% | 1.5e-36 | [-0.006, -0.004] |
| pearson_T | r | 235 | 0.045 | 0.123 | 0.059 | 0.142 | 0.36 | 92% | 5.3e-35 | [-0.088, -0.065] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.77 | 94% | 9.3e-35 | [-0.007, -0.005] |
| psnr_T | dB | 235 | -39.877 | -36.139 | -39.801 | -35.752 | nan | 92% | 4.9e-37 | [-4.443, -2.967] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.452 | 0.607 | 0.486 | 0.639 | 0.74 | 93% | 1.4e-11 | [-0.177, -0.101] |
| shape_1d |  | 71 | 0.063 | 0.099 | 0.067 | 0.106 | 0.64 | 86% | 2.3e-10 | [-0.050, -0.024] |
| rel_l2 |  | 71 | 0.327 | 0.466 | 0.338 | 0.491 | 0.70 | 93% | 4.1e-12 | [-0.186, -0.111] |
| rel_l2_T |  | 71 | 0.306 | 0.441 | 0.322 | 0.461 | 0.69 | 94% | 5.5e-12 | [-0.167, -0.107] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.83 | 90% | 2.4e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.70 | 94% | 6.2e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.038 | 0.083 | 0.050 | 0.098 | 0.45 | 90% | 1.1e-10 | [-0.056, -0.033] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.017 | 0.022 | 0.81 | 90% | 8.4e-10 | [-0.006, -0.002] |
| psnr_T | dB | 71 | -41.990 | -38.622 | -41.704 | -38.269 | nan | 94% | 3.2e-12 | [-4.070, -2.527] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.532 |
| oct_N | 20 | 0.789 |
| oct_NE | 7 | 0.635 |
| oct_E | 10 | 0.934 |
| oct_SE | 24 | 0.731 |
| oct_S | 42 | 0.636 |
| oct_SW | 48 | 0.425 |
| oct_W | 40 | 0.416 |
| oct_NW | 44 | 0.529 |
| north_N_NE_NW | 71 | 0.606 |
| not_north | 164 | 0.526 |
| array_in_view_gt5pct | 42 | 0.821 |
| array_absent_le5pct | 193 | 0.521 |
| zL_tercile_most_unstable | 79 | 0.704 |
| zL_tercile_middle | 78 | 0.564 |
| zL_tercile_least_unstable | 78 | 0.429 |
| zi_tercile_shallow | 86 | 0.549 |
| zi_tercile_middle | 77 | 0.493 |
| zi_tercile_deep | 72 | 0.545 |
| seed_shared_with_train | 231 | 0.540 |
| seed_not_in_train | 4 | 0.452 |

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
| fno | 0.1155 |
| fno_cone | 0.1155 |
| kljun | 0.0801 |
| les | 0.1527 |
