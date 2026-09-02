# Evaluation `final_seed2` on val (235 records, 1 member(s))

## fno

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 20.809 | 80.936 | 0.00 | 65% | 6.2e-22 | [-60.000, -30.000] |
| centroid | m | 235 | 63.386 | 91.855 | 68.516 | 102.180 | 0.69 | 81% | 6.4e-25 | [-36.173, -23.804] |
| overlap80 | Jaccard | 235 | 0.382 | 0.434 | 0.395 | 0.452 | 0.88 | 82% | 1.7e-26 | [-0.063, -0.038] |
| array_share | pp | 235 | 0.251 | 1.460 | 0.991 | 2.197 | 0.17 | 84% | 1.7e-25 | [-1.435, -1.025] |
| integral |  | 235 | 0.102 | 0.140 | 0.135 | 0.182 | 0.73 | 66% | 1.2e-08 | [-0.057, -0.016] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.618 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 9.296 | 24.085 | 0.00 | 41% | 0.00022 | [-30.000, +0.000] |
| centroid | m | 71 | 67.890 | 83.988 | 74.477 | 85.742 | 0.81 | 65% | 0.013 | [-23.767, -5.643] |
| overlap80 | Jaccard | 71 | 0.377 | 0.420 | 0.392 | 0.444 | 0.90 | 79% | 2.4e-06 | [-0.069, -0.022] |
| array_share | pp | 71 | 1.452 | 3.839 | 2.348 | 4.402 | 0.38 | 80% | 5.8e-07 | [-3.167, -1.392] |
| integral |  | 71 | 0.103 | 0.160 | 0.129 | 0.188 | 0.64 | 73% | 5.7e-06 | [-0.077, -0.012] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.623 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 7.143 | 13.571 | nan | 26% | 0.013 | [-30.000, +0.000] |
| centroid | m | 42 | 64.743 | 65.412 | 66.132 | 60.700 | 0.99 | 43% | 0.21 | [-16.402, +22.263] |
| overlap80 | Jaccard | 42 | 0.374 | 0.414 | 0.390 | 0.427 | 0.90 | 74% | 0.008 | [-0.069, -0.013] |
| array_share | pp | 42 | 3.308 | 5.004 | 3.978 | 5.594 | 0.66 | 69% | 0.012 | [-2.580, -0.536] |
| integral |  | 42 | 0.115 | 0.181 | 0.166 | 0.245 | 0.63 | 71% | 9.4e-05 | [-0.139, -0.024] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.626 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.474 | 0.631 | 0.502 | 0.662 | 0.75 | 91% | 6.6e-35 | [-0.176, -0.132] |
| shape_1d |  | 235 | 0.074 | 0.141 | 0.078 | 0.154 | 0.53 | 90% | 3e-35 | [-0.079, -0.057] |
| rel_l2 |  | 235 | 0.343 | 0.541 | 0.364 | 0.580 | 0.63 | 91% | 7.2e-37 | [-0.228, -0.175] |
| rel_l2_T |  | 235 | 0.331 | 0.515 | 0.353 | 0.555 | 0.64 | 91% | 8.6e-37 | [-0.219, -0.163] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.78 | 93% | 2.2e-34 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.63 | 91% | 1.6e-36 | [-0.006, -0.005] |
| pearson_T | r | 235 | 0.046 | 0.123 | 0.059 | 0.142 | 0.37 | 92% | 7.3e-35 | [-0.088, -0.065] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.77 | 92% | 2.5e-33 | [-0.007, -0.004] |
| psnr_T | dB | 235 | -39.759 | -36.139 | -39.834 | -35.752 | nan | 91% | 4.2e-37 | [-4.493, -2.965] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.456 | 0.607 | 0.486 | 0.639 | 0.75 | 93% | 2.4e-11 | [-0.181, -0.103] |
| shape_1d |  | 71 | 0.069 | 0.099 | 0.073 | 0.106 | 0.70 | 79% | 1e-08 | [-0.044, -0.019] |
| rel_l2 |  | 71 | 0.321 | 0.466 | 0.331 | 0.491 | 0.69 | 93% | 3.4e-12 | [-0.205, -0.110] |
| rel_l2_T |  | 71 | 0.298 | 0.441 | 0.318 | 0.461 | 0.68 | 93% | 6.2e-12 | [-0.181, -0.103] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.83 | 92% | 7.3e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.71 | 93% | 7e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.037 | 0.083 | 0.050 | 0.098 | 0.44 | 90% | 1.2e-10 | [-0.056, -0.034] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.018 | 0.022 | 0.83 | 89% | 6.7e-09 | [-0.005, -0.001] |
| psnr_T | dB | 71 | -41.873 | -38.622 | -41.829 | -38.269 | nan | 93% | 3.5e-12 | [-4.356, -2.589] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.526 |
| oct_N | 20 | 0.787 |
| oct_NE | 7 | 0.747 |
| oct_E | 10 | 0.789 |
| oct_SE | 24 | 0.725 |
| oct_S | 42 | 0.629 |
| oct_SW | 48 | 0.418 |
| oct_W | 40 | 0.388 |
| oct_NW | 44 | 0.579 |
| north_N_NE_NW | 71 | 0.647 |
| not_north | 164 | 0.499 |
| array_in_view_gt5pct | 42 | 0.822 |
| array_absent_le5pct | 193 | 0.496 |
| zL_tercile_most_unstable | 79 | 0.702 |
| zL_tercile_middle | 78 | 0.540 |
| zL_tercile_least_unstable | 78 | 0.389 |
| zi_tercile_shallow | 86 | 0.530 |
| zi_tercile_middle | 77 | 0.467 |
| zi_tercile_deep | 72 | 0.581 |
| seed_shared_with_train | 231 | 0.531 |
| seed_not_in_train | 4 | 0.487 |

## fno_cone

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 20.809 | 80.936 | 0.00 | 65% | 6.2e-22 | [-60.000, -30.000] |
| centroid | m | 235 | 63.386 | 91.855 | 68.516 | 102.180 | 0.69 | 81% | 6.4e-25 | [-36.173, -23.804] |
| overlap80 | Jaccard | 235 | 0.382 | 0.434 | 0.395 | 0.452 | 0.88 | 82% | 1.7e-26 | [-0.063, -0.038] |
| array_share | pp | 235 | 0.251 | 1.460 | 0.991 | 2.197 | 0.17 | 84% | 1.7e-25 | [-1.435, -1.025] |
| integral |  | 235 | 0.102 | 0.140 | 0.135 | 0.182 | 0.73 | 66% | 1.2e-08 | [-0.057, -0.016] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.618 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 9.296 | 24.085 | 0.00 | 41% | 0.00022 | [-30.000, +0.000] |
| centroid | m | 71 | 67.890 | 83.988 | 74.477 | 85.742 | 0.81 | 65% | 0.013 | [-23.767, -5.643] |
| overlap80 | Jaccard | 71 | 0.377 | 0.420 | 0.392 | 0.444 | 0.90 | 79% | 2.4e-06 | [-0.069, -0.022] |
| array_share | pp | 71 | 1.452 | 3.839 | 2.348 | 4.402 | 0.38 | 80% | 5.8e-07 | [-3.167, -1.392] |
| integral |  | 71 | 0.103 | 0.160 | 0.129 | 0.188 | 0.64 | 73% | 5.7e-06 | [-0.077, -0.012] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.623 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 7.143 | 13.571 | nan | 26% | 0.013 | [-30.000, +0.000] |
| centroid | m | 42 | 64.743 | 65.412 | 66.132 | 60.700 | 0.99 | 43% | 0.21 | [-16.402, +22.263] |
| overlap80 | Jaccard | 42 | 0.374 | 0.414 | 0.390 | 0.427 | 0.90 | 74% | 0.008 | [-0.069, -0.013] |
| array_share | pp | 42 | 3.308 | 5.004 | 3.978 | 5.594 | 0.66 | 69% | 0.012 | [-2.580, -0.536] |
| integral |  | 42 | 0.115 | 0.181 | 0.166 | 0.245 | 0.63 | 71% | 9.4e-05 | [-0.139, -0.024] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.626 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.474 | 0.631 | 0.502 | 0.662 | 0.75 | 91% | 6.6e-35 | [-0.176, -0.132] |
| shape_1d |  | 235 | 0.074 | 0.141 | 0.078 | 0.154 | 0.53 | 90% | 3e-35 | [-0.079, -0.057] |
| rel_l2 |  | 235 | 0.343 | 0.541 | 0.364 | 0.580 | 0.63 | 91% | 7.2e-37 | [-0.228, -0.175] |
| rel_l2_T |  | 235 | 0.331 | 0.515 | 0.353 | 0.555 | 0.64 | 91% | 8.6e-37 | [-0.219, -0.163] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.78 | 93% | 2.2e-34 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.63 | 91% | 1.6e-36 | [-0.006, -0.005] |
| pearson_T | r | 235 | 0.046 | 0.123 | 0.059 | 0.142 | 0.37 | 92% | 7.3e-35 | [-0.088, -0.065] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.77 | 92% | 2.5e-33 | [-0.007, -0.004] |
| psnr_T | dB | 235 | -39.759 | -36.139 | -39.834 | -35.752 | nan | 91% | 4.2e-37 | [-4.493, -2.965] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.456 | 0.607 | 0.486 | 0.639 | 0.75 | 93% | 2.4e-11 | [-0.181, -0.103] |
| shape_1d |  | 71 | 0.069 | 0.099 | 0.073 | 0.106 | 0.70 | 79% | 1e-08 | [-0.044, -0.019] |
| rel_l2 |  | 71 | 0.321 | 0.466 | 0.331 | 0.491 | 0.69 | 93% | 3.4e-12 | [-0.205, -0.110] |
| rel_l2_T |  | 71 | 0.298 | 0.441 | 0.318 | 0.461 | 0.68 | 93% | 6.2e-12 | [-0.181, -0.103] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.83 | 92% | 7.3e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.71 | 93% | 7e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.037 | 0.083 | 0.050 | 0.098 | 0.44 | 90% | 1.2e-10 | [-0.056, -0.034] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.018 | 0.022 | 0.83 | 89% | 6.7e-09 | [-0.005, -0.001] |
| psnr_T | dB | 71 | -41.873 | -38.622 | -41.829 | -38.269 | nan | 93% | 3.5e-12 | [-4.356, -2.589] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.526 |
| oct_N | 20 | 0.787 |
| oct_NE | 7 | 0.747 |
| oct_E | 10 | 0.789 |
| oct_SE | 24 | 0.725 |
| oct_S | 42 | 0.629 |
| oct_SW | 48 | 0.418 |
| oct_W | 40 | 0.388 |
| oct_NW | 44 | 0.579 |
| north_N_NE_NW | 71 | 0.647 |
| not_north | 164 | 0.499 |
| array_in_view_gt5pct | 42 | 0.822 |
| array_absent_le5pct | 193 | 0.496 |
| zL_tercile_most_unstable | 79 | 0.702 |
| zL_tercile_middle | 78 | 0.540 |
| zL_tercile_least_unstable | 78 | 0.389 |
| zi_tercile_shallow | 86 | 0.530 |
| zi_tercile_middle | 77 | 0.467 |
| zi_tercile_deep | 72 | 0.581 |
| seed_shared_with_train | 231 | 0.531 |
| seed_not_in_train | 4 | 0.487 |

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
| fno | 0.1235 |
| fno_cone | 0.1235 |
| kljun | 0.0801 |
| les | 0.1527 |
