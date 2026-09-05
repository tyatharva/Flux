# Evaluation `final_seed3` on val (235 records, 1 member(s))

## fno

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 22.596 | 80.936 | 0.00 | 65% | 5.5e-21 | [-60.000, -30.000] |
| centroid | m | 235 | 53.260 | 91.855 | 60.759 | 102.180 | 0.58 | 84% | 1.4e-31 | [-44.764, -31.445] |
| overlap80 | Jaccard | 235 | 0.378 | 0.434 | 0.393 | 0.452 | 0.87 | 85% | 6e-28 | [-0.066, -0.042] |
| array_share | pp | 235 | 0.265 | 1.460 | 0.977 | 2.197 | 0.18 | 86% | 1.4e-26 | [-1.430, -1.015] |
| integral |  | 235 | 0.106 | 0.140 | 0.137 | 0.182 | 0.76 | 60% | 2.1e-06 | [-0.052, -0.007] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.622 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 9.718 | 24.085 | 0.00 | 39% | 0.00033 | [-30.000, +0.000] |
| centroid | m | 71 | 56.753 | 83.988 | 61.087 | 85.742 | 0.68 | 75% | 3.1e-07 | [-40.095, -17.280] |
| overlap80 | Jaccard | 71 | 0.372 | 0.420 | 0.387 | 0.444 | 0.89 | 80% | 1.3e-07 | [-0.073, -0.029] |
| array_share | pp | 71 | 1.136 | 3.839 | 2.278 | 4.402 | 0.30 | 83% | 6.8e-08 | [-3.374, -1.719] |
| integral |  | 71 | 0.112 | 0.160 | 0.128 | 0.188 | 0.70 | 68% | 0.00011 | [-0.077, +0.002] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.628 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 7.857 | 13.571 | nan | 24% | 0.021 | [-30.000, +0.000] |
| centroid | m | 42 | 54.133 | 65.412 | 54.915 | 60.700 | 0.83 | 62% | 0.13 | [-23.218, +10.363] |
| overlap80 | Jaccard | 42 | 0.357 | 0.414 | 0.383 | 0.427 | 0.86 | 76% | 0.0017 | [-0.079, -0.022] |
| array_share | pp | 42 | 3.419 | 5.004 | 3.819 | 5.594 | 0.68 | 74% | 0.00089 | [-2.440, -0.619] |
| integral |  | 42 | 0.123 | 0.181 | 0.164 | 0.245 | 0.68 | 69% | 0.00038 | [-0.138, -0.017] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.643 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.476 | 0.631 | 0.498 | 0.662 | 0.75 | 91% | 4e-35 | [-0.177, -0.135] |
| shape_1d |  | 235 | 0.070 | 0.141 | 0.073 | 0.154 | 0.50 | 92% | 4.7e-37 | [-0.085, -0.061] |
| rel_l2 |  | 235 | 0.337 | 0.541 | 0.366 | 0.580 | 0.62 | 92% | 1e-36 | [-0.230, -0.177] |
| rel_l2_T |  | 235 | 0.330 | 0.515 | 0.354 | 0.555 | 0.64 | 91% | 8.8e-37 | [-0.220, -0.165] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.79 | 91% | 3.6e-34 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.64 | 91% | 1.4e-36 | [-0.006, -0.005] |
| pearson_T | r | 235 | 0.045 | 0.123 | 0.059 | 0.142 | 0.37 | 92% | 6.8e-35 | [-0.088, -0.065] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.77 | 92% | 3.8e-34 | [-0.007, -0.005] |
| psnr_T | dB | 235 | -39.748 | -36.139 | -39.810 | -35.752 | nan | 91% | 4.5e-37 | [-4.522, -2.921] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.452 | 0.607 | 0.481 | 0.639 | 0.75 | 93% | 1.5e-11 | [-0.184, -0.111] |
| shape_1d |  | 71 | 0.062 | 0.099 | 0.065 | 0.106 | 0.63 | 89% | 5.8e-11 | [-0.052, -0.026] |
| rel_l2 |  | 71 | 0.328 | 0.466 | 0.335 | 0.491 | 0.71 | 93% | 4.5e-12 | [-0.188, -0.114] |
| rel_l2_T |  | 71 | 0.301 | 0.441 | 0.320 | 0.461 | 0.68 | 93% | 6.4e-12 | [-0.170, -0.109] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.82 | 89% | 5.3e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.69 | 93% | 8.2e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.039 | 0.083 | 0.050 | 0.098 | 0.46 | 89% | 1.2e-10 | [-0.056, -0.034] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.018 | 0.022 | 0.80 | 87% | 2.6e-09 | [-0.006, -0.002] |
| psnr_T | dB | 71 | -41.756 | -38.622 | -41.770 | -38.269 | nan | 93% | 3.4e-12 | [-4.003, -2.569] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.514 |
| oct_N | 20 | 0.727 |
| oct_NE | 7 | 0.527 |
| oct_E | 10 | 0.923 |
| oct_SE | 24 | 0.759 |
| oct_S | 42 | 0.617 |
| oct_SW | 48 | 0.420 |
| oct_W | 40 | 0.409 |
| oct_NW | 44 | 0.511 |
| north_N_NE_NW | 71 | 0.593 |
| not_north | 164 | 0.498 |
| array_in_view_gt5pct | 42 | 0.802 |
| array_absent_le5pct | 193 | 0.491 |
| zL_tercile_most_unstable | 79 | 0.698 |
| zL_tercile_middle | 78 | 0.536 |
| zL_tercile_least_unstable | 78 | 0.404 |
| zi_tercile_shallow | 86 | 0.492 |
| zi_tercile_middle | 77 | 0.468 |
| zi_tercile_deep | 72 | 0.556 |
| seed_shared_with_train | 231 | 0.516 |
| seed_not_in_train | 4 | 0.484 |

## fno_cone

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 22.596 | 80.936 | 0.00 | 65% | 5.5e-21 | [-60.000, -30.000] |
| centroid | m | 235 | 53.260 | 91.855 | 60.759 | 102.180 | 0.58 | 84% | 1.4e-31 | [-44.764, -31.445] |
| overlap80 | Jaccard | 235 | 0.378 | 0.434 | 0.393 | 0.452 | 0.87 | 85% | 6e-28 | [-0.066, -0.042] |
| array_share | pp | 235 | 0.265 | 1.460 | 0.977 | 2.197 | 0.18 | 86% | 1.4e-26 | [-1.430, -1.015] |
| integral |  | 235 | 0.106 | 0.140 | 0.137 | 0.182 | 0.76 | 60% | 2.1e-06 | [-0.052, -0.007] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.622 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 9.718 | 24.085 | 0.00 | 39% | 0.00033 | [-30.000, +0.000] |
| centroid | m | 71 | 56.753 | 83.988 | 61.087 | 85.742 | 0.68 | 75% | 3.1e-07 | [-40.095, -17.280] |
| overlap80 | Jaccard | 71 | 0.372 | 0.420 | 0.387 | 0.444 | 0.89 | 80% | 1.3e-07 | [-0.073, -0.029] |
| array_share | pp | 71 | 1.136 | 3.839 | 2.278 | 4.402 | 0.30 | 83% | 6.8e-08 | [-3.374, -1.719] |
| integral |  | 71 | 0.112 | 0.160 | 0.128 | 0.188 | 0.70 | 68% | 0.00011 | [-0.077, +0.002] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.628 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 7.857 | 13.571 | nan | 24% | 0.021 | [-30.000, +0.000] |
| centroid | m | 42 | 54.133 | 65.412 | 54.915 | 60.700 | 0.83 | 62% | 0.13 | [-23.218, +10.363] |
| overlap80 | Jaccard | 42 | 0.357 | 0.414 | 0.383 | 0.427 | 0.86 | 76% | 0.0017 | [-0.079, -0.022] |
| array_share | pp | 42 | 3.419 | 5.004 | 3.819 | 5.594 | 0.68 | 74% | 0.00089 | [-2.440, -0.619] |
| integral |  | 42 | 0.123 | 0.181 | 0.164 | 0.245 | 0.68 | 69% | 0.00038 | [-0.138, -0.017] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.643 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.476 | 0.631 | 0.498 | 0.662 | 0.75 | 91% | 4e-35 | [-0.177, -0.135] |
| shape_1d |  | 235 | 0.070 | 0.141 | 0.073 | 0.154 | 0.50 | 92% | 4.7e-37 | [-0.085, -0.061] |
| rel_l2 |  | 235 | 0.337 | 0.541 | 0.366 | 0.580 | 0.62 | 92% | 1e-36 | [-0.230, -0.177] |
| rel_l2_T |  | 235 | 0.330 | 0.515 | 0.354 | 0.555 | 0.64 | 91% | 8.8e-37 | [-0.220, -0.165] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.79 | 91% | 3.6e-34 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.64 | 91% | 1.4e-36 | [-0.006, -0.005] |
| pearson_T | r | 235 | 0.045 | 0.123 | 0.059 | 0.142 | 0.37 | 92% | 6.8e-35 | [-0.088, -0.065] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.77 | 92% | 3.8e-34 | [-0.007, -0.005] |
| psnr_T | dB | 235 | -39.748 | -36.139 | -39.810 | -35.752 | nan | 91% | 4.5e-37 | [-4.522, -2.921] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.452 | 0.607 | 0.481 | 0.639 | 0.75 | 93% | 1.5e-11 | [-0.184, -0.111] |
| shape_1d |  | 71 | 0.062 | 0.099 | 0.065 | 0.106 | 0.63 | 89% | 5.8e-11 | [-0.052, -0.026] |
| rel_l2 |  | 71 | 0.328 | 0.466 | 0.335 | 0.491 | 0.71 | 93% | 4.5e-12 | [-0.188, -0.114] |
| rel_l2_T |  | 71 | 0.301 | 0.441 | 0.320 | 0.461 | 0.68 | 93% | 6.4e-12 | [-0.170, -0.109] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.82 | 89% | 5.3e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.69 | 93% | 8.2e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.039 | 0.083 | 0.050 | 0.098 | 0.46 | 89% | 1.2e-10 | [-0.056, -0.034] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.018 | 0.022 | 0.80 | 87% | 2.6e-09 | [-0.006, -0.002] |
| psnr_T | dB | 71 | -41.756 | -38.622 | -41.770 | -38.269 | nan | 93% | 3.4e-12 | [-4.003, -2.569] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.514 |
| oct_N | 20 | 0.727 |
| oct_NE | 7 | 0.527 |
| oct_E | 10 | 0.923 |
| oct_SE | 24 | 0.759 |
| oct_S | 42 | 0.617 |
| oct_SW | 48 | 0.420 |
| oct_W | 40 | 0.409 |
| oct_NW | 44 | 0.511 |
| north_N_NE_NW | 71 | 0.593 |
| not_north | 164 | 0.498 |
| array_in_view_gt5pct | 42 | 0.802 |
| array_absent_le5pct | 193 | 0.491 |
| zL_tercile_most_unstable | 79 | 0.698 |
| zL_tercile_middle | 78 | 0.536 |
| zL_tercile_least_unstable | 78 | 0.404 |
| zi_tercile_shallow | 86 | 0.492 |
| zi_tercile_middle | 77 | 0.468 |
| zi_tercile_deep | 72 | 0.556 |
| seed_shared_with_train | 231 | 0.516 |
| seed_not_in_train | 4 | 0.484 |

## Realisation floors beside each metric

| metric | floor | independent realisations | source |
|---|---|---|---|
| peak_x | 1 cell (30 m) run-to-run, both cases | 2 runs x 2 cases | `results/les_realisation_spread.txt` |
| peak_x | 0-24 m half-vs-half, convective | 4 cases | `docs/history/pass-4.md:547-560` |
| centroid | 46 m run-to-run (334 vs 380 m), convective | 2 runs x 1 case | `results/les_realisation_spread.txt` |
| centroid | 15-90 m half-vs-half, convective | 4 cases | `docs/history/pass-4.md:547-560` |
| centroid | 336 m p90 at 22.5 min of sub-windows | 18 sub-windows x 1 window | `docs/history/stages-0-2.md:310-380` |
| overlap80 | 0.592 half-vs-half at this grid | 1 window | `docs/history/stages-0-2.md:296-305` |
| overlap80 | 0.43-0.51 half-vs-half, convective | 4 cases | `docs/history/pass-4.md:547-560` |
| overlap80 | 0.56 two LPDM seeds on the same fields | 1 case | `results/les_realisation_spread.txt:30` |
| array_share | 5.65 -> 1.07 pp and 1.14 -> 0.47 pp run-to-run | 2 runs x 2 cases | `results/les_realisation_spread.txt` |
| array_share | 0.19 pp median within-window SE (release groups) | ~1000 records | `corpus/pairs_npz meta array_share_se, train+val` |
| integral | 1.44x and 1.20x run-to-run | 2 runs x 2 cases | `results/les_realisation_spread.txt` |
| integral | 5.5% two LPDM seeds on the same fields | 1 case | `results/les_realisation_spread.txt:30` |
| shape_l1_2d | 0.41 two LPDM seeds on the same fields | 1 case | `results/les_realisation_spread.txt:30` |
| shape_l1_2d | 0.92 two release ensembles, retired 60 m grid | 1 case | `results/stage5.txt:36; docs/history/stages-0-2.md:520-545` |
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
| fno | 0.1000 |
| fno_cone | 0.1000 |
| kljun | 0.0801 |
| les | 0.1527 |
