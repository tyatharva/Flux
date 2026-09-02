# Evaluation `final_seed4` on val (235 records, 1 member(s))

## fno

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 21.957 | 80.936 | 0.00 | 64% | 1.3e-20 | [-60.000, -30.000] |
| centroid | m | 235 | 54.070 | 91.855 | 60.852 | 102.180 | 0.59 | 84% | 4.7e-30 | [-44.473, -31.022] |
| overlap80 | Jaccard | 235 | 0.385 | 0.434 | 0.401 | 0.452 | 0.89 | 82% | 1.8e-24 | [-0.059, -0.037] |
| array_share | pp | 235 | 0.320 | 1.460 | 1.004 | 2.197 | 0.22 | 84% | 5.4e-26 | [-1.368, -0.961] |
| integral |  | 235 | 0.111 | 0.140 | 0.137 | 0.182 | 0.79 | 64% | 1.2e-07 | [-0.057, -0.010] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.615 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 9.718 | 24.085 | 0.00 | 41% | 0.00042 | [-30.000, +0.000] |
| centroid | m | 71 | 55.688 | 83.988 | 62.823 | 85.742 | 0.66 | 73% | 1.1e-06 | [-37.503, -15.089] |
| overlap80 | Jaccard | 71 | 0.376 | 0.420 | 0.387 | 0.444 | 0.90 | 80% | 1.1e-07 | [-0.070, -0.029] |
| array_share | pp | 71 | 1.038 | 3.839 | 2.286 | 4.402 | 0.27 | 82% | 9.9e-08 | [-3.375, -1.655] |
| integral |  | 71 | 0.115 | 0.160 | 0.130 | 0.188 | 0.72 | 72% | 4.3e-05 | [-0.077, +0.000] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.624 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 9.286 | 13.571 | nan | 19% | 0.058 | [-30.000, +0.000] |
| centroid | m | 42 | 52.296 | 65.412 | 56.572 | 60.700 | 0.80 | 57% | 0.33 | [-21.213, +10.366] |
| overlap80 | Jaccard | 42 | 0.377 | 0.414 | 0.388 | 0.427 | 0.91 | 76% | 0.0028 | [-0.075, -0.014] |
| array_share | pp | 42 | 3.590 | 5.004 | 3.896 | 5.594 | 0.72 | 69% | 0.0023 | [-2.399, -0.535] |
| integral |  | 42 | 0.122 | 0.181 | 0.165 | 0.245 | 0.68 | 74% | 0.00016 | [-0.136, -0.021] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.623 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.476 | 0.631 | 0.506 | 0.662 | 0.75 | 90% | 4.9e-34 | [-0.173, -0.130] |
| shape_1d |  | 235 | 0.072 | 0.141 | 0.074 | 0.154 | 0.51 | 92% | 5.3e-37 | [-0.083, -0.060] |
| rel_l2 |  | 235 | 0.339 | 0.541 | 0.364 | 0.580 | 0.63 | 91% | 2.2e-36 | [-0.232, -0.175] |
| rel_l2_T |  | 235 | 0.327 | 0.515 | 0.353 | 0.555 | 0.63 | 90% | 2.4e-36 | [-0.222, -0.162] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.79 | 91% | 1.4e-33 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.63 | 90% | 4.8e-36 | [-0.006, -0.005] |
| pearson_T | r | 235 | 0.046 | 0.123 | 0.060 | 0.142 | 0.37 | 90% | 2.6e-34 | [-0.087, -0.065] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.79 | 93% | 7.8e-34 | [-0.006, -0.004] |
| psnr_T | dB | 235 | -39.919 | -36.139 | -39.838 | -35.752 | nan | 90% | 1.7e-36 | [-4.461, -2.885] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.449 | 0.607 | 0.484 | 0.639 | 0.74 | 93% | 1.9e-11 | [-0.192, -0.107] |
| shape_1d |  | 71 | 0.061 | 0.099 | 0.067 | 0.106 | 0.61 | 89% | 1e-10 | [-0.050, -0.023] |
| rel_l2 |  | 71 | 0.312 | 0.466 | 0.335 | 0.491 | 0.67 | 92% | 7e-12 | [-0.201, -0.117] |
| rel_l2_T |  | 71 | 0.301 | 0.441 | 0.321 | 0.461 | 0.68 | 92% | 7.8e-12 | [-0.178, -0.107] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.82 | 89% | 7e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.70 | 92% | 1.1e-11 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.037 | 0.083 | 0.051 | 0.098 | 0.45 | 89% | 2.2e-10 | [-0.057, -0.034] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.018 | 0.022 | 0.81 | 92% | 2.5e-09 | [-0.006, -0.002] |
| psnr_T | dB | 71 | -41.947 | -38.622 | -41.771 | -38.269 | nan | 92% | 5.7e-12 | [-4.258, -2.778] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.549 |
| oct_N | 20 | 0.761 |
| oct_NE | 7 | 0.590 |
| oct_E | 10 | 0.976 |
| oct_SE | 24 | 0.626 |
| oct_S | 42 | 0.735 |
| oct_SW | 48 | 0.446 |
| oct_W | 40 | 0.401 |
| oct_NW | 44 | 0.537 |
| north_N_NE_NW | 71 | 0.583 |
| not_north | 164 | 0.456 |
| array_in_view_gt5pct | 42 | 0.812 |
| array_absent_le5pct | 193 | 0.520 |
| zL_tercile_most_unstable | 79 | 0.688 |
| zL_tercile_middle | 78 | 0.554 |
| zL_tercile_least_unstable | 78 | 0.417 |
| zi_tercile_shallow | 86 | 0.556 |
| zi_tercile_middle | 77 | 0.491 |
| zi_tercile_deep | 72 | 0.555 |
| seed_shared_with_train | 231 | 0.549 |
| seed_not_in_train | 4 | 0.563 |

## fno_cone

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 21.957 | 80.936 | 0.00 | 64% | 1.3e-20 | [-60.000, -30.000] |
| centroid | m | 235 | 54.070 | 91.855 | 60.852 | 102.180 | 0.59 | 84% | 4.7e-30 | [-44.473, -31.022] |
| overlap80 | Jaccard | 235 | 0.385 | 0.434 | 0.401 | 0.452 | 0.89 | 82% | 1.8e-24 | [-0.059, -0.037] |
| array_share | pp | 235 | 0.320 | 1.460 | 1.004 | 2.197 | 0.22 | 84% | 5.4e-26 | [-1.368, -0.961] |
| integral |  | 235 | 0.111 | 0.140 | 0.137 | 0.182 | 0.79 | 64% | 1.2e-07 | [-0.057, -0.010] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.615 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 9.718 | 24.085 | 0.00 | 41% | 0.00042 | [-30.000, +0.000] |
| centroid | m | 71 | 55.688 | 83.988 | 62.823 | 85.742 | 0.66 | 73% | 1.1e-06 | [-37.503, -15.089] |
| overlap80 | Jaccard | 71 | 0.376 | 0.420 | 0.387 | 0.444 | 0.90 | 80% | 1.1e-07 | [-0.070, -0.029] |
| array_share | pp | 71 | 1.038 | 3.839 | 2.286 | 4.402 | 0.27 | 82% | 9.9e-08 | [-3.375, -1.655] |
| integral |  | 71 | 0.115 | 0.160 | 0.130 | 0.188 | 0.72 | 72% | 4.3e-05 | [-0.077, +0.000] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.624 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 9.286 | 13.571 | nan | 19% | 0.058 | [-30.000, +0.000] |
| centroid | m | 42 | 52.296 | 65.412 | 56.572 | 60.700 | 0.80 | 57% | 0.33 | [-21.213, +10.366] |
| overlap80 | Jaccard | 42 | 0.377 | 0.414 | 0.388 | 0.427 | 0.91 | 76% | 0.0028 | [-0.075, -0.014] |
| array_share | pp | 42 | 3.590 | 5.004 | 3.896 | 5.594 | 0.72 | 69% | 0.0023 | [-2.399, -0.535] |
| integral |  | 42 | 0.122 | 0.181 | 0.165 | 0.245 | 0.68 | 74% | 0.00016 | [-0.136, -0.021] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.623 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.476 | 0.631 | 0.506 | 0.662 | 0.75 | 90% | 4.9e-34 | [-0.173, -0.130] |
| shape_1d |  | 235 | 0.072 | 0.141 | 0.074 | 0.154 | 0.51 | 92% | 5.3e-37 | [-0.083, -0.060] |
| rel_l2 |  | 235 | 0.339 | 0.541 | 0.364 | 0.580 | 0.63 | 91% | 2.2e-36 | [-0.232, -0.175] |
| rel_l2_T |  | 235 | 0.327 | 0.515 | 0.353 | 0.555 | 0.63 | 90% | 2.4e-36 | [-0.222, -0.162] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.79 | 91% | 1.4e-33 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.63 | 90% | 4.8e-36 | [-0.006, -0.005] |
| pearson_T | r | 235 | 0.046 | 0.123 | 0.060 | 0.142 | 0.37 | 90% | 2.6e-34 | [-0.087, -0.065] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.79 | 93% | 7.8e-34 | [-0.006, -0.004] |
| psnr_T | dB | 235 | -39.919 | -36.139 | -39.838 | -35.752 | nan | 90% | 1.7e-36 | [-4.461, -2.885] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.449 | 0.607 | 0.484 | 0.639 | 0.74 | 93% | 1.9e-11 | [-0.192, -0.107] |
| shape_1d |  | 71 | 0.061 | 0.099 | 0.067 | 0.106 | 0.61 | 89% | 1e-10 | [-0.050, -0.023] |
| rel_l2 |  | 71 | 0.312 | 0.466 | 0.335 | 0.491 | 0.67 | 92% | 7e-12 | [-0.201, -0.117] |
| rel_l2_T |  | 71 | 0.301 | 0.441 | 0.321 | 0.461 | 0.68 | 92% | 7.8e-12 | [-0.178, -0.107] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.82 | 89% | 7e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.70 | 92% | 1.1e-11 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.037 | 0.083 | 0.051 | 0.098 | 0.45 | 89% | 2.2e-10 | [-0.057, -0.034] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.018 | 0.022 | 0.81 | 92% | 2.5e-09 | [-0.006, -0.002] |
| psnr_T | dB | 71 | -41.947 | -38.622 | -41.771 | -38.269 | nan | 92% | 5.7e-12 | [-4.258, -2.778] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.549 |
| oct_N | 20 | 0.761 |
| oct_NE | 7 | 0.590 |
| oct_E | 10 | 0.976 |
| oct_SE | 24 | 0.626 |
| oct_S | 42 | 0.735 |
| oct_SW | 48 | 0.446 |
| oct_W | 40 | 0.401 |
| oct_NW | 44 | 0.537 |
| north_N_NE_NW | 71 | 0.583 |
| not_north | 164 | 0.456 |
| array_in_view_gt5pct | 42 | 0.812 |
| array_absent_le5pct | 193 | 0.520 |
| zL_tercile_most_unstable | 79 | 0.688 |
| zL_tercile_middle | 78 | 0.554 |
| zL_tercile_least_unstable | 78 | 0.417 |
| zi_tercile_shallow | 86 | 0.556 |
| zi_tercile_middle | 77 | 0.491 |
| zi_tercile_deep | 72 | 0.555 |
| seed_shared_with_train | 231 | 0.549 |
| seed_not_in_train | 4 | 0.563 |

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
| fno | 0.1196 |
| fno_cone | 0.1196 |
| kljun | 0.0801 |
| les | 0.1527 |
