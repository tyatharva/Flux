# Evaluation `final_seed0` on val (235 records, 1 member(s))

## fno

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 21.702 | 80.936 | 0.00 | 65% | 3.9e-21 | [-60.000, -30.000] |
| centroid | m | 235 | 59.184 | 91.855 | 63.928 | 102.180 | 0.64 | 84% | 7.5e-31 | [-39.336, -28.229] |
| overlap80 | Jaccard | 235 | 0.378 | 0.434 | 0.393 | 0.452 | 0.87 | 84% | 1.6e-28 | [-0.067, -0.047] |
| array_share | pp | 235 | 0.297 | 1.460 | 1.001 | 2.197 | 0.20 | 85% | 2.2e-26 | [-1.390, -0.979] |
| integral |  | 235 | 0.106 | 0.140 | 0.135 | 0.182 | 0.76 | 64% | 6.5e-08 | [-0.054, -0.012] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.622 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 9.718 | 24.085 | 0.00 | 39% | 0.00033 | [-30.000, +0.000] |
| centroid | m | 71 | 61.025 | 83.988 | 65.047 | 85.742 | 0.73 | 75% | 2.3e-06 | [-31.035, -14.234] |
| overlap80 | Jaccard | 71 | 0.367 | 0.420 | 0.387 | 0.444 | 0.87 | 83% | 3e-08 | [-0.078, -0.036] |
| array_share | pp | 71 | 1.276 | 3.839 | 2.339 | 4.402 | 0.33 | 83% | 1.4e-07 | [-3.288, -1.683] |
| integral |  | 71 | 0.105 | 0.160 | 0.127 | 0.188 | 0.66 | 72% | 1.4e-05 | [-0.075, -0.008] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.633 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 7.857 | 13.571 | nan | 24% | 0.021 | [-30.000, +0.000] |
| centroid | m | 42 | 58.209 | 65.412 | 55.763 | 60.700 | 0.89 | 57% | 0.27 | [-22.683, +13.528] |
| overlap80 | Jaccard | 42 | 0.366 | 0.414 | 0.385 | 0.427 | 0.88 | 76% | 0.00056 | [-0.071, -0.025] |
| array_share | pp | 42 | 3.465 | 5.004 | 3.880 | 5.594 | 0.69 | 74% | 0.0015 | [-2.454, -0.403] |
| integral |  | 42 | 0.121 | 0.181 | 0.164 | 0.245 | 0.67 | 69% | 0.00013 | [-0.141, -0.024] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.634 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.482 | 0.631 | 0.503 | 0.662 | 0.76 | 90% | 6e-35 | [-0.176, -0.130] |
| shape_1d |  | 235 | 0.072 | 0.141 | 0.076 | 0.154 | 0.51 | 91% | 8.6e-37 | [-0.082, -0.059] |
| rel_l2 |  | 235 | 0.343 | 0.541 | 0.366 | 0.580 | 0.63 | 90% | 9.1e-37 | [-0.228, -0.174] |
| rel_l2_T |  | 235 | 0.328 | 0.515 | 0.355 | 0.555 | 0.64 | 91% | 1e-36 | [-0.221, -0.166] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.78 | 91% | 1.8e-34 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.64 | 91% | 2.2e-36 | [-0.006, -0.004] |
| pearson_T | r | 235 | 0.045 | 0.123 | 0.059 | 0.142 | 0.37 | 91% | 1.2e-34 | [-0.087, -0.065] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.78 | 91% | 1.2e-33 | [-0.006, -0.004] |
| psnr_T | dB | 235 | -39.694 | -36.139 | -39.800 | -35.752 | nan | 91% | 6.3e-37 | [-4.411, -2.871] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.446 | 0.607 | 0.485 | 0.639 | 0.73 | 93% | 1e-11 | [-0.184, -0.108] |
| shape_1d |  | 71 | 0.067 | 0.099 | 0.069 | 0.106 | 0.67 | 85% | 1.7e-10 | [-0.048, -0.024] |
| rel_l2 |  | 71 | 0.324 | 0.466 | 0.336 | 0.491 | 0.70 | 90% | 2.7e-12 | [-0.188, -0.113] |
| rel_l2_T |  | 71 | 0.301 | 0.441 | 0.321 | 0.461 | 0.68 | 93% | 4.6e-12 | [-0.170, -0.110] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.83 | 89% | 2.9e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.71 | 93% | 6.2e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.036 | 0.083 | 0.050 | 0.098 | 0.43 | 90% | 1.2e-10 | [-0.057, -0.034] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.018 | 0.022 | 0.78 | 90% | 1.5e-09 | [-0.005, -0.002] |
| psnr_T | dB | 71 | -41.999 | -38.622 | -41.748 | -38.269 | nan | 93% | 2.9e-12 | [-3.984, -2.632] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.543 |
| oct_N | 20 | 0.772 |
| oct_NE | 7 | 0.631 |
| oct_E | 10 | 0.694 |
| oct_SE | 24 | 0.714 |
| oct_S | 42 | 0.608 |
| oct_SW | 48 | 0.425 |
| oct_W | 40 | 0.407 |
| oct_NW | 44 | 0.541 |
| north_N_NE_NW | 71 | 0.610 |
| not_north | 164 | 0.528 |
| array_in_view_gt5pct | 42 | 0.816 |
| array_absent_le5pct | 193 | 0.527 |
| zL_tercile_most_unstable | 79 | 0.700 |
| zL_tercile_middle | 78 | 0.543 |
| zL_tercile_least_unstable | 78 | 0.402 |
| zi_tercile_shallow | 86 | 0.540 |
| zi_tercile_middle | 77 | 0.484 |
| zi_tercile_deep | 72 | 0.564 |
| seed_shared_with_train | 231 | 0.545 |
| seed_not_in_train | 4 | 0.448 |

## fno_cone

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 21.702 | 80.936 | 0.00 | 65% | 3.9e-21 | [-60.000, -30.000] |
| centroid | m | 235 | 59.184 | 91.855 | 63.928 | 102.180 | 0.64 | 84% | 7.5e-31 | [-39.336, -28.229] |
| overlap80 | Jaccard | 235 | 0.378 | 0.434 | 0.393 | 0.452 | 0.87 | 84% | 1.6e-28 | [-0.067, -0.047] |
| array_share | pp | 235 | 0.297 | 1.460 | 1.001 | 2.197 | 0.20 | 85% | 2.2e-26 | [-1.390, -0.979] |
| integral |  | 235 | 0.106 | 0.140 | 0.135 | 0.182 | 0.76 | 64% | 6.5e-08 | [-0.054, -0.012] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.622 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 9.718 | 24.085 | 0.00 | 39% | 0.00033 | [-30.000, +0.000] |
| centroid | m | 71 | 61.025 | 83.988 | 65.047 | 85.742 | 0.73 | 75% | 2.3e-06 | [-31.035, -14.234] |
| overlap80 | Jaccard | 71 | 0.367 | 0.420 | 0.387 | 0.444 | 0.87 | 83% | 3e-08 | [-0.078, -0.036] |
| array_share | pp | 71 | 1.276 | 3.839 | 2.339 | 4.402 | 0.33 | 83% | 1.4e-07 | [-3.288, -1.683] |
| integral |  | 71 | 0.105 | 0.160 | 0.127 | 0.188 | 0.66 | 72% | 1.4e-05 | [-0.075, -0.008] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.633 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 7.857 | 13.571 | nan | 24% | 0.021 | [-30.000, +0.000] |
| centroid | m | 42 | 58.209 | 65.412 | 55.763 | 60.700 | 0.89 | 57% | 0.27 | [-22.683, +13.528] |
| overlap80 | Jaccard | 42 | 0.366 | 0.414 | 0.385 | 0.427 | 0.88 | 76% | 0.00056 | [-0.071, -0.025] |
| array_share | pp | 42 | 3.465 | 5.004 | 3.880 | 5.594 | 0.69 | 74% | 0.0015 | [-2.454, -0.403] |
| integral |  | 42 | 0.121 | 0.181 | 0.164 | 0.245 | 0.67 | 69% | 0.00013 | [-0.141, -0.024] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.634 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.482 | 0.631 | 0.503 | 0.662 | 0.76 | 90% | 6e-35 | [-0.176, -0.130] |
| shape_1d |  | 235 | 0.072 | 0.141 | 0.076 | 0.154 | 0.51 | 91% | 8.6e-37 | [-0.082, -0.059] |
| rel_l2 |  | 235 | 0.343 | 0.541 | 0.366 | 0.580 | 0.63 | 90% | 9.1e-37 | [-0.228, -0.174] |
| rel_l2_T |  | 235 | 0.328 | 0.515 | 0.355 | 0.555 | 0.64 | 91% | 1e-36 | [-0.221, -0.166] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.78 | 91% | 1.8e-34 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.64 | 91% | 2.2e-36 | [-0.006, -0.004] |
| pearson_T | r | 235 | 0.045 | 0.123 | 0.059 | 0.142 | 0.37 | 91% | 1.2e-34 | [-0.087, -0.065] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.78 | 91% | 1.2e-33 | [-0.006, -0.004] |
| psnr_T | dB | 235 | -39.694 | -36.139 | -39.800 | -35.752 | nan | 91% | 6.3e-37 | [-4.411, -2.871] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.446 | 0.607 | 0.485 | 0.639 | 0.73 | 93% | 1e-11 | [-0.184, -0.108] |
| shape_1d |  | 71 | 0.067 | 0.099 | 0.069 | 0.106 | 0.67 | 85% | 1.7e-10 | [-0.048, -0.024] |
| rel_l2 |  | 71 | 0.324 | 0.466 | 0.336 | 0.491 | 0.70 | 90% | 2.7e-12 | [-0.188, -0.113] |
| rel_l2_T |  | 71 | 0.301 | 0.441 | 0.321 | 0.461 | 0.68 | 93% | 4.6e-12 | [-0.170, -0.110] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.83 | 89% | 2.9e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.71 | 93% | 6.2e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.036 | 0.083 | 0.050 | 0.098 | 0.43 | 90% | 1.2e-10 | [-0.057, -0.034] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.018 | 0.022 | 0.78 | 90% | 1.5e-09 | [-0.005, -0.002] |
| psnr_T | dB | 71 | -41.999 | -38.622 | -41.748 | -38.269 | nan | 93% | 2.9e-12 | [-3.984, -2.632] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.543 |
| oct_N | 20 | 0.772 |
| oct_NE | 7 | 0.631 |
| oct_E | 10 | 0.694 |
| oct_SE | 24 | 0.714 |
| oct_S | 42 | 0.608 |
| oct_SW | 48 | 0.425 |
| oct_W | 40 | 0.407 |
| oct_NW | 44 | 0.541 |
| north_N_NE_NW | 71 | 0.610 |
| not_north | 164 | 0.528 |
| array_in_view_gt5pct | 42 | 0.816 |
| array_absent_le5pct | 193 | 0.527 |
| zL_tercile_most_unstable | 79 | 0.700 |
| zL_tercile_middle | 78 | 0.543 |
| zL_tercile_least_unstable | 78 | 0.402 |
| zi_tercile_shallow | 86 | 0.540 |
| zi_tercile_middle | 77 | 0.484 |
| zi_tercile_deep | 72 | 0.564 |
| seed_shared_with_train | 231 | 0.545 |
| seed_not_in_train | 4 | 0.448 |

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
| fno | 0.1187 |
| fno_cone | 0.1187 |
| kljun | 0.0801 |
| les | 0.1527 |
