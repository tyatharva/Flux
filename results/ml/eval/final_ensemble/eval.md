# Evaluation `final_ensemble` on val (235 records, 5 member(s))

## fno

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 21.574 | 80.936 | 0.00 | 64% | 1.7e-21 | [-60.000, -30.000] |
| centroid | m | 235 | 55.096 | 91.855 | 62.692 | 102.180 | 0.60 | 82% | 2.1e-30 | [-42.555, -30.329] |
| overlap80 | Jaccard | 235 | 0.378 | 0.434 | 0.393 | 0.452 | 0.87 | 84% | 4.6e-28 | [-0.067, -0.045] |
| array_share | pp | 235 | 0.286 | 1.460 | 0.988 | 2.197 | 0.20 | 86% | 2.2e-26 | [-1.404, -0.991] |
| integral |  | 235 | 0.104 | 0.140 | 0.136 | 0.182 | 0.75 | 63% | 6.7e-08 | [-0.054, -0.012] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.622 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 9.718 | 24.085 | 0.00 | 39% | 0.00033 | [-30.000, +0.000] |
| centroid | m | 71 | 57.979 | 83.988 | 64.629 | 85.742 | 0.69 | 72% | 6.5e-06 | [-33.842, -15.688] |
| overlap80 | Jaccard | 71 | 0.364 | 0.420 | 0.385 | 0.444 | 0.87 | 82% | 8.9e-08 | [-0.075, -0.033] |
| array_share | pp | 71 | 1.255 | 3.839 | 2.305 | 4.402 | 0.33 | 83% | 1.3e-07 | [-3.266, -1.625] |
| integral |  | 71 | 0.104 | 0.160 | 0.128 | 0.188 | 0.65 | 72% | 1.8e-05 | [-0.076, -0.008] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.636 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 7.857 | 13.571 | nan | 24% | 0.021 | [-30.000, +0.000] |
| centroid | m | 42 | 56.270 | 65.412 | 57.180 | 60.700 | 0.86 | 52% | 0.57 | [-20.792, +13.711] |
| overlap80 | Jaccard | 42 | 0.362 | 0.414 | 0.383 | 0.427 | 0.87 | 76% | 0.0014 | [-0.074, -0.022] |
| array_share | pp | 42 | 3.512 | 5.004 | 3.878 | 5.594 | 0.70 | 74% | 0.0022 | [-2.454, -0.509] |
| integral |  | 42 | 0.113 | 0.181 | 0.165 | 0.245 | 0.62 | 71% | 9.4e-05 | [-0.143, -0.022] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.638 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.473 | 0.631 | 0.497 | 0.662 | 0.75 | 91% | 2.5e-35 | [-0.181, -0.138] |
| shape_1d |  | 235 | 0.071 | 0.141 | 0.074 | 0.154 | 0.50 | 93% | 5.4e-37 | [-0.083, -0.060] |
| rel_l2 |  | 235 | 0.340 | 0.541 | 0.363 | 0.580 | 0.63 | 91% | 6.4e-37 | [-0.230, -0.178] |
| rel_l2_T |  | 235 | 0.328 | 0.515 | 0.352 | 0.555 | 0.64 | 91% | 7.2e-37 | [-0.222, -0.167] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.77 | 92% | 5.1e-35 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.64 | 91% | 1.4e-36 | [-0.006, -0.005] |
| pearson_T | r | 235 | 0.044 | 0.123 | 0.059 | 0.142 | 0.36 | 92% | 6.2e-35 | [-0.088, -0.066] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.77 | 94% | 1.2e-34 | [-0.007, -0.005] |
| psnr_T | dB | 235 | -39.832 | -36.139 | -39.867 | -35.752 | nan | 91% | 3.9e-37 | [-4.507, -2.943] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.442 | 0.607 | 0.479 | 0.639 | 0.73 | 93% | 1.2e-11 | [-0.188, -0.110] |
| shape_1d |  | 71 | 0.063 | 0.099 | 0.067 | 0.106 | 0.64 | 89% | 1.6e-10 | [-0.050, -0.024] |
| rel_l2 |  | 71 | 0.316 | 0.466 | 0.333 | 0.491 | 0.68 | 93% | 3.2e-12 | [-0.194, -0.115] |
| rel_l2_T |  | 71 | 0.303 | 0.441 | 0.319 | 0.461 | 0.69 | 93% | 5.7e-12 | [-0.174, -0.109] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.82 | 89% | 2.8e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.69 | 93% | 7.2e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.037 | 0.083 | 0.050 | 0.098 | 0.45 | 90% | 1.2e-10 | [-0.057, -0.034] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.018 | 0.022 | 0.79 | 93% | 1.3e-09 | [-0.006, -0.002] |
| psnr_T | dB | 71 | -41.958 | -38.622 | -41.818 | -38.269 | nan | 93% | 2.7e-12 | [-4.100, -2.713] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.526 |
| oct_N | 20 | 0.763 |
| oct_NE | 7 | 0.555 |
| oct_E | 10 | 0.759 |
| oct_SE | 24 | 0.711 |
| oct_S | 42 | 0.698 |
| oct_SW | 48 | 0.419 |
| oct_W | 40 | 0.391 |
| oct_NW | 44 | 0.508 |
| north_N_NE_NW | 71 | 0.597 |
| not_north | 164 | 0.505 |
| array_in_view_gt5pct | 42 | 0.800 |
| array_absent_le5pct | 193 | 0.502 |
| zL_tercile_most_unstable | 79 | 0.683 |
| zL_tercile_middle | 78 | 0.552 |
| zL_tercile_least_unstable | 78 | 0.392 |
| zi_tercile_shallow | 86 | 0.524 |
| zi_tercile_middle | 77 | 0.481 |
| zi_tercile_deep | 72 | 0.552 |
| seed_shared_with_train | 231 | 0.528 |
| seed_not_in_train | 4 | 0.464 |

## fno_cone

### all records

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 21.574 | 80.936 | 0.00 | 64% | 1.7e-21 | [-60.000, -30.000] |
| centroid | m | 235 | 55.096 | 91.855 | 62.692 | 102.180 | 0.60 | 82% | 2.1e-30 | [-42.555, -30.329] |
| overlap80 | Jaccard | 235 | 0.378 | 0.434 | 0.393 | 0.452 | 0.87 | 84% | 4.6e-28 | [-0.067, -0.045] |
| array_share | pp | 235 | 0.286 | 1.460 | 0.988 | 2.197 | 0.20 | 86% | 2.2e-26 | [-1.404, -0.991] |
| integral |  | 235 | 0.104 | 0.140 | 0.136 | 0.182 | 0.75 | 63% | 6.7e-08 | [-0.054, -0.012] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.622 / Kljun 0.566.
### N/NE/NW only (71 records)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 9.718 | 24.085 | 0.00 | 39% | 0.00033 | [-30.000, +0.000] |
| centroid | m | 71 | 57.979 | 83.988 | 64.629 | 85.742 | 0.69 | 72% | 6.5e-06 | [-33.842, -15.688] |
| overlap80 | Jaccard | 71 | 0.364 | 0.420 | 0.385 | 0.444 | 0.87 | 82% | 8.9e-08 | [-0.075, -0.033] |
| array_share | pp | 71 | 1.255 | 3.839 | 2.305 | 4.402 | 0.33 | 83% | 1.3e-07 | [-3.266, -1.625] |
| integral |  | 71 | 0.104 | 0.160 | 0.128 | 0.188 | 0.65 | 72% | 1.8e-05 | [-0.076, -0.008] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.636 / Kljun 0.580.
### array in view, LES share > 5% (42)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | 7.857 | 13.571 | nan | 24% | 0.021 | [-30.000, +0.000] |
| centroid | m | 42 | 56.270 | 65.412 | 57.180 | 60.700 | 0.86 | 52% | 0.57 | [-20.792, +13.711] |
| overlap80 | Jaccard | 42 | 0.362 | 0.414 | 0.383 | 0.427 | 0.87 | 76% | 0.0014 | [-0.074, -0.022] |
| array_share | pp | 42 | 3.512 | 5.004 | 3.878 | 5.594 | 0.70 | 74% | 0.0022 | [-2.454, -0.509] |
| integral |  | 42 | 0.113 | 0.181 | 0.165 | 0.245 | 0.62 | 71% | 9.4e-05 | [-0.143, -0.022] |

overlap80 is reported as 1 - Jaccard (smaller is better); the raw medians are FNO 0.638 / Kljun 0.586.
### shape and 2-D field metrics, all records (not in the composite; per-cell agreement sits on the noise floor)

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 235 | 0.473 | 0.631 | 0.497 | 0.662 | 0.75 | 91% | 2.5e-35 | [-0.181, -0.138] |
| shape_1d |  | 235 | 0.071 | 0.141 | 0.074 | 0.154 | 0.50 | 93% | 5.4e-37 | [-0.083, -0.060] |
| rel_l2 |  | 235 | 0.340 | 0.541 | 0.363 | 0.580 | 0.63 | 91% | 6.4e-37 | [-0.230, -0.178] |
| rel_l2_T |  | 235 | 0.328 | 0.515 | 0.352 | 0.555 | 0.64 | 91% | 7.2e-37 | [-0.222, -0.167] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.001 | 0.002 | 0.77 | 92% | 5.1e-35 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.010 | 0.016 | 0.64 | 91% | 1.4e-36 | [-0.006, -0.005] |
| pearson_T | r | 235 | 0.044 | 0.123 | 0.059 | 0.142 | 0.36 | 92% | 6.2e-35 | [-0.088, -0.066] |
| ssim_T |  | 235 | 0.020 | 0.025 | 0.022 | 0.027 | 0.77 | 94% | 1.2e-34 | [-0.007, -0.005] |
| psnr_T | dB | 235 | -39.832 | -36.139 | -39.867 | -35.752 | nan | 91% | 3.9e-37 | [-4.507, -2.943] |

### shape and 2-D field metrics, N/NE/NW only

| metric | unit | n | FNO median | Kljun median | FNO mean | Kljun mean | ratio | FNO wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_l1_2d |  | 71 | 0.442 | 0.607 | 0.479 | 0.639 | 0.73 | 93% | 1.2e-11 | [-0.188, -0.110] |
| shape_1d |  | 71 | 0.063 | 0.099 | 0.067 | 0.106 | 0.64 | 89% | 1.6e-10 | [-0.050, -0.024] |
| rel_l2 |  | 71 | 0.316 | 0.466 | 0.333 | 0.491 | 0.68 | 93% | 3.2e-12 | [-0.194, -0.115] |
| rel_l2_T |  | 71 | 0.303 | 0.441 | 0.319 | 0.461 | 0.69 | 93% | 5.7e-12 | [-0.174, -0.109] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.001 | 0.002 | 0.82 | 89% | 2.8e-10 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.010 | 0.014 | 0.011 | 0.015 | 0.69 | 93% | 7.2e-12 | [-0.005, -0.003] |
| pearson_T | r | 71 | 0.037 | 0.083 | 0.050 | 0.098 | 0.45 | 90% | 1.2e-10 | [-0.057, -0.034] |
| ssim_T |  | 71 | 0.016 | 0.020 | 0.018 | 0.022 | 0.79 | 93% | 1.3e-09 | [-0.006, -0.002] |
| psnr_T | dB | 71 | -41.958 | -38.622 | -41.818 | -38.269 | nan | 93% | 2.7e-12 | [-4.100, -2.713] |


Larger-is-better metrics (overlap80, pearson_T, ssim_T, psnr_T) are tabulated in their smaller-is-better form (1 - value, or -PSNR); the raw medians are in eval.json under `raw_medians`.


### composite (geometric mean of the five ratios) by group

| group | n | composite |
|---|---|---|
| all | 235 | 0.526 |
| oct_N | 20 | 0.763 |
| oct_NE | 7 | 0.555 |
| oct_E | 10 | 0.759 |
| oct_SE | 24 | 0.711 |
| oct_S | 42 | 0.698 |
| oct_SW | 48 | 0.419 |
| oct_W | 40 | 0.391 |
| oct_NW | 44 | 0.508 |
| north_N_NE_NW | 71 | 0.597 |
| not_north | 164 | 0.505 |
| array_in_view_gt5pct | 42 | 0.800 |
| array_absent_le5pct | 193 | 0.502 |
| zL_tercile_most_unstable | 79 | 0.683 |
| zL_tercile_middle | 78 | 0.552 |
| zL_tercile_least_unstable | 78 | 0.392 |
| zi_tercile_shallow | 86 | 0.524 |
| zi_tercile_middle | 77 | 0.481 |
| zi_tercile_deep | 72 | 0.552 |
| seed_shared_with_train | 231 | 0.528 |
| seed_not_in_train | 4 | 0.464 |

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
| fno | 0.1162 |
| fno_cone | 0.1162 |
| kljun | 0.0801 |
| les | 0.1527 |
