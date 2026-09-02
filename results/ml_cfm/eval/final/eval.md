# CFM evaluation `final` on val: 235 records, 160 samples (5 seeds x 32), FNO ensemble of 5

## CFM mean vs Kljun

### all records (235)

| metric | unit | n | CFM mean median | Kljun median | ratio | CFM mean wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 30.000 | 0.00 | 65% | 2e-21 | [-60.000, -30.000] |
| centroid | m | 235 | 51.701 | 91.855 | 0.56 | 80% | 2.7e-27 | [-47.677, -35.130] |
| overlap80 | Jaccard | 235 | 0.384 | 0.434 | 0.89 | 80% | 3.1e-25 | [-0.065, -0.041] |
| array_share | pp | 235 | 0.247 | 1.460 | 0.17 | 85% | 6.7e-26 | [-1.438, -1.027] |
| integral |  | 235 | 0.097 | 0.140 | 0.70 | 65% | 1.9e-09 | [-0.061, -0.018] |
| shape_l1_2d |  | 235 | 0.471 | 0.631 | 0.75 | 92% | 4.8e-35 | [-0.181, -0.138] |
| shape_1d |  | 235 | 0.064 | 0.141 | 0.45 | 93% | 2.5e-37 | [-0.090, -0.067] |
| rel_l2 |  | 235 | 0.334 | 0.541 | 0.62 | 92% | 2.3e-37 | [-0.236, -0.183] |
| rel_l2_T |  | 235 | 0.325 | 0.515 | 0.63 | 91% | 3.8e-37 | [-0.224, -0.171] |
| mae_T | asinh | 235 | 0.001 | 0.002 | 0.78 | 92% | 5.3e-34 | [-0.000, -0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.014 | 0.63 | 91% | 1.1e-36 | [-0.006, -0.005] |
| pearson_T | r | 235 | 0.045 | 0.123 | 0.37 | 91% | 1.2e-34 | [-0.088, -0.066] |
| ssim_T |  | 235 | 0.019 | 0.025 | 0.75 | 86% | 2.5e-31 | [-0.007, -0.005] |
| psnr_T | dB | 235 | -39.923 | -36.139 | nan | 91% | 2.2e-37 | [-4.590, -3.191] |

overlap80 is 1 - Jaccard here; raw medians CFM mean 0.616 / Kljun 0.566.

### N/NE/NW (71)

| metric | unit | n | CFM mean median | Kljun median | ratio | CFM mean wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 30.000 | 0.00 | 39% | 0.00017 | [-30.000, +0.000] |
| centroid | m | 71 | 52.230 | 83.988 | 0.62 | 69% | 3.3e-06 | [-42.623, -19.435] |
| overlap80 | Jaccard | 71 | 0.374 | 0.420 | 0.89 | 79% | 2.6e-07 | [-0.079, -0.027] |
| array_share | pp | 71 | 1.184 | 3.839 | 0.31 | 83% | 3.1e-07 | [-3.387, -1.734] |
| integral |  | 71 | 0.091 | 0.160 | 0.57 | 72% | 8.3e-07 | [-0.095, -0.025] |
| shape_l1_2d |  | 71 | 0.448 | 0.607 | 0.74 | 94% | 3.8e-11 | [-0.190, -0.123] |
| shape_1d |  | 71 | 0.061 | 0.099 | 0.61 | 90% | 1.3e-11 | [-0.053, -0.028] |
| rel_l2 |  | 71 | 0.315 | 0.466 | 0.68 | 93% | 1.3e-12 | [-0.206, -0.126] |
| rel_l2_T |  | 71 | 0.297 | 0.441 | 0.67 | 92% | 2.9e-12 | [-0.178, -0.113] |
| mae_T | asinh | 71 | 0.001 | 0.002 | 0.80 | 90% | 1.3e-09 | [-0.000, -0.000] |
| rmse_T | asinh | 71 | 0.009 | 0.014 | 0.68 | 92% | 3.6e-12 | [-0.006, -0.003] |
| pearson_T | r | 71 | 0.034 | 0.083 | 0.41 | 90% | 3.3e-10 | [-0.061, -0.034] |
| ssim_T |  | 71 | 0.015 | 0.020 | 0.76 | 76% | 7.9e-08 | [-0.006, -0.003] |
| psnr_T | dB | 71 | -42.378 | -38.622 | nan | 92% | 2.1e-12 | [-4.623, -2.926] |

overlap80 is 1 - Jaccard here; raw medians CFM mean 0.626 / Kljun 0.580.

### array in view (LES share > 5%) (42)

| metric | unit | n | CFM mean median | Kljun median | ratio | CFM mean wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | nan | 24% | 0.0067 | [-30.000, +0.000] |
| centroid | m | 42 | 57.553 | 65.412 | 0.88 | 48% | 0.46 | [-24.249, +15.083] |
| overlap80 | Jaccard | 42 | 0.370 | 0.414 | 0.89 | 76% | 0.0011 | [-0.081, -0.023] |
| array_share | pp | 42 | 3.479 | 5.004 | 0.70 | 74% | 0.0016 | [-2.830, -0.392] |
| integral |  | 42 | 0.092 | 0.181 | 0.51 | 76% | 4.3e-07 | [-0.155, -0.053] |
| shape_l1_2d |  | 42 | 0.465 | 0.598 | 0.78 | 88% | 2.7e-06 | [-0.183, -0.088] |
| shape_1d |  | 42 | 0.066 | 0.083 | 0.79 | 83% | 9.8e-07 | [-0.028, -0.012] |
| rel_l2 |  | 42 | 0.322 | 0.462 | 0.70 | 86% | 8.7e-09 | [-0.188, -0.098] |
| rel_l2_T |  | 42 | 0.307 | 0.437 | 0.70 | 86% | 4.6e-08 | [-0.163, -0.086] |
| mae_T | asinh | 42 | 0.001 | 0.002 | 0.84 | 83% | 0.00011 | [-0.000, -0.000] |
| rmse_T | asinh | 42 | 0.011 | 0.015 | 0.74 | 86% | 4.6e-08 | [-0.006, -0.002] |
| pearson_T | r | 42 | 0.041 | 0.071 | 0.59 | 83% | 9.4e-05 | [-0.044, -0.021] |
| ssim_T |  | 42 | 0.016 | 0.018 | 0.86 | 64% | 0.013 | [-0.005, +0.000] |
| psnr_T | dB | 42 | -41.570 | -38.627 | nan | 86% | 2.3e-08 | [-3.941, -2.046] |

overlap80 is 1 - Jaccard here; raw medians CFM mean 0.630 / Kljun 0.586.

## CFM mean vs FNO

### all records (235)

| metric | unit | n | CFM mean median | FNO median | ratio | CFM mean wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 0.000 | nan | 6% | 0.49 | [+0.000, +0.000] |
| centroid | m | 235 | 51.701 | 55.086 | 0.94 | 63% | 3.7e-06 | [-10.376, +0.137] |
| overlap80 | Jaccard | 235 | 0.384 | 0.378 | 1.02 | 49% | 0.28 | [-0.005, +0.013] |
| array_share | pp | 235 | 0.247 | 0.286 | 0.86 | 61% | 0.00082 | [-0.088, +0.004] |
| integral |  | 235 | 0.097 | 0.104 | 0.93 | 49% | 0.55 | [-0.018, +0.007] |
| shape_l1_2d |  | 235 | 0.471 | 0.473 | 1.00 | 57% | 0.017 | [-0.014, +0.010] |
| shape_1d |  | 235 | 0.064 | 0.071 | 0.90 | 68% | 7e-10 | [-0.010, -0.003] |
| rel_l2 |  | 235 | 0.334 | 0.340 | 0.98 | 54% | 0.085 | [-0.016, +0.005] |
| rel_l2_T |  | 235 | 0.325 | 0.328 | 0.99 | 55% | 0.069 | [-0.011, +0.007] |
| mae_T | asinh | 235 | 0.001 | 0.001 | 1.01 | 51% | 0.8 | [-0.000, +0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.009 | 0.99 | 55% | 0.1 | [-0.000, +0.000] |
| pearson_T | r | 235 | 0.045 | 0.044 | 1.02 | 60% | 0.031 | [-0.003, +0.003] |
| ssim_T |  | 235 | 0.019 | 0.020 | 0.98 | 62% | 0.00012 | [-0.001, +0.001] |
| psnr_T | dB | 235 | -39.923 | -39.833 | nan | 55% | 0.026 | [-0.515, +0.269] |

overlap80 is 1 - Jaccard here; raw medians CFM mean 0.616 / FNO 0.622.

### N/NE/NW (71)

| metric | unit | n | CFM mean median | FNO median | ratio | CFM mean wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 0.000 | nan | 1% | 0.32 | [+0.000, +0.000] |
| centroid | m | 71 | 52.230 | 57.933 | 0.90 | 62% | 0.02 | [-15.080, +2.098] |
| overlap80 | Jaccard | 71 | 0.374 | 0.364 | 1.03 | 44% | 0.26 | [-0.015, +0.018] |
| array_share | pp | 71 | 1.184 | 1.255 | 0.94 | 59% | 0.16 | [-0.510, +0.236] |
| integral |  | 71 | 0.091 | 0.104 | 0.87 | 48% | 0.67 | [-0.042, -0.003] |
| shape_l1_2d |  | 71 | 0.448 | 0.442 | 1.01 | 59% | 0.2 | [-0.030, +0.025] |
| shape_1d |  | 71 | 0.061 | 0.063 | 0.96 | 66% | 0.0012 | [-0.009, +0.002] |
| rel_l2 |  | 71 | 0.315 | 0.316 | 1.00 | 61% | 0.023 | [-0.031, +0.015] |
| rel_l2_T |  | 71 | 0.297 | 0.303 | 0.98 | 63% | 0.028 | [-0.023, +0.016] |
| mae_T | asinh | 71 | 0.001 | 0.001 | 0.97 | 51% | 0.65 | [-0.000, +0.000] |
| rmse_T | asinh | 71 | 0.009 | 0.010 | 0.99 | 63% | 0.053 | [-0.001, +0.001] |
| pearson_T | r | 71 | 0.034 | 0.037 | 0.91 | 68% | 0.012 | [-0.008, +0.004] |
| ssim_T |  | 71 | 0.015 | 0.016 | 0.95 | 62% | 0.14 | [-0.002, +0.001] |
| psnr_T | dB | 71 | -42.378 | -41.958 | nan | 63% | 0.032 | [-1.008, +0.199] |

overlap80 is 1 - Jaccard here; raw medians CFM mean 0.626 / FNO 0.636.

### array in view (LES share > 5%) (42)

| metric | unit | n | CFM mean median | FNO median | ratio | CFM mean wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | nan | 2% | 0.32 | [+0.000, +0.000] |
| centroid | m | 42 | 57.553 | 56.258 | 1.02 | 50% | 0.89 | [-13.220, +9.545] |
| overlap80 | Jaccard | 42 | 0.370 | 0.362 | 1.02 | 48% | 0.37 | [-0.022, +0.020] |
| array_share | pp | 42 | 3.479 | 3.512 | 0.99 | 57% | 0.18 | [-0.873, +0.502] |
| integral |  | 42 | 0.092 | 0.113 | 0.82 | 55% | 0.74 | [-0.050, +0.025] |
| shape_l1_2d |  | 42 | 0.465 | 0.452 | 1.03 | 52% | 0.24 | [-0.021, +0.035] |
| shape_1d |  | 42 | 0.066 | 0.070 | 0.94 | 57% | 0.14 | [-0.011, +0.006] |
| rel_l2 |  | 42 | 0.322 | 0.328 | 0.98 | 52% | 0.49 | [-0.030, +0.025] |
| rel_l2_T |  | 42 | 0.307 | 0.305 | 1.01 | 55% | 0.44 | [-0.019, +0.026] |
| mae_T | asinh | 42 | 0.001 | 0.001 | 1.01 | 43% | 0.35 | [-0.000, +0.000] |
| rmse_T | asinh | 42 | 0.011 | 0.011 | 1.00 | 55% | 0.65 | [-0.001, +0.001] |
| pearson_T | r | 42 | 0.041 | 0.041 | 1.02 | 60% | 0.24 | [-0.007, +0.007] |
| ssim_T |  | 42 | 0.016 | 0.016 | 1.00 | 43% | 0.31 | [-0.002, +0.002] |
| psnr_T | dB | 42 | -41.570 | -41.492 | nan | 55% | 0.42 | [-0.746, +0.646] |

overlap80 is 1 - Jaccard here; raw medians CFM mean 0.630 / FNO 0.638.

## CFM mean (filtered) vs FNO (filtered)

### all records (235)

| metric | unit | n | CFM mean (filtered) median | FNO (filtered) median | ratio | CFM mean (filtered) wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|
| peak_x | m | 235 | 0.000 | 0.000 | nan | 6% | 0.49 | [+0.000, +0.000] |
| centroid | m | 235 | 51.582 | 55.514 | 0.93 | 63% | 2e-06 | [-10.548, -0.262] |
| overlap80 | Jaccard | 235 | 0.382 | 0.378 | 1.01 | 47% | 0.33 | [-0.005, +0.013] |
| array_share | pp | 235 | 0.247 | 0.287 | 0.86 | 61% | 0.00085 | [-0.087, +0.004] |
| integral |  | 235 | 0.097 | 0.104 | 0.93 | 49% | 0.53 | [-0.018, +0.007] |
| shape_l1_2d |  | 235 | 0.470 | 0.472 | 1.00 | 57% | 0.018 | [-0.014, +0.010] |
| shape_1d |  | 235 | 0.064 | 0.071 | 0.90 | 68% | 4.9e-10 | [-0.010, -0.003] |
| rel_l2 |  | 235 | 0.334 | 0.340 | 0.98 | 54% | 0.085 | [-0.016, +0.005] |
| rel_l2_T |  | 235 | 0.325 | 0.328 | 0.99 | 55% | 0.069 | [-0.011, +0.007] |
| mae_T | asinh | 235 | 0.001 | 0.001 | 1.01 | 51% | 0.8 | [-0.000, +0.000] |
| rmse_T | asinh | 235 | 0.009 | 0.009 | 0.99 | 55% | 0.1 | [-0.000, +0.000] |
| pearson_T | r | 235 | 0.045 | 0.044 | 1.02 | 60% | 0.031 | [-0.003, +0.003] |
| ssim_T |  | 235 | 0.019 | 0.020 | 0.98 | 62% | 0.00011 | [-0.001, +0.001] |
| psnr_T | dB | 235 | -39.923 | -39.833 | nan | 55% | 0.026 | [-0.515, +0.269] |

overlap80 is 1 - Jaccard here; raw medians CFM mean (filtered) 0.618 / FNO (filtered) 0.622.

### N/NE/NW (71)

| metric | unit | n | CFM mean (filtered) median | FNO (filtered) median | ratio | CFM mean (filtered) wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|
| peak_x | m | 71 | 0.000 | 0.000 | nan | 1% | 0.32 | [+0.000, +0.000] |
| centroid | m | 71 | 52.133 | 58.297 | 0.89 | 62% | 0.017 | [-15.313, +1.763] |
| overlap80 | Jaccard | 71 | 0.372 | 0.364 | 1.02 | 45% | 0.31 | [-0.016, +0.016] |
| array_share | pp | 71 | 1.185 | 1.257 | 0.94 | 59% | 0.17 | [-0.514, +0.233] |
| integral |  | 71 | 0.090 | 0.104 | 0.87 | 48% | 0.69 | [-0.042, -0.003] |
| shape_l1_2d |  | 71 | 0.448 | 0.442 | 1.01 | 59% | 0.21 | [-0.030, +0.026] |
| shape_1d |  | 71 | 0.061 | 0.063 | 0.96 | 66% | 0.0012 | [-0.009, +0.002] |
| rel_l2 |  | 71 | 0.315 | 0.316 | 1.00 | 61% | 0.023 | [-0.031, +0.015] |
| rel_l2_T |  | 71 | 0.297 | 0.303 | 0.98 | 63% | 0.028 | [-0.023, +0.016] |
| mae_T | asinh | 71 | 0.001 | 0.001 | 0.97 | 51% | 0.66 | [-0.000, +0.000] |
| rmse_T | asinh | 71 | 0.009 | 0.010 | 0.99 | 63% | 0.053 | [-0.001, +0.001] |
| pearson_T | r | 71 | 0.034 | 0.037 | 0.91 | 68% | 0.012 | [-0.008, +0.004] |
| ssim_T |  | 71 | 0.015 | 0.016 | 0.95 | 62% | 0.13 | [-0.002, +0.001] |
| psnr_T | dB | 71 | -42.378 | -41.958 | nan | 63% | 0.032 | [-1.008, +0.199] |

overlap80 is 1 - Jaccard here; raw medians CFM mean (filtered) 0.628 / FNO (filtered) 0.636.

### array in view (LES share > 5%) (42)

| metric | unit | n | CFM mean (filtered) median | FNO (filtered) median | ratio | CFM mean (filtered) wins | Wilcoxon p | median diff 95% CI |
|---|---|---|---|---|---|---|---|---|
| peak_x | m | 42 | 0.000 | 0.000 | nan | 2% | 0.32 | [+0.000, +0.000] |
| centroid | m | 42 | 57.623 | 56.404 | 1.02 | 50% | 0.92 | [-12.811, +9.454] |
| overlap80 | Jaccard | 42 | 0.370 | 0.363 | 1.02 | 50% | 0.39 | [-0.023, +0.019] |
| array_share | pp | 42 | 3.469 | 3.516 | 0.99 | 57% | 0.19 | [-0.865, +0.497] |
| integral |  | 42 | 0.092 | 0.112 | 0.82 | 55% | 0.72 | [-0.050, +0.025] |
| shape_l1_2d |  | 42 | 0.464 | 0.451 | 1.03 | 52% | 0.23 | [-0.021, +0.035] |
| shape_1d |  | 42 | 0.066 | 0.070 | 0.94 | 57% | 0.15 | [-0.011, +0.006] |
| rel_l2 |  | 42 | 0.322 | 0.328 | 0.98 | 52% | 0.49 | [-0.030, +0.025] |
| rel_l2_T |  | 42 | 0.307 | 0.305 | 1.01 | 55% | 0.44 | [-0.019, +0.026] |
| mae_T | asinh | 42 | 0.001 | 0.001 | 1.01 | 43% | 0.33 | [-0.000, +0.000] |
| rmse_T | asinh | 42 | 0.011 | 0.011 | 1.00 | 55% | 0.65 | [-0.001, +0.001] |
| pearson_T | r | 42 | 0.041 | 0.041 | 1.02 | 60% | 0.24 | [-0.007, +0.007] |
| ssim_T |  | 42 | 0.016 | 0.016 | 0.99 | 43% | 0.32 | [-0.002, +0.002] |
| psnr_T | dB | 42 | -41.570 | -41.492 | nan | 55% | 0.42 | [-0.746, +0.646] |

overlap80 is 1 - Jaccard here; raw medians CFM mean (filtered) 0.630 / FNO (filtered) 0.637.

## Composite (geometric mean of the five production-metric ratios) by group

| group | n | CFM/Kljun | CFM filtered/Kljun | FNO/Kljun | FNO filtered/Kljun | CFM/FNO |
|---|---|---|---|---|---|---|
| all | 235 | 0.492 | 0.491 | 0.526 | 0.526 | 0.948 |
| oct_N | 20 | 0.743 | 0.743 | 0.763 | 0.764 | 0.979 |
| oct_NE | 7 | 0.496 | 0.496 | 0.554 | 0.557 | 0.915 |
| oct_E | 10 | 0.849 | 0.851 | 0.759 | 0.761 | 1.093 |
| oct_SE | 24 | 0.699 | 0.699 | 0.711 | 0.711 | 0.982 |
| oct_S | 42 | 0.647 | 0.646 | 0.698 | 0.698 | 0.927 |
| oct_SW | 48 | 0.372 | 0.371 | 0.419 | 0.420 | 0.909 |
| oct_W | 40 | 0.307 | 0.307 | 0.391 | 0.392 | 0.785 |
| oct_NW | 44 | 0.506 | 0.506 | 0.508 | 0.508 | 0.997 |
| north_N_NE_NW | 71 | 0.558 | 0.556 | 0.597 | 0.598 | 0.947 |
| not_north | 164 | 0.466 | 0.467 | 0.505 | 0.506 | 0.938 |
| array_in_view_gt5pct | 42 | 0.774 | 0.773 | 0.800 | 0.801 | 0.967 |
| array_absent_le5pct | 193 | 0.457 | 0.458 | 0.502 | 0.503 | 0.928 |
| zL_tercile_most_unstable | 79 | 0.635 | 0.635 | 0.683 | 0.681 | 0.944 |
| zL_tercile_middle | 78 | 0.510 | 0.510 | 0.552 | 0.552 | 0.938 |
| zL_tercile_least_unstable | 78 | 0.331 | 0.331 | 0.392 | 0.392 | 0.875 |
| zi_tercile_shallow | 86 | 0.457 | 0.456 | 0.524 | 0.523 | 0.897 |
| zi_tercile_middle | 77 | 0.452 | 0.453 | 0.481 | 0.482 | 0.951 |
| zi_tercile_deep | 72 | 0.537 | 0.537 | 0.552 | 0.554 | 0.977 |
| seed_shared_with_train | 231 | 0.497 | 0.496 | 0.528 | 0.528 | 0.952 |
| seed_not_in_train | 4 | 0.443 | 0.443 | 0.464 | 0.464 | 0.964 |

## Per-seed composites vs Kljun (each seed's own 32-sample mean)

| seed | composite |
|---|---|
| seed0 | 0.546 |
| seed1 | 0.488 |
| seed2 | 0.490 |
| seed3 | 0.474 |
| seed4 | 0.567 |

## Dependence of the mean on the sample count S

| S | composite vs Kljun | rel_l2 | shape_l1_2d | overlap80 (1-J) | array_share pp |
|---|---|---|---|---|---|
| 1 | 0.743 | 0.430 | 0.642 | 0.512 | 0.280 |
| 4 | 0.580 | 0.376 | 0.556 | 0.587 | 0.268 |
| 8 | 0.577 | 0.361 | 0.530 | 0.604 | 0.293 |
| 32 | 0.546 | 0.345 | 0.500 | 0.617 | 0.279 |
| 160 | 0.492 | 0.334 | 0.471 | 0.616 | 0.247 |

## The connected-component filter (rule A, 99.9% of |mass|)

| field | median mass removed | mean | max | median components | median kept | peak kept |
|---|---|---|---|---|---|---|
| cfm | 0.100% | 0.100% | 0.10% | 23 | 23 | 100% |
| fno | 0.100% | 0.100% | 0.10% | 16 | 16 | 100% |
| kljun | 0.100% | 0.100% | 0.10% | 1 | 1 | 100% |
| les | 0.099% | 0.099% | 0.10% | 3 | 3 | 100% |

Rule B: the LES target becomes single-connected at a median level tau* = 3.98e-01 of its peak (IQR 2.0e-01-7.9e-01); at that level the mass removed is cfm 69.15%, fno 67.60%, kljun 67.98%, les 70.72% (medians).

## Sample spread against the realisation floors

| group | n | array-share sd [pp] | 5-95% range [pp] | integral sd | integral 5-95% | peak_x sd [m] | between-sample overlap80 | shape L1 | centroid [m] | rel L2 |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 235 | 0.37 | 1.16 | 0.129 | 0.418 | 18 | 0.564 | 0.538 | 84 | 0.386 |
| north_N_NE_NW | 71 | 2.13 | 6.55 | 0.123 | 0.400 | 14 | 0.581 | 0.518 | 73 | 0.340 |
| array_in_view_gt5pct | 42 | 3.50 | 11.10 | 0.133 | 0.427 | 13 | 0.582 | 0.520 | 84 | 0.350 |

| quantity | floor | independent realisations | source |
|---|---|---|---|
| array_share [pp] | 5.34 between the two windows of case_2023111718 | 1 pair | `results/ml/eval/floor/pair_floor.json` |
| array_share [pp] | 4.58 and 0.67 run-to-run (5.65->1.07, 1.14->0.47) | 2 runs x 2 cases | `results/les_realisation_spread.txt` |
| array_share [pp] | 0.19 median within-window SE | ~1000 records | `corpus/pairs_npz meta array_share_se` |
| integral | 0.0007 between the two windows; 0.444 and 0.148 run-to-run | 1 pair; 2 x 2 | `pair_floor.json; les_realisation_spread.txt` |
| peak_x [m] | 0 between the two windows; 30 (one cell) run-to-run | 1 pair; 2 x 2 | `same` |
| centroid [m] | 51 between the two windows; 46 run-to-run | 1 pair; 2 x 1 | `same` |
| overlap80 | 0.507 between the two windows; 0.56 two LPDM seeds | 1 pair; 1 | `same` |
| shape_l1_2d | 0.630 between the two windows; 0.41 two LPDM seeds | 1 pair; 1 | `same` |

## Calibration: the LES target as one more draw from the sample set

| group | metric | n | PIT KS p | PIT mean | z sd | median |z| | cover 50% | cover 90% | sample sd (median) | |LES - mean| (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| all | array_share | 235 | 1.8e-05 | 0.448 | 1.21 | 0.67 | 0.50 | 0.88 | 0.371 | 0.237 |
| all | integral | 235 | 0.0017 | 0.559 | 1.29 | 0.77 | 0.44 | 0.80 | 0.129 | 0.097 |
| all | peak_x | 235 | 6.3e-05 | 0.516 | 1.08 | 0.56 | 0.85 | 0.95 | 18.343 | 11.438 |
| all | centroid_dist | 235 | 5.2e-08 | 0.569 | 0.69 | 0.47 | 0.63 | 0.97 | 72.408 | 35.366 |
| north_N_NE_NW | array_share | 71 | 0.061 | 0.451 | 1.24 | 0.70 | 0.45 | 0.85 | 2.134 | 1.167 |
| north_N_NE_NW | integral | 71 | 0.035 | 0.599 | 1.27 | 0.74 | 0.49 | 0.80 | 0.123 | 0.091 |
| north_N_NE_NW | peak_x | 71 | 0.01 | 0.494 | 0.88 | 0.49 | 0.90 | 0.97 | 14.114 | 6.750 |
| north_N_NE_NW | centroid_dist | 71 | 0.0012 | 0.613 | 0.65 | 0.59 | 0.55 | 1.00 | 68.075 | 40.129 |
| array_in_view_gt5pct | array_share | 42 | 0.032 | 0.554 | 1.77 | 0.99 | 0.33 | 0.74 | 3.503 | 3.524 |
| array_in_view_gt5pct | integral | 42 | 0.0087 | 0.669 | 1.46 | 0.75 | 0.50 | 0.71 | 0.133 | 0.092 |
| array_in_view_gt5pct | peak_x | 42 | 0.019 | 0.448 | 0.98 | 0.64 | 0.86 | 0.95 | 13.336 | 7.500 |
| array_in_view_gt5pct | centroid_dist | 42 | 0.012 | 0.608 | 0.63 | 0.55 | 0.57 | 1.00 | 70.528 | 39.726 |
| not_north | array_share | 164 | 0.00048 | 0.446 | 1.19 | 0.66 | 0.52 | 0.90 | 0.241 | 0.146 |
| not_north | integral | 164 | 0.042 | 0.542 | 1.29 | 0.84 | 0.42 | 0.80 | 0.133 | 0.105 |
| not_north | peak_x | 164 | 0.0013 | 0.525 | 1.15 | 0.73 | 0.82 | 0.95 | 23.908 | 15.938 |
| not_north | centroid_dist | 164 | 2.4e-05 | 0.551 | 0.71 | 0.42 | 0.66 | 0.96 | 75.233 | 31.720 |

## Sharpness (asinh space, interior)

| field | mean |grad| | high-k power fraction (k >= 32) |
|---|---|---|
| LES | 0.0012 | 0.0037 |
| Kljun | 0.0008 | 0.0009 |
| FNO | 0.0009 | 0.0017 |
| CFM mean | 0.0009 | 0.0021 |
| CFM sample | 0.0011 | 0.0033 |

## Cost

| seed | params | best epoch / run | val_mse_ref | gap | S | steps | solver | ms / record / sample | wall s |
|---|---|---|---|---|---|---|---|---|---|
| seed0 | 2.95 M | 60/111 | 0.000128 | x0.95 | 32 | 16 | euler | 23.66 | 1133 |
| seed1 | 2.95 M | 90/141 | 0.000117 | x0.99 | 32 | 16 | euler | 17.41 | 1288 |
| seed2 | 2.95 M | 90/141 | 0.000117 | x0.99 | 32 | 16 | euler | 17.36 | 1288 |
| seed3 | 2.95 M | 95/146 | 0.000117 | x1.00 | 32 | 16 | euler | 23.64 | 1400 |
| seed4 | 2.95 M | 120/171 | 0.000117 | x1.03 | 32 | 16 | euler | 18.96 | 1527 |

## Integral vs the asymptote 1 - z_m/z_i (median |error|)

- cfm: 0.1027
- fno: 0.1163
- kljun: 0.0801
- cfm_f: 0.1027
- fno_f: 0.1155
- kljun_f: 0.0810
- les: 0.1527
