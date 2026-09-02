# CFM calibration `final2`: 4 model variants, S = 64 samples each, val (235 records)

Bands: +-2 binomial sd around nominal — all (n 235): cover50 [0.43, 0.57], cover90 [0.86, 0.94]; north_N_NE_NW (n 71): cover50 [0.38, 0.62], cover90 [0.83, 0.97]; array_in_view_gt5pct (n 42): cover50 [0.35, 0.65], cover90 [0.81, 0.99]

| model | loss | init | sigma train / sample | steps | target | best epoch | val_mse_ref | val CRPS | ms/record/sample |
|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | fm | scratch | 0.1 / 0.1 | 16 | none | 90 | 1.172e-04 | nan | 8.1 |
| crps_share_ft | crps | seed1 | 0.1 / 0.1 | 2 | none | 0 | 1.163e-04 | 0.0048 | 1.0 |
| crps_pure_scratch | crps | scratch | 0.1 / 0.1 | 2 | none | 45 | 1.299e-04 | 0.0042 | 1.0 |
| crps_pure_ft | crps | seed1 | 0.1 / 0.1 | 2 | none | 20 | 1.273e-04 | 0.0043 | 1.0 |

## Calibration of the array share (LES as one more draw among S samples)

| model | group | n | S | cover50 | cover90 | in band | z sd | PIT KS p | CRPS [pp] | MAE of mean [pp] | spread/skill | sample sd [pp] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | all | 235 | 64 | 0.47 | 0.88 | y/y | 1.19 | 0.00077 | 0.674 | 0.951 | 0.82 | 0.35 |
| fm_seed1 | north_N_NE_NW | 71 | 64 | 0.44 | 0.85 | y/y | 1.26 | 0.48 | 1.606 | 2.285 | 0.84 | 2.09 |
| fm_seed1 | array_in_view_gt5pct | 42 | 64 | 0.31 | 0.74 | n/n | 1.55 | 0.057 | 2.747 | 3.846 | 0.81 | 3.88 |
| crps_share_ft | all | 235 | 64 | 0.27 | 0.66 | n/n | 1.88 | 9.2e-13 | 0.725 | 0.954 | 0.53 | 0.22 |
| crps_share_ft | north_N_NE_NW | 71 | 64 | 0.28 | 0.56 | n/n | 2.05 | 0.0013 | 1.707 | 2.262 | 0.55 | 1.23 |
| crps_share_ft | array_in_view_gt5pct | 42 | 64 | 0.21 | 0.45 | n/n | 2.52 | 0.00062 | 2.921 | 3.810 | 0.53 | 2.21 |
| crps_pure_scratch | all | 235 | 64 | 0.34 | 0.71 | n/n | 1.92 | 0.00014 | 0.751 | 0.967 | 0.45 | 0.24 |
| crps_pure_scratch | north_N_NE_NW | 71 | 64 | 0.28 | 0.58 | n/n | 2.15 | 0.017 | 1.792 | 2.314 | 0.46 | 1.14 |
| crps_pure_scratch | array_in_view_gt5pct | 42 | 64 | 0.17 | 0.40 | n/n | 2.88 | 8.6e-06 | 3.139 | 3.958 | 0.44 | 2.00 |
| crps_pure_ft | all | 235 | 64 | 0.51 | 0.89 | y/y | 1.07 | 0.083 | 0.674 | 0.961 | 0.79 | 0.39 |
| crps_pure_ft | north_N_NE_NW | 71 | 64 | 0.46 | 0.83 | y/y | 1.26 | 0.48 | 1.646 | 2.337 | 0.79 | 2.02 |
| crps_pure_ft | array_in_view_gt5pct | 42 | 64 | 0.31 | 0.76 | n/n | 1.48 | 0.043 | 2.749 | 3.877 | 0.77 | 3.72 |

## Calibration of the integral

| model | group | n | cover50 | cover90 | in band | z sd | CRPS | MAE of mean | spread/skill |
|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | all | 235 | 0.36 | 0.76 | n/n | 1.41 | 0.0977 | 0.1343 | 0.67 |
| fm_seed1 | north_N_NE_NW | 71 | 0.38 | 0.76 | n/n | 1.29 | 0.0852 | 0.1166 | 0.73 |
| fm_seed1 | array_in_view_gt5pct | 42 | 0.38 | 0.69 | y/n | 1.49 | 0.1190 | 0.1562 | 0.58 |
| crps_share_ft | all | 235 | 0.25 | 0.61 | n/n | 1.96 | 0.1015 | 0.1337 | 0.49 |
| crps_share_ft | north_N_NE_NW | 71 | 0.30 | 0.68 | n/n | 1.73 | 0.0878 | 0.1161 | 0.55 |
| crps_share_ft | array_in_view_gt5pct | 42 | 0.24 | 0.60 | n/n | 2.04 | 0.1231 | 0.1553 | 0.42 |
| crps_pure_scratch | all | 235 | 0.19 | 0.48 | n/n | 2.46 | 0.1058 | 0.1342 | 0.40 |
| crps_pure_scratch | north_N_NE_NW | 71 | 0.21 | 0.55 | n/n | 2.17 | 0.0923 | 0.1190 | 0.45 |
| crps_pure_scratch | array_in_view_gt5pct | 42 | 0.14 | 0.50 | n/n | 2.69 | 0.1242 | 0.1540 | 0.34 |
| crps_pure_ft | all | 235 | 0.29 | 0.68 | n/n | 1.66 | 0.0979 | 0.1338 | 0.58 |
| crps_pure_ft | north_N_NE_NW | 71 | 0.30 | 0.73 | n/n | 1.54 | 0.0871 | 0.1205 | 0.63 |
| crps_pure_ft | array_in_view_gt5pct | 42 | 0.29 | 0.71 | n/n | 1.84 | 0.1178 | 0.1565 | 0.50 |

## Peak and centroid (z sd, cover90)

| model | group | peak_x z sd | peak_x cover90 | centroid z sd | centroid cover90 |
|---|---|---|---|---|---|
| fm_seed1 | all | 1.14 | 0.93 | 0.83 | 0.94 |
| fm_seed1 | north_N_NE_NW | 0.99 | 0.96 | 0.76 | 0.94 |
| fm_seed1 | array_in_view_gt5pct | 0.98 | 0.95 | 0.77 | 0.95 |
| crps_share_ft | all | 2.24 | 0.85 | 0.94 | 0.93 |
| crps_share_ft | north_N_NE_NW | 2.09 | 0.92 | 0.88 | 0.93 |
| crps_share_ft | array_in_view_gt5pct | 1.83 | 0.90 | 0.84 | 0.93 |
| crps_pure_scratch | all | 1.53 | 0.95 | 0.94 | 0.89 |
| crps_pure_scratch | north_N_NE_NW | 0.89 | 0.99 | 0.93 | 0.89 |
| crps_pure_scratch | array_in_view_gt5pct | 0.92 | 0.95 | 0.95 | 0.86 |
| crps_pure_ft | all | 0.94 | 0.97 | 0.94 | 0.90 |
| crps_pure_ft | north_N_NE_NW | 0.78 | 0.99 | 0.88 | 0.89 |
| crps_pure_ft | array_in_view_gt5pct | 0.80 | 0.95 | 0.86 | 0.88 |

## Field CRPS (asinh space, cone cells, median over records) and the mean's metrics

Baseline composite range over the five final seeds: [0.474, 0.567] (rule: a mean has not regressed if inside it and val_mse_ref <= 1.20e-4).

| model | group | field CRPS | composite vs Kljun | vs baseline composite (p) | array share [pp] | centroid [m] | overlap80 (1-J) | integral | peak_x [m] | rel L2 |
|---|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | all | 0.00388 | 0.489 | - | 0.236 | 49.0 | 0.387 | 0.104 | 0 | 0.337 |
| fm_seed1 | north_N_NE_NW | 0.00402 | 0.539 | - | 1.168 | 47.8 | 0.379 | 0.086 | 0 | 0.294 |
| fm_seed1 | array_in_view_gt5pct | 0.00403 | 0.744 | - | 3.188 | 48.0 | 0.360 | 0.102 | 0 | 0.315 |
| crps_share_ft | all | 0.00438 | 0.486 | 0.995 (p 6.3e-13) | 0.219 | 50.7 | 0.392 | 0.104 | 0 | 0.336 |
| crps_share_ft | north_N_NE_NW | 0.00438 | 0.553 | 1.021 (p 0.34) | 1.215 | 49.1 | 0.384 | 0.089 | 0 | 0.302 |
| crps_share_ft | array_in_view_gt5pct | 0.00459 | 0.745 | 1.001 (p 0.41) | 3.262 | 48.0 | 0.365 | 0.098 | 0 | 0.312 |
| crps_pure_scratch | all | 0.00392 | 0.477 | 0.981 (p 5.9e-05) | 0.213 | 47.3 | 0.389 | 0.108 | 0 | 0.342 |
| crps_pure_scratch | north_N_NE_NW | 0.00404 | 0.582 | 1.063 (p 0.32) | 1.462 | 44.0 | 0.383 | 0.101 | 0 | 0.324 |
| crps_pure_scratch | array_in_view_gt5pct | 0.00421 | 0.762 | 1.023 (p 0.033) | 3.251 | 46.2 | 0.380 | 0.110 | 0 | 0.339 |
| crps_pure_ft | all | 0.00388 | 0.483 | 0.991 (p 0.017) | 0.224 | 48.4 | 0.383 | 0.107 | 0 | 0.351 |
| crps_pure_ft | north_N_NE_NW | 0.00415 | 0.564 | 1.037 (p 0.2) | 1.191 | 48.2 | 0.393 | 0.097 | 0 | 0.307 |
| crps_pure_ft | array_in_view_gt5pct | 0.00429 | 0.805 | 1.081 (p 0.67) | 3.750 | 52.4 | 0.384 | 0.110 | 0 | 0.351 |

## Verdicts

- **fm_seed1**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: no
- **crps_share_ft**: array-share coverage in band — all 50✗/90✗, north_N_NE_NW 50✗/90✗, array_in_view_gt5pct 50✗/90✗; mean regressed: no
- **crps_pure_scratch**: array-share coverage in band — all 50✗/90✗, north_N_NE_NW 50✗/90✗, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **crps_pure_ft**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
