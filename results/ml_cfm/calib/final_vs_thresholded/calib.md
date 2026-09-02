# CFM calibration `final_vs_thresholded`: 4 model variants, S = 64 samples each, val (235 records)

Bands: +-2 binomial sd around nominal — all (n 235): cover50 [0.43, 0.57], cover90 [0.86, 0.94]; north_N_NE_NW (n 71): cover50 [0.38, 0.62], cover90 [0.83, 0.97]; array_in_view_gt5pct (n 42): cover50 [0.35, 0.65], cover90 [0.81, 0.99]

| model | loss | init | sigma train / sample | steps | target | best epoch | val_mse_ref | val CRPS | ms/record/sample |
|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | fm | scratch | 0.1 / 0.1 | 16 | none | 90 | 1.172e-04 | nan | 12.4 |
| fm_seed2 | fm | scratch | 0.1 / 0.1 | 16 | none | 90 | 1.170e-04 | nan | 22.3 |
| thresh_seed0 | fm | scratch | 0.1 / 0.1 | 16 | sa99 | 105 | 1.327e-04 | 0.0042 | 16.0 |
| thresh_seed1 | fm | scratch | 0.1 / 0.1 | 16 | sa99 | 95 | 1.326e-04 | 0.0042 | 15.9 |

## Calibration of the array share (LES as one more draw among S samples)

| model | group | n | S | cover50 | cover90 | in band | z sd | PIT KS p | CRPS [pp] | MAE of mean [pp] | spread/skill | sample sd [pp] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | all | 235 | 64 | 0.47 | 0.88 | y/y | 1.17 | 4.4e-06 | 0.664 | 0.941 | 0.84 | 0.35 |
| fm_seed1 | north_N_NE_NW | 71 | 64 | 0.44 | 0.85 | y/y | 1.23 | 0.11 | 1.578 | 2.250 | 0.86 | 2.10 |
| fm_seed1 | array_in_view_gt5pct | 42 | 64 | 0.31 | 0.74 | n/n | 1.52 | 0.058 | 2.686 | 3.766 | 0.83 | 3.90 |
| fm_seed2 | all | 235 | 64 | 0.49 | 0.88 | y/y | 1.09 | 5.7e-05 | 0.660 | 0.943 | 0.87 | 0.39 |
| fm_seed2 | north_N_NE_NW | 71 | 64 | 0.48 | 0.86 | y/y | 1.14 | 0.35 | 1.580 | 2.255 | 0.89 | 2.24 |
| fm_seed2 | array_in_view_gt5pct | 42 | 64 | 0.31 | 0.76 | n/n | 1.43 | 0.16 | 2.660 | 3.752 | 0.86 | 3.99 |
| thresh_seed0 | all | 235 | 64 | 0.39 | 0.82 | n/n | 1.14 | 6.1e-11 | 0.662 | 0.950 | 0.83 | 0.36 |
| thresh_seed0 | north_N_NE_NW | 71 | 64 | 0.41 | 0.87 | y/y | 1.20 | 0.0071 | 1.576 | 2.245 | 0.85 | 2.11 |
| thresh_seed0 | array_in_view_gt5pct | 42 | 64 | 0.29 | 0.79 | n/n | 1.46 | 0.44 | 2.645 | 3.713 | 0.82 | 3.78 |
| thresh_seed1 | all | 235 | 64 | 0.47 | 0.86 | y/y | 1.23 | 0.0013 | 0.674 | 0.943 | 0.79 | 0.35 |
| thresh_seed1 | north_N_NE_NW | 71 | 64 | 0.46 | 0.83 | y/y | 1.26 | 0.3 | 1.614 | 2.271 | 0.81 | 1.96 |
| thresh_seed1 | array_in_view_gt5pct | 42 | 64 | 0.33 | 0.71 | n/n | 1.55 | 0.0093 | 2.751 | 3.800 | 0.79 | 3.70 |

## Calibration of the integral

| model | group | n | cover50 | cover90 | in band | z sd | CRPS | MAE of mean | spread/skill |
|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | all | 235 | 0.36 | 0.77 | n/n | 1.39 | 0.0978 | 0.1332 | 0.67 |
| fm_seed1 | north_N_NE_NW | 71 | 0.42 | 0.77 | y/n | 1.28 | 0.0859 | 0.1164 | 0.72 |
| fm_seed1 | array_in_view_gt5pct | 42 | 0.40 | 0.69 | y/n | 1.49 | 0.1211 | 0.1575 | 0.57 |
| fm_seed2 | all | 235 | 0.40 | 0.77 | n/n | 1.37 | 0.0966 | 0.1310 | 0.68 |
| fm_seed2 | north_N_NE_NW | 71 | 0.48 | 0.77 | y/n | 1.27 | 0.0879 | 0.1171 | 0.71 |
| fm_seed2 | array_in_view_gt5pct | 42 | 0.45 | 0.71 | y/n | 1.47 | 0.1245 | 0.1606 | 0.55 |
| thresh_seed0 | all | 235 | 0.35 | 0.77 | n/n | 1.39 | 0.0951 | 0.1327 | 0.70 |
| thresh_seed0 | north_N_NE_NW | 71 | 0.35 | 0.75 | n/n | 1.31 | 0.0851 | 0.1174 | 0.71 |
| thresh_seed0 | array_in_view_gt5pct | 42 | 0.38 | 0.67 | y/n | 1.49 | 0.1156 | 0.1527 | 0.58 |
| thresh_seed1 | all | 235 | 0.38 | 0.76 | n/n | 1.41 | 0.0968 | 0.1321 | 0.67 |
| thresh_seed1 | north_N_NE_NW | 71 | 0.42 | 0.76 | y/n | 1.34 | 0.0867 | 0.1176 | 0.70 |
| thresh_seed1 | array_in_view_gt5pct | 42 | 0.38 | 0.69 | y/n | 1.52 | 0.1184 | 0.1550 | 0.56 |

## Peak and centroid (z sd, cover90)

| model | group | peak_x z sd | peak_x cover90 | centroid z sd | centroid cover90 |
|---|---|---|---|---|---|
| fm_seed1 | all | 1.14 | 0.93 | 0.78 | 0.94 |
| fm_seed1 | north_N_NE_NW | 0.99 | 0.96 | 0.71 | 0.96 |
| fm_seed1 | array_in_view_gt5pct | 0.98 | 0.95 | 0.67 | 1.00 |
| fm_seed2 | all | 1.12 | 0.95 | 0.71 | 0.91 |
| fm_seed2 | north_N_NE_NW | 1.01 | 0.97 | 0.65 | 0.96 |
| fm_seed2 | array_in_view_gt5pct | 0.92 | 0.95 | 0.61 | 0.98 |
| thresh_seed0 | all | 1.15 | 0.95 | 0.83 | 0.90 |
| thresh_seed0 | north_N_NE_NW | 1.30 | 0.96 | 0.77 | 0.92 |
| thresh_seed0 | array_in_view_gt5pct | 0.93 | 0.95 | 0.74 | 0.93 |
| thresh_seed1 | all | 1.18 | 0.95 | 0.83 | 0.93 |
| thresh_seed1 | north_N_NE_NW | 1.04 | 0.97 | 0.74 | 0.94 |
| thresh_seed1 | array_in_view_gt5pct | 0.91 | 0.95 | 0.73 | 0.98 |

## Field CRPS (asinh space, cone cells, median over records) and the mean's metrics

Baseline composite range over the five final seeds: [0.474, 0.567] (rule: a mean has not regressed if inside it and val_mse_ref <= 1.20e-4).

| model | group | field CRPS | composite vs Kljun | vs baseline composite (p) | array share [pp] | centroid [m] | overlap80 (1-J) | integral | peak_x [m] | rel L2 |
|---|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | all | 0.00373 | 0.454 | - | 0.228 | 44.6 | 0.386 | 0.101 | 0 | 0.336 |
| fm_seed1 | north_N_NE_NW | 0.00391 | 0.503 | - | 1.228 | 47.5 | 0.371 | 0.078 | 0 | 0.294 |
| fm_seed1 | array_in_view_gt5pct | 0.00394 | 0.754 | - | 3.444 | 48.2 | 0.355 | 0.101 | 0 | 0.314 |
| fm_seed2 | all | 0.00372 | 0.460 | 1.010 (p 2.4e-05) | 0.230 | 48.7 | 0.381 | 0.097 | 0 | 0.337 |
| fm_seed2 | north_N_NE_NW | 0.00387 | 0.489 | 0.978 (p 0.025) | 1.149 | 49.4 | 0.367 | 0.073 | 0 | 0.296 |
| fm_seed2 | array_in_view_gt5pct | 0.00405 | 0.738 | 0.978 (p 0.025) | 3.211 | 51.6 | 0.355 | 0.091 | 0 | 0.322 |
| thresh_seed0 | all | 0.00380 | 0.469 | 1.025 (p 7.3e-05) | 0.249 | 44.8 | 0.384 | 0.104 | 0 | 0.338 |
| thresh_seed0 | north_N_NE_NW | 0.00389 | 0.517 | 1.022 (p 0.39) | 1.174 | 45.4 | 0.374 | 0.095 | 0 | 0.286 |
| thresh_seed0 | array_in_view_gt5pct | 0.00394 | 0.792 | 1.050 (p 0.16) | 3.677 | 52.0 | 0.359 | 0.111 | 0 | 0.312 |
| thresh_seed1 | all | 0.00372 | 0.460 | 1.010 (p 2.3e-06) | 0.227 | 46.2 | 0.389 | 0.102 | 0 | 0.337 |
| thresh_seed1 | north_N_NE_NW | 0.00389 | 0.506 | 1.005 (p 0.12) | 1.155 | 44.7 | 0.373 | 0.090 | 0 | 0.297 |
| thresh_seed1 | array_in_view_gt5pct | 0.00395 | 0.738 | 0.978 (p 0.12) | 3.224 | 47.8 | 0.360 | 0.096 | 0 | 0.312 |

## Verdicts

- **fm_seed1**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **fm_seed2**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **thresh_seed0**: array-share coverage in band — all 50✗/90✗, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **thresh_seed1**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
