# CFM calibration `final`: 15 model variants, S = 64 samples each, val (235 records)

Bands: +-2 binomial sd around nominal — all (n 235): cover50 [0.43, 0.57], cover90 [0.86, 0.94]; north_N_NE_NW (n 71): cover50 [0.38, 0.62], cover90 [0.83, 0.97]; array_in_view_gt5pct (n 42): cover50 [0.35, 0.65], cover90 [0.81, 0.99]

| model | loss | init | sigma train / sample | steps | target | best epoch | val_mse_ref | val CRPS | ms/record/sample |
|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | fm | scratch | 0.1 / 0.1 | 16 | none | 90 | 1.172e-04 | nan | 16.2 |
| fm_seed1+temp_global_in_sample | fm | scratch | 0.1 / 0.1 | 16 | none | - | nan | nan | 16.2 |
| fm_seed1+temp_grouped_in_sample | fm | scratch | 0.1 / 0.1 | 16 | none | - | nan | nan | 16.2 |
| fm_seed1+temp_global_crossfit | fm | scratch | 0.1 / 0.1 | 16 | none | - | nan | nan | 16.2 |
| fm_seed1+temp_grouped_crossfit | fm | scratch | 0.1 / 0.1 | 16 | none | - | nan | nan | 16.2 |
| fm_seed1_e2 | fm | scratch | 0.1 / 0.1 | 2 | none | 90 | 1.172e-04 | nan | 1.0 |
| fm_seed1_sig0.2 | fm | scratch | 0.1 / 0.2 | 16 | none | 90 | 1.172e-04 | nan | 8.2 |
| fm_seed1_sig0.3 | fm | scratch | 0.1 / 0.3 | 16 | none | 90 | 1.172e-04 | nan | 8.2 |
| fm_seed1_sig0.5 | fm | scratch | 0.1 / 0.5 | 16 | none | 90 | 1.172e-04 | nan | 8.2 |
| fm_sig0.3_trained | fm | scratch | 0.3 / 0.3 | 16 | none | 110 | 1.248e-04 | nan | 8.3 |
| crps_pure_ft | crps | seed1 | 0.1 / 0.1 | 2 | none | 20 | 1.273e-04 | 0.0043 | 1.0 |
| crps_blend_ft | fm+crps | seed1 | 0.1 / 0.1 | 2 | none | 20 | 1.248e-04 | 0.0043 | 1.0 |
| crps_pure_ft_S4 | crps | seed1 | 0.1 / 0.1 | 4 | none | 10 | 1.270e-04 | 0.0042 | 2.0 |
| thresh_seed0 | fm | scratch | 0.1 / 0.1 | 16 | sa99 | 105 | 1.327e-04 | 0.0042 | 8.2 |
| thresh_seed1 | fm | scratch | 0.1 / 0.1 | 16 | sa99 | 95 | 1.326e-04 | 0.0042 | 8.2 |

## Calibration of the array share (LES as one more draw among S samples)

| model | group | n | S | cover50 | cover90 | in band | z sd | PIT KS p | CRPS [pp] | MAE of mean [pp] | spread/skill | sample sd [pp] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | all | 235 | 64 | 0.47 | 0.88 | y/y | 1.19 | 0.00077 | 0.674 | 0.951 | 0.82 | 0.35 |
| fm_seed1 | north_N_NE_NW | 71 | 64 | 0.44 | 0.85 | y/y | 1.26 | 0.48 | 1.606 | 2.285 | 0.84 | 2.09 |
| fm_seed1 | array_in_view_gt5pct | 42 | 64 | 0.31 | 0.74 | n/n | 1.55 | 0.057 | 2.747 | 3.846 | 0.81 | 3.88 |
| fm_seed1+temp_global_in_sample | all | 235 | 64 | 0.56 | 0.92 | y/y | 1.00 | 0.0025 | 0.667 | 0.950 | 0.98 | 0.41 |
| fm_seed1+temp_global_in_sample | north_N_NE_NW | 71 | 64 | 0.55 | 0.93 | y/y | 1.05 | 0.5 | 1.591 | 2.285 | 1.01 | 2.49 |
| fm_seed1+temp_global_in_sample | array_in_view_gt5pct | 42 | 64 | 0.43 | 0.81 | y/y | 1.30 | 0.25 | 2.707 | 3.847 | 0.97 | 4.61 |
| fm_seed1+temp_grouped_in_sample | all | 235 | 64 | 0.56 | 0.92 | y/y | 1.00 | 0.0025 | 0.670 | 0.950 | 1.13 | 0.39 |
| fm_seed1+temp_grouped_in_sample | north_N_NE_NW | 71 | 64 | 0.56 | 0.94 | y/y | 0.96 | 0.27 | 1.601 | 2.285 | 1.17 | 2.90 |
| fm_seed1+temp_grouped_in_sample | array_in_view_gt5pct | 42 | 64 | 0.45 | 0.83 | y/y | 1.18 | 0.4 | 2.727 | 3.847 | 1.13 | 5.37 |
| fm_seed1+temp_global_crossfit | all | 235 | 64 | 0.56 | 0.92 | y/y | 1.01 | 0.0025 | 0.668 | 0.951 | 0.98 | 0.40 |
| fm_seed1+temp_global_crossfit | north_N_NE_NW | 71 | 64 | 0.54 | 0.93 | y/y | 1.07 | 0.5 | 1.596 | 2.287 | 1.00 | 2.57 |
| fm_seed1+temp_global_crossfit | array_in_view_gt5pct | 42 | 64 | 0.40 | 0.81 | y/y | 1.31 | 0.28 | 2.713 | 3.851 | 0.97 | 4.45 |
| fm_seed1+temp_grouped_crossfit | all | 235 | 64 | 0.54 | 0.91 | y/y | 1.06 | 0.0025 | 0.683 | 0.955 | 1.07 | 0.40 |
| fm_seed1+temp_grouped_crossfit | north_N_NE_NW | 71 | 64 | 0.52 | 0.93 | y/y | 1.13 | 0.42 | 1.642 | 2.302 | 1.11 | 2.69 |
| fm_seed1+temp_grouped_crossfit | array_in_view_gt5pct | 42 | 64 | 0.40 | 0.81 | y/y | 1.39 | 0.2 | 2.798 | 3.876 | 1.08 | 4.30 |
| fm_seed1_e2 | all | 235 | 64 | 0.27 | 0.62 | n/n | 1.94 | 3.4e-16 | 0.730 | 0.959 | 0.52 | 0.22 |
| fm_seed1_e2 | north_N_NE_NW | 71 | 64 | 0.28 | 0.55 | n/n | 2.09 | 0.00016 | 1.716 | 2.268 | 0.54 | 1.21 |
| fm_seed1_e2 | array_in_view_gt5pct | 42 | 64 | 0.21 | 0.45 | n/n | 2.56 | 0.00062 | 2.927 | 3.806 | 0.52 | 2.19 |
| fm_seed1_sig0.2 | all | 235 | 64 | 0.71 | 0.91 | n/y | 0.90 | 1.3e-07 | 1.221 | 1.653 | 0.63 | 0.78 |
| fm_seed1_sig0.2 | north_N_NE_NW | 71 | 64 | 0.59 | 0.85 | y/y | 1.12 | 0.0003 | 3.328 | 4.475 | 0.62 | 3.12 |
| fm_seed1_sig0.2 | array_in_view_gt5pct | 42 | 64 | 0.33 | 0.64 | n/n | 1.11 | 1.7e-11 | 5.690 | 7.579 | 0.60 | 5.79 |
| fm_seed1_sig0.3 | all | 235 | 64 | 0.64 | 0.84 | n/n | 1.72 | 5.5e-12 | 2.394 | 2.831 | 0.28 | 0.86 |
| fm_seed1_sig0.3 | north_N_NE_NW | 71 | 64 | 0.38 | 0.61 | n/n | 2.37 | 3.8e-10 | 6.980 | 8.131 | 0.27 | 3.13 |
| fm_seed1_sig0.3 | array_in_view_gt5pct | 42 | 64 | 0.02 | 0.24 | n/n | 2.09 | 6.5e-29 | 11.965 | 13.885 | 0.25 | 4.07 |
| fm_seed1_sig0.5 | all | 235 | 64 | 0.61 | 0.80 | n/n | 3.07 | 3.9e-15 | 3.176 | 3.559 | 0.18 | 0.96 |
| fm_seed1_sig0.5 | north_N_NE_NW | 71 | 64 | 0.37 | 0.51 | n/n | 4.43 | 2.1e-13 | 9.330 | 10.252 | 0.16 | 2.29 |
| fm_seed1_sig0.5 | array_in_view_gt5pct | 42 | 64 | 0.00 | 0.07 | n/n | 3.95 | 2.4e-39 | 15.944 | 17.399 | 0.14 | 2.74 |
| fm_sig0.3_trained | all | 235 | 64 | 0.53 | 0.89 | y/y | 0.99 | 0.013 | 0.656 | 0.930 | 0.96 | 0.39 |
| fm_sig0.3_trained | north_N_NE_NW | 71 | 64 | 0.46 | 0.86 | y/y | 1.12 | 0.3 | 1.586 | 2.236 | 0.99 | 2.21 |
| fm_sig0.3_trained | array_in_view_gt5pct | 42 | 64 | 0.33 | 0.76 | n/n | 1.35 | 0.25 | 2.655 | 3.714 | 0.96 | 4.23 |
| crps_pure_ft | all | 235 | 64 | 0.51 | 0.89 | y/y | 1.07 | 0.083 | 0.674 | 0.961 | 0.79 | 0.39 |
| crps_pure_ft | north_N_NE_NW | 71 | 64 | 0.46 | 0.83 | y/y | 1.26 | 0.48 | 1.646 | 2.337 | 0.79 | 2.02 |
| crps_pure_ft | array_in_view_gt5pct | 42 | 64 | 0.31 | 0.76 | n/n | 1.48 | 0.043 | 2.749 | 3.877 | 0.77 | 3.72 |
| crps_blend_ft | all | 235 | 64 | 0.52 | 0.89 | y/y | 1.11 | 0.5 | 0.672 | 0.952 | 0.78 | 0.36 |
| crps_blend_ft | north_N_NE_NW | 71 | 64 | 0.46 | 0.83 | y/y | 1.29 | 0.48 | 1.636 | 2.314 | 0.79 | 2.04 |
| crps_blend_ft | array_in_view_gt5pct | 42 | 64 | 0.29 | 0.71 | n/n | 1.55 | 0.06 | 2.746 | 3.855 | 0.77 | 3.77 |
| crps_pure_ft_S4 | all | 235 | 64 | 0.49 | 0.86 | y/n | 1.27 | 0.46 | 0.684 | 0.952 | 0.69 | 0.38 |
| crps_pure_ft_S4 | north_N_NE_NW | 71 | 64 | 0.42 | 0.77 | y/n | 1.43 | 0.12 | 1.668 | 2.312 | 0.69 | 1.77 |
| crps_pure_ft_S4 | array_in_view_gt5pct | 42 | 64 | 0.26 | 0.69 | n/n | 1.70 | 0.0046 | 2.796 | 3.844 | 0.67 | 3.21 |
| thresh_seed0 | all | 235 | 64 | 0.41 | 0.84 | n/n | 1.17 | 8.6e-09 | 0.667 | 0.952 | 0.82 | 0.36 |
| thresh_seed0 | north_N_NE_NW | 71 | 64 | 0.44 | 0.87 | y/y | 1.23 | 0.032 | 1.591 | 2.257 | 0.83 | 2.10 |
| thresh_seed0 | array_in_view_gt5pct | 42 | 64 | 0.33 | 0.79 | n/n | 1.49 | 0.16 | 2.683 | 3.755 | 0.81 | 3.77 |
| thresh_seed1 | all | 235 | 64 | 0.49 | 0.86 | y/n | 1.26 | 0.032 | 0.685 | 0.954 | 0.77 | 0.35 |
| thresh_seed1 | north_N_NE_NW | 71 | 64 | 0.45 | 0.82 | y/n | 1.29 | 0.57 | 1.646 | 2.308 | 0.79 | 1.95 |
| thresh_seed1 | array_in_view_gt5pct | 42 | 64 | 0.33 | 0.69 | n/n | 1.59 | 0.0065 | 2.817 | 3.881 | 0.77 | 3.68 |

## Calibration of the integral

| model | group | n | cover50 | cover90 | in band | z sd | CRPS | MAE of mean | spread/skill |
|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | all | 235 | 0.36 | 0.76 | n/n | 1.41 | 0.0977 | 0.1343 | 0.67 |
| fm_seed1 | north_N_NE_NW | 71 | 0.38 | 0.76 | n/n | 1.29 | 0.0852 | 0.1166 | 0.73 |
| fm_seed1 | array_in_view_gt5pct | 42 | 0.38 | 0.69 | y/n | 1.49 | 0.1190 | 0.1562 | 0.58 |
| fm_seed1+temp_global_in_sample | all | 235 | 0.43 | 0.84 | n/n | 1.18 | 0.0959 | 0.1343 | 0.80 |
| fm_seed1+temp_global_in_sample | north_N_NE_NW | 71 | 0.49 | 0.85 | y/y | 1.08 | 0.0842 | 0.1166 | 0.88 |
| fm_seed1+temp_global_in_sample | array_in_view_gt5pct | 42 | 0.45 | 0.74 | y/n | 1.25 | 0.1164 | 0.1560 | 0.69 |
| fm_seed1+temp_grouped_in_sample | all | 235 | 0.41 | 0.83 | n/n | 1.20 | 0.0961 | 0.1342 | 0.80 |
| fm_seed1+temp_grouped_in_sample | north_N_NE_NW | 71 | 0.49 | 0.87 | y/y | 1.00 | 0.0836 | 0.1164 | 0.96 |
| fm_seed1+temp_grouped_in_sample | array_in_view_gt5pct | 42 | 0.50 | 0.81 | y/y | 1.13 | 0.1156 | 0.1558 | 0.79 |
| fm_seed1+temp_global_crossfit | all | 235 | 0.44 | 0.82 | y/n | 1.19 | 0.0961 | 0.1343 | 0.80 |
| fm_seed1+temp_global_crossfit | north_N_NE_NW | 71 | 0.51 | 0.85 | y/y | 1.10 | 0.0845 | 0.1166 | 0.88 |
| fm_seed1+temp_global_crossfit | array_in_view_gt5pct | 42 | 0.48 | 0.74 | y/n | 1.28 | 0.1170 | 0.1560 | 0.69 |
| fm_seed1+temp_grouped_crossfit | all | 235 | 0.41 | 0.82 | n/n | 1.25 | 0.0968 | 0.1343 | 0.79 |
| fm_seed1+temp_grouped_crossfit | north_N_NE_NW | 71 | 0.49 | 0.83 | y/y | 1.17 | 0.0862 | 0.1166 | 0.91 |
| fm_seed1+temp_grouped_crossfit | array_in_view_gt5pct | 42 | 0.48 | 0.71 | y/n | 1.36 | 0.1200 | 0.1560 | 0.74 |
| fm_seed1_e2 | all | 235 | 0.26 | 0.60 | n/n | 1.99 | 0.1019 | 0.1338 | 0.48 |
| fm_seed1_e2 | north_N_NE_NW | 71 | 0.32 | 0.68 | n/n | 1.76 | 0.0886 | 0.1167 | 0.53 |
| fm_seed1_e2 | array_in_view_gt5pct | 42 | 0.26 | 0.62 | n/n | 2.07 | 0.1242 | 0.1560 | 0.42 |
| fm_seed1_sig0.2 | all | 235 | 0.10 | 0.43 | n/n | 0.66 | 0.3428 | 0.4943 | 0.60 |
| fm_seed1_sig0.2 | north_N_NE_NW | 71 | 0.08 | 0.45 | n/n | 0.59 | 0.3506 | 0.5068 | 0.59 |
| fm_seed1_sig0.2 | array_in_view_gt5pct | 42 | 0.14 | 0.43 | n/n | 0.69 | 0.3634 | 0.5176 | 0.61 |
| fm_seed1_sig0.3 | all | 235 | 0.00 | 0.00 | n/n | 0.66 | 1.7255 | 2.0875 | 0.30 |
| fm_seed1_sig0.3 | north_N_NE_NW | 71 | 0.00 | 0.00 | n/n | 0.61 | 1.7765 | 2.1433 | 0.30 |
| fm_seed1_sig0.3 | array_in_view_gt5pct | 42 | 0.00 | 0.00 | n/n | 0.66 | 1.9486 | 2.3302 | 0.29 |
| fm_seed1_sig0.5 | all | 235 | 0.00 | 0.00 | n/n | 0.73 | 5.1168 | 5.8024 | 0.21 |
| fm_seed1_sig0.5 | north_N_NE_NW | 71 | 0.00 | 0.00 | n/n | 0.70 | 5.1885 | 5.8800 | 0.21 |
| fm_seed1_sig0.5 | array_in_view_gt5pct | 42 | 0.00 | 0.00 | n/n | 0.71 | 5.6391 | 6.3476 | 0.20 |
| fm_sig0.3_trained | all | 235 | 0.48 | 0.81 | y/n | 1.17 | 0.0975 | 0.1334 | 0.79 |
| fm_sig0.3_trained | north_N_NE_NW | 71 | 0.49 | 0.79 | y/n | 1.12 | 0.0869 | 0.1184 | 0.84 |
| fm_sig0.3_trained | array_in_view_gt5pct | 42 | 0.43 | 0.76 | y/n | 1.31 | 0.1210 | 0.1625 | 0.65 |
| crps_pure_ft | all | 235 | 0.29 | 0.68 | n/n | 1.66 | 0.0979 | 0.1338 | 0.58 |
| crps_pure_ft | north_N_NE_NW | 71 | 0.30 | 0.73 | n/n | 1.54 | 0.0871 | 0.1205 | 0.63 |
| crps_pure_ft | array_in_view_gt5pct | 42 | 0.29 | 0.71 | n/n | 1.84 | 0.1178 | 0.1565 | 0.50 |
| crps_blend_ft | all | 235 | 0.34 | 0.74 | n/n | 1.46 | 0.0963 | 0.1335 | 0.66 |
| crps_blend_ft | north_N_NE_NW | 71 | 0.37 | 0.75 | n/n | 1.40 | 0.0861 | 0.1190 | 0.69 |
| crps_blend_ft | array_in_view_gt5pct | 42 | 0.36 | 0.71 | y/n | 1.64 | 0.1180 | 0.1575 | 0.55 |
| crps_pure_ft_S4 | all | 235 | 0.25 | 0.62 | n/n | 1.97 | 0.1011 | 0.1342 | 0.50 |
| crps_pure_ft_S4 | north_N_NE_NW | 71 | 0.25 | 0.69 | n/n | 1.83 | 0.0892 | 0.1197 | 0.54 |
| crps_pure_ft_S4 | array_in_view_gt5pct | 42 | 0.17 | 0.67 | n/n | 2.20 | 0.1194 | 0.1541 | 0.42 |
| thresh_seed0 | all | 235 | 0.31 | 0.75 | n/n | 1.41 | 0.0964 | 0.1351 | 0.69 |
| thresh_seed0 | north_N_NE_NW | 71 | 0.31 | 0.76 | n/n | 1.33 | 0.0854 | 0.1175 | 0.71 |
| thresh_seed0 | array_in_view_gt5pct | 42 | 0.33 | 0.67 | n/n | 1.49 | 0.1141 | 0.1515 | 0.59 |
| thresh_seed1 | all | 235 | 0.35 | 0.75 | n/n | 1.43 | 0.0972 | 0.1344 | 0.67 |
| thresh_seed1 | north_N_NE_NW | 71 | 0.38 | 0.75 | n/n | 1.36 | 0.0867 | 0.1203 | 0.70 |
| thresh_seed1 | array_in_view_gt5pct | 42 | 0.36 | 0.69 | y/n | 1.53 | 0.1167 | 0.1549 | 0.57 |

## Peak and centroid (z sd, cover90)

| model | group | peak_x z sd | peak_x cover90 | centroid z sd | centroid cover90 |
|---|---|---|---|---|---|
| fm_seed1 | all | 1.14 | 0.93 | 0.83 | 0.94 |
| fm_seed1 | north_N_NE_NW | 0.99 | 0.96 | 0.76 | 0.94 |
| fm_seed1 | array_in_view_gt5pct | 0.98 | 0.95 | 0.77 | 0.95 |
| fm_seed1+temp_global_in_sample | all | 0.96 | 0.96 | 0.70 | 0.97 |
| fm_seed1+temp_global_in_sample | north_N_NE_NW | 0.84 | 0.99 | 0.64 | 0.96 |
| fm_seed1+temp_global_in_sample | array_in_view_gt5pct | 0.82 | 0.98 | 0.65 | 0.95 |
| fm_seed1+temp_grouped_in_sample | all | 0.98 | 0.96 | 0.71 | 0.98 |
| fm_seed1+temp_grouped_in_sample | north_N_NE_NW | 0.84 | 0.99 | 0.60 | 0.99 |
| fm_seed1+temp_grouped_in_sample | array_in_view_gt5pct | 0.79 | 0.98 | 0.57 | 1.00 |
| fm_seed1+temp_global_crossfit | all | 0.96 | 0.96 | 0.70 | 0.98 |
| fm_seed1+temp_global_crossfit | north_N_NE_NW | 0.85 | 0.97 | 0.64 | 0.96 |
| fm_seed1+temp_global_crossfit | array_in_view_gt5pct | 0.83 | 0.95 | 0.65 | 0.95 |
| fm_seed1+temp_grouped_crossfit | all | 1.00 | 0.95 | 0.73 | 0.97 |
| fm_seed1+temp_grouped_crossfit | north_N_NE_NW | 0.93 | 0.97 | 0.65 | 0.96 |
| fm_seed1+temp_grouped_crossfit | array_in_view_gt5pct | 0.93 | 0.95 | 0.65 | 0.95 |
| fm_seed1_e2 | all | 2.25 | 0.86 | 0.96 | 0.91 |
| fm_seed1_e2 | north_N_NE_NW | 2.12 | 0.92 | 0.89 | 0.90 |
| fm_seed1_e2 | array_in_view_gt5pct | 1.85 | 0.90 | 0.86 | 0.90 |
| fm_seed1_sig0.2 | all | 0.41 | 1.00 | 0.55 | 0.67 |
| fm_seed1_sig0.2 | north_N_NE_NW | 0.25 | 1.00 | 0.53 | 0.58 |
| fm_seed1_sig0.2 | array_in_view_gt5pct | 0.28 | 1.00 | 0.56 | 0.50 |
| fm_seed1_sig0.3 | all | 0.30 | 1.00 | 0.82 | 0.03 |
| fm_seed1_sig0.3 | north_N_NE_NW | 0.16 | 1.00 | 0.92 | 0.01 |
| fm_seed1_sig0.3 | array_in_view_gt5pct | 0.16 | 1.00 | 0.95 | 0.00 |
| fm_seed1_sig0.5 | all | 0.32 | 0.61 | 1.10 | 0.00 |
| fm_seed1_sig0.5 | north_N_NE_NW | 0.25 | 0.46 | 1.29 | 0.00 |
| fm_seed1_sig0.5 | array_in_view_gt5pct | 0.25 | 0.31 | 1.30 | 0.00 |
| fm_sig0.3_trained | all | 1.11 | 0.95 | 0.66 | 0.95 |
| fm_sig0.3_trained | north_N_NE_NW | 0.96 | 0.97 | 0.58 | 0.99 |
| fm_sig0.3_trained | array_in_view_gt5pct | 0.94 | 0.95 | 0.57 | 1.00 |
| crps_pure_ft | all | 0.94 | 0.97 | 0.94 | 0.90 |
| crps_pure_ft | north_N_NE_NW | 0.78 | 0.99 | 0.88 | 0.89 |
| crps_pure_ft | array_in_view_gt5pct | 0.80 | 0.95 | 0.86 | 0.88 |
| crps_blend_ft | all | 1.20 | 0.94 | 0.88 | 0.94 |
| crps_blend_ft | north_N_NE_NW | 1.12 | 0.96 | 0.84 | 0.94 |
| crps_blend_ft | array_in_view_gt5pct | 1.00 | 0.95 | 0.80 | 0.93 |
| crps_pure_ft_S4 | all | 0.99 | 0.96 | 1.15 | 0.83 |
| crps_pure_ft_S4 | north_N_NE_NW | 0.80 | 0.99 | 1.12 | 0.82 |
| crps_pure_ft_S4 | array_in_view_gt5pct | 0.83 | 0.95 | 1.15 | 0.79 |
| thresh_seed0 | all | 1.15 | 0.95 | 0.89 | 0.92 |
| thresh_seed0 | north_N_NE_NW | 1.30 | 0.96 | 0.85 | 0.90 |
| thresh_seed0 | array_in_view_gt5pct | 0.93 | 0.95 | 0.85 | 0.90 |
| thresh_seed1 | all | 1.16 | 0.95 | 0.91 | 0.92 |
| thresh_seed1 | north_N_NE_NW | 1.04 | 0.97 | 0.80 | 0.93 |
| thresh_seed1 | array_in_view_gt5pct | 0.91 | 0.95 | 0.83 | 0.90 |

## Field CRPS (asinh space, cone cells, median over records) and the mean's metrics

Baseline composite range over the five final seeds: [0.474, 0.567] (rule: a mean has not regressed if inside it and val_mse_ref <= 1.20e-4).

| model | group | field CRPS | composite vs Kljun | vs baseline composite (p) | array share [pp] | centroid [m] | overlap80 (1-J) | integral | peak_x [m] | rel L2 |
|---|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 | all | 0.00388 | 0.489 | - | 0.236 | 49.0 | 0.387 | 0.104 | 0 | 0.337 |
| fm_seed1 | north_N_NE_NW | 0.00402 | 0.539 | - | 1.168 | 47.8 | 0.379 | 0.086 | 0 | 0.294 |
| fm_seed1 | array_in_view_gt5pct | 0.00403 | 0.744 | - | 3.188 | 48.0 | 0.360 | 0.102 | 0 | 0.315 |
| fm_seed1+temp_global_in_sample | all | 0.00383 | 0.487 | - | 0.236 | 48.7 | 0.387 | 0.103 | 0 | 0.337 |
| fm_seed1+temp_global_in_sample | north_N_NE_NW | 0.00403 | 0.544 | - | 1.214 | 47.6 | 0.379 | 0.087 | 0 | 0.296 |
| fm_seed1+temp_global_in_sample | array_in_view_gt5pct | 0.00406 | 0.748 | - | 3.275 | 48.3 | 0.359 | 0.101 | 0 | 0.318 |
| fm_seed1+temp_grouped_in_sample | all | 0.00385 | 0.488 | - | 0.236 | 48.7 | 0.387 | 0.104 | 0 | 0.337 |
| fm_seed1+temp_grouped_in_sample | north_N_NE_NW | 0.00409 | 0.551 | - | 1.270 | 48.0 | 0.378 | 0.087 | 0 | 0.299 |
| fm_seed1+temp_grouped_in_sample | array_in_view_gt5pct | 0.00409 | 0.753 | - | 3.388 | 48.7 | 0.360 | 0.100 | 0 | 0.321 |
| fm_seed1+temp_global_crossfit | all | 0.00384 | 0.488 | - | 0.236 | 48.7 | 0.386 | 0.104 | 0 | 0.337 |
| fm_seed1+temp_global_crossfit | north_N_NE_NW | 0.00404 | 0.544 | - | 1.225 | 47.7 | 0.378 | 0.086 | 0 | 0.297 |
| fm_seed1+temp_global_crossfit | array_in_view_gt5pct | 0.00405 | 0.747 | - | 3.261 | 48.3 | 0.359 | 0.101 | 0 | 0.317 |
| fm_seed1+temp_grouped_crossfit | all | 0.00388 | 0.481 | - | 0.223 | 48.8 | 0.387 | 0.103 | 0 | 0.337 |
| fm_seed1+temp_grouped_crossfit | north_N_NE_NW | 0.00402 | 0.555 | - | 1.298 | 47.7 | 0.380 | 0.087 | 0 | 0.302 |
| fm_seed1+temp_grouped_crossfit | array_in_view_gt5pct | 0.00406 | 0.745 | - | 3.276 | 48.4 | 0.358 | 0.099 | 0 | 0.315 |
| fm_seed1_e2 | all | 0.00439 | 0.491 | 1.003 (p 1e-11) | 0.226 | 51.2 | 0.392 | 0.104 | 0 | 0.337 |
| fm_seed1_e2 | north_N_NE_NW | 0.00439 | 0.563 | 1.035 (p 0.66) | 1.243 | 50.2 | 0.378 | 0.092 | 0 | 0.302 |
| fm_seed1_e2 | array_in_view_gt5pct | 0.00459 | 0.756 | 1.016 (p 0.45) | 3.357 | 50.7 | 0.359 | 0.099 | 0 | 0.312 |
| fm_seed1_sig0.2 | all | 0.00711 | 1.256 | 2.128 (p 2.6e-40) | 0.304 | 183.7 | 0.752 | 0.481 | 0 | 0.397 |
| fm_seed1_sig0.2 | north_N_NE_NW | 0.00781 | 1.613 | 2.405 (p 2.4e-13) | 1.665 | 222.1 | 0.780 | 0.510 | 0 | 0.339 |
| fm_seed1_sig0.2 | array_in_view_gt5pct | 0.00784 | 1.866 | 2.508 (p 4.5e-13) | 5.549 | 233.6 | 0.818 | 0.524 | 0 | 0.348 |
| fm_seed1_sig0.3 | all | 0.01994 | 2.349 | 3.511 (p 2.6e-40) | 0.352 | 390.8 | 0.874 | 2.057 | 0 | 0.645 |
| fm_seed1_sig0.3 | north_N_NE_NW | 0.02079 | 3.111 | 4.066 (p 2.4e-13) | 2.507 | 439.0 | 0.885 | 2.085 | 0 | 0.560 |
| fm_seed1_sig0.3 | array_in_view_gt5pct | 0.02164 | 3.385 | 4.548 (p 4.5e-13) | 11.504 | 467.5 | 0.900 | 2.254 | 0 | 0.553 |
| fm_seed1_sig0.5 | all | 0.05848 | 5.560 | 7.129 (p 2.6e-40) | 0.450 | 514.8 | 0.899 | 5.755 | 1080 | 1.524 |
| fm_seed1_sig0.5 | north_N_NE_NW | 0.06017 | 7.295 | 8.962 (p 2.4e-13) | 4.079 | 550.5 | 0.905 | 5.800 | 1140 | 1.244 |
| fm_seed1_sig0.5 | array_in_view_gt5pct | 0.06128 | 6.501 | 9.405 (p 4.5e-13) | 13.675 | 570.4 | 0.915 | 6.148 | 1140 | 1.218 |
| fm_sig0.3_trained | all | 0.00405 | 0.477 | 0.981 (p 0.0012) | 0.234 | 54.1 | 0.367 | 0.091 | 0 | 0.341 |
| fm_sig0.3_trained | north_N_NE_NW | 0.00421 | 0.542 | 1.005 (p 0.1) | 1.248 | 49.9 | 0.362 | 0.083 | 0 | 0.300 |
| fm_sig0.3_trained | array_in_view_gt5pct | 0.00429 | 0.778 | 1.046 (p 0.34) | 3.245 | 54.7 | 0.346 | 0.114 | 0 | 0.336 |
| crps_pure_ft | all | 0.00388 | 0.483 | 0.991 (p 0.017) | 0.224 | 48.4 | 0.383 | 0.107 | 0 | 0.351 |
| crps_pure_ft | north_N_NE_NW | 0.00415 | 0.564 | 1.037 (p 0.2) | 1.191 | 48.2 | 0.393 | 0.097 | 0 | 0.307 |
| crps_pure_ft | array_in_view_gt5pct | 0.00429 | 0.805 | 1.081 (p 0.67) | 3.750 | 52.4 | 0.384 | 0.110 | 0 | 0.351 |
| crps_blend_ft | all | 0.00387 | 0.477 | 0.981 (p 0.015) | 0.207 | 49.7 | 0.385 | 0.107 | 0 | 0.345 |
| crps_blend_ft | north_N_NE_NW | 0.00422 | 0.556 | 1.025 (p 0.41) | 1.056 | 49.8 | 0.394 | 0.100 | 0 | 0.306 |
| crps_blend_ft | array_in_view_gt5pct | 0.00440 | 0.800 | 1.075 (p 0.77) | 3.861 | 52.7 | 0.367 | 0.108 | 0 | 0.332 |
| crps_pure_ft_S4 | all | 0.00382 | 0.481 | 0.987 (p 0.011) | 0.224 | 47.4 | 0.380 | 0.108 | 0 | 0.346 |
| crps_pure_ft_S4 | north_N_NE_NW | 0.00412 | 0.574 | 1.052 (p 0.26) | 1.277 | 48.2 | 0.383 | 0.100 | 0 | 0.303 |
| crps_pure_ft_S4 | array_in_view_gt5pct | 0.00422 | 0.774 | 1.040 (p 0.5) | 3.578 | 51.6 | 0.360 | 0.103 | 0 | 0.348 |
| thresh_seed0 | all | 0.00394 | 0.506 | 1.027 (p 0.00021) | 0.260 | 47.8 | 0.388 | 0.110 | 0 | 0.341 |
| thresh_seed0 | north_N_NE_NW | 0.00400 | 0.560 | 1.032 (p 0.37) | 1.134 | 48.4 | 0.372 | 0.105 | 0 | 0.287 |
| thresh_seed0 | array_in_view_gt5pct | 0.00410 | 0.770 | 1.034 (p 0.16) | 3.488 | 50.2 | 0.348 | 0.109 | 0 | 0.313 |
| thresh_seed1 | all | 0.00390 | 0.480 | 0.985 (p 5.9e-06) | 0.223 | 48.0 | 0.387 | 0.104 | 0 | 0.339 |
| thresh_seed1 | north_N_NE_NW | 0.00397 | 0.541 | 1.003 (p 0.16) | 1.196 | 46.3 | 0.371 | 0.090 | 0 | 0.304 |
| thresh_seed1 | array_in_view_gt5pct | 0.00409 | 0.741 | 0.995 (p 0.16) | 3.080 | 48.9 | 0.361 | 0.101 | 0 | 0.312 |

## Verdicts

- **fm_seed1**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: no
- **fm_seed1+temp_global_in_sample**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✓/90✓; mean regressed: no
- **fm_seed1+temp_grouped_in_sample**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✓/90✓; mean regressed: no
- **fm_seed1+temp_global_crossfit**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✓/90✓; mean regressed: no
- **fm_seed1+temp_grouped_crossfit**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✓/90✓; mean regressed: no
- **fm_seed1_e2**: array-share coverage in band — all 50✗/90✗, north_N_NE_NW 50✗/90✗, array_in_view_gt5pct 50✗/90✗; mean regressed: no
- **fm_seed1_sig0.2**: array-share coverage in band — all 50✗/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **fm_seed1_sig0.3**: array-share coverage in band — all 50✗/90✗, north_N_NE_NW 50✗/90✗, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **fm_seed1_sig0.5**: array-share coverage in band — all 50✗/90✗, north_N_NE_NW 50✗/90✗, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **fm_sig0.3_trained**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **crps_pure_ft**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **crps_blend_ft**: array-share coverage in band — all 50✓/90✓, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **crps_pure_ft_S4**: array-share coverage in band — all 50✓/90✗, north_N_NE_NW 50✓/90✗, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **thresh_seed0**: array-share coverage in band — all 50✗/90✗, north_N_NE_NW 50✓/90✓, array_in_view_gt5pct 50✗/90✗; mean regressed: YES
- **thresh_seed1**: array-share coverage in band — all 50✓/90✗, north_N_NE_NW 50✓/90✗, array_in_view_gt5pct 50✗/90✗; mean regressed: YES

**Temperature scaling alone fixes coverage (cross-fitted): global tau = 1.19 -> YES; grouped tau (pred. array in view 1.38 / absent 1.14) -> YES.**
