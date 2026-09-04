# Reporting metrics on val (frozen recipe, 235 records)

## Training losses

- **FNO**: masked MSE + 0.03 x masked MAE, both in asinh space (global target scale) over the 122^2 interior, on the residual to Kljun; unweighted; selection on val masked MSE.
- **CFM**: conditional flow matching, velocity parameterisation: z_t = x_K + t (x_LES - x_K) + (1 - t) eps, eps ~ N(0, 0.1^2) on the cone cells, target v = (x_LES - x_K) - eps, MSE over the cone cells, t ~ U(0,1); asinh space anchored on Kljun; unweighted; selection on the val MSE of the 16-step Euler sample mean.

## Agreement composite: mean of four bounded ratios per record, each in [0, 1] with 1 = the LES; medians over the group

peak = min(x_peak) / max(x_peak) of the two; centroid = 1 - |c_model - c_LES| / (|c_model| + |c_LES|); overlap = Jaccard of the 80% source areas; integral = min(I) / max(I) of the two.

### all (n = 235)

| model | agreement | peak | centroid | overlap80 | integral |
|---|---|---|---|---|---|
| Kljun | 0.765 | 0.818 | 0.899 | 0.566 | 0.852 |
| FNO | 0.846 | 1.000 | 0.935 | 0.619 | 0.898 |
| CFM | 0.845 | 1.000 | 0.952 | 0.619 | 0.901 |
| two-window floor | 0.863 | 1.000 | 0.953 | 0.507 | 0.990 |

| model | rel. L2 | sliced W1 [m] | KL(LES‖model) [nats] | MS-SSIM (log grid) |
|---|---|---|---|---|
| Kljun | 0.540 | 69.0 | 0.362 | 0.943 |
| FNO | 0.340 | 46.3 | 0.256 | 0.942 |
| CFM | 0.334 | 36.6 | 0.244 | 0.946 |
| two-window floor | 0.403 | 32.6 | 0.454 | 0.916 |

### north_N_NE_NW (n = 71)

| model | agreement | peak | centroid | overlap80 | integral |
|---|---|---|---|---|---|
| Kljun | 0.803 | 0.867 | 0.913 | 0.580 | 0.832 |
| FNO | 0.854 | 1.000 | 0.929 | 0.632 | 0.899 |
| CFM | 0.860 | 1.000 | 0.945 | 0.632 | 0.919 |
| two-window floor | 0.863 | 1.000 | 0.953 | 0.507 | 0.990 |

| model | rel. L2 | sliced W1 [m] | KL(LES‖model) [nats] | MS-SSIM (log grid) |
|---|---|---|---|---|
| Kljun | 0.465 | 60.1 | 0.308 | 0.938 |
| FNO | 0.318 | 47.3 | 0.264 | 0.937 |
| CFM | 0.309 | 37.7 | 0.244 | 0.939 |
| two-window floor | 0.403 | 32.6 | 0.454 | 0.916 |

### array_in_view_gt5pct (n = 42)

| model | agreement | peak | centroid | overlap80 | integral |
|---|---|---|---|---|---|
| Kljun | 0.811 | 1.000 | 0.938 | 0.586 | 0.815 |
| FNO | 0.851 | 1.000 | 0.932 | 0.630 | 0.894 |
| CFM | 0.863 | 1.000 | 0.942 | 0.638 | 0.915 |
| two-window floor | 0.863 | 1.000 | 0.953 | 0.507 | 0.990 |

| model | rel. L2 | sliced W1 [m] | KL(LES‖model) [nats] | MS-SSIM (log grid) |
|---|---|---|---|---|
| Kljun | 0.461 | 39.4 | 0.292 | 0.935 |
| FNO | 0.328 | 42.1 | 0.262 | 0.932 |
| CFM | 0.318 | 37.2 | 0.237 | 0.933 |
| two-window floor | 0.403 | 32.6 | 0.454 | 0.916 |

## CRPS of the array share [pp], CFM samples (80 per record), median over records

| group | tau = 1 | tau = 1.19 |
|---|---|---|
| all | 0.154 | 0.162 |
| north_N_NE_NW | 0.996 | 1.051 |
| array_in_view_gt5pct | 2.046 | 2.081 |

Groups: all = every record; north_N_NE_NW = wind from N, NE or NW; array_in_view_gt5pct = records where the LES puts more than 5% of the footprint on the array. Medians over the group's records. Field metrics on the 122² interior with log floor ε = 1e-09 m⁻²: rel. L2 = ||model - LES|| / ||LES||; sliced W1 = mean over 64 directions of the 1-D Wasserstein-1 between the unit-mass positive parts [m]; KL = Σ P log(P/Q) with P = LES/ΣLES and Q = (model+ε)/Σ (0.008 nats smoothing bias at identical fields); MS-SSIM = 5-scale SSIM on the log10 grid (1 = identical). The floor row is window 1 vs window 0 of the one two-window case (a train record, n = 1), processed identically.
