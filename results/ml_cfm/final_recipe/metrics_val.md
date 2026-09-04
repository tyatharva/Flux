# Reporting metrics on val (frozen recipe, 235 records)

## Training losses

- **FNO**: masked MSE + 0.03 x masked MAE, both in asinh space (global target scale) over the 122^2 interior, on the residual to Kljun; unweighted; selection on val masked MSE.
- **CFM**: conditional flow matching, velocity parameterisation: z_t = x_K + t (x_LES - x_K) + (1 - t) eps, eps ~ N(0, 0.1^2) on the cone cells, target v = (x_LES - x_K) - eps, MSE over the cone cells, t ~ U(0,1); asinh space anchored on Kljun; unweighted; selection on the val MSE of the 16-step Euler sample mean.

## Composite = geometric mean over the five production quantities of median|error| / Kljun's median|error| (< 1 beats Kljun)

### all (n = 235)

| model | composite | peak_x [m] | centroid [m] | 1 - overlap80 | array share [pp] | integral |
|---|---|---|---|---|---|---|
| Kljun | 1.000 | 30.0 | 106.6 | 0.434 | 1.44 | 0.138 |
| FNO | 0.545 | 0.0 | 68.8 | 0.381 | 0.29 | 0.106 |
| CFM | 0.476 | 0.0 | 52.5 | 0.381 | 0.25 | 0.097 |
| two-window floor | - | 0.0 | 40.0 | 0.493 | 4.95 | 0.010 |

| model | log-MSE (dex²) | log-MSE, LES body | sliced W1 [m] | KL(LES‖model) [nats] | MS-SSIM (log grid) | rel. L2 |
|---|---|---|---|---|---|---|
| Kljun | 0.113 | 0.297 | 69.0 | 0.362 | 0.943 | 0.540 |
| FNO | 0.116 | 0.369 | 46.3 | 0.256 | 0.942 | 0.340 |
| CFM | 0.114 | 0.388 | 36.6 | 0.244 | 0.946 | 0.334 |
| two-window floor | 0.214 | 0.830 | 32.6 | 0.454 | 0.916 | 0.403 |

### north_N_NE_NW (n = 71)

| model | composite | peak_x [m] | centroid [m] | 1 - overlap80 | array share [pp] | integral |
|---|---|---|---|---|---|---|
| Kljun | 1.000 | 30.0 | 92.6 | 0.420 | 3.86 | 0.177 |
| FNO | 0.617 | 0.0 | 69.3 | 0.368 | 1.37 | 0.110 |
| CFM | 0.545 | 0.0 | 54.0 | 0.368 | 1.40 | 0.084 |
| two-window floor | - | 0.0 | 40.0 | 0.493 | 4.95 | 0.010 |

| model | log-MSE (dex²) | log-MSE, LES body | sliced W1 [m] | KL(LES‖model) [nats] | MS-SSIM (log grid) | rel. L2 |
|---|---|---|---|---|---|---|
| Kljun | 0.124 | 0.358 | 60.1 | 0.308 | 0.938 | 0.465 |
| FNO | 0.131 | 0.464 | 47.3 | 0.264 | 0.937 | 0.318 |
| CFM | 0.136 | 0.523 | 37.7 | 0.244 | 0.939 | 0.309 |
| two-window floor | 0.214 | 0.830 | 32.6 | 0.454 | 0.916 | 0.403 |

### array_in_view_gt5pct (n = 42)

| model | composite | peak_x [m] | centroid [m] | 1 - overlap80 | array share [pp] | integral |
|---|---|---|---|---|---|---|
| Kljun | 1.000 | 0.0 | 56.3 | 0.414 | 5.07 | 0.204 |
| FNO | 0.831 | 0.0 | 60.8 | 0.370 | 3.60 | 0.118 |
| CFM | 0.759 | 0.0 | 53.4 | 0.362 | 3.27 | 0.096 |
| two-window floor | - | 0.0 | 40.0 | 0.493 | 4.95 | 0.010 |

| model | log-MSE (dex²) | log-MSE, LES body | sliced W1 [m] | KL(LES‖model) [nats] | MS-SSIM (log grid) | rel. L2 |
|---|---|---|---|---|---|---|
| Kljun | 0.125 | 0.334 | 39.4 | 0.292 | 0.935 | 0.461 |
| FNO | 0.133 | 0.562 | 42.1 | 0.262 | 0.932 | 0.328 |
| CFM | 0.140 | 0.571 | 37.2 | 0.237 | 0.933 | 0.318 |
| two-window floor | 0.214 | 0.830 | 32.6 | 0.454 | 0.916 | 0.403 |

Medians over the group's records. Production errors are |model - LES| per record (overlap80 as 1 - Jaccard of the 80% source areas). Field metrics on the 122² interior with log floor ε = 1e-09 m⁻²: log-MSE = mean (log10(f+ε) - log10(LES+ε))² over the interior, and over the cells inside the LES's own 99.5% source area ('body'); sliced W1 = mean over 64 directions of the 1-D Wasserstein-1 between the unit-mass positive parts (placement/shape, blind to amplitude); KL = Σ P log(P/Q) with P = LES/ΣLES and Q = (model+ε)/Σ; MS-SSIM = 5-scale SSIM on the log10 grid (1 = identical). The floor row is window 1 vs window 0 of the one two-window case (a train record, n = 1), processed identically.

## Floor sensitivity (every 4th record, n = 59): medians of log-MSE / MS-SSIM / KL at four values of ε

| ε [m⁻²] | Kljun log-MSE / MS-SSIM / KL | FNO log-MSE / MS-SSIM / KL | CFM log-MSE / MS-SSIM / KL |
|---|---|---|---|
| 1e-09 | 0.117 / 0.945 / 0.360 | 0.111 / 0.946 / 0.249 | 0.113 / 0.950 / 0.231 |
| 1e-08 | 0.041 / 0.951 / 0.458 | 0.036 / 0.952 / 0.343 | 0.038 / 0.957 / 0.328 |
| 1e-07 | 0.010 / 0.954 / 1.093 | 0.008 / 0.961 / 1.002 | 0.008 / 0.965 / 0.991 |
| 1e-06 | 0.001 / 0.968 / 2.449 | 0.001 / 0.975 / 2.357 | 0.001 / 0.978 / 2.368 |

Read: log-MSE and MS-SSIM are dominated by the cells that sit at the floor in both fields, so their level is set by ε and the model ordering is inside the noise; KL's smoothing bias at identical fields is 0.008 nats at ε = 1e-9 and grows with ε. Sliced W1 and the composite do not depend on ε.
