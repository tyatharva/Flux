# Reporting metrics on val (frozen recipe, 235 records)

## Training losses

- **FNO**: masked MSE + 0.03 x masked MAE, both in asinh space (global target scale) over the 122^2 interior, on the residual to Kljun; unweighted; selection on val masked MSE.
- **CFM**: conditional flow matching, velocity parameterisation: z_t = x_K + t (x_LES - x_K) + (1 - t) eps, eps ~ N(0, 0.1^2) on the cone cells, target v = (x_LES - x_K) - eps, MSE over the cone cells, t ~ U(0,1); asinh space anchored on Kljun; unweighted; selection on the val MSE of the 16-step Euler sample mean.

## all (n = 235): medians over records

| model | peak distance error [m] | centroid error [m] | 80% source-area overlap (Jaccard) | integral error |
|---|---|---|---|---|
| Kljun | 30.0 | 106.6 | 0.566 | 0.138 |
| FNO | 0.0 | 68.8 | 0.619 | 0.106 |
| CFM | 0.0 | 52.6 | 0.619 | 0.098 |
| two-window floor | 0.0 | 40.0 | 0.507 | 0.010 |

| model | rel. L2 | sliced W1 [m] | KL(LES‖model) [nats] | MS-SSIM (log grid) |
|---|---|---|---|---|
| Kljun | 0.540 | 69.0 | 0.362 | 0.943 |
| FNO | 0.340 | 46.3 | 0.256 | 0.942 |
| CFM | 0.333 | 36.5 | 0.244 | 0.946 |
| two-window floor | 0.403 | 32.6 | 0.454 | 0.916 |

Octant groups (N n=20, NE n=7, E n=10, SE n=24, S n=42, SW n=48, W n=40, NW n=44) are in the JSON and the per-record .npz, for the wind-rose graphs. Production errors: |model - LES| of the upwind peak distance and of the field integral, the distance between the two mass centroids, and the Jaccard of the two 80% source areas (1 = identical). Field metrics on the 122² interior with log floor ε = 1e-09 m⁻²: rel. L2 = ||model - LES|| / ||LES||; sliced W1 = mean over 64 directions of the 1-D Wasserstein-1 between the unit-mass positive parts [m]; KL = Σ P log(P/Q) with P = LES/ΣLES and Q = (model+ε)/Σ (0.008 nats smoothing bias at identical fields); MS-SSIM = 5-scale SSIM on the log10 grid (1 = identical). The floor row is window 1 vs window 0 of the one two-window case (a train record, n = 1), processed identically.
