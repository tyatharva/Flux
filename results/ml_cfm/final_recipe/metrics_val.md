# Reporting metrics on val (frozen recipe, 235 records)

## Training losses

- **FNO**: masked MSE + 0.03 x masked MAE, both in asinh space (global target scale) over the 122^2 interior, on the residual to Kljun; unweighted; selection on val masked MSE.
- **CFM**: conditional flow matching, velocity parameterisation: z_t = x_K + t (x_LES - x_K) + (1 - t) eps, eps ~ N(0, 0.1^2) on the cone cells, target v = (x_LES - x_K) - eps, MSE over the cone cells, t ~ U(0,1); asinh space anchored on Kljun; unweighted; selection on the val MSE of the 16-step Euler sample mean.

## all (n = 235)

| model | peak distance RMSE [m] | centroid RMSE [m] | integral RMSE | overlap80 (Jaccard) | rel. L2 | sliced W1 [m] | JS distance [bits] | MS-SSIM (log grid) |
|---|---|---|---|---|---|---|---|---|
| Kljun | 118.4 | 130.1 | 0.258 | 0.548 | 0.578 | 75.3 | 0.362 | 0.937 |
| FNO | 47.1 | 86.7 | 0.190 | 0.607 | 0.362 | 50.8 | 0.290 | 0.937 |
| CFM | 48.8 | 68.8 | 0.197 | 0.607 | 0.358 | 40.6 | 0.287 | 0.939 |
| two-window floor (n = 1) | 0.0 | 40.0 | 0.010 | 0.507 | 0.403 | 32.6 | 0.362 | 0.916 |

Direction groups (N n=46, E n=24, S n=84, W n=81, N n=20, NE n=7, E n=10, SE n=24, S n=42, SW n=48, W n=40, NW n=44) are in the JSON and the per-record .npz, for the wind-rose graphs; sectors are 90 degrees centred on N/E/S/W, octants 45 degrees. Peak distance, centroid and integral: RMSE over records of the per-record error against the LES (|upwind peak distance difference|, distance between the mass centroids, |integral difference|). The rest are means over records of per-record scores: overlap80 = Jaccard of the two 80% source areas (1 = identical); rel. L2 = ||model - LES|| / ||LES|| on the 122² interior (0 = identical); sliced W1 = mean over 64 directions of the 1-D Wasserstein-1 between the unit-mass positive parts [m] (0 = identical); JS distance = sqrt of the Jensen-Shannon divergence in bits between the unit-mass positive parts (0 = identical, 1 = disjoint); MS-SSIM = 5-scale SSIM on the log10 grid with floor ε = 1e-09 m⁻² (1 = identical). The floor row is window 1 vs window 0 of the one two-window case (a train record, n = 1), processed identically: one error, not an RMSE.
