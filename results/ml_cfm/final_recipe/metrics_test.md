# Reporting metrics on test (frozen recipe, 294 records)

## Training losses

- **FNO**: masked MSE + 0.03 x masked MAE, both in asinh space (global target scale) over the 122^2 interior, on the residual to Kljun; unweighted; selection on val masked MSE.
- **CFM**: conditional flow matching, velocity parameterisation: z_t = x_K + t (x_LES - x_K) + (1 - t) eps, eps ~ N(0, 0.1^2) on the cone cells, target v = (x_LES - x_K) - eps, MSE over the cone cells, t ~ U(0,1); asinh space anchored on Kljun; unweighted; selection on the val MSE of the 16-step Euler sample mean.

## all (n = 294)

| model | peak distance RMSE [m] | centroid RMSE [m] | integral RMSE | overlap80 (Jaccard) | rel. L2 | sliced W1 [m] | JS distance [bits] | MS-SSIM (log grid) |
|---|---|---|---|---|---|---|---|---|
| Kljun | 104.0 | 129.3 | 0.240 | 0.548 | 0.565 | 75.0 | 0.359 | 0.937 |
| FNO | 33.1 | 92.8 | 0.184 | 0.604 | 0.365 | 53.5 | 0.292 | 0.937 |
| CFM | 30.6 | 69.3 | 0.190 | 0.604 | 0.359 | 40.9 | 0.286 | 0.941 |
| LES (perfect) | 0.0 | 0.0 | 0.000 | 1.000 | 0.000 | 0.0 | 0.000 | 1.000 |

Direction groups (N n=43, E n=38, S n=95, W n=118, N n=18, NE n=25, E n=9, SE n=27, S n=49, SW n=56, W n=68, NW n=42) are in the JSON and the per-record .npz, for the wind-rose graphs; sectors are 90 degrees centred on N/E/S/W, octants 45 degrees. Peak distance, centroid and integral: RMSE over records of the per-record error against the LES (|upwind peak distance difference|, distance between the mass centroids, |integral difference|). The rest are means over records of per-record scores: overlap80 = Jaccard of the two 80% source areas (1 = identical); rel. L2 = ||model - LES|| / ||LES|| on the 122² interior (0 = identical); sliced W1 = mean over 64 directions of the 1-D Wasserstein-1 between the unit-mass positive parts [m] (0 = identical); JS distance = sqrt of the Jensen-Shannon divergence in bits between the unit-mass positive parts (0 = identical, 1 = disjoint); MS-SSIM = 5-scale SSIM on the log10 grid with floor ε = 1e-09 m⁻² (1 = identical). The LES row is the LES target scored against itself: the perfect value of every column.
