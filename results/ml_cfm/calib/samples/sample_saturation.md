# Saturation of the CFM sample mean on val (pooled 800 samples from 5 seeds)

Floor = record-bootstrap sd of the composite (2000 resamples of the val records). S_sat = the S at which the fitted remaining improvement b S^-p equals the floor. Bands: parametric bootstrap of the curve (1000 refits), 2.5-97.5%.

| group | n | composite at S_max [record-bootstrap 95%] | floor (sd) | law | asymptote [95%] | S_sat [95%] | S at half the floor | excess at S_max / floor |
|---|---|---|---|---|---|---|---|---|
| all | 235 | 0.473 [0.438, 0.523] | 0.0219 | 1/sqrt(S) | 0.471 [0.462, 0.480] | 21 [2, 64] | 83 | 0.16 |
| north_N_NE_NW | 71 | 0.550 [0.433, 0.684] | 0.0641 | 1/sqrt(S) | 0.545 [0.528, 0.560] | 2 [0, 15] | 7 | 0.05 |
| array_in_view_gt5pct | 42 | 0.761 [0.656, 0.863] | 0.0515 | 1/sqrt(S) | 0.761 [0.748, 0.776] | 2 [0, 13] | 8 | 0.05 |

**S chosen from the fit: 70** (upper 97.5% band of S_sat on all records, rounded up to a multiple of 10, capped at S_max).

## Headline at S = 70: composite over 10 random 70-subsets (mean ± sd), paired tests from the first subset

| group | n | CFM | FNO | CFM/FNO | array share pp CFM / FNO / Kljun (p CFM vs FNO) | centroid m (p) | shape L1 (p) | rel L2 (p) |
|---|---|---|---|---|---|---|---|---|
| all | 235 | 0.481 ± 0.007 | 0.526 | 0.932 ± 0.010 | 0.222 / 0.286 / 1.460 (0.0012) | 48.8 vs 55.1 (8.9e-07) | 0.465 vs 0.473 (0.0029) | 0.336 vs 0.340 (0.066) |
| north_N_NE_NW | 71 | 0.556 ± 0.013 | 0.597 | 0.944 ± 0.017 | 1.248 / 1.255 / 3.839 (0.24) | 48.5 vs 57.9 (0.0014) | 0.440 vs 0.442 (0.028) | 0.309 vs 0.316 (0.012) |
| array_in_view_gt5pct | 42 | 0.769 ± 0.009 | 0.800 | 0.961 ± 0.012 | 3.393 / 3.512 / 5.004 (0.26) | 53.5 vs 56.3 (0.22) | 0.438 vs 0.452 (0.91) | 0.315 vs 0.328 (0.29) |
