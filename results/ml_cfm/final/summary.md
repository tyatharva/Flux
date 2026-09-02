# Phase 1 summary

6 runs in `results/ml_cfm/final`; baseline `seed*` n = 5.

## Baseline seed spread

| quantity | mean | sd | n |
|---|---|---|---|
| val_loss | 0.000119284 | 5.09e-06 | 5 |
| val_mse_ref | 0.000119284 | 5.09e-06 | 5 |
| composite | 0.512921 | 0.0406 | 5 |
| composite_north | 0.578064 | 0.0525 | 5 |

## GPU

utilisation mean 99% / p50 100% / p90 100%; memory mean 6523 MiB, max 7713 MiB; concurrent processes mean 2.8, max 4 (578 samples)

## Runs, as z-scores against the baseline seed spread (negative = better for losses and composites)

| run | val_mse_ref | z | composite | z | composite_north | z | gap x | best/epochs | r_array | rn_array | r_centroid | r_overlap | r_integral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long1000_seed0 | 0.00011635 | -0.6 | 0.570 | +1.4 | 0.639 | +1.2 | 0.99 | 105/206 | 0.18 | 0.32 | 0.74 | 0.99 | 0.79 |
| seed3 | 0.000116906 | -0.5 | 0.475 | -0.9 | 0.539 | -0.7 | 1.00 | 95/146 | 0.13 | 0.33 | 0.64 | 0.87 | 0.71 |
| seed2 | 0.000116958 | -0.5 | 0.489 | -0.6 | 0.534 | -0.8 | 0.99 | 90/141 | 0.17 | 0.27 | 0.56 | 0.88 | 0.69 |
| seed4 | 0.000116995 | -0.4 | 0.565 | +1.3 | 0.580 | +0.0 | 1.03 | 120/171 | 0.16 | 0.29 | 0.78 | 1.04 | 0.82 |
| seed1 | 0.00011718 | -0.4 | 0.488 | -0.6 | 0.573 | -0.1 | 0.99 | 90/141 | 0.16 | 0.34 | 0.54 | 0.88 | 0.74 |
| seed0 | 0.000128381 | +1.8 | 0.547 | +0.8 | 0.665 | +1.6 | 0.95 | 60/111 | 0.19 | 0.36 | 0.62 | 0.88 | 0.85 |

Spearman rank correlation across runs: val_mse_ref vs composite -0.14, vs composite_north +0.31.
