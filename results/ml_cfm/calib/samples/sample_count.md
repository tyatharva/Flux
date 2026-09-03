# Sample count study: 5 seeds x 160 samples (Euler 16), val

Sampling cost 8.2 ms per record per sample. Bands: sd over 5 random subsets.

## Pooled over seeds: composite vs Kljun of the S-sample mean

| S | all | north_N_NE_NW | array_in_view_gt5pct | rel_l2 (all) | shape_l1_2d (all) | array_share pp (all) |
|---|---|---|---|---|---|---|
| 5 | 0.518 ± 0.021 | 0.582 ± 0.042 | 0.790 ± 0.030 | 0.364 | 0.506 | 0.253 |
| 10 | 0.502 ± 0.010 | 0.574 ± 0.023 | 0.789 ± 0.022 | 0.344 | 0.478 | 0.228 |
| 20 | 0.493 ± 0.004 | 0.568 ± 0.017 | 0.778 ± 0.025 | 0.338 | 0.472 | 0.231 |
| 40 | 0.481 ± 0.007 | 0.556 ± 0.011 | 0.772 ± 0.023 | 0.336 | 0.467 | 0.225 |
| 80 | 0.485 ± 0.008 | 0.545 ± 0.013 | 0.767 ± 0.016 | 0.335 | 0.465 | 0.228 |
| 160 | 0.479 ± 0.005 | 0.551 ± 0.005 | 0.772 ± 0.010 | 0.335 | 0.463 | 0.219 |
| 320 | 0.479 ± 0.005 | 0.548 ± 0.005 | 0.763 ± 0.010 | 0.334 | 0.463 | 0.222 |
| 640 | 0.477 ± 0.002 | 0.553 ± 0.003 | 0.766 ± 0.004 | 0.334 | 0.464 | 0.216 |
| 800 | 0.473 ± 0.000 | 0.550 ± 0.000 | 0.761 ± 0.000 | 0.335 | 0.463 | 0.210 |

## Per seed: composite vs Kljun (all records)

| S | seed0 | seed1 | seed2 | seed3 | seed4 |
|---|---|---|---|---|---|
| 1 | 0.686 | 0.628 | 0.625 | 0.629 | 0.663 |
| 2 | 0.628 | 0.557 | 0.576 | 0.553 | 0.612 |
| 4 | 0.583 | 0.529 | 0.517 | 0.516 | 0.581 |
| 8 | 0.553 | 0.507 | 0.494 | 0.495 | 0.573 |
| 16 | 0.541 | 0.492 | 0.495 | 0.488 | 0.563 |
| 32 | 0.538 | 0.480 | 0.491 | 0.473 | 0.550 |
| 64 | 0.534 | 0.478 | 0.486 | 0.471 | 0.552 |
| 128 | 0.534 | 0.480 | 0.482 | 0.465 | 0.547 |
| 160 | 0.532 | 0.484 | 0.478 | 0.467 | 0.550 |

## Convergence law fits, err(S) = a + b S^-p (p = 1/2 or 1, whichever fits better)

| scope | group / metric | law | asymptote a | at max S | excess at max S | S for +1% | S for +2% | rms resid |
|---|---|---|---|---|---|---|---|---|
| pooled | all/composite | 1/sqrt(S) | 0.4713 | 0.4733 | 0.4% | 449 | 112 | 0.0026 |
| pooled | all/rel_l2 | 1/S | 0.3331 | 0.3354 | 0.7% | 43 | 21 | 0.0018 |
| pooled | all/array_share | 1/sqrt(S) | 0.2132 | 0.2097 | -1.6% | 1319 | 330 | 0.0048 |
| pooled | north_N_NE_NW/composite | 1/sqrt(S) | 0.5445 | 0.5503 | 1.1% | 248 | 62 | 0.0041 |
| pooled | array_in_view_gt5pct/composite | 1/sqrt(S) | 0.7615 | 0.7614 | -0.0% | 89 | 22 | 0.0032 |
| per_seed/seed0 | all/composite | 1/S | 0.5345 | 0.5321 | -0.5% | 30 | 15 | 0.0064 |
| per_seed/seed0 | all/rel_l2 | 1/S | 0.3500 | 0.3495 | -0.1% | 24 | 12 | 0.0020 |
| per_seed/seed0 | all/array_share | 1/sqrt(S) | 0.2733 | 0.2790 | 2.1% | 302 | 75 | 0.0039 |
| per_seed/seed0 | north_N_NE_NW/composite | 1/S | 0.6376 | 0.6406 | 0.5% | 20 | 10 | 0.0119 |
| per_seed/seed0 | array_in_view_gt5pct/composite | 1/sqrt(S) | 0.8259 | 0.8545 | 3.5% | 491 | 123 | 0.0133 |
| per_seed/seed1 | all/composite | 1/S | 0.4821 | 0.4845 | 0.5% | 31 | 15 | 0.0051 |
| per_seed/seed1 | all/rel_l2 | 1/S | 0.3395 | 0.3387 | -0.2% | 27 | 14 | 0.0021 |
| per_seed/seed1 | all/array_share | 1/sqrt(S) | 0.2070 | 0.2189 | 5.8% | 2090 | 523 | 0.0040 |
| per_seed/seed1 | north_N_NE_NW/composite | 1/S | 0.5464 | 0.5519 | 1.0% | 29 | 14 | 0.0052 |
| per_seed/seed1 | array_in_view_gt5pct/composite | 1/S | 0.7540 | 0.7542 | 0.0% | 25 | 12 | 0.0055 |
| per_seed/seed2 | all/composite | 1/S | 0.4826 | 0.4775 | -1.1% | 31 | 16 | 0.0075 |
| per_seed/seed2 | all/rel_l2 | 1/S | 0.3419 | 0.3411 | -0.2% | 26 | 13 | 0.0031 |
| per_seed/seed2 | all/array_share | 1/sqrt(S) | 0.2227 | 0.2247 | 0.9% | 1011 | 253 | 0.0083 |
| per_seed/seed2 | north_N_NE_NW/composite | 1/S | 0.5392 | 0.5320 | -1.3% | 32 | 16 | 0.0096 |
| per_seed/seed2 | array_in_view_gt5pct/composite | 1/S | 0.7517 | 0.7575 | 0.8% | 25 | 13 | 0.0079 |
| per_seed/seed3 | all/composite | 1/S | 0.4703 | 0.4674 | -0.6% | 34 | 17 | 0.0045 |
| per_seed/seed3 | all/rel_l2 | 1/S | 0.3362 | 0.3367 | 0.2% | 26 | 13 | 0.0018 |
| per_seed/seed3 | all/array_share | 1/sqrt(S) | 0.1769 | 0.1906 | 7.7% | 2756 | 689 | 0.0037 |
| per_seed/seed3 | north_N_NE_NW/composite | 1/S | 0.5345 | 0.5254 | -1.7% | 36 | 18 | 0.0102 |
| per_seed/seed3 | array_in_view_gt5pct/composite | 1/S | 0.7607 | 0.7561 | -0.6% | 25 | 12 | 0.0082 |
| per_seed/seed4 | all/composite | 1/S | 0.5513 | 0.5496 | -0.3% | 21 | 10 | 0.0042 |
| per_seed/seed4 | all/rel_l2 | 1/S | 0.3439 | 0.3436 | -0.1% | 25 | 13 | 0.0026 |
| per_seed/seed4 | all/array_share | 1/sqrt(S) | 0.2003 | 0.2104 | 5.1% | 1171 | 293 | 0.0063 |
| per_seed/seed4 | north_N_NE_NW/composite | 1/S | 0.5750 | 0.5940 | 3.3% | 29 | 15 | 0.0156 |
| per_seed/seed4 | array_in_view_gt5pct/composite | 1/S | 0.8154 | 0.8184 | 0.4% | 23 | 11 | 0.0137 |
