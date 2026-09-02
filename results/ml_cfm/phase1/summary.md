# Phase 1 summary

5 runs in `results/ml_cfm/phase1`; baseline `v_s0.1_seed*` n = 2.

## Baseline seed spread

| quantity | mean | sd | n |
|---|---|---|---|
| val_loss | 0.000122159 | 7.33e-07 | 2 |
| val_mse_ref | 0.000122159 | 7.33e-07 | 2 |
| composite | 0.517765 | 0.00815 | 2 |
| composite_north | 0.549199 | 0.00824 | 2 |

## GPU

utilisation mean 99% / p50 100% / p90 100%; memory mean 8731 MiB, max 12428 MiB; concurrent processes mean 3.9, max 5 (368 samples)

## Runs, as z-scores against the baseline seed spread (negative = better for losses and composites)

| run | val_mse_ref | z | composite | z | composite_north | z | gap x | best/epochs | r_array | rn_array | r_centroid | r_overlap | r_integral |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v_s0.1_seed0 | 0.000121641 | -0.7 | 0.524 | +0.7 | 0.543 | -0.7 | 1.05 | 105/156 | 0.18 | 0.30 | 0.55 | 0.90 | 0.82 |
| v_s0.1_seed1 | 0.000122677 | +0.7 | 0.512 | -0.7 | 0.555 | +0.7 | 1.03 | 90/141 | 0.16 | 0.34 | 0.58 | 0.90 | 0.81 |
| v_s0.3 | 0.000124848 | +3.7 | 0.512 | -0.7 | 0.551 | +0.3 | 1.10 | 110/161 | 0.16 | 0.28 | 0.62 | 0.89 | 0.77 |
| x_s0.1 | 0.000126524 | +6.0 | 0.462 | -6.9 | 0.553 | +0.5 | 1.12 | 125/176 | 0.13 | 0.35 | 0.52 | 0.91 | 0.72 |
| smoke60 | 0.000131774 | +13.1 | 0.516 | -0.2 | 0.613 | +7.8 | 0.99 | 59/60 | 0.15 | 0.31 | 0.60 | 0.90 | 0.85 |

Spearman rank correlation across runs: val_mse_ref vs composite -0.30, vs composite_north +0.70.
