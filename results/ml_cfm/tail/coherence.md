# Tail coherence on the two-window pair (case_2023111718, wdir 335 deg, a TRAIN record; n = 1 pair)

Cone cells 4322. Tail band = cone cells with f != 0 below the window's own source-area level; body = cells at or above it. Null Jaccard: each window's band placed independently with the same occupancy per shell (10 x' quantile bins x 5 |y'| bins); residual r: after removing shell means.

| level | part | n w0 | n w1 | |mass| frac w0 | Jaccard | null (expected / perm 95%) | J / null | resid Pearson (union) | resid Spearman (union) | resid Pearson (both-tail cells, n) | neg sign agree (null) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 (lev/peak 0.0105) | tail | 1692 | 1835 | 12.74% | 0.644 | 0.570 / [0.561, 0.581] | 1.13 | -0.015 | 0.125 | 0.037 (1382) | 0.19 (0.21) |
| 0.9 (lev/peak 0.0105) | body | 336 | 501 | 87.26% | 0.438 | 0.339 / [0.320, 0.361] | 1.29 | 0.868 | 0.504 | 0.848 (255) | 0.00 (0.03) |
| 0.95 (lev/peak 0.00493) | tail | 1508 | 1565 | 7.92% | 0.524 | 0.469 / [0.458, 0.480] | 1.12 | -0.101 | 0.074 | 0.060 (1056) | 0.19 (0.23) |
| 0.95 (lev/peak 0.00493) | body | 520 | 771 | 92.08% | 0.406 | 0.330 / [0.313, 0.348] | 1.23 | 0.869 | 0.366 | 0.853 (373) | 0.00 (0.05) |
| 0.99 (lev/peak 0.00136) | tail | 1113 | 1051 | 4.06% | 0.305 | 0.269 / [0.254, 0.284] | 1.13 | -0.232 | -0.111 | 0.037 (506) | 0.19 (0.28) |
| 0.99 (lev/peak 0.00136) | body | 915 | 1285 | 95.94% | 0.471 | 0.376 / [0.361, 0.391] | 1.25 | 0.870 | 0.287 | 0.859 (704) | 0.00 (0.07) |
| 0.999 (lev/peak 0.000321) | tail | 798 | 692 | 3.18% | 0.177 | 0.161 / [0.145, 0.175] | 1.10 | -0.317 | -0.296 | 0.097 (224) | 0.19 (0.36) |
| 0.999 (lev/peak 0.000321) | body | 1230 | 1644 | 96.82% | 0.547 | 0.448 / [0.436, 0.461] | 1.22 | 0.873 | 0.278 | 0.865 (1016) | 0.00 (0.09) |

**Verdict at the 99% level: NOISE** (noise if tail J/null <= 1.5 and resid r < 0.1 (signed); coherent if J/null >= 3 or r >= 0.3; else mixed).

## The realisation floor before and after the 99% source-area threshold (w1 vs w0)

| quantity | raw | thresholded | Kljun vs w0/w1 raw | Kljun_thr vs w_thr |
|---|---|---|---|---|
| rel_l2 | 0.4048 | 0.4029 | 0.44 / 0.3625 | 0.439 / 0.3613 |
| shape_l1_2d | 0.6298 | 0.6039 | 0.5769 / 0.5113 | 0.5551 / 0.5018 |
| overlap80 | 0.5072 | 0.537 | 0.5762 / 0.5625 | 0.5929 / 0.5625 |
| array_share | 5.341 | 4.993 | 1.31 / 2.035 | 0.4789 / 1.612 |
| centroid | 50.59 | 40.02 | 84.16 / 39.74 | 54.39 / 18.1 |
| integral | 0.0007178 | 0.009925 | 0.06457 / 0.05473 | 0.09511 / 0.07474 |
| peak_x | 0 | 0 | 0 / 0 | 0 / 0 |
| shape_1d | 0.0655 | 0.05833 | 0.0905 / 0.06464 | 0.07708 / 0.05839 |

Threshold removed 4.06% / 3.05% of |mass| (negative mass 3.09% / 2.08%), keeping 915 / 1285 of 2028 / 2336 non-zero cells; level / peak 0.00136 / 0.00146.

