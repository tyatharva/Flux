# The cut: what it removes, and the metrics before and after (val medians)

| field | 99% cut: |mass| removed | 99% cut: non-zero cells removed | 1e-8 floor: |mass| removed | 1e-8 floor: cells removed | 1e-8 as a fraction of the peak (median) |
|---|---|---|---|---|---|
| kljun | 1.00% | 80% | 0.16% | 74% | 4.6e-04 |
| fno | 1.19% | 69% | 0.57% | 60% | 4.7e-04 |
| cfm_mean800 | 1.46% | 68% | 0.71% | 59% | 4.8e-04 |
| cfm_mean70 | 1.54% | 68% | 0.83% | 59% | 4.9e-04 |
| cfm_one_sample | 6.38% | 63% | 5.56% | 50% | 4.5e-04 |
| LES target | 2.51% | 45% | 1.58% | 29% | - |

| field | composite all | north | in view |
|---|---|---|---|
| kljun | 1.000 | 1.000 | 1.000 |
| kljun_sa99 | 1.007 | 1.025 | 0.995 |
| kljun_abs1e-8 | 1.001 | 1.008 | 1.001 |
| fno | 0.526 | 0.597 | 0.800 |
| fno_sa99 | 0.551 | 0.612 | 0.814 |
| fno_abs1e-8 | 0.538 | 0.609 | 0.809 |
| cfm_mean800 | 0.473 | 0.550 | 0.761 |
| cfm_mean800_sa99 | 0.479 | 0.554 | 0.769 |
| cfm_mean800_abs1e-8 | 0.480 | 0.540 | 0.759 |
| cfm_mean70 | 0.490 | 0.555 | 0.767 |
| cfm_mean70_sa99 | 0.483 | 0.556 | 0.779 |
| cfm_mean70_abs1e-8 | 0.487 | 0.541 | 0.764 |
| cfm_one_sample | 0.743 | 0.835 | 1.052 |
| cfm_one_sample_sa99 | 0.682 | 0.748 | 0.987 |
| cfm_one_sample_abs1e-8 | 0.676 | 0.743 | 0.977 |

# Seed ensembles: composite vs Kljun of the pooled mean over k seeds (160 samples each), mean ± sd over the C(5,k) subsets

| k | n subsets | all | north | in view | best subset (all) | worst subset (all) |
|---|---|---|---|---|---|---|
| 1 | 5 | 0.502 ± 0.036 | 0.569 ± 0.048 | 0.788 ± 0.046 | 0.467 | 0.550 |
| 2 | 10 | 0.486 ± 0.017 | 0.552 ± 0.019 | 0.775 ± 0.016 | 0.468 | 0.518 |
| 3 | 10 | 0.481 ± 0.011 | 0.549 ± 0.012 | 0.767 ± 0.010 | 0.469 | 0.505 |
| 4 | 5 | 0.477 ± 0.004 | 0.550 ± 0.007 | 0.764 ± 0.003 | 0.472 | 0.481 |
| 5 | 1 | 0.473 ± 0.000 | 0.550 ± 0.000 | 0.761 ± 0.000 | 0.473 | 0.473 |
