# Calibration, sample count and the tail

The CFM follow-up of 2026-09-02, about 3 hours of wall time after the first CFM run. The code
is `ml_cfm/crps.py`, `ml_cfm/tailthresh.py`, `ml_cfm/calibrate.py` and `ml_cfm/sample_count.py`.
The gate is `bin/test_cfm.py` (26 checks). The numbers are in `results/ml_cfm/calib/` and
`results/ml_cfm/tail/`.

## The answers

1. **The under-dispersion where the array is in view is mostly a mis-scaling, not a
   mis-shaping.** A single global temperature τ = 1.19 on the sample spread, fitted on val and
   cross-fitted by record parity, brings the array-share coverage into the ±2 sd band on all
   three groups. The in-view group reaches the band's lower edge with z sd still 1.31, so
   about 30% residual under-dispersion survives a global rescale, and a group-specific τ is
   unstable across folds (1.02 vs 1.56 on n ≈ 20).
2. **CRPS training does not fix it.** Fine-tuning the flow on the pixelwise fair CRPS leaves
   the array-share coverage where it was. What it gives is a 2-step sampler as good as the
   16-step ODE.
3. **Raising σ at inference destroys the mean**, and a model trained at σ = 0.3 is the
   baseline again.
4. **The tail speckle is sampling noise** (one two-window pair, so n = 1).
5. **Thresholding the targets at their 99% source-area level changes nothing measurable.**

## Setup

Groups on val: all 235, N/NE/NW 71, array in view (LES share > 5%) 42. Coverage bands are
±2 binomial sd around nominal. "The mean has not regressed" means the composite of the
64-sample mean is inside the five final seeds' range [0.474, 0.567] and `val_mse_ref`
≤ 1.20e-4. Every variant is sampled with S = 64 on the same seed. The LES target is scored as
one more draw (PIT, z, coverage). The fair CRPS of the array share and integral are per
record, mean over the group. Spread/skill is rms sample sd × √((S+1)/S) / rmse of the sample
mean (1 = calibrated in the second moment). The field CRPS is the pixelwise fair CRPS in asinh
space over the cone cells, median over records.

## Temperature

Post-hoc temperature on the baseline's asinh-space samples, `T_s′ = T̄ + τ(T_s − T̄)`, with τ
fitted so the array-share z sd is 1:

| variant | τ | all c50 / c90 (z sd) | N/NE/NW c50 / c90 (z sd) | in view c50 / c90 (z sd) | integral in view c50 / c90 |
|---|---|---|---|---|---|
| none | 1 | 0.47 / 0.88 (1.19) | 0.44 / 0.85 (1.26) | 0.31 / 0.74 (1.55) | 0.38 / 0.69 |
| global, in-sample | 1.19 | 0.56 / 0.92 (1.00) | 0.55 / 0.93 (1.05) | 0.43 / 0.81 (1.30) | 0.45 / 0.74 |
| grouped, in-sample | 1.38 in view / 1.14 | 0.56 / 0.92 (1.00) | 0.56 / 0.94 (0.96) | 0.45 / 0.83 (1.18) | 0.50 / 0.81 |
| **global, cross-fitted** | 1.15 / 1.23 by fold | 0.56 / 0.92 (1.01) | 0.54 / 0.93 (1.07) | **0.40 / 0.81 (1.31)** | 0.48 / 0.74 |
| grouped, cross-fitted | 1.02–1.56 in view | 0.54 / 0.91 (1.06) | 0.52 / 0.93 (1.13) | 0.40 / 0.81 (1.39) | 0.48 / 0.71 |

The grouping is predictor-side (the CFM's own mean share > 5%, 40 records), because the LES
share is unknown at inference. Temperature does not change the sample mean in asinh space, and
the physical mean moves by < 0.01 in composite. The flow is well shaped and about 20%
under-dispersed overall. The integral stays under-dispersed after any array-share-fitted τ
(z sd 1.2–1.3). The frozen recipe ships τ = 1. At 1.19 the CRPS of the array share is 2–5%
worse, and coverage is not a reported metric.

## CRPS as a training objective (`results/ml_cfm/calib/runs/`)

The fair sample estimator, `mean_s|x_s − y| − Σ_{s<s'}|x_s − x_s'| / (S(S−1))`, per cell in asinh
space over the cone cells, on S samples drawn through the model's own Euler sampler inside
the loss (gradients through every step. `bin/test_cfm.py` checks the sorted form against the
pairwise one, the point-mass identity and unbiasedness at S = 2). Fine-tunes start from
`final/seed1`, AdamW 2e-4, batch 8, EMA, selection on the val field CRPS.

| run | loss | S / steps | best epoch / run | val CRPS | val_mse_ref | composite | in-view c50 / c90 | z sd | spread/skill |
|---|---|---|---|---|---|---|---|---|---|
| baseline, Euler 16 | fm | – / 16 | 90 / 141 | – | 1.172e-4 | 0.489 | 0.31 / 0.74 | 1.55 | 0.81 |
| baseline, Euler 2 | fm | – / 2 | same weights | – | – | 0.491 | 0.21 / 0.45 | 2.28 | 0.56 |
| crps_pure_ft | crps | 2 / 2 | 20 / 51 | 0.00426 | 1.273e-4 | 0.483 | 0.31 / 0.76 | 1.48 | 0.77 |
| crps_blend_ft (fm + 0.16·crps) | fm+crps | 2 / 2 | 20 / 51 | 0.00434 | 1.248e-4 | 0.477 | 0.29 / 0.71 | 1.55 | 0.77 |
| crps_pure_ft_S4 | crps | 4 / 4 | 10 / 30 | 0.00420 | 1.270e-4 | 0.481 | 0.26 / 0.69 | 1.70 | 0.67 |

λ = 0.16 makes the two terms equal at the checkpoint. All three CRPS runs reach their best
val CRPS by epoch 10–20 and then overfit (val/train MSE gap 1.22–1.26 against 0.99 before).
The t-augmentation that regularises the flow loss does not carry over to an objective that
samples through the model. The gain is real relative to the sampler it trains (the 2-step
flow model has field CRPS 0.00439, the CRPS-tuned 2-step model 0.00388, the 16-step value, at
1.0 ms per record per sample against 16.2 ms), but the array share is untouched: CRPS 2.75 pp
in view before and 2.75 / 2.75 / 2.80 after. The pixelwise objective is dominated by the
about 4000 tail cells of the cone where the spread is already right. The 44 array cells have
no weight in it.

Two conditional runs were triggered by the plan's rule (in-view cover90 < 0.81 after the
fine-tunes). Adding the array-share CRPS at weight 5 did not train at all (the val field CRPS
rose from the first evaluation. A scalar term on 44 cells with S = 2 is too noisy a gradient).
CRPS from scratch with a 2-step generator reached the baseline's field CRPS (0.00392) and
composite (0.477) in 45 epochs, but with half the flow's spread (in-view sample sd 2.0 pp
against 3.9, and the LES needs about 5. Coverage 0.17 / 0.40). Trained from nothing, the
pixelwise CRPS optimum at S = 2 is nearly deterministic on the near field.

## Raising σ

| σ at sampling | composite vs Kljun | in-view c50 / c90 | centroid [m] |
|---|---|---|---|
| 0.1 (trained) | 0.489 | 0.31 / 0.74 | 49 |
| 0.2 | 1.256 | 0.33 / 0.64 | 184 |
| 0.3 | 2.349 | 0.02 / 0.24 | 391 |
| 0.5 | 5.560 | 0.00 / 0.07 | 515 |

The network has only seen `z_0 = x_prior + N(0, 0.1²)`. Larger noise is out of distribution
and the mean is wrong before the spread can widen. A model *trained* at σ = 0.3 is the
baseline again (composite 0.477, in view 0.33 / 0.76, z sd 1.35). A 3× larger noise scale
widens the spread on the tail, not on the near field the array share depends on.

## The tail speckle (`results/ml_cfm/tail/`)

**Signal or noise.** Both windows of `case_2023111718` (wind from 335°) cropped to the
record's cone (4322 cells). Tail band = non-zero cells below the window's own source-area
level. Body = the rest. Null: each window's band placed independently with the same occupancy
per shell (10 downwind × 5 crosswind quantile bins).

| level | part | \|mass\| in band | Jaccard / null | r of both-tail cells (n) | negatives in the same place (null) |
|---|---|---|---|---|---|
| 90% | tail | 12.7% | 0.644 / 0.570 = 1.13 | 0.04 (1382) | 0.19 (0.21) |
| 90% | body | 87.3% | 0.438 / 0.339 = 1.29 | 0.85 (255) | – |
| 99% | tail | 4.1% | 0.305 / 0.269 = 1.13 | 0.04 (506) | 0.19 (0.28) |
| 99% | body | 95.9% | 0.471 / 0.376 = 1.25 | 0.86 (704) | – |
| 99.9% | tail | 3.2% | 0.177 / 0.161 = 1.10 | 0.10 (224) | 0.19 (0.36) |

The body passes the positive control (values correlate at 0.85 after removing the shell
means). The tail does not at any level. Its cells overlap the other window's at 10–13% above
what the layout alone predicts, the values where both windows have tail are uncorrelated, and
the negative cells are in the same place less often than chance. Verdict: noise, from one
pair.

**The threshold.** Cells below the 99% source-area level (median 0.33% of the peak) are
zeroed. It keeps 857 of 1566 non-zero cells (val medians) and removes a median 2.51% of
|mass| (1.00% of the positive mass by construction plus the whole negative lobe, median
1.53%). On the pair it changes the floor to rel. L2 0.405 → 0.403, shape L1 0.630 → 0.604,
overlap80 0.507 → 0.537, array share 5.34 → 4.99 pp, centroid 51 → 40 m.

**Training on thresholded targets** (two seeds): on `val_mse_ref` against the raw target they
are worse (1.33e-4 vs 1.17e-4) because the speckle variance moves from "explained" to
"residual". On the production metrics against raw targets they straddle the baseline
(composite 0.506 and 0.480 against 0.474–0.567). Scored against thresholded targets and
thresholded Kljun, every model gains the same about 0.03 (0.489 → 0.454 for the baseline, the
part of the score the noise was costing every emulator), and the two clean-trained models are
no better than the raw-trained ones even there. The raw-target model already averages the
speckle out. Thresholding the targets is not applied. `--score-target sa99` exists for the
reference side.

## How many samples the mean needs (`results/ml_cfm/calib/samples/`)

S = 32 per seed was a budget number. 128 extra samples per seed (Euler 16) give 160 per seed
and 800 pooled. The S-sample mean is scored at every S with random-subset repeats, the law
`err(S) = a + b·S^−p` is fitted, and saturation is defined against the val noise floor (the
record-bootstrap sd of the composite), so S is chosen from the fit rather than from the lowest
val value.

| group | n | composite at S = 800 [bootstrap 95%] | floor (sd) | asymptote a [95%] | S_sat [95%] | S at half the floor |
|---|---|---|---|---|---|---|
| all | 235 | 0.473 [0.438, 0.523] | 0.022 | 0.471 [0.462, 0.480] | 21 [2, 64] | 83 |
| N/NE/NW | 71 | 0.550 [0.433, 0.684] | 0.064 | 0.545 [0.528, 0.560] | 2 [0, 15] | 7 |
| array in view | 42 | 0.761 [0.656, 0.863] | 0.052 | 0.761 [0.748, 0.776] | 2 [0, 13] | 8 |

Per seed the curve follows 1/S and is within 1% of its asymptote by S ≈ 30. Pooled over five
seeds it follows 1/√S with `S_sat` = 21 (upper band 64). S = 70 pooled (14 per seed) is the
setting read from the fit. The first write-up's S-dependence table was confounded. Its
subsets for S ≤ 32 all came from seed 0, the weakest seed (asymptote 0.535 against
0.470–0.483 for seeds 1–3). More samples do not improve the result.

At S = 70 (ten random subsets, mean ± sd): composite vs Kljun 0.481 ± 0.007 (all),
0.556 ± 0.013 (N/NE/NW), 0.769 ± 0.009 (in view). CFM/FNO 0.93 / 0.94 / 0.96. Array share
0.222 pp vs the FNO's 0.286 (p 0.001). Centroid 48.8 vs 55.1 m (p 9e-7).

## Cost

CRPS fine-tune: 8–9 s per epoch at batch 8, S = 2, 2 steps (about 4× a flow epoch). The
S = 4 / 4-step variant is 4× that again. Evaluator: about 2 min per variant at S = 64.

## Limitations

One two-window pair for the coherence test (n = 1, a train record). Temperature is fitted and
evaluated on val with 2 folds. n = 42 in the group that matters, so the in-view coverage of
0.81 has a ±0.09 sampling band of its own. The CRPS runs are single fine-tunes from one seed.
The "regressed" flag fires on `val_mse_ref` > 1.20e-4 for every CRPS and thresholded run while
their composites are inside the seed range. Both numbers are reported and the composite is the
production one.
