# The emulator, day by day

The ML phase ran from 2026-09-02 to 2026-09-04, mostly unattended, with every decision logged
as it was made. This page is that log, condensed, with the decision record of the FNO's first
exploration round.

## 2026-09-02: the FNO (FNO_RESULT)

- Loader, features, model, losses, metrics and the two gates (`bin/test_ml_data.py`,
  `bin/test_ml_model.py`) built against `corpus_cone.h5`. The test split is refused by the
  loader and every read is logged.
- **Phase 1**, 38 short runs at K = 4, one factor at a time from a four-seed baseline, then a
  second round combining the round-1 winners with three seeds each. Decisions
  ([training](../emulator/training.md)): a local path is required. The auxiliary peak and
  integral terms hurt. The asinh knee stays at 1. Statics act only as a positional basis, and
  plain X/Y planes replace them. The architecture space is flat within the seed spread. The
  round-1 wins were the winner's curse.
- **Phase 2**, Optuna `fno_v2`: 120 trials in 3.4 h, 60 complete, 60 pruned, best #40 at
  1.1663e-4. A first study `fno_v1` was abandoned when its workers pruned 16 trials at epoch 0
  because they lacked the driver's pruner.
- **The haze round**: the cone gate plus λ_L1 = 0.03 removes the low-level haze and keeps the val
  loss inside the seed spread.
- **Final**: five seeds, ensemble composite 0.526 on val, 0.597 on N/NE/NW.

## 2026-09-02, 15:21–16:49 UTC: the CFM (CFM_RESULT)

- 15:21 plan approved. 15:25 a 60-epoch smoke run. 15:28 phase 1 starts (four runs, K = 4).
  15:29 code committed, pipeline running unattended.
- 15:59 phase 1 done. The winner is `v_s0.1_seed0` by the stated rule (the loss), with the
  x-prediction head's better composite recorded as a disagreement.
- 15:59–16:20 three final seeds. 16:22 the solver study (Euler 4–32 and Heun flat in the step
  count). 16:22 ALL DONE.
- 16:22–16:49 the extension: seeds 3 and 4 and a 1000-epoch run (stopped at epoch 105, the same
  place as the 500-epoch seeds).

## 2026-09-02, 22:12 – 2026-09-03, 00:23 UTC: calibration and the tail (CFM_CALIB_RESULT)

- 22:12 the follow-up approved. The coherence test of the tail speckle on the two-window pair.
- 22:20 batch A: CRPS fine-tunes (`crps_pure_ft`, `crps_blend_ft`, `crps_pure_ft_S4` at K = 3).
  22:22 the S = 4 run killed at epoch 0 for costing 4× per epoch. Relaunched at 22:32 beside
  batch B with 30 epochs.
- 22:31 batch B: two seeds on 99%-source-area-thresholded targets.
- 22:52 calibration at S = 64 over 11 variants. 23:14 done. The thresholded-target scoring.
- 23:15 batch C, triggered by the plan's rule (in-view cover90 < 0.81 after the fine-tunes): an
  array-share CRPS term at weight 5 (did not train), and CRPS from scratch (half the spread).
- 23:31 CALIB_DONE. 23:35 the second calibration pass. 23:36 write-up complete, gate PASS.
- 23:54 the sample-count study launched (128 extra samples per seed, Euler 16). 00:23 done:
  asymptote 0.471, `S_sat` 21 [2, 64] pooled, S = 70 chosen from the fit rather than the val
  argmin.

## 2026-09-03: the frozen recipe

- 20:38 UTC the recipe evaluated on val (`ml_cfm/final_recipe.py`): CFM 0.476 / FNO 0.545
  against Kljun. Seeds 0–3 for both models, 20 samples per CFM seed, physical-space mean, 99.5%
  source-area cut, LES positive-only and the temperature τ = 1.19 from the calibration study.

## 2026-09-04: metrics, figures and the test split

- `report_metrics` v1: losses stated, composite plus log-MSE, sliced W1, KL and MS-SSIM on val,
  the floor from the two-window pair. v2: an agreement composite of four bounded ratios, CRPS at
  τ = 1 vs 1.19. **Decision: τ → 1.0**, because the array-share CRPS is 2–5% worse at 1.19 and
  the reported metrics no longer include coverage. v3: production errors plus rel. L2, W1, KL,
  MS-SSIM over all records and eight octants, no composite, no CRPS. v4: RMSE for peak, centroid
  and integral, means for the rest, groups all + 4 sectors + 8 octants.
- Figures on val, five iterations (v1–v5): the showcase (four cases with a per-case metric
  table), the generative panel (sample cloud, mean contour, 90% band), sectors, distributions
  (ECDFs of per-record errors cut to the central 90%), and the domain map on Esri imagery with
  3DEP contours. Design choices settled along the way: Esri imagery under the footprint panels,
  full-saturation footprints, no wind arrow, the wind rose dropped from the sector panel.
- **The test run, on the user's go**: `test_predictions.py --allow-test` wrote FNO predictions
  and 20 CFM samples per seed (seeds 0–3, RNG 0). `report_metrics` and four figures on test.
  Results §8. Six audited test reads in total, all `allow_test`.
- Test figures re-picked with `--exclude`, then pinned: showcase `case_2025053115`,
  `case_2025020906`, `case_2025110421` (the S-sector case where the CFM leads on 7 of 8
  metrics. `--prefer-cfm-profile` was removed and the case pinned with `--case` instead),
  `case_2025042016`. Generative `case_2025031921`.
- The poster set: the five final figures at 600 dpi (showcase 18 × 12 in), scientific labels,
  no figure titles. 22 superseded figures deleted. Then the six figures of the first CFM run
  removed on request, leaving the five under `figures/poster/`.

The full per-run records behind this page (`results/ml/phase1/*/run.json`, `results/ml/phase2/trials/`,
`results/ml/haze/*/run.json`, `results/ml_cfm/TIMELINE.md`) were condensed here on 2026-09-04
and remain in the offline pre-cleanup archive of 2026-09-04. The summaries (`results/ml/phase1/summary.*`,
`results/ml/phase2/trials.tsv`, `results/ml/haze/summary.*`, `results/ml_cfm/*/summary.*`) are kept.
