# The FNO emulator: build, exploration, tuning, and the val evaluation against Kljun

2026-09-02. Code in `ml/`, gates in `bin/test_ml_{data,model}.py`, every number below in
`results/ml/` (the file is named beside it). **The test split was never read**: see §8.

## 1. What was built

An FNO that predicts a **residual on Kljun** in asinh space, conditioned on Kljun's six
scalars by FiLM (`docs/ML_TARGETS.md`). Inputs are exactly the six scalars and the Kljun
raster; the only other channels are the receptor geometry (distance, its exponential, and
the X/Y coordinate planes), which are identical for every record and carry no information
about the case.

- **Data** (`ml/data.py`, `ml/features.py`): `corpus/corpus_cone.h5`, rows of one split read
  by fancy indexing; the split of every row re-derived from its own datetime with
  `lpdm.corpus.split_of`; the file's own train-only `norm/` used as read (`asinh(x/s)`,
  `s_kljun = 2.170e-5`, `s_target = 2.426e-5` m⁻²); stability fed as `z_m/L = 28.5·inv_L`
  (O(1), no derived statistic); the pad excluded from the loss by `meta/valid_mask`; the cone
  rebuilt from the file's own rule with `bin/mask_cone.py` (every val target is exactly zero
  outside it).
- **Model** (`ml/model.py`): lift → `depth` × [spectral conv on the lowest `modes`² of
  `rfft2` + local conv] → FiLM(γ, β from the six scalars) → GELU → projection with a
  zero-initialised last layer. **A zero residual reproduces Kljun to 7.5e-8 relative**
  (`bin/test_ml_model.py`).
- **Metrics** (`ml/metrics.py`): the production functions and nothing else —
  `bin/fig_corpus_pairs.crosswind_integrated` → `bin/stage5_footprint.fy_metrics` for the
  peak distance, `lpdm.footprint.FootprintGrid.metrics_map` for the centroid and 80% area,
  `lpdm.footprint.source_area_overlap` for the Jaccard overlap, the raster form of
  `lpdm/driver.py`'s `cover_share` for the array share, plus stage5's `l1_rel`,
  `seed_leakage.d_shape`, and the standard 2-D field metrics (relative L2, MAE, RMSE,
  Pearson, SSIM, PSNR). The same estimator is applied to the LES target, to Kljun and to the
  emulator, so a comparison is between fields, never between estimators.

Verified on the way: the raster-based array share agrees with the touchdown-based
`meta/array_share` at r = 0.9995 (median 0.09 pp), so the static array mask and the target
share cells (`bin/test_ml_data.py`); `h` equals `meta/zi_achieved_m` exactly, so the
integral asymptote `1 − z_m/z_i` is a function of the input scalar.

Environment: the `LESNet` conda env (torch 2.5.1 + CUDA 11.8, Python 3.11) on the RTX 4080,
with h5py 3.16 and optuna 4.9 added; spec in `ml/environment.yml`. No Docker image carries
torch; this is the one analysis that runs on the host.

## 2. Kljun on val, the number to beat (235 records; N/NE/NW 71)

| metric | Kljun median error vs LES | realisation floor | n behind the floor |
|---|---|---|---|
| peak distance | 30 m (mean 81 m) | 30 m, one cell | 2 runs × 2 cases |
| centroid | 92 m | 15–90 m half-vs-half convective; 46 m run-to-run | 4; 2 |
| 80% source-area overlap | 0.566 | 0.59 half-vs-half; 0.51 between the two validation windows | 1; 1 |
| array share, N/NE/NW | 3.84 pp | 5.3 pp between the two validation windows; 0.19 pp within-window SE | 1; ~1000 |
| integral | 0.140 | 1.2–1.44× run-to-run | 2 × 2 |
| 2-D shape L1 | 0.63 | 0.63 between the two validation windows; 0.41–0.92 in the record | 1; 1 each |

Kljun is at the floor on the peak distance and on the 2-D shape, and within it on the
overlap. The room to beat it is the array share, the centroid, and the integral, and the
array share is where the site signal lives. Floors from `results/les_realisation_spread.txt`,
`docs/results/FOURTH_PASS_RESULTS.md:547-560`, `docs/results/STAGE2-6_RESULTS_V2.md:296-305`
and `results/ml/eval/floor/pair_floor.json` (the two windows of `case_2023111718`, scored by
this evaluator after both were cropped to the cone).

## 3. Phase 1: exploration (38 short runs, `results/ml/phase1/DECISIONS.md`)

Baseline B0 with four seeds: val masked asinh-MSE 1.2138e-4 ± 5.8e-7 (0.5%); metric
composite (geometric mean over five metrics of median|err_FNO| / median|err_Kljun|)
0.68 ± 0.12. The loss is precise; the composite is not (its seed spread is 18% of its value,
driven by the 80% overlap). Spearman correlation between the two across runs +0.59, so the
tuning objective is the loss.

Settled, in units of the seed spread:

- **Required**: a local path beside the spectral one (spectral-only stays at its initial
  loss, z +268); constant spatial channels of some kind (no statics and no X/Y, z +6.2).
- **Harmful**: the peak-location term (z +8.6 at λ = 1, centroid ratio 1.95) and the
  integral term (z +45 against the target integral, diverged against the asymptote);
  changing the asinh knee (z +5 to +6).
- **No effect within the spread**: width 16–64, depth 2–6, modes 8–32, direct vs residual
  head, per-record vs the file's global normalisation, z-scored L vs z_m/L, ×3 northerly
  weighting, λ = 0.1 on either auxiliary term.
- **Statics**: removing them cost z +6.2, but the same maps rotated 90° (wrong geography,
  same statistics) recovered it (z +1.1), and plain X/Y coordinate planes with no statics
  recovered it too (z +1.6). Their role was a positional basis, not the site. Per the
  standing prior that constant channels can only be a bias, and with 231 of 235 val records
  sharing a seed with train (so an un-shared-subset test has n = 4), the statics were dropped
  and the X/Y planes kept.
- **Winner's curse**: the round-1 single-run "wins" (depth 6, conv3×3, modes 24, statics C,
  z −2 to −3) did not survive combination (B1, three seeds, z −0.3 ± 2.2). With 27
  one-factor comparisons the expected extreme of null draws is about z −2.

GPU: one process reports 100% utilisation and ~3 GB; four concurrent runs bought **1.1×**
throughput (30 runs in 2178 s at K = 4 against 82 s solo). The premise that a 128² FNO would
not saturate a 4080 is false at width 32 and batch 16; concurrency was kept for its
independence, not its speed.

## 4. Phase 2: Optuna (`results/ml/phase2/study_summary.md`)

Study `fno_v2` on SQLite, TPE (multivariate, 12 start-up trials), median pruning after a
20-epoch warm-up, three worker processes, resumable. Fixed by Phase 1: residual head, no
statics, distance + X/Y channels, the file's normalisation at knee 1, z_m/L, no auxiliary
terms. Searched: modes 12–32, width {16, 24, 32, 48}, depth 3–6, local {1×1, 3×3}, lr
log[2e-4, 3e-3], weight decay log[1e-6, 0.1], batch {8, 16, 32}, FiLM hidden {32, 64, 128},
dropout [0, 0.3]. 150 epochs, patience 25.

**120 trials in 3.4 h: 60 complete, 60 pruned, 0 failed.** Best #40: modes 32, width 48,
depth 3, conv3×3, lr 2.6e-4, weight decay 0.019, dropout 0.22, batch 8, FiLM 32 —
28.4 M parameters, val loss 1.1663e-4 (8 baseline seed-sd below B0), composite 0.536.
fANOVA importance: modes 0.29, dropout 0.19, lr 0.18, width 0.11. **The top ten are within
7e-7 of each other, about one seed-sd, so they are tied**; all ten share width 48, batch 8,
FiLM 32, conv3×3, dropout ≥ 0.17.

A first study (`fno_v1`) was abandoned after 16 of its trials were pruned at epoch 0: the
workers had loaded the study without the driver's pruner and fell back to Optuna's default
zero-warm-up `MedianPruner`. Fixed by passing sampler and pruner to every worker
(`ml/phase2_optuna.py`); the five completed configurations were re-queued into `fno_v2`.

## 5. The haze round (`results/ml/haze/summary.md`)

The early baseline's panels (`results/ml/eval/early_b0x/octant_examples.png`) showed the one
visible defect: a low-level haze over the whole domain, at about a thousandth of the peak,
which the LES target never has. The cause is the objective, not convergence — in asinh
space that haze is a per-cell error of ~1e-3, squared 1e-6, under 1% of the converged loss
of 1.2e-4 — so it was attacked with two levers on the Optuna best, three seeds for the
reference and the gate, two for the rest:

| variant | val loss z (base spread 3.6e-7) | 80% overlap (Kljun 0.566) | area-80 / LES | 2-D shape L1 (Kljun 0.63) | integral error |
|---|---|---|---|---|---|
| Optuna best, as is | 0 | 0.52 / 0.54 / 0.57 | 1.17–1.53 | 0.53–0.54 | 0.110–0.124 |
| + cone gate | −1.2 / +0.9 / +2.6 | 0.49 / 0.55 / 0.59 | 1.31–1.83 | 0.50–0.52 | 0.113–0.140 |
| **+ cone gate + L1 λ 0.03** | **−0.2 / +0.1** | **0.615 / 0.616** | **0.98 / 1.01** | **0.47 / 0.48** | 0.105 / 0.107 |
| + cone gate + L1 λ 0.3 | +1.2 / +1.4 | 0.615 / 0.619 | 0.95 / 0.97 | 0.467 / 0.469 | 0.096 / 0.097 |
| + L1 λ 0.03, no gate | +6.4 | 0.614 | 0.98 | 0.477 | 0.096 |
| + cone gate, knee 0.3 | +4.6 / +11.1 | 0.53 / 0.59 | 1.35–1.60 | 0.49–0.51 | 0.105–0.125 |

The gate is `pred = cone ⊙ (Kljun + residual)` in asinh space, where the cone is Kljun's own
σ_y and the wind direction — inputs the model already has, so nothing new enters. The L1
term is a masked mean absolute error in asinh space added to the MSE (`ml/losses.py`).

**The L1 term is what removes the haze**; the gate alone does not, and the gate with the
L1 term keeps the val loss inside the seed spread where the L1 term alone costs six sd.
With λ = 0.03 the 80% overlap moves from below Kljun to above it, the 80% area lands on
the LES's, and the 2-D shape L1 falls to 0.48 against Kljun's 0.63 — below the 0.63 that
separates two realisations of one case, which is what a conditional mean should do. λ = 0.3
buys a little more shape and integral for a loss and centroid cost and was not taken. The
final configuration is the Optuna best with the cone gate and λ_L1 = 0.03.

## 6. The final model and the val evaluation

Configuration: Optuna best (#40) + cone gate + λ_L1 0.03: modes 32, width 48, depth 3, conv3×3, lr 2.6e-4, weight decay 0.019, dropout 0.22, batch 8, FiLM hidden 32; 28.4 M parameters; 150 epochs, patience 25. Five seeds (`results/ml/final/seed*/`), each with a checkpoint and val predictions; the ensemble is the mean of the five physical-space predictions (`results/ml/eval/final_ensemble/`).

| seed | val loss (file space) | best epoch / run | composite | N/NE/NW composite | val/train loss ratio | wall (K = 3) |
|---|---|---|---|---|---|---|
| seed0 | 1.1748e-04 | 99 / 125 | 0.543 | 0.610 | 1.06 | 740 s |
| seed1 | 1.1752e-04 | 107 / 133 | 0.532 | 0.606 | 1.07 | 783 s |
| seed2 | 1.1641e-04 | 84 / 110 | 0.526 | 0.647 | 1.05 | 659 s |
| seed3 | 1.1664e-04 | 107 / 133 | 0.514 | 0.593 | 1.07 | 542 s |
| seed4 | 1.1765e-04 | 79 / 105 | 0.549 | 0.583 | 1.04 | 436 s |
| **ensemble of 5** | — | — | **0.526** | **0.597** | — | — |

The selection rule in `ml/final.py` picked the best single seed (seed3, composite 0.514) over the ensemble (0.526); the margin is inside the seed spread of the composite (sd 0.014 over the five seeds), so **the ensemble is the recommended model for the test evaluation** — it is the lower-variance estimate of the same conditional mean, and every table below is the ensemble. The train/val gap is 1.04–1.07× on the loss; on the composite the train side is 0.532, 0.538, 0.543, 0.511, 0.539 against the val values above.

**All val records** (235 records)

| metric | FNO median | Kljun median | ratio | FNO wins | Wilcoxon p | floor |
|---|---|---|---|---|---|---|
| peak_x | 0.000 m | 30.000 m | 0.00 | 64% | 2e-21 | 30 m (one cell) |
| centroid | 55.096 m | 91.855 m | 0.60 | 82% | 2e-30 | 15–90 m (convective halves); 46 m run-to-run |
| overlap80 | 0.622 | 0.566 | 0.87 (on 1−J) | 84% | 5e-28 | 0.59 halves; 0.51 two windows |
| array_share | 0.286 pp | 1.460 pp | 0.20 | 86% | 2e-26 | 5.3 pp two windows; 0.19 pp within-window SE |
| integral | 0.104 | 0.140 | 0.75 | 63% | 7e-08 | 1.2–1.44× run-to-run |

**N/NE/NW, where the array signal lives** (71 records)

| metric | FNO median | Kljun median | ratio | FNO wins | Wilcoxon p | floor |
|---|---|---|---|---|---|---|
| peak_x | 0.000 m | 30.000 m | 0.00 | 39% | 0.0003 | 30 m (one cell) |
| centroid | 57.979 m | 83.988 m | 0.69 | 72% | 7e-06 | 15–90 m (convective halves); 46 m run-to-run |
| overlap80 | 0.636 | 0.580 | 0.87 (on 1−J) | 82% | 9e-08 | 0.59 halves; 0.51 two windows |
| array_share | 1.255 pp | 3.839 pp | 0.33 | 83% | 1e-07 | 5.3 pp two windows; 0.19 pp within-window SE |
| integral | 0.104 | 0.160 | 0.65 | 72% | 2e-05 | 1.2–1.44× run-to-run |

**Array in view (LES share > 5%)** (42 records)

| metric | FNO median | Kljun median | ratio | FNO wins | Wilcoxon p | floor |
|---|---|---|---|---|---|---|
| peak_x | 0.000 m | 0.000 m | both 0 | 24% | 0.02 | 30 m (one cell) |
| centroid | 56.270 m | 65.412 m | 0.86 | 52% | 0.6 | 15–90 m (convective halves); 46 m run-to-run |
| overlap80 | 0.638 | 0.586 | 0.87 (on 1−J) | 76% | 0.001 | 0.59 halves; 0.51 two windows |
| array_share | 3.512 pp | 5.004 pp | 0.70 | 74% | 0.002 | 5.3 pp two windows; 0.19 pp within-window SE |
| integral | 0.113 | 0.181 | 0.62 | 71% | 9e-05 | 1.2–1.44× run-to-run |

**Does it win only where the array is absent? No.** It wins on every metric in the N/NE/NW group (array share 1.26 pp against 3.84 pp, p = 1e-7; overlap 0.636 against 0.580). The margin is smallest exactly where the signal is largest: on the 42 array-in-view records the centroid is a tie (p = 0.57) and the array-share gain shrinks to 3.5 pp against 5.0 pp (p = 0.002), because those are the records whose realisation floor is largest (5.3 pp between two windows of one run at a 20% share). By octant the composite is N 0.76, NE 0.56 (n = 7), E 0.76, SE 0.71, S 0.70, SW 0.42, W 0.39, NW 0.51 — below 1 everywhere (`octant_ratios.png`). By stability tercile the advantage grows toward the least unstable third (composite 0.68 → 0.55 → 0.39); by z_i tercile it is flat (0.52, 0.48, 0.55). The shared-seed breakout is uninformative (231 against 4 records).

**Shape and 2-D field metrics** (all 235 records; not in the selection composite):

| metric | FNO | Kljun | FNO wins | Wilcoxon p | floor |
|---|---|---|---|---|---|
| shape_l1_2d | 0.473 | 0.631 | 91% | 3e-35 | 0.63 two windows; 0.41–0.92 in the record |
| shape_1d | 0.071 | 0.141 | 93% | 5e-37 | 0.065 two windows |
| rel_l2 | 0.340 | 0.541 | 91% | 6e-37 | 0.40 two windows |
| rel_l2_T | 0.328 | 0.515 | 91% | 7e-37 | 0.39 two windows |
| mae_T | 0.001 | 0.002 | 92% | 5e-35 | 0.0019 two windows |
| rmse_T | 0.009 | 0.014 | 91% | 1e-36 | 0.013 two windows |
| pearson_T | 0.956 | 0.877 | 92% | 6e-35 | 0.92 two windows |
| ssim_T | 0.980 | 0.975 | 94% | 1e-34 | 0.980 two windows |
| psnr_T | 39.832 | 36.139 | 91% | 4e-37 | 40.1 dB two windows |

The FNO is closer to the LES target than a second realisation of the same case is, on the 2-D shape (0.47 against the 0.63 between the two validation windows), the 1-D shape (0.071 against 0.065, at the floor), and the relative L2 (0.34 against 0.40). That is what a conditional mean should do: it sits nearer any one sample than samples sit to each other. Pearson r rises from 0.877 to 0.956, SSIM from 0.975 to 0.980, PSNR from 36.1 to 39.8 dB.

**Integral against the asymptote 1 − z_m/z_i**: median |error| LES 0.153, FNO 0.116, Kljun 0.080. The FNO learns the LES's departure from the asymptote (the advection non-closure PROJECT_BRIEF.md names), so it is further from the asymptote than Kljun and closer to the LES; the integral is not scored against the asymptote, and the integral term that would have done so hurt (§3).

Figures in `results/ml/eval/final_ensemble/`: `octant_examples.png` (one typical record per octant: LES / Kljun / FNO raw / FNO cone-cropped on one log scale with the crosswind-integrated profiles), `north_examples.png` (the four strongest-array N records), `north_residuals.png` (LES − Kljun against FNO − Kljun), `octant_ratios.png`. With the gate inside the model the raw and cone-cropped FNO are identical (cone keep fraction 1.000).

## 7. Limitations, stated

1. The model receptor is 30 m (aerodynamic 28.5 m); the instrument is 10 m. The emulator
   predicts a footprint the tower does not measure.
2. Every corpus record is unstable (z/L from −1.76 to −0.002); the emulator is undefined for
   the stable ~44% of QC'd hours.
3. Only ~15% of records carry the array signal; aggregate metrics are dominated by cases
   where Kljun and the LES agree by construction, so the N/NE/NW breakout is the one that
   matters.
4. 231 of 235 val records share an LES seed with a train record. Every val number here is
   subject to that channel; the rotated-map control argues against memorised geography, but
   it cannot rule out seed leakage in general.
5. The target is a single realisation per case; the emulator regresses to the conditional
   mean and cannot reproduce Monte-Carlo texture. Per-cell metrics are reported beside floors
   for that reason and kept out of the selection composite.
6. Static surface channels, which the spec listed, were tested and rejected (§3): with this
   corpus they act only as a positional basis.

## 8. The test split was never read

- `ml/data.py:load_split` raises `TestSplitForbidden` for `split="test"` unless
  `allow_test=True` is passed; nothing under `ml/` passes it. The only place the flag can be
  set is the `--allow-test` option of `ml/evaluate.py`, which was never invoked.
- `results/ml/loader_audit.jsonl` records every corpus read (file, split, rows,
  `allow_test`). It contains no line that loaded `test`; its one `test` line is the refusal
  logged by `bin/test_ml_data.py`, with `n = 0`.
- `grep -rn allow_test ml/ bin/test_ml_*.py` shows the default `False` and the CLI option
  only.

To run the test evaluation deliberately:
`python -m ml.evaluate --ckpt results/ml/final/seed*/best.pt --split test --allow-test --tag test_final`
