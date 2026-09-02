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

TODO after `results/ml/final/final.json` and `results/ml/eval/final_ensemble/eval.md`.

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
