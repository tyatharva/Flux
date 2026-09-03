# CFM follow-up: calibration (CRPS, σ, temperature) and the tail speckle

2026-09-02, ~3 h wall after `docs/results/CFM_RESULT.md`. Code: `ml_cfm/crps.py`,
`ml_cfm/tailthresh.py`, `ml_cfm/calibrate.py`, extensions of `ml_cfm/train.py` (`loss`,
`init_from`, `target_thresh`, `select`) and `ml_cfm/infer.py` (σ override); drivers
`ml_cfm/run_calib.sh`, `ml_cfm/run_calib2.sh`; gate `bin/test_cfm.py` (26 checks, PASS).
Every number below is in `results/ml_cfm/calib/` or `results/ml_cfm/tail/` (file named
beside it). Nothing under `ml/`, `results/ml/final/` or the first CFM run
(`results/ml_cfm/{phase1,final,eval}/`) was modified (the gate diffs them). **The test split
was never read** (§7).

## 0. The answers in one place

1. **The array-in-view under-dispersion is a mis-scaling, not a mis-shaping — mostly.** A single
   global temperature τ = 1.19 on the sample spread (fitted on val, cross-fitted by record
   parity) brings the array-share coverage into the ±2 sd band on all three groups: all
   0.56 / 0.92, N-NE-NW 0.54 / 0.93, array in view 0.40 / 0.81 (nominal 0.50 / 0.90). The
   in-view group lands on the band's lower edge with z sd still 1.31, so a residual
   under-dispersion of ~30% in that group survives a global rescale; a group-specific τ
   (1.38 in view / 1.14 elsewhere, predictor-side grouping) does not do better once
   cross-fitted, because the in-view τ is unstable across folds (1.02 vs 1.56 on n ≈ 20).
2. **CRPS training does not fix it.** Fine-tuning the flow on the pixelwise fair CRPS (alone,
   blended 1:1 with the flow loss, or with S = 4 / 4 steps) leaves the array-share coverage
   where it was (in view 0.31 / 0.76, 0.29 / 0.71, 0.26 / 0.69 against 0.31 / 0.74) and the
   spread-skill ratio at 0.67–0.79 against 0.81. What CRPS training does buy is a **2-step
   sampler as good as the 16-step ODE**: field CRPS 0.00388 at 1.0 ms per record per sample
   against 0.00388 at 16.2 ms, where the flow model sampled with 2 steps gives 0.00439 and a
   collapsed spread (2.19 pp vs 3.88 pp in view). The mean's MSE rises 9% (1.17 → 1.27e-4) while
   its production-metric composite is unchanged or slightly better (0.483 / 0.477 vs 0.489,
   Wilcoxon p 0.02). It overfits after ~20 epochs at this data size. The two conditional
   runs (§3) confirm it from the other side: an array-share term in the objective does not
   train at all, and CRPS from scratch produces a good mean with **half** the flow's spread
   (in view 0.17 / 0.40).
3. **Raising σ at inference destroys the mean** (composite 1.26 at σ 0.2, 5.6 at σ 0.5: the
   network is off-distribution) and a model *trained* at σ 0.3 is the baseline again
   (0.33 / 0.76 in view, composite 0.477).
4. **The tail speckle is sampling noise** (one two-window pair, so n = 1): the tail bands
   below the 99% source-area level overlap at 1.13× the layout-conditioned null and the
   values of cells that are tail in both windows correlate at r = 0.04, where the body gives
   1.25× and r = 0.86.
5. **Thresholding the targets at their own 99% source-area level removes a median 2.5% of
   |mass| (all of it the negative lobe plus the 1% positive tail), lowers the realisation
   floor modestly (shape L1 0.630 → 0.604, overlap80 0.507 → 0.537, centroid 51 → 40 m), and
   trains a model whose mean is inside the baseline seed range on the raw targets**
   (composite 0.506 / 0.480 vs 0.474–0.567) — no measurable gain, no loss; §5 has the scoring
   against thresholded targets and Kljun.

## 1. What was measured against what

Groups (val, n): all 235, N/NE/NW 71, array in view (LES share > 5%) 42. Coverage bands are
±2 binomial sd around nominal: cover50 [0.43, 0.57] / [0.38, 0.62] / [0.35, 0.65] and cover90
[0.86, 0.94] / [0.83, 0.97] / [0.81, 0.99]. "The mean has not regressed" = composite vs Kljun
of the 64-sample mean inside the five final seeds' range [0.474, 0.567] and `val_mse_ref`
≤ 1.20e-4. Every variant is sampled with S = 64 on the same seed; the LES target is scored
as one more draw (PIT, z, coverage) and the fair CRPS of the array share [pp] and integral
are per record, mean over the group; spread/skill = rms sample sd × √((S+1)/S) / rmse of the
sample mean (1 = calibrated in the second moment). The field CRPS is the pixelwise fair
CRPS in the training (asinh) space over the cone cells, median over records.
`results/ml_cfm/calib/final/calib.{md,json}`, figures `coverage.png`, `pit.png`.

## 2. CRPS as a training objective (`ml_cfm/crps.py`, runs in `results/ml_cfm/calib/runs/`)

The fair sample estimator, `mean_s|x_s − y| − Σ_{s<s'}|x_s − x_s'| / (S(S−1))`, evaluated per
cell in asinh space over the cone∩valid cells, on S samples drawn **through the model's own
Euler sampler inside the loss** (gradients through every step, one checkpoint per step;
`bin/test_cfm.py` checks the sorted form against the pairwise one, the point-mass identity,
and unbiasedness at S = 2). Fine-tunes start from `final/seed1` (val 1.17e-4, composite
0.488), AdamW 2e-4, batch 8, EMA, selection on the val field CRPS (S = 8, fixed draws).

| run | loss | S / steps | best epoch / run | val CRPS (own steps) | val_mse_ref | composite (64-sample mean) | in-view cover50 / 90 | z sd | spread/skill |
|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 (baseline, Euler 16) | fm | – / 16 | 90 / 141 | – | 1.172e-4 | 0.489 | 0.31 / 0.74 | 1.55 | 0.81 |
| fm_seed1, Euler 2 | fm | – / 2 | same weights | – | – | 0.491 | 0.21 / 0.45 | 2.28 | 0.56 |
| crps_pure_ft | crps | 2 / 2 | 20 / 51 | 0.00426 | 1.273e-4 | 0.483 | 0.31 / 0.76 | 1.48 | 0.77 |
| crps_blend_ft (fm + 0.16·crps) | fm+crps | 2 / 2 | 20 / 51 | 0.00434 | 1.248e-4 | 0.477 | 0.29 / 0.71 | 1.55 | 0.77 |
| crps_pure_ft_S4 | crps | 4 / 4 | 10 / 30 | 0.00420 | 1.270e-4 | 0.481 | 0.26 / 0.69 | 1.70 | 0.67 |

λ = 0.16 makes the two terms equal at the checkpoint (FM 7e-4, CRPS 4.4e-3, measured in the
smoke run). All three CRPS runs reach their best val CRPS by epoch 10–20 and then overfit
(train CRPS keeps falling, val rises; val/train MSE gap 1.22–1.26 against 0.99 before), so the
t-augmentation that keeps the flow loss honest does not carry over to an objective that
samples through the model.

**Reading.** The CRPS gain is real relative to the sampler it trains — the 2-step flow model
has field CRPS 0.00439 and the CRPS-tuned 2-step model 0.00388, the 16-step value — but the
scalar the calibration is about, the array share, is untouched: CRPS 2.75 pp in view before
and 2.75 / 2.75 / 2.80 after; the spread-skill ratio does not rise. The pixelwise objective
is dominated by the ~4000 tail cells of the cone where the spread is already right, and the
44 array cells carry no weight in it. §3 (conditional runs) adds the array share itself to
the objective and trains CRPS from scratch.

## 3. The conditional runs (`results/ml_cfm/calib/final2/`)

Triggered by the rule in the plan (in-view cover90 < 0.81 after the fine-tunes).

| run | loss | best epoch / run | val CRPS | val_mse_ref | composite (64-sample mean) | in view cover50 / 90 | z sd | spread/skill | in-view sample sd [pp] | field CRPS |
|---|---|---|---|---|---|---|---|---|---|---|
| fm_seed1 (baseline, Euler 16) | fm | 90 / 141 | – | 1.172e-4 | 0.489 | 0.31 / 0.74 | 1.55 | 0.81 | 3.88 | 0.00388 |
| crps_share_ft (field CRPS + 5 × array-share CRPS, from seed1) | crps | **0 / 21** | 0.00483 | 1.163e-4 | 0.486 | 0.21 / 0.45 | 2.52 | 0.53 | 2.21 | 0.00438 |
| crps_pure_scratch (CRPS alone, from scratch, 2 steps) | crps | 45 / 76 | 0.00421 | 1.299e-4 | 0.477 | **0.17 / 0.40** | 2.88 | 0.44 | 2.00 | 0.00392 |
| crps_pure_ft (§2, for reference) | crps | 20 / 51 | 0.00426 | 1.273e-4 | 0.483 | 0.31 / 0.76 | 1.48 | 0.77 | 3.72 | 0.00388 |

- **The share-weighted objective did not train.** With the array-share CRPS at weight 5 the
  val field CRPS rose from the first evaluation (0.00483 → 0.0054 by epoch 10) and the
  training loss stayed at 0.04–0.05, so the selection (on val field CRPS) kept epoch 0 —
  the baseline weights sampled with 2 steps, which is the collapsed-spread row of §2. A
  scalar term on 44 cells, with S = 2, is too noisy a gradient to move the network; a
  smaller weight would be a smaller version of the same thing.
- **CRPS from scratch gives a good mean and half the spread.** The 2-step generator trained
  on CRPS alone reaches the baseline's field CRPS (0.00392 vs 0.00388) and composite
  (0.477) in 45 epochs — the objective works as a regression-with-spread loss — but its
  array-share spread is 2.0 pp where the flow's is 3.9 pp and the LES needs ~5, so its
  coverage is the worst of the study (0.17 / 0.40 in view, spread/skill 0.44). The fine-tune
  keeps the flow's spread because it starts from it; trained from nothing, the pixelwise
  CRPS optimum at S = 2 is nearly deterministic on the near field.

Neither conditional run changes the answer of §2: CRPS is not the lever for this defect.
`results/ml_cfm/calib/final2/calib.{md,json}`, `coverage.png`, `pit.png`.


## 4. The cheap fixes

**Raising σ.** Sampling `final/seed1` with σ_sample > σ_train (Euler 16, S = 64):

| σ_sample | composite vs Kljun | in-view cover50 / 90 | centroid [m] |
|---|---|---|---|
| 0.1 (trained) | 0.489 | 0.31 / 0.74 | 49 |
| 0.2 | 1.256 | 0.33 / 0.64 | 184 |
| 0.3 | 2.349 | 0.02 / 0.24 | 391 |
| 0.5 | 5.560 | 0.00 / 0.07 | 515 |

The network has only seen `z_0 = x_prior + N(0, 0.1²)`; larger noise is out of distribution
and the mean is wrong before the spread can widen. A model trained at σ = 0.3
(`phase1/v_s0.3`, 64 samples) is the baseline again: composite 0.477, in-view 0.33 / 0.76,
z sd 1.35, spread/skill 0.96 in view (0.99 on all records) with cover90 still 0.76 and z sd
1.35 — a 3× larger noise scale widens the spread on the tail, not on the near field the
array share depends on.

**Post-hoc temperature** on the baseline's asinh-space samples, `T_s′ = T̄ + τ(T_s − T̄)`, τ
fitted so the array-share z sd is 1:

| variant | τ | all c50 / c90 (z sd) | N-NE-NW c50 / c90 (z sd) | in view c50 / c90 (z sd) | in-band | integral in view c50 / c90 |
|---|---|---|---|---|---|---|
| none | 1 | 0.47 / 0.88 (1.19) | 0.44 / 0.85 (1.26) | 0.31 / 0.74 (1.55) | ✓✓ ✓✓ ✗✗ | 0.38 / 0.69 |
| global, in-sample | 1.19 | 0.56 / 0.92 (1.00) | 0.55 / 0.93 (1.05) | 0.43 / 0.81 (1.30) | ✓✓ ✓✓ ✓✓ | 0.45 / 0.74 |
| grouped, in-sample | 1.38 in view / 1.14 | 0.56 / 0.92 (1.00) | 0.56 / 0.94 (0.96) | 0.45 / 0.83 (1.18) | ✓✓ ✓✓ ✓✓ | 0.50 / 0.81 |
| **global, cross-fitted** | 1.15 / 1.23 by fold | 0.56 / 0.92 (1.01) | 0.54 / 0.93 (1.07) | **0.40 / 0.81 (1.31)** | ✓✓ ✓✓ ✓✓ | 0.48 / 0.74 |
| grouped, cross-fitted | 1.02–1.56 in view | 0.54 / 0.91 (1.06) | 0.52 / 0.93 (1.13) | 0.40 / 0.81 (1.39) | ✓✓ ✓✓ ✓✓ | 0.48 / 0.71 |

The grouping is predictor-side (the CFM's own mean share > 5%, 40 records), because the LES
share is unknown at inference. Temperature does not change the sample mean in asinh space,
and the physical mean moves by < 0.01 in composite. **Verdict: a global τ = 1.19 is enough
to pass the bands, so the flow is well-shaped and ~20% under-dispersed overall; the
in-view group is ~30% under-dispersed even after the global rescale (z sd 1.31, cover90 on
the band edge), and the data cannot resolve a separate in-view τ (fold estimates 1.02 and
1.56).** The integral stays under-dispersed after any array-share-fitted τ (z sd 1.2–1.3).

## 5. The tail speckle (`results/ml_cfm/tail/`)

### 5.1 Signal or noise (`coherence.md`, `tail_pair.png`; n = 1 pair, a train record)

Both windows of `case_2023111718` (wdir 335°) cropped to the record's cone (4322 cells).
Tail band = non-zero cells below the window's own source-area level; body = the rest. Null:
each window's band placed independently with the same occupancy per shell (10 downwind ×
5 crosswind quantile bins), so the null already knows that tails live at the periphery.

| level (of the positive mass) | part | |mass| in band | Jaccard / null (perm 95%) | r of both-tail cells (n) | negatives in the same place (null) |
|---|---|---|---|---|---|
| 90% | tail | 12.7% | 0.644 / 0.570 = **1.13** ([0.561, 0.581]) | **0.04** (1382) | 0.19 (0.21) |
| 90% | body | 87.3% | 0.438 / 0.339 = 1.29 | **0.85** (255) | – |
| 99% | tail | 4.1% | 0.305 / 0.269 = **1.13** ([0.254, 0.284]) | **0.04** (506) | 0.19 (0.28) |
| 99% | body | 95.9% | 0.471 / 0.376 = 1.25 | **0.86** (704) | – |
| 99.9% | tail | 3.2% | 0.177 / 0.161 = 1.10 | 0.10 (224) | 0.19 (0.36) |

The body passes the positive control (values correlate at 0.85 after removing the shell
means); the tail does not at any level: its cells overlap the other window's at 10–13%
above what the layout alone predicts (outside the permutation band, so the excess is real
but small), the values where both windows have tail are uncorrelated, and the negative
cells are in the same place *less* often than chance. **Verdict: noise.** With one pair this
is one measurement, not a distribution; the same script runs on any further pair.

### 5.2 The threshold and what it removes (`threshold_corpus.json`)

Per record, cells below the 99% source-area level are zeroed (`ml_cfm/tailthresh.py:
threshold_sa`): the level is 0.33% of the peak (median; IQR 0.25–0.46%), the rule keeps 857
of 1566 non-zero cells (val medians) and removes a median 2.51% of |mass| (IQR 1.9–3.3%, max
10.6%) — 1.00% of the positive mass by construction plus the whole negative lobe (median
1.53% of |mass|). Train is the same (2.53%). By octant 2.0% (N) to 4.8% (E). Applied to
Kljun the same rule removes 0.998% (its tail is smooth and positive).

On the pair, thresholding both windows changes the floor to: rel L2 0.405 → 0.403, shape L1
0.630 → 0.604, overlap80 0.507 → 0.537, array share 5.34 → 4.99 pp, centroid 51 → 40 m,
integral difference 0.0007 → 0.0099 (the two windows lose 4.06% and 3.05%). A modest
tightening: most of the between-window difference is in the body.

### 5.3 Training on thresholded targets (`results/ml_cfm/tail/runs/`)

Two seeds of the final FM configuration with `target_thresh = sa99` (300 epochs, early stop
at 155 / 145, best 105 / 95). Selection stayed on `val_mse_ref` against the **raw** target so
the runs are comparable with the baseline; on that number they are worse (1.33e-4 vs
1.17e-4) because the model now predicts a clean tail against a speckled target — the
speckle variance moves from "explained" to "residual". On the production metrics of the
64-sample mean against the raw LES targets (`calib/final/calib.md`):

| model | composite vs Kljun, all | in view | vs baseline (p) | array share [pp] | centroid [m] | rel L2 | in-view cover50 / 90 |
|---|---|---|---|---|---|---|---|
| fm_seed1 (baseline) | 0.489 | 0.744 | – | 0.236 | 49.0 | 0.337 | 0.31 / 0.74 |
| thresh_seed0 | 0.506 | 0.770 | 1.027 (2e-4) | 0.260 | 47.8 | 0.341 | 0.33 / 0.79 |
| thresh_seed1 | 0.480 | 0.741 | 0.985 (6e-6) | 0.223 | 48.0 | 0.339 | 0.33 / 0.69 |

The two seeds straddle the baseline (the five raw-target seeds span 0.474–0.567), so the
thresholded targets neither help nor hurt the mean on raw targets, and the calibration is
the baseline's.

**Scored against the thresholded targets and thresholded Kljun** (`calib/final_vs_thresholded/
calib.md`; the LES loses a median 2.51% of |mass|, Kljun 1.00%), so the comparison is
fair to a model trained on clean targets:

| model (trained on) | composite vs Kljun_thr, all | in view | vs fm_seed1 (p) | field CRPS | in-view cover50 / 90 |
|---|---|---|---|---|---|
| fm_seed1 (raw) | 0.454 | 0.754 | – | 0.00373 | 0.31 / 0.74 |
| fm_seed2 (raw) | 0.460 | 0.738 | 1.010 (2e-5) | 0.00372 | 0.31 / 0.76 |
| thresh_seed0 (sa99) | 0.469 | 0.792 | 1.025 (7e-5) | 0.00380 | 0.29 / 0.79 |
| thresh_seed1 (sa99) | 0.460 | 0.738 | 1.010 (2e-6) | 0.00372 | 0.33 / 0.71 |

Every model, whatever it was trained on, gains the same ~0.03 in composite when the
speckle is removed from the reference (0.489 → 0.454 for the baseline: that 0.03 is the
part of the score the noise was costing every emulator), and the two models trained on
clean targets are no better than the two trained on raw ones **even on the clean targets**.
**The raw-target model already averages the speckle out** — it is unpredictable from the
inputs, so an MSE-trained mean ignores it — and the noise in the target costs the model
nothing that removing it from training would recover. Thresholding the targets is therefore
not applied; the two runs stay as the measurement. What the threshold is good for is the
reference side: scoring against the 99% source area removes a noise floor of ~0.03 in
composite that no model can beat, and `--score-target sa99` exists for that.


## 6. Cost

CRPS fine-tune: 8–9 s per epoch alone at batch 8, S = 2, 2 steps (≈4× a flow epoch); the
S = 4 / 4-step variant 4× that again. A CRPS-tuned model samples in 1.0 ms per record per
sample (Euler 2) against 16.2 ms for the 16-step flow; the two have the same field CRPS.
Evaluator: ~2 min per variant at S = 64 (sampling + 15,040 per-sample metric evaluations).

## 6a. How many samples the mean needs (`results/ml_cfm/calib/samples/`)

S = 32 per seed was a budget number, never measured. 128 extra samples per seed (Euler 16,
8 ms per record per sample) give 160 per seed and 800 pooled; the S-sample mean is scored
against the LES at every S with random-subset repeats, the law `err(S) = a + b S^-p` is
fitted, and saturation is defined against the val noise floor — the record-bootstrap sd of
the composite — so S is chosen from the fit, not from the lowest val value
(`sample_count.md`, `sample_saturation.md`, `sample_count.png`).

| group | n | composite at S = 800 [record-bootstrap 95%] | floor (sd) | asymptote a [95%] | S_sat [95%] | S at half the floor |
|---|---|---|---|---|---|---|
| all | 235 | 0.473 [0.438, 0.523] | 0.022 | 0.471 [0.462, 0.480] | 21 [2, 64] | 83 |
| N/NE/NW | 71 | 0.550 [0.433, 0.684] | 0.064 | 0.545 [0.528, 0.560] | 2 [0, 15] | 7 |
| array in view | 42 | 0.761 [0.656, 0.863] | 0.052 | 0.761 [0.748, 0.776] | 2 [0, 13] | 8 |

Per seed the curve follows 1/S and is within 1% of its asymptote by S ≈ 30, so the 32 per
seed already in use were at saturation for a single seed; pooled over five seeds it
follows 1/√S with S_sat = 21 (upper band 64). **S = 70 pooled (14 per seed) is the setting
read from the fit**; at S = 800 the remaining fitted improvement is 0.16 of the floor. The
first write-up's "S-dependence" table (0.743 → 0.492 from S = 1 to 160) was confounded:
its subsets for S ≤ 32 all came from seed 0, the weakest seed (asymptote 0.535 against
0.470–0.483 for seeds 1–3), so most of that fall was seed quality, not sample count. More
samples are not a lever: the asymptote is 0.471 and the 160-sample value in §4 of
`CFM_RESULT.md` (0.492) is one floor from it.

Headline at S = 70 (ten random 70-subsets, mean ± sd; paired tests from the first):
composite vs Kljun 0.481 ± 0.007 (all), 0.556 ± 0.013 (N/NE/NW), 0.769 ± 0.009 (in view);
CFM/FNO 0.93 / 0.94 / 0.96; array share 0.222 pp vs the FNO's 0.286 (p 0.001); centroid
48.8 vs 55.1 m (p 9e-7). The conclusions of `CFM_RESULT.md` §4 stand at the fitted S.

## 7. Limitations, and the test split

- One two-window pair for the coherence test (n = 1, a train record); the corpus-level
  statement is that the rule removes 2.5% of |mass| per record, not that every record's tail
  is noise.
- Temperature is fitted and evaluated on val (cross-fitted by record parity, 2 folds); n = 42
  in the group that matters, so the in-view coverage of 0.81 has a ±0.09 sampling band of its
  own.
- The CRPS runs are single fine-tunes from one seed; their differences to the baseline are
  inside the seed spread of the mean and were read as such.
- Rule-5 "regressed" flags in `calib.md` fire on `val_mse_ref` > 1.20e-4 for every CRPS and
  thresholded run while their composites are inside the seed range; both numbers are
  reported and the composite is the production one.

**The test split was never read.** `ml.data.load_split` refuses it without `allow_test=True`;
`grep -rnI allow_test ml_cfm/` shows only the `--allow-test` option of `ml_cfm/evaluate.py`
(`ml_cfm/calibrate.py` has no such option and loads `val` and `train` only);
`results/ml/loader_audit.jsonl` holds no `test` read with `n > 0`.
