# Prior-anchored flow matching beside the FNO: build, comparison, spread and calibration on val

2026-09-02. Code in `ml_cfm/` (nothing under `ml/` or `results/ml/final/` was modified; hashes
in `results/ml_cfm/ml_final_sha256_before.txt`), gate `bin/test_cfm.py`, every number below
in `results/ml_cfm/` (file named beside it). **The test split was never read** (§8).

## 1. What was built

Conditional flow matching on the straight-line (OT) path between the noised Kljun prior and
the LES target, in the file's asinh target space, with exactly the FNO's inputs plus `t`:

```
x_prior = cone ⊙ asinh(kljun / s_target)      d = x_les − x_prior      ε ~ N(0, σ²) on the cone
z_t     = x_prior + t·d + (1 − t)·ε            t ~ U(0, 1)
v       = d − ε                                the regression target (param = velocity)
sample  : z_0 = x_prior + ε,  z_{k+1} = z_k + Δt · v̂(z_k, t_k; scalars),  x̂_les = z_1
```

The coupling is fixed by the record (prior and target are already paired), so no minibatch
OT is needed. The network is a 2.95 M-parameter U-Net (widths 32/64/128/192, four levels,
GroupNorm, GELU) with FiLM on [sinusoidal(t), the six scalars] after every block and a
zero-initialised output conv, so the untrained model returns `x_prior + ε` and a σ = 0 sample
reproduces Kljun to 7.5e-8 relative (`bin/test_cfm.py`). Input channels: `z_t`, the Kljun
channel, distance (linear, exponential) and the X/Y planes. EMA 0.999, AdamW 1e-3 / 1e-4,
cosine, batch 16, grad-clip 5. Selection and early stopping on `val_mse_ref` of the 4-sample
ODE mean every 5 epochs, the same number the FNO was selected on. No adversarial term.

## 2. The comparison the spec asked for (`results/ml_cfm/phase1/summary.md`)

Four runs of 300 epochs at K = 4 (GPU 100%, 8.9 GB; ~30 min wall for all four):

| run | val_mse_ref (sample mean, S = 4) | z vs the seed pair | composite vs Kljun | best epoch |
|---|---|---|---|---|
| velocity, σ 0.1, seed 0 | 1.2164e-4 | −0.7 | 0.524 | 105 |
| velocity, σ 0.1, seed 1 | 1.2268e-4 | +0.7 | 0.512 | 90 |
| velocity, σ 0.3 | 1.2485e-4 | +3.7 | 0.512 | 110 |
| x-prediction, σ 0.1 | 1.2652e-4 | +6.0 | **0.462** | 125 |

**Velocity at σ = 0.1 was kept** by the stated rule (the loss). The x-prediction head lost on
the loss by six seed-sd but had the best metric composite of the four (0.462, z −6.9 on that
axis); with the two criteria disagreeing and the seed pair giving n = 2, the rule was
followed and the disagreement is recorded here rather than resolved.

**Solver study** (`phase1/solver_study.md`, seed 0, S = 8): Euler 4/8/16/32 and Heun 8/16
give the mean's `val_mse_ref` 1.31–1.36e-4 and composite 0.57–0.59 — flat in the step count.
Euler 16 was used for every sample below; Euler 4 would cost a quarter as much and changes
nothing measurable.

**Longer training does not help**: a 1000-epoch run of the winner stopped at epoch 105
(val 1.16e-4), the same place the 500-epoch seeds stopped (60–120). The t-augmentation
argument does not turn into a longer useful training curve at this data size.

## 3. The final model (`results/ml_cfm/final/`)

Five seeds, 500 epochs, patience 10 evaluations, 32 val samples each (160 pooled):

| seed | val_mse_ref | best epoch / run | val/train | composite (own 32-sample mean) |
|---|---|---|---|---|
| 0 | 1.28e-4 | 60 / 111 | 0.95 | 0.546 |
| 1 | 1.17e-4 | 90 / 141 | 0.99 | 0.488 |
| 2 | 1.17e-4 | 90 / 141 | 0.99 | 0.490 |
| 3 | 1.17e-4 | 95 / 146 | 1.00 | 0.474 |
| 4 | 1.17e-4 | 120 / 171 | 1.03 | 0.567 |
| **160-sample pooled mean** | | | | **0.492** |

Four of five seeds land on 1.17e-4, the FNO's number, with 2.95 M parameters against
28.4 M and no train/val gap (0.95–1.03 against the FNO's 1.04–1.07).

## 4. Sample mean vs LES, against Kljun and against the FNO (`results/ml_cfm/eval/final/eval.md`)

All 235 val records; median |error| against the LES target; win fraction and paired
Wilcoxon p for the CFM mean:

| metric | CFM mean | FNO ensemble | Kljun | CFM vs Kljun (wins, p) | CFM vs FNO (wins, p) | floor |
|---|---|---|---|---|---|---|
| peak_x [m] | 0 | 0 | 30 | 65%, 2e-21 | tie | 30 (one cell) |
| centroid [m] | 51.7 | 55.1 | 91.9 | 80%, 3e-27 | 63%, 4e-6 | 46–90 |
| overlap80 (Jaccard) | 0.616 | 0.622 | 0.566 | 80%, 3e-25 | 49%, 0.28 | 0.51–0.59 |
| array share [pp] | 0.247 | 0.286 | 1.460 | 85%, 7e-26 | 61%, 8e-4 | 5.3 (two windows) |
| integral | 0.097 | 0.104 | 0.140 | 65%, 2e-9 | 49%, 0.55 | 1.2–1.44× |
| shape L1 (2-D) | 0.471 | 0.473 | 0.631 | 92%, 5e-35 | 57%, 0.02 | 0.63 |
| shape (1-D) | 0.064 | 0.071 | 0.141 | 93%, 3e-37 | 68%, 7e-10 | 0.065 |
| rel L2 | 0.334 | 0.340 | 0.541 | 92%, 2e-37 | 54%, 0.09 | 0.40 |
| Pearson r (asinh) | 0.955 | 0.956 | 0.877 | 91%, 1e-34 | 60%, 0.03 | 0.92 |
| SSIM (asinh) | 0.981 | 0.980 | 0.975 | 86%, 3e-31 | 62%, 1e-4 | 0.980 |
| PSNR [dB] | 39.9 | 39.8 | 36.1 | 91%, 2e-37 | 55%, 0.03 | 40.1 |

**The CFM mean beats Kljun on every metric and essentially ties the FNO.** Against the FNO
it is significantly better on the centroid, array share, 1-D shape and SSIM and never
significantly worse; the composite is 0.492 against the FNO's 0.526 (ratio 0.948), below 1
in every octant except E (1.09, n = 10). By group (composite vs Kljun, CFM / FNO):
N/NE/NW 0.558 / 0.597; array in view 0.774 / 0.800; least-unstable tercile 0.331 / 0.392.
On the 42 array-in-view records CFM and FNO are indistinguishable on every metric
(p 0.18–0.89) and both beat Kljun on the array share (3.48 and 3.51 pp against 5.00 pp).

**The mean needs many samples.** Composite vs Kljun by sample count: S = 1 0.743, 4 0.580,
8 0.577, 32 0.546, 160 0.492; rel L2 0.430 → 0.334. A single sample is a realisation, not an
estimate of the mean, and the 32-sample mean is still 10% short of the 160-sample one.

**Integral vs the asymptote** 1 − z_m/z_i: LES 0.153, CFM 0.103, FNO 0.116, Kljun 0.080. Like
the FNO, the CFM learns part of the LES's departure from the asymptote.

## 5. The connected-component filter (`ml_cfm/ccfilter.py`)

Applied to Kljun, the FNO, the CFM mean and every sample; the LES scored unfiltered.

- **Rule A (keep components in descending |mass| until 99.9% of the original |mass| is
  accounted for)** is degenerate: on a gated field the non-zero support is the whole cone,
  so the support has to be taken at the record's own 99.9%-mass level, and at that level every
  component is needed to reach 99.9%. It removes exactly 0.100% of |mass| from every field
  (CFM, FNO, Kljun and the LES alike; 23 / 16 / 1 / 3 components, all kept) and changes no
  composite by more than 0.002.
- **Rule B (the level at which the LES target is single-connected)** is τ* = 0.40 of the
  peak (IQR 0.20–0.79): the LES targets are Monte-Carlo fields whose far tail is disconnected
  at any level below ~40% of the peak, and applying that level removes ~68–71% of the mass
  from every field including the LES itself. It is not a filter of artifacts.

The conclusion is that the isolated low-level cells the spec worried about carry no mass the
production metrics can see, and no threshold-free rule separates them from the LES's own
tail. Every table in `eval.md` is given unfiltered and Rule A filtered; they agree.

## 6. Spread and calibration — the result the FNO cannot give

Per record, 160 samples. **Between-sample spread** (medians): array-share sd 0.37 pp over all
records, **2.13 pp on N/NE/NW and 3.50 pp where the array is in view** (5–95% range 6.6 and
11.1 pp); integral sd 0.13; peak_x sd 18 m; between two samples of one record overlap80
0.564, 2-D shape L1 0.538, centroid distance 84 m, rel L2 0.386.

Against the floors: the two windows of `case_2023111718` differ by 5.34 pp in array share,
0.507 in overlap80, 0.63 in shape L1, 0.40 in rel L2 and 51 m in centroid; the two re-runs
differ by 4.58 and 0.67 pp and by 1.20–1.44× in integral. The sampled spread is the same
size as the realisation floor on the overlap, the shape, the L2 and the centroid, and about
two-thirds of it on the array share where the array is in view.

**Calibration on the 235 val LES targets, each treated as one more draw** (`calibration.png`):

| group | metric | n | PIT KS p | z sd | cover 50% | cover 90% |
|---|---|---|---|---|---|---|
| all | array share | 235 | 2e-5 | 1.21 | 0.50 | 0.88 |
| all | integral | 235 | 0.002 | 1.29 | 0.44 | 0.80 |
| all | peak_x | 235 | 6e-5 | 1.08 | 0.85 | 0.95 |
| all | centroid | 235 | 5e-8 | 0.69 | 0.63 | 0.97 |
| N/NE/NW | array share | 71 | 0.06 | 1.24 | 0.45 | 0.85 |
| array in view | array share | 42 | 0.03 | 1.77 | 0.33 | 0.74 |
| array in view | integral | 42 | 0.009 | 1.46 | 0.50 | 0.71 |

The 50% and 90% intervals on the array share cover 50% and 88% of the LES values over all
records and 45% / 85% on the northerly sectors: **usable error bars on the array share**,
the first this project has had. They are too narrow where the array is in view (33% / 74%,
z sd 1.77) and the integral's are too narrow everywhere (z sd 1.3–1.5). Peak and centroid
intervals are too wide (coverage 0.95–1.00, z sd 0.7–1.1). The PIT is not uniform anywhere
at n = 235 (KS p ≤ 0.06), so the spread is informative but not exactly calibrated; the
direction of every miss is stated above.

**Sharpness**: mean |∇| in asinh space LES 0.0012, CFM sample 0.0011, CFM mean 0.0009, FNO
0.0009, Kljun 0.0008; high-wavenumber power fraction LES 0.0037, sample 0.0033, mean 0.0021,
FNO 0.0017. Samples carry the LES's texture (`north_samples.png`: speckled tails, ragged
80% contours); the mean is as smooth as the FNO. Samples are not blurry, so no adversarial
term was added.

## 7. Cost

2.95 M parameters. Sampling on the RTX 4080 at batch 64, Euler 16 steps: 17–24 ms per record
per sample, so 32 samples cost ~0.6 s per record and the 160-sample val set 131 s per seed.
Euler 4 is four times cheaper at the same quality (§2). Training: 19–25 min per seed at K = 3.

## 8. Limitations, and the test split

1–5 as in `docs/results/FNO_RESULT.md` §7 (30 m receptor, no stable regime, ~15% of records
with the array signal, 231/235 val records sharing a seed with train, single-realisation
targets). In addition: the mean depends on S (§4); the spread is a property of one model
family and 160 samples, not a measured realisation distribution; the floors are n = 1 pair
and 2 re-runs; the filter of §5 is reported as degenerate rather than applied.

**The test split was never read.** `ml.data.load_split` raises `TestSplitForbidden` unless
`allow_test=True` is passed; nothing under `ml_cfm/` passes it (`grep -rnI allow_test
ml_cfm/` shows the `--allow-test` option of `ml_cfm/evaluate.py` only), and
`results/ml/loader_audit.jsonl` holds no `test` read with `n > 0`.

To evaluate on test deliberately, first write samples for the split (the trainer writes val
samples only), then `python -m ml_cfm.evaluate --seeds results/ml_cfm/final/seed{0..4}
--split test --allow-test --tag test_final`.
