# Training and selection

How the FNO's configuration was found, in three rounds, and how the CFM was trained beside
it. Every comparison is quoted in units of the baseline's seed spread, and "settled" means
beyond it. The records are `results/ml/phase1/{summary.*,DECISIONS.md}`, `results/ml/phase2/`,
`results/ml/haze/`, `results/ml/final/`, `results/ml_cfm/phase1/` and `results/ml_cfm/final/`.

## Phase 1: one factor at a time (38 short runs)

Baseline B0: residual head, width 32, modes 16, depth 4, spectral + 1×1 local path, channels =
Kljun + distance (linear, exp 300 m) + static maps (topo, log z0, array, water), the file's
global asinh scales, stability as `z_m/L`, MSE only, AdamW lr 1e-3, weight decay 1e-4,
batch 16, 80 epochs, patience 20. 4.22 M parameters. Four seeds:

| quantity | mean | sd |
|---|---|---|
| val masked asinh-MSE (`val_mse_ref`) | 1.2138e-4 | 5.8e-7 (0.5%) |
| composite (5-metric ratio to Kljun, < 1 wins) | 0.679 | 0.119 |
| composite, N/NE/NW | 0.772 | 0.140 |

The loss is precise enough that |z| ≥ 2 is real. The composite is not. Its seed spread is 18%
of its value, driven by the 80% source-area overlap, whose val median swung 0.37–0.60 across
four seeds of one configuration. Spearman correlation between loss and composite across runs
is +0.59, above the 0.5 threshold set in advance, so **the tuning objective is the loss** and
the composite is recorded as an attribute.

Round 1 (z on `val_mse_ref`, negative = better):

| factor | levels | z | decision |
|---|---|---|---|
| local path | none / 1×1 / 3×3 | +268 / 0 / −2.7 | required. Spectral-only barely trains (val loss stays at its initial 2.77e-4) |
| depth | 2 / 4 / 6 | +1.2 / 0 / −3.0 | search 3–6 |
| modes | 8 / 12 / 16 / 24 / 32 | +1.7 / +0.7 / 0 / −2.0 / −0.9 | search 12–32 |
| width | 16 / 32 / 64 | −0.3 / 0 / +1.0 | no effect. 64 costs 2.3× the time |
| head | residual / direct | 0 / +0.7 | residual kept (zero predicts Kljun exactly) |
| raster norm | file / per-record | 0 / +0.6 | file kept |
| asinh knee | 0.3 / 1 / 3 | +6.1 / 0 / +4.7 | knee 1 kept |
| distance channels | none / lin+exp / +XY | −0.9 / 0 / −1.6 | no effect on their own |
| stability scalar | z-scored L / `z_m/L` | +0.3 / 0 | `z_m/L` kept |
| peak-location term | λ 0 / 0.1 / 1 | 0 / +0.6 / +8.6 | hurts at λ = 1 (centroid ratio 1.95). Dropped |
| integral term | λ 0.1 / 1, vs target or asymptote | −0.8, −0.2 / +45, +268 | harmful at 1 (the asymptote run diverged at epoch 8). Dropped |
| northerly ×3 weighting | off / on | 0 / +0.4 | dropped |

The λ scale: the masked MSE converges near 1e-4 while both auxiliary terms are O(0.1), so
they have a fixed 1e-3 factor (`ml/losses.py:AUX_SCALE`), and λ = 1 means "about one
converged MSE". At that weight both hurt.

**Statics: the expected null, and what the control says.** Static surface maps are constant
across records and can only be a bias term, so the expected result was no effect. Instead,
removing them cost z +6.2. But the same maps rotated 90° (wrong geography, same statistics)
recovered it (z +1.1), and in round 2 plain X/Y coordinate planes with no statics recovered
it too (z +1.6). Their role was a positional basis that breaks the translation symmetry of the
spectral layers, not the site. With 231 of 235 val records sharing a seed with train, an
un-shared-subset test has n = 4 and cannot decide anything. The rotated-map control is the
test that can. The statics were dropped and the X/Y planes kept, which makes the emulator's
inputs exactly Kljun's six scalars, the Kljun raster and the receptor geometry.

**Winner's curse.** The round-1 single-run wins (depth 6, 3×3, modes 24, statics C, z −2 to
−3) did not survive combination. B1 (three seeds) landed at z −0.3 ± 2.2 against B0. With 27
one-factor comparisons the expected extreme of null draws is about z −2. The architecture
space is flat within the seed spread at this training length. What is real is the large-|z|
set.

GPU: one process reports 100% utilisation and about 3 GB. Four concurrent runs gave 1.1×
throughput (30 runs in 2178 s at K = 4 against 82 s solo). Concurrency was kept for its
independence, not its speed. Phase 2 used K = 3 for memory headroom.

## Phase 2: Optuna

Study `fno_v2` on SQLite, TPE (multivariate, 12 start-up trials), median pruning after a
20-epoch warm-up, three worker processes, resumable. Fixed by Phase 1: residual head, no
statics, distance + X/Y channels, the file's normalisation at knee 1, `z_m/L`, no auxiliary
terms. Searched: modes 12–32, width {16, 24, 32, 48}, depth 3–6, local {1×1, 3×3}, lr
log[2e-4, 3e-3], weight decay log[1e-6, 0.1], batch {8, 16, 32}, FiLM hidden {32, 64, 128},
dropout [0, 0.3]. 150 epochs, patience 25.

**120 trials in 3.4 h: 60 complete, 60 pruned, 0 failed.** Best #40: modes 32, width 48,
depth 3, 3×3, lr 2.6e-4, weight decay 0.019, dropout 0.22, batch 8, FiLM 32. 28.4 M
parameters. Val loss 1.1663e-4 (8 baseline seed-sd below B0). Composite 0.536. fANOVA
importance: modes 0.29, dropout 0.19, lr 0.18, width 0.11, FiLM hidden 0.09, batch 0.07,
weight decay 0.03, depth 0.02, local 0.02. The top ten are within 7e-7 of each other, about
one seed-sd, so they are tied. All ten share width 48, batch 8, FiLM 32, 3×3, dropout ≥ 0.17.

A first study (`fno_v1`) was abandoned after 16 of its trials were pruned at epoch 0. The
workers had loaded the study without the driver's pruner and fell back to Optuna's default
zero-warm-up `MedianPruner`. Fixed by passing sampler and pruner to every worker
(`ml/phase2_optuna.py`). The five completed configurations were re-queued.

## The haze round

The early baseline's panels showed one visible defect: a low-level haze over the whole
domain, at about a thousandth of the peak, which the LES target never has. The cause is the
objective, not convergence. In asinh space that haze is a per-cell error of about 1e-3,
squared 1e-6, under 1% of the converged loss of 1.2e-4. Two changes were tried on the Optuna
best, with three seeds for the reference and the gate and two for the rest:

| variant | val loss z (spread 3.6e-7) | 80% overlap (Kljun 0.566) | area-80 / LES | 2-D shape L1 (Kljun 0.63) | integral error |
|---|---|---|---|---|---|
| Optuna best | 0 | 0.52 / 0.54 / 0.57 | 1.17–1.53 | 0.53–0.54 | 0.110–0.124 |
| + cone gate | −1.2 / +0.9 / +2.6 | 0.49 / 0.55 / 0.59 | 1.31–1.83 | 0.50–0.52 | 0.113–0.140 |
| **+ cone gate + L1 λ 0.03** | **−0.2 / +0.1** | **0.615 / 0.616** | **0.98 / 1.01** | **0.47 / 0.48** | 0.105 / 0.107 |
| + cone gate + L1 λ 0.3 | +1.2 / +1.4 | 0.615 / 0.619 | 0.95 / 0.97 | 0.467 / 0.469 | 0.096 / 0.097 |
| + L1 λ 0.03, no gate | +6.4 | 0.614 | 0.98 | 0.477 | 0.096 |
| + cone gate, knee 0.3 | +4.6 / +11.1 | 0.53 / 0.59 | 1.35–1.60 | 0.49–0.51 | 0.105–0.125 |

The gate is `pred = cone ⊙ (Kljun + residual)` in asinh space. The L1 term is a masked mean
absolute error in asinh space added to the MSE. **The L1 term is what removes the haze.** The
gate alone does not, and the gate with the L1 term keeps the val loss inside the seed spread
where the L1 term alone costs six sd. With λ = 0.03 the 80% overlap moves from below Kljun to
above it, the 80% area matches the LES's, and the 2-D shape L1 falls to 0.48 against Kljun's
0.63, below the 0.63 that separates two realisations of one case. λ = 0.3 gains a little more
shape and integral at a loss and centroid cost and was not taken.

**Final FNO configuration**: the Optuna best plus the cone gate plus λ_L1 = 0.03. Five seeds.
The ensemble mean is the model ([results](results.md)).

## The CFM (`results/ml_cfm/phase1/`)

Four runs of 300 epochs at K = 4 (about 30 min wall for all four):

| run | val_mse_ref (sample mean, S = 4) | z vs the seed pair | composite vs Kljun | best epoch |
|---|---|---|---|---|
| velocity, σ 0.1, seed 0 | 1.2164e-4 | −0.7 | 0.524 | 105 |
| velocity, σ 0.1, seed 1 | 1.2268e-4 | +0.7 | 0.512 | 90 |
| velocity, σ 0.3 | 1.2485e-4 | +3.7 | 0.512 | 110 |
| x-prediction, σ 0.1 | 1.2652e-4 | +6.0 | 0.462 | 125 |

Velocity at σ = 0.1 was kept by the stated rule (the loss). The x-prediction head lost on the
loss by six seed-sd but had the best composite of the four. With the two criteria disagreeing
and n = 2 for the seed pair, the rule was followed and the disagreement recorded.

**Solver study** (seed 0, S = 8): Euler 4/8/16/32 and Heun 8/16 give the mean's
`val_mse_ref` 1.31–1.36e-4 and composite 0.57–0.59, flat in the step count. Euler 16 was used
for every sample. Euler 4 would cost a quarter as much and change nothing measurable.

**Longer training does not help.** A 1000-epoch run of the winner stopped at epoch 105, the
same place the 500-epoch seeds stopped (60–120). The final five seeds ran 500 epochs with
patience 10 evaluations ([results](results.md)).

## Selection rules that held

- Tune on the loss, report the composite. The loss has a 0.5% seed spread. The composite 18%.
- Retrain the best trial with five seeds before claiming anything. Trial-to-trial differences
  below about 1e-6 are noise.
- Select from a fitted curve, not the val argmin, when a parameter has an asymptote. The CFM's
  sample count was read from `err(S) = a + b·S^−p` against the val noise floor
  ([calibration](calibration.md)).
- Drop the worst seed of five for both models. The four-seed pools equal the five-seed pools.
