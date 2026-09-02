# Phase 1 decisions

Short runs (80 epochs, patience 20), val only, one factor at a time from a baseline B0 run
with four seeds. Every comparison is quoted in units of the baseline seed spread, and
"settled" means beyond it. Tables: `summary.md` (round 1, z-scores), `summary.tsv`,
`campaign_round1.json`, `campaign_round2.json`, `gpu_util.csv`.

B0: residual head, width 32, modes 16, depth 4, spectral + conv1×1, channels = Kljun +
distance (linear, exp 300 m) + statics B (topo, log z0, array, water), the file's global
asinh scales (knee 1), stability as z_m/L, MSE only, AdamW lr 1e-3 wd 1e-4, batch 16.
4.22 M parameters.

## Seed spread of B0 (n = 4)

| quantity | mean | sd |
|---|---|---|
| val masked asinh-MSE (file space, `val_mse_ref`) | 1.2138e-4 | 5.8e-7 (0.5%) |
| composite (5-metric ratio to Kljun, < 1 wins) | 0.679 | 0.119 |
| composite, N/NE/NW | 0.772 | 0.140 |

The val loss is precise enough that |z| ≥ 2 is a real effect. The metric composite is NOT:
its seed spread is 18% of its value, driven by the 80% source-area overlap, whose val
median swung 0.37–0.60 across four seeds of one configuration. Spearman rank correlation
across the 31 runs between `val_mse_ref` and the composite is +0.59 (+0.44 for the
northerly composite), above the 0.5 threshold set in the plan, so **Phase 2 optimises
`val_mse_ref`** and records the composite as an attribute.

## GPU utilisation and concurrency

One training process already reports 100% utilisation (nvidia-smi kernel-busy) and ~3 GB.
Solo baseline: 82 s. Round 1 at K = 4: 30 runs in 2178 s = 73 s per run, i.e. **1.1×
throughput from 4-way concurrency**, with per-run wall stretched to 230–620 s. Memory at
K = 4 was 11.9 GB mean, 15.5 GB max of 16.4 GB. The premise that a 128² FNO on a 33 MB
corpus would not saturate a 4080 is false at width 32; a batch of 16 keeps the FFT and
einsum kernels busy. Concurrency is therefore a convenience (independent processes, simple
resumption), not a speed-up, and Phase 2 uses K = 3 to keep memory headroom.

## Settled in round 1 (z on `val_mse_ref`, negative = better)

| factor | levels tried | z | decision |
|---|---|---|---|
| local path | none / conv1×1 / conv3×3 | +268 / 0 / **−2.7** | a local path is required (spectral-only barely trains: val loss stays at its initial 2.77e-4); conv3×3 better. Phase 2 searches {conv1×1, conv3×3}. |
| depth | 2 / 4 / 6 | +1.2 / 0 / **−3.0** | deeper is better at this length; Phase 2 searches 3–6. |
| modes | 8 / 12 / 16 / 24 / 32 | +1.7 / +0.7 / 0 / **−2.0** / −0.9 | 24 best; Phase 2 searches 12–32. |
| width | 16 / 32 / 64 | −0.3 / 0 / +1.0 | no effect within spread; width 64 costs 2.3× the time. Phase 2 searches {16, 24, 32, 48}. |
| head | residual / direct | 0 / +0.7 | no effect; **residual kept** (the design: zero predicts Kljun exactly). |
| raster norm | global (file) / per-record Kljun peak | 0 / +0.6 | no effect; **global kept** (the file's train-only constants). |
| asinh knee | 0.3 / 1 / 3 | +6.1 / 0 / +4.7 | both worse on the file-space loss (partly tautological: the reference space is knee 1); composites within spread. **Knee 1 kept**, not searched, so the Phase 2 objective stays comparable. |
| distance channels | none / lin+exp / lin+exp+XY | −0.9 / 0 / −1.6 | **no measurable effect** from the distance channels themselves; see statics. |
| stability scalar | z-scored L / z_m/L | +0.3 / 0 | no effect; **z_m/L kept** (no derived statistic needed). |
| peak-location term | λ 0 / 0.1 / 1 | 0 / +0.6 / **+8.6** | **hurts** at λ = 1 (centroid ratio 1.95) and does nothing at 0.1. Dropped. |
| integral term | λ 0.1 / 1, ref target or asymptote | −0.8, −0.2 / **+45**, **+268** | neutral at 0.1, harmful at 1 (the asymptote-referenced run diverged at epoch 8). Dropped. |
| northerly ×3 weighting | off / on | 0 / +0.4 | no effect on the loss or on the northerly composite. Dropped. |

The λ scale: the masked MSE converges near 1e-4 while both auxiliary terms are O(0.1), so
they carry a fixed 1e-3 factor (`ml/losses.py:AUX_SCALE`) and λ = 1 means "about one
converged MSE". At that weight both terms hurt.

## Statics: the expected null, and what the control says

Statics are constant across records and can only be a bias term, so the expected result
was no effect. Instead:

| statics | z (`val_mse_ref`) | composite |
|---|---|---|
| none | **+6.2** | 0.800 |
| B (topo, log z0, array, water) | 0 | 0.679 ± 0.119 |
| C (B + one-hot land cover + htFlux) | −2.5 | 0.580 |
| **B rotated 90° (wrong geography, same statistics)** | **+1.1** | 0.699 |

Removing the statics costs six seed-sd on the loss, but the rotated control recovers all
of it (z +1.1, inside the spread). So the gain is not the site: the maps act as a spatial
basis that breaks the translation symmetry of the spectral layers, which the radially
symmetric distance channels cannot do. The user's prior — that a statics effect would be
suspicious — is upheld in the sense that matters: the geography is irrelevant. Round 2
tests whether plain X/Y coordinate planes recover the same loss with no statics at all
(`statics_none_xy`) and carries the winning levels forward with and without statics
(`b1_*` with statics C, `b1x_*` with none + X/Y), three seeds each.

Seed leakage as the alternative explanation: 231 of 235 val records share a
`(seed_job, seed_rot)` with train, so an "un-shared subset" test has n = 4 and cannot
decide anything; the rotated-map control is the test that can, and it says the effect is
capacity, not memorised site geography.

## Per-metric picture on val (round 1, medians; Kljun in brackets)

- peak_x: FNO 0 m [30 m] median, 19–21 m [81 m] mean — the wind-axis peak is at the
  one-cell floor for both, and the FNO removes Kljun's outliers.
- array share, N/NE/NW: FNO 0.9–1.5 pp [3.84 pp]. On train the same number is 0.25 pp
  [1.22 pp]: a 4× train/val gap on this metric, against a realisation floor that the
  two-window pair puts at 5.3 pp for a 20% share (`eval/floor/pair_floor.json`).
- centroid: FNO 56–96 m [92 m]. integral: FNO 0.10–0.19 [0.14].
- 80% source-area overlap: FNO 0.37–0.60 [0.57] and area-80 ratios to the LES of 0.7–2.3
  across seeds — the FNO's weak metric. A smooth conditional-mean field carries diffuse
  low-level mass that a sample from the LES does not; whether it sits outside the cone
  (removable at evaluation) is measured on the final model.

## Round 2: the winning levels combined, three seeds each

| run | statics | dist | val_mse_ref z (per seed) | mean z | composite (mean) |
|---|---|---|---|---|---|
| B0 (reference) | B | lin+exp | −0.2, −0.1, +1.4, −1.0 | 0 | 0.679 |
| B1 = depth 6, conv3×3, modes 24 | C | lin+exp | −2.3, +2.0, −0.5 | −0.3 | 0.689 |
| B1x = B1 without statics | none | lin+exp+XY | +0.8, +1.2, −1.3 | +0.2 | 0.643 |
| statics_none_xy (B0 without statics) | none | lin+exp+XY | +1.6 | +1.6 | 0.643 |

**B1 does not beat B0 outside the seed spread.** The round-1 single-run effects of z −2 to
−3 did not survive combination. With 27 one-factor comparisons drawn against a spread
whose own estimate rests on four seeds, the expected extreme of 27 null draws is about
z −2, so those "wins" were the winner's curse, not effects. The architecture space
(width 16–64, depth 2–6, modes 8–32, either local path) is FLAT within the seed spread at
this training length. What is real is the large-|z| set: a local path is required, the
knee stays at 1, the auxiliary terms hurt at λ = 1, and constant spatial channels of some
kind are required.

**Statics closed.** Plain X/Y coordinate planes recover the no-statics deficit to within
the spread (z +6.2 → +1.6 with the planes; the B1 pair with and without statics is
identical, +0.2 vs −0.3). The statics' only role was as a positional basis, and the
rotated-map control had already shown the geography does not matter. Phase 2 therefore
runs with **no static channels** and the X/Y planes, which makes the emulator's inputs
exactly what the spec asked for — Kljun's six scalars, the Kljun raster and the receptor
geometry — with nothing site-specific entering except through the target.

## What Phase 2 searches

Fixed: residual head, no statics, distance lin+exp+XY, global file normalisation with
knee 1, z_m/L, no auxiliary terms, no weighting. Searched: modes 12–32, width
{16, 24, 32, 48}, depth 3–6, local {conv1×1, conv3×3}, lr log[2e-4, 3e-3], wd log[1e-6,
1e-1], batch {8, 16, 32}, FiLM hidden {32, 64, 128}, dropout [0, 0.3]. Objective
`val_mse_ref`; 150 epochs, patience 25, median pruning. Because the objective's seed
spread is 5.8e-7, trial-to-trial differences below ~1e-6 are noise, and the best trial is
retrained with five seeds before anything is claimed.
