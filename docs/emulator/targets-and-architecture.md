# Targets and architecture

Decided 2026-08-30, reversing the design of the day before. The target is the 122² raster
zero-padded to 128², the model is a Fourier neural operator (FNO) that predicts a residual on
Kljun conditioned on the six scalars by FiLM, and touchdowns are not saved. A flow-matching
model (CFM) built beside it on 2026-09-02 uses the same inputs and gives sample spread.

## What a record is

One `.npz` per case, written by `bin/make_pair.py`, self-contained and referencing nothing:

| array | shape | dtype | meaning |
|---|---|---|---|
| `scalars` | (6,) | f32 | `h`, `u*`, `σ_v`, `L`, `sin(wdir)`, `cos(wdir)`: Kljun's inputs, every one read off the LES window itself |
| `kljun` | (128, 128) | f32 | the official FFP on the target's own cell centres |
| `target` | (128, 128) | f32 | the LPDM flux footprint, signed, unclipped |
| `meta` | scalar | json | everything below |

Self-contained is the requirement, not a nicety. The corpus is generated on rented machines
that share no filesystem with each other; a record that points at a results directory does
not survive the trip.

**122 → 128 is a zero-pad of 3 cells, not a resize.** Every value is either a real LES column
or a structural zero. 128 is what the spectral transform wants; the pad extent is in
`meta["grid"]["pad"]` so the loss masks the border rather than learning to reproduce it. The
receptor's index is recorded in both frames, (61, 61) → (64, 64).

**The frame is north-up, and that is the site.** No rotation to a wind-aligned frame. The
emulator's job is to know that the array extends 250 m north and 60 m east of the tower and
that the lake is east-north-east. A wind-aligned frame throws that away and asks the model to
learn a rotationally symmetric function, which is what Kljun already is. Direction enters as
an input and indexes the site's geometry.

**`L` is written raw and the loader must not use it raw.** `L` is ±inf at exactly neutral, a
legitimate state, so the loader substitutes `meta["inv_L"]`, which is finite everywhere and is
the form the similarity functions use. Stability is fed to the model as `z_m/L = 28.5 · inv_L`.

**`h` is the TKE peak-fraction estimator, and the record says so**
(`meta["h_estimator"] = "tke_peak_fraction"`: 5% of the resolved-TKE profile's own peak,
bounded by the decay minimum, `lpdm/les_stats.py:bl_depth`). The project has two definitions
of `h` that differ by 7–21%; the seed stationarity gate uses a fixed 0.01 m²/s² threshold
because it scores a trend. One estimator everywhere in the training record.

**The Kljun channel is the official FFP.** `third_party/FFP/` is Natascha Kljun's v1.42,
vendored unmodified. `lpdm/kljun_ffp.py` re-evaluates its two separable factors at the
raster's own cell edges, so the two channels are on identical cells by construction, and
`bin/test_kljun_adapter.py` scores the adapter against the code it wraps at 9.4e-16. The
project's own earlier reimplementation was 1.25× wide in `σ_y` whenever `|L| > 5000`: the
official code resets `ol = −1e6` above `oln = 5000` and clips `scale_const` to 1.0, and ours
never reached the clip. That is exactly the near-neutral regime, the one place Kljun is
diagnostic rather than descriptive.

`meta` also carries: `datetime`, `parent_case`, `run_id`, `split`, `split_key`, `gate_state`
(the seed's stationarity verdict travels with every pair), `integral`, `integral_kljun`,
`integral_asymptote`, `peak_x_m`, `centroid_dist_m`, `centroid_bearing_deg`, `array_share`
with its standard error, the full `cover_share`, the git commit, the grid configuration, the
closure configuration, `floor_health`, the FFP validity conditions the case violates (the
official code only prints those), and any warnings. A share quoted without a standard error
cannot be compared to anything: an `h` fall-through defect once moved the array share 0.8
points against a 3.66-point SE and looked like a result.

## The target is the raster, and the touchdowns are gone

Within-cell cancellation is the integration. A footprint cell is a flux, and a flux is what
the positive and negative touchdowns in that cell sum to. The point process is a finer
description of the estimator, not of the physics, and the model's output has to be a raster
because that is what a footprint is used for.

What that costs, stated: on the two 30 m validation targets the negative lobe is 2.0–2.3% of
|flux| on the raster and 21–35% of |weight| in the touchdown sample. The raster stays signed
and nothing clips it, so the cancelled residual is still in the target; what is given up is
the `f = w⁺p⁺ − w⁻p⁻` decomposition, the leading candidate under the old design.
`--keep-touchdowns` still exists; it is not part of the corpus.

## Not generative, and the honest reason

"The pairs are deterministic" is contradicted by the project's own measurement: re-running an
identical case gave integral 1.463 → 1.019 and array share 5.65% → 1.07%. The target is a
sample from a conditional distribution. A deterministic model is right anyway because the
deliverable is the conditional mean footprint. An MSE regression in asinh space converges to
it, which is what an emulator of a 30-minute flux footprint is asked for; the realisation
spread is a property of the estimator and of turbulence, quantified by the two-window pairs.
(The CFM was then built to give that spread as well, and its mean ties the FNO; see below.)

## Splitting

Split by parent case, never by window, cell or touchdown. Two windows of one case share a
seed, an adjustment, a sounding and a surface. `meta["split_key"]` is the parent. In the
shipped corpus (`N_WINDOWS = 1`) the split is by calendar year, assigned at generation
([the dataset](../corpus/dataset.md)). Normalisation constants are computed on the training
split only.

## The FNO (`ml/`)

An FNO that predicts a residual on Kljun in asinh space, conditioned on the six scalars by
FiLM.

- **Inputs**: the Kljun raster in asinh space (`asinh(x/s)` with the file's own train-only
  `s_kljun = 2.170e-5`, `s_target = 2.426e-5` m⁻²), plus receptor geometry channels that are
  identical for every record: distance from the receptor (linear, and exponentially decaying
  with a 300 m scale so the near field has a channel whose gradient does not vanish) and the
  X/Y coordinate planes. Static terrain and land-cover channels were tested and dropped
  ([training](training.md)).
- **Model** (`ml/model.py`): lift → `depth` × [spectral convolution on the lowest `modes`² of
  `rfft2` + a local convolution] → FiLM(γ, β from the six scalars) → GELU → projection with a
  zero-initialised last layer. A zero residual reproduces Kljun to 7.5e-8 relative
  (`bin/test_ml_model.py`). FiLM because the scalars are global quantities; a spatial encoding
  would be six constant planes at 128² times the cost.
- **Residual on Kljun, not the raw footprint**: Kljun already gets the gross shape right over
  flat ground; the site-specific signal is the correction.
- **Gate**: `pred = cone ⊙ (Kljun + residual)`, where the cone is Kljun's own `σ_y` and the
  wind direction, inputs the model already has. It makes the raw and cone-cropped prediction
  identical.
- **Loss**: masked MSE in asinh space plus 0.03 × masked MAE, over the 122² interior. A
  peak-location term and an integral term (against `1 − z_m/z_i`, never against 1) were built
  and measured harmful. The asinh compresses exactly the dynamic range that noise dominates:
  the normalised per-cell L1 between two independent release ensembles of one window runs
  38–92%, while the peak agrees to zero cells and the 80% source areas overlap 40–60%.
- **Final configuration**: modes 32, width 48, depth 3, 3×3 local convolution, lr 2.6e-4,
  weight decay 0.019, dropout 0.22, batch 8, FiLM hidden 32; 28.4 M parameters; 150 epochs,
  patience 25; five seeds, ensemble mean in physical space.
- **Metrics** (`ml/metrics.py`) are the production functions and nothing else: the
  crosswind-integrated peak distance, the centroid and 80% area from `lpdm/footprint.py`,
  the Jaccard overlap of the 80% source areas, the raster form of the array share, the shape
  L1, and the standard 2-D field metrics. The same estimator is applied to the LES target, to
  Kljun and to the emulator, so a comparison is between fields, never between estimators.
  The raster array share agrees with the touchdown-based `meta/array_share` at r = 0.9995.

## The CFM (`ml_cfm/`)

Conditional flow matching on the straight-line path between the noised Kljun prior and the
LES target, in the same asinh space, with exactly the FNO's inputs plus `t`:

```
x_prior = cone ⊙ asinh(kljun / s_target)      d = x_les − x_prior      ε ~ N(0, σ²) on the cone
z_t     = x_prior + t·d + (1 − t)·ε            t ~ U(0, 1)
v       = d − ε                                the regression target (velocity parameterisation)
sample  : z_0 = x_prior + ε,  z_{k+1} = z_k + Δt · v̂(z_k, t_k; scalars),  x̂_les = z_1
```

The coupling is fixed by the record (prior and target are already paired), so no minibatch
optimal transport is needed. The network is a 2.95 M-parameter U-Net (widths 32/64/128/192,
four levels, GroupNorm, GELU) with FiLM on [sinusoidal(t), the six scalars] after every block
and a zero-initialised output convolution, so the untrained model returns `x_prior + ε` and a
σ = 0 sample reproduces Kljun to 7.5e-8 relative (`bin/test_cfm.py`). Input channels: `z_t`,
the Kljun channel, distance (linear, exponential) and the X/Y planes. EMA 0.999, AdamW
1e-3 / 1e-4, cosine, batch 16, gradient clip 5, σ = 0.1. Selection and early stopping on the
val MSE of the 4-sample (later 16-step Euler) mean every 5 epochs, the same number the FNO
was selected on. No adversarial term: samples carry the LES's texture without one.

Why the CFM gives what the FNO cannot: per record it draws samples whose spread is the
model's estimate of the realisation spread, so the array share comes with an error bar. Its
sample mean ties the FNO with a tenth of the parameters ([results](results.md)); the spread is
about 20% under-dispersed overall and 30% where the array is in view ([calibration](calibration.md)).

## Environment

The `LESNet` conda environment (torch 2.5.1 + CUDA 11.8, Python 3.11) on the RTX 4080, with
h5py 3.16 and optuna 4.9; spec in `ml/environment.yml`. No Docker image carries torch; the
emulator is the one analysis that runs on the host.
