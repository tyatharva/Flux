# The ML target format — decided, 2026-08-30

> **THIS REPLACES THE 2026-08-29 DESIGN AND REVERSES ITS CENTRAL DECISION.** That document
> argued the 122² raster was *not* the target and that a density fitted to the touchdown
> point process was. The target is now the raster, padded to 128²; **touchdowns are not
> saved at all**; and the architecture is an FNO residual on Kljun rather than a
> conditional normalizing flow. The reasoning that produced the old design is kept below
> under "What this reversed", because it is the record of how the question was arrived at
> and two of its measurements still bind.

## What a record is

One `.npz` per sampling window, written by `bin/make_pair.py --npz-dir`, self-contained and
referencing nothing:

| array | shape | dtype | meaning |
|---|---|---|---|
| `scalars` | (6,) | f32 | `h`, `u*`, `sigma_v`, `L`, `sin(wdir)`, `cos(wdir)` |
| `kljun` | (128, 128) | f32 | the official FFP on the target's own cell centres |
| `target` | (128, 128) | f32 | the LPDM flux footprint, **signed**, unclipped |
| `meta` | scalar | json | everything below |

26–47 kB each; ~2900 windows is ~100 MB for the whole corpus.

**SELF-CONTAINED IS THE REQUIREMENT, not a nicety.** The corpus is generated on rented
machines that share no filesystem with this one or with each other. A record that points at
`results/corpus/<tag>.npz` does not survive the trip, and one that assumes a grid directory
is present cannot be read at training time. Each machine also appends to a
`manifest.json` — case list, git commit, grid config, hostname — so a corpus assembled from
several boxes can be checked for gaps and for version skew instead of assumed homogeneous.

### 122 → 128 is a zero-pad of 3 cells, not a resize

`122 + 3 + 3 = 128` exactly. Every value in the padded array is either a real LES column or
a structural zero: nothing is resampled, rescaled, cropped or interpolated. 128 is what the
FNO's spectral transform wants; the pad extent is in `meta["grid"]["pad"]` so the loss masks
the border rather than learning to reproduce it. The receptor's index is recorded in **both**
frames (`ij_122`, `ij_128`) because the pad moves it — (61, 61) → (64, 64).

### The frame is north-up, and that is the site

Fixed map coordinates, receptor at the origin, **no rotation to a wind-aligned frame**. The
emulator's job is to know that the array extends 250 m north and 60 m east of the tower,
that the lake is east-north-east, and where the tree line is. A wind-aligned frame throws
that away and asks the model to learn a rotationally symmetric function — which is what
Kljun already is. Direction enters as an INPUT and indexes the site's geometry; it must not
be factored out of the coordinates.

### `L` is written raw, and the loader must not use it raw

`scalars[3]` is `L` because that is the named format. `L` is unbounded and is ±inf at exactly
neutral — a legitimate state, not corruption — so **the loader substitutes
`meta["inv_L"]`**, which is finite everywhere and is the form the similarity functions
actually use. `make_pair.py` prints a warning when `L` is non-finite. This is written down
in three places on purpose; it is the kind of thing that is discovered at training time.

### `h` is the TKE peak-fraction estimator, and the record says so

`meta["h_estimator"] = "tke_peak_fraction"`: 5% of the resolved-TKE profile's own peak,
bounded by the decay minimum (`lpdm/les_stats.py:bl_depth`). This project has **two**
definitions of `h` that differ by 7–21% — the seed stationarity gate uses a fixed
0.01 m²/s² threshold instead, because it scores a trend and a peak-normalised threshold
moves with the peak. A corpus that does not name which it used has an `h` channel nobody can
reproduce. One estimator everywhere in the training record; it is this one.

### The Kljun channel is the official FFP

`third_party/FFP/` is Natascha Kljun's own v1.42, vendored unmodified.
`lpdm/kljun_ffp.py` re-evaluates its two separable factors at the target raster's own cell
edges, so **the two channels are on identical cells by construction rather than by
assertion** — and `make_pair.py` checks the edges against the centres to 1e-9 m before it
does so. `bin/test_kljun_adapter.py` scores the adapter against the code it wraps at
**9.4e-16**.

The reimplementation this project used until now (`lpdm/kljun.py`) is **1.2500× wide in
`sigma_y` whenever `|L| > 5000`** — the official clips its `scale_const` to 1.0 and ours
does not reach the clip — which is exactly the flat/neutral end of the corpus. It survives
only for the gates already validated against it.

## The target is the raster, and the touchdowns are gone

**Within-cell cancellation IS the integration.** A footprint cell is a flux, and a flux is
what the positive and negative touchdowns in that cell sum to. The point process is a finer
description of the *estimator*, not of the physics: nothing the raster drops is a property
of the atmosphere, and the model's output has to be a raster in any case because that is
what a footprint is used for.

**What that costs, stated rather than implied.** Measured on the two 30 m production
targets, the negative lobe is 2.0–2.3% of |flux| on the raster and **21–35% of |weight| in
the touchdown sample** — the uncancelled version is far larger, and it is spatial rather
than noise (its centroid sits 1.2–1.6× further out than the positive lobe's). Dropping the
touchdowns drops the ability to model that structure directly. **The raster stays signed and
nothing clips it**, so the cancelled residual is still in the target and the model can still
produce negative cells; what is given up is the `f = w⁺p⁺ − w⁻p⁻` decomposition, which was
the leading candidate under the old design and is now out of scope.

`--keep-touchdowns` still exists and still works. It is simply not part of the corpus.

## The model

**FNO, predicting a residual on Kljun, conditioned on the six scalars by FiLM.**

- **Input channels**: Kljun (symlog-transformed); distance from the receptor, twice — linear
  and exponentially decaying, so the near field has a channel whose gradient does not vanish;
  static terrain and land cover, which are identical in every case and are what a
  site-calibrated emulator is *for*.
- **Conditioning**: FiLM on the 6 scalars. They are global quantities and a spatial encoding
  of them would be six constant planes, which is the same information at 128² times the cost.
- **Residual on Kljun, not the raw footprint.** Kljun already gets the gross shape right over
  flat ground; the site-specific signal is the correction.
- **Loss**: symlog MSE, plus a peak-location term, plus an integral term scored against
  **`1 − z_m/z_i`** and never against 1. The asymptote is Steinfeld et al. (2008), after
  Horst & Weil (1992): the fraction `z_m/z_i` of the column lies below the receptor and its
  flux never crosses it. At 30 m in an 800 m boundary layer that is 3.75%, the size of
  effects this project routinely gates on.
- **Log transform**: `log(x + eps) − log(eps)` with `eps = 1e-3`, applied symmetrically for
  signed values (`sign(x) · [log(|x| + eps) − log(eps)]`). Follows FootNet.
- **No U-Net baseline.** Deliberate: a second architecture is a second thing to validate and
  the FNO's resolution-independence is the property that matters here.

### Not generative — and the honest reason

**"The pairs are deterministic" is contradicted by this project's own measurement**, so it
is not the justification. Re-running an identical case — same restart, forcing, rotation and
code — gave integral 1.463 → 1.019 and array share 5.65% → 1.07%. The target is a *sample*
from a conditional distribution, not a fixed function of the inputs.

The reason a deterministic model is right anyway is that **the deliverable is the conditional
mean footprint**. A symlog-MSE regression converges to it, which is what an emulator of a
30-minute flux footprint is asked for; the realisation spread is a property of the estimator
and of turbulence, and the two-window pairs are what quantify it rather than something the
model should reproduce. Flow matching would let the model sample the spread — and there is no
use for a sample.

## Splitting, and the rule tightens

**Split by PARENT CASE. Never by window, never by cell, never by touchdown.**

A case runs 2.0 simulated hours and yields two footprints over disjoint field intervals
(1800 s adjustment, then two windows one output interval apart). `<case>_w0` and `<case>_w1`
share a seed, an adjustment, a sounding and a surface, so putting one in train and the other
in validation would leak almost everything that makes them what they are. **The effective
sample size for generalisation is the number of PARENTS — ~1469 — not the ~2900 pairs.**
`meta["split_key"]` is the parent and is what a loader must group on.

## Corpus assembly

1. Each machine writes `pairs_npz/<tag>.npz` per window and appends to its own
   `manifest.json`.
2. The npz files are shipped back and consolidated **locally** into one HDF5.
3. Splits are assigned by parent case at consolidation time and written into the file.
4. **Normalisation constants are computed on the TRAINING SPLIT ONLY** and stored in the
   file. Computing them over the whole corpus leaks the validation set's distribution into
   the inputs, which is the quietest possible form of the leak the split rule exists to
   prevent.

## What every record carries, and why it is there

`meta` also holds: `datetime`, `parent_case`, `window_index`, `gate_state` (the seed's
stationarity verdict — `INDETERMINATE` is the library's normal state and travels with every
pair), `integral`, `integral_kljun`, `integral_asymptote`, `peak_x_m`, `centroid_dist_m`,
`centroid_bearing_deg`, `array_share` **with its standard error**, the full `cover_share`,
the git commit, the grid config, the closure configuration, `floor_health`, the FFP validity
conditions the case violates (the official code only *prints* those), and any warnings.

A share quoted without a standard error cannot be compared to anything: the `h`
fell-through defect moved the array share 0.8 points against a 3.66-point SE and looked like
a result.

## What this reversed, and the two measurements that still bind

The 2026-08-29 design argued the raster was not the target, on the grounds that **binning is
what makes two realisations of the same conditions look 92% different**: the normalised
per-cell L1 between two independent release ensembles of one window runs 38–92%, while the
peak agrees to zero cells and the 80% source areas overlap 40–60%. That measurement stands
and it is the reason the loss is **not** raw per-cell MSE — symlog compresses exactly the
dynamic range that noise dominates, and the peak and integral terms score quantities that
are converged where the per-cell values are not.

The second is the negative-lobe fraction, 21–35% of |weight| uncancelled against ~2% on the
raster. It is why this document says plainly what dropping the touchdowns costs instead of
claiming the raster is lossless.

What did not survive is the conclusion: that a density fitted to the points was therefore
the right target. It required a two-flow signed decomposition to represent a target a third
of whose mass is negative, and it produced a model whose output was not the raster that a
footprint is used as.

**PROJECT_BRIEF.md still names the CNF as the primary architecture and the FNO as a benchmark.
That is now the wrong way round, and the dated block at the top of PROJECT_BRIEF.md says so.**
