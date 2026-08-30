# The ML target format — design, 2026-08-29

> **DESIGN ONLY. One thing is built: the touchdown persistence, and it had to be, because
> the window fields are deleted at the end of every case.** A touchdown not written at
> stage 5 is gone, and no amount of later design work brings it back. Everything else here
> — the density model, the loss, the heads — is decided at build time, after the seventh
> pass has said whether a 30 m receptor produces a footprint that responds to meteorology
> at all.

## Why the 122² raster is not the target

**Binning is what makes two realisations of the SAME conditions look 92% different.**
Measured (`bin/stage5_footprint.py`, the realisation-vs-realisation block): the normalised
per-cell L1 between two independent release ensembles of one window runs 38–92% while the
peak agrees to zero cells and the 80% source areas overlap 40–60%. A per-cell loss computed
on that raster is therefore mostly fitting Monte-Carlo noise, and the model that minimises
it is the model that best predicts the *sampling error* of the estimator.

The footprint the estimator actually produces is a **weighted point process**: a list of
touchdowns, each with a signed weight. The raster is one summary of it — a histogram with
a 24 m bin. Fitting a continuous density to the points removes the bin from the loss
entirely, and keeps the raster for the two things it is genuinely good for: figures, and
comparison against Kljun on identical cells.

## What is persisted, per case

`runs/<case>/touchdowns.npz`, written by `bin/stage5_footprint.py --keep-touchdowns N`
(driver support in `lpdm/driver.py`; the GPU path returns the same arrays off the device):

| array | dtype | meaning |
|---|---|---|
| `dx`, `dy` | f32 | receptor-relative displacement, **UNFOLDED** (metres, map frame) |
| `wt` | f32 | **signed** flux weight `w_release · 2/max(\|w_td\|, w_floor)` |
| `grp` | i16 | release-group index, so a sampling floor can be estimated from the sample |
| `age` | f32 | touchdown age, i.e. backward transit time (s) |
| `meta` | json | `n_touchdown_total`, `n_particles`, `sum_wt_total`, `weight_scale`, domain, receptor, `wind_angle_deg` |

Three decisions inside that table, each of which would be hard to undo later:

1. **UNFOLDED, not folded.** The raster folds touchdowns modulo the periodic domain,
   because the LES world is tiled and the land-cover attribution folds identically. The ML
   target wants the true displacement: folding is a modulo away, unfolding is not
   recoverable. (It also makes the wrap cap visible in the data rather than hidden.)
2. **SIGNED.** Negative values are physical — an elevated concentration maximum in the CBL,
   and wind turning with height in neutral air — and `bin/test_negative_lobes.py` measures
   what is being preserved: 5.8–11.1% of |flux| across twelve production convective
   footprints. See the open question below; this is the constraint that makes the obvious
   architecture not work.
3. **UNIFORMLY SUBSAMPLED, with the exact pre-subsample totals kept.** Bottom-k on an
   independent uniform key (CPU) or a Bernoulli filter (GPU): both are exactly uniform
   without replacement and stream in one pass. `weight_scale` is what a sample weight is
   multiplied by to reproduce the full estimator, and stage 5 asserts that the sample
   reproduces the full ensemble's integral. ~1e5 touchdowns is ~1.6 MB per case; 1469 cases
   is ~2.3 GB.

## The target itself

**Fixed north-up map coordinates, receptor at the origin, NO rotation to a wind-aligned
frame.** This is the decision the whole site rests on. The emulator's job is to know that
the array extends 250 m north and 60 m east of the tower, that the lake is to the
east-north-east, and that the tree line is where it is. A wind-aligned frame throws that
away and asks the model to learn a rotationally symmetric function — which is what Kljun
already is. Direction enters as an INPUT and indexes the site's geometry; it must not be
factored out of the coordinates.

**Shape is split from magnitude.**

```
target  =  A · p(dx, dy | inputs)          A = the integral, a scalar
                                            p = a normalised signed density
```

- the **integral head** regresses `A` against the physical ceiling **1 − z_m/z_i**
  (Steinfeld et al. 2008, after Horst & Weil 1992) rather than against 1. At 30 m in an
  800 m boundary layer that ceiling is 0.963, and the domain truncation sits below it. The
  residual `A/(1 − z_m/z_i)` is what the head actually predicts.
- the **shape head** predicts `p`, which integrates to 1 by construction and carries no
  information about how much flux there was.

Splitting them matters because the two have different error structures and different
physics: `A` is set by the boundary-layer depth and the domain, `p` by the surface and the
turbulence. A single head trained on the product spends its capacity on the easier one.

## The open question, flagged and NOT decided

**A normalizing flow is a non-negative density. It cannot represent the negative lobes that
`fix 2` exists to preserve.** That is a direct conflict between the project's stated primary
architecture and the target's own sign structure, and it is better named than quietly
resolved by clipping. Three candidates, to be decided once the negative-lobe magnitude is
measured on real 30 m targets:

| option | shape | cost |
|---|---|---|
| **signed decomposition** `f = w⁺p⁺ − w⁻p⁻` | two flows plus a mixing weight; each is a proper density and the CNF machinery is unchanged | two flows, and `w⁻` is small and therefore hard to estimate |
| **field model on the Kljun residual** (FNO / U-Net) | unconstrained, signs are free, and PROJECT_BRIEF.md already sanctions these as benchmarks | back on a raster, which is what this document exists to avoid — unless it is evaluated on a continuous loss (Sinkhorn on the signed measure) |
| **weighted MLE with signed weights** | one flow, fit by maximum likelihood with negative weights | non-standard; the likelihood is not a likelihood and the estimator can be biased in ways nobody has characterised for this problem |

**MEASURED, 2026-08-29, on the two 30 m production targets — and the answer is bigger than
the raster suggests, which is itself the point:**

| | on the 122² raster | in the TOUCHDOWN SAMPLE |
|---|---|---|
| `case_2023052519` convective | 2.32% of \|flux\|, 8.2% of cells | **33.0% of touchdowns, 20.9% of \|weight\|** |
| `case_2023121921` near-neutral | 2.03% of \|flux\|, 2.9% of cells | **40.1% of touchdowns, 35.0% of \|weight\|** |

The raster hides it, because within a 24 m cell the positive and negative touchdowns cancel
before anything is written down. **The target this document proposes — a density fitted to
the points — sees the uncancelled version, and a third to a half of its mass is negative.**
Its negative lobe sits 1.2–1.6× further out than the positive one, so the sign structure is
spatial and not noise.

That settles the open question against the naive route: **a single nonnegative flow cannot
represent this target**, and clipping would discard a fifth to a third of the signed mass
rather than a rounding error. Option 1 (`f = w⁺p⁺ − w⁻p⁻`) is viable because `w⁻` is now
known to be large enough to estimate; option 2 stops being merely a benchmark. Decide at
build time — but decide knowing the number is 21–35%, not 2%.

## What stays on the raster

- every figure
- the Kljun comparison, which must be on identical cells (`lpdm.kljun.footprint_on_static`)
- `bin/corpus_monitor.py`'s gates, which are QC on the estimator and not on the target
- the land-cover shares, which are accumulated in LES index space at nearest-grid-point and
  are deliberately not resampled

## Splitting

**By LES run, never by sample** (PROJECT_BRIEF.md). The effective sample size for generalisation is
the number of *runs*. With touchdowns as the target this is even more important than it was
with rasters: a random split over touchdowns would put the same window on both sides.

## Two footprints per case, and the split rule that follows — 2026-08-30

**A case now runs 2.0 simulated hours and yields TWO footprints**, over disjoint field
intervals (1800 s adjustment, then windows at 1800–4500 s and 4500–7200 s; window 2's
releases begin 900 s = `t_back` after window 1's end, so no field is shared). They are
persisted as `pairs/<case>_w0.json` and `_w1.json`, each carrying `parent` and
`window_index`.

**Why, and it is not to get more data cheaply.** Re-running an identical case — same
restart, forcing, rotation and code — gave integral 1.463 → 1.019 and array share
5.65% → 1.07%. That is turbulence REALISATION variance, and every error floor this project
quotes is measured *within* one realisation and is therefore too small. A second window is
a second draw at nearly the same condition for 0.75 h instead of a whole extra case.

**For the model this is a feature, not noise to be averaged away.** A noisy target is a
sample from the conditional distribution rather than a wrong target, and the density this
document proposes is fitted to samples — so two draws from one condition are exactly what
it wants. Averaging is for REPORTED numbers only (array share and the like), and those are
quoted with the across-realisation spread beside them.

**THE SPLIT RULE TIGHTENS: split by PARENT CASE, never by window and never by touchdown.**
PROJECT_BRIEF.md already says split by LES run; with two windows per run, `<case>_w0` and
`<case>_w1` share a seed, an adjustment, a sounding and a surface, so putting one in train
and the other in validation would leak almost everything that makes them what they are.
The effective sample size for generalisation is the number of PARENTS — ~1469 — and not the
~2900 pairs.

**What is measured on the first case that does this, and reported either way:** whether the
two windows are statistically independent (from the release groups' own decorrelation
ladder, and from |w0 − w1| against the within-footprint half-vs-half floor), and how far
`z_i` drifts between them. Near-replicates at one condition reduce realisation noise at that
condition; two different conditions are coverage and do NOT reduce it — in which case the
corpus still owes condition-bin averaging. `bin/window_independence.py`.
