# The wraparound cone: `target_cone` is the training target

**2026-09-01.** Numbers: `results/cone_mask_validation.txt`. Per record:
`results/cone_mask_per_record.tsv`. Figures: `figures/cone_mask_effect.png`, `figures/cone/`.
Code: `bin/mask_cone.py`.

## The framing

**The raw target is what the LES produced. The wraparound in it is an artifact of periodic
boundary conditions and is operationally meaningless — no tower measures it, and no emulator
should be asked to predict it. So the ML target must not carry it.**

`target_cone` is the training target. `target` is retained unchanged, so the raw simulation
can be trained on later if wanted.

## Why a cone and not a half-plane

The first attempt cut the downwind half-plane — every cell whose along-wind projection was
negative. **It leaves residual wraparound.** The periodic fold is **per axis and
independent**, so for a diagonal wind a particle can wrap in *x* alone and land back in the
**upwind** half as a thin off-axis streak. Single-axis wrap survives a half-plane cut
whenever the displacement exceeds `3660·max(|sin|, |cos|)`, which is why the streaks are thin
shells and why they vanish for axis-aligned winds.

Measured: the cone removes a median **1.18%** of `|f|` that lies *upwind* of the receptor —
material the half-plane could not see — up to **9.39%**, and **nonzero on 1360 of 1366
records**. On `case_2022030716` (wind FROM 303°) it is 2.41%.

The physical criterion is **crosswind spread, not the sign of a projection**. Real material
cannot be far off-axis at large along-wind distance, because `|y'|` is bounded by
`σ_y(x')` — and Kljun already computes `σ_y`, from the corpus's own input channel.

## The mask

    keep  ⟺  x' ≥ x_min   AND   |y'| ≤ max(k·σ_y(x'), y_min)

with `x' = x·sin_wdir + y·cos_wdir` (positive **upwind**) and `y' = x·cos_wdir − y·sin_wdir`.
`σ_y` comes from the official FFP v1.42 through `lpdm/kljun_ffp.py:ffp_profile` — the same
call that produced the `kljun` channel.

`y_min` floors the half-width because `σ_y → 0` at the receptor and a pure cone would pinch
the peak. `x_min` keeps the cone **empty downwind**, where a genuine footprint is zero by
construction.

**`k = 8`, `y_min = 90 m`, `x_min = 0 m`.** All three measured, not asserted.

## How `k` was chosen: the distribution is bimodal

The LES `|mass|` distribution against `q = |y'|/σ_y(x')`, averaged over 400 records:

| q | LES \|mass\|/bin | cumulative LES | cumulative Kljun |
|---|---|---|---|
| 0.00–0.25 | 16.837% | 16.84% | 19.506% |
| 2.00–2.25 | 2.100% | 84.64% | 97.355% |
| 4.00–4.25 | 0.011% | 87.51% | 99.997% |
| 6.00–6.25 | **0.000%** | 87.53% | 100.000% |
| 8.00–8.25 | **0.000%** | 87.53% | 100.000% |
| 10.00–10.25 | **0.000%** | 87.53% | 100.000% |
| 12.00–12.25 | 0.001% | 87.53% | 100.000% |
| 16.00–16.25 | 0.004% | 87.57% | 100.000% |
| 20.00–20.25 | 0.006% | 87.65% | 100.000% |
| q ≥ 24 (incl. downwind) | 12.241% | 100.00% | 100.000% |

**There are two populations and the valley between them is empty.** The LES carries
**0.0110%** of its `|mass|` in `q ∈ [5, 11)`, and then it **rises again** past `q ≈ 11`. The
footprint lives below `q ≈ 5`; the wrap lives above `q ≈ 11`. Kljun carries **0.00000%**
beyond `q = 6`.

`k = 8` is the middle of that valley. This is not a tuned number — **any `k` in [5, 11]
gives the same answer.**

## Sensitivity: the answer barely moves

| k | y_min | removed median | p95 | max | within 200 m **upwind** median | max | σ_v bias |
|---|---|---|---|---|---|---|---|
| 3 | 0 | 12.83% | 19.75 | 29.75 | 0.000% | 0.359% | 1.01 |
| 5 | 90 | 12.46% | 19.06 | 29.40 | 0.000% | 0.127% | 1.03 |
| **8** | **90** | **12.46%** | **19.06** | **29.40** | **0.000%** | **0.127%** | **1.03** |
| 12 | 120 | 12.45% | 19.05 | 29.40 | 0.000% | 0.057% | 1.03 |

Removed mass moves by 0.38 percentage points across a **factor of four in k**. That flatness
*is* the evidence: there is nothing between the footprint and the artifact to be sensitive to.

**It does not eat wide footprints.** The σ_v bias column is the top-decile / bottom-decile
ratio of removed mass: **1.03**, i.e. the broadest 10% of records by σ_v lose 3% more than
the narrowest 10%. σ_v spans 0.42–1.70 m/s across the corpus and the slope of removed mass
against σ_v is −0.25 %/(m/s) — **negative**, so if anything the wide cases lose slightly less.

**The within-200 m column is upwind only, and deliberately.** Downwind of the receptor a
genuine footprint is zero (next section), so removed mass there is artifact and including it
would swamp the very signal the column exists to show — it sits at ~1.02–1.14% for every
setting and discriminates nothing. Upwind, `y_min` discriminates properly: 0.219% at 60 m,
**0.127% at 90 m**, 0.084% at 120 m, against a 1.00% budget. `y_min = 90 m` binds only where
`k·σ_y < y_min`, i.e. `x' ≲ 30 m`.

## The `x_min` term was a bug fix, not a refinement

**The first cone bounded the downwind side at `x' ≥ −y_min` instead of `x' ≥ 0`.** That left
the `y_min` floor open over a strip 90 m deep and ~`2·max(8·σ_y(0), 90)` m wide directly
behind the receptor.

**It under-removed for axis-aligned winds specifically, and the mechanism says why.** The
periodic fold is per axis. For a near-axis-aligned wind the fold that matters happens in
**one** axis, so every wrapped particle lands on a single line through the receptor — straight
into that strip. Diagonal winds fold in both axes and scatter off-axis, where the cone already
catches them. So the artifact appeared as a bright rectangle at the tower on **every** N/S/E/W
record and on **no** diagonal one.

Two measurements fix `x_min = 0`.

**(a) The control.** Diagonal winds are the clean case, so their retained profile near the
receptor is what a genuine footprint looks like there. Retained `|mass|` per 30 m bin, % of
`|f|`, under the old rule:

| x' bin [m] | axis-aligned | diagonal | excess |
|---|---|---|---|
| −120…−90 | 0.0000 | 0.0000 | +0.0000 |
| −90…−60 | 0.0299 | **0.0000** | +0.0299 |
| −60…−30 | 0.0365 | **0.0000** | +0.0365 |
| −30…0 | 0.0277 | **0.0000** | +0.0277 |
| 0…30 | 0.0693 | 0.0117 | +0.0576 |

The diagonal control is **exactly 0.0000%** at every bin with `x' < 0`. A genuine footprint
puts nothing downwind. The axis-aligned group carried **0.0942%** there — that is the
rectangle, and it is artifact and nothing else.

**(b) Does wrap reach positive `x'`?** A fold shifts a particle by `3660·cos(off)` along the
wind axis for a wind `off` degrees from that axis, and the particle's own displacement is
capped at 3660 m, so folded material lands at `x' ≤ 3660·(1 − cos(off))`. If that reach
mattered, records further off-axis would carry **more** mass just upwind of the receptor:

| off-axis | n | predicted reach into x' > 0 | −90…−60 | −30…0 | **0…30** | 30…60 |
|---|---|---|---|---|---|---|
| 0–2° | 75 | 2.2 m | 0.0437 | 0.0380 | **0.0801** | 0.5266 |
| 2–4° | 66 | 8.9 m | 0.0489 | 0.0400 | **0.0691** | 0.5477 |
| 4–6° | 67 | 20.0 m | 0.0270 | 0.0271 | **0.0782** | 0.5585 |
| 6–8° | 61 | 35.6 m | 0.0226 | 0.0244 | **0.0691** | 0.4398 |
| 8–10° | 56 | 55.6 m | 0.0169 | 0.0155 | **0.0784** | 0.4573 |

The `x' < 0` bins **fall** with off-axis angle — as the wind turns away from an axis the
fold's crosswind shift grows and the cone already catches it — while the `x' ∈ [0, 30)` bin is
**flat** against a predicted reach growing from 2 m to 56 m. **Wrap does not measurably reach
positive `x'`.** `x_min = 0` is sufficient, and any larger value would cut genuine near-field
mass instead.

**The near-field peak cannot have moved.** The rule is unchanged for `x' ≥ 0`, so `target_cone`
is bit-identical to the previous version everywhere upwind; `x_min` only empties the downwind
side. Splitting the within-200 m removed mass confirms it: **upwind (where the peak is) median
0.000%, max 0.127%** — that is the number the acceptance criterion was written for. The
downwind part is median 0.000%, max 1.001%, and that is the strip, not the peak.

Corpus-wide the change is small, as expected for a narrow strip: removed `|mass|` went
median 12.43% → **12.46%**.

**Asserted after the rebuild: 0 of 1366 records carry any nonzero value at `x' < 0`.**

## What it removes

| | |
|---|---|
| \|mass\| removed | p5 6.71%, median **12.46%**, p95 19.06%, max 29.40% |
| of it upwind (half-plane blind spot) | median 1.18%, p95 3.76%, max 9.39%; nonzero on 1360/1366 |
| within 200 m, UPWIND (where the peak is) | median **0.000%**, max **0.127%** |
| within 200 m, downwind (the strip) | median 0.000%, max 1.001% |
| **Kljun \|mass\| removed by the same cone** | **max over all records 0.00000%** |

The input channel is untouched, so the cone removes essentially nothing a perfect emulator
would have to reproduce.

## The integral — reported, not claimed

| | raw | cone |
|---|---|---|
| integral, median | 1.0146 | 0.9461 |
| error vs asymptote, median | +0.0524 | −0.0166 |
| **median \|error\|** | **0.1443** | **0.1467** |
| records below the asymptote | 557 | 720 |
| G2b outside [0.6, 1.5] | 65 | 62 |
| negative lobe, median | 4.80% | 1.59% |

**The median |error| degrades, 0.1443 → 0.1467, and this is stated rather than buried.**
`r(|mass| removed, raw error) = −0.490` (Spearman −0.512) — the same sign the half-plane gave:
the records that lose the most wrap were the ones already *below* the asymptote, not the
inflated ones.

**Whatever inflates the footprint integral is not the wraparound.** The advection non-closure
already fits that observation and is already measured by this project — departure from the
asymptote tracks `w_bar` at the receptor with the right sign (subsidence 1.497x, updraft
0.916x, two cases of opposite sign). Testing it needs `w_bar` per record, which the corpus
does not carry.

**The cone is an operational cleanup, not an integral correction**, and it is justified on
that basis alone: it removes material the periodic boundary invented.

## What this mask can and cannot miss

**Production retires a trajectory at one domain length, so a particle cannot wrap twice.**
Verified inside the image that generated the corpus
(`ghcr.io/tyatharva/flux-seeds:7de9dee2a01d-fe0ce48d5dff06`), all three links:

    /flux/bin/run_corpus_case.sh      passes no --max-disp at all
    /flux/bin/stage5_footprint.py     --max-disp default is None
    /flux/lpdm/driver.py:281-282      if max_disp is None: max_disp = fs.Lx

and `fs.Lx = 122 × 30 = 3660 m`. The ninth-pass validation records at this geometry carry
`max_disp_used = 3660.0` in their stage-5 JSON; the raised-cap runs are separately named
(`_3L` = 10980, `_uncapped` = 8784) and are **not** corpus cases. The eighth-pass 3L
diagnostic did not leak into production.

**The consequence is the strong result.** Every wrapped particle is displaced by exactly one
domain length in x, in y, or in both, so it lands:

| wrapped in | lands | caught? |
|---|---|---|
| x only | off-axis by ~3660·\|cos_wdir\| | yes |
| y only | off-axis by ~3660·\|sin_wdir\| | yes |
| both | off-axis and/or downwind | yes |

**There is no double-wrapped material sitting on-axis for the cone to miss.** This replaces
the half-plane's limitations section, which had to concede exactly that possibility.

The residual limits are:

1. **Genuine far-off-axis material is removed with the wrap.** `k` was chosen inside an empty
   valley, so this is measured to be nothing: 0.0110% of LES `|mass|` in `q ∈ [5, 11)`.
2. **Genuine downwind contribution is removed.** A convective boundary layer puts a little
   influence downwind and the cone cannot tell it from wrap. Measured within 200 m downwind,
   where such a contribution would sit: median 0.000%, max 1.001% of `|f|`. The diagonal
   control says a genuine footprint puts **exactly nothing** there, so on this corpus that
   number is wrap — but the mask could not distinguish them if it were not.
3. **The near-field floor is a regulariser, not physics.** It binds only for `x' ≲ 30 m`, and
   only upwind — `x_min = 0` keeps the cone empty downwind. The earlier `x' ≥ −y_min` bound
   was the bug described above.
4. **The cone is a geometric test, not a trajectory test.** It says where mass ended up, not
   how it got there.

## The clean fix, if the corpus is ever rebuilt

Deposit the **unfolded** displacement at generation time: bin each touchdown by its cumulative
displacement from the receptor rather than by its folded LES column index, and let the raster
window truncate what leaves it. Then no wrapped material is deposited at all and no mask is
needed. It requires the touchdowns, which `docs/ML_TARGETS.md` decided not to save, so it is a
generation-time change and a full corpus regeneration.

## How it ships: two files, one dataset name

    corpus_raw.h5   46.3 MB   variant="raw"    `target` = the LES field as produced
    corpus_cone.h5  33.8 MB   variant="cone"   `target` = the same field, cone-cropped

Identical layout: the same `scalars`, `kljun`, `meta` and `norm` in both, and in both the
target is called **`target`**. A loader points at one file or the other and nothing else
changes — which is the reason they are separate files rather than two datasets in one.
`corpus_cone.h5` additionally carries `grid/cone_*` (the rule, k, y_min, x_min, the σ_y
source, the commit, the timestamp) and a root `source` attribute naming the file it came from.
`meta/u_mean_ms` is in both, so σ_y — and therefore the cone — is reproducible from either.

Verified against the pre-mask backup: `corpus_raw.h5`'s `scalars`, `kljun` and `target` are
**byte-identical**; `corpus_cone.h5`'s `scalars` and `kljun` are identical to the raw file's,
its `target` equals the raw target inside the cone and is exactly zero outside, and **0 of
1366 records carry any nonzero value at `x' < 0`**.

The pipeline is `consolidate_corpus.py --out corpus_raw.h5`, then `mask_cone.py`. The
intermediate `corpus.h5` that briefly held both targets, and `target_masked` from the retired
half-plane attempt, are both gone.
