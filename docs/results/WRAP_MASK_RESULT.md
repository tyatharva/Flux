# The wraparound mask: written, validated, and NOT a fix for the integral

**2026-09-01.** Full numbers: `results/wrap_mask_validation.txt`. Per record:
`results/wrap_mask_per_record.tsv`. Figure: `figures/wrap_mask_effect.png`. Code:
`bin/mask_wrap.py`.

## What was asked

The backward LPDM bins touchdowns by LES column index, **folded modulo the periodic
domain**, so a particle running more than one domain length (3660 m) upwind reappears
*downwind* of the tower. Kljun is identically zero downwind and a real 30 m footprint has
almost nothing there. The hypothesis: **downwind mass in the target is wrap, and removing it
should pull the footprint integral back toward its asymptote** `1 − z_m/z_i`.

`case_2022020316` is the motivating case — integral **1.6102** against an asymptote of
**0.9626**, where typical records run 1.04–1.19.

## What was built

`bin/mask_wrap.py` projects every cell centre onto the record's own upwind axis
`s = x·sin_wdir + y·cos_wdir` and zeroes every cell with `s < 0`. It writes a **new**
dataset `target_masked` beside `target` — nothing is overwritten, nothing is renormalised —
and stamps the convention into `grid/` so the mask is reproducible from the file alone.
The write is copy → modify → verify on 24 random records → atomic replace.

The premise was verified rather than assumed: **max over all 1366 records of Kljun's
downwind |mass| fraction is exactly `0.000e+00`.** The input channel has nothing downwind,
so the mask removes nothing a perfect emulator would need to reproduce.

## The result: the hypothesis does not survive

| | raw | masked |
|---|---|---|
| integral, median | 1.0146 | 0.9514 |
| error vs asymptote, median | +0.0524 | −0.0106 |
| **median \|error\|** | **0.1443** | **0.1455** |
| records below the asymptote | 557 | 712 |
| G2b outside [0.6, 1.5] | 65 | 59 |
| negative lobe, median | 4.80% | 2.11% |

Three findings, in order of weight:

**1. It does not close the gap.** Masking moved 53.8% of records closer to the asymptote —
barely better than a coin toss — and the median |error| went **slightly worse**. The named
case goes 1.6102 → 1.5823 against 0.9626: it removed 8.8% of |mass| and closed **4%** of
the gap.

**2. The SIGN of the correlation refutes it.** `r(|mass| removed, raw error) = −0.496`
(Spearman −0.504). The records that lose the *most* downwind mass are the ones that were
*already below* the asymptote; the records that were most inflated lose the *least*. Wrap
double-counting predicts the opposite sign. Top decile by mass removed: median raw error
−0.125. Bottom decile: **+0.273**.

**3. What is downwind is a near-uniform offset, not the spread.** The removed region is
**0.743 positive by |mass|** — so it is not pure shot noise and some of it is plausibly
genuine wrap — but its net is a median **0.058 in integral units on every record alike**. A
near-constant 6% offset cannot explain raw errors spread from −0.25 to +0.50.

**A candidate that does fit, already measured by this project:** the advection non-closure.
`PROJECT_BRIEF.md` records that departure from the asymptote tracks `w_bar` at the receptor with
the right sign — subsidence 1.497x, updraft 0.916x, on two cases of opposite sign. That is a
vertical-velocity effect, it predicts departures of **both** signs, and both-signed is what
the raw errors are. Testing it needs `w_bar` per record, which the corpus does not carry.

## The negative lobe, and a test that came back negative

The masked negative lobe is 2.11% against 4.80% raw, so **more than half the negative lobe
lives downwind**. Steinfeld's wind-turning negatives sit to the **right** of the upstream
direction, hence upwind, hence they should survive — so the drop looks like the mask cutting
physical signal. That prediction has a side to it, so it was checked rather than asserted:
of the negative mass that *survives*, the fraction lying to the right of the upwind axis is
a median **0.534**, with a right-hand majority in only 53.8% of records.

**There is no side preference.** The corpus negative lobe does not carry the Steinfeld
signature. That cuts both ways: the large drop is *not* evidence the mask cut Steinfeld
signal, because that signal is not identifiable in the raw target either.

## Known limits of the mask — state these wherever `target_masked` is used

1. **Double wrap is not caught.** A particle displaced more than two domain lengths (7320 m)
   lands back on the *upwind* side and is indistinguishable from real near-field influence.
   Far-tail material only, at the level of the speckle.
2. **Genuine downwind contribution is removed with the wrap.** A convective boundary layer
   does put a little influence downwind. The mask cannot tell it from wrap. The removed mass
   within 200 m of the receptor — where a genuine downwind contribution would sit — is a
   median 0.000% and at most 1.00%, so this limit is small but real.
3. **Pure crosswind wrap is not caught.** A particle wrapping across the domain crosswind
   lands at `s ≥ 0` and survives. That displacement is ~10 σ_y at this domain size, so the
   mass is negligible — but it is not zero and nothing here removes it.
4. **It is a half-plane, not a trajectory test.** A statement about where mass ended up, not
   about how it got there.

## The clean fix, if the corpus is ever rebuilt

**Deposit the unfolded displacement at generation time**: bin each touchdown by its
cumulative displacement from the receptor rather than by its folded LES column index, and
let the raster window truncate what leaves it. That removes all four limits at once, because
a wrapped particle then lands where it actually went instead of where the periodic domain
put it.

It needs the touchdowns, which `docs/ML_TARGETS.md` decided not to save, so it **cannot be
done post hoc** — it is a generation-time change and therefore a full corpus regeneration.
Record it here as the proper solution; it is not worth a regeneration on its own evidence,
because the thing it would fix is not what inflates the integral.

## Status of `target_masked`

It is in `corpus.h5`, it is reproducible from `grid/`, and the original `scalars`, `kljun`
and `target` are **byte-identical to the pre-mask backup** (verified against
`~/Desktop/corpus/corpus.h5`).

**Do not train on it as a correction to the integral.** It removes ~11% of |mass| and ~6% of
the integral from every record, pushes half the corpus below the asymptote, and cuts more
than half the negative lobe. It is a defensible **ablation** — train on both and compare —
not a fix.
