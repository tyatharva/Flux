# Figures

    cone/                     the nine pair figures on corpus_cone.h5. THE TRAINING SET.
    raw/                      the same nine on corpus_raw.h5, wraparound and all
    cone_mask_effect.png      how the cone was derived and what it did

Regenerate everything from the two corpus files alone:

    D="docker run --rm -v $PWD:/w -w /w -u $(id -u):$(id -g) -e MPLCONFIGDIR=/tmp/mpl \
       ghcr.io/tyatharva/flux-seeds:7de9dee2a01d-fe0ce48d5dff06"
    $D python3 bin/fig_corpus_pairs.py --h5 corpus/corpus_cone.h5 \
                                       --outdir figures/cone               # -> cone/
    $D python3 bin/fig_corpus_pairs.py --h5 corpus/corpus_raw.h5  \
                                       --outdir figures/raw                # -> raw/
    $D python3 bin/fig_cone_mask.py                                        # -> the PNG

The host python has no h5py, scipy or matplotlib, so this runs in-image like every other
analysis. `bin/fig_corpus_pairs.py` runs no LES and no LPDM — it opens the same file the
training loader will open, so a pair that is wrong here is wrong in the dataset.

## `cone/` and `raw/` — the (input, target) pairs

`cone/` plots corpus_cone.h5, the training set. `raw/` plots corpus_raw.h5, the raw LES
output with the periodic wraparound still in it. The two files have identical layout and
both call the target `target`, so these are the same nine figures on the same axes and the
two directories diff by eye.

| file | what it shows |
|---|---|
| `pair_anatomy_array.png` | one pair taken apart, for the record with the most solar array in the footprint (`case_2022020316`, 53.1%) |
| `pair_anatomy_typical.png` | the same anatomy for the **median** array-share record, so the layout is not a hand-picked success |
| `pairs_by_direction.png` | one pair per 45° wind sector — the frame check: the footprint swings with the wind while the array and the lake stay put |
| `pairs_array_signal.png` | the six records with the most array in view: the site-specific signal the emulator exists to learn |
| `pairs_random_{train,val,test}.png` | six **unselected** records per split (seeded draw), so this is what each split actually looks like |
| `corpus_inputs.png` | the six input scalars by split, the corpus wind rose, and array share against direction |
| `pairs_sanity.png` | corpus-wide checks: the G2b and G3b windows, the negative lobe, the zero pad, and the mean input beside the mean target |

### How to read a pair panel

Every raster is in the frame the corpus stores: **north-up map**, 30 m cells, receptor at
the centre of cell (64, 64), 122 real cells zero-padded to 128. The frame is *not*
wind-aligned — that is deliberate, and it is the first thing to check by eye.

* **green rectangle** — the solar array. It is a rectangle in EPSG:3071 and the tower is
  inside it, so it is in the *same place in all 1366 records*. If it moves, the frame is wrong.
* **cyan outline** — Lake Kegonsa. Also fixed.
* **star** — the receptor, at the origin.
* **dotted square** — the boundary of the 122 real cells; outside it is the zero pad.
* **arrow** — the mean flow. The source area must lie on the *other* side of it.
* **white contours** — 50% and 80% source area.
* **dashed cyan** — where the signed target is negative. Nothing clips it; the negative lobe
  is a median 4.8% of `|f|` raw and 1.6% after the cone.

The INPUT and TARGET panels of a row **share one colour scale**, spanning four decades below
the larger of the two peaks. Panels are not renormalised: the absolute scale is an input to
the loss, so it is what is plotted, and the integral is printed on the target instead.

In `raw/`, the speckled lobes off the wind axis and downwind are periodic **wrap**, not a
second footprint: touchdowns are binned by LES column index and folded modulo the domain, per
axis and independently, so a trajectory running more than one domain length reappears through
a seam. They are gone in `cone/`.

### What the sanity figure asserts

`bin/fig_corpus_pairs.py` re-derives the two gates `bin/corpus_monitor.py` defines and prints
them beside `corpus/FLAGGED.tsv`, which is the record of what the pipeline actually said:

    zero pad max |value|      0.000e+00  (exactly zero)
    outside G2b [0.6, 1.5]    65 of 1366   (FLAGGED.tsv: 65)
    outside G3b [0.4, 2.5]    187 of 1366  (FLAGGED.tsv: 187)
    median negative lobe      4.80% of |f|

Both counts reproduce the file exactly, which is what licenses the wind-axis reconstruction
the figures use. **G3b is a peak DISTANCE ratio, not a peak amplitude ratio** — the amplitude
ratio is a different number, is reported in the same panel, and nothing thresholds it.
Neither gate is an exclusion rule; see `corpus/README.md`.

## `cone_mask_effect.png` — how the cone was derived

`bin/mask_cone.py` reads `corpus_raw.h5` and writes `corpus_cone.h5`: everything outside a
wind-aligned cone `x' ≥ 0` and `|y'| ≤ max(8·σ_y(x'), 90 m)` is set to zero, where σ_y is
Kljun's own, from the corpus's own input channel.

The panel that justifies `k = 8` is the middle-left one. The LES mass distribution against
`q = |y'|/σ_y(x')` is **bimodal with an empty valley**: 0.0110% of `|mass|` in `q ∈ [5, 11)`,
then it rises again past `q ≈ 11`. The footprint is below `q ≈ 5`, the wrap above `q ≈ 11`,
and `k` sits between them — which is why removed mass moves only 0.4 pp across a factor of
four in `k`.

The cone also catches what a downwind half-plane cut could not. The periodic fold is per axis
and independent, so a particle that wraps in one axis alone lands back **upwind** as an
off-axis streak: a median 1.18% of `|f|`, up to 9.39%, on 1360 of 1366 records.

It is an **operational cleanup, not an integral correction** — the median |error| against the
`1 − z_m/z_i` asymptote goes 0.1443 → 0.1467, i.e. slightly worse, and that is stated rather
than hidden. Full write-up: `docs/results/CONE_MASK_RESULT.md`.

## What is not here any more

The 22 figures from the LES development passes were removed on 2026-09-01. They were made on
retired grids and, several of them, on the retired `sigma_w` closure, so they were superseded
on every absolute number; their inputs under `runs/*/window/` had already gone in the storage
cleanup, so they could not be regenerated in place either. The passes they illustrated are
written up in `docs/results/`, and `bin/make_figures.py`, `bin/fig_static.py`,
`bin/fig_gate6.py` and `bin/fig_closure.py` are still here if a regenerated window ever makes
them worth redrawing.
