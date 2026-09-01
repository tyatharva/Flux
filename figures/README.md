# Figures

    raw/                    the nine pair figures on the corpus target. THE DEFAULT SET.
    masked/                   the same nine on target_masked (the ablation, not a fix)
    wrap_mask_effect.png      why raw/ is still the default
    old/                      the 22 LES-development-pass figures

Regenerate everything from `corpus/corpus.h5` alone:

    D="docker run --rm -v $PWD:/w -w /w -u $(id -u):$(id -g) -e MPLCONFIGDIR=/tmp/mpl \
       ghcr.io/tyatharva/flux-seeds:7de9dee2a01d-fe0ce48d5dff06"
    $D python3 bin/fig_corpus_pairs.py                                     # -> raw/
    $D python3 bin/fig_corpus_pairs.py --target target_masked \
                                       --outdir figures/masked             # -> masked/
    $D python3 bin/fig_wrap_mask.py                                        # -> the PNG

The host python has no h5py, scipy or matplotlib, so this runs in-image like every other
analysis. `bin/fig_corpus_pairs.py` runs no LES and no LPDM — it opens the same file the
training loader will open, so a pair that is wrong here is wrong in the dataset.

## `raw/` — the (input, target) pairs, on the corpus target

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

Every raster is in the frame `corpus.h5` stores: **north-up map**, 30 m cells, receptor at
the centre of cell (64, 64), 122 real cells zero-padded to 128. The frame is *not*
wind-aligned — that is deliberate, and it is the first thing to check by eye.

* **green rectangle** — the solar array. It is a rectangle in EPSG:3071 and the tower is
  inside it, so it is in the *same place in all 1366 records*. If it moves, the frame is wrong.
* **cyan outline** — Lake Kegonsa. Also fixed.
* **star** — the receptor, at the origin.
* **dotted square** — the boundary of the 122 real cells; outside it is the zero pad.
* **arrow** — the mean flow. The source area must lie on the *other* side of it.
* **white contours** — 50% and 80% source area.
* **dashed cyan** — where the signed target is negative. The negative lobe is physical and
  nothing clips it; it carries a median 4.8% of `|f|`.

The INPUT and TARGET panels of a row **share one colour scale**, spanning four decades below
the larger of the two peaks. Panels are not renormalised: the absolute scale is an input to
the loss, so it is what is plotted, and the integral is printed on the target instead.

The faint speckled lobe **downwind** of the target is not a second footprint. Touchdowns are
binned by LES column index and folded modulo the periodic domain, so trajectories running
more than one domain length upwind reappear through the seam; displacement is capped at one
domain length, which is what bounds it.

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

## `masked/` and `wrap_mask_effect.png` — the wraparound ablation

`bin/mask_wrap.py` writes a second target into `corpus.h5`, `target_masked`, with every cell
on the **downwind** side of the receptor set to zero. The idea was that downwind mass is
periodic wrap — Kljun is identically zero downwind, verified at exactly `0.000e+00` over all
1366 records — and that removing it would pull the footprint integral back to its
`1 − z_m/z_i` asymptote.

**It does not.** `wrap_mask_effect.png` is the evidence, and its decisive panel is the middle
one of row 2: `r(|mass| removed, raw error) = −0.496`. The records that lose the most
downwind mass are the ones that were *already below* the asymptote. Wrap double-counting
predicts the opposite sign. Median |error| goes 0.144 → 0.146, i.e. slightly worse, and half
the corpus ends up below the asymptote.

`masked/` is the same nine figures on `target_masked`, for comparison. **Train on `target`.**
`target_masked` is a defensible ablation, not a correction. Full write-up:
`docs/results/WRAP_MASK_RESULT.md`.

## `old/`

The 22 figures from the LES development passes — closure experiments, static-case
comparisons, gate-6 panels, the 16 m and 24 m grid transects. They are the record of the
passes written up in `docs/results/` and are **superseded on absolute numbers**: they were
made on retired grids and, for several of them, on the retired `sigma_w` closure. Kept
because the write-ups that cite them are kept. Their inputs under `runs/*/window/` were
removed in the 2026-09-01 cleanup (`results/CLEANUP_INVENTORY.txt`), so regenerating them
needs a regenerated window first; `bin/make_figures.py`, `bin/fig_static.py`,
`bin/fig_gate6.py` and `bin/fig_closure.py` are the scripts that made them.
