# Figures

```
figures/cone/                 the nine pair figures on corpus_cone.h5, the training set
figures/cone_mask_effect.png  how the cone was derived and what it did
figures/poster/               the five final emulator figures, 600 dpi (see Results)
```

## Regenerate the pair figures

Everything under `figures/cone/` comes from the corpus file alone. The host Python has no
h5py, scipy or matplotlib, so this runs in the analysis image like every other analysis:

```bash
D="docker run --rm -v $PWD:/w -w /w -u $(id -u):$(id -g) -e MPLCONFIGDIR=/tmp/mpl \
   ghcr.io/tyatharva/flux-seeds:7de9dee2a01d-fe0ce48d5dff06"
$D python3 bin/fig_corpus_pairs.py --h5 corpus/corpus_cone.h5 --outdir figures/cone
$D python3 bin/fig_corpus_pairs.py --h5 corpus/corpus_raw.h5  --outdir /tmp/raw   # the raw variant, wraparound and all
$D python3 bin/fig_cone_mask.py                                                     # -> cone_mask_effect.png
```

`bin/fig_corpus_pairs.py` runs no LES and no LPDM. It opens the same file the training loader
opens, so a pair that is wrong here is wrong in the dataset. The file's `variant` attribute
tells it which corpus it opened. The raw-variant figures are not kept in the repository. They
are the same nine figures on the same axes, so the two directories can be compared by eye.

## The nine pair figures

| file | what it shows |
|---|---|
| `pair_anatomy_array.png` | one pair taken apart, for the record with the most solar array in the footprint (`case_2022020316`, 53.1%) |
| `pair_anatomy_typical.png` | the same anatomy for the median array-share record, so the layout is not a hand-picked success |
| `pairs_by_direction.png` | one pair per 45° wind sector. The frame check: the footprint swings with the wind while the array and the lake stay in place |
| `pairs_array_signal.png` | the six records with the most array in view: the site-specific signal the emulator exists to learn |
| `pairs_random_{train,val,test}.png` | six unselected records per split (seeded draw), so this is what each split actually looks like |
| `corpus_inputs.png` | the six input scalars by split, the corpus wind rose and array share against direction |
| `pairs_sanity.png` | corpus-wide checks: the G2b and G3b windows, the negative lobe, the zero pad and the mean input beside the mean target |

## How to read a pair panel

Every raster is in the frame the corpus stores: north-up map, 30 m cells, receptor at the
centre of cell (64, 64), 122 real cells zero-padded to 128. The frame is *not* wind-aligned.
That is intended, and it is the first thing to check by eye.

- **Green rectangle**: the solar array. A rectangle in EPSG:3071 with the tower inside it, so
  it is in the same place in all 1366 records. If it moves, the frame is wrong.
- **Cyan outline**: Lake Kegonsa. Also fixed.
- **Star**: the receptor, at the origin.
- **Dotted square**: the boundary of the 122 real cells. Outside it is the zero pad.
- **Arrow**: the mean flow. The source area must be on the other side of it.
- **White contours**: 50% and 80% source area.
- **Dashed cyan**: where the signed target is negative. Nothing clips it. The negative lobe is
  a median 4.8% of |f| raw and 1.6% after the cone.

The input and target panels of a row share one colour scale, spanning four decades below the
larger of the two peaks. Panels are not renormalised. The absolute scale is an input to the
loss, so it is what is plotted, and the integral is printed on the target instead.

In the raw variant, the speckled lobes off the wind axis and downwind are periodic wrap, not a
second footprint. Touchdowns are binned by LES column index and folded modulo the domain, per
axis and independently, so a trajectory that travels more than one domain length reappears
through a seam. They are gone in `cone/`.

## What the sanity figure asserts

`bin/fig_corpus_pairs.py` re-derives the two gates `bin/corpus_monitor.py` defines and prints
them beside `corpus/FLAGGED.tsv`, the record of what the pipeline actually reported:

```
zero pad max |value|      0.000e+00  (exactly zero)
outside G2b [0.6, 1.5]    65 of 1366   (FLAGGED.tsv: 65)
outside G3b [0.4, 2.5]    187 of 1366  (FLAGGED.tsv: 187)
median negative lobe      4.80% of |f|
```

Both counts reproduce the file exactly, which is what validates the wind-axis reconstruction
the figures use. G3b is a peak *distance* ratio, not a peak amplitude ratio. The amplitude
ratio is a different number, reported in the same panel, and nothing thresholds it. Neither
gate is an exclusion rule.

## `cone_mask_effect.png`

The panel that justifies `k = 8` is the middle-left one. The LES mass distribution against
`q = |y'|/σ_y(x')` is bimodal with an empty valley (0.0110% of |mass| in `q ∈ [5, 11)`, rising
again past `q ≈ 11`). The footprint is below `q ≈ 5` and the wrap above `q ≈ 11`, with `k`
between them. That is why removed mass moves only 0.4 percentage points across a factor of
four in `k`. The full derivation is on the [cone mask](../history/cone-mask.md) page.

## Figures that were retired

The 22 figures from the LES development passes were removed on 2026-09-01. They were made on
retired grids and, several of them, on the retired `σ_w` closure, so they were superseded on
every absolute number. Their inputs under `runs/*/window/` had already gone in the storage
cleanup, so they could not be regenerated. The passes they illustrated are written up in
[Development history](../history/overview.md). The scripts that drew them (`make_figures.py`,
`fig_static.py`, `fig_gate6.py`, `fig_closure.py`) were removed with the retired record on
2026-09-04 and remain in the offline pre-cleanup archive of 2026-09-04.
