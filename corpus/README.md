# Flux footprint corpus — 122² @ 30 m grid, 28.5 m receptor

1366 (input, target) pairs for the UW-Madison Kegonsa Solar Array footprint emulator.
Generated 2026-09-01 on 8 machines x 8 RTX 5090.

## Layout

    corpus_cone.h5  <- TRAIN ON THIS. 1366 records, 34 MB, gzip-4.
    corpus_raw.h5      the same corpus with the wraparound still in it, 46 MB
    INDEX.json         complete index: tag, datetime, split, machine, day, path
    FLAGGED.tsv        records failing a gate corpus_monitor.py already defines
    pairs_npz/         the 1366 source .npz, one per case
    logs/              the 8 orchestrator run logs
    provenance/        the 8 per-machine manifests (every one of 1945 days accounted for)

TWO FILES, IDENTICAL LAYOUT, ONE DATASET NAME. Both carry the same `scalars`, `kljun`,
`meta` and `norm`; they differ ONLY in `target`, and in both it is called `target`. So a
loader points at one file or the other and nothing else changes. Each file's root attribute
`variant` says which it is (`raw` or `cone`).

Contents of either file:

    scalars   (1366, 6)        float32   h, ustar, sigma_v, L, sin_wdir, cos_wdir
    kljun     (1366, 128, 128) float32   official FFP v1.42 on the target's own cell edges
    target    (1366, 128, 128) float32   LES + backward LPDM, SIGNED and UNCLIPPED
                                         in corpus_raw.h5:  as the pipeline produced it
                                         in corpus_cone.h5: cropped to the wind-aligned
                                                            cone. THE TRAINING TARGET.
    meta/                                datetime, run_id, split, split_index, gate_state,
                                         integral, peak_x_m, array_share, zi_achieved_m, ...
    norm/                                computed from the TRAIN SPLIT ALONE (837 records)

122 -> 128 is a zero-pad of 3 cells per side, not a resize. North-up map frame, receptor at
the centre cell (64, 64). Negative target values are physical and nothing clips them.

## Splits (hard-coded by calendar month, assigned at generation)

    train  837   2021, 2022, 2023 + 2026-02/04/06/08     (40 months)
    val    235   2024                                    (12 months)
    test   294   2025                                    (12 months)

## Read this before training

**Only ~15% of the corpus carries the site-specific signal.** The array is in the footprint
essentially only for northerly flow, and convective afternoons here favour SW/W:

    dir   corpus   site rose   mean array share
    N       6.9%      10.6%        30.3%
    NW     14.6%      14.5%         6.9%
    NE      6.4%      10.2%         2.7%
    W      21.4%      14.4%         0.3%
    SW     19.4%      14.3%         0.9%
    E       4.5%      10.4%         0.2%

202 of 1366 records (14.8%) have an array share above 5%; the median is 0.49%. Per split
that is train 14.7% / val 17.9% / test 12.6%, so every split can see it -- but an aggregate
metric over all 1366 is dominated by cases with no array in view, where Kljun and the LES
agree by construction. Weight the loss, or report the northerly subset separately.

(Measured N-wind mean array share 30.28% against the 30.7% Kljun predicts for N at
z_m = 30 m -- independent agreement to 1.4%.)

## corpus_cone.h5 -- the training set

The raw target is what the LES produced, and the wraparound in it is an artifact of periodic
boundary conditions: touchdowns are binned by LES column index and folded modulo the domain,
per axis and independently, so material running more than one domain length reappears through
a seam -- downwind of the tower, or upwind but far off the wind axis. No tower measures it and
no emulator should be asked to predict it, so the ML target does not carry it.

`bin/mask_cone.py` reads corpus_raw.h5 and writes corpus_cone.h5, zeroing everything
outside a wind-aligned cone:

    keep  <=>  x' >= 0  AND  |y'| <= max(8 * sigma_y(x'), 90 m)

sigma_y is Kljun's own, from the official FFP v1.42 -- the same call that made the `kljun`
channel. The rule, k, y_min and the sigma_y source are stamped into `grid/` (`cone_*`
attributes) so the mask is reproducible from this file alone; `meta/u_mean_ms` is carried for
the same reason.

All three parameters were measured, not picked. k = 8: the LES mass distribution against q = |y'|/sigma_y(x') is
BIMODAL with an empty valley -- 0.0110% of |mass| in q = [5, 11), rising again past q = 11 --
so the footprint is below q ~ 5, the wrap above q ~ 11, and any k in [5, 11] gives the same
answer. x_min = 0 likewise: diagonal winds scatter their wrap off-axis, and their retained
profile is EXACTLY 0.0000% at every bin with x' < 0, so a genuine footprint puts nothing
downwind; and wrap does not reach positive x' (the predicted reach grows from 2 m to 56 m
across 0-10 deg off-axis while the measured mass just upwind of the receptor is flat).

Removed mass is a median 12.46% of |f|; Kljun loses 0.00000%; removed mass within 200 m of
the receptor, UPWIND where the near-field peak is, is a median 0.000% and at most 0.127%.

It is an operational cleanup, NOT an integral correction: the median |error| against the
1 - z_m/z_i asymptote goes 0.1443 -> 0.1467. Whatever inflates the integral is not the
wraparound.

corpus_raw.h5 is retained unchanged so the raw simulation can be trained on later if
wanted; its `target` is byte-identical to what the pipeline produced.
Evidence: `results/cone_mask_validation.txt`, `results/cone_mask_per_record.tsv`,
`figures/cone_mask_effect.png`, `docs/results/CONE_MASK_RESULT.md`.

## Known gaps and caveats

* 166 days failed; six months are consequently empty: 2021-12, 2023-07, 2023-10,
  2024-01 (val), 2024-04 (val), 2026-08; 2021-06 and 2022-04 are partial. Machine 3 lost
  all eight GPUs to one fault 42% into its run. Verified NOT an input-space hole: 84-93%
  of a missing month's cases fall inside the retained months' p5-p95 on every scalar.
* TWO RECORDS WERE REJECTED and are in NEITHER file: `case_2022010915` and
  `case_2022122416` had `h` = 2371.979 m, which is `bl_depth`'s DAMP_FRAC search ceiling
  rather than a measured depth -- the model input was the bound, not the boundary layer.
  Their days are `missing` in the manifests with that reason and the .npz are kept under
  `provenance/rejected/`. Dropping them cut the train split's `h` std by 5%
  (2.379e+02 -> 2.265e+02), so they were skewing the normalisation as well.
* `FLAGGED.tsv` lists 231 records (16.9%) failing G2b (integral outside [0.6, 1.5]) or
  G3b (peak/Kljun-peak outside [0.4, 2.5]). NEITHER IS AN EXCLUSION RULE -- both are
  per-case pipeline sanity checks calibrated on a handful of validation cases, and an LES
  peak far from Kljun's may be the signal rather than an error. Ablate, do not filter.
* `provenance/PARTIAL_record_manifest_machine7_only.json` was found as
  `pairs_npz/manifest.json`. It indexes 196 of the 1368 records present at the time. Kept for provenance only.

## Reproducing / topping up

    image  ghcr.io/tyatharva/flux-seeds:7de9dee2a01d-fe0ce48d5dff06
    commit 7de9dee2a01daf12da90bab34df0db86822a17e7

The 30-seed library is baked into that image. The hour draw is seeded from the date, so
re-running a failed day reproduces exactly the case that would have been there.
