# The first corpus case: 2023-03-10 14:00 UTC, end to end

**2026-08-26.** The first (input, target) training pair produced by the whole pipeline —
HRRR sounding -> forcing -> per-case surface -> seed pick -> rotate + inject -> one
continuous 4200 s LES -> backward LPDM -> record. It ran **twice**: the first run produced a
complete, plausible pair whose corpus input `h` was the domain top and whose `sigma_w` floor
was consequently inflated up to 90,000x aloft. That is written up below because it is the
more useful half of the result.

## The run

| | |
|---|---|
| valid time | **2023-03-10 14:00 UTC** (08:00 CST) |
| seed | `seed_nbl-shallow_a000` **rot 3**, heading 341.7 deg (`--available-only`) |
| schedule | 1800 s adjustment + 600 s `t_back` + 1800 s releases = **4200 s = 1.1667 sim-h**, ONE invocation |
| steps / dumps | 287,280 / 841 written, **360 discarded**, 481 survive (8.4 GB) |
| wall | LES 19:31->19:40 window complete; **~69 min**, against a 74.2 min class -> **IN CLASS** |
| standing checks | `k0/k1` **0.502**; `turb_alive` **OK**, final/peak 100%; 0 `CORRUPTED` |
| surface, asserted on the RUN | window `z0m` matches the case grid to **1.5e-09**, `htFlux` to **3.3e-08**, array z0 **0.250 vs 0.100 = 2.50x** |
| releases | 450 x 700 = **315,000 particles**, 704,133 touchdowns; release period 1799.9995 s against 1800 asked (margin -5.0e-04 s, tolerance 0.5 s) |

## The footprint

| | LES + LPDM | Kljun on the identical cells |
|---|---|---|
| peak | **48 m** | 64 m |
| 80% of `f_y` within | **189 m** | 215 m |
| 80% source area | **1.0 ha** | 1.6 ha |
| centroid | **134 m at 320.1 deg** | 147 m at 331.1 deg |
| integral over the raster | **1.040** | 0.952 |
| 80% / 50% source-area overlap | **51% / 64%** | — |

**Array share 68.44% unwrapped (67.23% folded)**, against 1.03% of the box by area — a
**66x enrichment**. Over 10 independent release groups: mean **70.77%, sd 11.58, se 3.66**,
range 57.4-93.1. Kljun's own bearing curve at this heading and receptor height predicts
~79%, so the LES puts less of the footprint on the array than Kljun does, by about 3 of its
own standard errors.

**Kljun is DESCRIPTIVE here, not a target.** The receptor is inside the array and inside the
roughness sublayer, where MOST does not hold.

**On the integral being 1.040.** It **saturates** — 0.949 at 0.25 L, 1.017 at 0.50 L, 1.031
at 0.75 L, 1.040 at 1.00 L, and 1.040 at 1.5 L and 2.0 L — so it is not the runaway that
PROJECT_BRIEF.md says can only be a model inconsistency. The receptor sits at ground **-5.43 m**,
in a local depression, with a model-frame mean `w` of **-0.0976 m/s**; the double rotation
removes 96.2% of it and leaves **+0.00367 m/s** in the streamline frame. Over sloping ground
the turbulent flux genuinely is not the surface flux, and the residual is `w_bar` times the
concentration integral. For scale, this window's own half-vs-half floor is 80% overlap
**49%** and a centroid difference of **40 m**, so 4% on the integral sits well inside the
sampling noise of the thing it is measured from.

## Closure, and the receptor

| | |
|---|---|
| `sigma_w` floor factor at the receptor | **1.000 — inactive** |
| floor factor over the column | **1.00-1.31**, active below **86 m** (the model's own `sigma_w^2` peak) |
| floor-induced turnovers in `sigma_w^2` | **0** (must be 0) |
| resolved fraction of `sigma_w^2` at the receptor | **0.075** from the window mean (`sigma_w` 0.472, resolved 0.129), **0.093** from the final dump — i.e. **91-93% sub-grid** |
| `sigma_w/u*` | **1.40** against the surface-layer target 1.25 |

## The external check: LES `sigma_w` against the instrument

`data/raw/H_and_sigma_w.csv`, one year of half-hourly EC at this receptor, not used for
training, tuning or forcing. This case's sensible `H = 5.5 W/m2`:

| | p5 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|
| tower `sigma_w` at `H = +6 +/- 10 W/m2` (n = 3843) | 0.031 | 0.058 | **0.155** | 0.410 | 0.783 m/s |

**LES `sigma_w`(10 m) = 0.472 m/s — the 80th percentile, and just outside the IQR**
(3.05x the median). **That is expected and is not evidence of an LES error**, for a reason
the file itself cannot control for: `sigma_w` scales with `u*`, the record carries no wind
speed, and the `H ~ 6 W/m2` population is dominated by calm nights — its IQR spans a factor
of **7.08**. This case is a windy near-neutral morning at `u* = 0.337`, and MOST at that
`u*` predicts `1.25 u* = 0.421 m/s` against the LES's 0.472, i.e. **+12%**. The instrument
comparison is an order-of-magnitude check and it passes; the 12% against similarity is the
sharper number and it is the one to quote.

## Seed mismatch, and what 30 minutes actually absorbed

**The first measurement of the deferred adjustment study, on a real case.**

| axis | requested | seed at restart | gap | achieved | gap | closed |
|---|---|---|---|---|---|---|
| direction (deg) | 352.98 | 341.72 | **-11.26** | 331.19 | **-21.79** | **-10.53** |
| `z_i` (m) | 510.2 | 364.4 | **-145.8** | 559.0 | **+48.8** | **+97.0** |
| `G` (m/s) | 8.84 | 8.00 | -0.84 | — | — | reported, never costed |

**Direction did not close — it WIDENED by 10.5 deg.** The seed was still backing at
-3.15 %/h (~-8 deg/h) when it was frozen, and 30 min is 2.8% of a 17.6 h inertial period, so
the mean flow carried on turning the way it was already going rather than toward the case's
forcing. `bin/pick_seed.py` budgets "no, ~2.7 deg" of closure on this axis; the measured
value is **-10.5**, i.e. the axis is not merely uncloseable but can move away.

**`z_i` over-closed**: the 146 m gap shut and overshot by +48.8 m, a deepening of **292 m/h**
against the **+79 m/h** `pick_seed` budgets. The cause is in the two lids — the seed's cap is
+8 K/100 m at 300 m, while this case's own fitted profile is 2.61 K/km to 354 m and 6.03
above, a far weaker lid, so a 389 m seed placed under it deepens fast.

**Neither makes the pair wrong.** Inputs are read off the LES window, so the case simply
lands at (`h` 559, `wdir` 331.2) instead of (510, 353.0). What it does mean is that the seed
library's spacing buys **less** convergence on direction and **more** on depth than the
design assumed, and the corpus's coverage in direction is set by the seeds themselves rather
than by the requested headings.

## What the first run got wrong, and why it is the useful half

The first run produced a complete pair with **`h` = 2500 m — the top of the column**. The
depth estimator searched upward from the TKE peak for a first crossing, which assumes
monotone decay; this window decays to 0.058 m2/s2 at ~560 m and then **rises again to 0.33
at 1700 m**, almost entirely resolved `w` with no sub-grid part — internal waves in the
stable free atmosphere. The search walked through the boundary layer and fell through to
`z[-1]`.

`h` is not only a training feature: it sets the `sigma_w` floor's mixed-layer blend. Measured
with the production floor on the same profile:

| z | 18 m | 35 m | 52 m | 92 m | 200 m | 428 m | 559 m |
|---|---|---|---|---|---|---|---|
| factor, `h = 2500` (as it ran) | 1.47 | 3.23 | 4.70 | 7.50 | 20.0 | 523 | **90646** |
| factor, `h = 559` (corrected) | 1.00 | 1.30 | 1.13 | 1.00 | 1.00 | 1.00 | 1.00 |

**The factor at the receptor was 1.000 either way, which is why nothing complained** — but a
600 s backward trajectory released at 8.5 m samples 18-200 m, and that is the near field the
footprint is made of. What it cost, first run against second:

| | wrong `h` | corrected |
|---|---|---|
| peak | 64 m | **48 m** |
| 80% area | 1.2 ha | **1.0 ha** |
| integral | 1.010 | **1.040** |
| array share (unwrapped) | 67.68% | **68.44%** |

So the **array share barely moved** (0.8 points against a 3.66-point SE) while the
**near-field peak moved a full raster cell and A80 shrank 17%** — which is precisely
PROJECT_BRIEF.md's standing statement that the near field is closure-dominated and the shares are
comparatively robust.

**Two guards now exist because the run printed `1.00-381935.02 over the column` and carried
on**: `h` is refused if it reaches the top of the column, and the driver warns when the floor
exceeds 100x anywhere. `lpdm/les_stats.py:bl_depth` bounds every depth search by the decay
minimum, and `bin/seed_stationarity.py` imports it rather than keeping a copy.

## Files

`pairs/case_2023031014.json` + `pairs/index.jsonl`, `results/corpus/case_2023031014.{json,npz,txt}`,
`results/forcing/case_2023031014.json`, `results/soundings/case_2023031014.json`,
`results/pick/case_2023031014.json`.
