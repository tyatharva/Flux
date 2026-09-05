# The first corpus case: 2023-03-10 14:00 UTC

2026-08-26, on the 16 m grid with a 10 m receptor. The first (input, target) pair the whole
pipeline produced end to end: HRRR sounding → forcing → per-case surface → seed pick → rotate
and inject → one continuous 4200 s LES → backward LPDM → record. It ran twice, and the first run
is the more useful half of the result.

## Why this hour

Chosen from 2928 enumerated candidate hours (`results/candidates.tsv`). The seed
`seed_nbl-shallow_a000` had just been accepted ([seed rungs](seed-rungs.md)), so the case had
to be one it could serve: neutral (`w'θ_v'` +0.0054 K m/s), with the seed's achieved heading
251.7° putting rotation 3 at 341.7° against the case's predicted 353.0°, an 11.3° gap inside
the 15° half-spacing. Near-neutral (`z_m/L` −0.0043), `z_i` 510 m, `dz_i/dt` +4.1 %/h. And the
direction was the binding criterion: at a 10 m receptor Kljun's array share on the real map runs
84.1% for due north and 37.8% for east or west, and every alternative the seed reached sat on
the E/W axis. This one, 7° off north at about 79%, carried roughly double the array signal.
Forcing match `G = 8.84 m/s` against the rung's 8.0; base-state fit 0.042 K rms; no warnings.
The array's thermal contrast was +0.0020 K m/s (about 2.3 W/m²), 24× weaker than a midday
case, so this was very nearly the purely aerodynamic array case (`z0` 0.250 vs 0.100).

## The run

1800 s adjustment + 600 s `t_back` + 1800 s of releases = 4200 s = 1.1667 sim-h in one
invocation; 287,280 steps, 841 dumps written, 360 discarded, 481 kept (8.4 GB); about 69 min
of wall against a 74.2 min class. `k0/k1` 0.502, `turb_alive` OK (final/peak 100%), 0 `CORRUPTED`;
the window's `z0m` matched the case grid to 1.5e-9 and `htFlux` to 3.3e-8, array `z0` 0.250 vs
0.100 asserted on the run. 450 × 700 = 315,000 particles, 704,133 touchdowns; release period
1799.9995 s against 1800 asked, inside a 0.5 s tolerance.

## The footprint

| | LES + LPDM | Kljun on identical cells |
|---|---|---|
| peak | **48 m** | 64 m |
| 80% of `f_y` within | 189 m | 215 m |
| 80% source area | 1.0 ha | 1.6 ha |
| centroid | 134 m at 320.1° | 147 m at 331.1° |
| integral over the raster | 1.040 | 0.952 |
| 80% / 50% overlap | 51% / 64% | |

Array share 68.44% unwrapped (67.23% folded) against 1.03% of the box by area, a 66× enrichment;
over 10 independent release groups 70.77% ± 3.66 (sd 11.58, range 57.4–93.1). Kljun's own bearing
curve predicts about 79%, so the LES puts less on the array than Kljun does, by about 3 of its
own standard errors. Kljun is descriptive here, not a target: the receptor is inside the array
and inside the roughness sublayer, where MOST does not hold.

The integral of 1.040 saturates (0.949 at 0.25 L, 1.017 at 0.5 L, 1.040 at 1.0 L and beyond),
so it is not a wraparound runaway. The receptor sits in a local depression at −5.43 m with a
model-frame mean `w` of −0.0976 m/s; the double rotation removes 96.2% of it. Over sloping
ground the turbulent flux genuinely is not the surface flux. Closure: floor factor 1.000 at the
receptor (inactive), 1.00–1.31 over the column, 0 turnovers; 91–93% of `σ_w²` sub-grid at the
receptor; `σ_w/u*` 1.40 against the surface-layer 1.25.

**The external check.** One year of half-hourly eddy-covariance `σ_w` at this receptor
(`data/raw/H_and_sigma_w.csv`, never used for training or forcing): at `H = 6 ± 10 W/m²`
(n = 3843) the tower's `σ_w` runs p5 0.031, median 0.155, p95 0.783 m/s. The LES's 0.472 m/s is
the 80th percentile, 3.05× the median, and that is expected: the file carries no wind speed, the
low-flux population is dominated by calm nights (IQR a factor of 7.08), and this is a windy
near-neutral morning at `u* = 0.337`, where MOST predicts `1.25 u*` = 0.421 m/s against the LES's
0.472, +12%. The 12% against similarity is the sharper number.

## Seed mismatch, and what 30 minutes absorbed

The first measurement of the deferred adjustment study on a real case:

| axis | requested | seed at restart | gap | achieved | gap | closed |
|---|---|---|---|---|---|---|
| direction | 352.98° | 341.72° | −11.26° | 331.19° | −21.79° | **−10.53°** |
| `z_i` | 510.2 m | 364.4 m | −145.8 m | 559.0 m | +48.8 m | **+97.0 m** |
| `G` | 8.84 m/s | 8.00 | −0.84 | | | reported, never costed |

Direction did not close; it widened by 10.5° because the seed was still backing at about
−8°/h when frozen and 30 min is 2.8% of a 17.6 h inertial period. `z_i` over-closed: 292 m/h of
deepening against the +79 m/h budgeted, because the case's own lid (2.61 K/km to 354 m) is far
weaker than the seed's (+8 K/100 m). Neither makes the pair wrong; inputs are read off the LES
window, so the case lands at (`h` 559, `wdir` 331.2°) instead of (510, 353.0°). The library's
spacing buys less convergence on direction and more on depth than designed, which is why the
production library went to six base angles and `pick_seed.py` projects the seed's drift forward.

## What the first run got wrong

The first run produced a complete pair with `h = 2500 m`, the top of the column: the depth
estimator searched upward from the TKE peak for a first crossing, assuming monotone decay, and
this window decays to 0.058 m²/s² at about 560 m and then rises to 0.33 at 1700 m, resolved `w`
with no sub-grid part, internal waves in the stable free atmosphere. `h` sets the `σ_w` floor's
mixed-layer blend, and with `h = 2500` the floor factor ran 1.47 at 18 m, 7.50 at 92 m, 20.0 at
200 m, 523 at 428 m and 90,646 at 559 m, against 1.00–1.30 with `h = 559`. The factor at the
receptor was 1.000 either way, which is why nothing complained, but a 600 s backward trajectory
from 8.5 m samples 18–200 m.

| | wrong `h` | corrected |
|---|---|---|
| peak | 64 m | 48 m |
| 80% area | 1.2 ha | 1.0 ha |
| integral | 1.010 | 1.040 |
| array share | 67.68% | 68.44% |

The array share barely moved (0.8 points against a 3.66-point SE) while the near-field peak
moved a raster cell and A80 shrank 17%: the near field is closure-dominated and the shares are
comparatively robust. Two guards followed: `h` is refused if it reaches the top of the column,
and the driver warns when the floor exceeds 100× anywhere. `bl_depth` bounded every search by
the decay minimum, which the [ninth pass](pass-9.md) then found was itself insufficient on a
profile whose wave layer out-energised the boundary layer (traps §22).

The record itself (`case_2023031014`) was a 16 m, 10 m-receptor pair and was retired with that
configuration; `validation_pairs_retired/` was removed from the tree on 2026-09-04 and remains
in the offline pre-cleanup archive of 2026-09-04. `bin/test_floor_health.py` still reads its footprint JSON
from the gitignored `results/corpus/` as the live instance of the defect it detects.
