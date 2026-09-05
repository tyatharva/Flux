# Sixth pass: the `σ_w` closure, fixed and revalidated

2026-08-24, on the fifth pass's 16 m grid with a 10 m receptor. The convective well-mixed gate
passed in both directions for the first time. Three diagnoses were made and two were wrong. That
sequence is the result.

## What was wrong, after two wrong answers

| attempt | hypothesis | test | outcome |
|---|---|---|---|
| fifth pass | the taper manufactures a `σ_w²` maximum | rebuild so a maximum is structurally impossible | wrong: 0 turnovers, forward lowest-three 1.236 vs 1.260 |
| this pass | the drift is inconsistent (product rule, central differences) | additive delivery, bit-identical on `σ²` | wrong: still fails, and backward got worse, 1.014 → 1.152 |
| this pass | the **magnitude** of the inflation | a constant ×10 with no shape at all | **right**: fails at 1.370, worse than any shaped floor, while a constant ×1.673 passes at 1.130 |

A uniform ×10 has no taper, no turnover and `dsc/dz` exactly zero, and it fails worse than
every shaped floor tried, while the unmodified model passes both directions. Every hypothesis
about profile shape was excluded by two runs that cost 24 minutes of CPU. And the floor was
largest exactly where it was least justified:

| z [m] | resolved `ww` | `(2/3)e` | resolved fraction | old factor | new factor |
|---|---|---|---|---|---|
| 2.0 | 0.0012 | 0.3152 | 0.4% | 1.07 | 1.07 |
| 18.1 | 0.0979 | 0.1267 | 43.6% | 3.28 | 2.53 |
| 34.6 | 0.2495 | 0.0494 | 83.5% | 8.06 | 3.41 |
| 52.1 | 0.3776 | 0.0332 | 91.9% | **10.11** | 1.74 |
| 114.3 | 0.6138 | 0.0241 | 96.2% | 4.14 | 1.12 |
| 200.4 | 0.6872 | 0.0232 | 96.7% | 1.00 | 1.00 |

## The fix

**Sub-grid-fraction weighting**: `sc_eff = 1 + (sc − 1)·f_sgs` with `f_sgs = (2/3)e / (ww + (2/3)e)`.
The floor's justification is unresolved sub-filter variance, so it scales with how much of the
variance is sub-filter. Monotonicity is re-imposed afterwards because the two do not commute.

**`eps` scaled with `σ²`**: energy asserted at the filter scale should dissipate faster.
Leaving `eps` alone inflated `T_L = 2σ²/(C0·eps)` by the floor's own factor, so at `sc = 10` the
sub-grid memory length grew tenfold and the adaptive step saturated at `dt_max`.

Also fixed, and neither was the cause: the gate had its own drifted copy of the floor
without the displacement correction, and `dsc/dz` was a central difference of a
piecewise-linear interpolant rather than its exact derivative. The floor is now in one
place, `lpdm/sgs_floor.py:most_floor()`, and the gate and the footprints both import it.

## Gate D1: both regimes, both directions, three closures on identical fields

Counting-noise floor 5.48%.

| regime | closure | backward rms / lowest 3 | forward rms / lowest 3 | max factor | at receptor | turnovers | |
|---|---|---|---|---|---|---|---|
| neutral | no floor | 4.61% / 1.001 | 3.96% / 1.041 | | | | PASS |
| neutral | **production** | 5.00% / 0.989 | 4.04% / 1.023 | 1.23 | 1.000 | 0 | **PASS** |
| neutral | retired taper | 4.98% / 0.967 | 5.30% / 1.047 | 2.45 | 1.000 | 9 | PASS |
| convective | no floor | 3.75% / 1.027 | 6.37% / 1.091 | | | | PASS |
| convective | **production** | 4.57% / 1.037 | 6.89% / 1.097 | 3.49 | 1.591 | 0 | **PASS** |
| convective | retired taper | 6.88% / 0.961 | 11.34% / **1.226** | 12.17 | 1.673 | 11 | **FAIL** |

The production closure is within counting noise of the unmodified model in both regimes and
directions while supplying 3.49× where the variance is sub-grid. **The neutral gate still
passes the retired closure**, with nine turnovers and a factor of 2.45. That is not a weaker
version of the convective test but a test of a different model, because the floor is inert
neutrally. That became [standing rule 6](../reference/standing-rules.md). Gate D2: the flat
controls converge from below and are quoted against Kljun on identical cells, never against 1
(neutral 0.880 vs 0.956, convective 0.897 vs 0.950). The standing regression passed against
the fifth-pass baseline.

## What the retired closure changed

Same fields, releases and seed. Only the floor differs:

| case | array % new | array % legacy | difference | `x80` new | `x80` legacy |
|---|---|---|---|---|---|
| nbl_wN | 54.98 | 53.81 | −1.17 | 302 | 308 |
| nbl_wE | 50.86 | 51.26 | +0.40 | 291 | 272 |
| nbl_wS | 55.96 | 56.76 | +0.80 | 285 | 267 |
| nbl_wW | 37.05 | 38.44 | +1.39 | 421 | 378 |
| cbl_wN | 75.24 | 77.50 | +2.25 | 153 | 131 |
| **cbl_wE** | 64.18 | **82.64** | **+18.46** | 118 | **52** |
| **cbl_wS** | 75.66 | **87.46** | **+11.81** | 122 | **65** |
| cbl_wW | 47.08 | 49.38 | +2.30 | 149 | 129 |

The retired closure inflated the convective array share by +8.71 points on average and +18.46
at worst, against a neutral effect of +0.35. On the easterly it put 80% of the footprint inside
52 m against 118 m. The fifth pass had quoted a "2–4%" systematic uncertainty, estimated from
the integral overshoot, which is a poor proxy because the closure redistributes the footprint
radially without changing how much lands. And **the floor itself, correctly applied, adds
+8.40 points of convective array share and shortens `x80` from 400 to 227 m** (no floor 23.52%
and 400 m. Production 31.92% and 227 m on the flat convective control).

## Production, regenerated

Weighted floor, `eps`-consistent, `--raise-topo`, receptor at a fractional level 8.500 m above
the raised surface = 10.000 m above bare ground. Neutral array shares rose 13.9 points on
average, and that is `--raise-topo`, not the closure. At `z0_array = 0.10 m` the array was
aerodynamically identical to the cropland it replaced, so its entire neutral signal was zero.
`z0_array = 0.25 m` and `topoPos` raised by `d` restore a 2.5× roughness contrast and put the
first model level above panel top. Convective integrals 0.971–1.193, neutral 0.905–1.095, both
near 1. Values above 1 over the real surface are the advective non-closure (the by-displacement
decomposition saturates flat beyond one domain length), and the flat controls converge from
below.

**The compaction ratio, deconfounded.** The fifth pass's 2.57× compared a neutral case with an
inert floor against a convective one with a factor-10 floor. With the same closure on both sides
it is 1.88×. A quarter of the apparent compaction was the closure.

## The compaction ratio is closure-dependent and its sign does not survive

Flat controls, 80% source area (2-D, the metric to settle it on):

| regime | closure | A80 [ha] | A50 [ha] | `x80` [m] |
|---|---|---|---|---|
| neutral | floor off | 3.763 | 0.461 | 433 |
| neutral | floor on | 3.814 | 0.461 | 440 |
| convective | floor off | **6.579** | 0.819 | 400 |
| convective | floor on | **2.867** | 0.461 | 227 |

On area the ratio is 0.57× (convective broader) with the floor off and 1.33× (more compact) with
it on. The sign flips. `x80` cannot see it because the no-floor convective footprint is broader
crosswind as well as long. The flip is one-sided (the floor changes convective A80 by −56.4%
and neutral by +1.4%), and the paired null is the neutral row. Both closures pass Gate D1,
because well-mixedness tests self-consistency, not whether `σ_w` has the right magnitude.
What separates them is the resolved fraction at the receptor. Neutrally the LES's total `σ_w`
already meets surface-layer similarity (repair ×1.000). Convectively it is 23% short. So the
no-floor convective case is the under-resolution artifact, not a physical alternative. **Quote
the compaction ratio with its closure. The band across the closure choice is 0.57× to 1.33×, and
it changes the sign.** Resolving the claim needs a grid where the receptor is resolved
(`Δ ≲ 2.9 m`, about 22×, out of reach) or eddy-covariance `σ_w` from the tower itself.

## Infrastructure, because two silent bugs got through

`bin/preflight.sh`. Every analysis step asserts its output exists. `bin/test_parallel_lpdm.py`
asserts the forked LPDM is bit-identical to the serial one (16 chunks with per-chunk seeds,
6.8× on 12 workers, field cache shared copy-on-write). Resumable sampling windows stamped by
configuration. The campaign runs the GPU and the CPU at once.

## Limitations at this configuration

The receptor was inside the roughness sublayer over the array. The first model level was
1.997 m above the raised surface. `z/Δ = 0.99`. Tree cells have `ln(z_first/z0) = 0.69`. These
are the reasons the [seventh pass](pass-7.md) retired the 10 m receptor.

Removed from the tree on 2026-09-04 (in the offline pre-cleanup archive of 2026-09-04): `bin/run_pass6*.sh`,
`bin/floor_bias.py`, `bin/compaction_check.py`, `bin/window_stationarity.py`,
`bin/pass6_tables.py`, `results/g16p6*`, `results/g16r_*`, `results/pass6_*`. Kept:
`lpdm/sgs_floor.py`, `bin/test_sgs_floor.py`, `bin/test_parallel_lpdm.py`, `bin/stage4_wellmixed.py`,
`results/cbl_*.npz` (read by `bin/test_negative_lobes.py`).
