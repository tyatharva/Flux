# Containment: does the footprint fit in the box?

Measured 2026-08-30 on the 24 m grid (2928 m box), the deferred gate from the seventh pass;
re-measured on the 30 m grid (3660 m) in the ninth pass. Raw output `results/containment_gate.txt`,
`results/containment_gate_targets.txt`; the decomposition that motivated it
`results/integral_decomposition.txt`.

## Why the existing gate could not ask the question

`corpus_monitor` G2a checks `|I(2L)/I(1L) − 1| ≤ 1e-3` and every case passes it, because
production retires a trajectory at one domain length, so the by-displacement curve is flat past
1 L by construction. G2a tests that the cap *binds*. Containment needs the cap raised
(`stage5_footprint.py --max-disp`), and then the curve separates influence that has genuinely
run out from influence still accumulating because the trajectory re-entered turbulence it
already sampled.

## The measurement at 2928 m

Three runs, uncapped to 3 L = 8784 m:

| displacement | flat/neutral control | convective target | near-neutral target |
|---|---|---|---|
| 0.25 L | 0.391 | 0.814 | 0.366 |
| 0.50 L | 0.608 | 0.995 | 0.573 |
| 0.75 L | 0.716 | 1.020 | 0.696 |
| **1.00 L (the cap)** | **0.785** | **1.034** | **0.750** |
| 1.50 L | 0.840 | 1.044 | 0.829 |
| 2.00 L | 0.840 | 1.043 | 0.862 |
| 3.00 L | 0.833 | 1.043 | 0.868 |

| gate | flat/neutral | convective | near-neutral |
|---|---|---|---|
| C1 saturates by 1 L | **FAIL** +8.8% | ok +1.4% | **FAIL** +7.3% |
| C2 `x80` inside the box | ok 1557 / 2342 m | ok 776 m | ok 1581 m |
| C3 what the cap hides | **FAIL** +6.1% | ok +0.8% | **FAIL** +15.8% |

The convective footprint fits in 2928 m. The neutral one does not: the flat control needs 1.5
domain lengths before its integral stops growing, and the cap removes 6.1%. **C2 passes where C1
and C3 fail**: `x80` is a crosswind-integrated 1-D measure and the missing influence lives in a
long thin tail it does not see. Scoring containment on `x80` alone would have called this
contained.

**The mitigating fact**: Kljun on the identical cells was truncated by the same amount, 0.867 of
its asymptote against the LES's 0.874. So an LES-vs-Kljun comparison on the 2928 m box was fair,
and the corpus's targets would carry a systematic about 12% truncation deficit that is a property
of the domain, shared across cases of a regime.

**And a correction it forced**: the two targets' capture curves had said `t_back` = 600 s reaches
99.6–100% of the 900 s integral; the flat/neutral control says 600 s reaches only 91.5% and 750 s
96.5%. Neutral has the longest footprint and sets `t_back`. 900 s stands.

## The options, costed

| | domain | captures of the saturated integral | cost per case |
|---|---|---|---|
| as-is | 122² at 24 m = 2928 m | 93.5% | 0.63 GPU-h |
| 146² at 24 m | 3504 m | about 96.7% | +48% |
| 186² at 24 m | 4464 m, past the 1.5 L saturation | about 100% | +132% (about 820 → 2150 GPU-h) |
| **122³ at 30 m** | **3660 m** | **97.5%** (measured in the ninth pass) | **+3%** |

The decision was 122³ at 30 m ([eighth pass](pass-8.md)): 25% more box for 3% more cost, and the
acceptance became "the neutral integral saturates by 2.5 L".

## At 3660 m: the acceptance passes, the parity does not

Ninth pass, run 4, the flat/neutral control with the cap raised to 3 L = 10,980 m: 0.712 at 1 L,
0.736 at 1.25 L, 0.738 at 1.5 L, 0.730 at 2.0, 2.5 and 3.0 L. `|I(2.5L) − I(2.0L)|/I(2.5L)` = 0.0%
against a 2% tolerance: **the neutral integral saturates by 2.5 L**, flat to three decimals from
2.0 L, and the production cap at 1 L captures 97.5% of it against 93.5% at 2928 m. The stricter
questions still fail: saturation by 1 L at +7.2%, what the cap hides at +2.5%; `x80` 1733 m
inside a 2928 m bar.

**But the LES-Kljun parity that licensed accepting truncation does not hold at 3660 m**: the LES
retains 0.765 of its asymptote against Kljun's 0.923, a 15.8-point gap where 2928 m had 0.7. Not
the Kljun fix (both implementations give 0.8263 on the 24 m control). The LES footprint is what
changed in a nearly identical flow, with the sub-grid fraction at the receptor 85.1% → 90.4% and
only 44.9% of particles reaching the surface within `t_back`. Any relative claim against Kljun on
the neutral flat control carries that gap explicitly ([limitations](../limitations-and-future-work.md),
item 4).

The convective target of the same pass, with the cap raised to 3 L, was still climbing +2.1%
over the last quarter domain, marginally at the 2% bar, with an integral 1.050× its asymptote
explained by mean subsidence at the receptor rather than by truncation.

`bin/containment_gate.py` and `results/containment_24m.txt` were removed from the tree on
2026-09-04 (at the `pre-cleanup-2026-09-04` tag); the gate's logic lives in
`bin/stage5_footprint.py --max-disp` and the by-displacement ladder it reports.
