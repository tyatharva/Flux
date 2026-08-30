# The containment gate at 2928 m: **FAIL for neutral, PASS for convective**

**2026-08-30.** The deferred gate from the seventh pass, now measured. Full raw output in
`results/containment_gate.txt`; the decomposition that motivated it is in
`results/integral_decomposition.txt`.

## What was asked, and why the existing gate could not ask it

`corpus_monitor` G2a checks `|I(2L)/I(1L) − 1| ≤ 1e-3` and every case passes it — because
production retires a trajectory at one domain length, so the by-displacement curve is flat
past 1 L **by construction**. G2a tests that the cap BINDS. Containment needs the cap
RAISED (`stage5_footprint.py --max-disp`, new), and then the curve separates two things
that are identical at the cap: influence that has genuinely run out, and influence that is
still accumulating because the trajectory has re-entered turbulence it already sampled.

## The measurement

Three runs, all uncapped to 3 L = 8784 m.

| displacement | **flat/neutral control** | convective target | near-neutral target |
|---|---|---|---|
| 0.25 L (732 m) | 0.391 | 0.814 | 0.366 |
| 0.50 L | 0.608 | 0.995 | 0.573 |
| 0.75 L | 0.716 | 1.020 | 0.696 |
| **1.00 L (the cap)** | **0.785** | **1.034** | **0.750** |
| 1.25 L | 0.819 | 1.038 | 0.792 |
| 1.50 L | **0.840** | 1.044 | 0.829 |
| 2.00 L | 0.840 | 1.043 | 0.862 |
| 2.50 L | 0.833 | 1.043 | 0.879 |
| 3.00 L | 0.833 | 1.043 | 0.868 |

| gate | flat/neutral | convective | near-neutral |
|---|---|---|---|
| C1 integral saturates by 1 L | **FAIL** +8.8% | ok +1.4% | **FAIL** +7.3% |
| C2 x80 inside the box | ok 1557 / 2342 m | ok 776 m | ok 1581 m |
| C3 what the cap hides | **FAIL** +6.1% | ok +0.8% | **FAIL** +15.8% |

**The convective footprint fits in 2928 m. The neutral one does not.** The flat/neutral
control — the binding case, and the one PLAN.md nominates — needs **1.5 domain lengths**
before its integral stops growing, and the cap removes 6.1% of it.

## The mitigating fact, and it is a real one

**Kljun on the identical cells is truncated by the same amount.**

| | integral | / the `1 − z_m/z_i` asymptote |
|---|---|---|
| LES + LPDM | 0.833 | **0.874** |
| Kljun FFP, same box, same cells | 0.83 | **0.867** |

The two lose the same fraction to the finite domain, to within 0.7 points. So an
LES-vs-Kljun comparison on this box is fair; what is not fair is comparing either against 1
or against the asymptote. The corpus's targets carry a systematic ~12% truncation deficit
that is a **property of the domain**, shared across cases of the same regime, and an
emulator trained on them reproduces the truncated footprint self-consistently.

## C2 passes where C1 and C3 fail, and that is worth naming

`x80` is a **crosswind-integrated 1-D** measure and it fits comfortably (1557 m against a
2342 m bar). The integral does not, because the missing influence lives in a long thin tail
that a 1-D 80%-containment distance does not see. **Scoring containment on `x80` alone
would have called this contained.** PLAN.md's own wording — "the 80% source area must
resolve before the wrap cap" — is the version that would have been read that way; the
integral's approach slope is what actually answers it.

## The options, costed

| | domain | what it captures of the saturated integral | cost per case |
|---|---|---|---|
| **as-is** | 122² @ 24 m = 2928 m | 0.785 / 0.840 = **93.5%** | 0.63 GPU-h |
| 146² @ 24 m | 3504 m | ~96.7% | ~0.93 (+48%) |
| **186² @ 24 m** | **4464 m**, past the 1.5 L saturation | **~100%** | ~1.46 (+132%) |

186² is the fourth pass's own grid and keeps `(N+6) = 192` divisible by the benchmarked
`1×2×64` block. It would take the corpus from ~820 GPU-h to ~2150.

**This is a grid decision and it belongs to the user.** Recorded, not taken.

## And one correction it forces

The two targets' capture curves said `t_back = 600 s` reaches 99.6% and 100.0% of the 900 s
integral, and that was reported. **The flat/neutral control says 600 s reaches only 91.5%**,
and 750 s 96.5%. Neutral has the longest footprint in the corpus and it is the case that
sets `t_back`. **900 s stands**; 600 s would truncate neutral cases by a further 8.5%. The
earlier statement was recorded as "not changed", which is now why it did no harm.
