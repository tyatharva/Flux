# Sixth pass — the sigma_w closure, fixed and revalidated

**Status: COMPLETE.** The convective well-mixed gate passes in both directions for the
first time in this project. The eight production footprints are regenerated on the
corrected closure and on the raised surface, and the cost of the retired closure is
measured on identical fields rather than inferred.

Three diagnoses were made and two of them were wrong. That sequence is the result, not an
embarrassment to be tidied away, so it is recorded first.

---

## 1. What was actually wrong, after two wrong answers

| attempt | hypothesis | test | outcome |
|---|---|---|---|
| fifth pass | the taper manufactures a `sigma_w^2` maximum | rebuild so a maximum is structurally impossible | **wrong** — 0 turnovers, forward lo3 1.236 vs 1.260 |
| this pass | the drift is inconsistent (product rule, central differences) | additive delivery, bit-identical on `sigma^2` | **wrong** — still fails, and BACKWARD got worse, 1.014 -> 1.152 |
| this pass | the MAGNITUDE of the inflation | constant x10, no shape at all | **right** — fails at 1.370, worse than any shaped floor, while constant x1.673 passes at 1.130 |

The constant-factor experiment is what settled it. A uniform x10 has no taper, no turnover
and a `dsc/dz` that is exactly zero, and it fails worse than every shaped floor tried;
the unmodified model passes both directions. **Every hypothesis about profile shape was
excluded by two runs that cost 24 minutes of CPU.**

**And the floor was largest exactly where it was least justified:**

| z (m) | resolved `ww` | `(2/3)e` | resolved fraction | old factor | new factor |
|---|---|---|---|---|---|
| 2.0 | 0.0012 | 0.3152 | **0.4%** | 1.07 | 1.07 |
| 18.1 | 0.0979 | 0.1267 | 43.6% | 3.28 | 2.53 |
| 34.6 | 0.2495 | 0.0494 | **83.5%** | 8.06 | 3.41 |
| 52.1 | 0.3776 | 0.0332 | **91.9%** | **10.11** | 1.74 |
| 114.3 | 0.6138 | 0.0241 | **96.2%** | 4.14 | 1.12 |
| 200.4 | 0.6872 | 0.0232 | 96.7% | 1.00 | 1.00 |

## 2. The fix

**Sub-grid-fraction weighting.** `sc_eff = 1 + (sc - 1) f_sgs`, with
`f_sgs = (2/3)e / (ww + (2/3)e)`. The floor's justification is unresolved sub-filter
variance, so it scales with how much of the variance is sub-filter. Applied to the factor,
after which monotonicity is **re-imposed** — the two do not commute, because `f_sgs` falls
faster than the raw factor rises and their product dips.

**`eps` scaled with `sigma^2`.** Energy asserted at the filter scale should dissipate
faster, not slower. Leaving `eps` alone inflated `T_L = 2 sigma^2/(C0 eps)` by the floor's
own factor, so at `sc = 10` the sub-grid memory length grew tenfold — at `z = 35 m`,
comparable to the height itself — and the adaptive step saturated at `dt_max`.

Also fixed, and neither was the cause: the gate carried its own drifted **copy** of the
floor without the displacement correction, and `dsc/dz` was a central difference of a
piecewise-linear interpolant rather than its exact derivative.

## 3. Gate D1 — both regimes, both directions, three closures on identical fields

Counting-noise floor 5.48%.

| regime | closure | backward rms / lo3 | forward rms / lo3 | max fac | at receptor | turnovers | |
|---|---|---|---|---|---|---|---|
| neutral | no floor | 4.61% / 1.001 | 3.96% / 1.041 | — | — | — | PASS |
| neutral | **production** | 5.00% / 0.989 | 4.04% / 1.023 | 1.23 | 1.000 | **0** | **PASS** |
| neutral | retired taper | 4.98% / 0.967 | 5.30% / 1.047 | 2.45 | 1.000 | 9 | PASS |
| convective | no floor | 3.75% / 1.027 | 6.37% / 1.091 | — | — | — | PASS |
| convective | **production** | 4.57% / 1.037 | 6.89% / 1.097 | 3.49 | 1.591 | **0** | **PASS** |
| convective | retired taper | 6.88% / 0.961 | 11.34% / **1.226** | 12.17 | 1.673 | 11 | **FAIL** |

The production closure sits **within counting noise of the unmodified model** in both
regimes and both directions while supplying 3.49x where the variance is genuinely
sub-grid. That is what a floor is supposed to look like.

**The neutral gate still passes the retired closure**, which carries nine turnovers and a
factor of 2.45. It is not a weaker version of the convective test; it is a test of a
different model, because the floor is inert neutrally (receptor factor 1.000). That is now
a standing rule in PROJECT_BRIEF.md.

**Gate D2** — the flat-control integrals converge from below and are quoted against Kljun
on the identical cells, never against 1: neutral **0.880** vs Kljun 0.956; convective
**0.897** vs Kljun 0.950.

**The standing regression PASSES** against the fifth-pass baseline: peak +0 m,
area80 +0.589 of 5.000, integral -0.034 of 0.080, overlap -0.065 of 0.150.

## 4. What the retired closure was worth — same fields, same releases, same seed

Only the floor differs between the two columns, so the difference is the closure and
cannot be a different turbulence realisation.

| case | array% new | array% legacy | difference | relative | `x80` new | `x80` legacy |
|---|---|---|---|---|---|---|
| nbl_wN | 54.98 | 53.81 | **-1.17** | -2.1% | 302 | 308 |
| nbl_wE | 50.86 | 51.26 | +0.40 | +0.8% | 291 | 272 |
| nbl_wS | 55.96 | 56.76 | +0.80 | +1.4% | 285 | 267 |
| nbl_wW | 37.05 | 38.44 | +1.39 | +3.8% | 421 | 378 |
| cbl_wN | 75.24 | 77.50 | +2.25 | +3.0% | 153 | 131 |
| **cbl_wE** | 64.18 | **82.64** | **+18.46** | **+28.8%** | 118 | **52** |
| **cbl_wS** | 75.66 | **87.46** | **+11.81** | **+15.6%** | 122 | **65** |
| cbl_wW | 47.08 | 49.38 | +2.30 | +4.9% | 149 | 129 |

**The retired closure inflated the convective array share by +8.71 points on average and
by +18.46 points at worst — against a neutral effect of +0.35 points.** The mechanism is
visible in `x80`: on the easterly the retired closure put 80% of the footprint inside
**52 m** against 118 m, a 2.3x compaction onto a tower that stands inside the array.

**The fifth pass quoted a systematic uncertainty of "2-4%" on the convective array shares.
That was wrong by roughly an order of magnitude on two of the four directions.** It was
estimated from the integral overshoot, which is a poor proxy: the integrals barely move
(cbl_wE 0.971 vs 1.029) while the share moves 18 points, because the closure redistributes
the footprint radially without changing how much of it lands.

Radial distribution on the flat convective control, cumulative share of weight inside each
radius:

| closure | <50 m | <100 | <150 | **<250** | <400 | <900 | array% | `x80` |
|---|---|---|---|---|---|---|---|---|
| **production** | 22.64 | 55.04 | 68.71 | **80.93** | 87.66 | 97.84 | 31.92 | 227 m |
| retired taper | 22.73 | 56.36 | 70.77 | **82.97** | 89.40 | 98.55 | 32.27 | 206 m |
| no floor | 16.22 | 43.39 | 56.18 | **70.11** | 79.56 | 96.24 | 23.52 | 400 m |

Note the third row: **the floor itself — correctly applied — is worth +8.40 points of
convective array share and shortens `x80` from 400 to 227 m.** It is not a small
correction, which is why getting it wrong cost so much.

## 5. Production, regenerated

Weighted floor, `eps`-consistent, `--raise-topo`, receptor released at a fractional level
8.500 m above the raised surface = **10.000 m above bare ground**, `--exact-agl`.

| case | dir | array% | vs 5th pass | integral | 80% area (ha) | Kljun (ha) | ratio | `x80` |
|---|---|---|---|---|---|---|---|---|
| nbl_wN | 319.5 | 54.98 | **+13.50** | 1.095 | 2.202 | 1.562 | 1.41 | 302 |
| nbl_wE | 46.6 | 50.86 | **+14.20** | 0.905 | 1.920 | 1.613 | 1.19 | 291 |
| nbl_wS | 147.2 | 55.96 | +9.81 | 1.090 | 2.227 | 1.562 | 1.43 | 285 |
| nbl_wW | 238.4 | 37.05 | **+17.96** | 1.032 | 3.379 | 1.510 | 2.24 | 421 |
| cbl_wN | 338.7 | 75.24 | -6.25 | 1.193 | 1.408 | 1.869 | 0.75 | 153 |
| cbl_wE | 70.0 | 64.18 | +0.77 | 0.971 | 1.152 | 1.997 | 0.58 | 118 |
| cbl_wS | 169.0 | 75.66 | +1.57 | 1.080 | 1.229 | 1.946 | 0.63 | 122 |
| cbl_wW | 257.9 | 47.08 | **+18.08** | 1.129 | 1.382 | 1.792 | 0.77 | 149 |

**The neutral array share rose 13.9 points on average, and that is `--raise-topo`, not the
closure.** At `z0_array = 0.10 m` the array was aerodynamically identical to the WorldCover
cropland it replaced, so its entire neutral signal was zero and the share was pure
geometry. Raising `topoPos` by `d` and setting `z0_array = 0.25 m` restores a 2.5x
roughness contrast and puts the first model level above panel top. **The array is now
visible to the neutral flow**, which is the point of the treatment.

**The compaction ratio, deconfounded.** The fifth pass's 2.57x compared a neutral case
with an inert floor against a convective case with a factor-10 one:

    fifth pass  2.784 / 1.082 ha = 2.57x   [different closures either side]
    sixth pass  2.432 / 1.293 ha = 1.88x   [same closure both sides]

**Roughly a quarter of the apparent neutral-to-convective compaction was the closure**, not
the physics.

**The integrals.** Convective **0.971-1.193** (was 0.768-1.180); neutral **0.905-1.095**
(was 0.803-1.005). Both tightened, and both now sit near 1 rather than spanning it. Values
above 1 over the real surface are expected and are not the wrap-around artifact — the
by-displacement decomposition saturates flat beyond one domain length (cbl_wN: 1.157 at
0.5 L, 1.192 at 1.0 L, 1.193 at 2.0 L), and the residual is `w_bar` times the concentration
integral over sloping ground, which is the advection non-closure that makes EC hard in
complex terrain. The double rotation removes 95-96% of the model-frame mean `w`
(-0.138 -> +0.005 m/s on cbl_wN) and what is left is the physical part.

**The flat controls are the diagnostic ones and they converge from below**, 0.880 and
0.897 — so the estimator is not manufacturing flux.

## 6. Infrastructure, because two silent bugs got through today

- **`bin/preflight.sh`** parses every python entry point and shell driver in ten seconds,
  and the campaign refuses to start without it. A duplicate keyword argument in
  `stage5_footprint.py` reached a running campaign and was launched six times.
- **Every analysis step asserts its output exists.** The analysis is piped into `grep`, so
  bash reports GREP's exit status and a traceback lands quietly in a redirected `.txt`.
- **`bin/test_parallel_lpdm.py`** asserts the parallel LPDM is bit-identical to the serial
  one, including that `td_particle` is offset into the full ensemble — an error that would
  be wrong but in range, and therefore silent.
- **Sampling windows are resumable**, stamped by configuration, so a kill during analysis
  no longer costs 42 minutes of GPU.
- **The LPDM is forked across cores**: 16 chunks with per-chunk seeds, so worker count is a
  pure performance knob. **6.8x on 12 workers**, bit-identical. The field cache is shared
  copy-on-write, so N workers cost no extra RAM.
- **The campaign runs the GPU and the CPU at once** — the production LES chain starts first
  and in the background and the control batteries run against it.

## 7. Known limitations, unchanged

1. The receptor is inside the **roughness sublayer** over the array; Kljun is not a
   reference there and the floor is an extrapolation. Quote the **46-66% anchor-sensitivity
   band** against a 38% sampling floor with any near-field number.
2. The first model level is **1.997 m above the raised surface = 3.50 m above bare ground**,
   `ln(z/z0) = 2.08`. The array's surface exchange is parameterised, not resolved.
3. **`z/Delta` = 0.99** at the receptor; the near field is closure-dominated — which this
   pass has now measured rather than asserted: the floor is worth **8.40 points** of
   convective array share.
4. **Tree cells have `ln(z_first/z0) = 0.69`**, 23.5% of the box.
5. Real terrain reaches **~784 m**; land cover is real to the seam.
6. Deep convective boundary layers are constrained — Phase E measured the cap as
   non-binding at a 10 m receptor (p ~ 0.54).
