# Sixth pass — the sigma_w closure

**Status: the closure is NOT fixed, and the reason is now measured rather than guessed.**
The fifth pass's diagnosis was wrong; the defect it named is real, is fixed, and was not
what broke the gate. No production footprints were regenerated, because regenerating them
on a closure that fails its own gate would only have moved the error somewhere harder to
find.

Everything below is one flat neutral and one flat convective control window at
`122^3 @ 16 m`, 2400 s each, 481 dumps, receptor 10 m. Every closure is scored on
**identical fields with identical releases**, so a difference between rows is the closure
and nothing else.

---

## 1. What the fifth pass said, and why it was wrong

`FIFTH_PASS_RESULTS.md` §5b attributed the convective well-mixed failure to a spurious
`sigma_w^2` maximum manufactured by the floor's `0.1h-0.2h` factor taper. The floor was
rebuilt so that maximum is structurally impossible — the target is made non-decreasing by a
running maximum, bounded by the model's variance at the resolved profile's own peak, and
switched off at and above it; `bin/test_sgs_floor.py` checks the properties on adversarial
profiles and `lpdm/driver.py` asserts them at runtime.

It worked, and it changed nothing:

| convective closure | floor-induced turnovers | forward lowest 3 bins |
|---|---|---|
| retired taper | 11 | 1.260 |
| **restructured, monotone** | **0** | **1.236** |

**The turnover was a real defect. It was not the one that fails the gate.**

## 2. The measurements

Counting-noise floor 5.48% in both regimes (~666 particles per bin).

### 2a. Neutral — everything passes, and that is the problem

| closure | backward rms / lo3 | forward rms / lo3 | turnovers | max factor |
|---|---|---|---|---|
| no floor | 5.70% / 1.001 PASS | 3.97% / 1.026 PASS | — | — |
| **restructured** | 5.54% / 1.002 PASS | 4.26% / 1.024 PASS | **0** | 1.29 |
| retired taper | 5.90% / 1.017 PASS | 4.79% / 1.081 PASS | 9 | 2.46 |

The restructured floor is **indistinguishable from the unmodified model** in both
directions while supplying variance up to 1.29x, which is what a floor should look like.

**And the fifth pass's "neutral results are unaffected" was wrong.** That was inferred from
the receptor factor being 1.000. The retired taper was active to 2.46 between 30 and 90 m
and put **nine** turnovers into neutral `sigma_w^2` — and still passed, because 8.1% in the
forward lowest-three-bins clears a `max(5%, 3 sigma) = 16.4%` threshold. **The neutral gate
cannot detect this class of defect at all.** It is not a weaker version of the convective
test; it is a test of a different closure, because the floor is nearly inert neutrally.

### 2b. Convective — nothing with a large factor passes

| closure | backward rms / lo3 | forward rms / lo3 | turnovers | max factor |
|---|---|---|---|---|
| **no floor** | 5.19% / 1.013 PASS | 6.08% / 1.090 PASS | — | — |
| **constant x1.673** | 4.60% / 0.974 PASS | 7.61% / 1.130 PASS | — | 1.67 |
| restructured, multiplicative | 5.09% / 1.014 PASS | 13.26% / **1.236** FAIL | 0 | 10.24 |
| restructured, additive | 9.93% / **1.152** FAIL | 11.44% / **1.232** FAIL | 0 | 10.24 |
| retired taper | 7.85% / **1.036** FAIL | 13.46% / **1.260** FAIL | 11 | 12.17 |
| **constant x10** | 5.96% / 1.069 PASS | 21.34% / **1.370** FAIL | — | 10.0 |

Read the first, second and last rows together. **The base model is well mixed in both
directions convectively.** A constant inflation of 1.673 — the factor the floor needs at
the receptor — is also well mixed. A constant inflation of 10 — with no height dependence,
no taper, no turnover and a `dsc/dz` term that is exactly zero — **fails forward at 1.370,
worse than any shaped floor.**

**So the failure is the MAGNITUDE of the inflation, not its shape, gradient or delivery.**
Every hypothesis about profile shape is excluded by the constant-factor rows.

### 2c. Two things that were excluded along the way

**Non-stationarity and mean vertical motion**, `bin/window_stationarity.py`: the
window-mean slab `w` below 400 m is **4.9e-4 m/s, 0.26% of sigma_w**, so a uniform release
is not being advected anywhere; `z_i` drifts +45 m/h (+7% across the window) and receptor
`sigma_w` -5.8%. Real but far too small for a 24% pile-up.

**A genuine drift inconsistency, found and fixed, which turned out not to be the cause.**
`lpdm/fields.py` builds `dsig2dz` as a central difference on the LES levels which is then
4-D interpolated — it is *not* the derivative of the interpolant that samples `e`. A
multiplicative floor enters that term with weight `sc` (**10.2 convectively, 1.29
neutrally**, which is exactly why neutral passes), and the product rule's two terms nearly
cancel: at `z = 25 m`, `sc*ds2z = -0.0257` against `(2/3)e*dsc/dz = +0.0232`, summing to
`-0.001`. Additive delivery — `sigma^2 = (2/3)e + delta(z)` — removes both problems and is
verified **bit-identical on `sigma^2`**. It still fails, and it makes the BACKWARD
direction (the one footprints use) worse, 1.014 -> 1.152. Multiplicative therefore stays
the default **on measurement**, and additive is kept behind `--sgs-most-form` as a recorded
negative result.

The same class of error was fixed in `dsc/dz` itself, which was a central difference of a
piecewise-linear interpolant; it is now the exact derivative of the curve the model
samples.

## 3. Why a factor of 10 is unphysical anyway

The floor exists because at `z/Delta ~ 1` the LES resolves almost none of `sigma_w` at the
receptor. That is true **at the receptor** and false above it:

| z (m) | resolved `ww` | `(2/3)e` | resolved fraction | floor factor |
|---|---|---|---|---|
| 2.0 | 0.0012 | 0.3152 | **0.4%** | 1.07 |
| 18.1 | 0.0979 | 0.1267 | 43.6% | 3.28 |
| 34.6 | 0.2495 | 0.0494 | **83.5%** | 8.06 |
| 52.1 | 0.3776 | 0.0332 | **91.9%** | 10.11 |
| 114.3 | 0.6138 | 0.0241 | **96.2%** | 4.14 |

At 34-52 m the LES resolves 84-92% of its own vertical variance. There is no sub-grid
deficit there. What the floor is "repairing" is a disagreement between the LES's `sigma_w`
and a surface-layer MOST extrapolation evaluated at `0.1 z_i`, and it repairs it by
inflating the small remaining sub-grid part tenfold.

That inflation has a consequence the closure never accounted for: **`sigma^2` is raised and
`eps` is not**, so the Langevin timescale `T_L = 2 sigma^2/(C0 eps)` is raised by the same
factor. At `sc = 10` the sub-grid memory length `sigma T_L` grows tenfold — at `z = 35 m`
it becomes comparable to the height itself — and the adaptive step saturates at
`dt_max = 1 s`. The stochastic component stops being a sub-grid perturbation. That is
consistent with the dose-response: 1.67x is fine, 10x is not.

## 4. What this leaves open — and it is a design decision, not a bug

The floor needs a **magnitude bound**, and the measurements give one: **~1.7 passes, 10
fails**, on this window, at this grid. Bracketing that ceiling is cheap (CPU only, ~12 min
per point, the control windows are the only inputs). The candidates:

1. **Weight the floor by the sub-grid fraction**, so it supplies sub-grid variance in
   proportion to how much of the variance is actually sub-grid. Parameter-free, and it
   collapses to ~1 exactly where the LES resolves the motion.
2. **Cap the factor** at the measured ceiling and state the cap.
3. **Raise `eps` with `sigma^2`** so `T_L` is preserved. Physically the floor asserts more
   energy at the filter scale, which should dissipate faster, not slower.
4. **Abandon the floor convectively** and carry the resolved deficit as a stated bias.

**This is a decision about the science, not a defect to repair, and it belongs to the
user.** PROJECT_BRIEF.md already records that the choice of `sigma_w` anchor is worth **46-66%
shape L1** against a 38% sampling floor — so bounding the floor will move the near-field
peak, which is the whole reason the floor exists. Choosing among the four above changes the
footprints, not just the gate.

## 5. What IS settled and committed

- The floor is **one implementation** (`lpdm/sgs_floor.py`); the well-mixed gate used to
  carry its own drifted copy with no displacement correction, so it was validating a
  closure the footprints do not use.
- The floor cannot introduce a turnover in `sigma_w^2` — structurally, asserted at runtime,
  covered by tests on adversarial profiles.
- `dsc/dz` is the exact derivative of the interpolant the model samples.
- Closure profiles are persisted into every footprint JSON, so the `sigma_w^2` profile
  behind a result survives deletion of the window fields.
- **Neutral Gate D1 PASSES both directions** and the standing flat/neutral regression
  passes against the fifth-pass baseline (integral -0.019 of 0.080, area80 -0.154 of
  5.000). Neutral production is unaffected by any of this.
- Sampling windows are resumable, stamped by configuration.
- `--raise-topo` surfaces are built for both regimes (`data/grid16r_nbl`,
  `data/grid16_raised`) and the receptor treatment for them is settled: release at
  `z_target = 8.5 m` with `--exact-agl`, since the tower cell's `topoPos` is raised by
  exactly 1.500 m and the instrument is 10 m above BARE GROUND.

## 6. The neutral radial bias, which did get measured

Cumulative share of footprint weight inside each radius, flat/neutral, identical fields:

| closure | <50 m | <100 | <150 | **<250** | <400 | <900 | array % | integral | x80 |
|---|---|---|---|---|---|---|---|---|---|
| restructured | 15.92 | 43.31 | 58.14 | **71.99** | 81.73 | 94.79 | 30.10 | 0.895 | 368 m |
| retired taper | 15.91 | 42.79 | 59.42 | **75.65** | 86.69 | 97.97 | 29.67 | 0.870 | 305 m |
| no floor | 16.08 | 43.70 | 58.71 | **73.27** | 84.35 | 96.10 | 30.72 | 0.885 | 332 m |

The retired taper pulls **+3.67 points** of weight inside 250 m and shortens `x80` by 63 m
— the compaction its spurious maximum predicts. Neutrally it is worth **-0.42 points** of
array share (-1.4% relative), because at this westerly the array's chord is only 120 m and
near-field concentration buys little of it.

**The convective equivalent — the number this exercise existed to produce — is not
reported, because there is no convective closure that passes its gate to report it
against.** Quoting an array-share correction derived from two closures that both fail
would be worse than quoting nothing.
