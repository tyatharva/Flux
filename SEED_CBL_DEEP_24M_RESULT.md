# `cbl-deep` at 2928 m: the lock-in is largely gone, the depth overshoot is not

**2026-08-29**, `jobs24/seed_cbl-deep_a000`, 122³ @ 24 m, 30 m receptor.

## Why this seed was run

At 1952 m this rung **locked in** (`SEED_CBL_DEEP_RESULT.md`): the peak wavelength of `w`
at mid-depth pinned at exactly `L` in every one of the last five dumps, with mode 1 carrying
**53.9–72.0%** of the variance, against the **50.2%** Phase E had accepted at `L/z_i` 2.28
and the **4.8%** of its compliant reference at 4.56. The rung was deleted from the library
and the corpus lost its deepest convective coverage — which is exactly where `z_i` correlates
with heat flux and the array's flux enhancement is largest.

The 2928 m box is the direct test. **The lock-in constraint is domain WIDTH, not height.**

## The result

| | 1952 m box | **2928 m box** |
|---|---|---|
| `L/z_i` (spectral diagnostic's own `z_i` = 986 m) | 1.53 – 1.98 | **2.97** |
| **mode-1 share of mid-depth `w` variance** | **53.9 – 72.0%** | **19.3 – 23.1%** |
| peak wavelength | `L` in **5 of 5** dumps | `L` in **3 of 4**, `L/2` in one |
| `r(L/2)` | positive (along-wind rolls) | **−0.076 to +0.067**, i.e. zero |

Reference points, so the number means something: Phase E's **compliant** case at `L/z_i`
4.56 had **4.8%** in mode 1; its **deep** case at 2.28 had **50.2%** and was ACCEPTED — its
footprint was statistically indistinguishable from the compliant one (array share −1.88
points, SE 3.03, t = −0.62, p ≈ 0.54).

**So 19.3–23.1% at `L/z_i` = 2.97 sits well below the level Phase E measured and accepted,
and `r(L/2)` is no longer anti-correlated. By Phase E's own standard the box no longer
organises the thermals.** The honest residual: the peak wavelength still lands on mode 1 in
three dumps of four. A spectrum whose largest mode is the box is not yet a spectrum with no
box in it — but the mode's SHARE is what makes it an artifact, and that has fallen by a
factor of three.

## What did NOT get fixed: the depth

| `z_i` definition | 1952 m box | **2928 m box** | target |
|---|---|---|---|
| fixed 0.01 m²/s² TKE (gated) | 1276 m | **1308 m** | 950 m |
| 5% of running peak (corpus currency) | 1055 m | **1065 m** | 950 m |
| spectral diagnostic | — | 986 m | |

**The overshoot is a property of the rung spec, not of the box** — the two boxes agree to
within 3% on both depth definitions. `w'θ'_v = 0.160 K m/s` under a 0.08 K/m cap with
−25 m/h subsidence builds a deeper layer than 950 m asks for, and it is still deepening at
+6.8 %/h (fixed) / +1.6 %/h (peak fraction) at the ceiling.

At 2928 m that overshoot is no longer fatal — `L/z_i` is 2.24 (fixed) to 2.97 (spectral),
all above the 2.0 corpus floor, where at 1952 m every definition put it at 1.53–1.98, below
it. **But 1308 m is above the 1250 m band ceiling**, so the rung as specified builds a state
the corpus would refuse a CASE for. Re-specifying it to a lower `w'θ'_v`, or a stronger cap,
is a rung-ladder decision and is flagged rather than made.

## Stationarity: FAIL, all seven INDETERMINATE, none drifting

Ran to the 3.0 sim-h ceiling. Achieved `u*` 0.574, `U(10 m)` 8.12, direction FROM 265.1,
drift at freeze **−4.71 deg/h**, `σ_w/u*` 1.046 at the receptor.

`bin/seed_budget.py`, fixed 2.0 h window swept over end times: **not in band anywhere up to
3.0 h**, zero of five immune limits resolving at any end time, and `z_i` reads DRIFTING from
2.25 h onward — consistent with a layer that is still deepening. This rung needs more than
the ceiling, and the report is the margin rather than a stop time.

## Cost

0.48 GPU-h per simulated hour, 3.0 sim-h in one invocation, ~87 min wall — half the 16 m
grid's rate.
