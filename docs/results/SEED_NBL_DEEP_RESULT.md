# `seed_nbl-deep_a000` — FAILED its gate on `TKE/u*^2`, and the gate cannot resolve it

**Run 2026-08-26 23:35 → 2026-08-27 02:29. Verdict: FAIL. Not extended, not respecced, and
no corpus case was built on it.**

`bin/pick_seed.py` refuses a seed whose gate says `pass: false`, so this is enforced rather
than remembered — the exclusion was already exercised on `seed_sbl-weak_a030`.

## The run: in class, and clean everywhere except the one limit

| | |
|---|---|
| job | `jobs/seed_nbl-deep_a000` |
| rung | `nbl-deep`, neutral, `z_i` target 550 m, `G = 12.0 m/s` from 270 deg |
| steps | **738,720 in ONE invocation**, `dt = 0.01461988 s` |
| simulated | 3.000 h, 37 dumps at 300 s |
| wall | **2.865 h = 171.9 min**, 13.962 ms/step off 37 dump mtimes, no pauses |
| **wall-to-sim** | **0.955 GPU-h per simulated hour**, **+0.2% on the sanctioned SEED CLASS 0.953 — IN CLASS** |
| artifact | `seed_restart.nc` 69.9 MB |
| the library at this rate | 15 x 2.86 h = **43.0 GPU-h**, exactly as projected |

Eight of the nine acceptance items pass:

| check | result |
|---|---|
| `CORRUPTED` / `#NaN` / `#Inf`, completion banner | **0 / 0 / 0**, present |
| `htFlux` read back out of the run's own dump | **+0.000000**, the flux it was asked for |
| `k0/k1` (accuracy CFL) | **0.122** (must be < 1) |
| **`turb_alive`, a VERDICT not a SKIP** | **OK** — max `e_res/U_ref^2` 5.00e-03, `u*` trend -9.1 %/h, final/peak **76%** |
| Ozmidov at the 10 m receptor | **`L_O/Delta = 1818.5`** against a 10.0 requirement; surface layer (z <= 50 m) min 336.1, median 636.4 |
| resolved fraction of `sigma_w^2` at the receptor | **0.043** — 95.7% sub-grid, as expected neutrally at `z/Delta ~ 1` |
| **Gate C2** | **PASS — bit-for-bit.** 23 variables, 0 differ, worst 0.000e+00 |
| **90-degree rotation check** | **PASS.** Four turns the identity bit-for-bit; file == production index map bit-for-bit at every rotation; wind vector departs from the exact turn by <= 4.5e-20; scalar slab moments invariant to 4.5e-16; FROM bearing exactly -90 per turn |

## RE-SCORED UNDER THE CORRECTED GATE, 2026-08-27 — and the FAIL did not survive

The `+8.13 %/h` above was produced by the **column-mean** TKE at a **1.5 h** window. Both
were wrong, and fixing them is not loosening anything — the 5.0 %/h threshold is untouched.

**(a) The gated TKE is now the BOUNDARY-LAYER AVERAGE.** The column mean divides by the
whole 2500 m box, so it rises mechanically as `z_i` rises even in an equilibrated layer.
It is not scale-free and was never a fair thing to trend. The BL-average has nearly the
same VALUE across rungs — **1.5024 `nbl-deep` against 1.4218 `nbl-shallow`, 5.7% apart** —
where the column mean differs by **44%**. **This was wrong even when it PASSED**:
`nbl-shallow` passed partly on being shallower.

**(b) The scoring window is 2.0 h, measured rather than inherited.** Swept 1.0–2.5 h on
both seeds: 2.0 h improves the four oscillation-immune limits by ~50% in margin, crosses
`sigma_v/u*` into resolvable on `nbl-shallow`, and stops short of 2.25–2.5 h where the
window reaches back into the cold start and the `z_i` trend blows up to +9.0 and +9.6 %/h.

**(c) A limit whose threshold sits within 3 SE of its measurement now returns
INDETERMINATE**, not PASS and not FAIL — the same refusal as `turb_alive` declining to let
a SKIP read as a pass. An INDETERMINATE limit still fails the run: a seed whose
stationarity is unestablished is not a seed. What changes is that it is refused for the
honest reason.

### Both seeds, corrected gate, no re-runs

| | `nbl-deep` | | | | `nbl-shallow` | | |
|---|---|---|---|---|---|---|---|
| quantity | trend | SE | margin | verdict | trend | SE | verdict |
| `U/u*` | +0.24 | 0.08 | 10.2 SE | ok | +0.21 | 0.11 | ok (7.2 SE) |
| `sigma_v/u*` | +1.93 | 0.80 | **1.3 SE** | **INDET** | +1.14 | 0.51 | ok (3.6 SE) |
| `sigma_w/u*` | +0.59 | 0.23 | 6.3 SE | ok | +0.23 | 0.33 | ok (5.3 SE) |
| **`TKE_BL/u*^2`** | **−0.15** | 4.88 | **1.0 SE** | **INDET** | +2.28 | 4.15 | **INDET (0.7 SE)** |
| `z_i` | +3.35 | 1.15 | **0.3 SE** | **INDET** | +1.31 | 0.66 | **INDET (2.5 SE)** |
| Kljun `x_peak` | +0.18 | 0.07 | 11.0 SE | ok | +0.18 | 0.12 | ok (7.0 SE) |
| Kljun `x90` | +0.15 | 0.06 | 13.8 SE | ok | +0.15 | 0.10 | ok (8.7 SE) |

**`nbl-deep`'s `TKE_BL/u*^2` trend is −0.15 %/h — indistinguishable from zero.** The
`+8.13` that failed it was an artifact of the retired form and the inherited window.
**It does NOT fail with a resolvable margin**, so the "the rung needs more than 3.0 h"
conclusion is NOT reached.

**What it does instead is come out INDETERMINATE on three limits, with none drifting —
and so does the ACCEPTED seed, on two.** Neither seed's stationarity is established. The
first corpus pair rests on a seed in that state.

### Widening the window does not fix the two that matter, and that is the finding

`n_eff` at every width from 1.0 h to 2.5 h:

| limit | `n_eff` range over 1.0–2.5 h | resolves? |
|---|---|---|
| `U/u*`, `sigma_w/u*`, `x_peak`, `x90` | 9–31 | yes, 3–16 SE |
| `sigma_v/u*` | 13–31 | on `nbl-shallow` only, from 1.75 h |
| **`TKE_BL/u*^2`** | **3.0–5.5** | **no, at any width** |
| **`z_i`** | **3.0–9.6** | **no, at any width** |

`TKE_BL/u*^2` and `z_i` decorrelate on the **eddy turnover** (`h/u*` = 1258–1345 s here),
not on the 300 s dump interval, so a 2.0 h window is ~5.4 turnovers and holds about five
independent samples of them **however finely it is sampled**. **Dumping more often cannot
help. What is short is the RUN**, and that is a budget decision, not a scoring one.

## The rung selection rationale was wrong

`nbl-deep` was chosen to probe spin-up adequacy on the expectation that `h/u*` scales with
depth, buying ~5 turnovers against ~9 at `nbl-shallow`. **On achieved numbers it is 7.4
against 8.1** — because `u*` scales with `G` (12 vs 8 m/s) and cancels most of the depth
increase.

Across the whole library, with `u*` from the drag-law scaling the two measured seeds
confirm (`0.2815/8.0` and `0.4292/12.0`, both ≈ `G/28`), and convective rungs turning over
on `z_i/w*`:

| rung | regime | `z_i` | `G` | `u*`~ | `w*` | turnover | **turnovers in 3.0 h** |
|---|---|---|---|---|---|---|---|
| **`nbl-deep`** | neutral | 550 | 12.0 | 0.426 | — | 1292 s | **8.4** |
| `nbl-shallow` | neutral | 300 | 8.0 | 0.284 | — | 1058 s | **10.2** |
| `cbl-deep` | convective | 950 | 11.0 | 0.390 | 1.71 | 557 s | 19.4 |
| `cbl-mid` | convective | 700 | 9.0 | 0.319 | 1.36 | 515 s | 21.0 |
| `cbl-shallow` | convective | 450 | 7.0 | 0.248 | 0.96 | 469 s | 23.0 |

**`nbl-deep` IS the most spin-up-marginal rung — by 18%, not by 80%.** The two neutral
rungs sit together at the bottom and the three convective ones are 2–3x better, because
`w*` dominates the turnover there. So the experiment had far less discriminating power
than it was designed with, and in the event it discriminated nothing: both neutral seeds
return INDETERMINATE on the same limits.

**The genuinely spin-up-marginal rung has therefore NOT been identified**, and on this
table it may not exist: the library's whole spread is 8.4–23.0 turnovers, and the question
"is 3.0 h enough" is not answerable by the estimator at all, for either neutral rung, at
any scoring window a 3.0 h run supports.

## What was done with the seeds

`nbl-deep` was **not** accepted, **not** extended and **not** respecced, and no target case
was built on it. `bin/pick_seed.py` now separates DRIFTING from INDETERMINATE: the first is
refused outright, the second only with an explicit `--allow-indeterminate`, and any pair
built that way carries `seed.gate_state = INDETERMINATE` and a warning in the training
record. `nbl-deep` is additionally excluded by name.
