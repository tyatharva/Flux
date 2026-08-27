# `seed_cbl-deep_a000` — LOCKED IN. The rung does not fit the box.

**Run 2026-08-27 10:32 → 13:28. Verdict: FAIL, on the failure mode this rung was chosen to
test. Not extended, not respecced, and no corpus case was built on it.**

This is a **different and stronger** failure than `nbl-deep`'s. That one was an artifact of
a badly-posed estimator and did not survive the gate correction. This one is a measured
property of the flow, it is unambiguous, and it is not fixable by scoring.

## The run itself was clean and in class

| | |
|---|---|
| steps | 738,720 in ONE invocation, `dt = 0.01461988 s` |
| wall | **2.898 h = 173.9 min**, 14.123 ms/step off 37 dump mtimes, no pauses |
| **wall-to-sim** | **0.966 GPU-h per simulated hour**, **+1.4% on the sanctioned SEED CLASS — IN CLASS** |
| `CORRUPTED` / `#NaN` / `#Inf` | **0 / 0 / 0**, completion banner present |
| `k0/k1` | **0.113** (must be < 1) |
| **`turb_alive`** | **OK** — max `e_res/U_ref^2` 1.19e-02, `u*` trend −6.8 %/h, final/peak **83%** |
| Ozmidov at the receptor | **no constraint** — the surface layer is unstratified (`N^2 <= 0` below 50 m). Convection produces the largest eddies rather than limiting them; this is the easy regime for this grid, and stable is the hard one. |
| resolved `sigma_w^2` at the receptor | **0.101** — 89.9% sub-grid, against 0.036 neutral |

## THE FAILURE: the box is organising the thermals

`bin/domain_adequacy.py spectra`, last five dumps, 2-D spectrum of `w` at mid-depth:

| dump | `z_i` | `L/z_i` | peak `lambda` | `lambda/z_i` | **mode-1 share** | `r(L/2)` |
|---|---|---|---|---|---|---|
| FE_SEED.656640 | 987 m | 1.98 | **1952 m** | 1.98 | **66.0%** | +0.651 |
| FE_SEED.677160 | 987 m | 1.98 | **1952 m** | 1.98 | **72.0%** | +0.643 |
| FE_SEED.697680 | 987 m | 1.98 | **1952 m** | 1.98 | **60.6%** | +0.449 |
| FE_SEED.718200 | 987 m | 1.98 | **1952 m** | 1.98 | **53.9%** | +0.331 |
| FE_SEED.738720 | 987 m | 1.98 | **1952 m** | 1.98 | **58.3%** | +0.530 |

**The peak wavelength is pinned at exactly `L` = 1952 m in every dump**, and mode 1 carries
**53.9–72.0%** of the mid-depth `w` variance. An unconstrained CBL peaks near
`lambda ~ 1.5 z_i`; here `lambda/z_i = 1.98` because `lambda` cannot exceed the box.

For scale, Phase E measured **50.2%** in the deep case it ACCEPTED (at `L/z_i = 2.28`) and
**4.8%** in the compliant one. This sits **above** the share that was accepted, at a
tighter `L/z_i`.

**`r(L/2)` is POSITIVE here, not negative, and that does not weaken the result.** The
diagnostic's note expects strong anti-correlation from a mode-1 pattern, which holds for a
structure varying along the shifted axis. At `G = 11 m/s` a sheared CBL organises into
along-wind ROLLS, so shifting along-wind by `L/2` samples the same roll and correlates
positively. The spectral evidence — peak pinned at `L`, most of the variance in mode 1 —
is the diagnostic that does not depend on orientation.

## And the depth overshot past what the domain supports

| estimator | `z_i` | `L/z_i` |
|---|---|---|
| target | 950 m | 2.05 |
| **fixed TKE threshold** (the gated one) | **1276 m** | **1.53** |
| **peak fraction** (the corpus currency `pick_seed` matches on) | **1055 m** | **1.85** |
| **theta gradient** (`zi_from_theta`) | **987 m** | **1.98** |

**Every estimator is at or above the 976 m the box supports at `L >= 2 z_i`, and every
`L/z_i` is below the 2.0 corpus floor.** The rung was specified at 950 m — 26 m under the
limit — and entrained straight past it. A 3.0 h spin-up under `w'th_v' = 0.16 K m/s` with a
capping inversion at 950 m does not hold 950 m.

## The gate, for completeness

| quantity | mean | trend | SE | `n_eff` | limit | margin | verdict |
|---|---|---|---|---|---|---|---|
| `U/u*` | 10.2809 | +0.23 | 0.26 | 10.1 | 1.0 | 3.0 SE | ok |
| `sigma_v/u*` | 2.2029 | +0.60 | 3.09 | 5.1 | 3.0 | 0.8 SE | INDETERMINATE |
| `sigma_w/u*` | 1.1579 | −1.19 | 0.40 | 15.5 | 2.0 | 2.0 SE | INDETERMINATE |
| `TKE_BL/u*^2` | 2.8849 | +4.44 | 8.58 | 7.9 | 5.0 | 0.1 SE | INDETERMINATE |
| `z_i` | 1276.29 | +2.00 | 2.17 | 8.8 | 3.0 | 0.5 SE | INDETERMINATE |
| Kljun `x_peak` | 36.0671 | +0.21 | 0.25 | 10.2 | 1.0 | 3.1 SE | ok |
| Kljun `x90` | 496.1430 | +0.18 | 0.21 | 10.2 | 1.0 | 3.9 SE | ok |

**Four INDETERMINATE, none drifting** — one more than either neutral seed, and
`sigma_w/u*` joins them here because a convective window's second moments scatter more.
**The gate is not what rejects this seed.** Lock-in is a property no stationarity limit
looks for: the flow can be perfectly steady and still be organised by the box, and this one
very nearly is.

## What this costs the library, stated

The `cbl-deep` rung as specified is **not usable in a 1952 m domain**. That is 6 of the 30
seeds (one per base angle) and it removes the corpus's deepest convective coverage — which
is exactly where PROJECT_BRIEF.md already records the corpus is thinnest and the bias is worst:
`z_i` and surface heat flux correlate at **+0.43** (CONUS404) and **+0.49** (HRRR), so the
excluded deep-CBL hours carry **1.5–3.6x** the heat flux of the retained ones, and the
array's flux enhancement is largest exactly there.

**Not proposed and not applied**, because the rung definition is a design decision: the
options are a shallower `cbl-deep` (~700–800 m target, accepting that the deepest convective
hours are unrepresentable), or the `218^2 @ 16 m` box already costed at 3.2x, or accepting
mode-1 contamination and stating it. Phase E's evidence bears on the third — it found a
footprint at 50.2% mode-1 statistically indistinguishable from a compliant one at a 10 m
receptor — but this seed sits above that share, and Phase E tested the FOOTPRINT rather
than a seed the whole corpus restarts from.
