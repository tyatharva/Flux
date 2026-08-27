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

## The failure

| quantity | mean | trend | limit | used | |
|---|---|---|---|---|---|
| **`TKE/u*^2`** | 1.1430 | **+8.13 %/h** | 5.0 | **163%** | **DRIFTING** |
| `sigma_v/u*` | 2.0740 | +2.69 %/h | 3.0 | 90% | ok |
| `z_i` | 660.02 | +2.29 %/h | 3.0 | 76% | ok |
| `sigma_w/u*` at the receptor | 1.2155 | +0.32 %/h | 2.0 | 16% | ok |
| `U/u*` (Kljun `Pi_4`) | 10.6434 | +0.05 %/h | 1.0 | 5% | ok |
| Kljun `x_peak` | 37.6158 | +0.01 %/h | 1.0 | 1% | ok |
| Kljun `x90` | 513.8319 | +0.01 %/h | 1.0 | 1% | ok |

### It is NOT the depth-growth artifact, which was the obvious suspect

`TKE/u*^2` uses the **column mean over the whole 2500 m domain**, so it rises as `z_i` rises
even in a layer that is otherwise equilibrated. Recomputing it as a **boundary-layer
average** (integrated to `z_i`, divided by `z_i`) removes only 1.3 of the 8.13 points:

| | column [gated] | BL-average |
|---|---|---|
| `nbl-deep` | **+8.13 %/h** | **+6.81 %/h** |
| `nbl-shallow` | +4.32 %/h | +4.42 %/h |

The BL-average also has almost the same MEAN in both rungs — **1.4814 deep, 1.4262
shallow** — so it is the scale-free form and it still fails. The depth explanation is real
but small.

### And the turnover argument, measured, is much weaker than projected

The rung was chosen on the expectation that `h/u*` scales with depth, buying ~5 turnovers
here against ~9 at `nbl-shallow`. On the ACHIEVED numbers it does not, because `u*` scales
with `G` (12 vs 8 m/s) and cancels most of the depth increase:

| | `z_i` | `u*` | `h/u*` | turnovers in 3.0 h |
|---|---|---|---|---|
| `nbl-deep` | 660 m | 0.4502 | 1466 s | **7.4** |
| `nbl-shallow` | 389 m | 0.2936 | 1326 s | **8.1** |

9% fewer turnovers, not 45% fewer. So "the deep rung is under-spun" is not established by
the turnover count, and the 3.0 h budget is not obviously wrong for this rung.

### THE GATE CANNOT TELL PASS FROM FAIL ON THIS LIMIT

Scoring the trend against **its own sampling standard error**, AR(1)-corrected because
dumps are 300 s apart while the eddy turnover is 1300–1500 s:

| quantity | trend | trend SE | `n_eff` of 19 | limit | limit is … | resolvable? |
|---|---|---|---|---|---|---|
| `U/u*` | +0.05 | 0.06 | 19.0 | 1.0 | 16.2 SE away | yes |
| `x90` | +0.01 | 0.05 | 19.0 | 1.0 | 18.2 SE away | yes |
| `x_peak` | +0.01 | 0.07 | 19.0 | 1.0 | 15.1 SE away | yes |
| `sigma_w/u*` | +0.32 | 0.23 | 19.0 | 2.0 | 7.2 SE away | yes |
| `sigma_v/u*` | +2.69 | 1.22 | 17.8 | 3.0 | **0.3 SE away** | **no** |
| **`TKE/u*^2`** | **+8.13** | **3.09** | **7.5** | 5.0 | **1.0 SE away** | **no** |
| `z_i` | +2.29 | 1.07 | 3.6 | 3.0 | **0.7 SE away** | **no** |

**Three of the seven limits sit inside their own estimator's noise.** The four that do not
are exactly the ratios PROJECT_BRIEF.md's rule says ride the inertial oscillation together, plus
the two Kljun geometry terms that inherit their immunity — margins of 7 to 18 SE.

The same measurement on the ACCEPTED seed shows the verdict is close to a coin flip in
both directions: `nbl-shallow` PASSED at **+4.32 ± 3.46 %/h**, i.e. **1.2 SE from zero and
0.2 SE from its limit**. Its pass carried essentially no information about this quantity.

And the trend does not converge — it **oscillates**, on sliding 1.5 h windows:

| window (h) | `nbl-deep` | `nbl-shallow` |
|---|---|---|
| 0.50–2.00 | +13.04 | +2.01 |
| 0.75–2.25 | -0.93 | +0.66 |
| 1.00–2.50 | -2.03 | +4.54 |
| 1.25–2.75 | +3.26 | **+6.51** |
| **1.50–3.00 (scored)** | **+8.13 FAIL** | **+4.32 PASS** |

**`nbl-shallow` would have FAILED had its run ended 15 minutes earlier.** The accepted seed
and the rejected one are not distinguishable by this limit.

## What is NOT claimed, and what was not done

The gate was **not changed**, the run was **not extended**, the rung was **not respecced**,
and no target case was built on this seed. Whether a limit whose threshold sits 1 SE from
its measurement should gate a 43 GPU-h library is a design decision, and the numbers for
it are above.

The mechanism, stated plainly: `TKE/u*^2` is a ratio in `u*` but the domain has only
`(1952/660)^2 ~ 9` energy-containing eddies across it, so the domain-mean TKE carries large
sampling scatter, its residuals are autocorrelated at `rho = +0.43`, and 19 dumps are worth
**7.5 independent samples**. A least-squares slope through that reports the scatter as much
as any drift — the same shape as the `z_i` staircase already recorded in
`FASTEDDY_TRAPS.md` §16, and the same shape as PROJECT_BRIEF.md's own standing rule that a
tolerance must be the size of the failure it is looking for.
