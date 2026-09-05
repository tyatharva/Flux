# The seed rungs, one at a time

Before the 30-seed library was spun in one go on 2026-08-31, individual seeds were run on the
16 m and 24 m grids to find out what a rung does over a spin-up and whether the gate could tell.
Four of those runs changed the design.

## `seed_nbl-shallow_a000` (16 m, 2026-08-26): the first accepted seed, after the estimator was fixed

The first full-length seed the library produced: 738,720 steps in one invocation (chaining had
just been retired), 3.000 sim-h, 37 dumps at 300 s, 0.956 GPU-h per simulated hour, exactly in
class. It was clean everywhere: 0 `CORRUPTED`, `htFlux` read back as +0.000000, `k0/k1` 0.119,
`turb_alive` a real OK (final/peak 71%), `L_O/Δ` 485 at the receptor, Gate C2 bit-for-bit on 23
variables, the new 90° rotation check bit-for-bit at every turn with the FROM bearing moving
exactly −90° per turn. Achieved: `u*` 0.2815, `U(10 m)` 3.005, `z_i` 364 m against a 300 m
target, receptor wind from 251.7° for a 270° forcing: 18.3° of Ekman backing, 5.2° off the
23.5° `pick_seed.py` assumed for an unspun seed.

**And it failed its gate on one limit: `z_i` at +11.67 %/h against 3.0**, with the other six
passing by wide margins (`U/u*` +0.14, `x_peak` −0.21, `x90` −0.17). The boundary layer was not
deepening; the estimator was. `z_i` as 5% of the running TKE peak tracks a peak that falls with
`u*` on the inertial oscillation (peak −15.67 %/h, `u*` −9.61); the gated depth was −0.885
correlated with the peak it was normalised by; a fixed 0.01 m²/s² threshold said +1.87 %/h, the
settled-peak version +1.71, the inversion base +2.33. It was also a staircase of four model
levels. Nothing was re-run or re-specified to obtain a pass; the numbers were handed over and
the gated `z_i` was switched to the fixed threshold the same day. Re-scored on the same 37
dumps the seed **passed** at +1.87 %/h on two levels spanning 14 m, and became the library's
first accepted seed. The target case it served is [the target case](target-case.md).

## `seed_nbl-deep_a000` (16 m, 2026-08-27): the gate cannot resolve two of its limits

3.000 sim-h at 0.955 GPU-h/sim-h, eight of nine acceptance items passing (`k0/k1` 0.122,
`turb_alive` OK, `L_O/Δ` 1818, C2 bit-for-bit, rotation exact), refused on `TKE/u*²` at
+8.13 %/h. Re-scored under the corrected gate the FAIL did not survive: the gated TKE became the
boundary-layer average (the column mean divides by the whole 2500 m box and rises mechanically
as `z_i` rises; the BL-average is 1.50 vs 1.42 across the two neutral rungs where the column
mean differs 44%), the scoring window became 2.0 h (measured over 1.0–2.5 h; longer reaches
into the cold start), and a limit within 3 SE of its threshold became INDETERMINATE. The trend
was then −0.15 %/h, indistinguishable from zero.

What it came out as instead was INDETERMINATE on three limits with none drifting, and so did
the accepted `nbl-shallow` on two. **Widening the window does not fix the two that matter**:
`TKE_BL/u*²` and `z_i` have `n_eff` of 3.0–5.5 and 3.0–9.6 at every width from 1.0 to 2.5 h,
because they decorrelate on the eddy turnover (`h/u*` 1258–1345 s), not on the 300 s dump
interval. Dumping more often cannot help; what is short is the run. The rung-selection
rationale was also wrong: `nbl-deep` was meant to be the spin-up-marginal rung with about 5
turnovers against 9, but `u*` scales with `G` and cancels most of the depth increase, so the two
neutral rungs sit at 8.4 and 10.2 turnovers in 3.0 h and the convective rungs at 19–23.

## `seed_cbl-deep_a000` (16 m, 2026-08-27): locked in, the rung does not fit the box

Clean and in class (`k0/k1` 0.113, `turb_alive` OK, 0.966 GPU-h/sim-h), and unusable for a
reason no stationarity limit looks for. The 2-D spectrum of `w` at mid-depth had its peak pinned
at exactly `L` = 1952 m in every one of the last five dumps, with mode 1 carrying 53.9–72.0% of
the variance, where an unconstrained convective layer peaks near 1.5 `z_i` and the fifth pass's
compliant reference at `L/z_i` 4.56 had 4.8%. `r(L/2)` was positive rather than the expected
anti-correlation because at `G = 11 m/s` a sheared layer organises into along-wind rolls. And
the depth overshot: 1276 m (fixed threshold), 1055 m (peak fraction), 987 m (θ gradient) against
a 950 m target, every `L/z_i` below the 2.0 corpus floor. A 3.0 h spin-up under 0.16 K m/s with
a cap at 950 m does not hold 950 m. The rung was deleted from the 16 m library, taking the
corpus's deepest convective coverage, exactly where `z_i` and heat flux correlate (+0.43
CONUS404, +0.49 HRRR) and the array's flux enhancement is largest.

## `cbl-deep` at 2928 m (24 m grid, 2026-08-29): the lock-in is largely gone, the overshoot is not

The direct test of the box: mode-1 share 19.3–23.1% at `L/z_i` 2.97 (against 53.9–72.0%), peak
wavelength at `L` in 3 of 4 dumps rather than 5 of 5, `r(L/2)` −0.076 to +0.067. By the fifth
pass's own standard (50.2% accepted as footprint-indistinguishable) the box no longer organises
the thermals. The depth overshoot is a property of the rung, not the box: 1308 m (fixed) and
1065 m (peak fraction) against 1276 and 1055 at 1952 m, still deepening at +6.8 %/h at the
ceiling. At 2928 m that is no longer fatal for `L/z_i` (2.24–2.97), but 1308 m is above the
1250 m band ceiling, so the rung as specified builds a state the corpus would refuse a case
for. Re-specifying it was flagged, not made; the production library kept `cbl-deep` at a 950 m
target and 0.160 K m/s, and every corpus case picks its seed on achieved depth anyway.
Stationarity: all seven INDETERMINATE, none drifting; the immune limits in band nowhere up to
3.0 h. 0.48 GPU-h per simulated hour, half the 16 m rate.

## What these four settled

- Gate on ratios that ride the inertial oscillation, and on a fixed-threshold `z_i`.
- INDETERMINATE is the library's normal state; a 3.0 h (later 2.0 h) run cannot resolve the
  eddy-turnover quantities at any scoring window.
- The box width, not height, is what a deep convective rung needs; 30 m on 3660 m is wider
  still.
- Seeds are matched on what they achieved, and a case's inputs are read off its own window, so
  a rung overshooting its target is a coverage question, not a correctness one.

The 16 m seeds (`jobs/`), the 24 m seeds (`jobs24/`) and `results/seed_nbl-shallow_a000.txt`,
`results/nbl_a000_zi_diagnosis.txt`, `results/seed_indeterminate.txt`, `bin/zi_diagnose.py`,
`bin/seed_indeterminate.py`, `bin/seed_compare.py`, `bin/seed_tke_rescore.py` were removed
from the tree on 2026-09-04 and remain at the `pre-cleanup-2026-09-04` tag. The production
library's records are under `seeds/` and `results/seed_library/`.
