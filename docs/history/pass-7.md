# Seventh pass: a 30 m receptor on 122³ at 24 m

2026-08-29. The 10 m configuration was retired on two measurements. The footprint peak did
not respond to meteorology: 48 m in all three target cases (three soundings, three rotations,
three directions), max/min 1.00×, while the 80% area spanned 2.54×. And LES `σ_w` at the
receptor ran 2.33–2.99× the tower median with the closure floor inactive and about 90%
sub-grid. The near field was closure output. Refining `Δ` to 2.9 m would cost about 22×, so
the receptor was raised instead: **30 m on 122³ at 24 m**, a 2928 m box. The real tower stays
at 10 m; the emulator now predicts a footprint the instrument does not measure, deliberately.

## Configuration

| | |
|---|---|
| grid | 122³ at 24 m; receptor exactly 30.000 m on the `k = 3` cell centre; `dz_sfc` 8.5583 m, `d_zeta` 24.691358, `verticalDeformFactor` 0.346601, `zCeiling` 3000 m |
| `Δ`, `z/Δ` | 17.05 m, 1.76 (against 10.09 m and 0.99 at 16 m) |
| cost | 0.481 GPU-h per simulated hour, measured |
| `dt` | boundary between CFL 1.55 (`k0/k1` 0.132) and 1.60 (8.511); production 1.3442, `dt = 5/169 s`, 17% margin. Not the retired 24 m grid's 1.64 despite identical anisotropy |
| B5, rotation | PASS at the fp32 floor; four turns the identity bit-for-bit |
| taper | pad 10, real terrain to 1224 m |

**The lake is back and Gate A1 fails in one regime**: water 8.78% of the box (1307 cells);
worst-case footprint water share 17.45% in very stable easterly conditions, which the corpus
does not contain, and 7.38% over the regimes it does. Both numbers recorded. **The array
leaves the footprint for E/W winds**: Kljun at 30 m gives N 30.7%, E 0.04% (against 80.6% and
29.9% at 10 m). "This tower measures the array in every wind direction" is a 10 m statement.
The `z_i` band moved to 300–1250 m (floor `10 z_m`; ceiling the lower of `L ≥ 2 z_i` and the
domain height): day coverage 75.0% → 80.4%, 1370 → 1469 cases, but the deep exclusion still
carries 2.33× the heat flux of the accepted hours.

## Five fixes

1. **The integral asymptote is `1 − z_m/z_i`, not 1** (Steinfeld et al. 2008): 3.75% at 30 m in
   an 800 m layer. Reported per case and quoted beside Kljun on identical cells.
2. **Negative footprint values are physical and nothing clips them**, now asserted
   (`bin/test_negative_lobes.py`). Across twelve convective footprints the negative lobe was
   5.8–11.1% of |flux| with its centroid 2.5–5× further out than the positive lobe's.
3. **The tower `σ_w` check translated from 10 to 30 m** (`bin/sigma_w_tower.py`): invert
   `σ_w(10) = 1.25 u* φ_w(10/L)` for `u*`, predict at 30 m; the lift is 1.006–1.238×. Caveats:
   the tower file carries no wind speed, and the stable branch is an upper bound.
4. **That check became an acceptance gate**: stage 7c refuses a case whose window `σ_w` falls
   outside the tower IQR for its own `H`.
5. **Steinfeld's spin-up accelerator** for neutral rungs, with `htFlux` zeroed in the restart
   file and read back.

Plus touchdown persistence (uniform bottom-k on an independent key, signed, unfolded), which
[the target design](../emulator/targets-and-architecture.md) later decided not to use.

## The GPU LPDM: accepted

`lpdm/cuda/` (then in the fork tree), built as `lib/liblpdm.so` and driven from `lpdm/gpu.py`.
A VRAM ring holds `t_back` of history at fp16; the backward ensemble integrates in-kernel with
fp64 particle state; the whole production closure is transliterated. Acceptance on a 900 s
convective window with both paths reading the same fp16 fields (`results/gpu_lpdm_acceptance.json`):
ingest PASS (`eps` median relative 2.0e-3, the fp16 floor); footprint within the CPU path's
own half-vs-half floor on peak, centroid, A80, integral and array share; backward well-mixed
PASS (lowest three 0.999 vs 0.995); signed weights PASS (negative lobe 7.7% vs 9.0%); 0.06 s
against 9.3 s on 12 CPU workers, 153×. Forward D1 failed in both paths on that window (1.093
vs 1.095), which establishes it is the window (900 s of a layer 1800 s old), not the port. Not
done: the in-FastEddy hook, which the [eighth pass](pass-8.md) built as the in-process hand-off.

## The resolution split

Resolved fraction of `σ_w²` at the receptor at `z/Δ = 1.76`: convective 47.5% resolved, neutral
13.6%. The fourth pass had measured 52.3% / 85.5% sub-grid at a 30 m receptor on a different
domain, so the fraction is a property of `z/Δ` and regime. Stated before the targets ran: a
30 m receptor takes the convective half of the corpus out of the closure-dominated regime and
leaves the neutral half in it.

Containment, first evidence: the integral by trajectory displacement on a convective control
was still climbing at 0.75 L (+3.4% over the last quarter), so the flat/neutral containment
gate was deferred but required ([containment](containment.md)).

## Seeds: `nbl-deep` and `cbl-deep`

`nbl-deep` with the accelerator ran to the 3.0 sim-h ceiling at 0.483 GPU-h/sim-h, every
battery item passing (`k0/k1` 0.135, `turb_alive` OK, `L_O/Δ` 311, C2 bit-for-bit), and the
gate returned five INDETERMINATE, none DRIFTING. `bin/seed_budget.py` found the immune limits in
band nowhere up to 3.0 h. `cbl-deep`: the lock-in largely gone (mode-1 share 19.3–23.1% against
53.9–72.0% in the 1952 m box; `r(L/2)` zero), the depth overshoot unchanged at 1308 m, above
the 1250 m band ceiling ([seed rungs](seed-rungs.md)).

## The deciding test: the peak moves

Pre-registered before either target ran (`results/deciding_test_preregistration.txt`).

| | A `case_2023052519`, convective | B `case_2023121921`, near-neutral |
|---|---|---|
| forcing | `H` 333 W/m², HRRR `z_i` 970 m | `H` 22 W/m², `z_i` 447 m |
| achieved `L` / `z_i` / direction | −25.5 m / 1229 m / 89.2° | −732 m / 937 m / 177.0° |
| **LES peak** | **144 m** | **288 m** |
| its own half-vs-half peak floor | 24 m (1 cell) | 0 m |
| Kljun peak on identical cells | 144 m | 168 m |
| A80 | 20.22 ha (Kljun 23.73) | 23.21 ha (Kljun 12.56) |
| integral / asymptote | 1.463 / **1.497×** | 0.888 / **0.916×** |
| array share ± SE (10 groups) | 5.65 ± 1.44% | 1.14 ± 0.37% |
| sub-grid fraction at the receptor | 34.0% | 75.6% |
| floor factor at the receptor | 1.912 | 1.000, inert |
| `σ_w` vs the translated tower | 0.78× median, outside the IQR | 1.14×, inside |

**|Δpeak| = 144 m, six times the larger floor, ordering matching Kljun's**, max/min 2.00×
against 1.00× at 10 m. **And it is not the closure**: recomputed with the floor off, the peak is
bit-identical in both cases (144 and 288 m), while the array share moves (5.65 → 3.72%) and the
integral (1.463 → 1.634) do. The 144 m separation is the LES's.

Settled for free: `t_back` = 600 s captures 99.6–100% of the 900 s integral at a 30 m
receptor, and the peak is at its final value from 150 s; production kept 900 s. And **the
integral's departure from the asymptote tracks the mean vertical velocity at the receptor with
the right sign**: A in subsidence (`W = −0.099 m/s`) at 1.497×, B in an updraft (+0.342) at
0.916×. The advective non-closure, measured on two cases of opposite sign.

What the test did not establish: both targets came off the same seed at different rotations
(direction dominated the cost), so it is not a two-seed test; case A fails the `σ_w` acceptance
gate and is not a usable corpus target by the criterion this pass adopted; neither footprint
is fully contained (A's integral still rising 1.447 → 1.463 over the last quarter domain); and
the seed mismatch was large and the adjustment widened A's direction gap from 23.1° to 35.2°.

New traps: §19 (a grid constant that is really a grid property, five instances), §19b–d
([FastEddy traps](../reference/fasteddy-traps.md)).

Removed from the tree on 2026-09-04 (at the `pre-cleanup-2026-09-04` tag): `runs/g24_*`
except the base templates and the two directories tests name, `results/g24_*`,
`bin/run_pass7.sh`, `bin/seed_budget.py`'s outputs, `jobs24/`.
