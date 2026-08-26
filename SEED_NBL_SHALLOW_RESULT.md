# The first seed: `nbl-shallow`, base angle 0 — FAIL on one limit, and the limit is an artifact

**2026-08-26.** The first full-length seed the library has ever produced. It ran clean, it
is bit-for-bit restartable, it rotates exactly, its turbulence is alive, and it came in on
budget — and it **FAILED its own stationarity gate**, on one of seven quantities.

This file records the run, the failure, and the measurement that says what the failure is.
**Nothing here was re-run, extended or re-specified to obtain a pass**, and the target case
that was to be built on this seed was NOT run: `bin/pick_seed.py` refuses a seed whose gate
says FAIL, and overriding it is the move this exercise exists to avoid.

---

## The run

| | |
|---|---|
| job | `jobs/seed_nbl-shallow_a000` |
| rung | `nbl-shallow`, neutral, `z_i` target 300 m, `G = 8.0 m/s` from 270 deg |
| steps | **738,720 in ONE invocation** (chaining retired), `dt = 0.01461988 s` |
| simulated | 3.000 h, a 300 s stationarity dump (37 of them) |
| wall | **2.869 h = 172.1 min** |
| rate | **13.982 ms/step** measured off 37 dump mtimes, no pauses |
| **wall-to-sim** | **0.956 GPU-h per simulated hour** |
| against the sanctioned SEED CLASS 0.953 | **+0.4% — IN CLASS** |
| against the manifest's own projection 1.019 | -6.1% (the planner is deliberately pessimistic) |
| artifact | `seed_restart.nc` **73.27 MB** against a 73.3 MB estimate (-0.0%) |
| the library at this rate | 15 x 2.87 h = **43.0 GPU-h** against the projected 43 |

## What passed

| check | result |
|---|---|
| `CORRUPTED` / `#NaN` / `#Inf` in the log | **0 / 0 / 0**, completion banner present |
| `htFlux` read back out of the run's own dump | **+0.000000**, the flux the run was asked for |
| `k0/k1` (accuracy-CFL) | **0.119** (must be < 1; ~9 means `dt` past the boundary) |
| **`turb_alive`, a real VERDICT and not a SKIP** | **OK** — `max e_res/U_ref^2 = 4.70e-03` against a 2.0e-3 floor, `u*` trend -9.6 %/h against a -35 limit, final/peak **71%** against a 50% minimum |
| Ozmidov at the 10 m receptor | **`L_O/Delta = 484.68`** against a 10.0 requirement. Surface layer (z <= 50 m) min 70.3, median 156.7. For scale, the stable seeds that collapsed measured **3.57** here. |
| resolved fraction of `sigma_w^2` at the receptor | **0.036** — i.e. **96.4% sub-grid**, as expected for neutral at `z/Delta ~ 1` |
| **Gate C2** (restart with `Nt` = restart step, re-dump, diff) | **PASS — bit-for-bit.** 23 variables compared, **0 differ, worst 0.000e+00** |
| **90-degree rotation check** (`bin/rotation_check.py`, new) | **PASS.** Four turns are the identity bit-for-bit; the FILE `prep_restart.py` writes matches the imported production index map bit-for-bit at every rotation; the slab-mean wind vector departs from the exact turn by <= **1.01e-20**; scalar slab moments invariant to **7.85e-16**; and the FROM bearing moves **exactly -90 per turn**, which is the sign convention `pick_seed.py` picks every corpus case on and previously had only as a comment |

## Achieved vs asked

| | asked | achieved |
|---|---|---|
| geostrophic | 8.0 m/s FROM 270.0 deg | — |
| receptor wind direction | — | **FROM 251.72 deg**, i.e. **Ekman backing 18.3 deg** (PROJECT_BRIEF.md: 12-24 deg neutrally) |
| `z_i` | 300 m | **364 m** (the gate's diagnostic; see below — 371 m absolute, 337 m from the theta gradient) |
| `u*`, `U(10 m)`, `sigma_w` | — | 0.2815, 3.005 m/s, 0.3414 |

`pick_seed.py`'s nominal Ekman estimate for an unspun neutral seed is 23.5 deg, which would
have put this seed's heading at 246.5. It achieved 251.7 — the estimate is **5.2 deg off**,
about a sixth of a library direction bin. That is the first measurement of the quantity
every unbuilt seed's heading currently rests on.

---

## The failure: `z_i` at +11.67 %/h against a 3.0 %/h limit

Six of seven limits pass with enormous margin. Scored on the last 1.5 h:

| quantity | mean | trend | limit | |
|---|---|---|---|---|
| `U/u*` (Kljun `Pi_4`) | 10.6929 | **+0.14 %/h** | 1.0 | ok — 14% of its limit |
| `sigma_v/u*` | 2.0486 | +1.01 %/h | 3.0 | ok |
| `sigma_w/u*` at the receptor | 1.2219 | +0.07 %/h | 2.0 | ok |
| `TKE/u*^2` | 0.7955 | +4.32 %/h | 5.0 | ok |
| **`z_i`** | 364.43 | **+11.67 %/h** | 3.0 | **DRIFTING — 389% of its limit** |
| Kljun `x_peak` | 38.2729 | -0.21 %/h | 1.0 | ok |
| Kljun `x90` | 521.2624 | -0.17 %/h | 1.0 | ok |

**The boundary layer is not deepening at 11.67 %/h. The estimator is.**

`z_i` here is *the height at which resolved TKE falls to 5% of its own peak* — a threshold
**relative to a peak that is moving**. Over the same scored window:

| diagnostic | mean | trend |
|---|---|---|
| `z_i`, 5% of the INSTANTANEOUS peak (**the gated one**) | 364.4 m | **+11.67 %/h** |
| `z_i`, fixed threshold 0.01 m2/s2 | 389.3 m | **+1.87 %/h** |
| `z_i`, 5% of the run's own SETTLED peak | 370.8 m | **+1.71 %/h** |
| `z_i` from the theta gradient (inversion base) | 336.5 m | **+2.33 %/h** |
| peak resolved TKE (*the normaliser*) | 0.3308 m2/s2 | **-15.67 %/h** |
| `u*` | 0.2936 | **-9.61 %/h** |

`bin/zi_diagnose.py` (`results/nbl_a000_zi_diagnosis.txt`) settles which of the two the
gated number is tracking: **the gated depth is -0.885 correlated with the peak it is
normalised by**, the fixed-threshold depth only -0.379. It is also a **staircase** -- the
peak-normalised depth takes **4 distinct values** over the last 2 h, because it can only
land on a model level, and a straight line fitted through a staircase reports a trend
whatever the layer does.

**Three independent depths, three different thresholds, one answer**: 1.71, 1.87 and
2.33 %/h, all inside the 3 %/h limit. Only the peak-normalised one is outside, and it is
outside by 4x.

The peak is falling because `u*` is falling, and `TKE ~ u*^2` predicts **-19.2 %/h** against
the -15.7 measured. A falling peak lowers the 5%-of-peak threshold, which pushes the
crossing height **up**. Both physical depths sit inside the 3 %/h limit; the gated one does
not.

### This is `FASTEDDY_TRAPS.md` §16 with the sign reversed — and it corrects §16

§16 recorded this estimator falling 154 -> 81 m while nothing shrank, because the peak was
**growing** during spin-up. It then concluded:

> **In a converged state it is fine** — the peak is steady, so the threshold is steady.

**That conclusion is wrong, and this run is the counter-example.** A doubly-periodic neutral
Ekman layer forced by a constant geostrophic wind does not have a steady `u*` on any
affordable timescale: `f = 9.94e-5`, the inertial period is **17.6 h**, and `u*` falls for
the first quarter of it. At 3.0 h this run is squarely inside that falling quarter, `u*` is
down 9.6 %/h, and the TKE peak with it. **The peak is not steady in a converged state; it
rides the same oscillation the whole gate was designed to be immune to.**

And that is the sharp form of the finding. `bin/seed_stationarity.py` exists because gating
on `u*` failed this project's spin-ups twice for a reason that was never a modelling error,
so every limit is a RATIO that rides the oscillation in both its terms. `z_i` is the one
gated quantity that is **not** such a ratio, and it inherits the oscillation anyway —
through the threshold rather than through the value.

### The corroboration: the two gated quantities that depend on `z_i` are flat

Kljun's `x_peak` and `x90` take `z_i` as an input and are gated at 1.0 %/h. They came in at
**-0.21 %/h and -0.17 %/h**, and `x_peak` spans **38.0-38.4 m across the whole scored
window against a 16 m raster cell**. A `z_i` genuinely moving at 11.67 %/h could not leave
them there. This is the same thing PROJECT_BRIEF.md already records from the other direction: at a
10 m receptor Kljun's only `z_i` channel is `1/(1 - z_m/h)`, worth **1.0 percentage point**
of array share over `h = 200-1200 m`.

---

## What was NOT done, and why

- **The run was not extended.** The instruction was to report a failure, not to buy a pass
  with more simulated hours — and more hours would make this worse, not better: `u*` keeps
  falling until ~4.4 h.
- **The regime was not re-specified.**
- **The gate was not changed.** Whether the `z_i` limit should be scored against an absolute
  threshold, or dropped in favour of the theta-gradient depth, or kept as it is because a
  seed that trips it deserves a second look, is a **design decision and belongs to the
  user**. The numbers needed to make it are all above.
- **The target case was not run.** `bin/pick_seed.py` refuses a seed whose gate says FAIL,
  and this seed's `return/stationarity.json` says exactly that. The case that was selected
  for it — **2023-03-10 14:00 UTC**, chosen and staged offline — is described in the report
  and is ready to run against whatever seed the user decides to accept.

## Files

`jobs/seed_nbl-shallow_a000/return/` (73.4 MB: restart, gate JSON and text, logs, manifest),
`results/seed_nbl-shallow_a000.txt`, `results/seed_rotation_check.{txt,json}`,
`results/seed_ozmidov.txt`, `results/seed_turb_alive.txt`.
