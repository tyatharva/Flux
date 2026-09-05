# Fifth pass: a 10 m receptor on 122³ at 16 m

2026-08-22. Rebuilt around the real instrument height. The grid was chosen for corpus
economics. An 8-case campaign costs about 12 GPU-h here against 42 at 186² at 10 m. Every
phase ran and every gate passed, about 30 GPU-h, unattended via `bin/run_campaign.sh`. What
it gained and what it cost are both recorded. The configuration was retired two passes later
because the near field at a 10 m receptor turned out to be closure output.

## Configuration

| | value | how it was fixed |
|---|---|---|
| grid | 122 × 122 × 122, `dx = dy = 16 m`, 1952 m domain | `(N + 6) = 128` in all three |
| vertical | `d_zeta` 20.576132, `verticalDeformFactor` 0.194059, `dz_sfc` 3.9933 m | `bin/vgrid.py`. Receptor exactly at 10.000 m on `k = 2` |
| block | `1x2x64` | fastest of nine legal shapes, 0.01475 s/step |
| `dt` | 0.0146417 s, CFL 1.35 | measured accuracy boundary about 1.51, 10% margin. Terrain did not lower it |
| cost | 0.94–0.99 GPU-h per simulated hour | measured |
| `z/Δ` at the receptor | **0.99** | the sub-grid gate was retired |
| taper | pad 12 (192 m). Real terrain to 784 m | measured knee |

## Gates

| gate | verdict | number |
|---|---|---|
| A1 water share | PASS | worst case 0.01% over every direction × stability (8 water cells of 14,884) |
| B3 flat `dt` | PASS | boundary about 1.51, not the 1.64 of the 24–30 m grids |
| B4 terrain `dt` | PASS | clean to CFL 1.40. `bin/k0k1_by_slope.py` conditions on slope because the domain-mean check cannot see a few ringing columns |
| B5 restart injection | PASS | `topoPos` 9.5e-7, `z0m` 1.5e-9, `htFlux` exact |
| B6 90° equivariance | PASS | rotation exact to 1.2e-14 |
| B7 subsidence | PASS | θ warms at 1.10× the prescribed rate, after the moisture-off fix (patch 0004) |
| C1 stationarity | PASS | `U/u*` +0.03 %/h. Kljun `x_peak` +0.06 %/h |
| C2 restart resume | PASS | bit-for-bit on every field |
| C3 convective similarity | PASS | shallow `z_i` 428 m, `w*/u*` 2.56, entrainment 0.223. Deep 857 m, 3.11, 0.172 |
| D1 well-mixed | PASS | backward rms 4.26%, forward 4.12%, against a 5.48% counting floor |
| D2 integral | PASS | LES 0.914 vs Kljun 0.956 on the identical domain, converging from below |
| D3 error floor | PASS, narrowly | half-vs-half overlap 43% vs LES-vs-Kljun 33% |
| E domain adequacy | PASS | `L ≥ 2 z_i` not binding, p ≈ 0.54 |
| F production directions | PASS | explicable difference in every direction |

## Three gates that were specified wrong

- **C1 gated on `u*`.** A doubly-periodic neutral Ekman layer under a constant geostrophic
  wind does not settle to a fixed `u*` on any affordable timescale. The inertial period is
  17.6 h. `u*` fell for 4.4 h and then rose. `U/u*` moved +0.03 %/h while numerator and
  denominator each moved +6.3 %/h. The gate now tests the footprint's own controlling
  parameters ([seed library](../les/seed-library.md)).
- **C2 demanded bit equality over a whole segment.** FastEddy is chaotic in fp32. The testable
  claim is the restart *read*, which is bit-for-bit.
- **B4's check could not see the problem** (a domain mean cannot see 1.7% of cells).

## The flat/neutral control

One 2400 s window, 315,700 particles over exactly 30 minutes. **`t_back` = 600 s, measured.**
The integral reaches 98.9% of its 600 s value at 500 s. The peak (64 m) is converged at every
`t_back` down to 60 s. The array share converged to 0.1 points by 400 s while the integral
needed 500 s. Window sizing is set by the tail, not the signal. Backward transit from 10 m:
median 59 s (60–95 s predicted from `z/σ_w`). Ensemble convergence from 12 independent 150 s
sub-windows: the peak converged in one sub-window. The centroid reached 80 m at p90 by 12.5 min.
At this receptor the 30-minute averaging period was well past what convergence needed.

`x80` was not a reliable statistic at this grid. Its half-vs-half floor was 203 m, 54% of
itself, because the tail is carried by rare large-weight touchdowns (the top 0.1% carry 9.6%).

## Phase E: domain adequacy, the decision experiment

Two convective windows in the same 1952 m domain, identical in everything including surface heat
flux (0.1363 K m/s virtual), with `z_i` separated by the capping inversion and subsidence alone:
`L/z_i` = 4.56 and 2.28. **The lock-in artifact was present and not subtle.** The deep case
put 50.2% of mid-depth `w` variance in mode 1 (wavelength = L exactly) against 4.8% in the
shallow one. **And the footprint did not notice.** Peak 64 m in both, centroid −2.2 m,
`x80` +12.9 m, array share 38.63 ± 5.96% vs 36.76 ± 7.52%, difference −1.88 points, SE 3.03
over 10 release groups, p ≈ 0.54. So `L ≥ 2 z_i` is not binding for a 10 m footprint and
convective-midday coverage went from 19.3% to 60.9%.

**The first answer was wrong.** Run with two halves per case the gate said DIFFERS, missing by
2% of a 1.19-point "floor" that was one difference with one degree of freedom. The actual sd
over 10 groups was 6.0–7.5 points. That became [standing rule 5](../reference/standing-rules.md).

## Phase F: production directions

Neutral: array share 20.08% (W, 239°), 45.79% (S, 147°), 37.68% (E, 47°), 42.13% (N, 320°),
a 2.3× swing against an array occupying 1.03% of the domain (19–44× enrichment). At 30 m the
swing had been about 370×. Here the tower is inside the array and sees it from every
direction. The chord is capped by the array's 120 m width, so absolute share by direction is
the discriminator, not the ratio. Predicting the share from the LES's own `f_y` moved the
prediction toward the measurement in every case, and the ratios agreed exactly. The LES
footprint was 40–150% broader than Kljun's in 80% area with its peak one cell further out.

Convective: 81.36% (N), 74.13% (S), 63.02% (E), 29.14% (W), mean 1.67× the neutral shares.
**The sign of the disagreement with Kljun flipped between regimes.** The LES says convection
makes the footprint 2–3× more compact. Kljun's 80% area barely moves and moves the wrong way.
That structural failure of the analytic model was the residual the emulator existed to learn.
Water was 0.00–0.05% of every footprint.

**The displacement-height sensitivity**, three treatments of the same convective northerly
with the instrument at 10.0 m above bare ground: baseline (`z0` 0.10, flat) 81.36%. Bracket
(`z0` 0.25, flat) 82.26%. Raised (`z0` 0.25, `topoPos` +1.5 m, fractional receptor) 84.12%.
The surface representation accounted for 2.76 points (0.90 roughness, 1.86 displacement),
measured where the share is closest to saturation. Kljun predicts +8.3 points on a crosswise
direction. `--raise-topo` was adopted as the physically better treatment.

## What was found that changed the science

1. The lake left the study (0.05% of the domain) and it cost nothing at this receptor.
2. Array shares on the real 2-D rectangle were 1.4–1.6× the idealised crosswind-integrated
   estimates, and the N/E ratio fell to 2.69× neutral.
3. Displacement height was absent and is first order. Kljun at `z_m` 10 → 8.5 m moves the E/W
   share 29.9% → 38.2%. `d` now enters the sub-layer log law, the floor and Kljun's `z_m`.
4. The `z_i` cap is expensive and biased. Excluded deep hours have 1.51× the heat flux.
5. **The array's heat flux was the wrong quantity.** The fourth pass applied sensible-flux
   ratios to a run that needs the virtual flux. Renormalised with the Bowen ratio the array
   multiplier falls 1.60 → 1.376 and water rises 0.12 → 0.151. The array-to-water contrast
   falls about 32% ([site](../problem/site.md)).
6. At `z0_array = 0.10 m` the array is aerodynamically invisible (WorldCover's cropland value).

## Bugs found

FastEddy: subsidence unusable dry (patch 0004). Ours: `stage5_footprint.py` never passed
`z_target`, so every footprint would have been computed at 30 m, with four other hard-coded
30 m receptors. `regression_flat.sh` deleted the window fields before the well-mixed gate could
see them. The convective adequacy stages used the neutral base file.

## The principal open finding: the floor was not well mixed convectively

The neutral gate says nothing about the convective closure, because the floor is inert
neutrally (factor 1.000) and 1.57–1.68 convectively. Run convectively, Gate D1 failed forward
(lowest three bins 1.258). The diagnosis given here, a spurious `σ_w²` maximum at the taper's
inner edge, **was wrong**. The [sixth pass](pass-6.md) found the cause was the magnitude of the
inflation, and that the retired closure had inflated the convective array share by up to 18
points, not the "2–4%" estimated here. The sub-grid fraction at the receptor was 96.5% neutral
and 90.7–91.4% convective. Reaching 40% would have needed `dx ≈ 3 m`.

Removed from the tree on 2026-09-04 (in the offline pre-cleanup archive of 2026-09-04): `runs/g16_*`,
`results/g16_*`, `bin/run_pass5.sh`, `bin/run_campaign.sh`, `bin/regression_flat.sh`,
`bin/phase2*.sh`, `bin/phase3.sh`, `bin/phaseB_b5b6.sh`, `bin/b6_convective.sh`,
`bin/spin_cbl.sh`, `bin/run_directions.sh`, `bin/pick_tback.py`, `bin/upwind_transect.py`,
`bin/stage6_predict.py`, `bin/subgrid_gate.py`, `bin/subgrid_apriori.py`, `bin/make_figures.py`,
`bin/fig_static.py`, `bin/fig_gate6.py`, `bin/fig_closure.py`, the 16 m seed library `jobs/`.
