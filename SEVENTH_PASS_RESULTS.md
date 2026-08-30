# Seventh pass — 30 m receptor on 122³ @ 24 m

**2026-08-29.** Branch `library-states`. Everything below is measured; where a number is a
projection it says so.

---

## Why the configuration changed

Two measurements from the completed 10 m validation, and neither is an argument:

1. **The peak did not respond to meteorology.** 48 m in all three target cases — three
   soundings, three rotations, three achieved directions — max/min **1.00×**, while A80
   spanned 2.54×. A peak that cannot move cannot beat Kljun, whose peak does.
2. **LES `σ_w` at the receptor ran 2.33–2.99× the tower median** with the closure floor
   INACTIVE (receptor factor 1.000, ~90% sub-grid). The near field was closure output.

`PROJECT_BRIEF.md` had recorded the mechanism as a risk when the 16 m grid was chosen — the
energy-containing eddy scale goes as `z` while `Δ` does not, so at `z/Δ ≈ 1` the isotropic
`(2/3)e_sgs` partition sets `σ_w`. It materialised. Refining `Δ` to ≤ 2.9 m costs ~22× and
was ruled out long ago, so the receptor was raised instead: **30 m on 122³ @ 24 m**, a
2928 m box. The real tower stays at 10 m; the model receptor is a deliberate methodological
choice for resolution adequacy, and the emulator now predicts a footprint the instrument
does not measure.

---

## Phase A — the configuration

| item | result |
|---|---|
| grid | 122³ @ 24 m, receptor **exactly 30.000000000 m** on the k=3 cell centre, `dz_sfc` 8.5583 m, `d_zeta` 24.691358, `verticalDeformFactor` 0.346601, `zCeiling` 3000 m |
| Δ / z/Δ | **17.05 m / 1.76** against 10.09 m / 0.99 at 16 m |
| cost, MEASURED | **0.481 GPU-h per simulated hour** (14.22 ms/step) against 0.94–0.99 at 16 m |
| **B3 flat `dt`, RE-MEASURED** | boundary between **1.55** (`k0/k1` 0.132) and **1.60** (8.511). Production `CFL_3d` 1.3442, `dt` 0.0295858 s = 5/169 s — **17% margin** |
| B5 restart injection | **PASS** — topoPos 4.2e-08, z0m 1.5e-09, zPos 0.0, htFlux 4.0e-08, all at the fp32 storage floor |
| rotation check (static) | **PASS** — four turns the identity bit-for-bit, FROM bearing exact at every turn |
| taper | `pad = 10` re-measured as the knee at this spacing: real terrain to **1224 m**, slope p99 0.1361, max slope unchanged at 0.2020 |

**The accuracy boundary is NOT the retired 24 m grid's.** That grid had the same anisotropy
(`dx/dz_sfc` = 2.804) and its boundary was ~1.64; this one is ~1.575. The margin rule
survives, the number does not — which is exactly why `PROJECT_BRIEF.md` says re-measure at every
grid.

### The lake is back, and Gate A1 fails in one regime

| | 1952 m box | **2928 m box** |
|---|---|---|
| water | 0.05% (8 cells) | **8.78% (1307 cells)** |
| worst-case footprint water share | 0.01% | **17.45%** |

**Gate A1 FAILS at 17.45% against a 10% threshold — and that worst case is very stable
easterly, a regime the corpus does not contain** (`STABLE_REGIME_RESULT.md`). Over the
regimes it does contain the worst case is **7.38%** (neutral easterly), which passes. Both
numbers are recorded; quoting only the second would be the mistake this project keeps
making. What is genuinely new is that easterly footprints now carry real water, with a
virtual flux ratio of 0.151 — a buoyancy heterogeneity the 10 m box did not have.

### The array leaves the footprint for E/W winds

Kljun on the real map at `z_m = 30 m`, neutral: array share **N 30.7%, E 0.04%**, against
**80.6% / 29.9%** at 10 m. `x_peak` is 126–206 m over all stability classes and the array
reaches only 60 m east and west of the tower, so for E/W winds the peak is well past it.
**"This tower measures the solar array in every wind direction" is a 10 m statement and is
false at 30 m.** The directional signal is now presence-versus-absence (N/E ratio 50–760×),
and Gate F must lean on absolute share by direction rather than on any ratio.

### The corpus band, and what widening it bought

`z_i` **300–1250 m** (was 100–976): the floor is `10 z_m` and tracks the receptor; the
ceiling is the lower of the width constraint `L ≥ 2 z_i` (1464 m) and the domain-height
constraint (~1250 m). Measured on identical code: day coverage **75.0% → 80.4%**,
**1370 → 1469 cases** over five years.

**It bought cases, not fairness.** Among runnable unstable hours the deep exclusion still
carries **2.33×** the mean surface heat flux of the accepted set (136 vs 58 W/m²) against
2.44× before, and the widening adds only **+4.0%** of runnable unstable hours — because the
excluded population sits at a median `z_i` of 1716 m, far above either ceiling.

---

## Phase A — the five fixes

1. **The integral asymptote is `1 − z_m/z_i`, not 1** (Steinfeld et al. 2008, after Horst &
   Weil 1992). At 10 m in an 800 m CBL that is 1.25% and invisible; at 30 m it is **3.75%**,
   the size of effects this project routinely gates on. Reported by `stage5_footprint.py`,
   carried into every pair, and quoted by `corpus_monitor` G2b beside Kljun-on-identical-
   cells — which stays the primary reference because it also carries the domain truncation.
2. **Negative footprint values are physical and nothing clips them.** Audited and now
   asserted (`bin/test_negative_lobes.py`): the estimator is signed by construction, CIC
   takes signed weights, the persisted array is unclipped, and the `np.maximum(f, 0)` calls
   that exist are metric-side and deliberate. **Measured across twelve production
   convective footprints the negative lobe carries 5.8–11.1% of |flux|**, and its centroid
   sits 2.5–5× further out than the positive lobe's — the wind-turning mechanism rather
   than the CBL elevated-maximum one.
3. **The tower `σ_w` check is translated 10 → 30 m** (`bin/sigma_w_tower.py`): invert
   `σ_w(10) = 1.25 u* φ_w(10/L)` for `u*` by fixed point, then predict at 30 m; `u*` is
   constant through the surface layer and `H` is a surface flux, so only `φ_w` moves. `φ_w`
   is **imported** from `lpdm/sgs_floor.py`. The lift is **1.006–1.238×** and is above 1 in
   BOTH regimes, because the ratio is exactly `φ_w(30/L)/φ_w(10/L)` and `ζ` triples with
   height. At the convective end the tower says `σ_w(30) = 0.848 m/s`, IQR [0.772, 0.942].
   Two stated caveats: the file carries no wind speed so the IQR spans ~2×; and on the
   stable side surface-layer MOST makes `σ_w` rise with height where a real SBL has it
   fall, so those bins are an upper bound — costless only because the corpus has no stable
   cases.
4. **That check is now an acceptance gate**, not a diagnostic: `run_corpus_case.sh` stage 7c
   refuses a case whose window `σ_w` falls outside the tower IQR for its own `H`.
5. **Steinfeld's spin-up accelerator** on neutral rungs: 3000 s at `surflayer_wth = +0.05
   K m/s`, then the run at the rung's own flux, restarting with `htFlux` **zeroed in the
   file** and read back to confirm — because `htFlux` is IO-registered and the `.in` cannot
   override it.

Plus the one thing that had to ship rather than be designed: **touchdown persistence**. The
window fields are deleted at the end of every case, so a touchdown not captured at stage 5
is gone. `ML_TARGETS.md` has the design; the persistence is built, uniform (bottom-k on an
independent key), signed, unfolded, and asserted to reproduce the full ensemble's integral.

---

## Phase A — the GPU-resident LPDM: **ACCEPTED**

`SRC/LPDM/CUDA/` on the `kegonsa` fork, built as `lib/liblpdm.so` and driven from
`lpdm/gpu.py`. A VRAM ring buffer holds `t_back` of history at fp16 (six fields, 4.2 MB per
field per snapshot); the backward ensemble integrates in-kernel with **fp64 particle
state**. The whole production closure is transliterated, not reimplemented: the
f_sgs-weighted MOST floor with its product-rule drift and ε-scaling with both sides floored,
the displacement-corrected sublayer log law, the surface-normal touchdown weight, the
one-domain-length wrap cap, the double-rotation flux weight, signed CIC deposition.

Acceptance (`results/gpu_lpdm_acceptance.txt`), on a 900 s convective window, both paths
reading the same fp16 fields with the same floor table and the same release times:

| test | result |
|---|---|
| (d) ingest — `kDerive` vs `lpdm/fields.py` | **PASS**, `eps` median rel 2.0e-03 (the fp16 floor), `dsig2dz` 0.0 |
| (a) footprint vs the CPU path's own half-vs-half floor | **PASS** on peak, centroid, A80, integral and array share |
| (b) well-mixed, 200,000 particles | **backward PASS** — GPU lowest-three 0.999 against CPU 0.995, agreeing to 0.004 where three combined SE is 0.060 |
| (c) signed weights | **PASS** — negative lobe 7.7% GPU against 9.0% CPU |
| cost | 9.3 s CPU (12 forked workers) vs 0.06 s GPU — **153×** |

**Forward D1 fails in BOTH paths on that window** (lowest-three 1.093 CPU, 1.095 GPU,
agreeing to 0.002). The control is what establishes that it is the window — 900 s of a
convective layer 1800 s old — and not the port. It is not a substitute for Gate D1 on a seed
window, which is still owed.

**What is not done, and it is the whole deployment argument:** the in-FastEddy hook. Until
the ring buffer is filled from the live device fields inside the time loop, a corpus case
still writes ~16.6 GB of scratch and reads it back. The integrator is validated; the
plumbing is not built.

---

## The resolution split — the number that predicts the rest

Resolved fraction of `σ_w²` at the receptor, same grid, same `z/Δ = 1.76`:

| regime | resolved | sub-grid |
|---|---|---|
| convective | 47.5% | **52.5%** |
| **neutral** | 13.6% | **86.4%** |

The fourth pass measured 52.3% / 85.5% at a 30 m receptor on a *different* domain
(186² @ 24 m), so this reproduces to within a point — the resolved fraction is a property of
`z/Δ` and the regime, exactly as claimed. Against the retired 10 m configuration: neutral
96.4% → 86.4% sub-grid (resolved 3.6% → 13.6%, a factor of 3.8), convective ~90% → 52.5%.

**Stated before the targets ran:** a 30 m receptor takes the *convective* half of the corpus
out of the closure-dominated regime and leaves the *neutral* half in it. If the peak moves,
expect the convective case to be why.

---

## Containment — first evidence, and it is not reassuring

The integral by trajectory displacement, on a convective control window:

| cap | integral |
|---|---|
| 0.25 L (732 m) | 0.814 |
| 0.50 L (1464 m) | 0.951 |
| 0.75 L (2196 m) | 1.009 |
| **1.00 L (2928 m)** | **1.043** ← the production cap |
| 1.50 L, 2.00 L | 1.043 (unchanged: the cap binds, by construction) |

**Still climbing at 0.75 L — +3.4% over the last quarter — so influence has not run out
inside one domain length.** Gate G2a passes trivially because the ratio past the cap is
1.000 by construction; it tests that the cap BINDS, not that the footprint is contained.
The flat/neutral containment gate is where this gets settled, and it is deferred but
required before any corpus.

---

## Phase B — `nbl-deep` + accelerator: **FAIL, and nothing is drifting**

Ran to the 3.0 sim-h ceiling. Cost **0.483 GPU-h/sim-h**, 86.9 min wall, −49% against the
sanctioned seed class.

| item | result |
|---|---|
| log | 0 CORRUPTED, 0 NaN, 0 Inf, completion banner present |
| `k0/k1` | 0.135 |
| `turb_alive` | **OK** (a real OK, not a SKIP) |
| Ozmidov at the receptor | `L_O/Δ` **311** against a requirement of 10 |
| **Gate C2** | **PASS bit-for-bit**, 23 variables, 0 differ |
| rotation | **PASS**, exact at all four turns |
| seven limits | **FAIL — five INDETERMINATE, none DRIFTING** |

Achieved: `u*` 0.4058, `U(10 m)` 6.502, `σ_w` 0.4172, `z_i` **684 m** against a 550 m target
(the accelerator's convective burn-in built it deeper), direction FROM 249.9 (Ekman backing
+20.1°), drift at freeze **−6.13 deg/h**.

`σ_v/u*` and Kljun `x90` are `ok`; `z_i`, `TKE_BL/u*²`, `σ_w/u*`, `U/u*` and `x_peak` are
INDETERMINATE. **`bin/seed_budget.py` sweeps a fixed 2.0 h window over end times and finds
the immune limits in band nowhere up to 3.0 h** — one of five resolves at 2.75 h. So this
rung needs more than the ceiling, and the honest report is the margin at the ceiling rather
than a stop time.

**And the live watcher structurally cannot measure the budget on a 3 h run.** Its scoring
window has to be a trailing fraction of elapsed time, so it never reaches the 2.0 h width
those trends need; it runs to the ceiling, and reporting that as "the measured stop time"
would be reporting the ceiling. That is why `seed_budget.py` exists.

---

## Phase C — `cbl-deep`: the lock-in is largely gone

Full write-up in `SEED_CBL_DEEP_24M_RESULT.md`.

| | 1952 m box | **2928 m box** |
|---|---|---|
| `L/z_i` (spectral `z_i` 986 m) | 1.53 – 1.98 | **2.97** |
| **mode-1 share of mid-depth `w` variance** | **53.9 – 72.0%** | **19.3 – 23.1%** |
| peak wavelength | `L` in 5 of 5 dumps | `L` in 3 of 4, `L/2` in one |
| `r(L/2)` | positive | **−0.076 to +0.067**, i.e. zero |

Phase E's compliant reference at `L/z_i` 4.56 had **4.8%** in mode 1; its deep case at 2.28
had **50.2%** and was ACCEPTED as footprint-indistinguishable (p ≈ 0.54). **19.3–23.1% sits
well below the level Phase E accepted, and `r(L/2)` is no longer anti-correlated — by Phase
E's own standard the box no longer organises the thermals.** The honest residual: the peak
wavelength still lands on mode 1 in three dumps of four.

**The depth overshoot is unchanged and belongs to the rung, not the box** — 1308 m (fixed
threshold) / 1065 m (peak fraction) against 1276 / 1055 at 1952 m. `L/z_i` is now 2.24–2.97,
all above the 2.0 floor; but 1308 m exceeds the 1250 m band ceiling, so the rung as specified
builds a state the corpus would refuse a CASE for. Re-specifying it is flagged, not made.

Stationarity **FAIL — all seven INDETERMINATE, none drifting**; `seed_budget.py` finds the
immune limits in band nowhere up to 3.0 h, with `z_i` reading DRIFTING from 2.25 h onward,
consistent with a layer still deepening. Cost 0.48 GPU-h/sim-h.

---

## THE DECIDING TEST: **THE PEAK MOVES**

Pre-registered before either target ran (`results/deciding_test_preregistration.txt`).

| | A `case_2023052519` convective | B `case_2023121921` near-neutral |
|---|---|---|
| forcing | `H` 333 W/m², `w'θ'_v` 0.291, HRRR `z_i` 970 m | `H` 22 W/m², `w'θ'_v` 0.019, `z_i` 447 m |
| achieved `u*` / `U(30)` / `σ_w` | 0.457 / 5.06 / 0.659 | 0.568 / 8.50 / 0.636 |
| achieved `L` / `z_i` / direction | **−25.5 m** / 1229 m / 89.2° | **−732 m** / 937 m / 177.0° |
| **LES peak** | **144 m** | **288 m** |
| its own half-vs-half \|Δpeak\| floor | **24 m** (1 cell) | **0 m** |
| Kljun peak on identical cells | 144 m | 168 m |
| centroid | 334 m at 76.6° | 335 m at 176.4° |
| A80 | 20.22 ha (Kljun 23.73) | 23.21 ha (Kljun 12.56) |
| integral / the `1 − z_m/z_i` asymptote | 1.463 / **1.497×** | 0.888 / **0.916×** |
| array share ± SE (10 groups) | **5.65 ± 1.44%** | **1.14 ± 0.37%** |
| sub-grid fraction at the receptor | 34.0% | 75.6% |
| floor factor at the receptor | 1.912 | **1.000 — inert** |
| `σ_w` vs the translated tower | 0.78× median, **OUTSIDE the IQR** | 1.14× median, **INSIDE** |
| seed | `cbl-deep` rot 2, gap 23.1° | `cbl-deep` rot 1, gap 31.5° |
| health gate | all `ok` | all `ok` |

**|Δpeak| = 144 m against the larger of the two cases' own floors, 24 m — six times it —
and the LES ordering matches Kljun's.** At 10 m the peak was 48 m in all three targets,
max/min 1.00×; here max/min is **2.00×**.

### And it is not the closure. The no-op control says so.

Each footprint was recomputed on the identical window with the `σ_w` floor OFF:

| | peak ON | peak OFF | integral ON | OFF | array share ON | OFF |
|---|---|---|---|---|---|---|
| A convective | 144 m | **144 m** | 1.463 | 1.634 | 5.65% | 3.72% |
| B near-neutral | 288 m | **288 m** | 0.888 | 0.886 | 1.14% | 1.07% |

**The peak is bit-identical with and without the closure in both cases**, and for B the
floor is inert at the receptor to begin with (factor 1.000). So the 144 m separation is the
LES's, not the floor's. At 10 m the same floor was worth +8.40 points of array share and
shortened `x80` from 400 m to 227 m.

### Two things the pair also settled, for free

**`t_back` at a 30 m receptor: 600 s is enough.** The capture curve, from touchdown ages
already in hand:

| `t_back` | A: fraction of the 900 s integral | B |
|---|---|---|
| 150 s | 73.5% | 53.5% |
| 300 s | 89.6% | 84.6% |
| 450 s | 97.8% | 98.7% |
| **600 s** | **99.6%** | **100.0%** |
| 750 s | 99.9% | 100.0% |

and **the peak is at its final value at every mark from 150 s in both cases**. Production
ran 900 s (the fourth pass's value); 600 s would cost 300 s of simulated time per case,
6.7% of the class. Recorded, not changed.

**The integral's departure from the asymptote tracks the mean vertical velocity at the
receptor, with the right sign.** A sits in mean subsidence (`W = −0.099 m/s`) and integrates
to 1.497× the ceiling; B sits in a mean updraft (`W = +0.342 m/s`) and integrates to 0.916×.
That is the advection non-closure `PROJECT_BRIEF.md` already describes — over a slope the turbulent
flux genuinely is not the surface flux — and it is now measured on two cases with opposite
signs rather than asserted.

### What the test does NOT establish

- **Both targets came off the same seed, at different rotations.** `pick_seed` chose
  `cbl-deep` for the near-neutral case too, because direction dominated the cost (31.5°
  against `nbl-deep`'s best 43.7°) even though `nbl-deep`'s depth was much closer. That
  makes the peak difference *harder* to explain as a seed artifact, not easier — it is the
  same turbulence field, re-indexed — but it does mean the pair is not a two-seed test.
- **Case A fails the `σ_w` acceptance gate** (0.78× the tower median, outside the IQR). By
  the criterion this pass adopted it is not a usable corpus target. Reported, not waived:
  one refusal in two is not a rate, and the gate is doing exactly what it was built for.
- **Neither footprint is fully contained.** A's integral is still rising 1.447 → 1.463 over
  the last quarter-domain, B's 0.840 → 0.888 (+5.7%). The wrap cap is binding on real
  influence in both.
- **Seed mismatch is large and the adjustment did not close it.** A: `z_i` 1065 → requested
  970 → achieved **1229 m**; direction 77.0 → requested 54.0 → achieved **89.2°**, i.e. the
  gap WIDENED from 23.1° to 35.2°. B: `z_i` 1065 → 447 → **937 m** (+490 m); direction
  167.1 → 198.5 → **177.0°**, gap 31.5° → 21.5°, the one case that closed. Both windows
  back at **+9.0 to +9.3 deg/h** where the seed was backing at **−4.71 deg/h** at freeze —
  **opposite in sign**, which is new: on the 16 m library both cases inherited the seed's
  sign and only overshot its magnitude.

---

## New traps, all in `FASTEDDY_TRAPS.md` §19

- **19** — a grid constant that is really a grid property, five instances in one day
  (`surflayer_z0`, the gate's receptor, the representability screen's `z_m` and `z0`,
  `ozmidov`'s `dx`, the battery's final-dump assumption). Every one would have produced a
  plausible number.
- **19b** — a `z/L` column that is at 10 m and does not say so; reading it as 30 m
  understated every unstable case threefold and changed which target looked most convective.
- **19c** — a comparison harness that built its own release times and produced a 27% gap
  that looked exactly like the port bug it exists to detect.
- **19d** — editing a shell driver while it is running; bash reads by byte offset.
