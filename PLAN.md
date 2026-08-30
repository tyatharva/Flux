# Staged Plan — EIGHTH PASS, 122^3 @ 30 m, the in-process hand-off, two footprints per case

> **THE SEVENTH PASS ANSWERED ITS QUESTION: THE PEAK MOVES.** 144 m and 288 m on two
> pre-registered targets, six times the larger case's own half-vs-half floor, ordering
> matching Kljun, and identical with the `sigma_w` floor OFF — so the LES moved it, not the
> closure. That settles the 30 m receptor. What it left open is three things, and this pass
> is those three.

## Where this pass stands

| phase | item | verdict |
|---|---|---|
| **0** | grid solve, surfaces at the re-measured taper knee (pad 12) | **done** |
| 0 | **Gate A1' water at 3660 m** | **FAIL 11.58% over corpus regimes** — and it is the SITE, not the box (below) |
| 0 | a priori resolution cost of dx 24 -> 30, no GPU | **done** — 87.2% of resolved variance kept at a 4dx filter |
| 0 | `lpdm/dumpsrc.py` + `bin/test_dumpsrc.py` | **PASS bit-identical**; caught a real step-parsing bug |
| 0 | `SRC/IO/io_lpdmonline.c` + `lpdm/ringsrc.py` + `bin/test_ringsrc.py` | **PASS bit-identical**; caught a mixed flux estimator |
| 0 | `stage5_footprint.py --ring` end to end | **PASS 0.00e+00** on an identical 60-snapshot window |
| **1** | 30 m bring-up: cost, flat `dt` boundary, k0/k1, turb_alive | **PASS** — boundary **1.50-1.55**, production `dt` = 5/162 s (CFL 1.3502, 10.0% margin), **0.479 GPU-h/sim-h** |
| 1 | B5 restart injection, static rotation check | pending |
| **2** | seeds `cbl-deep`, `nbl-deep` (3.0 sim-h ceiling) | pending |
| **3** | flat/neutral control: regression, D1-neutral, **containment at 2.5 L** | pending |
| **4** | two dual-output 2.0 h cases: in-process acceptance, D1-convective, window independence, peak-motion re-check | pending |

## Why 3660 m, and what it costs

The containment gate FAILED for neutral at 2928 m (`CONTAINMENT_RESULT.md`). 186^2 @ 24 m
buys full containment at **+132%** (820 -> 2150 GPU-h); 122^3 @ 30 m buys 25% more box for
**3%**. A relative claim against Kljun does not need full containment, and the parity number
is what makes that defensible: **LES 0.874 of its asymptote against Kljun's 0.867 on
identical cells.** Acceptance is that the neutral integral SATURATES by 2.5 L, not that it
is complete.

**GATE A1 FAILS AND THE BIGGER BOX IS WHY — but not in the way that would argue for
shrinking it.** Corpus-regime worst case 11.58% (neutral easterly) against 7.38% for the
same direction and regime at 2928 m, while Kljun's `x90` moved only 1615 -> 1665 m. The
physical footprint is the same; the 2928 m box was replacing the lake between 1464 and
1830 m east with a periodic re-sample of its own land. **The old PASS was truncation.**
Recorded as a site limitation; ~20% of the wind rose (E and NE) carries 6-12% water and
every other direction carries ~0%.

## The three items

1. **THE IN-PROCESS HAND-OFF — built, CPU-tested, GPU acceptance owed.** `~20 GB` of window
   scratch per case becomes `~3 MB`. Two things the design settled that the brief assumed
   otherwise: the ring holds a FULL WINDOW (541 slots, 6 fields, 12.0 GB) rather than
   `t_back` (180 slots, 4.0 GB), because the `sigma_w` floor is built from whole-window
   statistics and a shorter ring forces the floor onto partial ones — an estimator change
   wearing a plumbing change's clothes; and the route into the VRAM ring is host memory
   over tmpfs rather than CUDA IPC, because the validated fill entry point takes host
   pointers and the hop costs ~4 ms per 5 s of model time.
2. **THE GRID**, above.
3. **TWO FOOTPRINTS PER CASE.** 1800 s adjustment + two 2700 s windows; window 2's releases
   begin 900 s after window 1's end, so the two windows' FIELD INTERVALS ARE DISJOINT by
   construction. Both are separate training pairs, tagged by parent, always on the same
   side of the split. Two things get measured on the first case that does it, and both get
   reported: **(a)** are the two windows independent — from the 20 release groups'
   decorrelation ladder, and from |w0 - w1| against the within-footprint half-vs-half floor
   (~floor x sqrt2 if independent, ~0 if near-duplicates); **(b)** how far `z_i` drifts
   between them — near-replicates at one condition REDUCE realisation noise, two conditions
   are coverage and do NOT, and which one it is decides whether the corpus still owes
   condition-bin averaging.

## Acceptance for the hand-off, on the GPU

Run at `lpdmOnlineSelector = 2` so one LES produces both paths from bit-identical buffers —
the only way to score the hand-off without comparing across two turbulence realisations,
which on this project differ by 44% in the integral.

- **(a)** CPU-from-disk vs GPU-from-ring, per footprint, within the half-vs-half floor.
- **(b)** Gate D1 well-mixed, both directions, both regimes, THROUGH the ring.
- **(c)** signed weights preserved; negative-lobe share reported on both paths.
- **(d)** fp16 parity is inherent to (a): disk is CF-packed 16-bit, the ring is IEEE fp16,
  and the ring is the more accurate of the two.
- plus `window_stats`-from-disk vs from-ring, which is what catches a slot-ordering or
  rho-division error in the producer.

**If (a) or (b) fails: fall back to the CPU path with dumps, and say so.**

## Stop rules

- In-process (a) or (b) fails -> CPU fallback, said plainly.
- **Peak separation collapses at dx 30 -> stop; that is a grid decision and it is the
  user's.** The 24 m deciding-test result does NOT certify this grid: the sub-grid fraction
  is re-measured at every grid and never carried.
- Water gate over 10% in corpus regimes -> already true, recorded, not a blocker.

## Still deferred, still required before any corpus

1. **Gate D1 on the production closure, both regimes, both directions, on a SEED window.**
   Acceptance (b) exercises it through the ring, which is necessary and not sufficient.
2. **Rung re-spacing.** `nbl-shallow`'s 300 m target sits on the `z_i` floor.
3. **`zCeiling`.** The 3660 m width would support `z_i` to 1830 m at `L >= 2 z_i`; the band
   stays 300-1250 m because the VERTICAL sets it. Flagged, not done.

---

# Staged Plan — SEVENTH PASS, 30 m receptor on 122^3 @ 24 m

> **THE SIXTH PASS'S CONFIGURATION IS RETIRED, 2026-08-29.** The 10 m receptor produced a
> footprint whose peak did not move: 48 m in all three target cases across three soundings,
> three rotations and three directions, max/min **1.00x**, while A80 spanned 2.54x — and
> LES `sigma_w` at the receptor ran 2.33-2.99x the tower median with the closure floor
> INACTIVE. The near field was closure output. See the dated block at the top of
> `PROJECT_BRIEF.md` for the new configuration and everything it reverses. **Every absolute
> distance below this line, and in every earlier pass, is a 10 m number.**

## Where this pass stands

| phase | item | verdict |
|---|---|---|
| **A** | grid solve, surfaces, geometry, coverage | **done** |
| A | B1 launch + measured cost | **PASS** 0.481 GPU-h/sim-h |
| A | **B3 flat `dt`, RE-MEASURED** | **PASS** boundary 1.55/1.60, production 17% below |
| A | B5 restart injection at 24 m | **PASS** topoPos 4.2e-08, z0m 1.5e-09, htFlux 4.0e-08 |
| A | static rotation check | **PASS** bit-for-bit, FROM bearing exact at all four turns |
| A | Gate A1' water | **FAIL 17.45%** very stable easterly; **PASS 7.38%** over corpus regimes |
| A | five analysis fixes + touchdown persistence | **done** |
| A | **GPU LPDM acceptance** | **PASS** (a)(b)(c)(d); 153x; forward D1 fails in BOTH paths on the bring-up window |
| A | `ML_TARGETS.md` | **done** (design only, plus the persistence that had to ship) |
| **B** | `nbl-deep` + accelerator, 3.0 h ceiling | **ran clean, gate FAIL** — 5 INDETERMINATE, none drifting; budget not in band up to the ceiling |
| **C** | `cbl-deep` — the lock-in re-test | **mode-1 share 53.9–72.0% → 19.3–23.1%**, below what Phase E accepted; depth still overshoots to 1308 m |
| C | pre-target go/no-go: sub-grid fraction at the receptor | **GO** — 52.5% convective against ~52% expected |
| C | two targets, then **THE DECIDING TEST** | **PASS — the peak moves 144 m against a 24 m floor, ordering matches Kljun, and the no-op control shows it is not the closure** |

## THE DECIDING TEST, pre-registered

**Does the peak MOVE between the two targets?** `|Δpeak|` must exceed each target's own
half-vs-half floor (the 30 m ensemble table gives peak p90 = **60 m**, ~2.5 raster cells,
where the 10 m table gave 0 m) AND order the way Kljun's `x_peak` orders the two cases.
Expected if the configuration worked: peaks **150-250 m** upwind, on the array for a
northerly and past it for E/W, share falling with A80 growing to ~3-8 ha against the array's
1.03% of the box by area. **If the peak still does not move, the receptor is still
closure-dominated and the configuration has not worked — say so plainly and stop.**

A cheaper leading indicator runs first, off the convective control window and before any
target: **the sub-grid fraction of `sigma_w^2` at the receptor, expected ~52%** against
~90% at 10 m. If it does not land near that, the peak test is likely to fail and it is one
number rather than two more GPU-hours.

## Deferred, and REQUIRED BEFORE ANY CORPUS

1. ~~The flat/neutral containment gate.~~ **RUN 2026-08-30: FAIL for neutral, PASS for
   convective** (`CONTAINMENT_RESULT.md`). The flat/neutral control's integral needs
   **1.5 domain lengths** to stop growing and the cap removes 6.1%; the convective target
   saturates by 1 L with the cap removing 0.8%. `x80` PASSES in both, because it is
   crosswind-integrated and the missing influence is a long thin tail — the wording above
   ("does the 80% source area resolve") would have called this contained, and the
   integral's approach slope is what actually answers it.
   **Mitigating and measured: Kljun on the identical cells is truncated to 0.867 of its own
   asymptote against the LES's 0.874**, so the two lose the same fraction and an
   LES-vs-Kljun comparison on this box is fair. The corpus's targets carry a systematic
   ~12% truncation deficit that is a property of the domain.
   **The grid decision is now the user's, with the numbers**: as-is captures 93.5% of the
   saturated integral at 0.63 GPU-h per case; 146² @ 24 m ~96.7% at +48%; **186² @ 24 m
   (4464 m, past saturation, and the fourth pass's own grid) ~100% at +132%**, taking the
   corpus from ~820 to ~2150 GPU-h.
2. **Gate D1 on the production closure, both regimes, both directions, on a SEED window.**
   The acceptance suite ran D1 through both integrators on a 900 s bring-up window and
   **both failed forward** (lowest-three 1.093 CPU, 1.095 GPU, agreeing to 0.002). That is
   a statement about that window — 900 s of a convective layer 1800 s old — and it is not a
   substitute for the gate. Until it runs, the closure at 24 m has no evidence.
3. **The in-FastEddy hook for the GPU LPDM.** The integrator is validated; the ring buffer
   is not yet filled from the live device fields inside the time loop, so a case still
   writes ~16.6 GB of scratch. Eliminating that is the entire deployment argument for the
   port.
4. **Rung re-spacing.** The `z_i` band moved to 300-1250 m and `nbl-shallow`'s 300 m target
   now sits exactly ON the floor. Whether the five-rung ladder should be re-spaced is a
   post-validation decision and is flagged, not made.

---

# Staged Plan — Fifth Pass, 10 m receptor, 122^3 @ 16 m

> **THE CORPUS PHASE IS NOW `LIBRARY_PLAN.md`, 2026-08-25.** "After Phase F: corpus design,
> wind-rose stratification with a directional floor" is answered there and answered
> differently: the corpus is **~1825 HRRR-forced cases**, one per day over five years, not
> a stratified sweep — so the wind rose enters through the weather itself rather than
> through a sampling rule. The 18 **seed states** are pre-spun turbulence that exists only
> to delete each case's 3 h spin-up. **The forcing source changed from CONUS404 to HRRR**;
> see the section at the top of `PROJECT_BRIEF.md`.

> **SIXTH PASS COMPLETE, 2026-08-24 — the sigma_w closure.** `SIXTH_PASS_RESULTS.md`.
> The convective well-mixed gate passes in both directions for the first time; production
> is regenerated on the corrected closure and on `--raise-topo`. The retired closure was
> inflating the convective array share by up to 18.46 points. **The target pipeline is
> done; the ML phase below is next.**

> **STATUS: COMPLETE, 2026-08-22.** Every phase ran and every gate passed. Results in
> `FIFTH_PASS_RESULTS.md`. Absolute distances from earlier passes do not carry over;
> methodology, traps and closure findings do.
>
> | phase | verdict |
> |---|---|
> | A offline geometry, `z_i` coverage, displacement | **PASS** (Gate A1 water 0.01%) |
> | B smoke batch, 8 items | **PASS** (flat `dt` boundary 1.51; terrain does not lower it) |
> | C spin-ups: neutral + 2 convective | **PASS** (C1, C2, C3 x2) |
> | D flat/neutral control | **PASS** (D1 well-mixed, D2 integral, D3 floor) |
> | E domain adequacy | **PASS** -- `L >= 2 z_i` is not binding, p ~ 0.54 |
> | F 8 production directions + displacement sensitivity | **PASS** (Gate F explicable) |
>
> Total ~30 GPU-h. The campaign ran unattended via `bin/run_campaign.sh` (resumable,
> gated, self-freezing) with `bin/run_pass5.sh` for the gate chain.

Goal: **one validated configuration at a 10 m receptor, a flat/neutral control footprint,
the domain-adequacy answer, and then the production directions.**

> **RETIRED 2026-08-26 — CHAINING IS GONE AND SO IS THE CAP.** A seed and a target case
> are each ONE continuous FastEddy invocation; the only restart left in the project is
> seed -> target. That removes `FASTEDDY_TRAPS.md` §17's failure mode structurally: every
> segment boundary was a restart READ, which overwrites every IO-registered field with
> whatever the restart file holds. **The price is stated rather than hidden** — a seed is
> ~2.9 h wall and a target case ~74 min, both past the old cap, and neither can be split.
> A killed run now costs the whole run. Measured first: chained vs unchained is
> **0.89-1.08x the run-to-run reproducibility floor** (`bin/test_unchained.py`), so
> chained results carry.

**HARD RULE (RETIRED): no single run over 1 hour wall.** At 0.0149 s/step that was **1.02
simulated hours per segment**, so a spin-up was 5-6 chained segments and a whole sampling
window fit in ONE.

**Validate the configuration as ONE thing.** Everything lands together, gets a batch of
short smoke runs, then one full window. The fourth pass established that the expensive
failures are interactions, and those only show up end to end.

---

## The grid, settled

`122 x 122 x 122` @ `dx = dy = 16 m`, domain 1952 m, receptor on a cell centre at exactly
10.000000 m (`k = 2`), `dz_sfc = 3.9933 m`, `d_zeta = 20.576132`,
`verticalDeformFactor = 0.194059`, `zCeiling 2500 m`, `dampingLayerDepth 500 m`, block
`1 x 2 x 64`, `dt = 0.0146417 s` flat (`CFL_3d = 1.35`). **0.0149 s/step measured**,
~0.97 GPU-h per simulated hour. Solved and re-derivable with `bin/vgrid.py`.

Chosen for corpus economics: an 8-case campaign costs ~12 GPU-h against 42 at `186^2 @ 10 m`.
The costs are `z/Delta ~ 0.99` at the receptor, a footprint peak 1.7-5.7 cells from the
tower, and the `z_i` cap below.

---

## Phase A — offline. COMPLETE, committed (a6048c0)

| item | result |
|---|---|
| `bin/vgrid.py` | solves the vertical grid from FastEddy's `zDeform`; reproduces the retired 24 m grid exactly |
| `bin/phaseA_geometry.py` | **Gate A1 PASS**: worst-case water share **0.01%** vs a 10% threshold |
| `bin/zi_coverage.py` | `L>=4 z_i` covers **19.3%** of convective midday; `L>=2 z_i` **60.9%**; the cap is biased (excluded hours carry **1.51x** the heat flux) |
| displacement height | into the LPDM sub-layer log law, the MOST floor, and Kljun's `z_m`; worth **3.6-8.4 points** of array share |
| `bin/prep_surface.py` | rebuilt at 122^2 @ 16 m; virtual-flux conversion; `dmap.npy`; `--raise-topo`; taper `pad = 12` measured as the knee |
| the sounding fit | **impossible** — CONUS404 hourly has no atmospheric profiles. The inversion is a CONTROL on `z_i`, not a target. |

Also fixed: `stage5_footprint.py` never passed `z_target`, so every footprint would have
landed on the level nearest **30 m**. Four other hard-coded 30 m receptors with it.

---

## Phase B — smoke batch. 7/8 COMPLETE, committed (5fe9c77)

B1 launch, B2 thread blocks, B3 flat `dt`, B5 restart injection, B6 equivariance,
B7 subsidence, B8 halo check — **all PASS**. See `PROJECT_BRIEF.md` Status for the numbers.

**B4 — terrain `dt`. PASS, and it did not lower `dt` at all.** A ladder from `CFL_3d` 1.00
to 1.40 over the real surface, branched off a state already adjusted to the terrain, stayed
clean at every rung. **And the standing check could not have told you either way**:
`docker/k0k1_check.py` is a DOMAIN MEAN, terrain amplification is local, and only 1.7% of
this domain exceeds slope 0.14 -- a few ringing columns cannot move a 14,884-cell average.
`bin/k0k1_by_slope.py` conditions on slope and is what actually establishes the result.

Method, learned from B3: a cold start at 500 steps produces `ww[1]` below the `k0k1_check.py`
floor and the check SKIPs, so the ladder only detects the gross acoustic failure. Branch the
ladder off a developed state instead.

---

## Phase C — spin-ups (one continuous invocation each; chaining retired)

| state | target | `w'th_v'` land | simulated | GPU-h |
|---|---|---|---|---|
| neutral | — | 0 | ~5 h | ~4.9 |
| convective shallow | `z_i ~ 490 m` (`L = 4 z_i`) | **0.1363** | ~5 h | ~4.9 |
| convective deep | `z_i ~ 976 m` (`L = 2 z_i`) | **0.1363** | ~8 h | ~7.8 |

`stabilityScheme = 2` + `lsfSelector = 1` + `lsf_horMnSubTerms = 1`. `surflayer_wth` is the
**domain mean of the per-cell virtual map**, not the cropland reference — a flat spin-up has
no restart injection, so the scalar IS its flux.

**The two convective states carry the SAME surface flux on purpose.** `z_i` is separated by
the capping inversion and subsidence alone, so `u*` and `L` match and Phase E tests the BOX
rather than surface-layer physics. The `z_i`-conditioned fluxes (0.095 shallow / 0.142 deep)
are for the Phase F corpus, not for this pair.

- **Gate C1** stationarity: domain TKE plateau, `u*` trend within ~2 sigma.
- **Gate C2** the saved restart restarts bit-for-bit at this grid.
- **Gate C3** (convective) CBL similarity via `bin/cbl_check.py`. It must read the prescribed
  `htFlux`, not the resolved covariance at `k = 0`.

---

## Phase D — the flat/neutral control, and `t_back`

**First full window, and the standing regression.** The only place Kljun is diagnostic, and
the canary that found two real bugs on its first run.

**`t_back` cannot be measured offline any more** — the saved 24 m window fields are gone,
only single spin-up/adjustment dumps remain. So it is measured HERE, for free: run the
control **generously long** (`t_back = 600 s`, window 2400 s, 0.65 GPU-h, one segment) with
`--rel-seconds 1800` and `--tback-marks 60,100,150,200,250,300,400,500`. `compute_footprint`
builds that capture curve from touchdown ages at no extra cost. Read the convergence point
off it and fix production `t_back`. Expect ~150-250 s.

Also record once: well-mixed in the production closure (`--sgs-most`, displacement active);
the integral against Kljun on identical cells; the half-vs-half error floor; the sub-grid
fraction of `sigma_w^2` (**reported, not gated**); the ensemble-convergence curve.

- **Gate D1** well-mixed — backward rms within the counting-noise floor, lowest three bins in
  tolerance. Non-negotiable.
- **Gate D2** the integral converges **from below** with the wrap cap, compared against Kljun
  on the same box, never against 1. Kljun says 92.6-94.1% lies within 930 m here.
- **Gate D3** half-vs-half 80% overlap clearly above the LES-vs-Kljun overlap.

Then re-baseline `bin/regression_flat.sh` at this grid.

---

## Phase E — domain adequacy. THE DECISION EXPERIMENT (~1.3 GPU-h)

Two convective windows in the **same** 1952 m box at `z_i ~ 488` (`L = 4 z_i`) and `z_i ~ 976`
(`L = 2 z_i`), identical in everything including surface heat flux. `bin/domain_adequacy.py`
reports two independent things:

1. **Lock-in, diagnosed directly** — the 2-D spectrum of `w` at mid-depth and the horizontal
   autocorrelation at `L/2`. An unconstrained CBL peaks near `lambda ~ 1.5 z_i`; a locked one
   pins to mode 1 (`lambda = L`), takes an anomalous share of the variance, and goes strongly
   anti-correlated at half a domain length. This detects the artifact **without reference to
   the footprint**, which is what makes the pair interpretable at all.
2. **Footprint observables** against **both** the Phase D half-vs-half floor **and** the Kljun
   null for this `z_i` pair (**+0.24 points** of array share, **-0.9%** in `x90`).

**Gate E.** Agreement inside the floor *and* no spectral lock-in means `L >= 2 z_i` is
acceptable for this observable, convective-midday coverage goes **19.3% -> 60.9%**, and
122^3 covers the corpus. **Disagreement: stop and report the size of the error.** The fix is
`218^2 @ 16 m` (`L = 3488 m`, 3.2x cost, 53.0% coverage at `L >= 4 z_i`) and that is a grid
decision, which belongs to the user.

---

## Phase F — production directions and the displacement-height sensitivity

Four directions per regime by 90-degree re-index, then restart onto the real surface with
~20 min adjustment before sampling. 8 cases ~7.4 GPU-h.

**The displacement-height sensitivity is a paper result**, measured on the northerly (largest
array share), three windows, ~2.0 GPU-h:

| treatment | array `z0` | `topoPos` | receptor |
|---|---|---|---|
| **baseline** | 0.10 m | flat | `k = 2`, exact 10.000 m |
| **raised** | 0.25 m | +1.5 m over the array | fractional, `--exact-agl`, `z_agl = 8.5 m` |
| **bracket** | 0.25 m | flat | `k = 2` (isolates `z0` from the raise) |

Recorded either way: at `z0_array = 0.10` the array is **aerodynamically identical to the
cropland it replaced** (`WORLDCOVER_Z0[40] = 0.10`), so its entire neutral signal is zero and
only the convective heat-flux contrast distinguishes it. `prep_surface.py` warns when the two
coincide.

**Gate F — explicable difference, and it is weaker at 10 m than the plan used to claim.** The
real map gives a N-vs-E/W array-share ratio of **2.69x** neutral, not the ~3.7x the idealised
table suggested and not the ~370x measured at 30 m. The discriminators are:

- **absolute array share by direction**, predicted from the array chord and the LES's own
  `f_y`, against measured — at 30 m these agreed to 0.6%, which is far sharper than a ratio;
- the **near-field peak position** against `bin/upwind_transect.py`;
- **terrain response**, now a larger fraction of what is left once the array is subtracted.

Over terrain and over the array Kljun is descriptive, never a target — the receptor is inside
the RSL there. Quote the anchor-sensitivity band (**46-66% shape L1** against a 38% sampling
floor) alongside any near-field number.

---

## After Phase F

Only then: corpus design, wind-rose stratification with a directional floor, CNF
implementation. **The CNF raster is `122 x 122`** — the LES interior, no halos, confirmed by
`ncdump`.

**Corpus design is done and is `LIBRARY_PLAN.md`.** It replaces wind-rose stratification
with per-day HRRR forcing, which samples the rose by construction; the directional floor
survives only as the seed library's uniform 30-degree spacing, which is a spacing of
RESTART POINTS and not of corpus cases.

Do not start ML work before Phase F passes. A trained model on a broken target pipeline looks
exactly like a trained model on a correct one.

---

## Known limitations to state wherever the corpus is described

> **RE-CUT AT 30 m / 24 m, 2026-08-29.** Items 1, 2, 5 and 7 below were written for a 10 m
> receptor on a 1952 m box and are LARGELY UNDONE by the new configuration; items 3, 4 and
> 6 change value. The seventh-pass list is here and the retired one is kept below it,
> because the reasoning is still the record of how each was arrived at.
>
> **A. The model receptor is 30 m and the instrument is 10 m.** A deliberate methodological
>    choice for resolution adequacy, not a correction. The emulator predicts a footprint the
>    physical tower does not measure, and every comparison with the tower — including the
>    `sigma_w` gate — goes through a MOST translation whose stable branch is an upper bound.
> **B. Containment at 2928 m is not established for neutral.** Deferred item 1 above.
> **C. Gate D1 has no evidence at this grid.** Deferred item 2 above; forward D1 failed in
>    both integrators on the bring-up window.
> **D. The lake is back: 8.78% of the box, and up to 7.38% of a corpus-eligible footprint.**
>    Gate A1 fails outright (17.45%) in very stable easterly conditions, which the corpus
>    excludes. Water's virtual flux ratio is 0.151, so easterly cases now carry a real
>    buoyancy heterogeneity that 10 m cases did not.
> **E. The array leaves the footprint for E/W winds.** Kljun array share at `z_m = 30 m`:
>    N 30.7%, E 0.04% neutral. The directional signal is presence-versus-absence.
> **F. The `z_i` band 300-1250 m is still biased against strong convection.** The
>    deep-excluded unstable hours carry **2.33x** the mean surface heat flux of the accepted
>    ones (136 vs 58 W/m2), barely better than the old band's 2.44x — the widening buys
>    +4.0% of runnable unstable hours and +5.4 points of day coverage, not a fix.
> **G. `nbl-shallow`'s 300 m target sits exactly on the new `z_i` floor.** Rung re-spacing
>    is open.
> **H. The GPU LPDM is validated but not yet in-process**, so the corpus still writes and
>    re-reads ~16.6 GB per case.

### The retired 10 m list, kept for its reasoning



0aa. **INDETERMINATE IS THREE DIFFERENT SITUATIONS AND THEY NEED DIFFERENT ANSWERS.**
   Measured 2026-08-30 (`bin/seed_indeterminate.py`, `results/seed_indeterminate.txt`) by
   sweeping the scoring-window width and watching whether the TREND moves or the SE does:
   * **RESOLVED and never said so** — `sigma_v/u*` (nbl-deep) and Kljun `x90` (both seeds)
     are inside their thresholds by more than 3 SE at 2.0-2.5 h.
   * **NOISY, SE-LIMITED — a longer run buys nothing.** `U/u*` and `x_peak` on nbl-deep,
     `sigma_v/u*` on cbl-deep: the SE is flat across a 2.5x range of window widths
     (1.00x, 1.05x), so INDETERMINATE is simply the state.
   * **NOISY but the SE is shrinking — a longer run WOULD resolve them.** `sigma_w/u*` on
     nbl-deep (SE 1.9x), and `U/u*`, `x_peak`, `TKE_BL/u*^2` on cbl-deep (1.8-2.2x over
     1.0-2.5 h, close to the ideal 1/T).
   * **TRENDING AWAY from band** — `z_i` in BOTH seeds, monotonically (+2.31 -> +4.08 and
     +0.53 -> +7.57 %/h) with a falling SE. A longer run resolves these into a **FAIL**,
     not a pass, and cbl-deep's is the depth overshoot showing up in the gate.
   **AND THE "n_eff SATURATES AT 3-5" CLAIM IS A NEUTRAL STATEMENT.** cbl-deep carries
   n_eff 12-17 on the wind-frame limits, because a CBL decorrelates on `z_i/w* ~ 540 s`
   against a 300 s dump interval where a neutral layer decorrelates on `h/u* ~ 1700 s`.

0. **THE LIBRARY'S SEEDS HAVE UNESTABLISHED STATIONARITY, AND THAT IS THE NORMAL STATE
   RATHER THAN AN EXCEPTION.** Two of the seven gated limits — **`TKE_BL/u*^2` and `z_i`**
   — cannot be resolved against their own thresholds in a 3.0 h spin-up, at **any** scoring
   window. Both decorrelate on the **eddy turnover** (`h/u*` = 1258–1345 m/s at these
   rungs), not on the 300 s dump interval, so the effective sample size `n_eff` **saturates
   at 3–5** from a 1.0 h window to a 2.5 h one. **Dumping more often cannot help: it is the
   RUN LENGTH that is short, not the sampling.**

   Every seed is therefore expected to return **INDETERMINATE** on those two — not PASS and
   not FAIL. `bin/seed_stationarity.py` reports each trend's AR(1)-corrected SE and `n_eff`
   and refuses a verdict when the threshold sits within 3 SE of the measurement;
   `bin/run_corpus_case.sh` runs with `ALLOW_INDETERMINATE=1` as its **default operating
   mode**; and `seed.gate_state = INDETERMINATE` is stamped onto **every pair**, with a
   warning in the training record. No threshold is loosened by any of this, and a seed with
   a **DRIFTING** limit is still refused outright — a stronger and different statement that
   no flag admits.

   **The first corpus pair, `case_2023031014`, carries this state.** Its seed
   `seed_nbl-shallow_a000` is INDETERMINATE on `TKE_BL/u*^2` and `z_i`; it was accepted
   before the gate could say so. What IS established for it, at 3.6–8.7 SE of margin, is
   the four limits the inertial oscillation cancels in — `U/u*`, `sigma_v/u*`,
   `sigma_w/u*` — and the two Kljun geometry terms that inherit their immunity.

0b. **THE `cbl-deep` RUNG IS LOCKED IN AND UNUSABLE AT THIS BOX SIZE.** Measured
   2026-08-27 (`SEED_CBL_DEEP_RESULT.md`): the mid-depth `w` spectrum pins to `lambda = L`
   with **53.9-72.0%** of the variance in mode 1, and the depth overshoots to 987-1276 m
   against a 976 m limit. The corpus therefore has **no deep convective rung**, and the
   exclusion is biased in the same direction as every other depth exclusion here — deep
   CBLs carry 1.5-3.6x the heat flux of the retained hours.

1. **The receptor may be inside the roughness sublayer over the array.** MOST does not hold
   there, so Kljun is not a reference over the array and the `sigma_w` floor is extrapolated.
2. **The first model level is at 1.997 m, at or below panel top.** The array's surface
   exchange is parameterised, not resolved. Dominant known modelling uncertainty.
3. **Deep convective boundary layers are constrained.** `L >= 4 z_i` caps `z_i` at 488 m and
   covers 19.3% of convective midday; the cap is biased against strong convection. Phase E
   measures whether it is binding for a 10 m footprint.
4. **The lake is outside the domain** — 0.05% of cells, 0.01% of any footprint. Measured.
5. **The near field is closure-dominated** at `z/Delta = 0.99`, worse than 1.76 at 24 m,
   because the eddy scale shrinks with height and `Delta` does not.
6. **Real terrain extends to ~784 m** from the tower after the taper ring; land cover is real
   to the seam.
7. **Tree cells have `ln(z_first/z0) = 0.69`** — 23.5% of the box, where the surface-layer
   scheme has almost no room. Inherent to `dx = 16 m` with a 10 m receptor.

---

## Working agreement

- Report the gate result explicitly before moving on. Commit at every passed gate.
- Prefer reading FastEddy's own source over inferring behaviour from documentation. Every
  capability claim in PROJECT_BRIEF.md carries a file and line number; keep it that way.
- Every script greps for `CORRUPTED` and tests `np.isfinite(...).all()` FIRST. `inf` is not
  NaN, and a NaN passes every `>` comparison.
- Stop early only if a gate fails twice, a fix needs FastEddy source changes beyond the
  existing fork, a run projects far past its budget, or a result would
  change the grid decision.
