# The stable regime at `dx = 16 m` — measured, 2026-08-25

> **Result.** A stable boundary layer at the canonical benchmark regime **cannot be
> sustained on this grid**, and the reason is resolution rather than forcing, numerics or
> a bug. At the *healthy* dump — an hour before anything looked wrong — the Ozmidov scale
> at the receptor is **3.6 Δ** against **318 Δ** neutrally, and **0.2%** of `sigma_w^2` is
> resolved against 2.7% neutral and 12.1% convective. GABLS1, the standard stable LES
> benchmark, runs that regime at **`dx = 6.25 m`**: 2.6× finer, **17× the cells** for this
> domain.
>
> **Every standing check passed while it died.** Finiteness, a clean `CORRUPTED` grep, and
> `k0/k1` at **0.442** — a comfortable pass. `k0/k1` is a `dt` check, not a physics check.
>
> **Weakening the stratification was tried and did not work.** A second seed at half the
> stability (`z/L = 0.044`, the *median* stable hour at this site) collapsed on the same
> timeline, with every decoupling signature absent — `Ri_g` 0.043 against a critical 0.25,
> normal Ekman backing, an ordinary 12 K/km inversion. The surface layer stayed healthy
> while the resolved energy drained upward, 92 m → 1825 m.
>
> **So stable is excluded from the corpus.** The cost is smaller than it looks in cases and
> larger than it looks in coverage: day coverage falls only **80.4% → 75.0%** (1370 cases
> from 1826 days), because enumeration finds a usable hour on almost every day — but the
> emulator is **undefined in stable conditions**, which is 44% of the site's QC'd hours.

This is a methods-section finding, not a maintenance note. It belongs wherever the corpus
is described, alongside the roughness-sublayer caveat and the retired sub-grid gate. The
short form lives in `PROJECT_BRIEF.md`; the operational fix lives in `FASTEDDY_TRAPS.md` §15 and
§16; this is the evidence.

---

## 1. What was run, and what happened

**Two collapses, from two different causes.** They are easy to conflate and they are not
the same thing — the first was fixed, and the second happened anyway.

### 1a. The cold start (fixed — `FASTEDDY_TRAPS.md` §15)

`sbl` at `G = 6 m/s`, `w'th' = -0.020 K m/s`, cold-started, 1.25 simulated hours:

| | t = 0.25 h | t = 1.25 h |
|---|---|---|
| `u*` | 0.219 m/s | **0.043 m/s** |
| `z_i` | 209 m | 61 m |
| `z/L` at the 10 m receptor | ~0.3 | **+34.8** |
| `dtheta/dz` at the first level | — | **2551 K/km** |

Above 66 m the mean wind sat at **exactly** the geostrophic 6.000 m/s — no Ekman turning
at all — with `Ri_g ~ 1e8`. The boundary layer had decoupled from the flow.

**Cause: runaway surface cooling under a prescribed-flux boundary condition.** At `t = 0`
there is no turbulence, so the cooling builds a near-discontinuous inversion at the first
model level before any can develop, and that stratification then prevents it developing.
GABLS1 prescribes a cooling *rate* for exactly this reason.

**Fix: a neutral warm-up segment** (`warmup_segments` in `bin/make_seed_jobs.py`), so real
turbulence exists before the cooling is switched on. That is also how a stable boundary
layer forms in nature — out of the evening transition — so the seed became more physical
rather than less. It worked: the next run developed a proper Ekman layer.

### 1b. The warm-started run, at GABLS1's own regime (NOT fixed)

`sbl` respec'd to `G = 8 m/s`, `w'th' = -0.012 K m/s` — which *is* GABLS1 (Beare et al.
2006: `G = 8 m/s`, cooling 0.25 K/h ≈ `-0.012 K m/s`) — with one neutral warm-up segment,
3.0 simulated hours, base angle 30°.

**It was healthy for 1.75 simulated hours and then collapsed.**

| | healthy, t = 1.50 h | final, t = 3.00 h |
|---|---|---|
| `u*` | 0.2361 m/s | **0.0984** |
| `z/L` at the receptor | +0.123 | **+2.67** peak |
| `Ri_g` through the layer | 0.03–0.05 | — |
| receptor wind direction | 228.8° (**11.2° Ekman backing** from a 240° forcing) | 200.2° (39.8°) |
| resolved TKE, peak of profile | 5.97e-1 m²/s² | **4.56e-2** (8% of its own peak) |

All seven stationarity limits fail. **Binds last: Kljun `x_peak` at 6989% of its limit**
(+69.89 %/h). `u*` trend **−75.5 %/h**.

**The warm-up is not the fault this time.** The layer *was* turbulent — 11.2° of Ekman
backing and `Ri_g` of 0.03–0.05 are a coupled, sheared, turbulent boundary layer, not a
decoupled one. It died later, and for a different reason.

---

## 2. The cause is resolution, and it is measured at the HEALTHY dump

This is the part that matters, because it is diagnosable **before** the GPU time is spent.

The **Ozmidov scale** is the largest eddy stratification permits to overturn:

```
    L_O = sqrt(eps / N^3)
```

Above `L_O`, buoyancy wins and motion is wave-like rather than turbulent. An LES can only
represent turbulence in the band `Delta < l < L_O`. **`L_O/Delta ~ 1` means there is no
band at all**: the model is running a sub-grid closure and calling it a boundary layer.

`eps` and the mixing length are **FastEddy's own**, imported from `lpdm/fields.py` rather
than re-chosen for the diagnostic (`l = min(0.76 sqrt(e)/N, Delta)`, `eps = 0.93 e^(3/2)/l`),
so the number is comparable with what the LPDM is actually driven by.

Measured with `bin/ozmidov.py`, `Delta = 10.09 m` at the receptor
(`results/ozmidov_regimes.txt`):

| regime | run, time | `L_O/Delta` at the 10 m receptor | `L_O/Delta` surface layer (min–median) | resolved `sigma_w^2` at receptor |
|---|---|---|---|---|
| **stable** (GABLS1 regime) | `seed_sbl_a030`, **healthy** t = 0.92 h | **3.57** | 2.41 – 5.21 | **0.2%** |
| neutral | `g16_spin`, t = 6.26 h | **318.07** | 43.2 – 92.7 | 2.7% |
| convective | `g16_cbl_shallow`, t = 1.79 h | *unstratified* — no constraint | — | 12.1% |

**A factor of 89 between stable and neutral at the same receptor, on the same grid.** The
verdict does not turn on where a threshold is drawn.

Two things follow that are worth stating separately.

**The column median is not the answer, and quoting it would have hidden this.** `L_O/Delta`
rises steeply with height as `N` falls: over the whole stratified column to 200 m the
median is **8.97**, which reads far healthier than the 2.4–4.0 the surface layer actually
has. The receptor is at 10 m and the footprint is made in the lowest few tens of metres.
`bin/ozmidov.py` therefore scores at the receptor and prints the column median labelled
*reported, not scored*.

**Convection has no Ozmidov constraint at all.** Below 50 m the convective case has
`N^2 <= 0`: buoyancy *produces* the largest eddies there rather than limiting them. That
is precisely why convective cases are the easy case for this grid and stable ones are the
hard case, and it is why a gate passed convectively says nothing about stable — the
standing rule in `PROJECT_BRIEF.md` about inferring one regime from another applies here in its
sharpest form.

---

## 3. Why every standing check passed while it died

| check | reading at the collapse | why it missed |
|---|---|---|
| exit status | 0 | FastEddy exits 0 on fully-NaN fields; never trusted alone |
| `CORRUPTED` grep | clean | fields stayed finite; nothing was numerically wrong |
| `np.isfinite(...).all()` | true | same |
| **`k0/k1`** | **0.442 — a comfortable pass** | **it is a RATIO BETWEEN TWO LEVELS and both went quiet together** |
| column-integrated TKE | *rising* | that integral is gravity-wave variance aloft, which grows as turbulence dies (`FASTEDDY_TRAPS.md` §15) |
| `z_i` as 5% of peak TKE | falling | a diagnostic artifact in its own right (`FASTEDDY_TRAPS.md` §16) |

**`k0/k1` is a `dt` check, not a physics check**, and nothing in this project asked whether
a boundary layer still existed. `docker/turb_alive.py` now does, and it runs wherever
`k0/k1` runs — through `docker/check_run.sh`, which `docker/run_case.sh` routes every
FastEddy invocation through, so the coverage is total.

**And the obvious metric for it was also wrong, which is the same lesson one level up.**
The natural choice is resolved TKE in surface-layer units, `e_res/u*^2` — near 5 neutrally,
regime- and wind-independent. On this run's own series it reads **11.72 at the healthy peak
and 4.71 after the collapse**: squarely inside the healthy band, and *higher* than the
healthy neutral run's 3.93. It cannot see the death because `u*` dies with the turbulence
and the ratio is preserved. The check scales against the **forcing** instead —
`max_k e_res / U_ref^2`, `U_ref` being the geostrophic wind, which cannot collapse — where
healthy is `4.6–9.1e-3` and dead is `6.3–7.1e-4`, a factor of 6.5 with nothing between.

---

## 4. What it would cost to run stable properly

GABLS1 uses **`dx = 6.25 m`**. Over the same 1952 m box that is **2.6× finer and ~17× the
cells**, at roughly 17× the GPU time per simulated hour — the corpus is ~1470 cases at
~1.1 sim-h each, so this is not a decision about one run.

The framing that actually generalises is **depth relative to the filter width**, not the
spacing itself:

| | `z_i` | `Delta` | `z_i/Delta` |
|---|---|---|---|
| GABLS1 | ~180 m | 6.25 m | **28.8** |
| the collapsed rung | 150 m | 10.09 m | **14.9** |
| `sbl-weak` (this decision) | ~280 m | 10.09 m | **27.7** |

A **deeper, weakly stratified** layer is resolved at `dx = 16 m` in the same relative sense
GABLS1 is at 6.25 m. That is the whole basis for the restriction below, and it is why the
knob chosen was **more wind, not less cooling**: `z/L` falls as `u*^-3` while `eps` rises as
`u*^3`, so more wind buys weaker stratification *and* a larger Ozmidov scale at once, and
it deepens the layer as well (`z_i ~ u*^2`). Weakening the flux buys only the first.

---

## 5. How much of this site is weakly stable

Measured over three independent sources with three different determinations of `u*`
(`bin/stable_fraction.py`, `results/stable_fraction.txt`). Each has a different failure
mode, so they are reported side by side rather than averaged.

| source | median stable `z/L` | runnable (`z/L <= 0.10`) share of stable hours | cost as a share of the whole record |
|---|---|---|---|
| the tower's own `H` + `sigma_w`, 1 y of half-hours | 0.056 | **65.8%** | 15.2% |
| HRRR, the corpus's own forcing source | 0.063 | **64.9%** | 15.4% |
| CONUS404, 45 y, `u*` carried directly | 0.071 | **60.7%** | 14.0% |

All at the standing `u* >= 0.15` QC. **Every figure is also reported without it**, because
that QC cuts exactly the hours in question — stability suppresses `u*`, so a QC on `u*`
preferentially deletes strongly stable hours and inflates the weakly-stable share. Without
it the runnable share falls to 45.2 / 43.8 / 27.6%.

So the exclusion is a **bounded limitation, not a lost regime**: about two thirds of the
site's stable hours survive it.

---

## 6. The decision, and the test of it

**Restrict the library and the corpus to weakly stable conditions, `z/L <= 0.10` at 10 m.**

- The `sbl` rung becomes **`sbl-weak`**: `G = 10 m/s`, `w'th' = -0.012 K m/s` (unchanged),
  `z_i` target 280 m, one neutral warm-up segment. Placed at the site's **median** stable
  hour, not at the edge of the band, so a pass licenses the band and a failure is
  unambiguous.
- `bin/select_times.py` gains `--max-zol` (default 0.10, **stable side only** — a
  convective hour is never rejected for being convective), solved through the
  stability-corrected log law rather than the neutral one, because neutral inversion
  overstates `u*` and would wave through the very hours the screen exists to reject.

### The test, and what each outcome means

One run, `seed_sbl-weak_a030` — deliberately the same base angle as the collapsed one, so
the comparison is direct. 3.0 simulated hours, four chained segments under the 1 h wall cap.

- **Holds with turbulence alive** → weakly stable enters the library, and the corpus states
  its stable coverage as **bounded at `z/L <= 0.10`** rather than complete.
- **Collapses too** → stable is excluded outright, and that is the result. No fourth respec.

It collapsed. What follows is the second branch.

**Judged on `docker/turb_alive.py`, not on the stationarity gate alone.** The two ask
different questions and the stable case is exactly where they diverge: a layer can be
drifting (gate FAIL) while perfectly turbulent, which is a usable seed run for longer, and
it can be dead while `k0/k1` passes, which is not usable at all.

### THE RESULT: it collapsed too. Stable is excluded.

`seed_sbl-weak_a030`, 3.0 simulated hours, fresh (not resumed), one neutral warm-up
segment. Artifacts in `results/retired_sbl_weak/`.

| t (h) | `u*` | `z/L` | Ekman backing | `\|U-G\|` aloft | `Ri_g` @20 m | `dθ/dz` @2 m | `zTKE95` |
|---|---|---|---|---|---|---|---|
| 0.75 (cooling starts) | 0.2794 | 0.074 | 8° | 0.0001 | −0.000 | −0.0 | **92 m** |
| 1.50 | 0.3334 | **0.044** | 7° | 0.344 | 0.012 | 7.1 | 559 m |
| 3.00 | **0.1848** | 0.253 | 21° | 0.467 | 0.043 | 12.4 | **1825 m** |

`u*` ends at **40% of its own peak** and still falling at −40 %/h; resolved TKE is at
**5%** of its peak. All seven stationarity limits fail. **Halving the stratification bought
a slightly slower death and nothing else.**

**And this is NOT the cold-start failure the warm-up was written to fix.** Every
decoupling signature is absent: `Ri_g` peaked at **0.043** against a critical 0.25, the
Ekman backing was normal and *increasing*, the inversion reached an ordinary 12 K/km rather
than the 2551 K/km of the cold-started run, and the flow aloft **departed** from geostrophic
instead of pinning to it. The surface layer was healthy the whole way down.

What moved was **where the energy sat**: the height holding 95% of the column TKE ran
**92 m → 1825 m**. The turbulence was not destroyed by stratification at the surface; it
failed to be *resolved* there, and what the TKE integral still counts is wave energy aloft.
`bin/sbl_diagnose.py` scores both signatures; the control confirms it discriminates — run
on the retired GABLS1-regime seed it reports *starved **and** decoupled* (`Ri_g` 0.198,
flow pinned to 0.002 m/s of geostrophic). The two runs sit at different points on one road.

**One alternative not excluded, stated.** A stably stratified periodic box with a 500 m
Rayleigh sponge is also a configuration in which upward-propagating wave energy can
accumulate rather than radiate away, and `zTKE95` reaching 1825 m in a 2500 m domain is
consistent with that as well. The load-bearing evidence for starvation is not the energy
aloft — it is the Ozmidov and resolved-fraction measurement **at the healthy dump**, which
is independent of anything that happens later and above.

**DECISION: stable is excluded from the corpus.** `bin/select_times.py --max-zol` defaults
to **0.0**; the `sbl` rung is deleted and the library is **5 rungs × 3 base angles = 15
seeds**.

**What it costs, and the surprise is that it is small in cases and large in coverage:**

| | with weak stable (`z/L ≤ 0.10`) | stable excluded (`z/L ≤ 0`) |
|---|---|---|
| days carrying a case | 80.4% | **75.0%** |
| cases from 1826 days | 1469 | **1370** |
| convective share of selected | 40.5% | **65.2%** |
| stable / very-stable cases | 24 / 0 | **0 / 0** |

**Only 5.4 points of day coverage**, because enumeration finds a neutral or unstable hour
on almost every day — 44% of QC'd *hours* are stable, but they are not the only hours those
days offer. **The loss is a regime, not a sample size.** The emulator will be undefined in
stable conditions and must not be extrapolated into them. 18 of 69 retained cases (26%)
still fall outside 06–18 LST, so the corpus is not purely daytime.

**Not attempted: a third respec.** Two runs at a factor of ~3 apart in `z/L` died on the
same timeline, and the mechanism measured at the healthy dump does not depend on the
forcing. A third attempt chosen to obtain a pass would be tuning, not measurement.

---

## 7. Limitations text, ready to use

> **The corpus contains no stable cases, and the emulator is undefined in stable
> conditions.** At `dx = 16 m` with a 10 m receptor the Ozmidov scale — the largest eddy
> stratification permits to overturn — is only 3.6 × the LES filter width at `z/L ~ 0.2`
> and 6.9 × at `z/L ~ 0.04`, against 318 × in a neutral layer. There is no resolved
> turbulent band at the receptor, and two independent seeds, a factor of three apart in
> stratification, both lost their turbulence within about two simulated hours of the
> cooling being applied. The failure is resolution, not forcing: in both runs the surface
> layer stayed healthy (`Ri_g` well below critical, normal Ekman turning, an ordinary
> surface inversion) while the resolved variance drained upward into internal waves. The
> standard stable benchmark, GABLS1, runs this regime at `dx = 6.25 m` — 17× the cells for
> this domain, which the corpus economics do not permit.
>
> **Stable conditions are ~44% of this site's quality-controlled hours**, so this is a
> substantial restriction on where the emulator may be applied. It is a much smaller
> restriction on the corpus itself: because a case is selected from any acceptable hour of
> each day rather than from a fixed time, day coverage falls only from 80.4% to 75.0%
> (1370 cases from 1826 days), and 26% of retained cases still fall outside 06–18 local
> time. The model must not be extrapolated into stable conditions on the strength of that
> nocturnal coverage — those hours are near-neutral or weakly unstable, not stable.
>
> This compounds with, and is separate from, the two standing near-field limitations: the
> receptor may sit inside the roughness sublayer over the array, and the resolved fraction
> of `sigma_w^2` at `z/Delta ~ 1` is small in every regime (0.2% stable, 2.7% neutral,
> 12.1% convective), which is why the sub-grid gate was retired and the closure's influence
> is quoted as a band rather than a correction.

---

## 8. Reproduce it

```bash
# the resolution diagnosis, all three regimes
./docker/pyrun.sh bin/ozmidov.py jobs/seed_sbl_a030/output/FE_SEED.225720 --top 120
./docker/pyrun.sh bin/ozmidov.py runs/g16_spin/output/FE_G16.1540000    --top 120
./docker/pyrun.sh bin/ozmidov.py runs/g16_cbl_shallow/output/FE_CBL.440000 --top 120

# what k0/k1 says about the same dumps, and what the physics check says
./docker/pyrun.sh docker/k0k1_check.py  jobs/seed_sbl_a030/output/FE_SEED.738720
./docker/pyrun.sh docker/turb_alive.py --calibrate "jobs/seed_sbl_a030/output/FE_SEED.*"

# how much of the site is weakly stable, three sources
./docker/pyrun.sh bin/stable_fraction.py

# the case-selection consequence
./docker/pyrun.sh bin/select_times.py --zi-max 976 --max-zol 0.10
```

Artifacts: `results/ozmidov_regimes.txt`, `results/stable_fraction.txt`,
`results/time_selection.txt`, `results/retired_sbl_gabls1/`.

---

## 9. What this changes about how we work

1. **`k0/k1` is a `dt` check and was being read as a health check.** Two collapsed runs
   scored 0.442 and 0.72 on it. `docker/turb_alive.py` now runs beside it everywhere,
   through `docker/check_run.sh`.
2. **A ratio between two quantities that die together cannot detect the death.** That is
   true of `k0/k1` (two levels) and of `e_res/u*^2` (resolved TKE and `u*`), and it is why
   the alive check scales against the forcing instead.
3. **Score a profile where the answer is needed, not where it averages well.** The column
   median of `L_O/Delta` reads 8.97; the receptor reads 3.57.
4. **Diagnose the grid at the HEALTHY dump.** Every number in §2 was available an hour
   before the collapse, at zero extra cost.
5. **A regime a gate has never run in is unknown, not fine.** Convection has no Ozmidov
   constraint at all; a convective pass is silent about stable by construction.
