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
> **The consequence is bounded, and the bound is measured.** 61–66% of this site's QC'd
> stable hours sit at `z/L <= 0.10`, weak enough to carry, so the corpus keeps weakly
> stable cases and excludes the rest. That costs **14–15% of the whole QC'd record**.

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

