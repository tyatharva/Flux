# Flux Footprint Emulator — Kegonsa Solar Array

Train an emulator that predicts 2-D flux footprints for the UW-Madison Kegonsa Solar Array
eddy-covariance tower from **only the scalar inputs Kljun et al. (2015) uses**, and beats
Kljun at that site.

**Scope is deliberately narrow: a site-calibrated emulator for ONE tower.** Zero transfer to
other sites, and that is an accepted, stated limitation. Do not add scope.

**Architecture is an FNO, not a CNF** (changed 2026-08-30, see `docs/ML_TARGETS.md`). The target is
the 122² raster zero-padded to 128², the model predicts a residual on Kljun conditioned on the
six scalars by FiLM, and touchdowns are not saved.

LES (FastEddy) and the backward LPDM are **offline target generators**. They are never part of
inference.

---

## STATUS — 2026-09-01: THE CORPUS EXISTS. THE NEXT STEP IS ML.

**1366 training pairs in `corpus/corpus.h5`** (46 MB), generated on 8 machines x 8 RTX 5090.
`corpus/README.md` is the dataset's own documentation; read it before training.

| | |
|---|---|
| records | **1366** — train 837 / val 235 / test 294 |
| arrays | `scalars` (N,6), `kljun` (N,128,128), `target` (N,128,128), all float32 |
| normalisation | in `norm/`, computed from the **train split alone** |
| index / flags | `corpus/INDEX.json`, `corpus/FLAGGED.tsv` |
| provenance | `corpus/provenance/` — 8 machine manifests, every one of 1945 days accounted for |
| image that made it | `ghcr.io/tyatharva/flux-seeds:7de9dee2a01d-fe0ce48d5dff06` (30-seed library baked in) |

**READ THIS BEFORE TRAINING: only ~15% of the corpus carries the site-specific signal.** The
array is in the footprint essentially only for northerly flow, and convective afternoons here
favour SW/W, so the corpus rose is skewed away from the signal:

| dir | corpus | site rose | mean array share |
|---|---|---|---|
| **N** | 6.9% | 10.6% | **30.3%** |
| NW | 14.6% | 14.5% | 6.9% |
| NE | 6.4% | 10.2% | 2.7% |
| W | 21.4% | 14.4% | 0.3% |
| SW | 19.4% | 14.3% | 0.9% |
| E | 4.5% | 10.4% | 0.2% |

202 of 1366 records (14.8%) have an array share above 5%; the median is **0.49%**. Per split
that is train 14.7% / val 17.9% / test 12.6%, so every split sees it — but **an aggregate
metric over all 1366 is dominated by cases with no array in view, where Kljun and the LES
agree by construction.** Weight the loss, or report the northerly subset separately.
(Measured N-wind array share 30.28% against the 30.7% Kljun predicts for N at `z_m = 30 m` —
independent agreement to 1.4%.)

**Corpus gaps.** 166 days failed; six months are empty — 2021-12, 2023-07, 2023-10,
**2024-01 (val)**, **2024-04 (val)**, 2026-08; 2021-06 and 2022-04 are partial. Machine 3 lost
all 8 GPUs to one fault 42% into its run (stage 7 timed out waiting for the LES to stage into
`/dev/shm`; root cause unresolved, the LES logs are gone). **Verified NOT an input-space
hole**: 84–93% of a missing month's cases fall inside the retained months' p5–p95 on every
scalar. A top-up is possible any time — the failed days are named in the manifests and the
hour draw is date-seeded, so a re-run reproduces exactly the cases that would have been there.

**Two records were rejected** (`case_2022010915`, `case_2022122416`): `h` = 2371.979 m is
`bl_depth`'s `DAMP_FRAC` search ceiling, not a measured depth. Their days are `missing` with
that reason; the `.npz` are in `corpus/provenance/rejected/`. Dropping them cut the train
split's `h` std by 5%.

**`corpus/FLAGGED.tsv` lists 231 records (16.9%)** failing G2b (integral outside [0.6, 1.5])
or G3b (peak/Kljun-peak outside [0.4, 2.5]). **Neither is an exclusion rule** — both are
per-case pipeline sanity checks calibrated on a handful of validation cases, and an LES peak
far from Kljun's may be the signal. Ablate; do not filter by default.

---

## Configuration — 122³ @ 30 m, receptor at 30 m

**The MODEL receptor is 30 m; the real instrument is at 10 m.** A deliberate methodological
choice for resolution adequacy, not a correction — at 10 m the footprint peak did not respond
to meteorology (48 m in all three targets, max/min 1.00x) because the near field was closure
output, not LES output. Say so wherever the emulator is described.

| | |
|---|---|
| grid | **122 x 122 x 122** at `dx = dy = 30 m`, domain **3660 m**. `(N+6) = 128 = 2^7` in all three. |
| vertical | `d_zeta` 24.691358, `verticalDeformFactor` 0.346601, `zCeiling` 3000 m, `dz_sfc` 8.5583 m, **k = 3 at exactly 30.000000000 m** |
| receptor | 30 m above bare ground; the array surface is raised 1.5 m (`--raise-topo`), so the aerodynamic height is **28.5 m** and that is what every record carries |
| `dt` | **0.0308642 = 5/162 s**, `CFL_3d` 1.3502 — 10.0% below the measured accuracy boundary |
| thread block | **1 x 2 x 64** |
| cost | **0.479 GPU-h/sim-h** measured; a case is ~0.36 GPU-h at 8-way |
| `Delta` / `z/Delta` | 19.78 m / 1.52 |
| geometric-mean `z0` | 0.0615 m; water 13.61%, array 0.30% (44 cells) |
| taper knee | pad 12 — real geography to 1470 m |

`bin/vgrid.py --dx 30 --nx 122 --receptor 30 --k 3 --zceiling 3000` re-derives the grid.
`runs/g30_base/base.in` is the template; `data/grid30_raised/` is the production surface.

**A case is 1.25 simulated hours and the footprint is its last 30 minutes.** 145,800 steps =
4500.000 s, verified arithmetically:

| clock | event | step |
|---|---|---|
| T − 1.25 h | restart from the seed; adjustment begins | 0 |
| T − 0.75 h | adjustment complete (`ADJ_S` 1800 s) | 58,320 |
| T − 0.50 h | first release (needs `t_back` = 900 s) | 87,480 |
| T | last release; window closes | 145,800 |

The earliest field a backward trajectory can reach is the adjustment end, **enforced twice** —
`run_window.sh` deletes the adjustment's dumps and refuses unless the earliest survivor is
step `A_NT`, and `stage5_footprint.py --t-min` refuses independently.

`N_WINDOWS = 1`. A second window was measured and cut: on both validation cases the two
windows were near-duplicates in shape (median `|w0 − w1|` / the within-footprint floor 0.19
and 0.33, where independent draws give ~√2). `N_WINDOWS = 2` stays supported for a
spread-estimating model.

### Splits — hard-coded by calendar month, assigned at generation

`lpdm/corpus.py:SPLITS`. Whole years to val and test, so a split boundary never falls inside a
synoptic system and seasonal coverage is complete on each side.

| split | months | |
|---|---|---|
| train | 2021, 2022, 2023 + 2026-02/04/06/08 | 40 |
| val | 2024 | 12 |
| test | 2025 | 12 |

A month not named is not in the corpus and `split_of` refuses it. Disjointness asserted at
import. The split is checked against the case's own timestamp; disagreement is fatal.

### How a case is chosen

Per day, draw a round hour **without replacement** from the 24, seeded from the date alone
(so a re-run reproduces the sequence). Screens: HRRR present, `z/L < 0`, `z_i` in 300–1250 m,
`|dz_i/dt| < 15 %/h`. An hour is spent the moment it is drawn, so a day terminates at an
accepted hour or as a **MISSING DAY with a reason**. **No rose weighting and no direction
stratification** — the weather supplies the rose.

**The sounding is the HRRR analysis valid at EXACTLY T, not T−1**, because forcing is constant
through the run: the LES is initialised from the sounding and integrates 1.25 h under fixed
geostrophic wind and surface flux, so it never evolves from a T−1 state toward a T state.

**Day yield is meteorological: 78% overall but 38% in June and 57% in July** — a summer CBL is
inside the 300–1250 m band only while growing at 17–45 %/h and is past it by the time growth
falls under 15 %/h.

---

## Hardware and environment

**Two environments differing in exactly one thing: the CUDA toolkit.** Distro, MPI, NetCDF and
gcc (Ubuntu 22.04 / OpenMPI 4.1.2 / NetCDF 4.8.1 / gcc 11.4) are held **IDENTICAL** so the
toolkit is the only variable.

| | workstation | rented boxes |
|---|---|---|
| GPU | RTX 4080, Ada, `sm_89` | 8–16x RTX 5090, Blackwell, `sm_120` |
| image | `flux-fasteddy:cuda118` (`Dockerfile`) — toolchain only | `flux-seeds:<commit>` (`Dockerfile.blackwell`) — code baked in |
| CUDA / arch | **11.8**, `sm_89` cubin only | **13.0.1**, real SASS `sm_75 … sm_120`, no PTX |

- **The 11.8 pin is a FLOOR, superseded for deployment only.** Its highest target is `sm_90`,
  so it cannot reach a 5090 with SASS *or* PTX. Every published result came out of the 11.8
  image and that image is unchanged. The upgrade's effect on physics is measured at
  **0.97–1.12x the model's own run-to-run floor** (`bin/test_toolkit_parity.py`).
- **`nvcc -dlink` silently drops PTX**, so there is no JIT fallback to have — hence real SASS
  for seven architectures. `--list-elf` and `--list-ptx` are different questions; assert both.
- FastEddy **v5.0.1**, fork branch `kegonsa`, **fp32** (confirmed in source; never
  re-litigate). **Not bitwise reproducible** — two runs differ ~1e-4 relative in velocity after
  200 steps, and any "did my change matter?" test compares against that floor, not zero.
- **Restart is a true bit-for-bit state resume.** Requires netCDF; `ioOutputMode = 1` is not.
- **Docker only.** The host python has no scipy or h5py; every analysis script runs in-image.

---

## THE STANDING RULES

These are the expensive lessons. Each has cost real GPU time at least once.

### 1. Validate the state the model actually LOADED, never the config handed to it

**Five instances, each of which produced a plausible wrong number rather than an error**, and
each found by looking at an artifact rather than a setting:

| configured | what the model actually had | found by |
|---|---|---|
| a per-cell convective `htFlux` map | **all zeros** — the case would have run neutral silently | reading the field out of the file |
| receptor at 10 m | every footprint landed on the level nearest the **30 m** default | reading the call, not the flag |
| `dt` inside the stability limit | inside the **accuracy-vs-stability window**: exits 0, prints nothing, near-surface `w` is acoustic noise | `k0/k1` on the dump |
| `surflayer_wth = −0.012` in the `.in` | **`+0.000000`** in every dump | reading `htFlux` out of the dump |
| a `.in` template in the image | **absent** — `.dockerignore` took it; 81 cases, 0 records, on all 8 machines | reproducing the build locally |

**Every parameter that is ALSO an IO-registered field is a property of the restart FILE, not
of the `.in`** — `htFlux`, `z0m`, `z0t`, `tskin`, `topoPos`, `zPos`, `xPos`, `yPos`. For those
the `.in` is a request and the restart is the answer. The rule is wider than that list: a
default silently taken, a parameter silently reverted for being out of range, and a `dt`
inside the accuracy window are all the same shape.

**So every step asserts on the artifact it produced, and on the QUANTITY, not the presence of
a file.**

### 2. A check that stubs the thing it is checking is a statement about the harness

The 8-machine dry run was green while the image had no `.in` template, because `--stub`
replaces the screener and the case and **opens no file the case path reads**. Every seed-side
artifact was asserted at build; no case-side one was. Fixes: the build asserts the case path's
inputs *and* that the template is `Nz = 122`; `run_corpus` refuses at startup by name; and
`docs/DEPLOY.md` §C2 runs **one real case with only the LES stubbed** in ~4 min with no GPU. That
last one is what actually closes the gap.

### 3. A diagnostic is only as scale-free as its reference

> A diagnostic whose DENOMINATOR or REFERENCE varies with anything but the quantity being
> measured will report that variation as signal.

It fails quietly every time — the number stays finite, the check runs, the verdict prints.
**Four instances, and the fourth is the FIX that was applied to the second:**

| diagnostic | the reference that moved | what it reported instead |
|---|---|---|
| `z_i` at 5% of the *running* TKE peak | the peak falls with `u*²` on the inertial oscillation | **+11.67 %/h** of deepening while three independent depths said +1.71 to +2.33 |
| `TKE/u*²` over the *whole column* | the column is mostly free atmosphere, so it scales with `z_i/H` | two rungs **44%** apart in a quantity that is **5.7%** apart when scale-free — and it FAILED a seed |
| `k0/k1`, first-to-second level `w` variance | both levels collapse together when the layer dies | **0.442, a clean pass**, through a stable seed whose boundary layer had died |
| `TKE_BL/u*²` — *the fix applied to row 2* | `u*²` falls ~10 %/h on the oscillation, and the averaging depth `z_i` entrains upward | **+22.5 %/h "drifting"** on a rung whose absolute BL TKE is **flat**, and the WRONG SIGN on another |

**The fix is one of two things, never a looser threshold:** make the reference scale-free
(`z_i` moved to a fixed 0.01 m²/s² threshold), or pair the check with one that fails
differently (`docker/turb_alive.py` runs everywhere `k0/k1` runs and answers *is there any
turbulence at all?* — a SKIP from it is not a PASS).

**And the diagnostic for the diagnostic is its own sampling error.** `bin/seed_stationarity.py`
reports each trend's AR(1)-corrected SE and `n_eff` and returns **INDETERMINATE** rather than
PASS or FAIL when the threshold sits inside that spread.

### 4. A tolerance must be the size of the failure it is looking for

Not the smallest number you can write down. `--strict-rel` exists to catch losing a **5 s**
dump and scored against `1e-6` s, so it failed a correct production run on half a
millisecond, at stage 7, after 74 minutes of GPU. One lost dump exceeds that deficit by a
factor of **10,016**. Tolerances are now expressed in the unit of the thing they protect
(dumps, output intervals), and **the margin is printed on SUCCESS too** — a configuration
designed to sit at zero margin leaves no evidence of how close it came unless the passing
path says so.

The same rule with the sign flipped: a tolerance must also **reject nonsense**, not just
excess. `t_start_s` and `t_end_s` rounded to different precisions gave sub-100 ms days a
NEGATIVE duration, and the load-balance check passed on "−73% imbalance" and "108% of wall
time saved".

### 5. A tolerance measured from ONE difference is not a tolerance

It has one degree of freedom and its own sampling error is the size of the thing it bounds.
Phase E said DIFFERS on a 2-group floor and PASSED at p ≈ 0.54 on a 10-group one — a factor
of **5** in the estimated floor from nothing but the number of groups. Use `--cover-groups N`
(N ≥ 8), quote the standard error, and never quote a tolerance without saying how many
independent realisations went into it.

**And score a second moment against its own sampling spread, not a number you picked.** The
convective B6 gate first used a fixed 3e-2 on `sigma_w²` and reported DIFFERS at 3.587e-2.
Re-scored against the block standard error of the same field it is **0.38x** one
realisation's own spread. The reframing is not a loosening; it is what located the problem.

### 6. Validation must exercise the production code path AND the production regime

- **Wrong regime.** The neutral well-mixed gate passed a closure carrying **nine** turnovers
  in `sigma_w²`, because the floor is nearly inert neutrally (receptor factor 1.000 there,
  1.59 convectively). Treat a regime where a component is inert as *no evidence at all* about
  that component.
- **Wrong code path.** `stage4_wellmixed.py` carried its own COPY of the `sigma_w` floor,
  which had drifted from the production one. Gates import the production function; they never
  reimplement it.
- **Quote the no-op control alongside the result.** The convective failure was localised by
  scoring the same window with no floor at all. A gate result without its control says only
  "a number came out".

### 7. Assert on the artifact, not on the exit status

Analyses get piped into `grep`, so bash reports GREP's status and a python traceback lands
quietly in a redirected `.txt` — a `SyntaxError` in `stage5_footprint.py` was launched six
times that way. Every step checks the JSON it was supposed to write, and `bin/preflight.sh`
parses every python entry point and shell driver before a campaign starts (~10 s; the drivers
refuse without it).

Same rule for exit codes: `jobs/run_seed.sh` was `[ "$VERDICT" = "PASS" ] || exit 1` and
returned **1 for all thirty** seeds. A status identical for every outcome discriminates
nothing, in the dangerous direction. **The verdict lives in the artifact.**

### 8. One run per directory, or it is not a series

FastEddy names a dump `<outFileBase>.<step>`, so a directory that has held two runs holds two
families with OVERLAPPING step numbers — sorting the union on the step interleaves them into
a "history" with two different states at the same time. Every glob of a dump directory filters
on one base name.

### 9. Every script greps for `CORRUPTED` and tests `np.isfinite(...).all()` FIRST

`inf` is not NaN, and a NaN passes every `>` comparison.

---

## Settled by measurement — do not re-derive

### `dt` is set by the acoustic CFL, and the ACCURACY limit is below the stability limit

FastEddy is fully compressible with RK3, **no acoustic sub-stepping and no CFL machinery at
all** — `dt` is a mandatory user constant, never computed or checked. Tutorial values are
hand-picked and mutually inconsistent; never copy them.

`CFL_3d = c·dt·sqrt(2/dx² + 1/dz_sfc²)`, `c = 347.2 m/s`.

| | CFL_3d | behaviour |
|---|---|---|
| stability limit | ~1.79 | NaN, `CORRUPTED` |
| **accuracy limit** | **grid-dependent** | above it: **silent** grid-scale acoustic noise |
| production, 122³ @ 30 m | **1.3502** | 10% margin |

**The accuracy boundary is a property of the grid, must be RE-MEASURED on every one, and does
NOT interpolate with anisotropy:** 122³ @ 16 m (`dx/dz` 4.007) → ~1.51; @ 24 m (2.804) →
1.55–1.60; **@ 30 m (3.505) → 1.50–1.55**. This grid's anisotropy sits *between* the other two
and its boundary at the *bottom* of their range. The transition is sharp — `k0/k1` is **0.130
at CFL 1.50** and **8.857 at 1.55**, a factor of **68 across 0.05 of CFL** — and `turb_alive`
reads OK at every rung, so `k0/k1` is the only check that sees it.

**Verify every run with `docker/diag_near_surface.py`: first-level `w` variance ratio `k0/k1`
must be `< 1`** (~0.27 when correct). Near 9 means `dt` is too large. **`k0k1_check.py` is a
DOMAIN MEAN and is structurally blind to terrain-driven local noise** — `bin/k0k1_by_slope.py`
conditions on slope and is the terrain-aware form.

**Terrain amplifies the effective CFL** as `CFL_3d·sqrt(1 + (slope·dx/dz)²)`, but measured at
122³ @ 16 m it did **not** lower the boundary. Re-measure; never carry the number. Vertical
stretching is not a speed lever — with `dx` fixed, even an infinitely coarse vertical relaxes
the 3-D CFL by at most `sqrt(3/2)`.

### Restart overwrites grid and surface fields — the trap and the lever

`hydro_coreInit()` runs before the restart read, which walks the entire registered variable
list — including `xPos`, `yPos`, `zPos`, `topoPos`, `z0m`.

- **Trap.** Restarting a FLAT spin-up with a `topoFile` set leaves correct terrain-following
  metrics but silently overwrites the *diagnostic* `zPos`/`topoPos` with flat values. The LES
  is right and the output coordinates are wrong, so the LPDM places every particle at the
  wrong height with nothing to indicate it.
- **Lever.** The same mechanism is the ONLY way to give FastEddy v5.0.1 spatially varying
  roughness or heat flux. `z0m` is a 2-D field with no input path; writing it into the restart
  file works with no source change. `bin/prep_stage6.py` does this.

### The footprint estimator

- **The raster IS the LES grid.** Touchdowns are binned by LES column index, folded modulo the
  periodic domain. Cloud-in-cell deposition — exactly conservative, 0.67x the per-cell noise of
  nearest-grid-point.
- **Negative values are physical and nothing clips them.** Signed by construction; the negative
  lobe carries **5.8–11.1%** of `|flux|`. The `np.maximum(f, 0)` calls are metric-side.
- **The integral asymptotes to `1 − z_m/z_i`, not to 1** (Steinfeld 2008): the fraction
  `z_m/z_i` of the column lies below the receptor and its flux never crosses it. At 30 m that
  is **3.75%** — the size of effects this project gates on. **Departure from the asymptote
  tracks `w_bar` at the receptor with the right sign** (subsidence → 1.497x, updraft → 0.916x):
  the advection non-closure, measured on two cases with opposite signs.
- **Periodic wrap double-counts. Cap trajectory displacement at one streamwise domain length.**
  And **an integral that crosses 1 and keeps climbing cannot be truncation** — a finite backward
  time can only lose influence — so it is always a model inconsistency.
- **Rescaling sub-grid variance breaks well-mixedness unless the drift is rescaled with it.**
  Thomson's reverse-time drift contains `d(sigma²)/dz`; with a height-dependent `sc(z)` it is
  `sc·dsig2dz + (2/3)·e·dsc/dz`, and the second term is the larger.
- **A clip on one side of a ratio is a bug on the other.** `sigma²` was clipped at 1e-6 and the
  denominator of `eps ∝ sigma²` was not, so above the boundary layer `eps` was inflated a
  millionfold and every particle pinned at `dt_min`. **The only symptom was a 4x slowdown.**
- **The touchdown weight uses the surface-normal approach rate** `|d(z−z_ground)/dt|`, not
  `|w|`; over sloping ground the `2/|w|` weight explodes. Flat ground hides this completely.
- **The LPDM is forked into 16 fixed chunks with per-chunk seeds**, so worker count is a pure
  performance knob and cannot change a result (asserted bit-identical, 1 worker vs 12). Workers
  are FORKED so the field cache is shared copy-on-write — do not switch to spawn.

### The `sigma_w` closure

The floor is **weighted by the sub-grid fraction**, `sc_eff = 1 + (sc − 1)·f_sgs` with
`f_sgs = (2/3)e / (ww + (2/3)e)`, and **`eps` is scaled with `sigma²`** so
`T_L = 2σ²/(C0·eps)` is preserved. Gate D1 passes in both regimes with **0 turnovers**.

**The cause of the old failure was the MAGNITUDE of the inflation, not its shape** — a
constant x10 with no taper failed forward at 1.370 while a constant x1.673 passed at 1.130.
Two earlier diagnoses were both wrong and each cost a rebuild. (`docs/results/SIXTH_PASS_RESULTS.md`.)

**The near field is closure-dominated and that is a number:** the floor is worth **+8.40
points** of convective array share and shortens `x80` from 400 to 227 m; the retired closure
inflated that share by up to **+18.46 points**.

**Quote the compaction ratio with its closure.** Floor OFF gives 0.57x (convective BROADER),
floor ON 1.33x (convective more compact) — and **both PASS Gate D1**, because well-mixedness
tests SELF-CONSISTENCY, not whether `sigma_w` has the right magnitude. **Settle regime
comparisons on AREA, not `x80`.**

**The sub-grid fraction is reported, never gated** — 52.5% convective, 86.4% neutral here;
reaching 40% at `z = 10 m` would need `dx ~ 3–4 m`, ~22x this configuration.

### `bl_depth` measures the SURFACE-ATTACHED layer

`lpdm/les_stats.py:surface_layer_top` bounds the estimate at the minimum terminating the
surface-attached layer. **The test is "a second layer that out-energises the first", not "the
first local minimum"** — the first strict local minimum above the surface peak is usually
noise, and using it moved `h` on 15 of 47 stored profiles by up to 331 m on profiles with no
wave layer at all. What identifies a wave layer is that it carries more resolved TKE than the
boundary layer under it, so the column's global maximum lands in it.

`bin/test_bl_depth.py` re-derives `h` for all 47 stored profiles and requires **EXACT**
equality — no physics between the two, only arithmetic. **47 of 47.**

### Kljun is Natascha Kljun's own code

`third_party/FFP/calc_footprint_FFP.py` is the official v1.42, vendored unmodified.
`lpdm/kljun_ffp.py` re-evaluates its two separable factors at our north-up cell centres and
**reimplements no formula**; it agrees with the code it wraps to **9.4e-16**.

**Our own reimplementation was 1.25x wide in `sigma_y` whenever `|L| > 5000`** — the official
resets `ol = -1e6` above `oln = 5000` and clips `scale_const` to 1.0, while `lpdm/kljun.py`
short-circuits at `|L| > 1e5` and never reaches the clip. That is exactly the near-neutral
regime, which is the one place Kljun is diagnostic rather than descriptive.

### Neutral stationarity is about `u(z_m)/u*`, not about `u*`

A doubly-periodic neutral Ekman layer forced by a constant geostrophic wind does not settle to
a fixed `u*` on any affordable timescale — `f = 9.94e-5`, so the **inertial period is 17.6 h**.
Over five windows `u*` moves **18%** while `U/u*` moves **0.6%**. Gate on the ratio: both terms
ride the oscillation together, and Kljun's `x_peak`/`x90` inherit the immunity.

**`z_i` is the one exception** — a length, with no `u*` to cancel, so it can only be made immune
by how it is MEASURED. The gated definition is a **FIXED 0.01 m²/s²** threshold; `window_stats`
still produces the corpus input `h` as a peak fraction, and the two differ by 7–21%. **The gate
measures a TREND and needs a threshold that does not move; the matcher compares a VALUE and
needs the definition the corpus inputs use.** And **a linear trend through a staircase reports
the staircase** — `z_i` only lands on model levels, so the gate prints the distinct-level count
and span beside the trend.

### The LES hands fields to the LPDM in RAM

`SRC/IO/io_lpdmonline.c` on the fork, behind `lpdmOnlineSelector` (default 0, so every
existing `.in` is bit-identical). **1 = stage only** (production); **2 = stage AND write
netCDF** (acceptance — one run producing both paths from the same bytes). Per case this is
**3.6 MB persisted against 19 GB**, and the two paths agree to **0.00e+00** — bit-identity is
asserted, not a tolerance, because there is no physics between them. `docs/results/NINTH_PASS_RESULTS.md`.

**What streaming cannot reach.** The 12.0 GB field cache is not buildup — it IS the window,
and `compute_footprint` is a CPU integrator that random-accesses all of it, so host residency
floors at the cache. **"The ring is in VRAM" is aspirational: today it is a HOST fp16 cache
and the GPU LPDM is not on the production path.** The ring holds a full window rather than
`t_back` because the `sigma_w` floor is built from whole-window statistics — a shorter ring
would force it onto partial ones, an estimator change wearing a plumbing change's clothes.

### The seed library

**30 seeds, 2.0 sim-h each**, in `jobs30/seed_*/return/`, baked into the deployment image.
Spun in 0.936 h wall / 13.24 GPU-h on 16x RTX 5090, all 30 complete, finite and passing the
full acceptance battery. **0.189 GPU-h/sim-h at 16-way against 0.469 single-GPU in the same
image — contention costs nothing, it is 2.5x faster.** Full evidence
`docs/results/SEED_LIBRARY_RESULT.md`.

**Do NOT carry 0.189 to a corpus estimate.** A seed runs FastEddy and nothing else; a case
also runs the LPDM and the ring.

**Seed selection uses the WHOLE library; `ALLOW_DRIFTING` defaults to `any`.** A seed is an
INITIAL CONDITION, not a corpus point — the case restarts from it, adjusts under its OWN
sounding's forcing, and every ML input comes from `window_stats` over the footprint's own
window, so the pair is self-consistent whatever the seed's drift state. Refusing a seed
removes a RESTART POINT without removing any error. `gate_state` is stamped on every pair.
**"11 of 30 accepted" was never a quality statement** — the strict gate returns INDETERMINATE
when the threshold sits within 3 SE, which at these magnitudes admits the *worse-measured*
seed.

**Owed:** seed runs should return the SCORED SERIES, not only the verdicts fitted to it.


---

## Site

- UW-Madison Kegonsa Solar Array, southern Wisconsin.
- **Tower coordinate, SURVEYED: `42.957160, -89.292362`** (EPSG:3071 577719.1, 276299.5).
  Single source of truth: `TOWER_LON/TOWER_LAT` in `bin/prep_stage6.py`.
- **EC tower measurement height ~10 m AGL.** The model receptor is 30 m — see above.
- **Solar array — THE TOWER IS INSIDE IT.** 60 m east and west, 250 m north, 100 m south:
  120 x 350 m, 4.20 ha. A rectangle in EPSG:3071; nothing about it depends on the wind.
- Land cover from **ESA WorldCover v200 (2021)**, 10 m; terrain from **USGS 3DEP** 1/3-arcsec,
  both in `data/raw/` (gitignored). Roughness per class (water 1e-4, grass 0.03, cropland 0.10,
  built 0.5, tree 1.0), then the array rectangle overrides it — **WorldCover labels the array
  as cropland**, because it does not see photovoltaics.
- **Terrain is tapered at the wrap seams; land cover is NOT.** Terrain height enters the
  coordinate transform and its metric tensor, so a seam step is a numerical cliff. Roughness
  and heat flux are local boundary conditions, where a seam is just a coastline.

### Panels, and the surface heat flux

Panels are a **bulk surface patch** — elevated `z0`, displacement height `d ≈ 1.5 m`, raised
heat flux — never explicit geometry, because row spacing is 5–7 m. Production uses
`--raise-topo`: `z0_array = 0.25` against cropland's 0.10 (2.50x). **At `z0_array = 0.10` the
array is aerodynamically IDENTICAL to the cropland it replaced and its entire neutral signal
is zero**; `prep_surface.py` warns when the two coincide.

**A stable case is a roughness-only array case.** The per-class flux table is a **daytime**
enhancement table with no nocturnal equivalent, and at night the physics inverts, so a stable
case gets a **uniform negative `htFlux`**. That makes stable the only place the array's
roughness effect is isolated from its heat-flux effect. Never claim a thermal array response
at night.

**`htFlux` is per-cell and is the VIRTUAL flux**, because the run is dry and buoyancy is what
it is for. Literature ratios are sensible-flux; the conversion is Bowen- and therefore
class-dependent, `w'θ_v' = w'θ'·(1 + 0.0735/B)`. Cropland (the reference, B = 0.4) 1.000;
**array (B = 4) 1.376** from a 1.60 sensible ratio; built 1.314; tree 1.100; grassland 1.066;
**water (B = 0.15) 0.151** from 0.12. **Working in virtual flux COMPRESSES the wet–dry
contrast** — array-to-water falls ~32% — because the wetter surface's latent flux buys
buoyancy back. That is correct, and is what running dry trades for.

**The `.in` scalar `surflayer_wth` must be the DOMAIN MEAN of that map, not the cropland
reference** — a flat spin-up has no restart injection, so the scalar IS its flux.

**Albedo has no pathway, and that is not an omission.** There is no radiation scheme;
`surflayerSelector = 1` prescribes the kinematic heat flux directly, so what albedo would
control is subsumed by `htFlux`.

### Rotation

Direction is set by **rotating the geostrophic vector**, not the map, so the surface is
bit-identical for every direction and any directional difference is flow rather than a
resampling artifact. A square periodic domain with `dx = dy` over a flat uniform surface is
exactly equivariant under 90° rotation. **Achieved direction is not forcing direction** —
Ekman turning measured off the library is 5.2° convective (n=18), 16.9° neutral (n=12). Label
cases by ACHIEVED direction.

### FastEddy capabilities confirmed in source

- **Geostrophic forcing takes a linear vertical gradient** (`z_Ug`, `z_Vg`, `Ug_grad`,
  `Vg_grad`), all `PARAM_MANDATORY` even when zero.
- **`stabilityScheme = 2` gives a 4-segment piecewise-linear base-state theta profile**
  (`zStableBottom{,2,3}`, `stableGradient{,2,3}`). `bin/sounding_to_forcing.py` fits a per-case
  HRRR profile to those six numbers — **0.04–0.27 K rms** over the LES column.
- **Subsidence needs `lsfSelector = 1` AND `lsf_horMnSubTerms = 1`.** Inputs are **per hour**.
  It is an ADVECTION tendency on U/V/THETA against the slab-mean gradient — there is no `W`
  term, so `w` never acquires it. **Fixed a real FastEddy bug on our fork:** with
  `moistureSelector = 0` it died with an illegal memory access, because the qv slab-mean and
  `Frhs_qv` are unconditional while the arrays are allocated only when moisture is on.

---

## Known limitations — state these wherever the corpus is described

1. **The model receptor is 30 m; the instrument is 10 m.** The emulator predicts a footprint
   the physical tower does not measure. Tower comparisons go through a MOST translation whose
   stable branch is an upper bound.
2. **Only ~15% of the corpus carries the array signal**, and the rose is skewed away from the
   directions that do. See STATUS.
3. **Gate A1 (water share) FAILS** — 11.58% worst case over corpus regimes against a 10%
   threshold. **It is the site, not the box:** Kljun's `x90` barely moved (1665 vs 1615 m), so
   the physical footprint is the same; what changed is that a 3660 m box HOLDS it where a
   2928 m box was replacing the lake with a periodic re-sample of its own land. **The old PASS
   was truncation.** E and NE are ~20% of the rose and carry 6–12% water; the rest ~0%.
4. **The LES loses tail that Kljun does not at 3660 m** — LES retains 0.756 of its asymptote
   against Kljun's 0.929, where at 2928 m the two were at parity (0.874 vs 0.867). **An
   LES-vs-Kljun comparison on this box is no longer the fair one that parity licensed.**
5. **There are NO stable cases and the emulator is undefined there** (~44% of QC'd hours). A
   stable seed was healthy 1.75 sim-h then collapsed, and the cause is resolution, measured at
   the healthy dump an hour before anything looked wrong: `L_O/Delta` = **3.57** at the receptor
   against 318 neutral, a factor of **89**. GABLS1 runs that regime at `dx = 6.25 m`, 17x the
   cells. Weakening the stratification was tried and failed the same way.
6. **The near field is closure-dominated** — 52.5% sub-grid convective, 86.4% neutral. Quote
   the anchor-sensitivity band (**46–66% shape L1** against a 38% sampling floor) with any
   near-field number.
7. **Every seed is DRIFTING or INDETERMINATE on at least one limit, and that is the normal
   state.** `TKE_BL/u*²` and `z_i` decorrelate on the eddy turnover, not the dump interval, so
   `n_eff` saturates at 3–5 whatever the scoring window — **dumping more often cannot help; the
   RUN is short.** The neutral rungs are genuinely short at 2.0 sim-h.
8. **The GPU LPDM is validated but is not the production integrator**, so host residency floors
   at the 12.0 GB field cache rather than at one or two snapshots.
9. **Seed grouping in the split is not settled.** `bin/seed_leakage.py` found no fingerprint at
   the un-confounded receptor — sharing a seed made two cases *less* alike (2.47 vs 1.55 floors)
   — but n = 1 same-seed pair there, so it is weak evidence, not a licence to drop grouping.

## Ruled out — do not propose these

- **STILT** — replaced by this project's own backward LPDM.
- **Mesoscale coupling** (`hydroBCs=1`, GenICBCs, cell perturbation) — the fetch requirement
  would consume most of the domain. Periodic instead.
- **LES-to-LES nesting**, **NSCBC**, **512³ domains** — schedule / unnecessary / infeasible.
- **Running FastEddy backwards in time** — mathematically impossible, not a code limitation.
  Reversing t and u flips the sign of the SGS stress term, giving negative eddy viscosity and
  the backward heat equation. Backward LPDM steps *particles* backward through *forward-stored*
  fields.
- **Multiple virtual tower locations** — would inject unexplained variance. One fixed tower.
- **Surface fields as ML inputs** — out of scope.
- **`surflayer_idealsine` / a diurnal cycle within a run** — both branches assign a SCALAR to
  `htFlux`, overwriting the per-cell map that gives the array its enhancement.
- **Moisture (`moistureSelector = 1`)** — run dry, prescribe the virtual heat flux instead.
- **Sub-grid-fraction < 40% as a gate** — retired; unreachable at any affordable grid.
- **A neutral well-mixed PASS as evidence about the convective closure** — the floor is nearly
  inert neutrally. It passed a closure carrying NINE turnovers.
- **Stable corpus cases at this grid** — see limitation 5.
- **Online footprint calculation inside FastEddy** — IO is ~3% of compute so it solves a
  problem we do not have, and it would be a worse estimator: forward tracers resolve source
  *tiles*, so the footprint's resolution becomes the number of tracers you can afford and the
  near field would be the coarsest part.
- **Fitting `stabilityScheme = 2` to a CONUS404 MEAN sounding** — `conus404_hourly` has no
  time-varying atmospheric profiles at all. (A **per-case fit to a real HRRR profile** is the
  corpus mechanism and is different.)
- **CONUS404 as a forcing.** It sets sweep ranges and is the 45-year climatology the site is
  characterised by. **HRRR forces the runs.**

---

## Repository layout

```
corpus/                  <- THE DATASET: corpus.h5, INDEX.json, README.md, provenance/
bin/                     <- entry points, gates, tests      lpdm/  <- LPDM, estimator, Kljun
docs/                    <- every document except this one. docs/README.md indexes them.
figures/                 <- the corpus pair figures; figures/old/ is the LES-pass record
jobs30/seed_*/return/    <- the 30-seed production library
runs/g{16,24,30}_base/   <- the .in TEMPLATES every case is built from. LOAD-BEARING.
data/                    <- raw geography (gitignored) and the built model grids
results/                 <- every scored artifact
validation_pairs_30m/    <- 2 ninth-pass validation records. NOT the corpus.
Dockerfile               <- CUDA 11.8, toolchain only. Every published result. Frozen.
Dockerfile.blackwell     <- CUDA 13.0, code baked in, sm_75..sm_120. The deployable one.
FastEddy-model-5.0.1/    <- the fork. Gitignored by the main repo.
```

**`docs/README.md` is the index of every document and says which are current.** The live
ones: `docs/FASTEDDY_TRAPS.md` every trap that has cost GPU time, read before running ·
`docs/DEPLOY.md` running on rented boxes · `docs/PLAN.md` the staged path and per-pass
verdicts · `docs/ML_TARGETS.md` the FNO target design · `docs/LIBRARY_PLAN.md` seed library
and corpus design. `docs/results/` holds the twenty per-pass and per-experiment write-ups —
**superseded on absolute numbers by this file**, kept for methodology and for how each
conclusion was reached.

`bin/fig_corpus_pairs.py` regenerates every figure in `figures/` from `corpus/corpus.h5`
alone, and re-derives the G2b and G3b counts against `corpus/FLAGGED.tsv` as it goes.
`figures/README.md` says how to read a pair panel.

**2026-09-01: 121 GB of LES scratch was removed** (`runs/*/{output,window}`, the `jobs*` dumps,
and a verified byte-identical duplicate seed library). Inventory, what was kept and why, and
the nine tests that now need a regenerated window: **`results/CLEANUP_INVENTORY.txt`**.

## Working agreement

- Report the gate result explicitly before moving on. Commit at every passed gate.
- Prefer reading FastEddy's own source over inferring behaviour from documentation. Every
  capability claim here carries a file and line number; keep it that way.
- Run the GPU and the CPU at once — the LPDM is an offline analysis and needs no GPU.
- Stop early only if a gate fails twice, a fix needs FastEddy source changes beyond the
  existing fork, a run projects far past its budget, or a result would change the grid
  decision.
