# Ninth pass — the official FFP, the streamed hand-off, the .npz corpus record

**2026-08-30.** What this pass was for: finish the seed → target pipeline so corpus
generation can move to rented GPUs. Build first, then validate on two seeds and two targets.

Everything below is measured on this machine unless it says otherwise. Where a threshold is
not met, the number is reported and labelled — a threshold not met is a measurement, not a
failure; FAIL is reserved for the model breaking, the physics breaking, or the simulation
being unrealistic.

---

## 0. Kljun is now the official FFP, and our reimplementation was 25% wide where it matters

`third_party/FFP/` holds Natascha Kljun's own `calc_footprint_FFP.py` **v1.42**, vendored
unmodified with its ISC licence, retrieved 2026-08-30 from
<https://footprint.kljun.net/downloads/v1.42/FFP_Python.zip> (zip sha256
`50e188c5…64ba`; file hashes in `third_party/FFP/PROVENANCE.md`). `lpdm/kljun_ffp.py` is a
thin adapter that re-evaluates the official code's own two separable factors at our
north-up cell centres. **No formula is reimplemented.**

`bin/test_kljun_adapter.py`, on ten input sets spanning the corpus:

| part | what | result |
|---|---|---|
| **A** | the adapter against the code it wraps — **asserted** | worst **9.4e-16** of peak. **PASS** |
| **B** | `lpdm/kljun.py` against the official — **measured** | see below |
| **C** | the static-raster drop-in on 122² @ 30 m | **6.2e-3** against the cell-averaged `f_ci` |

**Part B found one real divergence and it is in the regime the standing regression lives
in.** `f_ci` agrees to 1e-14 and `x_peak` exactly, everywhere. `sigma_y` does not:

| case | `sigma_y` ours / official |
|---|---|
| every case with \|L\| < 5000 | **1.0000** |
| `L = −50000` | **1.2245** |
| `L = +inf`, the flat/neutral control | **1.2500** |

The official resets `ol = −1e6` above its own `oln = 5000` and then evaluates
`scale_const = 1e-5·|ol/z_m| + 0.80`, which at `z_m = 30 m` is **1.133 and is CLIPPED to
1.0**. `lpdm/kljun.py` short-circuits `|L| > 1e5` straight to `0.8` and never reaches the
clip, so it divides `sigma_y` by 0.8. **The flat/neutral control is exactly where `|L|` is
effectively infinite**, so the one place Kljun is diagnostic rather than descriptive is the
one place the reimplementation was wrong.

Part C's second number is a trap worth recording: scored against cell **centres** the same
raster reads **8.0e-1**, because `f_ci` climbs three orders of magnitude across one 30 m
cell in the near field. The raster is a cell **average** by construction and only a
cell-average reference means anything.

`bin/stage5_footprint.py` and the `.npz` writer now both take Kljun from the official code.
`lpdm/kljun.py` survives for the gates already validated against it, and
`bin/test_kljun_adapter.py` keeps the two from drifting apart silently.

---

## 1. The hand-off was not streaming: it moved ~20 GB from disk to RAM

**Found by measurement, not by failure.** Every check passed while it was true.
`RingConsumer.drain_until_pause` returned the whole window as a list — 541 × 36.5 MB =
**19.7 GB** — and `FieldSet` retained that list for its own lifetime *on top of* its own
12.0 GB fp16 cache, because `window_stats` then opened every dump a **second** time. Peak
~32 GB, on a machine class that is being chosen for rented boxes.

Deleting the tmpfs file releases the **producer's** backpressure and nothing else. The
consumer's own bound was a separate statement and nothing was making it.

**The fix fuses the two passes into one.** `lpdm/les_stats.py:WindowAccumulator` is the
estimator as an accumulator and `window_stats()` is now a *thin loop over it*, so there is
one implementation rather than two that agree; `FieldSet.load()` feeds it and releases each
handle; `RingConsumer.iter_until_pause()` yields instead of collecting.

**Identity is asserted at exactly zero** — there is no physics between two schedules over
the same arithmetic:

| comparison | result |
|---|---|
| `window_stats` new vs the pre-refactor code from git, 2 real 541-dump windows, 2 receptor levels (one fractional) | **0.000e+00** across 25 fields |
| streamed vs batched `FieldSet` cache, 9 arrays | **bit-identical** |
| time axis and cadence | identical |
| streamed vs batched `window_stats`, 25 fields | **0.000e+00** |
| staging files left behind | 0 |

**Host residency, measured in isolation** (each route in its own process, 24 snapshots):

| | batched | streamed |
|---|---|---|
| peak RSS | **1.754 GB** | **0.937 GB** |
| snapshots retained | 24 (876 MB) | 1–2 (73 MB) |
| `h`, `u*` returned | 1144.0 m, 0.5135 | identical |

Extrapolated to a production 541-snapshot window: **~31.7 GB → ~12.4 GB.**

**What streaming cannot reach, said rather than implied.** The 12.0 GB field cache is not
buildup — it IS the window, and `compute_footprint` is a **CPU** integrator that
random-accesses all of it. Host residency floors at the cache. Reaching one or two
snapshots needs the window in VRAM and the integration there (`lpdm/gpu.py`), which is an
**integrator** change rather than a plumbing one and remains a deferred item. The directive's
"~37–74 MB steady state" is therefore met for the *staging* and for the *snapshots*, and
not for total host RSS.

**The `/dev/shm` guard**: `lpdm/hostwatch.py:ShmGuard` refuses a staging filesystem smaller
than `(queue + 2) × snapshot` at second zero, naming the figure. Docker's default is 64 MB
and a snapshot here is 36.5 MB.

---

## 2. The training record is one self-contained `.npz`

`bin/make_pair.py --npz-dir` writes `pairs_npz/<tag>.npz`: `scalars` (6,), `kljun`
(128, 128) f32, `target` (128, 128) f32, and a `meta` blob, plus a per-machine
`manifest.json`. **26–47 kB each**, so ~2900 windows is ~100 MB.

- **122 → 128 is a zero-pad of 3 cells, not a resize.** Verified on a real record: the
  border is exactly zero on both channels, the interior carries 4106 non-zero cells of
  14 884, and the receptor moves (61, 61) → (64, 64), recorded in both frames.
- **The signed target survives**: 1414 negative cells, 3.7% of |flux|, unclipped.
- **The Kljun channel is re-evaluated on the target's own cell edges**, so identity is
  structural rather than asserted; the edges are checked against the centres to 1e-9 m first.
- `meta["h_estimator"] = "tke_peak_fraction"`, named because this project has two
  definitions of `h` that differ by 7–21%.
- `L` is written raw as the format names it, with `inv_L` beside it in `meta` — `L` is
  ±inf at exactly neutral, which is legitimate and not trainable.

**The six existing records were regenerated**, and the stale `z0_geometric_m = 0.1435`
literal was wrong in both directions:

| records | recorded | true |
|---|---|---|
| four 16 m cases | 0.1435 | **0.14488** (1% off; the literal predates `--raise-topo`) |
| two 24 m cases | 0.1435 | **0.08323** — wrong by **72%** |

Their Kljun integrals move 0.07–0.75% with the official FFP; the largest,
`case_2023101222` at 0.9594 → 0.9522, is the `sigma_y` clip.

---

## 3. The in-process hand-off runs a real two-window case

`bin/run_corpus_case.sh RING=1` launches the LES in the background and consumes each window
live. Measured on an 800 s two-window smoke case at `lpdmOnlineSelector = 2`, 122³ @ 24 m:

| | |
|---|---|
| staged snapshots vs netCDF dumps written | **162 vs 162** — the two paths agree exactly |
| peak staging directory | **58.3 MB = 1.6 snapshots** (2 files), against a producer queue depth of 4 (146 MB) and a whole window of 2.2 GB |
| peak consumer host RSS | 1.74 GB against a 1.35 GB field cache |
| LES paused per window boundary | **1.6 s** |
| both pauses fired and resumed | yes; LES exited 0, `k0/k1` 0.322, `turb_alive` OK |
| deferred surface read-back | z0m 1.5e-09, htFlux 3.3e-08 |

**The requirement holds: one to two snapshots staged, never accumulating across a window,
and the consumer deletes after read — which is what releases the producer's backpressure.**

### Three bugs the smoke run found, all of which would have cost production GPU time

1. **Consecutive windows cannot share a boundary dump through the ring.** The consumer
   deletes each snapshot as it reads it, so window 0 consumes the boundary and window 1
   starts one output interval late: its release period came out **195.0 s against 200 s**
   and `--strict-rel` refused it. At production geometry that is every second window of the
   corpus, failing after ~1 GPU-h per case. Windows are now spaced one output interval
   apart on **both** paths.
2. **Killing the shell is not stopping the LES.** The driver's trap killed the backgrounded
   subshell and left the FastEddy *container* holding the GPU; the next run was refused with
   no hint of why. Containers are now named and removed by name.
3. **`bin/seed_accept.sh` is not a CPU analysis.** Its Gate C2 restarts the seed and
   re-dumps — that is an LES. Backgrounding it raced run 2 into the one-FastEddy-at-a-time
   refusal and killed the case before a step ran. Its own guard did not help: a check
   cannot serialise, only queueing can. The battery is now foreground and `wait_for_gpu()`
   blocks before every launch.

---

## 4. The two datetimes, and why

Selected from `results/candidates.tsv`, which was already built — so choosing them cost no
network and no GPU.

| | convective | neutral |
|---|---|---|
| valid time (UTC) | **2023-11-17T18:00** | **2023-11-21T20:00** |
| `z_i` | 681 m | 711 m |
| `dz_i/dt` | +13.1 %/h | **+0.8 %/h** |
| SHTFL | 179 W/m² | **1 W/m²** (z/L ≈ −0.005) |
| wind FROM | 331° | 326° |
| wind speed | 4.9 m/s | 5.0 m/s |
| seed chosen | `seed_cbl-mid_a015` rot 3, **1.0° direction gap**, +19 m depth gap | `seed_nbl-deep_a015` rot 3, 4.4° gap |

**The direction is not incidental.** At a 30 m receptor Kljun's array share is 30.7% for a
northerly footprint and 0.04% for an easterly one, so the array signal this site exists to
measure is present only in the northern sector. **The pre-registered 24 m convective target
was easterly** (achieved 88.8°, array share 1.07%), which is why it is not reused — and the
cost of that choice is losing the like-for-like comparison with its 144 m peak. Both new
cases sit in the same northern sector, so they are comparable to each other on the array.

Both dates are inside HRRR v4 (from 2020-12-02).

---

## 5. Validation

*(runs in progress; this section is completed as they land)*

### Run 2 — convective target `case_2023111718`, two windows, through the live ring

**Ran end to end at production scale.** 234 883 steps = 2.0 sim-h + one output interval,
1442 snapshots, both pauses fired and resumed, LES exited 0, 16:36:49 wall.

**The hand-off, measured on a live LES:**

| | |
|---|---|
| staged snapshots vs netCDF dumps written | **1442 vs 1442** |
| snapshots expected vs delivered, per window | **541 vs 541** |
| adjustment snapshots drained and dropped | 360 |
| streamed cadence | 5.0000 s, margin **0.000%** of a 0.5 s bar |
| release period | **1800.0000 s**, margin **+0.00e+00 s** |
| **peak staging directory** | **58.3 MB = 1.6 snapshots (2 files)**, against a queue depth of 4 (146 MB) and a whole window of **19.7 GB** |
| **peak consumer host RSS** | **12.45 GB** = the 11.98 GB field cache + one snapshot, against **~31.7 GB** before the streaming fix |
| **LES pause per window boundary** | **45.4 s** |
| window load | 2345 s, i.e. LES-limited and therefore free |

**Acceptance (a): CPU-from-disk vs from-ring — PASS at 0.27× the run's own half-vs-half
floor.**

| observable | ring | disk | \|diff\| | × floor |
|---|---|---|---|---|
| peak | 180 m | 180 m | **0** | 0.00 |
| centroid | 350.0 m | 356.8 m | 6.8 m | 0.23 |
| A80 | 14.40 ha | 14.85 ha | 0.45 ha | 0.01 |
| x80 | 570.2 m | 578.2 m | 8.0 m | 0.27 |
| integral | 0.9827 | 0.9974 | 0.0147 | 0.015 of the asymptote |
| **array share** | **25.47%** | **25.45%** | **0.02 points** | 0.00 |

**(c)** the negative-lobe weight fraction is **0.390 on both paths** — signed weights
preserved. **(d)** fp16 parity is what (a) measures: the disk path is CF-packed to 16 bit
and the ring is raw fp32. And `window_stats` — the check that would catch a slot-ordering
or rho-division error in the producer — agrees to **1.3e-07 relative across 20 scalars**,
i.e. fp32 roundoff.

**Acceptance (b): Gate D1 well-mixed, production closure, BOTH directions — PASS.**
max \|ratio − 1\| **19.39%**, rms 7.58%, lowest three bins **1.047**, against a 5.48%
counting-noise floor and a max(10%, 4σ = 21.9%) bar. **This is the deferred item PLAN.md
records as having no evidence at this grid; the convective half now has it.**
The script's composite verdict prints FAIL, and it is worth saying why: it is
`ok and frac > 0.5`, where `frac = 37.7%` is the fraction of 20 000 particles reaching the
surface within a 900 s `t_limit`. That is a `t_back`-sizing statistic against a picked
constant, not a statement about the closure.

**The footprint, and the array signal the northerly choice was for:**

| | w0 | w1 |
|---|---|---|
| peak | **180 m** | **180 m** |
| centroid | 350 m at 324.7° | 354 m at 316.0° |
| A80 | 14.4 ha | 27.9 ha |
| integral | 0.983 | 1.021 |
| **array share** | **25.47%** (unwrapped 26.12%) | **19.49%** (20.72%) |
| achieved direction | 335.3° | 325.9° |
| achieved `h` | 986 m | 1063 m |
| `L` | −53.3 m | −46.1 m |
| **sub-grid fraction of σ_w² at the receptor** | **59.3%** | **57.7%** |

The array is **0.30% of the box by area**, so 25.47% is an **85× enrichment** — against
1.07% for the pre-registered easterly 24 m target. Kljun on identical cells gives peak
150 m, A80 23.2 ha, integral 0.894. The sub-grid fraction lands at 59.3% against ~56%
predicted a priori from the spectra.

**THE NO-OP CONTROL: the peak is the LES's, not the closure's.** The identical window
recomputed with the σ_w floor OFF:

| | floor ON | floor OFF |
|---|---|---|
| **peak** | **180 m** | **180 m** |
| x80 | 570 m | 701 m |
| A80 | 14.4 ha | 19.9 ha |
| array share | 25.47% | 18.11% (**+7.37 points** from the floor) |
| integral | 0.983 | 0.959 |

The peak does not move; the shape does. The +7.37 points is consistent with the +8.40
recorded at 10 m.

**Containment (cap raised to 3 L = 10 980 m):** C1 **FAIL at +2.1%** over the last
quarter-domain against a 2% bar — i.e. marginally at it; C2 ok (x80 599 m against 2928 m);
C3 ok (+1.1% hidden by the cap). The integral over the `1 − z_m/z_i` asymptote is **1.050**,
above 1, which truncation cannot cause: the receptor sits in mean subsidence
(**W = −0.160 m/s**), which is the advective non-closure this project already measures with
the right sign. **The binding containment case is the flat/neutral control, not this one.**

**Two windows: near-duplicate in shape, but a different condition.** `bin/window_independence.py`:

| observable | w0 | w1 | \|diff\| | floor | × floor |
|---|---|---|---|---|---|
| peak | 180.0 | 180.0 | 0.0 | 30.0 | **0.00** |
| centroid | 350.0 | 353.9 | 3.8 | 331.2 | 0.01 |
| x80 | 570.2 | 735.5 | 165.3 | 439.8 | 0.38 |
| array share (group mean) | 32.88% | 19.95% | 12.93 pts | 7.55 | 1.71 (inside 3 SE) |

Median \|diff\|/floor **0.19** → NEAR-DUPLICATE. The release groups' own lagged
autocorrelation puts the decorrelation time at **180 s** and the two windows' release
periods are **1800 s** apart, so they are independent draws — and yet the footprints agree
to a fifth of the within-window floor. **The realisation variance that motivated the
two-window design is far smaller here than the 44% on record**: the integrals are 0.983 and
1.021, 3.8% apart, against 1.463 → 1.019 measured on an identical 24 m re-run.

**`z_i` moved +77 m between the windows**, so the pair is **coverage, not replication**: it
does not reduce noise at a fixed condition, and the corpus still owes averaging within
condition bins. *(Note: the independence tool compares the MEAN over release groups, which
is why its w0 array share reads 32.88% where the pooled ensemble share is 25.47%. Both are
legitimate; the pooled value is the share, the group statistics are the comparison basis.)*

**σ_w against the tower — OUTSIDE the IQR at 0.68× the median, and the reason is not what
it looks like.** LES σ_w 0.575 m/s against a tower median of 0.848 for the +179 W/m² bin.
The obvious explanation, a wind-speed mismatch, is **wrong**: the bin's `ustar_median` is
**0.457** against this case's **0.468**, matched to 2.4%. The real decomposition:

| | σ_w/u* | ζ at 30 m | 1.25·φ_w at that ζ |
|---|---|---|---|
| tower bin | 1.855 | −0.737 | 1.844 |
| this case (LES, unfloored) | **1.230** | −0.532 | **1.718** |

Two things follow. The tower curve **is** MOST — `bin/sigma_w_tower.py` inverts
σ_w(10) = 1.25 u* φ_w for u* and re-predicts at 30 m, so its 1.855 is its own 1.844 — and
part of the gap is that the bin sits at a more unstable ζ than this case. Corrected for
that, the LES still resolves-plus-models only **0.716×** what MOST wants at its own
stability. The σ_w floor lifts the receptor value to **1.51 u*, i.e. 0.88× MOST**, closing
about half the gap. **This is the opposite sign to the 10 m failure**, where the LES ran
2.33–2.99× HIGH with the floor inactive.

**Which of the three this is:** a **measurement**. Not the model breaking, not the physics
breaking. The stage-7c gate is doing its job — refusing to call a case clean when its σ_w
is outside the tower's band — and the band is a MOST extrapolation conditioned on heat flux
alone, which PROJECT_BRIEF.md already records as spanning ~2×.

### Run 3 — neutral seed `seed_nbl-deep_a015`, 3.0 sim-h ceiling, accelerator on

3000 s of burn-in at `surflayer_wth = +0.05`, restart with `htFlux` **zeroed in the file**
(it is IO-registered and the `.in` cannot override it), then 340 200 steps = **2.917 sim-h**
in one invocation, 35 dumps. The watcher scored at 0.5/0.75/1.0/1.25 h and never entered
band — its drifting count ran 2 → 4 → 1 → 0 — so the run went to the ceiling.

**Verdict: FAIL — `z_i` DRIFTING at +5.76 %/h against a 3 %/h limit**, plus INDETERMINATE on
`sigma_v/u*`, `sigma_w/u*` and `TKE_BL/u*^2`. But three limits are **established ok, and
they are the three the footprint's geometry actually rides on**:

| limit | value | trend | verdict |
|---|---|---|---|
| `U/u*` (Kljun Π₄) | 16.358 | **+0.00 %/h** | ok |
| Kljun `x_peak` | 178.7 m | −0.27 %/h | ok |
| Kljun `x90` | 1493.5 m | −0.14 %/h | ok |
| `z_i` | 678.5 m | **+5.76 %/h** | **DRIFTING** |

**THIS CONFIRMS A PREDICTION THE PROJECT MADE IN ADVANCE.** PLAN.md item 0aa, from a
scoring-window sweep run before any of this: *"**TRENDING AWAY from band** — `z_i` in BOTH
seeds, monotonically (+2.31 → +4.08 and +0.53 → +7.57 %/h) with a falling SE. **A longer run
resolves these into a FAIL, not a pass.**"* It did, at +5.76 %/h.

**And the consequence is structural.** `bin/pick_seed.py` refuses a DRIFTING seed outright
and no flag admitted it, so **the neutral half of the corpus became unbuildable** — and no
longer spin-up fixes it, because a neutral Ekman layer's depth keeps growing for several
inertial periods (35–50 simulated hours). PROJECT_BRIEF.md makes exactly this argument for `u*`,
where the fix was to gate on a RATIO; `z_i` is the one gated quantity with no ratio to take,
and PROJECT_BRIEF.md says so explicitly.

**The fallback turned out to be more dangerous than the refusal.** With `--available-only`
leaving one spun seed in the library, `pick_seed` fell back to the **convective**
`seed_cbl-mid_a015` for a neutral case and said so: *"no neutral seed in the library; fell
back to convective. 30 min will NOT convert one regime into the other."* That is
disqualifying in a way the drifting seed is not — a neutral target restarted from a
convective state is a decaying CBL for the length of the case, and no label on the pair
fixes a wrong TARGET. Run 5 was killed 13 minutes in and re-run against the neutral seed
under a new, narrow, **default-off** `--allow-drifting`, which stamps
`gate_state = DRIFTING` on every pair and which the corpus driver never sets.

**Whether the corpus should use it is a design decision and it belongs to the user.** The
numbers to decide on are above.

### Run 4 — the flat/neutral control, and the containment acceptance

87 480 steps = 2700 s, 541 dumps, LES clean: `k0/k1` **0.136**, `turb_alive` OK, exit 0.
First run at this grid, so `results/regression_baseline_g30.json` is a baseline write rather
than a comparison.

**THE CONTAINMENT ACCEPTANCE THIS GRID WAS CHOSEN FOR: PASS.** The by-displacement ladder,
cap raised to 3 L = 10 980 m:

| displacement | integral | / I(1 L) |
|---|---|---|
| 0.25 L | 0.420 | 0.591 |
| 0.50 L | 0.581 | 0.816 |
| 0.75 L | 0.661 | 0.928 |
| **1.00 L** (the production cap) | **0.712** | 1.000 |
| 1.25 L | 0.736 | 1.034 |
| 1.50 L | 0.738 | 1.036 |
| 2.00 L | **0.730** | 1.025 |
| **2.50 L** | **0.730** | 1.025 |
| 3.00 L | **0.730** | 1.025 |

`|I(2.5L) − I(2.0L)| / I(2.5L)` = **0.0%** against the 2% `SATURATE_TOL`. **The neutral
integral saturates by 2.5 L**, which is the acceptance PLAN.md sets, and it is flat to three
decimals from 2.0 L. The production cap at 1 L captures **97.5%** of the saturated integral,
against **93.5%** at 2928 m — the bigger box bought four points.

The stricter question still fails, and both numbers belong in the record: **C1 (saturates by
1 L) FAILS at +7.2%** and **C3 (what the cap hides) at +2.5%**, against +8.8% and +6.1% at
2928 m. C2 passes (x80 1733 m against a 2928 m bar).

**GATE D1, NEUTRAL, BOTH DIRECTIONS: PASS.** max |ratio − 1| **9.86%**, rms 4.70%, lowest
three bins **1.043**, against a 5.48% counting floor. **With run 2's convective PASS, the
deferred item — "Gate D1 on the production closure, both regimes, both directions" — now has
evidence in both regimes at this grid for the first time.** (The composite "stage 4" verdict
prints FAIL on the same `frac > 0.5` particle-count criterion as run 2: 44.9% here.)

**THE NO-OP CONTROL BEHAVES DIFFERENTLY NEUTRALLY, AND THAT IS THE POINT OF RUNNING IT IN
BOTH REGIMES.** Convectively the peak was **identical** with the floor on and off. Neutrally
it moves:

| | floor ON | floor OFF |
|---|---|---|
| peak | 270 m | **330 m** |
| x80 | 1624 m | 1666 m |
| A80 | 27.8 ha | 29.2 ha |
| integral | 0.723 | 0.700 |

The receptor here is **90.4% sub-grid** in σ_w² (85.1% at 24 m), so the neutral flat control
is the closure-dominated end of the corpus and the floor is not inert in it.

**AND THE KLJUN-PARITY ARGUMENT DOES NOT SURVIVE THE MOVE TO 3660 m.** At 2928 m the LES
retained **0.874** of its `1 − z_m/z_i` asymptote against Kljun's **0.867** on identical
cells — parity to 0.7 points, and that parity is what made accepting the truncation
defensible ("both models lose the same tail, so a RELATIVE claim survives"). At 3660 m:

| | 24 m / 2928 m | 30 m / 3660 m |
|---|---|---|
| LES / asymptote | **0.874** | **0.765** |
| Kljun / asymptote | **0.867** | **0.923** |
| gap | 0.7 points | **15.8 points** |

**It is not the Kljun fix.** That was the obvious suspect — the flat control is exactly the
`|L| > 5000` regime where `lpdm/kljun.py` is 1.25× wide in σ_y — and it is wrong:
recomputing Kljun on the 2928 m control's *identical* cells with the official FFP gives
**0.8263, the same to four decimals**. The integral is insensitive to σ_y because σ_y stays
small against a 1464 m box half-width, so widening it moves almost nothing out of the box.
The recorded parity is real.

**It is the LES.** The two controls are nearly the same flow — u\* 0.410 vs 0.405, U/u\*
15.99 vs 16.27, `z_i` 643 m in both, σ_v 0.577 vs 0.584 — and Kljun agrees (peak 192 vs
180 m, x80 931 vs 964 m, A80 13.1 vs 13.1 ha). The **LES** footprint is the thing that
changed: peak 240 → 270–330 m, x80 1557 → 1733 m, A80 31.8 → 34.0 ha. Two measured
candidates, and this pass does not separate them: the sub-grid fraction at the receptor rose
**85.1% → 90.4%** at the coarser grid, and only **44.9% of particles reach the surface within
the 900 s `t_back`** (median transit **307 s**, against 118 s convectively).

**Which of the three this is:** a **measurement**, and the one that most changes what can be
claimed. The acceptance PLAN.md set (saturation by 2.5 L) passes; the *justification* it
rests on (LES–Kljun parity under truncation) holds at 2928 m and does **not** hold at
3660 m. Any relative claim against Kljun on the neutral flat control now has to carry the
15.8-point gap explicitly.

### Run 1 — convective seed `seed_cbl-mid_a015`, 1.0 sim-h ceiling

Ran clean: 106 920 steps = **0.917 sim-h** in one invocation (the ceiling rounds DOWN to a
whole number of dumps), 11 dumps, ~26 min wall.

**The stationarity battery could not return a verdict, and the reason is the ceiling
itself.** `bin/seed_stationarity.py` scores a **2.0 h** window; the run is 0.92 h, so the
window necessarily includes step 0 — the cold start, where `u* = 0` by construction. Every
ratio with `u*` in its denominator came back `inf` or `nan`, and all seven gated limits
returned INDETERMINATE. This is **not** the familiar "n_eff saturates at 3–5" INDETERMINATE
that PROJECT_BRIEF.md documents; it is the gate being handed a window it cannot score.

What *is* measurable at the ceiling:

| | value | |
|---|---|---|
| `u*` | 0.4751 | trending **+45.7 %/h** |
| `z_i` (fixed 0.01 m²/s² threshold, gated) | 646 m | **+143.6 %/h** against a 3 %/h limit |
| `z_i` (5% of running peak — the corpus currency) | 609 m | **+126.8 %/h** |
| eddy turnovers (`z_i/w*` ≈ 489 s) | **≈ 6.8** | |

So the **turbulence** is developed — nearly seven turnovers — while the **depth** is still
in free growth at ~900 m/h against a 700 m rung target. `bin/seed_accept.sh` reports
"NOT IN BAND ANYWHERE up to 0.92 h", which is the honest form: the margin at the ceiling
rather than a stop time.

**Which of the three this is:** not the model breaking and not the physics breaking. It is
a **measurement**: the 1.0 sim-h convective ceiling is shorter than this rung needs, and
shorter than the gate can score at all. The seed was still produced and is usable
(`ALLOW_INDETERMINATE=1` is the library's normal operating mode and stamps
`gate_state` onto every pair), and the consequence to watch is where the target's achieved
`z_i` lands, which run 2 measures rather than predicts.
