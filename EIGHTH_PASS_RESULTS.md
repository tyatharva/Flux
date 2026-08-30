# Eighth pass — 122³ @ 30 m, the in-process hand-off, two footprints per case

**In progress, 2026-08-30.** Phases 0 and 1 are complete and committed. Everything below
is measured unless it says otherwise.

---

## 1. The grid

`122 × 122 × 122 @ dx = dy = 30 m`, domain **3660 × 3660 m**. The vertical grid is
**identical to the retired 24 m solution** — `d_zeta` 24.691358,
`verticalDeformFactor` 0.346601, `dz_sfc` 8.5583 m, **k = 3 at exactly 30.000000000 m** —
so only the horizontal spacing changes. `Delta` 19.78 m, `z/Delta` **1.52**,
`dx/dz_sfc` **3.505**.

| | 24 m (retired) | **30 m** |
|---|---|---|
| domain | 2928 m | **3660 m** |
| cost, MEASURED | 0.481 GPU-h/sim-h | **0.479** at the production `dt` (14.781 ms/step) |
| geometric-mean `z0` | 0.0832 m | **0.0615 m** |
| water in the box | 8.78% | **13.61%** (2026 cells) |
| array in the box | 0.50% | **0.30%** (44 cells) |
| taper knee, MEASURED | pad 10 | **pad 12**, real geography to 1470 m |

**The taper knee was re-measured rather than carried.** pad 12 is the smallest ring at
which the taper stops being the steepest cell in the domain: max slope 0.1709, which is the
real terrain's own value, against 0.1787 at pad 10 and 0.1920 at pad 8. Below 12 the taper
itself starts setting the terrain `dt`, which is the wrong thing to pay for.

### Why 3660 m and not 186² @ 24 m

The containment gate failed for neutral at 2928 m (`CONTAINMENT_RESULT.md`): the flat
control's integral needed 1.5 domain lengths to stop growing and the cap removed 6.1%.
186² @ 24 m buys full containment at **+132%** (820 → 2150 GPU-h). 122³ @ 30 m buys 25%
more box for **3%**. A relative claim against Kljun does not need full containment, and the
parity number is what makes that defensible: **the LES retains 0.874 of its asymptote
against Kljun's 0.867 on identical cells**, so both models lose the same tail. The
acceptance is that the neutral integral **saturates by 2.5 L**, not that it is complete.

### The flat `dt` accuracy boundary, and what it says about the rule

A ladder from CFL_3d 1.30 to 1.70, branched off a developed state (`results/g30_bringup.txt`):

| CFL_3d | `dt` (s) | `k0/k1` | turb_alive | verdict |
|---|---|---|---|---|
| 1.30 | 0.0297170 | **0.130** | OK | OK |
| 1.40 | 0.0320029 | **0.130** | OK | OK |
| 1.45 | 0.0331459 | **0.130** | OK | OK |
| 1.50 | 0.0342888 | **0.130** | OK | OK |
| 1.55 | 0.0354318 | **8.857** | OK | **REJECT** |
| 1.60 | 0.0365748 | 8.433 | OK | REJECT |
| 1.65 | 0.0377177 | 8.078 | OK | REJECT |
| 1.70 | 0.0388607 | 7.591 | OK | REJECT |

**The boundary is between 1.50 and 1.55 — a factor of 68 in `k0/k1` across 0.05 of CFL.**
`turb_alive` reads OK at every rung, including the four that are grid-scale acoustic noise,
which is exactly why PROJECT_BRIEF.md pairs the two checks: `k0/k1` is the only one that sees this
failure, and `turb_alive` is the only one that sees a dead boundary layer.

**And it does not interpolate with anisotropy:**

| grid | `dx/dz_sfc` | boundary |
|---|---|---|
| 122³ @ 16 m | 4.007 | ~1.51 |
| 122³ @ 24 m | 2.804 | 1.55–1.60 |
| **122³ @ 30 m** | **3.505** | **1.50–1.55** |

This grid's anisotropy sits *between* the other two and its boundary sits at the *bottom*
of their range. Anisotropy was the last remaining candidate for a rule that would let the
number be predicted; it is not one. **The boundary is a property of the grid and must be
measured on every grid** — which is what PROJECT_BRIEF.md already said, now on a third independent
data point.

**Production: `dt = 5/162 s = 0.0308642`, CFL_3d 1.3502**, exactly 10.0% below the last
clean rung. 5/162 lands the 5 s cadence, the 300 s spin-up cadence, a 2.0 sim-h case
(233,280 steps) and a 3.0 sim-h seed (349,920) all on integer step counts.
**0.479 GPU-h per simulated hour** at that `dt`.

---

## 2. Gate A1 fails, and it is the site rather than the box

| | 2928 m | **3660 m** |
|---|---|---|
| worst case, all directions and stabilities | 17.45% | **25.93%** |
| worst case over **corpus regimes** | 7.38% | **11.58%** (neutral easterly) |
| Kljun `x90`, neutral | 1615 m | **1665 m** |

**The physical footprint did not change** — `x90` moved 50 m, and the small change is the
box's own geometric-mean `z0` moving, not the meteorology. What changed is that a 3660 m
box *holds* that footprint and a 2928 m box did not: the lake between 1464 and 1830 m east
was being replaced by a periodic re-sample of the box's own land.

**So the 2928 m PASS was truncation.** Shrinking the box to recover it would be passing the
gate by hiding exactly what the gate asks about. Recorded as a site limitation: E and NE are
~20% of the wind rose and carry 6–12% water; every other direction carries ~0%.

---

## 3. Coarsening to 30 m costs less resolution than z/Delta suggests

`bin/subgrid_apriori.py`, with **no GPU at all**: the 2-D spectrum of `w` on the receptor
level of the windows already on disk, split at each candidate grid's cutoff.

| to `dx` | `Delta` | `z/Delta` | kept @2dx | kept @4dx | kept @6dx |
|---|---|---|---|---|---|
| 24 (reference) | 17.02 | 1.76 | 100.0% | 94.6% | 78.3% |
| **30** | **19.75** | **1.52** | **99.7%** | **87.2%** | **64.7%** |
| 36 | 22.30 | 1.35 | 98.8% | 78.3% | 53.4% |

Neutral agrees closely (86.2% at 4dx). So the sub-grid fraction at the receptor moves
**52.5% → ~56% convective** and **86.4% → ~87% neutral** — a real cost, and nowhere near the
~90% that made a 10 m receptor closure output. Read it as a **lower bound** on the
degradation: a coarser run also produces different large scales, and its sub-grid model
returns some of what was filtered.

**This is a prediction, not the verdict.** The sub-grid fraction is re-measured on each
target's own window, and the deciding test is re-run at this grid rather than inherited.

---

## 4. The LES hands its fields to the LPDM in RAM

`SRC/IO/io_lpdmonline.c` (fork, 313 lines) + `lpdm/ringsrc.py` + `lpdm/dumpsrc.py`, behind
`lpdmOnlineSelector` (PARAM_OPTIONAL, default 0). **~20 GB of window scratch per case
becomes ~3 MB.** Over the corpus that is ~14 TB → ~4 GB.

Two design points where the obvious reading of the brief turned out to be wrong:

- **The ring holds a FULL WINDOW (541 slots, 6 fields, 12.0 GB), not `t_back`
  (180 slots, 4.0 GB).** The `sigma_w` floor is built from whole-window statistics
  (`lpdm/sgs_floor.py:62` consuming `window_stats`), so a shorter ring forces integration
  at each release time, which forces the floor onto partial-window statistics — an
  estimator change wearing a plumbing change's clothes. Sizing: a slot is
  122×123×123 wrap-padded, **3.691 MB per field fp16**, so 6 × 541 = 11.98 GB beside
  FastEddy's ~1.6 GB on a 16 GB card.
- **The route into the VRAM ring is host memory over tmpfs, not CUDA IPC.**
  `lpdmDevicePushSnapshotHost` takes HOST pointers by design — the wrap-pad, the fp16 cast
  and the eps/dsig2dz derivation are the device kernels already validated against the CPU
  path. The hop costs ~4 ms of PCIe per 5 s of model time against ~2.5 s of compute. IPC
  would buy that 4 ms and cost a new packing kernel on a validated integrator's critical
  path. **The ring is still in VRAM.**

### What is established, and what is not

| test | scope | result |
|---|---|---|
| `bin/test_dumpsrc.py` | the reader indirection, both windows | **PASS, bit-identical** |
| `bin/test_ringsrc.py` | the consumer, both windows | **PASS, bit-identical** vs an fp32 reference |
| `stage5 --ring` end to end | an identical 60-snapshot window | **PASS, 0.00e+00** on integral, asymptote, wrapped fraction and every `window_stats` field |
| `bin/test_lpdmonline.py` | **producer ↔ consumer, on a real LES** | **PASS, exact** — 23 snapshots, all 8 fields, max \|diff\| 0.000e+00 |

Bit-identity rather than a tolerance, because there is no physics between the two paths and
the correct tolerance is exactly zero. The CPU tests say plainly what they do not cover.

**The producer↔consumer test is the half no CPU test could reach.** One LES at
`lpdmOnlineSelector = 2` stages AND writes from the same `ioBuffField` at the same point in
the writer, so the two artifacts are comparable without running the LES twice and
differencing two turbulence realisations (44% in the integral on this project). All 23
snapshots agree exactly on u, v, w, theta, TKE_0, fricVel, z0m and invOblen, and
`window_stats` and `FieldSet` through both sources are bit-identical. That excludes a wrong
field order, a kji/ijk transpose, a pre-rho-division snapshot, an off-by-one ring slot, a
2-D field read with 3-D extents, and a truncated write.

Still owed on the hand-off: **Gate D1 well-mixed through the ring, both directions both
regimes**, and a footprint computed end to end from a live LES rather than from a replayed
window. Neither is a format question; both need a seed.

### Five real bugs found by building it

**`window_stats` was mixing two surface-flux estimators inside one window.** It used the
`htFlux` variable when a dump carried it and derived the flux per cell otherwise, on the
premise — written in the code — that `ioLPDMmode` never writes `htFlux`. That premise is
stale: `ioLPDMfullFrq` writes a FULL dump at every multiple, and a full dump carries it. On
`case_2023052519`, **2 of 12 sampled dumps took one branch and 10 took the other**, so the
window mean was a mean of neither. The two agree to **1.3e-7**, so nothing published is
visibly wrong; what was wrong is that the estimator depended on the OUTPUT MODE. Found by
the ring, which carries no `htFlux` and so could not reproduce the mixture.

Three more came out of running it against a real LES, and are in `FASTEDDY_TRAPS.md`
§20f–h: a 504-character comment in the `.in` template segfaults FastEddy's parser before it
starts (now refused by `docker/run_case.sh`, which says which line); `0` was the "disabled"
sentinel for a pause step and `0` is a real step; and the ring consumer used
delete-after-read to mark a snapshot consumed, so `keep=True` re-read the same 23 forever
until the container was OOM-killed.

**`window_stats` parsed the timestep with its own `rsplit` inside a `try/except`**, so a
non-path handle fell through to the dump INDEX and the time axis spanned 9 units instead of
82,134 — four orders of magnitude, silently. Caught by `test_dumpsrc.py` on its first run,
in the same block that already documents a prior bug of that shape.

---

## 5. Two footprints per case

1800 s adjustment + two 2700 s windows in one invocation. Window 2's releases begin 900 s
(`= t_back`) after window 1's end, so **the two windows' field intervals are disjoint by
construction**. `0.99 GPU-h per case, 0.50 per footprint` against 0.60 for one window at
24 m.

**Why**: re-running an identical case gave integral 1.463 → 1.019 and array share
5.65% → 1.07%. That is turbulence realisation variance, and every floor this project quotes
is a *within*-realisation floor and therefore too small.

**Whether it is worth it is a measured question**, and `bin/window_independence.py` asks it
two ways that fail differently — the release groups' own decorrelation ladder, and
|w0 − w1| against the within-window half-vs-half floor — alongside the `z_i` drift, because
near-replicates at one condition reduce realisation noise there and two different conditions
do not.

**The split rule tightens**: `<case>_w0` and `<case>_w1` share a seed, an adjustment, a
sounding and a surface, so `split_key` is the PARENT and the effective sample size for
generalisation stays **~1469**, not ~2938. Enforced in `bin/make_pair.py`, which was
splitting on the tag.

---

## 6. Still owed

1. **Gate D1 well-mixed through the ring**, both directions both regimes, and a footprint
   from a live LES rather than a replayed window. The FORMAT half of the acceptance has
   passed exactly; what is left needs a seed. **If D1 fails through the ring, fall back to
   the CPU path and say so.**
3. **The containment acceptance**: the neutral integral must saturate by 2.5 L.
4. **The deciding test, re-run.** The 24 m result does not certify this grid.
5. **Gate D1 on a seed window** on the production closure.
6. Rung re-spacing; `zCeiling` (the 3660 m width would support `z_i` to 1830 m at
   `L ≥ 2 z_i`, but the vertical still caps the band at 1250 m).
