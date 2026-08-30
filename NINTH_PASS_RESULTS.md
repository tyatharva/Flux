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
