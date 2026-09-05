# Ninth pass: the official FFP, the streamed hand-off, the corpus record

2026-08-30. The last validation pass before generation moved to rented GPUs: build first, then
two seeds and two targets at production scale. About 5.1 GPU-h. Where a threshold was not met
the number is reported and labelled; FAIL was reserved for the model breaking.

## The report in one place

| item | result |
|---|---|
| Kljun verification | official FFP v1.42 vendored; adapter faithful to 9.4e-16. The project's reimplementation was 1.2500× wide in `σ_y` at \|L\| > 5000 |
| Gate D1 well-mixed, both directions, both regimes | PASS convective (run 2, max \|ratio − 1\| 19.39%) and PASS neutral (run 4, 9.86%) |
| does the peak move between windows w0 and w1 | no: identical, 180 m and 180 m, against a 30 m floor; median \|diff\|/floor 0.19 |
| neutral integral saturates by 2.5 L | PASS at 0.0%: 0.730 at 2.0, 2.5 and 3.0 L; the 1 L cap keeps 97.5% |
| sub-grid fraction of `σ_w²` at the receptor | 59.3% / 57.7% convective (about 56% predicted); 90.4% on the neutral flat control |
| LES `σ_w` vs the tower at 30 m | convective outside the IQR at 0.68× the median; neutral inside at 1.19× |
| measured GPU-h per simulated hour | 0.525 at selector 2, 0.512 net of pauses, against 0.479 at bring-up; +7% is the double write |
| seed stop times | neither seed stopped early |
| pause per window boundary | 45.4, 46.8, 46.9 s |
| peak host RSS / staging | 12.45 GB (the field cache) and 58.3 MB = 1.6 snapshots, against 19.7 GB of window |
| persisted per case | 3.6 MB against 19 GB of window dumps |

**What was failing rather than measured: the `h` estimator on neutral profiles whose wave layer
is stronger than their boundary layer** (traps §22). It refused nothing, returned 2372 m, drove
the `σ_w` floor to 1.2e+04, and the corpus monitor's G1 caught it two stages later. It blocked
the neutral half of the corpus until `bl_depth` was redefined on the surface-attached layer the
next day (48 of 47 stored profiles exact; the refused case then gave 448 m and passed every
scoreable gate; `results/pass10/`).

## Kljun is the official FFP

`third_party/FFP/` holds Kljun's own `calc_footprint_FFP.py` v1.42, vendored unmodified with
its ISC licence, retrieved 2026-08-30 from `footprint.kljun.net/downloads/v1.42/FFP_Python.zip`
(hashes in `third_party/FFP/PROVENANCE.md`). `lpdm/kljun_ffp.py` re-evaluates the official
code's two separable factors at the north-up cell centres; no formula is reimplemented.
`bin/test_kljun_adapter.py` scores the adapter against the code it wraps at 9.4e-16 on ten input
sets, and found one real divergence in `lpdm/kljun.py`: `σ_y` 1.2245× wide at `L = −50000` and
1.2500× at `L = +inf`, because the official code resets `ol = −1e6` above `oln = 5000` and clips
`scale_const` to 1.0 while ours short-circuited `|L| > 1e5` to 0.8. The flat/neutral control,
the one place Kljun is diagnostic, is exactly where `|L|` is effectively infinite. A trap for
scoring the static raster: against cell centres it reads 8.0e-1 because `f_ci` climbs three
orders of magnitude across one 30 m cell; the raster is a cell average and only a
cell-average reference means anything (6.2e-3).

## The hand-off was not streaming

Every check passed while it moved about 20 GB from disk to RAM (traps §21a). The fix fused the
two passes over the window into one accumulator; identity asserted at exactly zero across 25
`window_stats` fields, 9 cache arrays, both windows, two receptor levels; peak RSS 1.754 →
0.937 GB on 24 snapshots, extrapolated 31.7 → 12.4 GB for a production window. What streaming
cannot reach: the 12.0 GB field cache is the window. `lpdm/hostwatch.py:ShmGuard` refuses a
staging filesystem smaller than `(queue + 2) × snapshot` at second zero (Docker's default
`/dev/shm` is 64 MB; a snapshot is 36.5 MB).

## The training record

One self-contained `.npz` per window ([targets](../emulator/targets-and-architecture.md)),
26–47 kB. Verified on a real record: the 3-cell pad exactly zero on both channels, 4106 non-zero
interior cells, receptor (61, 61) → (64, 64), 1414 negative cells carrying 3.7% of |flux|,
unclipped. The six existing records were regenerated because a `z0_geometric_m = 0.1435`
literal was wrong in both directions (true 0.14488 at 16 m, 0.08323 at 24 m: 72% off).

## Run 2: the convective target `case_2023111718`, two windows, through the live ring

2023-11-17 18Z, `z_i` 681 m, `SHTFL` 179 W/m², wind from 331°, seed `cbl-mid_a015` rot 3 with
a 1.0° direction gap. Chosen northerly because at 30 m Kljun's array share is 30.7% for a
northerly and 0.04% for an easterly, and the pre-registered 24 m convective target had been
easterly (array share 1.07%). 234,883 steps = 2.0 sim-h plus one output interval, 1442
snapshots staged and 1442 written, both pauses fired, LES exited 0.

Acceptance (a), CPU-from-disk against from-ring, PASS at 0.27× the run's own half-vs-half floor:
peak 180 m on both, centroid 6.8 m apart, A80 0.45 ha, integral 0.9827 vs 0.9974, array share
25.47% vs 25.45%; the negative-lobe weight fraction 0.390 on both; `window_stats` agreeing to
1.3e-7 across 20 scalars. Acceptance (b), Gate D1 convective through the ring, both directions:
PASS (max 19.39%, rms 7.58%, lowest three 1.047 against a 5.48% floor and a 21.9% bar).

The footprint: peak 180 m in both windows, array share **25.47%** (w0) and 19.49% (w1) against
0.30% of the box by area, an 85× enrichment; Kljun on identical cells 150 m, 23.2 ha, 0.894.
**The no-op control**: with the floor off the peak is still 180 m while `x80` goes 570 → 701 m,
A80 14.4 → 19.9 ha and the array share 25.47 → 18.11% (+7.37 points from the floor, consistent
with the +8.40 at 10 m). Containment with the cap raised to 3 L: +2.1% over the last quarter
domain, marginal; the integral over the asymptote is 1.050 in mean subsidence
(`W = −0.160 m/s`).

**The two windows are near-duplicates in shape but a different condition**: peak 0.0 of a 30 m
floor apart, centroid 3.8 m, median |diff|/floor 0.19, while the release groups decorrelate in
180 s against an 1800 s separation, so they are independent draws. Integrals 0.983 and 1.021,
3.8% apart, against 1.463 → 1.019 on an identical 24 m re-run. `z_i` moved +77 m between them,
so the pair is coverage, not replication.

`σ_w` against the tower: outside the IQR at 0.68× the median, and not a wind-speed mismatch
(`u*` matched to 2.4%). Decomposed: the tower bin sits at a more unstable ζ, and corrected for
that the LES resolves-plus-models 0.716× what MOST wants at its own stability; the floor lifts
it to 0.88×. The opposite sign to the 10 m failure. A measurement, and the stage-7c gate doing
its job.

## Run 3: the neutral seed `seed_nbl-deep_a015`, 3.0 sim-h ceiling, accelerator on

FAIL: `z_i` DRIFTING at +5.76 %/h against 3, plus INDETERMINATE on `σ_v/u*`, `σ_w/u*` and
`TKE_BL/u*²`, while the three limits the footprint's geometry rides on were established ok
(`U/u*` +0.00 %/h, `x_peak` −0.27, `x90` −0.14). This confirmed a prediction made in advance
from a scoring-window sweep: `z_i` in neutral seeds trends away from band with run length,
because a neutral Ekman layer's depth keeps growing for several inertial periods. Since
`pick_seed.py` refused a DRIFTING seed outright, the neutral half of the corpus became
unbuildable, and the fallback (a convective seed for a neutral case; run 5 was killed 13
minutes in for it) was worse than the refusal. That handed the decision back, and it was
decided the next day: first `--allow-drifting zi-neutral`, then on 2026-08-31 the whole
library ([seed library](../les/seed-library.md)).

## Run 4: the flat/neutral control, and the containment acceptance

2700 s, 541 dumps, `k0/k1` 0.136, `turb_alive` OK. **Containment PASS**: the by-displacement
ladder with the cap raised to 3 L gives 0.712 at 1 L, 0.738 at 1.5 L, 0.730 at 2.0, 2.5 and
3.0 L; `|I(2.5L) − I(2.0L)| / I(2.5L)` = 0.0% against a 2% tolerance, and the production cap
keeps 97.5% of the saturated integral against 93.5% at 2928 m. The stricter questions still
fail (saturates by 1 L: +7.2%; what the cap hides: +2.5%). **Gate D1 neutral, both directions:
PASS** (max 9.86%, rms 4.70%, lowest three 1.043). The no-op control behaves differently
neutrally: the peak moves 270 → 330 m with the floor off, because the receptor is 90.4%
sub-grid here.

**And the Kljun-parity argument does not survive 3660 m.** LES 0.765 of its asymptote against
Kljun's 0.923 (a 15.8-point gap) where at 2928 m it was 0.874 against 0.867. Not the Kljun fix
(both implementations give 0.8263 on the 24 m control; the integral is insensitive to `σ_y`).
The LES footprint is what changed in a nearly identical flow (`u*` 0.410 vs 0.405, `z_i` 643 m
in both): peak 240 → 270–330 m, `x80` 1557 → 1733 m, with the sub-grid fraction 85.1% → 90.4%
and only 44.9% of particles reaching the surface within `t_back` (median transit 307 s
against 118 convectively). Any relative claim against Kljun on the neutral flat control now
carries that gap ([limitations](../limitations-and-future-work.md)).

## Run 5: the neutral target `case_2023112120`, refused

The plumbing worked exactly as on the convective target (1442 vs 1442, 58.3 MB staging,
12.46 GB RSS, `σ_w` inside the tower IQR at 1.19×), and the case was refused because `h` came
out 2372 m: resolved TKE decays to 0.14 at 760 m and then rises to 1.91 at 2011 m, internal
waves in the stable free atmosphere, and `argmax` landed there. The integral came out 1.212 and
the array share 1.52% where the northerly geometry predicts about 25%. Both windows carried it;
`results/pass9/refused/` keeps the evidence. The neutral half of this pass rests on run 4, not
on this target. Recomputed on the corrected `h` the next day (700 m by the surface-attached
definition; `results/pass10/`), the floor fell from 1.2e+04 to 1.05 and G1 through G3b all
passed, with an integral 1.265× its asymptote explained by `w̄ = −0.140 m/s`.

## Run 1: the convective seed at a 1.0 sim-h ceiling

Ran clean to 0.917 sim-h (the ceiling rounded down to a whole dump, a defect fixed the next
day), nearly seven eddy turnovers, but the stationarity battery could not score it: its 2.0 h
window reached step 0 where `u* = 0`, so every ratio was `inf` or `nan`. The seed was usable
(`ALLOW_INDETERMINATE=1` is the library's normal mode) and the ceiling became 2.0 sim-h for
every rung.

## Three bugs the smoke run found before production

Consecutive windows cannot share a boundary dump through the ring (traps §21b); killing the
shell is not stopping the LES (§21c); and `bin/seed_accept.sh` is an LES, not a CPU analysis,
so backgrounding it raced the case into the one-FastEddy-at-a-time refusal.

Removed from the tree on 2026-09-04 (at the `pre-cleanup-2026-09-04` tag): `bin/run_pass9.sh`,
`bin/pass9_accept.sh`, `bin/pass9_flat.sh`, `bin/handoff_accept.py`. Kept: `results/pass9/`,
`results/pass10/`, `results/streaming*.txt`, `results/lpdmonline_acceptance.txt`,
`results/kljun_adapter.json`, `results/kljun_parity.json`, `validation_pairs_30m/` (the two
windows of `case_2023111718`, the realisation-floor pair the emulator is scored against).
