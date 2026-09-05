# FastEddy traps, and the ones we made ourselves

Failure modes that produce a normal exit, no warning, and wrong numbers. Every one cost real
GPU time before it was found. They are numbered stably because code comments cite them
("traps §12"). The short forms are the [standing rules](standing-rules.md); this page carries
the diagnosis and the evidence. Several are consequences of this project's own patches or
drivers; the rest are documented behaviours of FastEddy v5.0.1 that are worked around rather
than reported.

## 1. `ioLPDMmode` skipping `rho` silently produced `±inf` in five prognostic fields

Found 2026-08-20; the one that got closest to poisoning a result. A 400-step smoke test with
the first cut of `ioLPDMmode` exited 0, printed no `****CORRUPTED***` banner, wrote every file
with every expected variable, and had `u`, `v`, `w`, `theta`, `TKE_0` entirely `±inf`, with
`scale_factor = inf` on the packed variables. `docker/k0k1_check.py` passed it: the ratio was
`inf/inf = nan`, and `nan < 1.0` is False, but so is `nan > threshold`.

**Cause.** `SRC/IO/io_netcdf.c:486` has `#define NORHO` with no matching `#undef`. The
comment says it belongs to the reader; the define is in force further down the same
translation unit, inside the writer, where `u`, `v`, `w`, `theta`, `TKE_0` are stored
flux-conservatively and divided by `ioBuffFieldRho` on the way out. That buffer is populated
by a `memcpy` when `rho` comes past in the registered-variable list. The first cut skipped
`rho` during *processing*, not only during writing, so the buffer held uninitialised heap.

**Fix.** Two predicates: `lpdmSkipProcess` never skips `rho` (it primes the buffer);
`lpdmSkipWrite` still does not write it. Patch 0002.

**What changed.** `inf` is not `CORRUPTED` (the banner tests for NaN). A NaN passes every `>`
test, so gates require `np.isfinite(ratio) and ratio < limit`. A `#define` with no `#undef` is
in force for the whole translation unit. Verify a new output path on field *values*, not file
structure: every structural check passed.

## 2. `****CORRUPTED***` still exits 0

A fully NaN field returns exit status 0. `docker/check_run.sh` greps the log and inspects the
newest dump; the exit code is never trusted alone.

## 3. A missing restart file does not abort

FastEddy prints `Error: No such file or directory`, continues with x/y/z dimensions of 0, and
produces a run in which every cell is NaN, exit 0. `run_case.sh` verifies `inPath + inFile`
exists before spending GPU time.

## 4. The restart timestep is parsed from the filename

`SRC/TIME_INTEGRATION/time_integration.c:104` runs `sscanf` on the characters after the first
`.` in `inFile`. `restart.nc` leaves `simTime_itRestart` uninitialised. Used deliberately:
naming a restart `FE_RST.0` resets the step counter, which keeps `frqOutput` dividing the
absolute step across a restart.

## 5. `frqOutput` is tested against the absolute step, and finer than `NtBatch` is ignored

`for(it = simTime_it; it < Nt; it += NtBatch)` with `if(it % frqOutput == 0)` inside
(`SRC/FEMAIN/FastEddy.c:400,423`). A restart step that is not a multiple of `frqOutput` writes
exactly one dump; `frqOutput < NtBatch` writes two. For a sampling window set
`NtBatch = frqOutput`.

## 6. `Nt` is an absolute target step, not a step count

A restart from step 500 with `Nt = 500` performs zero timesteps, writes one dump, exits 0.

## 7. Restart overwrites `zPos`, `topoPos`, `z0m` from the restart file

`hydro_coreInit()` runs before `ioReadNetCDFinFileSingleTime()`, which walks the whole
registered variable list. Restarting a flat spin-up with a `topoFile` set leaves correct
terrain-following metrics but flat *diagnostic* coordinates in every later dump. Also the only
lever that gives v5.0.1 a spatially varying `z0m`; `bin/prep_restart.py` uses it.

## 8. `tBz` cannot exceed 64

CUDA's `maxThreadsDim[2]` is 64, so `1x1x128` is rejected. FastEddy reports this one cleanly.

## 9. `tBx > 1` breaks memory coalescing

`i <- threadIdx.x` (`cuda_hydroCoreDevice.cu:648`) while `kStride = 1`. Measured at 17% for
the `4x4x16` block shipped through Stage 1 on the 16 m grid. Production uses `1x2x64`.

## 10. `lsf_horMnSubTerms = 1` traps instantly when `moistureSelector = 0`

Found 2026-08-22. `lsfSelector = 1`, `lsf_horMnSubTerms = 1`, `moistureSelector = 0`: the run
writes its step-0 dump and dies on the first timestep with `GPUassert: an illegal memory
access was encountered`. Loud, by luck: `cuda_lsfSlabMeans()` launches the qv slab-mean kernel
unconditionally over `moistScalars_d[0]`, and `cudaDevice_lsfRHS` writes `Frhs_qv[ijk]`
unconditionally, while `cuda_moistureDeviceSetup()` allocates both only when
`moistureSelector > 0`. Upstream subsidence is only usable with moisture on. Patch 0004 guards
both; rho, u, v and theta subsidence is bit-for-bit unchanged.

The plan had asked the smoke test to confirm that `w` acquires the prescribed subsidence. It
never will: `cudaDevice_lsfRHS` adds the tendency to U, V, THETA and qv; there is no `W_INDX`
term. Subsidence is a large-scale advection tendency against the slab-mean gradient. The real
test is `d<θ>/dt = −w_sub d<θ>/dz`.

## 11. A clip on one side of a ratio inflated `eps` a millionfold, and only the slowness showed

Ours. The neutral well-mixed battery ran 39 minutes where the convective one took 10, on
windows of identical size. The `σ_w` floor scales `eps` with `σ²` to preserve
`T_L = 2σ²/(C0 eps)`; the numerator was clipped at 1e-6 and the denominator at 1e-12. Above
the boundary layer 51.3% of cells carry `(2/3)e < 1e-6`, so the ratio evaluated to 1e6 there,
`T_L` collapsed, and every particle above `z_i` pinned at `dt_min`. Median ratio 6.29 where it
should be 1.00.

**Fix.** `eps = eps * max(sig2 / max(sig2_raw, 1e-6), 1.0)`: ratio max 2.00, `T_L` 18.49 →
14.77 s where the floor is active, 0% of particles at `dt_min`. Pinned by
`bin/test_sgs_floor.py`, which also asserts the retired denominator really did blow up.

A guard on one side of a division is a bug on the other side. And watch the wall clock: two
runs of the same size differing 4× in duration are telling you something.

## 12. Piping an analysis into `grep` hides its traceback, and the driver carries on

Ours. `pyrun.sh bin/stage5_footprint.py ... 2>&1 | grep -v ... > out.txt`: bash reports
grep's exit status, Python died with a `SyntaxError`, the traceback went into a redirected
`.txt`, and the driver launched the same broken analysis six times, each after its own
multi-minute field load. `set -o pipefail` does not help when nothing reads the file.

**Fix.** Every analysis step asserts its output JSON exists and is non-empty, and
`bin/preflight.sh` parses every Python entry point and shell driver before a campaign
starts. Assert on the artifact the step was supposed to produce.

## 13. An out-of-range parameter does not stop FastEddy; it silently uses the default

Found 2026-08-25, before it cost anything. `SRC/PARAMETERS/parameters.c:308-315` prints
`ERROR: parameter '%s' value %g is outside limits`, increments `numErrors`, and never assigns
`*var`, which keeps its compiled-in default. `SRC/FEMAIN/FastEddy.c:96` never tests the return
code of `hydro_coreGetParams()`; nothing consults `numErrors`. The whole consequence is one
`printf` into a log grepped for `CORRUPTED`.

`stableGradient{,2,3}` are queried over `[FLT_MIN, FLT_MAX]` (strictly positive) with defaults
0.1, 0.03, 0.03 K/m, so a base state that passed 0.0 for a 0.4 K/km lapse would run with a
250× stronger inversion. Same for `surflayer_wth` (range [−5, 5], default 0: a convective
case runs neutral), `surflayer_z0` ([1e-12, 1], default 0.1), `thetaAmplitude` ([0, 2]).

**Fix.** `bin/sounding_to_forcing.py` clips every gradient into [1e-4, 0.5] K/m and raises if
a value is still out of range or the stable-layer bases are unordered (unordered bases make
the middle branch of `hydro_core.c:1786-1800` unreachable). `bin/test_sounding.py` re-checks
all ten written parameters against limits read out of `hydro_core.c` itself. Grep logs for
`outside limits` beside `CORRUPTED`.

A numerical near-miss recorded because the fp32 reasoning is wrong: the hydrostatic integral
carries `(1/g)·log(1 + g·dz/theta)`, which looks like it loses the neutral limit at
`g = FLT_MIN` in fp32. The literal `1.0` is a double, the subexpression promotes, and it is
accurate to about 1e-13 relative even at the floor. The positivity constraint is a
parameter-range rule, not a numerical one.

## 14. Combining traps 4 and 6 turns a zero-timestep echo into a full integration

Found 2026-08-25 writing Gate C2. A "restart and re-dump, then diff" check reported 10 of 23
variables differing, `u` by 2.65 m/s: a real integration, not roundoff. The test copied the
returned dump to `FE_RST.0` (trap 4: the counter resets to 0) and set `Nt = 20520` (trap 6:
20520 real steps from a counter at 0). Each trap alone is harmless; together they turn a no-op
into a five-minute integration.

**Fix.** Name the restart for the step it holds (`FE_RST.20520`) and set `Nt`, `NtBatch` and
`frqOutput` to that step (trap 5 otherwise eats the dump). `bin/c2_restart_check.sh` takes the
step as an explicit argument and re-scored the same artifact as PASS, 0 of 23 differing. When
a restart check reports a difference far above the 1e-4 floor, suspect the test's step
bookkeeping before suspecting the restart.

## 15. A cold-started stable boundary layer collapses under a prescribed heat flux

Found 2026-08-25. The run completes, exits 0, greps clean, has finite fields and `k0/k1` 0.72,
and is not a boundary layer: `u*` 0.219 → 0.043 m/s over an hour, `z_i` 209 → 61 m, `z/L`
+34.8 at the receptor, 2551 K/km at the first level, the mean wind at exactly the geostrophic
6.000 m/s above 66 m with `Ri_g` of order 1e8. Column TKE was rising the whole time: internal
gravity waves above the level where the stress had already vanished.

**Cause.** Runaway surface cooling. `surflayerSelector = 1` prescribes a fixed kinematic heat
flux; a cold start has no turbulence to mix the cooling away, so a near-discontinuous inversion
forms at the first level within minutes and suppresses the turbulence that would relieve it.
GABLS1 prescribes a cooling rate for this reason. The forcing itself is sustainable
(`u* ≈ 0.30` at `G = 6 m/s` gives `z/L ≈ 0.10`).

**Fix.** Not a boundary-condition change (the per-cell `htFlux` map is the whole surface
treatment): run a stable rung's first segment neutral so turbulence exists before cooling is
switched on (`warmup_segments`). The stable rung was later retired anyway
([stable regime](../history/stable-regime.md)).

`k0/k1`, finiteness and a clean grep do not establish that a run contains turbulence. A wind
equal to the geostrophic value over most of the column is the signature of decoupling; rising
TKE in a stratified flow is not evidence of turbulence.

## 16. `z_i` as "5% of the peak TKE" moves with the peak

Two instances, mirror images. During spin-up the peak-relative depth falls (154 → 81 m over 20
minutes) while `u*` is healthy, because the threshold is relative to a peak that grew 25×
while TKE at 150 m grew 8×; against a fixed threshold the layer is monotonically deepening.
That reading killed a run for the wrong reason.

Then on 2026-08-26 it failed one: `seed_nbl-shallow_a000`, the first seed to reach its gate
healthy, was rejected on `z_i` at +11.67 %/h against a limit of 3 while the other six limits
passed. The gated depth was −0.885 correlated with the peak it was normalised by; the peak was
falling at −15.67 %/h because `u*` falls for the first quarter of the 17.6 h inertial period.
A fixed-threshold depth (0.01 m²/s²) said +1.87 %/h; the inversion base said +2.33 %/h; and
Kljun's `x_peak` and `x90`, gated ten times tighter, came in at −0.21 and −0.17 %/h. It was
also a staircase: four distinct levels over 2 h.

**Fix (the user's, same day).** The gated `z_i` is the fixed 0.01 m²/s² threshold
(`bin/seed_stationarity.py:ZI_ABS`); the peak-fraction depth is reported beside it; and
`bin/pick_seed.py` keeps matching on the peak fraction, because that is the currency
`window_stats` produces the corpus input `h` in. The same dumps re-scored: PASS at +1.87 %/h,
the library's first accepted seed. The gate prints the distinct-level count and span beside the
trend and flags any count ≤ 4. Checked across regimes: the fixed threshold runs 7–21% deeper,
growing with regime intensity, so the two definitions are not interchangeable.

## 17. `surflayer_wth` in the `.in` is inert after a restart

Trap 7 pointed at the surface flux. `htFlux` is IO-registered (`hydro_core.c:1309`), so on any
restart the file's `htFlux` wins and the `.in`'s `surflayer_wth` is discarded. A segment whose
`.in` said `−0.012` wrote `+0.000000` in every dump, clean by every other measure. In a chained
run a scalar flux from segment 1 propagated through every later segment. The mechanism is also
the lever (`bin/case_surface.py`, `bin/prep_restart.py`): change the restart *file*, not the
`.in`. Chaining was retired on 2026-08-26, so the only restart left is seed → target, where the
surface is written deliberately. Assert on the flux the dump carries, not the flux its `.in`
requested. Any parameter that is also an IO-registered field is a restart-file property.

## 18. Retiring the chain removed §17's mechanism and created four of our own

A seed and a case are each one continuous invocation. Four defects were introduced or exposed,
all in our drivers, none of which had ever been executed because the dry run stopped at stage 4.

- **18a.** `run_window.sh` deleted its own input restart: `rm -f $D/FE_RST.*` then
  `cp $RST $D/FE_RST.0`, where after unchaining `$RST` *was* `$D/FE_RST.0`. Staging is now a
  no-op when `readlink -f` says the two paths are one file, and the result is asserted.
- **18b.** `--strict-rel` failed the production configuration on half a millisecond. It exists
  to catch losing a 5 s dump and scored against 1e-6 s; `dt` carried to 8 decimals gives
  `frqOutput·dt = 5 s` only to 1.04e-6, which over 840 dumps accumulated to a 4.99e-4 s
  deficit, and a production case raised at stage 7 after 74 minutes of GPU. One lost dump is
  10,016× that deficit. The tolerance is now a tenth of the measured output interval, and the
  achieved margin is printed on success too.
- **18c.** Two runs in one directory make a series with two states at the same time.
  `seed_stationarity.py` globbed `*.[0-9]*` and sorted on the step with no test that the dumps
  came from one run; four directories still held `FE_SMOKE.*` beside `FE_SEED.*`.
  `turb_alive.py`, `k0k1_check.py` and `ozmidov.py` expanded siblings the same way. Mixed
  families are refused by name, and every sibling expansion filters on the anchor's base name.
- **18d.** A seed that failed its gate was selectable, because `achieved` is stamped last. A job
  that died between the gate and the stamp left a manifest with no verdict, and `pick_seed.py`
  tested only `achieved.pass is False`. Live instance: `seed_sbl-weak_a030`, whose collapse is
  the subject of the stable-regime page, was returned as the best available seed. The verdict
  is read from the gate's own JSON now; a return directory with neither verdict nor restart is
  an unfinished job.

## 19. A grid constant that is really a grid property: five instances in one day

Moving the receptor to 30 m and the grid to 24 m surfaced the same defect five times, each of
which would have produced a plausible number.

| file | the literal | what it should be | what it would have done |
|---|---|---|---|
| `bin/make_seed_jobs.py` | `Z0 = 0.1435` | the geometric-mean `z0` of the grid's own `z0m.npy` (0.0832 at 24 m, 42% smaller) | spun every seed up over the wrong surface |
| `bin/seed_stationarity.py` | `--zm 10.0 --k 2` defaults | the manifest's `gate.zm`, `gate.k` | scored Kljun at 10 m and read `σ_w` off 21.4 m |
| `bin/sounding_to_forcing.py` | `zm_recept = 10.0`, `z0 = 0.1435` | `<grid>/meta.npy`, `<grid>/z0m.npy` | `z_i,min = 10 z_m` at 100 m instead of 300 m |
| `bin/ozmidov.py` | `--dx 16.0 --receptor 10.0` | the manifest's `grid.dx`, `gate.zm` | scored `L_O/Δ` at the wrong height with the wrong `Δ` |
| `bin/seed_accept.sh` | `LAST = "$OUTBASE.$TOTAL"` | the newest dump on disk | with open-ended seeds `Nt` is a ceiling, so the file often does not exist |

The fix is not "pass a flag": read the value off the artifact the run will actually use. The
job manifest carries `gate: {zm, k}` and `grid: {dx, nz, zceiling, deform, domain_m,
surflayer_z0}`.

- **19b.** A `z/L` column is at *some* height, and the column name does not say which.
  `zm_over_L` in `results/selected_times*.tsv` was at 10 m; read as 30 m it understated every
  unstable case by a factor of three (−1.59 vs −4.76 for the most convective hour) and changed
  which target got chosen. Store the height with a normalised quantity.
- **19c.** A comparison harness that builds its own version of the thing compared.
  `bin/test_gpu_lpdm.py` released over `[t0 + t_back, ...]` while the driver releases over
  `[t_last − rel_seconds, t_last]`: both 76 releases, a different 300 s of a spinning-up
  layer, a 27% integral gap that looked exactly like a port bug. Derive shared inputs from one
  place, the production one.
- **19d.** Editing a shell driver while it is running corrupts it, silently and later. Bash
  reads a script incrementally by byte offset; insert lines above the current point during a
  90-minute child and the next command executed is the middle of a different one. It happened
  to `run_pass7.sh` twice. Never edit a shell script that is currently executing; kill the
  wrapper only, let the child finish its artifact, relaunch against a driver whose completed
  steps are no-ops. Python does not have this problem, which is a reason to keep logic in
  `bin/*.py` and the shell thin.
- **19e.** The mean of a ratio is not the ratio of the means: `L` wrong by 148×.
  `lpdm/les_stats.py` recovered the surface flux as `−(mean u*)³ (mean θ) (mean invOblen)/(κg)`;
  `invOblen ∝ 1/u*³`, so the mean is set by the cells with the smallest `u*` (domain mean
  −0.146, minimum −89 where `u*` is 0.036). Flux 43.09 vs 0.2906 K m/s, `L` −0.17 vs
  −25.45 m, `z/L` −166 vs −1.12; every downstream number finite and plausible. Hid for three
  near-neutral pairs. Fix: reconstruct the numerator per cell and average that; the corrected
  mean reproduces the grid's own `htFlux.npy` domain mean to four decimals.

## 20. The in-process hand-off, and what it cost before it worked

Added 2026-08-30 with `io_lpdmonline.c` (patch 0005).

- **20a.** A container gets 64 MB of `/dev/shm` whatever the host has; a 60-snapshot staging
  attempt died at 2.2 GB with `ENOSPC`, mid-window, after the GPU time was spent. All three
  container wrappers mount the host staging root at an identical path inside and out, because
  `lpdmOnlineDir` is written into the `.in` in one container and polled in another.
- **20b.** Backpressure must be counted, not flagged. The producer blocked on a `backlog` file
  nothing created; wired to a consumer flag, a dead consumer would have filled tmpfs hundreds of
  dumps later. The producer counts `snap.*.ok` in the directory itself.
- **20c.** `window_stats` mixed two surface-flux estimators inside one window: it used the
  `htFlux` variable when a dump carried it (full dumps under `ioLPDMfullFrq`) and derived the
  flux per cell otherwise. 2 of 12 sampled dumps took one branch. The two agree to 1.3e-7, so
  nothing published was visibly wrong; what was wrong is that the estimator depended on the
  output mode. Found by the ring, which carries no `htFlux`. Derived per cell for every dump now.
- **20d.** §19d at one remove: `bin/g30_bringup.sh` was blocked inside `docker/run_case.sh`,
  which was blocked inside `docker run`; adding eight lines to `run_case.sh` above that point
  made bash resume eight lines earlier in meaning and run the same `docker run` a second time.
  The rule is "do not edit any script in the call tree of a running job".
- **20e.** The final dump is not at step `Nt`. The loop overshoots to the first `NtBatch`
  multiple at or past `Nt` (a run asking for 60571 with `NtBatch = 2000` ends at 62000). Take
  the newest dump of the one run family.
- **20f.** A line of 256 characters or more in the `.in` segfaults FastEddy before it starts.
  `parameters.c:28` sets `MAXLEN 256` and reads with `fgets`; the continuation fragment has no
  `=`, `strchr` returns NULL, and `str_trim(NULL)` segfaults with a six-frame libc backtrace
  that mentions neither the file nor the line. The offending line was a 504-character `dt`
  comment in `runs/g30_base/base.in`. `docker/run_case.sh` refuses any `.in` with a line at or
  over 255 characters and says which. A blank line would crash the same way.
- **20g.** `0` cannot be the "disabled" sentinel for a step number, because 0 is a step. The
  first acceptance run paused at step 0 and waited for a resume marker nothing would write.
  Guarded on `> 0`.
- **20h.** `keep=True` turned the consumer's drain loop into an unbounded accumulator: deleting
  a snapshot after reading it was also what removed it from the ready list, so with `keep` the
  same 23 snapshots were re-read every 2 ms until the container was OOM-killed at exit 137 with
  no output (block-buffered stdout). When a flag changes whether a side effect happens, check
  whether that side effect was also carrying a control-flow invariant.

A snapshot the producer stages incompletely is indistinguishable downstream from a good one,
so `lpdmOnlineFlush` refuses a partial snapshot and the consumer checks `np.isfinite` on every
3-D field it reads.

## 21. The hand-off's second pass, and the two things streaming cost

- **21a.** The hand-off removed about 20 GB of disk and moved it to RAM, and every check
  passed. `drain_until_pause` returned every snapshot of the window as a list (541 × 36.5 MB =
  19.7 GB), and `FieldSet` retained it on top of its own 12.0 GB fp16 cache because
  `window_stats` opened every dump a second time. Peak about 32 GB, on a path whose whole
  argument is a rented box. Deleting the tmpfs file releases the *producer's* backpressure and
  nothing else. Fix: the two passes are fused into one (`WindowAccumulator`; `window_stats()`
  is a thin loop over it; `FieldSet.load()` releases each handle; `iter_until_pause()` yields).
  Peak RSS 1.754 → 0.937 GB on 24 snapshots, identical `h` and `u*`; `bin/test_streaming.py`
  asserts identity at exactly zero across 9 cache arrays and all 25 `window_stats` fields. A
  resource freed on one side of an interface is not freed on the other. What streaming cannot
  reach: the 12.0 GB field cache is the window, random-accessed by a CPU integrator.
- **21b.** Consecutive windows cannot share a boundary dump through the ring: the consumer
  deletes each snapshot as it reads it, so window 1 began one output interval late, and
  `--strict-rel` (with its §18b tolerance) refused a 195.0 s release period against 200 s.
  Windows are spaced `W_NT + frqOutput` apart, and the run is one output interval longer per
  extra window (7205 s, not 7200, for two). `N_WINDOWS = 1` is unchanged.
- **21c.** Killing the shell is not stopping the LES. The driver's `trap` killed the subshell
  and left the FastEddy container running, holding the GPU, so the next run was refused by the
  concurrency guard with an opaque container name. `docker/run_case.sh` names the container
  deterministically from the case directory and the trap removes it by name.

## 22. `h` was measured in the wrong fluid, and only the closure noticed

From the ninth pass's neutral target, `case_2023112120`. The resolved-TKE profile decays to
0.14 at 760 m and then a layer aloft is thirteen times stronger, essentially all resolved `w`:
internal-wave activity in the stable free atmosphere. `bl_depth`'s `argmax` picked 2011 m, the
downward search ran from there, and `h` came out 2372 m. Three guards were in place and none
fired: the `h ≥ 0.98 z[-1]` refusal (2372 m is under a 2960 m column), the decay-minimum bound
(it assumes the peak is in the boundary layer), and `turb_alive` and `k0/k1` (the LES was
fine). What fired was the closure's own health gate two stages later: `corpus_monitor.py` G1
reported the `σ_w` floor adding its most variance at 681 m where the LES already resolves
94.9%, a factor of 1.2e+04, and printed "Check st['h'] = 2372 m". The footprint's integral was
1.212 and its array share 1.52% where the geometry predicts about 25%.

A fix that bounds a search inherits every assumption the search's starting point makes. The
estimator now walks up from the ground on a smoothed profile to the surface-attached layer's
first minimum and requires the global peak to lie inside it. The test is "a second layer that
out-energises the first", not "the first local minimum" (using the first strict minimum moved
`h` on 15 of 47 stored profiles by up to 331 m on profiles with no wave layer). `h` is
bit-identical on all 47 stored profiles (`bin/test_bl_depth.py`, exact equality, 47 of 47), and
the broken profile now gives 448 m.

## 23. The CUDA 11.8 pin was a floor, not a ceiling, and moving it cost four things

Written 2026-08-31 for `Dockerfile.blackwell`. A 5090 is `sm_120`; nvcc 11.8's highest target
is `sm_90`, so the pin had to move for deployment only. The 11.8 image is unchanged and every
published LES result came out of it.

**It is a compiler problem, not a JIT problem.** 11.8 cannot emit `compute_120` PTX; its
newest virtual architecture is `compute_90`. And `-arch=sm_89` embeds a cubin and no PTX at
all, so the seed driver's reassurance that a newer card would "JIT from PTX, slower but
correct" was false for every binary this project had built. On a 5090 it would have been
`no kernel image is available for execution on the device`, after the banner had printed.

**And `nvcc -dlink` silently drops PTX**, measured on a three-line program:

| step | `--list-elf` | `--list-ptx` |
|---|---|---|
| `nvcc -gencode ...code=compute_90 -dc a.cu -o a.o` | sm_89, sm_120 | compute_90, compute_80 |
| `nvcc -gencode ...code=compute_90 a.o -dlink -o dl.o` | sm_89, sm_120 | **none** |
| the same flags, whole-program | sm_89, sm_120 | compute_90, compute_80 |

FastEddy uses separate compilation because `__device__` functions cross translation units, so
the PTX is in the objects and never reaches the binary, with no warning. The shipped image
carries both halves of the comparison: `liblpdm.so` is one translation unit built whole-program
and keeps its PTX; `FastEddy` lists none. The fallback is real SASS for seven architectures,
`sm_75 … sm_120` (83 s of build, 18 MB against 5 MB), which is the better answer anyway:
hydro-core kernels are register-heavy and JIT defers register allocation to load time.
`--list-elf` and `--list-ptx` are different questions; assert on both.

**CUDA 13.0's two frictions, both one line.** CCCL requires C++17 (`fecuda_PlugIns.cu:17`
pulls `<cub/cub.cuh>` for the slab means the subsidence forcing uses; the dialect is raised
from the make command line, not the diagnostic suppressed). Four `cudaDeviceProp` members
(`clockRate`, `deviceOverlap`, `computeMode`, `kernelExecTimeoutEnabled`) were removed; FastEddy
reads them in one diagnostic printf, patched under `#if CUDART_VERSION >= 13000` so one tree
builds under both toolkits. NetCDF, HDF5 and MPI needed nothing: none links against CUDA.

**The warning set is identical**: the same nine pre-existing upstream printf-format warnings
under 11.8 and 13.0, zero new, zero suppressed; the image build fails on a tenth.

**The physics did not move, scored against the model's own floor** (`bin/test_toolkit_parity.py`,
200 steps, `seed_nbl-deep_a015`, cold start, same physical GPU):

| | 11.8 vs 13.0 | 11.8 vs 11.8 | ratio |
|---|---|---|---|
| u | 2.4509e-4 | 2.5272e-4 | 0.97 |
| v | 2.4438e-4 | 2.3651e-4 | 1.03 |
| w | 4.0478e-4 | 3.8383e-4 | 1.05 |
| theta | 8.8501e-4 | 7.9346e-4 | 1.12 |

Against the shipped image's own baked binary: 1.07 / 1.06 / 1.00 / 1.11. The initial condition
is bit-identical across all three because `hydro_core.c:1881` draws the theta perturbation
with `rand()` seeded at `FastEddy.c:113` by `srand(mpi_rank_world + 12345)`, and both images
are Ubuntu 22.04 with the same glibc. Moving the distro with the toolkit would have made this a
measurement of `rand()`; that is why the distro is pinned.

**Three tooling traps of the same shape.** `cuobjdump --list-elf` and `--list-ptx` name images
differently (`sm_120.cubin` and `sm_120.ptx`, not `compute_120.ptx`), so a `compute_` pattern
finds nothing in a binary that has PTX. `[^\n]` is not a newline class in POSIX ERE (GNU grep
reads it as "not a backslash and not the letter n" and truncates every diagnostic at its
first `n`; the development host's ugrep accepts it, which is how it was found).
`read -r x < /proc/<pid>/cmdline` assigns and then returns 1 (NUL-separated, no trailing
delimiter), so a per-GPU mutex written `{ read ... } || continue` never refused anything; it is
`tr`-based now and `docker/run_case.sh` scans for its own FastEddy five seconds after launch.

Not claimed: bitwise reproducibility across architectures, and that 13.0 is required (12.8 also
targets `sm_120`; 13.0 was tried first because it matches the r580 driver generation on the
target boxes).
