# FastEddy traps, and one we created ourselves

Failure modes that produce a **normal exit, no warning, and wrong numbers**. Every one of
these cost real GPU time before it was found. They are recorded here, in *our* repository,
rather than reported upstream: several are consequences of our own fork, and the rest are
documented behaviours of v5.0.1 that we work around rather than ask NCAR to change.

The short forms live in `PROJECT_BRIEF.md`; this file carries the diagnosis and the evidence.

---

## 1. `ioLPDMmode` skipping `rho` silently produced `±inf` in five prognostic fields

**The one that got closest to poisoning a result.** Found 2026-08-20.

### Symptom

With the first cut of `ioLPDMmode` on the `kegonsa` fork, a 400-step smoke test:

- exited **0**
- printed **no** `****CORRUPTED***` banner
- wrote every file it was asked to write, with every expected variable present
- and had `u`, `v`, `w`, `theta`, `TKE_0` **entirely `+inf` / `-inf`**, with
  `scale_factor = inf` and `add_offset = inf` on the packed variables

`docker/k0k1_check.py` **passed it**, because the check is `k0/k1 < 1.0` and
`nan < 1.0` is `False`... but the ratio was `inf/inf = nan`, and the script's
"is it above the threshold" test therefore did not fire either. A NaN slips through
every comparison written as `x > threshold`.

### Cause

`SRC/IO/io_netcdf.c:486` contains

```c
#define NORHO  //if defined, it was in ioGetNetCDFinFileVars up above
```

with **no matching `#undef`**. The comment says it belongs to the *reader*, and it is easy
to read that as "this only affects the read path". It does not: the `#define` is still in
force further down the same translation unit, inside the **writer**. So in
`ioPutNetCDFoutFileVars`, `u`, `v`, `w`, `theta` and `TKE_0` are stored
flux-conservatively (`rho*u`, …) and each is divided by `ioBuffFieldRho` on the way out.

`ioBuffFieldRho` is populated by a `memcpy` that happens **when `rho` itself comes past in
the registered-variable list**. `rho` precedes the five prognostics in that list, so in
upstream operation the buffer is always primed before it is used.

Our first cut skipped `rho` in the same place it skipped everything else the LPDM does not
read — that is, it skipped `rho` during *processing*, not merely during *writing*. The
`memcpy` never ran, `ioBuffFieldRho` held uninitialised heap, and all five fields were
divided by garbage. On this machine the garbage was zeros, so the division produced `inf`
rather than a plausible-but-wrong number.

### Fix

Split the one predicate into two:

```c
static int lpdmSkipProcess(const char *n){   /* gather and un-flux-convert? */
   if(ioLPDMmode == 0){ return 0; }
   if(!strcmp(n, "rho")){ return 0; }        /* MUST be processed: primes ioBuffFieldRho */
   return !lpdmIsKept(n);
}
static int lpdmSkipWrite(const char *n){ ... }  /* ...but rho is still not WRITTEN */
```

`rho` is processed for its side effect and then discarded. Fork commit `692a3cd`.

### What this changes about how we work

1. **`inf` is not `CORRUPTED`.** FastEddy's corruption banner tests for NaN, and `inf` is
   not NaN. `x/0` where `x != 0` gives `inf`, and `inf` propagates through arithmetic
   without ever becoming NaN until an `inf - inf` or `inf/inf` appears. Our own checks now
   test `np.isfinite(...).all()`, never `np.isnan(...).any()`.
2. **A NaN passes every `>` test.** Any gate written as `if value > limit: fail` is a gate
   that a NaN walks straight through. `k0k1_check.py` was rewritten to require
   `np.isfinite(ratio) and ratio < limit`.
3. **A `#define` with no `#undef` is in force for the whole translation unit.** Do not
   trust a comment about where it "was" used. `io_netcdf.c` is ~1500 lines and `NORHO`
   spans two functions of it.
4. **Verify a new output path on field VALUES, not on file structure.** Every structural
   check — variable present, right dtype, right shape, right attributes — passed. Only
   reading the numbers caught it.

---

## 2. `****CORRUPTED***` still exits 0

Documented in `PROJECT_BRIEF.md`; repeated here because trap 1 is its sharper form. A fully-NaN
field returns exit status 0. `docker/check_run.sh` greps the log **and** inspects the newest
dump; the exit code is never trusted alone.

## 3. A missing restart file does not abort

FastEddy prints `Error: No such file or directory`, continues with x/y/z dimensions of 0,
and produces a run in which every cell of every field is NaN — exit 0. `run_case.sh` now
verifies `inPath`+`inFile` exists before spending GPU time.

## 4. The restart timestep is parsed from the FILENAME

`SRC/TIME_INTEGRATION/time_integration.c:104` runs `sscanf` on the characters after the
first `.` in `inFile`. A name like `restart.nc` leaves `simTime_itRestart`
**uninitialised**. Used deliberately: naming a restart `FE_RST.0` resets the step counter,
which is how `frqOutput` is kept dividing the absolute step across a restart.

## 5. `frqOutput` is tested against the ABSOLUTE step, and finer than `NtBatch` is ignored

`for(it = simTime_it; it < Nt; it += NtBatch)` with `if(it % frqOutput == 0)` *inside* the
loop (`SRC/FEMAIN/FastEddy.c:400,423`). Two consequences: a restart step that is not a
multiple of `frqOutput` writes exactly one dump; and `frqOutput < NtBatch` writes two.
For a sampling window, set `NtBatch = frqOutput`.

## 6. `Nt` is an absolute target step, not a step count

A restart from step 500 with `Nt = 500` performs zero timesteps, writes one dump, exits 0.

## 7. Restart overwrites `zPos`/`topoPos`/`z0m` from the restart file

`hydro_coreInit()` runs before `ioReadNetCDFinFileSingleTime()`, which walks the whole
registered variable list. Restarting a FLAT spin-up with a `topoFile` set leaves correct
terrain-following metrics but flat *diagnostic* coordinates in every later dump. Also the
only lever that gives v5.0.1 a spatially varying `z0m`, which is what
`bin/prep_stage6.py` uses.

## 8. `tBz` cannot exceed 64

CUDA's `maxThreadsDim[2]` is 64, so `1x1x128` is rejected even though `Nz + 6 = 128`
permits it arithmetically. FastEddy reports this one cleanly.

## 9. `tBx > 1` breaks memory coalescing

`i <- threadIdx.x` (`cuda_hydroCoreDevice.cu:648`) while `kStride = 1`. Cost measured at
**17%** for the `4x4x16` block we shipped through Stage 1. Use `1x2x64`.

## 10. `lsf_horMnSubTerms = 1` traps instantly when `moistureSelector = 0`

Found 2026-08-22, on the first attempt to spin up a dry boundary layer with subsidence.

### Symptom

`lsfSelector = 1`, `lsf_horMnSubTerms = 1`, `moistureSelector = 0`. The run writes its
step-0 dump, then dies on the first timestep:

```
GPUassert: an illegal memory access was encountered ../TIME_INTEGRATION/CUDA/cuda_timeIntDevice.cu 135
```

Unusually for this codebase, it is LOUD -- nonzero exit, no completion banner. That is luck,
not design: it is the same bug as trap 1 and differs only in what the bad pointer happened
to be.

### Cause

Two moisture accesses in the subsidence path are not guarded by `moistureSelector`, while
the memory they touch is allocated only when it is:

- `cuda_lsfSlabMeans()` launches the qv slab-mean kernel unconditionally over
  `&moistScalars_d[0]`;
- `cudaDevice_lsfRHS` writes `Frhs_qv[ijk]` unconditionally inside the
  `lsf_horMnSubTerms_d == 1` branch.

`cuda_moistureDeviceSetup()` allocates `moistScalars_d` and `moistScalarsFrhs_d` inside
`if (moistureSelector > 0)` (`cuda_moistureDevice.cu:41`). Dry, both pointers are
unassigned globals.

So **upstream v5.0.1 subsidence is only usable with moisture on**. This project runs dry by
decision (PROJECT_BRIEF.md), so the two had to be reconciled.

### Fix

Both guarded, fork commit on `kegonsa`. `lsf_numPhiVars` stays 5 and the qv profile slot
stays zeroed by the existing `cudaMemset`, so rho/u/v/theta subsidence is bit-for-bit
unchanged and a moist run is untouched.

### And a plan error it exposed

PLAN.md asked the smoke test to "confirm that `w` acquires the prescribed slab-mean
subsidence". **It never will, and it should not.** `cudaDevice_lsfRHS` adds the subsidence
tendency to `Frhs_HC[U_INDX]`, `[V_INDX]`, `[THETA_INDX]` and `Frhs_qv` -- there is no
`W_INDX` term. Subsidence here is a large-scale vertical ADVECTION tendency applied against
the slab-mean profile gradient, not a resolved vertical motion. Checking `w` would have
"failed" a correct implementation. The real test is differential on the theta profile:
`d<theta>/dt = -w_sub d<theta>/dz`.

---

## 11. A clip on one side of a ratio inflated `eps` a millionfold — and only the SLOWNESS showed

**Ours, not FastEddy's.** Cost about an hour of CPU and produced one withdrawn result.

### Symptom

The neutral well-mixed battery ran **39 minutes** where the convective one took 10, on
windows of identical size. No error, no warning, no obviously wrong number — the run would
have completed and reported a plausible verdict.

### Cause

The `sigma_w` floor scales `eps` with `sigma^2` so `T_L = 2 sigma^2/(C0 eps)` is preserved.
The ratio was written

```python
sig2 = np.maximum(sig2, 1e-6)                       # numerator CLIPPED
eps  = eps * (sig2 / np.maximum(sig2_raw, 1e-12))   # denominator NOT
```

Above the boundary layer the sub-grid TKE goes to zero. **Measured on a neutral window:
51.3% of cells carry `(2/3)e < 1e-6`**, and for those the ratio evaluates to `1e-6/1e-12`
= **1e6**. `eps` was inflated by that factor through the free atmosphere, `T_L` collapsed,
and every particle above `z_i` pinned at `dt_min` — wrong, and slow, and only the slow part
was visible. The median ratio came out **6.29 where it should be 1.00**.

### Fix

Floor both sides at the same value, so the ratio is exactly 1 wherever the clip is what is
being compared:

```python
eps = eps * np.maximum(sig2 / np.maximum(sig2_raw, 1e-6), 1.0)
```

Verified: ratio max **2.00** against 1e6, `T_L` 18.49 -> 14.77 s where the floor is active
(the intended effect), **0%** of particles at `dt_min`. Pinned by `bin/test_sgs_floor.py`,
which also asserts the retired denominator really did blow up.

### What this changes about how we work

**A guard on one side of a division is a bug on the other side.** Whenever a clipped
quantity appears in a ratio, the other operand needs the same clip or the ratio is
meaningless exactly where the clip binds — which is where the physics has gone quiet and
nobody is looking.

**And watch the wall clock as a diagnostic.** Two runs of the same size that differ 4x in
duration are telling you something about the numerics. This one produced no wrong number
anyone could see; the only signal it emitted was time.

---

## 12. Piping an analysis into `grep` hides its traceback, and the driver carries on

**Ours.** A duplicate keyword argument in `bin/stage5_footprint.py` reached a running
campaign and was launched **six times**, each after its own multi-minute field load.

### Cause

```bash
./docker/pyrun.sh bin/stage5_footprint.py ... 2>&1 | grep -vE 'batch [0-9]+/' > out.txt
```

Bash reports the exit status of the LAST command in a pipeline, which is `grep`. Python
died with a `SyntaxError`, the traceback went quietly into a redirected `.txt`, `grep`
succeeded, and the driver moved on. `set -o pipefail` is set in these scripts and still
does not help, because the redirect makes the failure invisible to the eye and the
subsequent stages never look at the file.

### Fix

Two, both cheap:

- **Check the PRODUCT, not the exit status.** Every analysis step now asserts its output
  JSON exists and non-empty, and dumps the tail of the log if not.
- **`bin/preflight.sh`** parses every python entry point and shell driver in about ten
  seconds, and the campaign refuses to start without it. Ten seconds would have caught
  this before any GPU time was spent.

### What this changes about how we work

**A gate that reads an exit status is reading whatever the shell last did, not whatever
you meant.** Assert on the artifact the step was supposed to produce.

---

## 13. An out-of-range parameter does NOT stop FastEddy — it silently uses the default

Found 2026-08-25 while building the sounding-forced corpus, before it cost anything.

### Symptom

There is none. A run with an out-of-range parameter launches, integrates, exits 0, and
produces perfectly finite fields — of a *different case* than the one in the `.in` file.

### Cause

`SRC/PARAMETERS/parameters.c:308-315`:

```c
/*Ensure the value is within the appropriate range limits*/
if((val < min) || (val > max)){
   entry->val->state = VALUE_STATE_INVALID;
   retCode = PARAM_ERROR_INVALID_FLOAT;
   numErrors++;
   printf("ERROR: parameter '%s' value %g is outside limits [%g,%g].\n",
           name, val, min, max);
}else { ... *var = val; }
```

Note what does **not** happen: `*var` is never assigned. It keeps whatever the caller
initialised it to a few lines earlier — the compiled-in default. And
`SRC/FEMAIN/FastEddy.c:96` calls

```c
errorCode = hydro_coreGetParams();
```

and never tests `errorCode`. Neither does any other `*GetParams()` call site. `numErrors`
is a file-static that nothing consults before the run starts.

So the entire consequence of an out-of-range value is **one `printf` into a log that this
project greps for `CORRUPTED`**, which it will not match.

### Why it matters here more than it looks

`stableGradient`, `stableGradient2` and `stableGradient3` are queried over
`[FLT_MIN, FLT_MAX]` (`hydro_core.c:642,646,650`) — i.e. **strictly positive**; zero is
rejected. Their compiled-in defaults are **0.1, 0.03 and 0.03 K/m**. So a per-case base
state that asked for a 0.4 K/km free-atmosphere lapse and passed `0.0` would run with
**0.1 K/m — a 250x stronger inversion** — and `z_i` would come out wrong in every window
with nothing anywhere to say so.

The same applies to `surflayer_wth` (range `[-5, 5]`, default 0.0 → a convective case
would run **neutral**), `surflayer_z0` (`[1e-12, 1]`, default 0.1) and `thetaAmplitude`
(`[0, 2]`).

### Fix

Guarantee the ranges upstream rather than hoping downstream.

- `bin/sounding_to_forcing.py` clips every gradient into `[1e-4, 0.5]` K/m and **raises**
  if a value is still out of range after rounding, or if the stable-layer bases come out
  unordered (`b1 <= b2 <= b3`) — unordered bases make the middle branch of
  `hydro_core.c:1786-1800` unreachable, which is its own silent wrong answer.
- `bin/test_sounding.py` re-checks all ten written parameters against the limits **read
  out of `hydro_core.c` itself**, not against remembered ones.
- Any campaign that greps a log should grep for `outside limits` alongside `CORRUPTED`.

### A numerical near-miss worth recording

The hydrostatic pressure integral carries `(1/g)*log(1 + g*dz/theta)`, which looks like it
would lose the neutral limit as `g -> 0` in fp32: at `g = FLT_MIN`, `1.0f + 1e-38f` rounds
to `1.0f`, `log` gives 0, and the segment drops out of the integral entirely.

It does not happen. The literal `1.0` in that expression is a **double**, so the whole
subexpression promotes and `log` is the double version — accurate to ~1e-13 relative even
at the 1e-4 K/m floor. **The positivity constraint is a parameter-range rule, not a
numerical one.** Recorded because the fp32 reasoning is the obvious one and it is wrong.


---

## 14. Combining traps 4 and 6 turns a zero-timestep echo into a full integration

Not a new FastEddy behaviour — a new way to get two documented ones wrong at once, found
2026-08-25 while writing Gate C2 for the seed library.

### Symptom

A "restart and re-dump, then diff" check reported **10 of 23 variables differing**, `u` by
**2.65 m/s** and `pressure` by 7.42. That is not fp32 roundoff and not the known
non-reproducibility floor (~1e-4 relative); it is a real integration. The obvious reading
is that restart is not bit-for-bit, which would contradict two earlier verifications at
two different grids.

### Cause

The test did this:

```bash
cp returned_dump.nc  $D/FE_RST.0          # <-- trap 4
sed -e 's|^Nt = .*|Nt = 20520|'  ...      # <-- trap 6
```

**Trap 4**: the restart step is `sscanf`'d from the characters after the first `.` in
`inFile`, so `FE_RST.0` sets `simTime_itRestart = 0` — the counter is *reset*, which is
normally the useful half of that trap.

**Trap 6**: `Nt` is an absolute target step. From a counter at 0, `Nt = 20520` is
**20520 real timesteps**, not the intended zero.

Each trap is documented above and each is individually harmless here. Together they turn
the intended no-op into a five-minute integration, and the diff faithfully reports it.

### Fix

Name the restart for the step it actually holds, and set `Nt` to that same step:

```bash
cp returned_dump.nc  $D/FE_RST.20520
sed -e 's|^Nt = .*|Nt = 20520|' -e 's|^NtBatch = .*|NtBatch = 20520|' \
    -e 's|^frqOutput = .*|frqOutput = 20520|'  ...
```

`frqOutput` must divide the step too, or **trap 5** eats the dump and the check reports
"produced nothing". `bin/c2_restart_check.sh` takes the step as an explicit argument rather
than inferring it, and re-scored the same returned artifact as **PASS, bit-for-bit, 0 of 23
variables differing**.

### What this changes about how we work

The existing B5 test got this right by using `Nt = 1` and comparing the **step-0** dump —
the restart read echoed back before any timestep touches it. That works too, and it is
worth knowing there are two correct formulations and one seductive wrong one. When a
restart check reports a difference far larger than the ~1e-4 non-reproducibility floor,
**suspect the test's step bookkeeping before suspecting the restart**.

---

## 15. A cold-started stable boundary layer collapses under a prescribed heat flux

Found 2026-08-25, by running it for 1.25 simulated hours and watching it happen.

### Symptom

The run completes, exits 0, greps clean for `CORRUPTED`, has finite fields everywhere and a
perfectly respectable `k0/k1` of 0.72. The stationarity gate would score it. And it is not a
boundary layer.

| | t = 0.25 h | t = 1.25 h |
|---|---|---|
| `u*` | 0.219 m/s | **0.043 m/s** |
| `z_i` | 209 m | **61 m** |
| `z/L` at the 10 m receptor | ~0.3 | **+34.8** |
| `dtheta/dz` at the first level | — | **2551 K/km** |

Above 66 m the mean wind sat at **exactly** the geostrophic 6.000 m/s — no Ekman turning at
all — with a gradient Richardson number of order **1e8**. The boundary layer had decoupled
from the flow entirely. Column-integrated "TKE" was *rising* the whole time, which is what
makes this look survivable: that variance is internal gravity waves, not turbulence, and it
sits above the level where the stress has already gone to zero.

### Cause

Runaway surface cooling, and it is a property of the boundary condition rather than of
FastEddy. `surflayerSelector = 1` prescribes a **fixed kinematic heat flux**. At `t = 0` a
cold start has no turbulence — only the `thetaPerturbation` seeding — so the prescribed
cooling has nothing to mix it away and builds a near-discontinuous inversion at the first
model level within minutes. That stratification then suppresses the very turbulence that
would have relieved it, and the feedback is positive: less mixing, stronger inversion, less
mixing. It is why GABLS1 (Beare et al. 2006) prescribes a surface **cooling rate** rather
than a flux.

**The forcing is not the fault.** With turbulence already present, `u* ~ 0.30` at
`G = 6 m/s` gives `L ~ 100 m` and `z/L ~ 0.10` at the receptor — weakly stable, and
perfectly sustainable. The same forcing is fine; the cold start is not.

### Fix

Not a change of boundary condition — this project's whole surface treatment is built on the
per-cell `htFlux` map, and `surflayer_tr` cannot carry it. **Remove the cold start
instead**: a stable rung runs its first segment NEUTRAL (`surflayer_wth = 0`), so real
turbulence exists before the cooling is switched on. `bin/make_seed_jobs.py` carries it as
`warmup_segments`, `jobs/run_seed.sh` applies it, and it is 0 for every other rung.

That is also how a stable boundary layer forms in nature — out of the evening transition of
a neutral or convective one — so the seed is more physical rather than less.

### What this changes about how we work

**`k0/k1`, finiteness and a clean `CORRUPTED` grep do not establish that a run contains
turbulence.** All three passed here. The diagnostics that caught it were `u*`, `z/L`, and
the mean wind profile — a wind that equals the geostrophic value *exactly* over most of the
column is the signature, because a coupled boundary layer always shows Ekman turning.

And **rising TKE is not evidence of turbulence in a stratified flow.** Check where it sits
relative to the stress: variance that peaks well above the level where `|tau|` has vanished,
at `Ri_g >> 0.25`, is wave motion.


---

## 16. z_i as "5% of the peak TKE" falls while the layer is growing

> **SECOND INSTANCE, MIRROR IMAGE, 2026-08-26 — and this one FAILED a run rather than
> killing one.** `seed_nbl-shallow_a000` (neutral, 3.0 sim-h, the first seed ever to reach
> its gate healthy) was rejected on `z_i` at **+11.67 %/h against a limit of 3**, while the
> other six limits passed with margins of 0.05-2.9% per window. Measured with
> `bin/zi_diagnose.py` (`results/nbl_a000_zi_diagnosis.txt`):
>
> | quantity, last 1.5 h | mean | trend |
> |---|---|---|
> | `z_i`, 5% of the RUNNING PEAK — **gated** | 364.4 m | **+11.67 %/h** |
> | `z_i`, fixed threshold 0.01 m²/s² | 389.3 m | +1.87 %/h |
> | peak resolved TKE — *the normaliser* | 0.3308 | **−15.67 %/h** |
> | `u*` | 0.2936 | −9.61 %/h |
>
> The gated depth is **−0.885 correlated with the peak it is normalised by**; the
> fixed-threshold depth is −0.379. The peak is falling because `u*` is falling, and `u*`
> falls for the first quarter of the 17.6 h inertial period — which PROJECT_BRIEF.md already
> records as *not* a stationarity failure ("Neutral stationarity is a statement about
> `u(z_m)/u*`, not about `u*`"). The layer's own depth is flat at 386-399 m from t = 1.0 h.
>
> It is also a **staircase**: the peak-normalised depth takes 4 distinct values over the
> last 2 h, and a straight line fitted through a staircase reports a trend whatever the
> layer does.
>
> **NO CHANGE HAS BEEN MADE TO THE GATE.** The seed FAILED and that is the recorded
> result. Whether `z_i` should be defined on a fixed threshold is a decision about what
> stationarity means for this project, and changing a gate immediately after it fails is
> how a gate stops meaning anything.

Found 2026-08-25, after it had already caused one run to be killed for the wrong reason.

### Symptom

During spin-up, `z_i` diagnosed as *the highest level where resolved TKE exceeds 5% of its
maximum* falls steadily — 154 -> 147 -> 120 -> 81 m over 20 simulated minutes — while `u*`
stays perfectly healthy at 0.26-0.40. It reads as a boundary layer collapsing.

### Cause

The threshold is **relative to a peak that is itself growing**. Measured on the same dumps:

| t (h) | peak TKE | TKE at 150 m | `z_i` (5% of peak) | `z_i` (absolute) |
|---|---|---|---|---|
| 0.08 | 0.0011 | 0.00005 | 154 m | 30 m |
| 0.17 | 0.0056 | 0.00016 | 147 m | 61 m |
| 0.25 | 0.0148 | 0.00033 | 120 m | 76 m |
| 0.33 | 0.0276 | 0.00040 | **81 m** | **81 m** |

The surface peak grew **25x** while the TKE at 150 m grew **8x**. Nothing shrank. Against a
fixed absolute threshold the layer is monotonically *deepening*, which is what a spinning-up
shear-driven boundary layer does.

### Where this matters

The same definition is used in `lpdm/les_stats.py:window_stats` (where it becomes the
corpus INPUT `h`), in `bin/seed_stationarity.py` (where its trend is one of the seven gated
quantities), and in `bin/pick_seed.py` via the achieved `z_i`.

**In a converged state it is fine** — the peak is steady, so the threshold is steady, and
that is the state all three of those actually consume. The gate scores the last 1.5 h of a
3 h run, by which time the peak has stopped growing. **It is only unreliable while the
turbulence is still organising**, which is exactly when someone is most likely to be
watching it to decide whether a run is healthy.

### Fix

Not a redefinition — changing it would break comparability with every prior pass, and it is
correct where it is used. **Watch `u*` and an absolute-threshold depth alongside it when
judging a spinning-up run**, and do not conclude anything from the relative `z_i` alone
before the peak has settled.

### What this changes about how we work

The judgement that killed a run here was "`u*` is healthy but `z_i` is collapsing, so the
state is wrong". Half of that was a real measurement and half was an artifact, and the two
were indistinguishable without looking at the profile. **When two diagnostics of the same
state disagree, plot the profile before believing either** — the scalar that looks alarming
is not automatically the informative one.

---

## 17. `surflayer_wth` in the .in is inert after a restart

> **THE MECHANISM IS STILL TRUE; ITS MAIN OPPORTUNITY IS GONE (2026-08-26).** Segment
> chaining is retired — a seed and a target case are each ONE continuous FastEddy
> invocation — so there is no longer a restart READ at every segment boundary waiting to
> overwrite `htFlux` with whatever the previous dump happened to hold. **The one restart
> that remains is seed -> target**, and `bin/prep_restart.py` / `bin/prep_stage6.py` write
> the surface into it deliberately, which is the same mechanism used as a LEVER rather than
> walked into. Keep reading: the trap is what a restart does, and the project still does
> one.

Found 2026-08-25, one segment after building a feature that depended on the opposite.

### Symptom

A segment's `.in` says `surflayer_wth = -0.012`. Every dump it writes carries
`htFlux = +0.000000`. The run is clean by every other measure and nothing in the log
mentions the flux at all.

### Cause

**Trap 7, pointed at the surface flux instead of at the terrain.** `htFlux` is
IO-registered (`hydro_core.c:1309`), `hydro_coreInit()` runs before the restart read, and
the restart read walks the whole registered variable list. So on any restart the file's
`htFlux` wins and the `.in`'s `surflayer_wth` is discarded.

This is already documented from the other direction — it is exactly the lever
`bin/prep_stage6.py` and `bin/case_surface.py` use to give FastEddy a spatially varying
surface, and PROJECT_BRIEF.md says so. What is easy to miss is the consequence for a CHAINED run:
**a scalar flux written into segment 1's `.in` propagates through every later segment
regardless of what their `.in` files say**, because each one restarts from the previous
one's dump.

That silently defeated a two-phase seed design whose entire point was to change the flux
between segments.

### Fix

Change the restart FILE, not the `.in`:

```python
with Dataset(restart_path, "a") as ds:
    ds["htFlux"][:] = np.full(shape, target, dtype="f4")
```

`jobs/run_seed.sh` does this before the first post-warm-up segment. It is idempotent, so it
also covers the resume path — which an `.in`-only version never could, because a resumed
chain reads its flux from a dump written long before.

### What this changes about how we work

**Assert on the flux the dump CARRIES, not the flux its `.in` requested.** Those two
differed for an entire 45-minute segment here with no error, no warning and no difference
in any other diagnostic. `jobs/run_seed.sh` now checks every segment's dump against the
value that segment was supposed to run with, and fails the job if they disagree.

The general form: **any FastEddy parameter that is also an IO-registered field is a
restart-file property, not an `.in` property.** `z0m`, `z0t`, `tskin`, `topoPos` and
`zPos` are in the same category.

---

## 18. Retiring the chain removed §17's mechanism and created four of our own

Chaining was retired on 2026-08-26: a seed and a target case are each ONE continuous
FastEddy invocation, and the only restart left in the project is seed -> target. That
deletes §17 **structurally** rather than by assertion — there is no longer a segment
boundary at which an IO-registered field can be silently inherited.

What it did not delete is the class of mistake. **Four defects were introduced or exposed
by the change, all in our own drivers, and none of them had ever been executed**: the
corpus dry run stops after stage 4, so stages 5-8 had not run since the change. Each was
found by walking the pass path deliberately rather than by a failure, and each would have
produced either a dead run or a plausible wrong number.

### 18a. `run_window.sh` deleted its own input restart

```bash
rm -f "$D"/window/* "$D"/FE_RST.*      # <- deletes the restart
cp -f "$RST" "$D/FE_RST.0" || die ...  # <- then copies it from itself
```

`bin/run_corpus_case.sh` stages the seed-derived restart at `$D/FE_RST.0` and passes that
path in as `$RST`. Before unchaining, `$RST` was the ADJUSTMENT RUN's final dump, in a
different directory, and could not collide with the destination. Afterwards it is the same
file, so the stale-clean removes it and the copy fails. The case dies before FastEddy is
launched — loud, but only if you run it.

**Fix:** the staging is a no-op when `readlink -f` says the two paths are the same file,
and the result is asserted with `[ -s ... ]` either way.

### 18b. `--strict-rel` failed the production configuration on half a millisecond

The guard exists to catch losing a DUMP off the head of the window — 5 s at the production
cadence — and it scored against a tolerance of `1e-6` s.

`dt` is carried in the `.in` to 8 decimals (`0.01461988`), so `frqOutput * dt` is 5 s only
to `1.04e-6`. Over the 840 dumps of a 4200 s case that accumulates:

| quantity | value |
|---|---|
| `fs.t[0]` (step 123120) | 1799.9996256 s |
| `t_last` (step 287280) | 4199.9991264 s |
| release period achieved | **1799.9995008 s** |
| deficit against 1800 | **4.99e-4 s** |
| one lost dump would be | 5 s = **10,016x** the deficit |

So the production case raised `ValueError` at **stage 7**, after 74 minutes of GPU, with
the fields already on disk, on a rounding artifact.

**Fix:** the tolerance is one tenth of the measured output interval — 0.5 s here — which
separates a missing dump from `dt` rounding by four orders of magnitude on both sides. The
achieved margin is now printed on SUCCESS too: this configuration is designed to sit at
exactly zero margin, and a guard that only speaks when it fires leaves no evidence of how
close a run came.

**The general form is worth more than the fix.** *A tolerance must be the size of the
failure it is looking for, not the size of the smallest number you can write down.* This
project already has the mirror-image lesson recorded in PROJECT_BRIEF.md — scoring a second moment
against an arbitrary 3e-2 instead of against its own sampling spread — and this is the same
error with the sign flipped.

### 18c. Two runs in one directory make a series that holds two states at the same time

FastEddy names a dump `<outFileBase>.<step>`. `bin/seed_stationarity.py` was handed a
DIRECTORY and globbed `*.[0-9]*`, sorting on the step number with no test that the dumps
came from one run. Four seed job output directories still held `FE_SMOKE.0` and
`FE_SMOKE.20520` from an August smoke test; the real run writes `FE_SEED.0` and
`FE_SEED.20520`. Sorting the union by step interleaves them into a "history" containing two
different states at t = 0 and two more at t = 300 s — and reports a stationarity verdict on
it.

`docker/turb_alive.py`, `docker/k0k1_check.py` and `bin/ozmidov.py` expanded "the siblings
in this directory" the same way. `k0k1_check.py`'s docstring already said *"or any sibling
of the same run"* and did not enforce it.

**Fix:** the smoke dumps are moved aside, mixed families are refused by name with both
listed, and every sibling expansion filters on the anchor dump's own base name.

### 18d. A seed that FAILED its gate was selectable, because the verdict is stamped last

`jobs/run_seed.sh` writes `achieved` into the return manifest as its LAST step. A job that
died after the gate and before that stamp leaves a manifest with **no verdict at all** —
and `bin/pick_seed.py` tested only `achieved.pass is False`, so such a job read as
*unjudged* rather than as *failed* and was ranked normally.

Live: `seed_sbl-weak_a030` — the weakly-stable seed whose collapse is the entire subject of
`STABLE_REGIME_RESULT.md`, whose `return/stationarity.json` says `pass: false`, whose
manifest says nothing, and which `pick_seed.py` returned as the best available seed in the
library.

**Fix:** the verdict is read from the gate's own JSON; the manifest is a convenience, not
the record. A return directory with neither a verdict nor a restart is reported as an
UNFINISHED JOB rather than falling back to its index entry, which would present a run that
got part way and stopped as one that never started.
