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
