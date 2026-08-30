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
> **A third depth agrees with the fixed-threshold one.** The height of maximum
> `dtheta/dz` below 1500 m -- the inversion base, which is what "the top of a capped
> neutral layer" physically means -- runs **336.5 m at +2.33 %/h**, also inside the limit.
> Two independent physical depths say the layer is steady; only the peak-normalised one
> says otherwise.
>
> **The corroboration is inside the gate's own output.** Kljun's `x_peak` and `x90` take
> `z_i` as an input and are gated **ten times tighter**, at 1.0 %/h. They came in at
> **-0.21** and **-0.17 %/h**, with `x_peak` spanning **38.0-38.4 m across the whole scored
> window against a 16 m raster cell**. A depth genuinely moving at 11.67 %/h cannot leave
> them there. (PROJECT_BRIEF.md records the same thing from the other side: at a 10 m receptor
> Kljun's only `z_i` channel is worth **1.0 percentage point** of array share over
> `h = 200-1200 m`.)
>
> **This CORRECTS the paragraph under "Where this matters" below**, which claimed the
> estimator "is fine in a converged state -- the peak is steady". It is not. A neutral
> Ekman layer has no steady `u*` on any affordable timescale, so it has no steady TKE peak
> either, and `TKE ~ u*^2` puts the predicted peak trend at **-19.2 %/h** against the -15.7
> measured. The sharp form: `bin/seed_stationarity.py` makes every limit a RATIO precisely
> so both terms ride the oscillation together -- and `z_i` is the one gated quantity that
> is not a ratio, so it inherits the oscillation anyway, through the threshold instead of
> through the value.
>
> ~~**NO CHANGE HAS BEEN MADE TO THE GATE.**~~ **RESOLVED THE SAME DAY, BY THE USER:** the
> gated `z_i` is now the **fixed 0.01 m2/s2 threshold** (`bin/seed_stationarity.py:ZI_ABS`),
> the peak-fraction depth is reported beside it, and `bin/pick_seed.py` keeps MATCHING on
> the peak fraction because that is the currency `lpdm/les_stats.py:window_stats` produces
> the corpus input `h` in. The gate measures a trend and needs a threshold that does not
> move; the matcher compares a value and needs the definition the corpus inputs use.
>
> **The same dumps were re-scored — no re-run, same seven limits, one estimator changed —
> and the seed PASSED**: `z_i` 389.3 m at **+1.87 %/h** against the 3.0 limit, and
> `x_peak`/`x90` moved from -0.21/-0.17 to **+0.09/+0.08 %/h**, both further inside a limit
> ten times tighter. `seed_nbl-shallow_a000` is the library's first accepted seed.
>
> **And the staircase is now printed, because the passing number needs it too.** That
> +1.87 %/h sits on **2 distinct model levels spanning 14 m**. A least-squares slope through
> two levels reports which two levels were visited as much as any drift, so the gate prints
> the level count and the span alongside the trend and flags any count <= 4. The span --
> 14 m on a 389 m depth over 1.5 h, 3.6% — is the evidence that the layer is steady; the
> trend alone is nearly uninformative in either direction, whichever side of the limit it
> falls on.
>
> **Checked across regimes before adopting**, because a fixed threshold is not scale-free:
>
> | rung | peak TKE | 0.01 as % of peak | `z_i` 5%-peak | `z_i` fixed | domain top |
> |---|---|---|---|---|---|
> | `nbl-shallow` | 0.331 | 3.02% | 364 m | 389 m | 2500 m |
> | neutral `g16_spin` | 0.487 | 2.05% | 414 m | 455 m | 2500 m |
> | `cbl-shallow` | 1.084 | 0.92% | 508 m | 598 m | 2500 m |
> | `cbl-deep` | 1.430 | 0.70% | 976 m | 1186 m | 2500 m |
>
> It runs **7-21% deeper** and the offset **grows with regime intensity**, so the two
> definitions are not interchangeable — which is exactly why both are recorded and why the
> matcher was left on the old one. It reaches the domain top in none of them.

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

~~**In a converged state it is fine** — the peak is steady, so the threshold is steady, and
that is the state all three of those actually consume. The gate scores the last 1.5 h of a
3 h run, by which time the peak has stopped growing. **It is only unreliable while the
turbulence is still organising.**~~ **WRONG — see the box at the top of this section.** The
peak is not steady in a converged neutral state; it rides the 17.6 h inertial oscillation
along with `u*`, and the estimator is unreliable there too, in the opposite direction.

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

---

## 19. A GRID CONSTANT THAT IS REALLY A GRID PROPERTY — five instances in one day

Moving the receptor to 30 m and the grid to 24 m surfaced the same defect five times, in
five different files, and **every instance would have produced a plausible number rather
than an error**. They are collected here because the shape is the point, not any one of
them:

> **A number that was correct for one grid, written as a literal, is a bug the moment a
> second grid exists — and the failure is silent, because the literal is still a number of
> the right kind.**

| file | the literal | what it should be | what it would have done |
|---|---|---|---|
| `bin/make_seed_jobs.py` | `Z0 = 0.1435` | geometric-mean `z0` of the grid's own `z0m.npy` (**0.0832** at 24 m, 42% smaller — the box now reaches the lake and coarser cells average tree with crop) | spun **every seed in the library** up over the wrong surface |
| `bin/seed_stationarity.py` | `--zm 10.0 --k 2` defaults | the manifest's `gate.zm` / `gate.k` | scored Kljun `x_peak`/`x90` at 10 m and read `sigma_w` off level k=2 (**21.4 m**) instead of the receptor |
| `bin/sounding_to_forcing.py` | `zm_recept = 10.0`, `z0 = 0.1435` | read off `<grid>/meta.npy` and `<grid>/z0m.npy` | `z_i_min = 10 z_m` would have been **100 m instead of 300 m**, admitting cases whose receptor is outside the surface layer |
| `bin/ozmidov.py` (via `seed_accept.sh`) | `--dx 16.0 --receptor 10.0` defaults | the manifest's `grid.dx` and `gate.zm` | scored `L_O/Delta` at the wrong height with the wrong `Delta` |
| `bin/seed_accept.sh` | `LAST = "$OUTBASE.$TOTAL"` | the newest dump on disk | with open-ended seeds, `Nt` is a **ceiling**, so this file often does not exist |

**The fix is the same in every case and it is not "pass a flag": read the value off the
ARTIFACT the run will actually use.** The grid directory already carries `meta.npy` and
`z0m.npy`; the job manifest now carries `gate: {zm, k}` and `grid: {dx, nz, zceiling,
deform, domain_m, surflayer_z0}`. A flag can be forgotten; a value read from the file the
case runs on cannot disagree with it.

### 19b. A z/L column is at SOME height, and the column name does not say which

`results/selected_times*.tsv` has a column `zm_over_L`. It is `z/L` **at 10 m**, because
`bin/select_times.py:classify(..., zm=10.0)` — and a scan that read it as a 30 m value
understated every unstable case by exactly a factor of three, putting the most convective
hour in the corpus at `30/L = -1.59` when it is really `-4.76`. It changed which candidate
looked most convective and therefore which target got chosen. Caught because the resulting
Kljun `x_peak` range (121-169 m) looked implausibly narrow and was re-derived; the
corrected range is 103-169 m.

**Whenever a normalised quantity is stored, store the height with it or put it in the
name.** This is the ratio rule's cousin: a ratio whose reference is unstated is a number
whose meaning depends on a file you have to go and read.

### 19c. A comparison harness that builds its own version of the thing being compared

`bin/test_gpu_lpdm.py` released the GPU ensemble over `[t0 + t_back, t0 + t_back +
rel_seconds]` while `lpdm/driver.py` releases over `[t_last - rel_seconds, t_last]`. Both
are 76 release times, both are "the same configuration" by inspection, and they are a
different 300 seconds of a spinning-up convective layer. It showed up as a **27% integral
gap** between CPU and GPU and looked exactly like a port bug — the thing the harness exists
to detect — until the release times were printed side by side.

**A harness that compares two implementations must derive their shared inputs from ONE
place, and preferably from the production one.** The fix was to rebuild the release times
using the driver's own rule rather than a re-implementation of it, which is the same
principle as "gates import the production function, they never reimplement it".

### 19d. Editing a shell driver WHILE IT IS RUNNING corrupts it, silently and later

`bash` does not read a script into memory. It reads it incrementally and remembers a BYTE
OFFSET, so a driver that is blocked for ninety minutes inside a seed run resumes by seeking
to an offset in a file that may no longer be the file it started. Insert twenty lines above
that point and the next command it executes is the middle of a different one.

This project's drivers are exactly the wrong shape for that: `bin/run_pass7.sh`,
`bin/run_campaign.sh` and `jobs/run_seed.sh` all block for an hour or more inside a single
child, and the temptation to improve the next stage while the current one runs is constant.
It happened here -- `run_pass7.sh` was edited twice while blocked inside a 90-minute seed.

**The rule: never edit a shell script that is currently executing.** The safe move, and the
one taken, is to KILL THE WRAPPER ONLY -- the long-running child is a separate process and
carries on, finishes its own artifact staging, and the wrapper is relaunched afterwards
against a driver whose completed steps are all no-ops. That last property is what makes it
safe, and it is why every stage in these drivers checks for its own artifact before doing
anything (`have_seed`, `.window_complete`, `pairs/$TAG.json`).

A python entry point does not have this problem -- CPython reads and compiles the whole
file at import -- which is another reason to put logic in `bin/*.py` and leave the shell
drivers as thin sequencing.

### 19e. The mean of a ratio is not the ratio of the means — L wrong by 148x

`lpdm/les_stats.py` needs the surface heat flux to form `L`. The fork's lean LPDM output
does not carry `htFlux`, so the code fell back to `invOblen`, which FastEddy writes as

    invOblen = -kappa g htFlux / (u*^3 theta)          (cuda_surfaceLayerDevice.cu:426)

and recovered the flux as `-(mean u*)^3 (mean theta) (MEAN invOblen)/(kappa g)`. That is the
mean of a ratio whose **denominator is u*^3**, so the average is set by whichever cells
happen to have the smallest friction velocity — and over a real surface with a strong flux
there are always some. Measured on `case_2023052519`: `invOblen` has a domain mean of
−0.146 and a **minimum of −89**, where `u*` is 0.036 against a mean of 0.457.

| | value |
|---|---|
| hfx, mean-of-the-ratio (the bug) | **43.09 K m/s** (window mean) |
| hfx, per-cell then averaged | **0.2906 K m/s** |
| the case grid's own domain-mean htFlux | **0.2906 K m/s** |
| `L` | **−0.17 m** vs **−25.45 m** |
| `z/L` at the receptor | **−166** vs **−1.12** |

`L` feeds Kljun's `x_peak` and `x90` (the reference the whole deciding test is scored
against), the `σ_w` floor's `ζ`, and the pair's own `L` input. Nothing complained. Every
downstream number was finite, of the right sign, and of a plausible order.

**It hid for three corpus pairs** because the 10 m cases were near-neutral: at
`w'θ' ≈ 0.015 K m/s` the two forms agree closely, and it took a case with a 333 W/m² flux
over a heterogeneous surface to separate them by two orders of magnitude.

**The fix is one line moved inside the mean** — `(-(u*_c^3) θ_c iL_c/(κg)).mean()` is
`htFlux_c.mean()` by construction. The general rule, which is the ratio rule again wearing
a different hat:

> **Never average a quantity whose denominator varies across the thing you are averaging
> over. Reconstruct the numerator per element and average THAT.**

And the confirmation that made it certain rather than probable: the corrected window mean
reproduces the case grid's own `htFlux.npy` domain mean to four decimals. **An independent
artifact agreeing to four digits is what turns a plausible fix into a verified one.**

## §20 — the in-process hand-off, and the three things it cost before it worked

Added 2026-08-30 with `SRC/IO/io_lpdmonline.c`. None of these produced an error message on
their own; all three produced a plausible wrong number or a silent stall, which is this
file's whole subject.

**20a. A container gets 64 MB of `/dev/shm`, whatever the host has.** The host tmpfs is
32 GB; a 60-snapshot staging attempt died at 2.2 GB with `ENOSPC`. Worse, the failure lands
wherever the *writer* happens to be — for the producer that is mid-window, hundreds of
dumps in, after the GPU time is already spent. All three container wrappers now mount the
host staging root explicitly, **at an identical path inside and out**, because
`lpdmOnlineDir` is written into the `.in` that FastEddy reads in one container and polled
by the analysis in another: a path that means two different things in two containers is
the same shape as a constant duplicated on both sides of an interface.

**20b. Backpressure must be counted, not flagged.** The first version had the producer
block while a `backlog` file existed — and nothing created it. Had it been wired to a flag
the consumer sets, the LES's memory bound would have depended on the consumer still being
alive to clear it, so a dead consumer would have filled tmpfs and killed the run several
hundred dumps later, far from the cause. The producer now counts `snap.*.ok` in the
directory itself and says what it is waiting for; a dead consumer stalls the LES loudly.

**20c. `window_stats` was mixing two surface-flux estimators inside one window, and had
been for as long as `ioLPDMfullFrq` has existed.** It used the `htFlux` variable when a
dump carried it and derived the flux per cell otherwise, on the premise (written in the
code) that `ioLPDMmode` never writes `htFlux`. That premise is stale: `ioLPDMfullFrq`
writes a FULL dump at every multiple of its setting, and a full dump carries `htFlux`. On
`case_2023052519`, 2 of 12 sampled dumps took one branch and 10 took the other, so the
window mean was a mean of neither. **The two agree to 1.3e-7, so nothing published is
visibly wrong — what was wrong is that the estimator depended on the OUTPUT MODE**, which
is the "a diagnostic is only as scale-free as its reference" rule with the reference being
an IO setting. Now derived per cell for every dump under every mode.

It was found by the ring, which carries no `htFlux` and therefore could not reproduce the
mixture — a second implementation of the same read disagreeing with the first is how a
branch nobody knew was there becomes visible. That is an argument for the indirection in
`lpdm/dumpsrc.py` rather than against it: one reader, two sources, and a test that demands
bit-identity between them.

**And the general form, which is the reason this section exists at all:** a snapshot the
producer stages incompletely is indistinguishable, downstream, from a good one — the
consumer would interpolate the previous step's field and produce a plausible footprint.
`lpdmOnlineFlush` therefore REFUSES a partial snapshot rather than writing what it has, and
the consumer checks `np.isfinite` on every 3-D field it reads. Both are cheap; neither
would exist if the format had been trusted.

**20d. §19d again, and this time to a script I was not editing.** §19d says: do not edit a
shell script while bash is executing it, because bash reads by byte offset and resuming
after an insertion lands mid-command. What happened here is the same mechanism at one
remove. `bin/g30_bringup.sh` was blocked inside `./docker/run_case.sh`, which was itself
blocked inside its `docker run`. Adding the tmpfs mount (§20a) inserted eight lines into
`docker/run_case.sh` **above** that point. When the container exited, bash resumed
`run_case.sh` at its saved offset — now eight lines earlier in meaning — and **ran the same
`docker run` a second time**, starting a fresh 15-minute cold start on top of the one that
had just finished.

The symptom was not an error. It was a log that restarted at step 10000 after reaching
58000, a `FE_BU.62000` next to a `FE_BU.0` written two seconds later, and one `run_case.sh`
pid owning two sequential containers. **The generalisation §19d was missing: the rule is
not "do not edit the script you launched", it is "do not edit ANY script in the call tree
of a running job."** A driver, its helper, and the helper's helper are all open files.

**20e. The final dump is not at step `Nt`.** FastEddy's loop is
`for(it = simTime_it; it < Nt; it = it + NtBatch)`, so it overshoots to the first `NtBatch`
multiple at or past `Nt` — a run asking for 60571 with `NtBatch = 2000` ends at **62000**.
Naming the expected artifact `FE_BU.$Nt` therefore names a file that never exists, and the
failure surfaces as "no developed state" *after* the GPU time has been spent. Take the
newest dump of the one run family instead, which is what `bin/seed_accept.sh` already had
to learn and what `bin/g30_bringup.sh` now does.

**20f. A LINE OF 256 CHARACTERS OR MORE IN THE `.in` SEGFAULTS FastEddy BEFORE IT STARTS.**
`parameters.c:28` sets `MAXLEN 256` and the parser reads with `fgets(strBuff, MAXLEN, ...)`,
so a longer line is split across two reads. The first piece parses; the **continuation is a
fragment with no `=`**, `strchr` returns NULL, and `parameters.c:126-133` calls
`str_trim(valueBuff)` on it:

```c
valueBuff = strchr(strBuff, '=');
if(valueBuff != NULL){ *valueBuff++ = 0; }
...
valueBuff = str_trim(valueBuff);      /* NULL when the line had no '=' */
```

What you see is `Signal: Segmentation fault (11)`, `Failing at address: (nil)`, and a
six-frame libc backtrace with no mention of the input file or the line — so the natural
reading is that the model or the build is broken. It is a **comment being too long**.

Cost here: one acceptance run, and it would have cost every run of the pass, because the
offending line was in `runs/g30_base/base.in` — a 504-character `dt` comment I had written
myself to record the measured accuracy boundary. The reasoning belongs in `PROJECT_BRIEF.md`; the
`.in` gets a pointer. **`docker/run_case.sh` now refuses any `.in` with a line at or over
255 characters before spending GPU time**, and says which line.

Note the same code path means a **blank line** would crash too, for the same reason — no
`#`, non-zero length, no `=`. Existing templates happen not to have one.

**20g. `0` cannot be the "disabled" sentinel for a step number, because 0 is a step.**
`lpdmOnlinePause1/2` default to 0 meaning off, and the guard read
`if((tstep != lpdmOnlinePause1) && (tstep != lpdmOnlinePause2)){ return 0; }` — so the
first acceptance run paused at step 0, after ONE snapshot, before any window could exist,
and waited for a resume marker no consumer had reason to write. Nothing was wrong except
that "off" and "pause at the very first output" were spelled the same way. Guarded on
`> 0` now, which costs nothing: a pause at step 0 can never be wanted.

**20h. `keep=True` turned the consumer's drain loop into an unbounded accumulator.**
`RingConsumer._ready()` listed every staged snapshot, and deleting one after reading it was
what removed it from the list. With `keep=True` — the acceptance comparison, where the
staged files ARE the artifact — nothing was deleted, so the same 23 snapshots were re-read
every 2 ms until the container was OOM-killed. **Exit 137, and because stdout was
redirected and therefore block-buffered, not one line of output to say where.** `_ready()`
now excludes what has already been read, which makes `keep` orthogonal to termination.

The generalisation: when a flag changes whether a side effect happens, check whether that
side effect was also carrying a control-flow invariant. Here "delete after read" was
silently doing double duty as "mark as consumed".

---

## §21 — the hand-off's second pass, and the two things streaming cost

Added 2026-08-30, while turning the hand-off from *staged* into *streamed*. §20 got the
snapshots out of the filesystem; these are what it took to get them out of RAM as well, and
both were found by a smoke run rather than by the production one they would have broken.

**21a. THE HAND-OFF REMOVED ~20 GB OF DISK AND MOVED IT TO RAM, and every check passed
while it did.** `RingConsumer.drain_until_pause` returned every snapshot of the window as a
list — 541 × 36.5 MB = **19.7 GB** — and `FieldSet` then retained that list for its own
lifetime *on top of* its own 12.0 GB fp16 cache, because `window_stats` opened every dump a
SECOND time after the cache was built. Peak ~32 GB.

Nothing complained. The staging directory was correctly bounded (the producer blocks at
`lpdmOnlineQueue`), the consumer correctly deleted each file after reading it, and the
footprints were correct. **Deleting the tmpfs file releases the PRODUCER's backpressure and
nothing else**; only dropping the Python references releases the consumer's own memory, and
the two are easy to conflate because the first is what the protocol talks about.

It matters because the whole argument for the in-process path is that a corpus can be
generated on a rented box, and 32 GB of host RAM is not something a rented single-GPU box
reliably has.

The fix is that the two passes are fused into one: `lpdm/les_stats.py:WindowAccumulator` is
the estimator as an accumulator, `window_stats()` is now a *thin loop over it* so there is
one implementation rather than two that agree, `FieldSet.load()` feeds it and calls
`MemDump.release()` on each handle, and `RingConsumer.iter_until_pause()` yields instead of
collecting. Measured in isolation on 24 snapshots: peak RSS **1.754 → 0.937 GB**, with both
routes returning identical `h` and `u*`. `bin/test_streaming.py` asserts identity at
**exactly zero** — there is no physics between two schedules over the same arithmetic — and
gets it across the 9 cache arrays, the time axis, and all 25 `window_stats` fields.

**The generalisation: a resource freed on one side of an interface is not freed on the
other.** The protocol's "delete after read" is about the producer's bound. The consumer's
bound is a separate statement and nothing was making it.

**And what streaming CANNOT reach is said rather than implied**: the 12.0 GB field cache is
not buildup, it IS the window, and `compute_footprint` is a CPU integrator that
random-accesses all of it. Host residency floors at the cache. Reaching one or two snapshots
needs the window in VRAM and the integration there — an INTEGRATOR change, not a plumbing
one, and a deferred item.

**21b. CONSECUTIVE WINDOWS CANNOT SHARE A BOUNDARY DUMP THROUGH THE RING, and `--strict-rel`
is what said so.** A two-window case naively spans `[A, A+W]` and `[A+W, A+2W]`, sharing the
dump at `A+W`. On the disk path that is harmless — nothing is deleted, both windows read it.
Through the ring it is impossible: the consumer deletes each snapshot as it reads it, so
window 0 consumes the boundary and window 1 begins one output interval late.

Measured on the first two-window ring run: window 1's release period came out **195.0 s
against the 200 s asked for**, and `--strict-rel` refused it. At production geometry that is
**every second window of the corpus**, failing after ~1 GPU-h per case is spent.

Note which check caught it. `FASTEDDY_TRAPS.md` §18b is about `--strict-rel` failing a
correct run on half a millisecond, and the fix there was to score against a tenth of the
measured output interval instead of against `1e-6`. That tolerance is what made this one a
clean refusal rather than either a silent 2.7% short averaging period or another false
alarm: the deficit is one whole output interval, 10× the bar.

Windows are now spaced `W_NT + frqOutput` apart and the run is one output interval longer
per extra window, so every window owns a full `W_NT` of fields and delivers exactly
`W_NT/frqOutput + 1` snapshots — **on both paths**, which also keeps the CPU-from-disk
versus from-ring acceptance a comparison of the same window rather than of two schedules.
`N_WINDOWS = 1` is arithmetically unchanged.

**21c. KILLING THE SHELL IS NOT STOPPING THE LES.** The ring runs `run_window.sh` in the
background while the LPDM consumes in the foreground. The driver's `trap` killed the
subshell — and left the FastEddy *container* running, holding the GPU. The next run was then
REFUSED by `docker/run_case.sh`'s concurrency guard, whose message says a run is already in
progress and gives a container name that means nothing to anyone reading it. Two minutes of
confusion, and on an unattended campaign it would have been the whole campaign.

`docker/run_case.sh` now names the container deterministically from the case directory and
the driver's trap removes it by name. **A process supervisor that does not know about
containers is not supervising the work**, only the shell that asked for it.
