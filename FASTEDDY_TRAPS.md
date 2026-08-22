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
