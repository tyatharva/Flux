# Stage 0a — Gate Result: **PASS**

Date: 2026-08-17
Image: `flux-fasteddy:cuda118` · FastEddy `kegonsa` @ `e0cd2f3` (= upstream v5.0.1, zero divergence)

---

## Gate evidence

| Check | Result |
|---|---|
| Image builds | PASS (~2 min, apt-only) |
| MPI Fortran-binding assertion | PASS (`Fort mpif.h: yes (all)`) |
| FastEddy builds | PASS, exit 0 |
| Binary owned by host user, not root | PASS (`1000:1000`) |
| Compiled GPU arch | PASS — `cuobjdump` reports `FastEddy.1.sm_89.cubin` |
| Links correct libs | PASS — `libmpi.so.40`, `libcudart.so.11.0`, `libcurand.so.10`, `libnetcdf.so.19` |
| **Example03_SBL** runs to completion | **PASS**, exit 0, "Your FastEddy simulation is complete!" |
| **Example01_NBL** runs to completion | **PASS**, exit 0 — *the case that previously failed* |
| `too many resources requested for launch` | **absent** in both runs |
| `CRITICAL ERROR` / `*_FAIL` / `MPI_ERR` | 0 occurrences in both runs |
| Output structurally correct | PASS — dims + 32 vars, all dumps written |
| Fields finite, `rho > 0` | PASS |
| Stratification matches published spec | PASS (below) |

### Physical validation against published tutorial specifications

Not merely "it ran" — the initial states reproduce the documented case setups.

**Example03_SBL (GABLS1, Kosović & Curry 2000).** Spec: 265 K constant below 100 m,
then +0.01 K/m. Grid `dz = 3.125 m`, so k=32 is z=100 m.

```
theta(z) horiz-mean:  k=0: 265.00 K   k=30: 265.00 K   k=61: 265.92 K   k=121: 267.80 K
```

k=121 is z=378 m; spec predicts 265 + 0.01*(378-100) = **267.78 K**, observed **267.80 K**.

**Example01_NBL.** Spec: neutral 300 K well-mixed layer under a capping inversion.

```
theta(z) horiz-mean:  k=0: 300.00 K   k=14: 300.00 K   k=29: 300.00 K   k=57: 313.50 K
```

Constant 300.00 K through the mixed layer, inversion aloft, as documented.

---

## Measured performance

| Case | Grid | Cells | GPU peak | Time/step | Comp/step | IO per dump |
|---|---|---|---|---|---|---|
| Example03_SBL | 128x126x122 | 1.97 M | 1,764 MiB | 0.0228 s | 0.0186 s | 0.41-0.45 s |
| Example01_NBL | 640x634x58 | 23.53 M | 6,378 MiB | 0.2717 s | 0.2188 s | 5.29 s |

Per-cell compute cost agrees to **1.7 %** across a 12x range in grid size
(9.453 vs 9.297 ns/cell/step), so extrapolation is well founded.

### Memory headroom for Stage 1

NBL at 23.53 M cells used **6.4 GiB of 16 GiB**. The production grid is 7.73 M
cells — **3x smaller**. Memory is not a constraint for Stage 1.

---

## Flag: projected production run time exceeds docs/PLAN.md's threshold

At 9.37 ns/cell/step, the production grid `434x146x122` (7,730,408 cells) costs
**0.0725 s/step**. Taking `dt = 0.0267 s` (scaling NBL's dt/dx to 10 m):

| Simulated | Steps | Wall clock |
|---|---|---|
| 30 min | 67,500 | **1.36 h** |
| 2 h | 270,000 | **5.44 h** |
| 3 h | 405,000 | **8.15 h** |

docs/PLAN.md Stage 1: *"If a single run exceeds ~4 hours, stop and flag it — the corpus
arithmetic needs revisiting."* A 3 h run is **8.15 h**. Flagged. Stage 1 measures
this directly rather than by extrapolation; treat these as the prior.

## Flag: output volume, given no z-subset option exists

FastEddy has **no vertical-subset output** (`io.c` reads only `ioOutputMode`,
`inFile`, `outFileBase`, `frqOutput`, `towerIOSelector`). Measured dump size is
**76.6 B/cell**, so a production dump is **0.59 GB**:

| Window at 5 s cadence | Dumps | Volume |
|---|---|---|
| 30 min | 360 | **213 GB** |
| 3 h | 2,160 | **1.28 TB** |

PROJECT_BRIEF.md budgets 11-33 GB/run assuming a `z < 400 m` subset. That subset does not
exist, so the real figure is ~10x higher. Resolution belongs to Stage 3: either add
a k-range limit to `SRC/IO/io_binary.c` (a genuine `kegonsa` fork commit) or reduce
the written variable set.

---

## Corrections to prior assumptions

1. **CUDA-aware MPI is NOT required.** FastEddy passes raw device pointers to
   `MPI_Isend`/`MPI_Irecv` under `hydroBCs==2` with no `numProcs==1` bypass, which
   looks like it demands a CUDA-aware MPI. Tested rather than assumed: a
   from-source OpenMPI 4.1.6 `--with-cuda` was built and compared against stock
   Ubuntu OpenMPI 4.1.2 (`mpi_built_with_cuda_support:false`). Field differences at
   t=200 were **statistically indistinguishable from the model's own run-to-run
   nondeterminism** (rho 3.70e-6 vs 3.81e-6 baseline; theta 7.0e-4 vs 6.4e-4).
   inst.txt's stock `openmpi-bin libopenmpi-dev` is correct and much cheaper.

2. **MPI Fortran bindings ARE required**, despite FastEddy having no Fortran source.
   Its C code broadcasts with `MPI_INTEGER`/`MPI_CHARACTER` — Fortran datatype
   handles — in 60 places across 8 files. Building MPI with `--disable-mpi-fortran`
   makes the model abort with `MPI_ERR_TYPE: invalid datatype`. Arguably an upstream
   bug (`MPI_INT`/`MPI_CHAR` are the correct C spellings); stock Ubuntu OpenMPI
   carries Fortran bindings, so inst.txt's recipe avoids it for free.

3. **FastEddy is not bitwise reproducible.** Identical binary, identical image, two
   runs: fields differ by ~1e-4 relative in velocity after 200 steps. Relevant to
   the LES corpus — a run cannot be reproduced exactly, only statistically.

4. **Thread blocks.** Every shipped tutorial uses 256 threads/block (Examples 01-04
   `1x4x64`, 05-10 `1x8x32`). PROJECT_BRIEF.md's `4x4x16` is also 256 and equally valid.
   The divisibility rule is enforced on **per-rank, halo-inclusive** extents
   (`SRC/GRID/grid.c:222-240`); Nz is never decomposed.

5. **Precision (Stage 0b, answered).** Hardwired fp32: bare `float` on every
   prognostic field, `MPI_FLOAT` in halo exchange, `NC_FLOAT` in the writer. No
   build switch, no typedef. Confirmed in output: all fields `NC_FLOAT`.

## inst.txt reconciliation

| inst.txt line | Verdict |
|---|---|
| `openmpi-bin libopenmpi-dev` | **Correct** — supplies the required Fortran bindings; CUDA-awareness proven unnecessary |
| `libnetcdf-dev`, `libhdf5-openmpi-dev` | **Correct** — NetCDF-C is required (`NC_NETCDF4` forced at `io_netcdf.c:611`) |
| `libnetcdff-dev`, `libpnetcdf-dev`, ParallelIO | **Not used by FastEddy** — links only `-lm -lmpi -lstdc++ -lcurand -lcudart -lnetcdf`. Installed anyway to stay faithful to working notes; belongs to the MPAS sections |
| `apt install nvidia-cuda-toolkit` | **Wrong for this GPU** — Ubuntu 22.04 ships CUDA 11.5, which predates sm_89. The 11.8 base image is required |
| `gcc-13 / gfortran-13` | **Wrong for nvcc** — CUDA 11.8 supports gcc <= 11; Ubuntu's default gcc-11 is used |
