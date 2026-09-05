# FastEddy and the patches

flux-kegonsa runs NCAR's FastEddy, a GPU-resident large-eddy simulation model, at release **v5.0.1**
with a six-patch series applied. There is no fork to clone. FastEddy is fully compressible,
fp32, with a third-order Runge–Kutta integrator. It is the offline generator of every training
target and is never part of inference.

## Fetching and building

```bash
fasteddy/fetch.sh                          # NCAR v5.0.1 + 6 patches -> ./FastEddy-model-5.0.1/
docker/run.sh ./docker/build_fasteddy.sh   # compile for the workstation's GPU, in flux-fasteddy:cuda118
```

`fasteddy/fetch.sh` clones the `v5.0.1` tag (commit `e0cd2f3efcdff8ea543668228a983a95953478ca`,
asserted), applies `fasteddy/patches/0001…0006` with `patch -p1` and verifies all 254 source
files against `fasteddy/MANIFEST.sha256`. If the destination already exists it verifies it
instead of overwriting, so a hand-edited tree is refused rather than built. The result is
byte-identical to the retired fork's `kegonsa` branch at `0ce48d5dff06`, the
`flux.fasteddy.revision` label on every published image, except for `SRC/LPDM/CUDA`, which is
now in this repository as `lpdm/cuda/`. `Dockerfile.blackwell` runs the same script at image
build time.

## The six patches

| patch | files | what it does | upstream? |
|---|---|---|---|
| 0001 `ioLPDMmode` | `SRC/IO/io.c`, `io.h`, `io_netcdf.c` | New optional input, default 0. At 1, a dump contains only the fields a backward LPDM reads, with the 3-D prognostics CF-packed to 16 bit. A 30-min window at 5 s falls from 61 GB to 15.4 GB on the 24 m grid. Not restartable, because rho and pressure are absent | no |
| 0002 | `io_netcdf.c` | rho must still be processed, and the packed range must avoid the fill value | with 0001 |
| 0003 `ioLPDMfullFrq` | `io.c`, `io.h`, `io_netcdf.c` | New optional input, default 0. At N, any dump whose absolute step is a multiple of N is written in full upstream form while the rest stay lean. A long window can then be chained and a field cache built from the first surviving dump. Verified: the interleaved full dump has the same variables and dtypes as a mode-0 dump and differs from an independent run only at the ~1e-4 nondeterminism floor | no |
| 0004 moisture-off guard | `SRC/HYDRO_CORE/CUDA/cuda_largeScaleForcingsDevice.cu` | `lsf_horMnSubTerms = 1` with `moistureSelector = 0` computes the qv slab mean and `Frhs_qv` unconditionally while those arrays are allocated only when moisture is on. The result is an illegal memory access. The patch guards both | **yes, a real bug** |
| 0005 `lpdmOnlineSelector` | `SRC/FEMAIN/FastEddy.c`, `io.c`, `io.h`, `io_lpdmonline.c` (new), `io_netcdf.c` | In-process hand-off of a completed sampling window to the LPDM through host RAM. 0 = upstream. 1 = stage only (production). 2 = stage and also write netCDF (acceptance). `io.c` textually includes the new file, so the Makefile is untouched | no |
| 0006 CUDA 13 guard | `SRC/FECUDA/fecuda_Device.cu` | CUDA 13.0 removed four `cudaDeviceProp` members that a diagnostic printf reads. A `CUDART_VERSION` guard around that one place. No physics | **yes** |

Every default is 0, so an unmodified `.in`, including every NCAR tutorial case, runs through
the patched binary exactly as through the release. No time-integration, grid or advection
file is touched, so NCAR solver fixes merge cleanly. Upstream `develop` is two commits past
v5.0.1 and touches none of the seven patched files.

Patches 0001, 0003 and 0005 hook FastEddy's IO registration and main loop. v5.0.1 has no plugin
interface for those, so they must be in FastEddy's own source. A patch series over a pinned
release is the smallest change that does it.

## The in-process hand-off (patch 0005)

`lpdmOnlineInit(Nx, Ny, Nz, dt)` runs after `timeInit` and after the grid, so it sees the
values the run will actually use rather than the ones the `.in` requested. It allocates
staging and publishes the snapshot format to `lpdmOnlineDir`. `lpdmOnlinePause(it)` passes a
completed window to the LPDM and blocks until it has integrated. It is placed after the output
block so the pause step's own snapshot is already staged, because the `σ_w` floor is built
from whole-window statistics. A pause is not a restart. There is no exit, no re-read and no
IO-registered field overwritten. `lpdmOnlineFinish()` tells the consumer the run is over.

Per case this persists 3.6 MB against the 19 GB the window used to write. The staged and
netCDF paths agree to 0.00e+00. Bit-identity is asserted, not a tolerance, because there is no
physics between them. Selector 2 costs +7% (IO is 0.0012 s of a 0.0159 s step) and exists so
the acceptance comparison can be made at all. Streaming cannot reduce the 12.0 GB fp16 field
cache. That cache is not buildup, it *is* the window, and the CPU integrator random-accesses
all of it, so host residency cannot fall below the cache. The ring holds a full window rather
than `t_back` because the `σ_w` floor needs whole-window statistics.

## Build details that matter

**Two images that differ in exactly one thing, the CUDA toolkit.** Ubuntu 22.04, OpenMPI 4.1.2,
NetCDF 4.8.1 and gcc 11.4 are held identical.

| | workstation `Dockerfile` | deployment `Dockerfile.blackwell` |
|---|---|---|
| CUDA | 11.8, `sm_89` cubin only | 13.0.1, real SASS `sm_75 … sm_120`, no PTX |
| code | toolchain only. FastEddy is compiled in the bind-mounted tree by `docker/build_fasteddy.sh` | code baked in. FastEddy fetched and compiled at image build |
| results | every published LES result came out of it. Frozen | the seed library and the corpus |

The 11.8 pin is a floor, superseded for deployment only. Its highest target is `sm_90`, so it
cannot reach a 5090 with SASS or PTX. The upgrade's effect on physics is measured at
0.97–1.12× the model's own run-to-run floor (`bin/test_toolkit_parity.py`, 200 steps).

**The Makefile is not edited.** It is written for NCAR's Casper/Derecho module environment,
where a wrapper sets every include and library path. Outside that environment, three lines are
wrong, and all three are plain `=` assignments, so `make` command-line overrides win.
`ARCH_CU_FLAGS` (`-arch=sm_80` → the target list). `OTHER_INCLUDES` (an undefined
`${NCAR_ROOT_MPI}` → explicit MPI, CUDA and NetCDF include paths). `TEST_LDFLAGS` (`-L.` → the
library paths). CUDA 13.0 needs a fourth. CCCL requires C++17 (`fecuda_PlugIns.cu:17` includes
`<cub/cub.cuh>` for the slab means the subsidence forcing uses), so `TEST_CU_CFLAGS` is
overridden to `-std=c++17` rather than suppressing the diagnostic. The architecture list is one
string used in both the `-dc` compile and the `-dlink` step, so the two agree by construction.
Builds are serial. The `FastEddy_devlink.o` rule has no prerequisites and begins with `rm -rf`.

**`nvcc -dlink` silently drops PTX.** FastEddy is built with separate compilation, so even
asking for `-gencode arch=compute_90,code=compute_90` puts PTX in the object files and none
in the executable. There is no JIT fallback. That is why the image has real SASS for seven
architectures. `cuobjdump --list-elf` and `--list-ptx` answer different questions. The image
asserts both, and asserts the opposite result on `liblpdm.so`, which is compiled whole-program
and does keep its PTX.

**Warnings are compared as a set**, not counted or suppressed. The build emits exactly nine
pre-existing upstream printf-format warnings, the same nine from the same files under CUDA
11.8 and 13.0 (`docker/expected_warnings.txt`). A tenth stops the build.

**MPI needs Fortran bindings even though FastEddy has no Fortran.** Its C code broadcasts with
`MPI_INTEGER` and `MPI_CHARACTER` (60 occurrences across 8 files). A build with
`--disable-mpi-fortran` aborts at startup with `MPI_ERR_TYPE`. Stock OpenMPI is used, not a
CUDA-aware build. FastEddy passes device pointers to `MPI_Isend`/`Irecv` under `hydroBCs = 2`,
which looks as if it needs CUDA awareness, so this was tested. At one rank the two builds differ
at the model's own nondeterminism floor (rho 3.70e-6 vs 3.81e-6, theta 7.0e-4 vs 6.4e-4).
`OMPI_MCA_plm=isolated` is set because the default `rsh` launcher aborts when it cannot find
`ssh`.

**Not bitwise reproducible.** Two runs on one GPU differ about 1e-4 relative in velocity and
about 7e-4 K in theta after 200 steps, from the block-retirement order of an `atomicAdd` in
the slab-mean reduction. Every "did my change matter?" test compares against that floor.
`FastEddy.c:113` seeds with `srand(mpi_rank_world + 12345)`, a fixed seed at one rank.

## The GPU LPDM (`lpdm/cuda/`)

`cuda_lpdmDevice.cu` and its header are a GPU backward Lagrangian dispersion model, validated
against the CPU integrator in `lpdm/model.py` by `bin/test_gpu_lpdm.py`. They were written in
the fork tree but were never part of FastEddy's build. `docker/build_lpdm.sh` compiles them
whole-program into `lib/liblpdm.so` (not tracked, because it is an architecture-specific build
product). `lpdm/gpu.py` drives it from Python with the same arrays `lpdm/fields.py` builds for
the CPU path, so the acceptance test compares integrators and not loaders. It is not on the
production path. See [limitations](../limitations-and-future-work.md).

## Two patches that should go upstream

0004 (the moisture-off subsidence crash) and 0006 (the CUDA 13 `cudaDeviceProp` guard) fix
defects in the release and contain no project-specific behaviour. Each is a self-contained
patch file under `fasteddy/patches/`.
