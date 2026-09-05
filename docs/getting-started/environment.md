# Environment

Everything runs in Docker except the emulator, which runs in a conda environment on the host.
The host Python has no scipy, h5py or netCDF4 by design. Every analysis script runs in the image.

## Two images that differ in exactly one thing

| | `Dockerfile` (workstation) | `Dockerfile.blackwell` (deployment) |
|---|---|---|
| image | `flux-fasteddy:cuda118` | `flux-seeds:<flux-commit>-fe<fasteddy-id>` on GHCR |
| CUDA | 11.8, `sm_89` cubin only | 13.0.1, real SASS `sm_75, 80, 86, 89, 90, 100, 120`, no PTX |
| GPU | RTX 4080 (Ada, `sm_89`) | 8–16 × RTX 5090 (Blackwell, `sm_120`) |
| contents | toolchain only. FastEddy is compiled in the bind-mounted checkout | the whole repository, FastEddy fetched and compiled, the 30-seed library baked in |
| use | every LES development result, local analysis, the emulator's figures | the seed library and the corpus, on rented machines |

The rest is held identical so that the toolkit is the only variable: Ubuntu 22.04, OpenMPI
4.1.2, NetCDF 4.8.1, HDF5, gcc 11.4. The base image is pinned by digest. The upgrade changes
the physics by 0.97–1.12× the model's own run-to-run floor
([FastEddy and the patches](../les/fasteddy-and-patches.md)).

The pip layers are not version-pinned: numpy, scipy, netCDF4, h5py, xarray, pandas, dask,
scikit-image, mpi4py, eccodes ≥ 1.7, cfgrib ≥ 0.9.10, herbie-data ≥ 2024.3.0, pyproj ≥ 3.6,
s3fs, scikit-learn. They are the analysis stack. No LES physics passes through them, and
`IMAGE_PROVENANCE.txt` inside a deployment image records what was installed. Herbie has a
version floor because its archive source list changes as NOAA moves buckets. `eccodes` comes
from the pip wheel and the apt package is not installed. The two disagree on definitions paths.

## Building the workstation image

```bash
docker build -t flux-fasteddy:cuda118 .     # about 2 minutes of apt plus the pip layers
fasteddy/fetch.sh                           # NCAR v5.0.1 + the patches -> FastEddy-model-5.0.1/
docker/run.sh ./docker/build_fasteddy.sh    # compile for sm_89 in the bind-mounted tree
docker/build_lpdm.sh                        # optional: the GPU LPDM as lib/liblpdm.so
```

`docker/run.sh` runs a command in the container with the repository mounted at `/work`, as
your user (`--user $(id -u):$(id -g)`, `HOME=/tmp`), with `--gpus all`. `docker/pyrun.sh`
runs a project Python script the same way (`docker/pyrun.sh bin/test_bl_depth.py`).
`docker/run_case.sh` runs one FastEddy case with the concurrency guard, the `.in` line-length
check, the log grep and the `k0/k1` check. All three check `FLUX_NATIVE=1`, which the
deployment image sets, and then run directly instead of calling `docker run`.

Two environment settings in the image are required. `OMPI_MCA_plm=isolated` stops OpenMPI's
default launcher from looking for an ssh binary that a single-node container does not have.
`CUDA_HOME`, `MPI_ROOT` and `NETCDF_ROOT` supply the include and library paths that FastEddy's
Makefile expects NCAR's module wrapper to set.

The build asserts `ompi_info | grep "Fort mpif.h: yes"`. FastEddy broadcasts with
`MPI_INTEGER` and `MPI_CHARACTER`, so MPI without Fortran bindings aborts at startup.

## Building the deployment image

```bash
bin/fetch_assets.sh seeds      # the 30 restarts must be at seeds/*/return/seed_restart.nc
docker/build_image.sh          # -> flux-seeds:<flux-sha>-fe<fasteddy-id>, and :latest
```

The build uses the classic builder (`DOCKER_BUILDKIT=0`). This machine has BuildKit without the
buildx component, and the classic builder reads exactly one `.dockerignore`. That is why there
is one file for both Dockerfiles. The build refuses a dirty tree unless `FLUX_ALLOW_DIRTY=1`.
It fetches FastEddy, compiles it with `-std=c++17` for seven architectures and builds
`liblpdm.so`. It then asserts on the artifacts: the SASS list of both binaries, no PTX in
FastEddy and PTX present in `liblpdm.so`, the nine-warning baseline, 30 seed restarts of
identical size with their manifests and verdicts, the case path's inputs by name and a
2400 MB ceiling on the whole `/flux` tree. [Deployment](../les/deployment.md) has the Vast
procedure.

## The emulator's environment

```bash
conda env create -f ml/environment.yml     # the LESNet env: torch 2.5.1 + CUDA 11.8, Python 3.11, h5py, optuna
conda activate LESNet
```

No Docker image contains torch. Training ran on the RTX 4080. An FNO seed takes 7–13 min and a
CFM seed 19–25 min, one process at 100% utilisation and about 3 GB. Four concurrent runs give
1.1× the throughput of one.

## The host

The workstation runs Linux with an RTX 4080 and Docker with the NVIDIA container toolkit.
A host-side toolchain was installed alongside (CUDA toolkit, gcc-13 and gfortran-13 from the
toolchain PPA, OpenMPI, HDF5/NetCDF/PnetCDF, ParallelIO, an LLVM/Flang build). It was for other
atmospheric-model work on the same machine and no result here depends on it. The reproducible
environments are the two Dockerfiles. Two of those host choices are wrong for FastEddy and are
the reason the images exist. Ubuntu 22.04's `nvidia-cuda-toolkit` is CUDA 11.5, which predates
`sm_89`. nvcc 11.8 accepts gcc ≤ 11, not gcc-13.

## Tooling used by the repository

| tool | where | why |
|---|---|---|
| `bin/preflight.sh` | host + image | parses every Python entry point and shell driver, on the host and in the container (3.12 vs 3.10), before any campaign |
| `docker/check_run.sh` | image | greps a run's log for `CORRUPTED`, NaN, errors and the completion banner, and scores the newest dump's `k0/k1` |
| `docker/diag_near_surface.py`, `docker/k0k1_check.py`, `bin/k0k1_by_slope.py` | image | the `dt` accuracy check, domain-mean and slope-conditioned |
| `docker/turb_alive.py` | image | checks that there is any turbulence at all. Runs everywhere `k0/k1` runs |
| `docker/verify_image.sh` | deployment image | SASS against the attached cards, then a 200-step run |
| `bin/test_*.py` | image or LESNet | the gates. See [scripts](../reference/scripts.md) |
