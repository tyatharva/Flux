# FastEddy: upstream release plus six patches

Flux runs NCAR's FastEddy **v5.0.1** with a small patch series on top. There is no fork to
clone. `fasteddy/fetch.sh` fetches the release, checks its commit, applies the patches, and
verifies every source file against `MANIFEST.sha256`. The result lands in
`FastEddy-model-5.0.1/`, the path every build script expects.

```bash
fasteddy/fetch.sh                      # -> ./FastEddy-model-5.0.1/
docker/run.sh ./docker/build_fasteddy.sh   # compile it in the CUDA 11.8 image
```

`Dockerfile.blackwell` runs the same script at image build time, so an image never depends
on a checkout that happens to be on the build host.

## The pin

| | |
|---|---|
| upstream | https://github.com/NCAR/FastEddy-model, tag `v5.0.1`, commit `e0cd2f3efcdff8ea543668228a983a95953478ca` |
| patches | six files in `patches/`, applied in order with `patch -p1`; series sha256 in `UPSTREAM` |
| result | byte-identical to the retired fork's `kegonsa` branch at `0ce48d5dff06`, minus `SRC/LPDM/CUDA` (moved to `lpdm/cuda/`). That commit is the `flux.fasteddy.revision` label on every published image |

## What each patch does

| patch | files | purpose | upstream? |
|---|---|---|---|
| 0001 `ioLPDMmode` | `SRC/IO/io.c`, `io.h`, `io_netcdf.c` | New optional input (default 0). At 1, a dump carries only the fields a backward LPDM reads, with the 3-D prognostics CF-packed to 16 bit. A 30-min window at 5 s falls from 61 GB to 15.4 GB on the 24 m grid. These dumps are not restartable: rho and pressure are absent by construction | no, project-specific |
| 0002 fix for 0001 | `io_netcdf.c` | rho must still be processed, and the packed range must avoid the fill value | with 0001 |
| 0003 `ioLPDMfullFrq` | `io.c`, `io.h`, `io_netcdf.c` | New optional input (default 0). At N, any dump whose absolute step is a multiple of N is written in full upstream form while every other dump stays lean, so a long window is chainable under a wall-time cap. Verified: the interleaved full dump has the same variables and dtypes as a mode-0 dump and differs from an independent run only at the ~1e-4 nondeterminism floor | no |
| 0004 moisture-off guard | `SRC/HYDRO_CORE/CUDA/cuda_largeScaleForcingsDevice.cu` | With `lsf_horMnSubTerms = 1` and `moistureSelector = 0`, upstream computes the qv slab mean and `Frhs_qv` unconditionally while the arrays are only allocated when moisture is on. Illegal memory access. The patch guards both | **yes, a real bug** |
| 0005 `lpdmOnlineSelector` | `SRC/FEMAIN/FastEddy.c`, `io.c`, `io.h`, `io_lpdmonline.c` (new), `io_netcdf.c` | In-process hand-off of a completed sampling window to the LPDM through host RAM instead of the filesystem. 0 = upstream behaviour; 1 = stage only (production); 2 = stage and also write netCDF (acceptance, one run producing both paths from the same bytes; they agree to 0.00e+00). `io.c` textually includes the new file, so the Makefile is untouched | no |
| 0006 CUDA 13 guard | `SRC/FECUDA/fecuda_Device.cu` | CUDA 13.0 removed four `cudaDeviceProp` members that a diagnostic printf reads. A `CUDART_VERSION` guard around that one place. No physics | **yes** |

Every default is 0, so an unmodified `.in`, including every NCAR tutorial case, runs through
the patched binary exactly as through the release.

No time-integration, grid or advection file is touched, so NCAR solver bug fixes merge
cleanly. Build customisation (architecture list, C++ dialect, include and library paths) is
applied as `make` command-line overrides in `docker/build_fasteddy.sh` and
`Dockerfile.blackwell`; nothing edits the Makefile.

## Why these are patches and not an adapter

Patches 0001, 0003 and 0005 hook FastEddy's IO registration and its main loop. There is no
plugin interface for that in v5.0.1, so the changes have to live in FastEddy's own source.
A patch series over a pinned release is the smallest thing that does it, and `git am` or
`patch -p1` of the series onto `e0cd2f3` reproduces the retired fork's tree exactly. The
series applies to upstream `develop` too: its two commits since v5.0.1 touch none of the
seven patched files.

## Files

```
fasteddy/
  UPSTREAM            the URL, tag, commit, and the series sha256 that fetch.sh enforces
  fetch.sh            fetch + patch + verify; verifies an existing tree instead of overwriting it
  patches/0001..0006  the series, in order
  MANIFEST.sha256     sha256 of every source file of the patched tree (254 files)
```

Regenerate the manifest after changing a patch:

```bash
FE_DIR=/tmp/fe fasteddy/fetch.sh && (cd /tmp/fe && find . -type f -print0 | sort -z | xargs -0 sha256sum) > fasteddy/MANIFEST.sha256
```
