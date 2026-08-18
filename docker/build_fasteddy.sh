#!/usr/bin/env bash
# Build FastEddy inside the flux-fasteddy container, in the bind-mounted host tree.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# FastEddy's SRC/FEMAIN/Makefile is written for NCAR's Casper/Derecho module
# environment, where the `ncarcompilers` wrapper silently injects every include
# and library path. Off that environment three lines are wrong:
#
#   Makefile:39  ARCH_CU_FLAGS  = -arch=sm_80              # A100 only; we are Ada/sm_89
#   Makefile:46  OTHER_INCLUDES = -I${NCAR_ROOT_MPI}/include  # var undefined -> "-I/include"
#   Makefile:48  TEST_LDFLAGS   = -L.                      # no CUDA/NetCDF/MPI lib paths
#
# All three are plain `=` assignments, so make command-line overrides win. That
# lets us fix the build WITHOUT editing the fork, keeping its diff against
# upstream v5.0.1 empty (see FASTEDDY_VERSION.txt).
#
# Note we do NOT add -O2 to host code, even though the Makefile sets no -O flag
# at all (nvcc still defaults device code to -O3). Matching NCAR's build exactly
# means any failure is ours, not an optimization artifact. Revisit once the
# Stage 0a gate is green.
set -euo pipefail

FE_DIR="${FE_DIR:-/work/FastEddy-model-5.0.1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MPI_ROOT="${MPI_ROOT:-/opt/openmpi}"
NETCDF_ROOT="${NETCDF_ROOT:-/usr}"
SM="${SM:-sm_89}"
JOBS="${JOBS:-$(nproc)}"

echo "=== FastEddy build ==="
echo "  tree      : ${FE_DIR}"
echo "  arch      : ${SM}"
echo "  cuda      : ${CUDA_HOME}"
echo "  mpi       : ${MPI_ROOT}"
echo "  netcdf    : ${NETCDF_ROOT}"
echo

MAKE_ARGS=(
  TEST_CC=mpicc
  "ARCH_CU_FLAGS=-arch=${SM}"
  "OTHER_INCLUDES=-I${MPI_ROOT}/include -I${CUDA_HOME}/include -I${NETCDF_ROOT}/include"
  "TEST_LDFLAGS=-L. -L${CUDA_HOME}/lib64 -L${MPI_ROOT}/lib -L${NETCDF_ROOT}/lib/x86_64-linux-gnu"
)

if [ "${CLEAN:-1}" = "1" ]; then
  make -C "${FE_DIR}/SRC/FEMAIN" "${MAKE_ARGS[@]}" clean || true
fi

# NOTE: intentionally serial by default. The Makefile's FastEddy_devlink.o rule
# has no prerequisites and begins with `rm -rf`, which races under -j.
if [ "${JOBS}" != "1" ] && [ "${ALLOW_PARALLEL:-0}" = "1" ]; then
  make -C "${FE_DIR}/SRC/FEMAIN" -j"${JOBS}" "${MAKE_ARGS[@]}"
else
  make -C "${FE_DIR}/SRC/FEMAIN" "${MAKE_ARGS[@]}"
fi

BIN="${FE_DIR}/SRC/FEMAIN/FastEddy"
echo
echo "=== verify ==="
ls -l "${BIN}"
echo "--- linked libraries ---"
ldd "${BIN}" | grep -E "mpi|cud|netcdf|hdf5" || true
echo "--- compiled GPU architectures (must include sm_89) ---"
cuobjdump --list-elf "${BIN}" | head -20
