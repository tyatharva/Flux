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
echo "--- compiled GPU architectures ---"
cuobjdump --list-elf "${BIN}" | head -20
echo "--- PTX images ---"
cuobjdump --list-ptx "${BIN}" 2>&1 | head -5
# THIS BINARY HAS NO PTX AND THEREFORE NO JIT FALLBACK. Two independent reasons, and the
# second one means the first cannot simply be fixed by adding a -gencode:
#
#   1. -arch=sm_89 is shorthand for -gencode arch=compute_89,code=sm_89. It embeds a cubin
#      and NO PTX. jobs/run_seed.sh used to reassure the operator that a newer architecture
#      would "JIT from PTX, slower but correct" -- that was false for every binary this
#      script has ever produced.
#   2. MEASURED: nvcc -dlink DROPS every PTX image from the fatbin, silently. FastEddy is
#      built with separate compilation (-dc then -dlink), so even asking for
#      -gencode arch=compute_90,code=compute_90 puts PTX in the .o files and none of it in
#      the executable. docs/FASTEDDY_TRAPS.md 23.
#
# So this binary runs on ${SM} and on nothing else. That is fine for the workstation, which
# has one card. The DEPLOYABLE image (Dockerfile.blackwell) carries real SASS for seven
# architectures instead, and asserts each one.
_HAVE=$(cuobjdump --list-elf "${BIN}" | sed "s/.*\.\(sm_[0-9]*\)\.cubin/\1/" | sort -u | tr "\n" " ")
case " ${_HAVE} " in
  *" ${SM} "*) echo "  ${SM} present. NO PTX, so this binary runs on ${SM} ONLY.";;
  *) echo "  FATAL: asked for ${SM}, binary carries [${_HAVE}]" >&2; exit 1;;
esac
