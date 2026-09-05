#!/usr/bin/env bash
# Build the GPU-resident LPDM as a shared library, inside the CUDA image.
#
# ONE TRANSLATION UNIT. lpdm/cuda/cuda_lpdmDevice.cu is the GPU backward LPDM, validated
# against the CPU integrator in lpdm/model.py (bin/test_gpu_lpdm.py). It is built here as
# liblpdm.so and driven from Python: lpdm/gpu.py feeds it the arrays lpdm/fields.py already
# built, so both integrators see bit-identical inputs and the only difference left is the
# integrator itself. It is not linked into FastEddy and is not on the production path.
set -uo pipefail
FLUX_ROOT="${FLUX_ROOT:-/home/atyagi/Flux}"
ARCH="${SM_ARCH:-sm_89}"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "${FLUX_ROOT}":/work -w /work --entrypoint nvcc "${FLUX_IMAGE:-flux-fasteddy:cuda118}" \
  -O3 -std=c++11 -arch="${ARCH}" -Xcompiler -fPIC -shared \
  -I lpdm/cuda \
  lpdm/cuda/cuda_lpdmDevice.cu \
  -o lib/liblpdm.so
rc=$?
# ASSERT ON THE ARTIFACT, not the exit status (docs/reference/fasteddy-traps.md 12).
if [ ! -s "${FLUX_ROOT}/lib/liblpdm.so" ]; then
  echo "FATAL: nvcc produced no lib/liblpdm.so (rc=$rc)" >&2; exit 1
fi
echo "  built lib/liblpdm.so ($(stat -c%s "${FLUX_ROOT}/lib/liblpdm.so") bytes, ${ARCH})"
