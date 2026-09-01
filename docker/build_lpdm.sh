#!/usr/bin/env bash
# Build the GPU-resident LPDM as a shared library, inside the CUDA image.
#
# ONE TRANSLATION UNIT, TWO FRONT-ENDS. The same cuda_lpdmDevice.cu is (a) linked into
# FastEddy for production, where the ring buffer is filled from the live device fields, and
# (b) built here as liblpdm.so and driven from Python for the acceptance suite. The second
# is what makes acceptance a test of the INTEGRATOR rather than of two different loaders:
# lpdm/gpu.py feeds it the arrays lpdm/fields.py already built, so both paths see
# bit-identical inputs and the only difference left is the integrator itself.
set -uo pipefail
FLUX_ROOT="${FLUX_ROOT:-/home/atyagi/Flux}"
ARCH="${SM_ARCH:-sm_89}"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "${FLUX_ROOT}":/work -w /work --entrypoint nvcc "${FLUX_IMAGE:-flux-fasteddy:cuda118}" \
  -O3 -std=c++11 -arch="${ARCH}" -Xcompiler -fPIC -shared \
  -I FastEddy-model-5.0.1/SRC/LPDM/CUDA \
  FastEddy-model-5.0.1/SRC/LPDM/CUDA/cuda_lpdmDevice.cu \
  -o lib/liblpdm.so
rc=$?
# ASSERT ON THE ARTIFACT, not the exit status (docs/FASTEDDY_TRAPS.md 12).
if [ ! -s "${FLUX_ROOT}/lib/liblpdm.so" ]; then
  echo "FATAL: nvcc produced no lib/liblpdm.so (rc=$rc)" >&2; exit 1
fi
echo "  built lib/liblpdm.so ($(stat -c%s "${FLUX_ROOT}/lib/liblpdm.so") bytes, ${ARCH})"
