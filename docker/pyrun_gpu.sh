#!/usr/bin/env bash
# As docker/pyrun.sh, but with the GPU attached -- for lpdm/gpu.py and the LPDM
# acceptance suite, which need a device. Kept separate so the ordinary analysis path
# cannot silently take a GPU that a FastEddy run is using.
#   docker/pyrun.sh docker/diag_near_surface.py runs/foo/output/FE_X.1000
#   docker/pyrun.sh - <<'PY' ... PY      (script on stdin)
# --entrypoint python3 skips the image's CUDA banner scripts.
set -uo pipefail
# See docker/run_case.sh: repo root is overridable for off-machine jobs.
FLUX_ROOT="${FLUX_ROOT:-/home/atyagi/Flux}"
# LPDM_WORKERS is forwarded so the campaign can size the fork pool per stage;
# without it every container would silently fall back to one core.
exec docker run --rm -i --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e LPDM_WORKERS="${LPDM_WORKERS:-1}" \
  -v ${FLUX_ROOT}:/work -w /work --entrypoint python3 flux-fasteddy:cuda118 "$@"
