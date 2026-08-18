#!/usr/bin/env bash
# Run a project python script inside the container (host lacks netCDF4).
#   docker/pyrun.sh docker/diag_near_surface.py runs/foo/output/FE_X.1000
#   docker/pyrun.sh - <<'PY' ... PY      (script on stdin)
# --entrypoint python3 skips the image's CUDA banner scripts.
set -uo pipefail
exec docker run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /home/atyagi/Flux:/work -w /work --entrypoint python3 flux-fasteddy:cuda118 "$@"
