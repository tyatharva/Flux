#!/usr/bin/env bash
# Run a project python script inside the container (host lacks netCDF4).
#   docker/pyrun.sh docker/diag_near_surface.py runs/foo/output/FE_X.1000
#   docker/pyrun.sh - <<'PY' ... PY      (script on stdin)
# --entrypoint python3 skips the image's CUDA banner scripts.
set -uo pipefail
# See docker/run_case.sh: repo root is overridable for off-machine jobs.
FLUX_ROOT="${FLUX_ROOT:-/home/atyagi/Flux}"
# LPDM_WORKERS is forwarded so the campaign can size the fork pool per stage;
# without it every container would silently fall back to one core.
# THE IN-PROCESS LPDM STAGING DIRECTORY, MOUNTED AT AN IDENTICAL PATH ON BOTH SIDES.
# Docker gives a container 64 MB of /dev/shm by default -- measured, after a 60-snapshot
# staging attempt died with ENOSPC at 2.2 GB -- so the host tmpfs has to be mounted
# explicitly. The path is the SAME inside and out on purpose: lpdmOnlineDir is written
# into the .in that FastEddy reads in one container and polled by the analysis in another,
# and a path that means two different things in two containers is the kind of quiet
# mismatch this project keeps paying for. Created on demand; a no-op when unused.
FLUX_RINGROOT="${FLUX_RINGROOT:-/dev/shm/flux}"
mkdir -p "${FLUX_RINGROOT}" 2>/dev/null || true
exec docker run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e LPDM_WORKERS="${LPDM_WORKERS:-1}" \
  -v ${FLUX_ROOT}:/work -v ${FLUX_RINGROOT}:${FLUX_RINGROOT} -w /work --entrypoint python3 flux-fasteddy:cuda118 "$@"
