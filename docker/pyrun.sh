#!/usr/bin/env bash
# Run a project python script inside the container (host lacks netCDF4).
#   docker/pyrun.sh docker/diag_near_surface.py runs/foo/output/FE_X.1000
#   docker/pyrun.sh - <<'PY' ... PY      (script on stdin)
# --entrypoint python3 skips the image's CUDA banner scripts.
#
# TWO MODES, ONE SCRIPT.
#
#   FLUX_NATIVE unset (the host)  -- shell out to `docker run`, exactly as before.
#   FLUX_NATIVE=1  (inside the portable image) -- exec python3 directly.
#
# The native branch exists because the deployable image BAKES THE CODE IN and runs the
# project's own drivers from inside a container, where there is no docker daemon to shell
# out to and a container cannot launch a sibling. Every caller passes REPO-RELATIVE paths
# (because the host branch mounts the repo at /work), so the native branch only has to
# `cd` to the repo root for those same paths to resolve -- and an ABSOLUTE path, which the
# orchestrator passes for job directories living on the mounted output volume, resolves on
# its own. There is no path translation in native mode because there is no mount to
# translate through.
set -uo pipefail
# See docker/run_case.sh: repo root is overridable for off-machine jobs.
FLUX_ROOT="${FLUX_ROOT:-/home/atyagi/Flux}"
# LPDM_WORKERS is forwarded so the campaign can size the fork pool per stage;
# without it every container would silently fall back to one core.

if [ "${FLUX_NATIVE:-0}" = "1" ]; then
  # THE THREAD CAPS ARE NOT COSMETIC HERE. numpy/OpenBLAS opens one thread per core by
  # default -- MEASURED at 32 threads in this image on this machine -- and the orchestrator
  # runs up to 16 acceptance batteries at once. Uncapped that is 512 runnable threads
  # fighting for the cores the 16 FastEddy host processes also need, which does not fail,
  # it just makes everything slower and makes the measured GPU-h per seed meaningless.
  # Set only if the caller has not: an explicit value always wins.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  export LPDM_WORKERS="${LPDM_WORKERS:-1}"
  cd "${FLUX_ROOT}" || { echo "FATAL: no FLUX_ROOT ${FLUX_ROOT}" >&2; exit 1; }
  exec python3 "$@"
fi

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
  -v ${FLUX_ROOT}:/work -v ${FLUX_RINGROOT}:${FLUX_RINGROOT} -w /work --entrypoint python3 "${FLUX_IMAGE:-flux-fasteddy:cuda118}" "$@"
