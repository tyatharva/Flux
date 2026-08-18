#!/usr/bin/env bash
# Run a command inside the flux-fasteddy container with the project bind-mounted.
#
#   docker/run.sh ./docker/build_fasteddy.sh
#   docker/run.sh bash -lc 'cd runs/stage0a_smoke && mpirun -np 1 ... FastEddy ./case.in'
#
# --user $(id -u):$(id -g) keeps every artifact owned by you on the host, and
# sidesteps OpenMPI's refusal to run as root. HOME=/tmp because that UID has no
# passwd entry inside the container.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/atyagi/Flux}"
IMAGE="${IMAGE:-flux-fasteddy:cuda118}"

exec docker run --gpus all --rm -it \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "${PROJECT_ROOT}:/work" \
  -w /work \
  "${IMAGE}" \
  "$@"
