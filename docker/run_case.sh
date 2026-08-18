#!/usr/bin/env bash
# Run one FastEddy case in the container. usage: run_case.sh <case_dir> <case_file.in>
set -euo pipefail
CASE_DIR="$1"; CASE_FILE="$2"
exec docker run --gpus all --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /home/atyagi/Flux:/work -w "/work/${CASE_DIR}" flux-fasteddy:cuda118 \
  mpirun -np 1 /work/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy "./${CASE_FILE}"
