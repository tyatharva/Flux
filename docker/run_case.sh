#!/usr/bin/env bash
# Run one FastEddy case in the container and SCORE IT.
# usage: run_case.sh <case_dir> <case_file.in> [logfile]
#
# Always routes through check_run.sh, because FastEddy exits 0 on fully-NaN fields.
set -uo pipefail
CASE_DIR="$1"; CASE_FILE="$2"
LOG="${3:-/tmp/flux-logs/$(basename "$CASE_FILE" .in).log}"
mkdir -p "$(dirname "$LOG")"
docker run --gpus all --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /home/atyagi/Flux:/work -w "/work/${CASE_DIR}" flux-fasteddy:cuda118 \
  mpirun -np 1 /work/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy "./${CASE_FILE}" > "$LOG" 2>&1
RC=$?
exec "$(dirname "$0")/check_run.sh" "$LOG" "$RC"
