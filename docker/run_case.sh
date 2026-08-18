#!/usr/bin/env bash
# Run one FastEddy case in the container and SCORE IT.
# usage: run_case.sh <case_dir> <case_file.in> [logfile]
#
# Always routes through check_run.sh, because FastEddy exits 0 on fully-NaN fields.
set -uo pipefail
# One GPU, one run. Two FastEddy containers writing the same output/ silently
# interleave their dumps and corrupt both -- and it looks like a mysteriously stalled
# run, not an error. Refuse rather than race.
if [ -n "$(docker ps -q --filter ancestor=flux-fasteddy:cuda118)" ]; then
  echo "  REFUSED: a FastEddy container is already running:" >&2
  docker ps --filter ancestor=flux-fasteddy:cuda118 --format '    {{.Names}} {{.Status}} {{.Command}}' >&2
  exit 2
fi
CASE_DIR="$1"; CASE_FILE="$2"
LOG="${3:-/tmp/claude-1000/$(basename "$CASE_FILE" .in).log}"
mkdir -p "$(dirname "$LOG")"
docker run --gpus all --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /home/atyagi/Flux:/work -w "/work/${CASE_DIR}" flux-fasteddy:cuda118 \
  mpirun -np 1 /work/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy "./${CASE_FILE}" > "$LOG" 2>&1
RC=$?
# Score the log AND the newest dump (accuracy-CFL k0/k1 check) in one place.
LAST=$(ls -t "/home/atyagi/Flux/${CASE_DIR}/output"/*.[0-9]* 2>/dev/null | head -1)
LAST_REL="${LAST#/home/atyagi/Flux/}"
exec "$(dirname "$0")/check_run.sh" "$LOG" "$RC" ${LAST_REL:+"$LAST_REL"}
