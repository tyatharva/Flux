#!/usr/bin/env bash
# Run one FastEddy case in the container and SCORE IT.
# usage: run_case.sh <case_dir> <case_file.in> [logfile]
#
# Always routes through check_run.sh, because FastEddy exits 0 on fully-NaN fields.
set -uo pipefail
# One GPU, one run. Two FastEddy containers writing the same output/ silently
# interleave their dumps and corrupt both -- and it looks like a mysteriously stalled
# run, not an error. Refuse rather than race.
#
# Match on the FastEddy BINARY, not on the image: analysis scripts share the image and
# only use the CPU, so an image-level filter blocks the very runs it is meant to protect.
busy=""
for c in $(docker ps -q --filter ancestor=flux-fasteddy:cuda118); do
  if docker inspect -f '{{json .Config.Cmd}}' "$c" 2>/dev/null | grep -q 'FEMAIN/FastEddy'; then
    busy="$busy $c"
  fi
done
if [ -n "$busy" ]; then
  echo "  REFUSED: a FastEddy run is already in progress:" >&2
  docker ps --format '    {{.Names}} {{.Status}} {{.Command}}' --filter "id=${busy## }" >&2
  exit 2
fi
# A MISSING RESTART FILE DOES NOT ABORT FastEddy. It prints "Error: No such file or
# directory", carries on with x,y,z dimensions of 0, and produces a run in which every
# cell of every field is NaN -- while still exiting 0. That cost a 30-minute segment once.
# Check the file exists before spending GPU time on it.
CASE_DIR="$1"; CASE_FILE="$2"
_cf="/home/atyagi/Flux/${CASE_DIR}/${CASE_FILE}"
if [ -f "$_cf" ]; then
  _ip=$(grep -oP '^inPath\s*=\s*\K[^#[:space:]]*' "$_cf" || true)
  _if=$(grep -oP '^inFile\s*=\s*\K[^#[:space:]]*' "$_cf" || true)
  if [ -n "${_if:-}" ] && [ ! -f "/home/atyagi/Flux/${CASE_DIR}/${_ip}${_if}" ]; then
    echo "  REFUSED: restart file ${_ip}${_if} not found relative to ${CASE_DIR}" >&2
    echo "           (FastEddy would run to completion and write only NaN)" >&2
    exit 3
  fi
fi
LOG="${3:-/tmp/flux-logs/$(basename "$CASE_FILE" .in).log}"
mkdir -p "$(dirname "$LOG")"
docker run --gpus all --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /home/atyagi/Flux:/work -w "/work/${CASE_DIR}" flux-fasteddy:cuda118 \
  mpirun -np 1 /work/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy "./${CASE_FILE}" > "$LOG" 2>&1
RC=$?
# Score the log AND the newest dump (accuracy-CFL k0/k1 check) in one place.
LAST=$(ls -t "/home/atyagi/Flux/${CASE_DIR}/output"/*.[0-9]* 2>/dev/null | head -1)
LAST_REL="${LAST#/home/atyagi/Flux/}"
exec "$(dirname "$0")/check_run.sh" "$LOG" "$RC" ${LAST_REL:+"$LAST_REL"}
