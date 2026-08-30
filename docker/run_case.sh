#!/usr/bin/env bash
# Run one FastEddy case in the container and SCORE IT.
# usage: run_case.sh <case_dir> <case_file.in> [logfile]
#
# Always routes through check_run.sh, because FastEddy exits 0 on fully-NaN fields.
set -uo pipefail
# Repo root is a variable so a seed job can run on a rented GPU whose checkout is
# somewhere else. Defaults to this machine, so nothing that already works changes.
FLUX_ROOT="${FLUX_ROOT:-/home/atyagi/Flux}"
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
_cf="${FLUX_ROOT}/${CASE_DIR}/${CASE_FILE}"
# A LINE OF 256 CHARACTERS OR MORE IN THE .in SEGFAULTS FastEddy BEFORE IT STARTS.
# parameters.c:28 sets MAXLEN 256 and reads with fgets(strBuff, MAXLEN, ...), so a longer
# line is split. The first piece parses; the CONTINUATION is a fragment with no '=', and
# parameters.c:126-133 then calls str_trim(valueBuff) on the NULL that strchr returned.
# The result is "Signal: Segmentation fault ... Failing at address: (nil)" with a six-frame
# libc backtrace and no mention of the file or the line -- so the natural reading is that
# the model is broken, not that a COMMENT is too long. Cost: one acceptance run, and it
# would have cost every run of the pass, because the offending line was in the TEMPLATE.
if [ -f "$_cf" ]; then
  _long=$(awk 'length($0) >= 255 {print NR": "length($0)" chars"}' "$_cf" | head -3)
  if [ -n "$_long" ]; then
    echo "  REFUSED: ${CASE_FILE} has line(s) at or over 255 characters, which FastEddy's" >&2
    echo "           parameter parser cannot read (parameters.c:28 MAXLEN 256). It would" >&2
    echo "           segfault at address (nil) with no reference to the file:" >&2
    echo "$_long" | sed 's|^|             line |' >&2
    exit 4
  fi
fi
if [ -f "$_cf" ]; then
  _ip=$(grep -oP '^inPath\s*=\s*\K[^#[:space:]]*' "$_cf" || true)
  _if=$(grep -oP '^inFile\s*=\s*\K[^#[:space:]]*' "$_cf" || true)
  if [ -n "${_if:-}" ] && [ ! -f "${FLUX_ROOT}/${CASE_DIR}/${_ip}${_if}" ]; then
    echo "  REFUSED: restart file ${_ip}${_if} not found relative to ${CASE_DIR}" >&2
    echo "           (FastEddy would run to completion and write only NaN)" >&2
    exit 3
  fi
fi
LOG="${3:-/tmp/flux-logs/$(basename "$CASE_FILE" .in).log}"
mkdir -p "$(dirname "$LOG")"
# THE IN-PROCESS LPDM STAGING DIRECTORY, MOUNTED AT AN IDENTICAL PATH ON BOTH SIDES.
# Docker gives a container 64 MB of /dev/shm by default -- measured, after a 60-snapshot
# staging attempt died with ENOSPC at 2.2 GB -- so the host tmpfs has to be mounted
# explicitly. The path is the SAME inside and out on purpose: lpdmOnlineDir is written
# into the .in that FastEddy reads in one container and polled by the analysis in another,
# and a path that means two different things in two containers is the kind of quiet
# mismatch this project keeps paying for. Created on demand; a no-op when unused.
FLUX_RINGROOT="${FLUX_RINGROOT:-/dev/shm/flux}"
mkdir -p "${FLUX_RINGROOT}" 2>/dev/null || true
# A NAME, SO THE CONTAINER CAN BE STOPPED BY SOMETHING OTHER THAN LUCK.
# The in-process hand-off runs this in the BACKGROUND while the LPDM consumes in the
# foreground, and a driver that has to abandon the case kills the shell it launched -- which
# does NOT stop the container. Measured: a killed smoke run left FastEddy holding the GPU,
# and the next run was refused by the guard above with no hint of why. The name is derived
# from the case directory, and the guard above already forbids two FastEddy runs at once, so
# it cannot collide with a legitimate second run.
FE_NAME="${FE_CONTAINER_NAME:-flux-fe-$(echo "$CASE_DIR" | tr -c 'A-Za-z0-9_.-' '-')}"
docker rm -f "$FE_NAME" >/dev/null 2>&1 || true
docker run --gpus all --rm --name "$FE_NAME" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v ${FLUX_ROOT}:/work -v ${FLUX_RINGROOT}:${FLUX_RINGROOT} -w "/work/${CASE_DIR}" flux-fasteddy:cuda118 \
  mpirun -np 1 /work/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy "./${CASE_FILE}" > "$LOG" 2>&1
RC=$?
# Score the log AND the newest dump (accuracy-CFL k0/k1 check) in one place.
# Read outPath from the CASE FILE rather than assuming ./output/. Sampling windows write
# to ./window/, and assuming ./output/ silently scored the leftover adjustment dump
# instead -- so the standing accuracy check was passing on a file the run never touched.
_op=$(grep -oP '^outPath\s*=\s*\K[^#[:space:]]*' "$_cf" 2>/dev/null || true)
_op="${_op:-./output/}"
LAST=$(ls -t "${FLUX_ROOT}/${CASE_DIR}/${_op}"/*.[0-9]* 2>/dev/null | head -1)
LAST_REL="${LAST#${FLUX_ROOT}/}"
exec "$(dirname "$0")/check_run.sh" "$LOG" "$RC" ${LAST_REL:+"$LAST_REL"}
