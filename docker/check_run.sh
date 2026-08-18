#!/usr/bin/env bash
# Score a FastEddy run log. usage: check_run.sh <logfile> [exit_code] [dump.nc ...]
#
# WHY THIS EXISTS: FastEddy prints "****CORRUPTED*** --- (#NaN, #Inf)" when a field
# goes non-finite, but it does NOT change its exit status -- a run whose every field
# is NaN still exits 0. Exit code alone is therefore not a correctness test.
# Nothing in this project may trust a run without going through this script.
#
# It also runs the standing near-surface accuracy-CFL check (k0/k1 < 1) on any dumps
# passed as extra arguments. That failure mode is even quieter than NaN: the run
# completes, prints nothing, and the near-surface resolved w is grid-scale acoustic
# noise instead of turbulence. See docker/k0k1_check.py.
set -uo pipefail
LOG="$1"; RC="${2:-0}"; shift 2 2>/dev/null || shift $#
fail=0

# `grep -c` prints 0 AND exits 1 on no-match, so `|| true` (never `|| echo 0`).
count() { grep -cE "$1" "$LOG" 2>/dev/null || true; }

corrupt=$(count "CORRUPTED")
# '#NaN'/'#Inf' are FastEddy's own report tokens; the signed forms catch printed values.
# Deliberately NOT a bare /inf/ -- that matches the "inFile" parameter echo.
nanhit=$(count '#NaN|#Inf|[-+[:space:]](nan|inf)([[:space:]]|$)')
errs=$(count "CRITICAL ERROR|_FAIL|MPI_ERR|too many resources|Segmentation")
done_ok=$(count "simulation is complete")

[ "${RC:-0}" -ne 0 ]      && { echo "  FAIL: nonzero exit ($RC)"; fail=1; }
[ "${corrupt:-0}" -ne 0 ] && { echo "  FAIL: $corrupt CORRUPTED field report(s)"; fail=1; }
[ "${nanhit:-0}" -ne 0 ]  && { echo "  FAIL: $nanhit NaN/Inf mention(s)"; fail=1; }
[ "${errs:-0}" -ne 0 ]    && { echo "  FAIL: $errs error string(s)"; fail=1; }
[ "${done_ok:-0}" -eq 0 ] && { echo "  FAIL: no 'simulation is complete' banner"; fail=1; }

# Standing accuracy-CFL check on any dumps handed to us.
if [ "$#" -gt 0 ]; then
  "$(dirname "$0")/pyrun.sh" docker/k0k1_check.py "$@" || fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "  RUN OK (exit 0, no CORRUPTED, no NaN/Inf, completion banner, k0/k1 < 1)"
else
  echo "  >>> RUN REJECTED: $LOG"
fi
exit $fail
