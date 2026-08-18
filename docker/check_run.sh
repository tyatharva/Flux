#!/usr/bin/env bash
# Score a FastEddy run log. usage: check_run.sh <logfile> [exit_code]
#
# WHY THIS EXISTS: FastEddy prints "****CORRUPTED*** --- (#NaN, #Inf)" when a field
# goes non-finite, but it does NOT change its exit status -- a run whose every field
# is NaN still exits 0. Exit code alone is therefore not a correctness test.
# Nothing in this project may trust a run without going through this script.
set -uo pipefail
LOG="$1"; RC="${2:-0}"
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

if [ "$fail" -eq 0 ]; then
  echo "  RUN OK (exit 0, no CORRUPTED, no NaN/Inf, completion banner present)"
else
  echo "  >>> RUN REJECTED: $LOG"
fi
exit $fail
