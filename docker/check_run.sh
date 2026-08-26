#!/usr/bin/env bash
# Score a FastEddy run log. usage: check_run.sh <logfile> [exit_code] [dump.nc ...]
#
# WHY THIS EXISTS: FastEddy prints "****CORRUPTED*** --- (#NaN, #Inf)" when a field
# goes non-finite, but it does NOT change its exit status -- a run whose every field
# is NaN still exits 0. Exit code alone is therefore not a correctness test.
# Nothing in this project may trust a run without going through this script.
#
# It also runs TWO standing checks on any dumps passed as extra arguments:
#
#   k0/k1 < 1        docker/k0k1_check.py -- the accuracy-CFL check. Quieter than NaN:
#                    the run completes, prints nothing, and the near-surface resolved w
#                    is grid-scale acoustic noise instead of turbulence.
#
#   turbulence alive docker/turb_alive.py -- and it is here because k0/k1 IS NOT ENOUGH.
#                    k0/k1 read 0.442, a comfortable pass, on a stable boundary layer
#                    whose turbulence had entirely collapsed (u* 0.236 -> 0.098, the flow
#                    above 66 m exactly geostrophic). It is a ratio between two levels,
#                    so it survives both levels going quiet together. It is a dt check,
#                    not a physics check, and this project had nothing that asked whether
#                    a boundary layer still existed.
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

# Standing checks on any dumps handed to us. BOTH run; neither substitutes for the other.
SKIPPED=""
if [ "$#" -gt 0 ]; then
  _k=$("$(dirname "$0")/pyrun.sh" docker/k0k1_check.py "$@" 2>&1); _rc=$?
  echo "$_k"; [ "$_rc" -ne 0 ] && fail=1
  case "$_k" in *"k0/k1 SKIP"*) SKIPPED="$SKIPPED k0/k1";; esac
  _t=$("$(dirname "$0")/pyrun.sh" docker/turb_alive.py "$@" 2>&1); _rc=$?
  echo "$_t"; [ "$_rc" -ne 0 ] && fail=1
  case "$_t" in *"turb-alive SKIP"*) SKIPPED="$SKIPPED turbulence-alive";; esac
fi

# A SKIP IS NOT A PASS, AND THE BANNER USED TO SAY IT WAS. Observed live: a collapsed
# stable segment produced `k0/k1 SKIP (turbulence undeveloped)` -- because the near-surface
# variance had fallen BELOW the floor by dying -- and `turb-alive SKIP`, and this script
# printed "RUN OK ... k0/k1 < 1, turbulence alive". Neither check had rendered a verdict.
# The banner now names what was actually established.
VERD="k0/k1 < 1, turbulence alive"
[ -n "$SKIPPED" ] && VERD="but NO VERDICT from:$SKIPPED -- these established NOTHING"
if [ "$fail" -eq 0 ]; then
  echo "  RUN OK (exit 0, no CORRUPTED, no NaN/Inf, completion banner, $VERD)"
else
  echo "  >>> RUN REJECTED: $LOG"
fi
exit $fail
