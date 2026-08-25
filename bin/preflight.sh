#!/usr/bin/env bash
# Compile every python entry point and shell driver BEFORE a campaign spends GPU time.
#
# A duplicate keyword argument in stage5_footprint.py reached a running campaign and was
# only discovered when the analysis had already been launched six times, each after its
# own field load -- because the error lands in a redirected .txt and the driver reads the
# exit status of the pipeline, not of python. Ten seconds here would have caught it.
set -uo pipefail
# The repo root is discovered, not hardcoded, for the same reason docker/run_case.sh's is:
# a seed job runs on a rented GPU whose checkout is somewhere else entirely.
cd "${FLUX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
bad=0
for f in bin/*.py lpdm/*.py docker/*.py; do
  python3 -c "import ast,sys; ast.parse(open('$f').read())" 2>&1 | sed "s|^|  $f: |" || bad=1
  python3 -c "import ast,sys
try: ast.parse(open('$f').read())
except SyntaxError as e: print('  SYNTAX %s:%s %s' % ('$f', e.lineno, e.msg)); sys.exit(1)"  || bad=1
done
for f in bin/*.sh docker/*.sh jobs/*.sh; do
  [ -e "$f" ] || continue
  bash -n "$f" || { echo "  SHELL $f"; bad=1; }
done
# ENTRY POINTS MUST ALSO ANSWER --help. A clean parse says nothing about a NameError at
# module scope or an argparse definition that raises -- both of which look exactly like a
# working script until the moment a campaign calls one. --help exercises import and
# argument construction and nothing else, so it costs milliseconds and touches no data.
for f in bin/hrrr_sounding.py bin/sounding_to_forcing.py bin/make_seed_jobs.py          bin/pick_seed.py bin/make_pair.py bin/seed_stationarity.py          bin/test_sounding.py; do
  [ -e "$f" ] || continue
  out=$(python3 "$f" --help 2>&1) || { echo "  ENTRY $f: $(echo "$out" | tail -3)"; bad=1; }
done
[ $bad -eq 0 ] && echo "preflight OK: $(ls bin/*.py lpdm/*.py | wc -l) python, $(ls bin/*.sh docker/*.sh jobs/*.sh 2>/dev/null | wc -l) shell"
exit $bad
