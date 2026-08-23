#!/usr/bin/env bash
# Compile every python entry point and shell driver BEFORE a campaign spends GPU time.
#
# A duplicate keyword argument in stage5_footprint.py reached a running campaign and was
# only discovered when the analysis had already been launched six times, each after its
# own field load -- because the error lands in a redirected .txt and the driver reads the
# exit status of the pipeline, not of python. Ten seconds here would have caught it.
set -uo pipefail
cd /home/atyagi/Flux
bad=0
for f in bin/*.py lpdm/*.py docker/*.py; do
  python3 -c "import ast,sys; ast.parse(open('$f').read())" 2>&1 | sed "s|^|  $f: |" || bad=1
  python3 -c "import ast,sys
try: ast.parse(open('$f').read())
except SyntaxError as e: print('  SYNTAX %s:%s %s' % ('$f', e.lineno, e.msg)); sys.exit(1)"  || bad=1
done
for f in bin/*.sh docker/*.sh; do bash -n "$f" || { echo "  SHELL $f"; bad=1; }; done
[ $bad -eq 0 ] && echo "preflight OK: $(ls bin/*.py lpdm/*.py | wc -l) python, $(ls bin/*.sh docker/*.sh | wc -l) shell"
exit $bad
