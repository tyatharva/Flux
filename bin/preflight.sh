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
# === AND PARSE THEM WHERE THEY ACTUALLY RUN =========================================
# THE HOST PYTHON IS NOT THE PYTHON THESE FILES EXECUTE ON. Every analysis runs inside
# flux-fasteddy:cuda118 (the host has no scipy), and the two interpreters are not the same
# version -- 3.12 on this host against 3.10 in the image at the time of writing. A file
# that parses here can be a SyntaxError there, and it lands in a redirected .txt after the
# GPU time is already spent, which is the exact failure this script exists to prevent.
# Observed: an f-string reusing the outer quote inside a nested expression, legal from
# 3.12 and fatal on 3.10, cleared the host pass and died in the container.
#
# One docker invocation for the whole tree, so it costs a second or two. Skipped with a
# loud line if the image is not present -- a rented GPU may be building it still.
if docker image inspect flux-fasteddy:cuda118 >/dev/null 2>&1; then
  # -i IS LOAD-BEARING. Without it docker does not attach stdin, `python3 -` reads an
  # empty program, prints nothing and exits 0 -- a silent PASS on every file. Caught
  # here by a deliberate 3.12-only canary that this check waved through.
  cpy=$(docker run --rm -i -v "$PWD:/w" -w /w flux-fasteddy:cuda118 python3 - <<'PYC' 2>&1
import ast, glob, sys
bad = 0
for f in sorted(glob.glob("bin/*.py") + glob.glob("lpdm/*.py") + glob.glob("docker/*.py")):
    try:
        ast.parse(open(f).read())
    except SyntaxError as e:
        print("  CONTAINER SYNTAX %s:%s %s" % (f, e.lineno, e.msg)); bad = 1
print("  container python %s: %s" % (sys.version.split()[0], "clean" if not bad else "FAILED"))
sys.exit(bad)
PYC
  ) || bad=1
  echo "$cpy"
  # ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS. An exit 0 from a container that
  # never ran the program is indistinguishable from a clean pass unless the pass
  # has to SAY something.
  case "$cpy" in *"container python"*) :;; *)
    echo "  FATAL: the container parse produced no verdict line -- it did not run."
    bad=1;; esac
else
  echo "  NOTE: flux-fasteddy:cuda118 not present; python was parsed ONLY by the host "
  echo "        interpreter ($(python3 -V 2>&1 | cut -d' ' -f2)), which is not the one "
  echo "        the analysis runs on. Build the image before trusting this pass."
fi

for f in bin/*.sh docker/*.sh fasteddy/*.sh; do
  [ -e "$f" ] || continue
  bash -n "$f" || { echo "  SHELL $f"; bad=1; }
done
# ENTRY POINTS MUST ALSO ANSWER --help. A clean parse says nothing about a NameError at
# module scope or an argparse definition that raises -- both of which look exactly like a
# working script until the moment a campaign calls one. --help exercises import and
# argument construction and nothing else, so it costs milliseconds and touches no data.
for f in bin/hrrr_sounding.py bin/sounding_to_forcing.py bin/make_seed_jobs.py          bin/pick_seed.py bin/make_pair.py bin/seed_stationarity.py bin/corpus_monitor.py bin/test_floor_health.py bin/direction_drift.py bin/case_compare.py          bin/test_sounding.py; do
  [ -e "$f" ] || continue
  out=$(python3 "$f" --help 2>&1) || { echo "  ENTRY $f: $(echo "$out" | tail -3)"; bad=1; }
done
[ $bad -eq 0 ] && echo "preflight OK: $(ls bin/*.py lpdm/*.py | wc -l) python, $(ls bin/*.sh docker/*.sh fasteddy/*.sh 2>/dev/null | wc -l) shell"
exit $bad
