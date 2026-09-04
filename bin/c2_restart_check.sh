#!/usr/bin/env bash
# Gate C2: does a returned seed restart BIT-FOR-BIT?
#
#   usage: bin/c2_restart_check.sh <restart.nc> <step> [template.in]
#          bin/c2_restart_check.sh seeds/seed_x/return/seed_restart.nc 20520
#
# THE TWO TRAPS THIS TEST IS MADE OF, and getting either backwards silently turns a
# zero-timestep echo into a full integration:
#
#   TRAP 4  The restart step is parsed from the FILENAME -- sscanf on the characters after
#           the first '.' (time_integration.c:104). So the restart file must be NAMED for
#           the step it holds. Calling it FE_RST.0 resets the counter to zero.
#   TRAP 6  Nt is an ABSOLUTE target step, not a count. Restarting from step N with Nt = N
#           performs zero timesteps, writes one dump, and exits 0.
#
# Combine them wrongly -- FE_RST.0 with Nt = 20520 -- and FastEddy runs 20520 real steps
# and the comparison reports u differing by 2.65 m/s. Which is what happened the first time
# this was run, and is why the step is now an explicit argument rather than an assumption.
#
# TRAP 5 matters too: frqOutput is tested against the ABSOLUTE step, so the step must be a
# multiple of frqOutput or nothing is written. frqOutput = step handles that.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
RST="${1:?usage: c2_restart_check.sh <restart.nc> <step> [template.in]}"
STEP="${2:?the ABSOLUTE step the restart holds -- see trap 4}"
# NO DEFAULT TEMPLATE. It used to be `runs/g16_base/base.in` -- the RETIRED 16 m grid --
# on the only GPU step of the acceptance battery. Every caller passes the seed's own
# seed.in, so nothing was broken; but this is the identical hazard docs/PLAN.md records for
# run_corpus_case.sh's 16 m default and for results/tback_production.txt: a plausible,
# complete, wrong-grid result the moment a caller stops passing the argument. And that
# directory is not even in the deployable image, so the failure would be a confusing
# "no template" rather than a wrong grid -- worse in a different way. Require it.
TMPL="${3:?the .in template to derive the C2 case from -- pass the seed job own seed.in.
      There is deliberately no default, because the old one was the retired 16 m grid}"
# THE SCRATCH DIRECTORY IS PER-CHECK, NOT FIXED, AND THAT IS NOT A TIDINESS POINT.
# It was the literal `runs/c2_check`, and the first line of work here is `rm -rf $D`. On a
# one-GPU workstation only one C2 can be running at a time, so a fixed name is safe by
# construction. On a 16-GPU box sixteen acceptance batteries run at once, and each one
# would delete the 73 MB restart the other fifteen had just staged, then compare whatever
# survived -- producing a bit-for-bit FAIL on a perfectly good seed, or a PASS against
# another seed's restart. Default derived from the restart's own job so it is unique
# without the caller having to think about it; C2_DIR overrides.
D="${C2_DIR:-runs/c2_check_$(basename "$(dirname "$(dirname "$(readlink -f "$RST")")")")}"
die(){ echo "FATAL: $*" >&2; exit 1; }
[ -f "$RST" ] || die "no restart at $RST"
[ -f "$TMPL" ] || die "no template at $TMPL"

rm -rf "$D"; mkdir -p "$D/output"
cp -f "$RST" "$D/FE_RST.$STEP" || die "copy"
sed -e "s|^Nt = .*|Nt = $STEP|" -e "s|^NtBatch = .*|NtBatch = $STEP|" \
    -e "s|^frqOutput = .*|frqOutput = $STEP|" -e 's|^inPath = .*|inPath = ./|' \
    -e "s|^inFile = .*|inFile = FE_RST.$STEP|" -e 's|^outPath = .*|outPath = ./output/|' \
    -e 's|^outFileBase = .*|outFileBase = FE_C2|' -e 's|^topoFile = .*|topoFile = |' \
    "$TMPL" > "$D/c2.in"
echo "########## GATE C2: restart $(basename "$RST") at step $STEP, re-dump, diff ##########"
L="${LOGDIR:-${TMPDIR:-/tmp}/flux-logs}"; mkdir -p "$L"
# The log name follows the scratch directory for the same reason the directory does.
#
# A REFUSAL IS NOT A FAILURE, AND SCORING ONE PRODUCES A FALSE "GATE C2: FAIL".
# docker/run_case.sh exits 2 (a FastEddy is already on this GPU), 3 (the restart file is
# missing) or 4 (a .in line is too long) WITHOUT EVER LAUNCHING. In those cases no FE_C2.*
# is written, the scorer below finds nothing, and prints "FAIL: the re-dump produced
# nothing" -- a bit-for-bit failure attributed to a restart that was never tested. The
# host path guards against this with a docker-ps check in bin/seed_accept.sh; the native
# path has no such check by design, because the per-GPU mutex is the guard. So the exit
# code has to be read here.
./docker/run_case.sh "$D" c2.in "$L/$(basename "$D").log"; RC_C2=$?
case "$RC_C2" in
  0) ;;
  2|3|4)
    echo "  DEFERRED: run_case.sh refused to launch (exit $RC_C2) -- the GPU is busy, the"
    echo "  restart is missing, or the .in is unreadable. NOTHING WAS TESTED. THIS IS NOT"
    echo "  A PASS and it is NOT a FAIL: rerun step 6 when the device is free."
    exit 0 ;;
  *) echo "  (run_case reported non-zero ($RC_C2) after launching; scoring the artifact anyway)" ;;
esac

./docker/pyrun.sh - "$D" "$STEP" <<'PY'
import glob, sys, numpy as np
from netCDF4 import Dataset
D, step = sys.argv[1], sys.argv[2]
outs = sorted(glob.glob(f"{D}/output/FE_C2.*"))
if not outs:
    print("  FAIL: the re-dump produced nothing -- check the step and frqOutput (trap 5)")
    sys.exit(1)
print(f"  re-dump: {', '.join(o.split('/')[-1] for o in outs)}")
a = Dataset(f"{D}/FE_RST.{step}"); b = Dataset(outs[-1])
n = nbad = 0; worst = 0.0
for v in a.variables:
    if v not in b.variables:
        continue
    x, y = np.asarray(a[v][:]), np.asarray(b[v][:])
    if x.shape != y.shape or x.dtype.kind not in "fi":
        continue
    n += 1
    if not np.array_equal(x, y):
        nbad += 1
        d = float(np.abs(x.astype("f8") - y.astype("f8")).max())
        worst = max(worst, d)
        print(f"    {v}: differs, max |diff| {d:.3e}")
print(f"  {n} variables compared, {nbad} differ, worst {worst:.3e}")
print(f"  GATE C2: {'PASS -- bit-for-bit' if nbad == 0 else 'FAIL'}")
sys.exit(0 if nbad == 0 else 1)
PY
