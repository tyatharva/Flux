#!/usr/bin/env bash
# Gate C2: does a returned seed restart BIT-FOR-BIT?
#
#   usage: bin/c2_restart_check.sh <restart.nc> <step> [template.in]
#          bin/c2_restart_check.sh jobs/seed_x/return/seed_restart.nc 20520
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
TMPL="${3:-runs/g16_base/base.in}"
D=runs/c2_check
die(){ echo "FATAL: $*" >&2; exit 1; }
[ -f "$RST" ] || die "no restart at $RST"
[ -f "$TMPL" ] || die "no template at $TMPL"

rm -rf $D; mkdir -p $D/output
cp -f "$RST" "$D/FE_RST.$STEP" || die "copy"
sed -e "s|^Nt = .*|Nt = $STEP|" -e "s|^NtBatch = .*|NtBatch = $STEP|" \
    -e "s|^frqOutput = .*|frqOutput = $STEP|" -e 's|^inPath = .*|inPath = ./|' \
    -e "s|^inFile = .*|inFile = FE_RST.$STEP|" -e 's|^outPath = .*|outPath = ./output/|' \
    -e 's|^outFileBase = .*|outFileBase = FE_C2|' -e 's|^topoFile = .*|topoFile = |' \
    "$TMPL" > $D/c2.in
echo "########## GATE C2: restart $(basename "$RST") at step $STEP, re-dump, diff ##########"
L="${LOGDIR:-${TMPDIR:-/tmp}/flux-logs}"; mkdir -p "$L"
./docker/run_case.sh $D c2.in "$L/c2.log" || echo "  (run_case reported non-zero; scoring the artifact anyway)"

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
