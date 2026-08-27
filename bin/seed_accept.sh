#!/usr/bin/env bash
# THE FULL ACCEPTANCE BATTERY FOR ONE SEED, in one place so every seed gets the same one.
#
# The first seed's battery was assembled by hand from PROJECT_BRIEF.md and PLAN.md, which is
# fine once and unrepeatable fifteen times: the risk is not that a check fails, it is that
# a check is quietly SKIPPED for one rung and the library ends up with seeds that were
# held to different standards. Every item below is run for every seed, and an item that
# cannot run says so rather than being absent.
#
#   usage: bin/seed_accept.sh jobs/seed_nbl-deep_a000 [--wall-seconds N]
#
# ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS (FASTEDDY_TRAPS.md 12). Every step here is
# piped into tee or grep, so $? belongs to the last element of the pipe. Verdicts are
# re-read from the JSON each tool writes, and a missing JSON is a failure.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
JOB_ARG="${1:?usage: seed_accept.sh <job_dir> [--wall-seconds N]}"; shift
JOB="$(cd "$JOB_ARG" && pwd)"; JOB_REL="${JOB#$ROOT/}"
WALL=""; [ "${1:-}" = "--wall-seconds" ] && { WALL="$2"; shift 2; }

OUT="$JOB/return/acceptance.txt"; : > "$OUT"
say(){ echo -e "\n=== $* ===" | tee -a "$OUT"; }
tee_(){ tee -a "$OUT"; }

read -r NAME DT TOTAL WTH OUTBASE REGIME < <(python3 -c "
import json;m=json.load(open('$JOB/manifest.json'));r=m['run']
print(m['job'],r['dt'],r['steps_total'],m['target']['wth_virtual'],r['outFileBase'],m['regime'])")
LAST="$JOB/output/$OUTBASE.$TOTAL"
[ -f "$LAST" ] || { echo "FATAL: no final dump $LAST" >&2; exit 1; }
echo "########## acceptance battery: $NAME ($REGIME) ##########" | tee_
date '+%F %H:%M:%S' | tee_

# ---- 1. the log ------------------------------------------------------------------
say "1. the log: CORRUPTED / NaN / Inf, and the completion banner"
LOG="$JOB/return/run.log"
for pat in CORRUPTED '#NaN' '#Inf'; do
  n=$(grep -c -- "$pat" "$LOG" 2>/dev/null || true); echo "  $pat: ${n:-0}" | tee_
  [ "${n:-0}" = "0" ] || echo "  *** FAIL: $pat present" | tee_
done
grep -qiE "Ending FastEddy|Completed|SUCCESS" "$LOG" && echo "  completion banner: present" | tee_ \
  || echo "  completion banner: ABSENT" | tee_

# ---- 2. cost, artifact, the seven limits, which binds last ------------------------
say "2. seed_report: cost, artifact, the seven limits"
./docker/pyrun.sh bin/seed_report.py "$JOB_REL" ${WALL:+--wall-seconds "$WALL"} \
    --out "$JOB_REL/return/seed_report.json" 2>&1 | tee_

# ---- 3. the accuracy CFL ---------------------------------------------------------
say "3. k0/k1 (accuracy CFL; ~9 means dt is past the boundary)"
./docker/pyrun.sh docker/k0k1_check.py "${LAST#$ROOT/}" 2>&1 | tee_

# ---- 4. is the turbulence ALIVE -- a VERDICT, never a SKIP ------------------------
# k0/k1 is a dt check, not a physics check. A stable seed collapsed with k0/k1 at 0.442
# throughout (PROJECT_BRIEF.md), so this runs everywhere k0/k1 runs and a SKIP is not an answer.
say "4. turb_alive (the physics check; a SKIP is not a PASS)"
./docker/pyrun.sh docker/turb_alive.py "${LAST#$ROOT/}" \
    --json "$JOB_REL/return/turb_alive.json" 2>&1 | tee_
python3 -c "
import json,sys
try: d=json.load(open('$JOB/return/turb_alive.json'))
except Exception as e: print('  *** FAIL: turb_alive wrote no json (%s)'%e); sys.exit()
v=d.get('status')
print('  VERDICT: %s'%(v if v else 'NO VERDICT -- treat as FAIL'))
print('  (a SKIP is not a PASS: k0/k1 is a dt check and stayed at 0.442 through a stable')
print('   seed whose boundary layer had died -- this is the physics check)')" | tee_

# ---- 5. is the grid resolving the stratification ---------------------------------
say "5. Ozmidov scale in Delta at the receptor (the stable rungs died at 3.57)"
./docker/pyrun.sh bin/ozmidov.py "${LAST#$ROOT/}" 2>&1 | tee_

# ---- 6. Gate C2: the saved restart restarts bit-for-bit ---------------------------
say "6. Gate C2: restart with Nt = restart step, re-dump, diff byte-for-byte"
# C2 IS THE ONE STEP THAT NEEDS THE GPU, and docker/run_case.sh refuses a second FastEddy
# container. Say which it is rather than emitting a confusing refusal, so a battery run
# while another seed is on the GPU produces a legible "not yet" instead of a FAIL.
if [ "${SKIP_C2:-0}" = "1" ]; then
  echo "  SKIP_C2=1: deferred. THIS IS NOT A PASS -- rerun before accepting the seed." | tee_
elif [ -n "$(docker ps -q --filter ancestor=flux-fasteddy:cuda118)" ]; then
  # MATCH ON THE IMAGE, the way docker/run_case.sh does. Matching on {{.Command}}
  # does not work: docker truncates it to "/opt/nvidia/nvidia_..." and the grep
  # for FastEddy never fires, so this guard fell through and C2 reported a FAIL
  # that was really a refusal.
  echo "  DEFERRED: a FastEddy container is already running, and only one may run at a" | tee_
  echo "  time. THIS IS NOT A PASS. Rerun step 6 when the GPU is free." | tee_
else
  ./bin/c2_restart_check.sh "$JOB/return/seed_restart.nc" "$TOTAL" "$JOB/seed.in" 2>&1 | tee_
fi

# ---- 7. the 90-degree re-index the whole library rests on -------------------------
say "7. rotation check (static; every corpus case is picked on this convention)"
./docker/pyrun.sh bin/rotation_check.py "$JOB_REL/return/seed_restart.nc" \
    --json "$JOB_REL/return/rotation_check.json" 2>&1 | tee_

# ---- 8. direction: backing, drift, and the projection ----------------------------
say "8. Ekman backing and direction drift"
./docker/pyrun.sh bin/direction_drift.py 2>&1 | tail -30 | tee_

# ---- 9. CONVECTIVE ONLY: is the box organising the thermals? ----------------------
# cbl-deep sits at L/z_i = 2.05, the corpus floor and just outside the 2.28 Phase E
# measured. The failure mode there is not collapse but domain-scale circulation, which
# the stationarity limits cannot see -- so it is diagnosed directly, on the spectrum.
if [ "$REGIME" = "convective" ]; then
  say "9. lock-in: the 2-D spectrum of w at mid-depth (mode-1 share, r at L/2)"
  DUMPS=$(ls -1 "$JOB/output/$OUTBASE".* | sort -t. -k2 -n | tail -4 | tr '\n' ' ')
  ./docker/pyrun.sh bin/domain_adequacy.py spectra $DUMPS 2>&1 | tee_
else
  say "9. lock-in diagnostic: N/A, this rung is $REGIME"
fi

say "battery complete -> $OUT"
date '+%F %H:%M:%S' | tee_
