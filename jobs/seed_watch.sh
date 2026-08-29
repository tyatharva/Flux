#!/usr/bin/env bash
# Watch a running seed and stop it as soon as it is stationary. THE BUDGET IS MEASURED.
#
# WHY. The 3.0 simulated hours every seed used to run was derived once, at dx = 16 m, on
# nbl-shallow, as "the first duration where all seven limits passed". It is not a property
# of the library -- convective rungs turn over on z_i/w* (T* ~ 350 s) and neutral ones on
# h/u* (~1500 s), a factor of four -- and it does not transfer to a different grid at all.
# So the seed runs OPEN-ENDED and this watcher decides when it is done. The measured stop
# times are what set the library's real budget.
#
# 3.0 SIMULATED HOURS IS A HARD CEILING, NOT A TARGET. A seed that has not entered band by
# then stops there and that IS the result -- no extension, no respec. Nt in the .in is the
# ceiling; this script is what usually ends the run earlier.
#
# THE CRITERION IS THE OSCILLATION-IMMUNE LIMITS, and only those. U/u*, sigma_v/u*,
# sigma_w/u* and Kljun's x_peak and x90 are ratios (or functions of ratios) whose numerator
# and denominator ride the 17.6 h inertial oscillation together, so they can reach a band
# and stay there. TKE_BL/u*^2 and z_i cannot be resolved against their thresholds at ANY
# window width in a 3 h run -- n_eff saturates at 3-5 because they decorrelate on the eddy
# turnover, not on the dump interval -- so requiring them here would mean never stopping
# early and would misreport WHY. They are still SCORED, and a DRIFTING verdict on any limit
# blocks the stop: unestablished stationarity is not stationarity, but neither is it drift.
#
# usage: seed_watch.sh <job_dir> &      (run_seed.sh starts it; it exits when the run does)
set -uo pipefail
JOB="$(cd "${1:?usage: seed_watch.sh <job_dir>}" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"
POLL_SIM_S="${POLL_SIM_S:-1800}"     # score every this many SIMULATED seconds
MIN_SIM_S="${MIN_SIM_S:-3600}"       # never stop before this much simulated time
SCORE_H="${SCORE_H:-2.0}"            # trailing window the gate scores, h

read -r NAME DT FRQ TOTAL WTH OUTBASE ZM ZK < <(python3 - "$JOB/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); r=m["run"]; g=m.get("gate",{})
print(m["job"], r["dt"], r["frqOutput"], r["steps_total"], m["target"]["wth_virtual"],
      r["outFileBase"], g.get("zm",10.0), g.get("k",2))
PY
) || exit 0

echo "  [watch] $NAME: polling every ${POLL_SIM_S} simulated s, floor ${MIN_SIM_S} s, "\
     "ceiling $(python3 -c "print(f'{$TOTAL*$DT:.0f}')") s"
last_scored=0
while true; do
  sleep 60
  # Has the run ended on its own?
  pgrep -f "FEMAIN/FastEddy" >/dev/null 2>&1 || \
    docker ps --format '{{.Image}}' | grep -q flux-fasteddy || { echo "  [watch] run ended"; exit 0; }
  LAST=$(ls -1 "$JOB/output/$OUTBASE".[0-9]* 2>/dev/null | sort -t. -k2 -n | tail -1)
  [ -n "$LAST" ] || continue
  STEP=${LAST##*.}
  SIM=$(python3 -c "print(int($STEP*$DT))")
  [ "$SIM" -ge "$MIN_SIM_S" ] || continue
  [ $((SIM - last_scored)) -ge "$POLL_SIM_S" ] || continue
  # THE NEWEST DUMP MAY STILL BE BEING WRITTEN. Score only once its size has been stable
  # for one poll -- a half-written netCDF is not a state, and the gate would either throw
  # or, worse, read a truncated field and print a number.
  s1=$(stat -c%s "$LAST"); sleep 5; s2=$(stat -c%s "$LAST")
  [ "$s1" = "$s2" ] || continue
  last_scored=$SIM
  SW=$(python3 -c "print(f'{min($SCORE_H, $SIM/3600.0*0.5):.3f}')")
  ./docker/pyrun.sh bin/seed_stationarity.py "${JOB#$ROOT/}/output" --dt "$DT" \
      --wth "$WTH" --zm "$ZM" --k "$ZK" --score-h "$SW" \
      --json "${JOB#$ROOT/}/output/.watch.json" --label "$NAME" >/dev/null 2>&1
  V=$(python3 - "$JOB/output/.watch.json" <<'PY'
import json,sys
IMMUNE={"U/u* (Kljun Pi_4)","sigma_v/u*","sigma_w/u* at the receptor",
        "Kljun x_peak","Kljun x90"}
try: d=json.load(open(sys.argv[1]))
except Exception: print("ERR 0 0"); raise SystemExit
rows=d.get("gated",[])
imm=[r for r in rows if r["name"] in IMMUNE]
drift=[r["name"] for r in rows if r["ok"] is False]
inband = bool(imm) and all(r["ok"] is True for r in imm) and not drift
print(("INBAND" if inband else "WAIT"), len(imm), len(drift))
PY
)
  set -- $V
  echo "  [watch] $(date +%H:%M:%S) sim ${SIM}s (score ${SW} h): $1 "\
       "(immune ok=$2, drifting=$3)"
  if [ "$1" = "INBAND" ]; then
    echo "$STEP" > "$JOB/output/.early_stop"
    echo "  [watch] STOPPING: the oscillation-immune limits are in band at step $STEP "\
         "= $(python3 -c "print(f'{$SIM/3600.0:.2f}')") simulated hours"
    for c in $(docker ps -q --filter ancestor=flux-fasteddy:cuda118); do
      if docker inspect -f '{{json .Config.Cmd}}' "$c" 2>/dev/null | grep -q 'FEMAIN/FastEddy'; then
        # SIGINT-then-wait, so the current dump completes rather than being truncated.
        docker stop -t 30 "$c" >/dev/null 2>&1
      fi
    done
    exit 0
  fi
done
