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
# THE CEILING THE RUN ACTUALLY HAS, NOT THE MANIFEST'S. Every jobs30 manifest carries
# steps_total = 349920 (3.0 sim-h) and SEED_CEILING_H -- 2.0 h by default -- is applied
# inside jobs/run_seed.sh, which now exports the resulting step count. Without it this
# script announced a 10800 s ceiling for a run that was going to stop at 7200. The watcher
# does not ACT on the number, but a log that names a ceiling the run never had is the same
# defect just fixed in bin/seed_accept.sh, and the fix costs one line.
TOTAL="${SEED_TOTAL_STEPS:-$TOTAL}"

echo "  [watch] $NAME: polling every ${POLL_SIM_S} simulated s, floor ${MIN_SIM_S} s, "\
     "ceiling $(python3 -c "print(f'{$TOTAL*$DT:.0f}')") s"
last_scored=0
# NATIVE MODE: THE WATCHER MUST FIND, AND STOP, EXACTLY ITS OWN RUN.
#
# On the host both of those were machine-global -- liveness was `pgrep -f FEMAIN/FastEddy`
# OR "any container from the image", and the stop was `docker stop` on EVERY container
# whose command matched FastEddy. On a one-GPU workstation, where only one FastEddy may
# run at a time by construction, global and own are the same set. On a 16-GPU box they are
# not: the first seed to reach band would stop the other fifteen, mid-dump, and each of
# them would then be judged on a truncated run.
#
# docker/run_case.sh puts each run in its own process group with setsid and writes the
# group leader's pid to <job>/.fe.pid, so both questions have an exact answer here.
NATIVE="${FLUX_NATIVE:-0}"
PIDF="$JOB/.fe.pid"
# THE PID IS ONLY TRUSTED IF ITS START TIME STILL MATCHES. docker/run_case.sh writes
# "<pid> <starttime>", where starttime is field 22 of /proc/<pid>/stat -- ticks since boot.
# A pid alone is not an identity: a container killed mid-seed leaves this file, a fresh
# container's PID namespace starts at 1, and the low number it holds very likely names some
# OTHER live process by then. Without the check the watcher would eventually send
# `kill -TERM -- -<pgid>` at it.
fe_pid(){
  local p st cur
  [ -f "$PIDF" ] || return 1
  read -r p st < "$PIDF" 2>/dev/null || return 1
  [ -n "$p" ] || return 1
  cur=$(awk '{print $22}' "/proc/$p/stat" 2>/dev/null) || return 1
  if [ -n "$st" ] && [ "$st" != "0" ] && [ -n "$cur" ] && [ "$st" != "$cur" ]; then
    echo "  [watch] .fe.pid names pid $p but its start time differs ($st vs $cur):" >&2
    echo "          that is a DIFFERENT process. Refusing to act on it." >&2
    return 1
  fi
  printf '%s' "$p"
}
# THE WATCHER STARTS BEFORE THE RUN DOES, so "no pid file" means two different things and
# they must not be conflated. run_seed.sh backgrounds this script and only then calls
# run_case.sh, which writes .fe.pid once FastEddy is launched -- a second or two later.
# Without the latch the very first liveness check would find no pid file, conclude "run
# ended", and exit: the watcher would be gone before the run began, no seed would ever
# stop early, and every one of them would burn the full ceiling with nothing to say why.
SEEN=0
fe_alive(){
  if [ "$NATIVE" = "1" ]; then
    local p; p=$(fe_pid)
    if [ -n "$p" ]; then SEEN=1; kill -0 "$p" 2>/dev/null; return; fi
    [ "$SEEN" = "0" ]     # not started YET is alive; started and gone is not
  else
    pgrep -f "FEMAIN/FastEddy" >/dev/null 2>&1 || \
      docker ps --format '{{.Image}}' | grep -q flux-fasteddy
  fi
}
fe_stop(){
  if [ "$NATIVE" = "1" ]; then
    local p; p=$(fe_pid)
    [ -n "$p" ] || { echo "  [watch] no pid file; cannot stop"; return; }
    # SIGTERM to the whole GROUP (mpirun and its FastEddy child), then wait, exactly as
    # `docker stop -t 30` did: give the current dump a chance to complete rather than
    # truncating a netCDF that the gate would then read as a state.
    kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
    for _ in $(seq 30); do kill -0 "$p" 2>/dev/null || return; sleep 1; done
    kill -KILL -- "-$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null
  else
    for c in $(docker ps -q --filter ancestor="${FLUX_IMAGE:-flux-fasteddy:cuda118}"); do
      if docker inspect -f '{{json .Config.Cmd}}' "$c" 2>/dev/null | grep -q 'FEMAIN/FastEddy'; then
        # SIGINT-then-wait, so the current dump completes rather than being truncated.
        docker stop -t 30 "$c" >/dev/null 2>&1
      fi
    done
  fi
}

while true; do
  sleep 60
  # Has the run ended on its own?
  fe_alive || { echo "  [watch] run ended"; exit 0; }
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
# THE SECOND FIELD IS "HOW MANY IMMUNE LIMITS ARE IN BAND", not "how many were found".
# It printed len(imm) -- the number of immune ROWS -- so a poll where all five existed but
# only two had resolved read "immune ok=5 ... WAIT", which looks like a contradiction and
# hides the actual reason for waiting: ok is None (INDETERMINATE) is neither True nor
# False, so it blocks the stop without appearing in the drifting count.
nok=sum(1 for r in imm if r["ok"] is True)
nind=sum(1 for r in imm if r["ok"] is None)
print(("INBAND" if inband else "WAIT"), f"{nok}/{len(imm)}", len(drift), nind)
PY
)
  set -- $V
  echo "  [watch] $(date +%H:%M:%S) sim ${SIM}s (score ${SW} h): $1 "\
       "(immune in band $2, drifting=$3, immune indeterminate=$4)"
  if [ "$1" = "INBAND" ]; then
    echo "$STEP" > "$JOB/output/.early_stop"
    echo "  [watch] STOPPING: the oscillation-immune limits are in band at step $STEP "\
         "= $(python3 -c "print(f'{$SIM/3600.0:.2f}')") simulated hours"
    fe_stop
    exit 0
  fi
done
