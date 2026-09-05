#!/usr/bin/env bash
# Run ONE seed spin-up, end to end, on whatever machine this checkout is sitting on.
#
#   usage: bin/run_seed.sh seeds/seed_cbl-mid_a030
#          bin/run_seed.sh seeds/seed_cbl-mid_a030 --dry-run
#          bin/run_seed.sh seeds/seed_cbl-mid_a030 --restart-over   discard a partial run
#
# NO ABSOLUTE PATHS. The repo root is discovered from this script's own location, so a
# rented GPU can check the repo out anywhere; FLUX_ROOT is exported for docker/run_case.sh
# and docker/pyrun.sh, which is why those two now read it instead of a literal.
#
# NO SHARED STATE. Everything this job writes lands under its own job directory. Two seed
# jobs on two machines cannot interact; two on ONE machine are serialised by
# docker/run_case.sh, which refuses to start a second FastEddy container.
#
# NO CHAIN, AND THEREFORE NO RESUME. A seed is ONE continuous FastEddy invocation --
# 738,720 steps, 3.0 simulated hours, ~2.9 h wall. Chaining was retired 2026-08-26 and
# with it the entire failure mode of docs/reference/fasteddy-traps.md 17: a restart READ overwrites every
# IO-registered field (htFlux, z0m, z0t, tskin, topoPos, zPos) with whatever the restart
# file holds, so every segment boundary was an opportunity to inherit state the .in does
# not describe. It cost a whole segment of a stable seed running at zero surface flux
# while its .in asked for -0.012. THE ONLY RESTART LEFT IN THE PROJECT IS SEED -> TARGET.
#
# The price, stated plainly: a killed job now costs the whole run rather than one segment,
# and the one-hour-per-run wall cap no longer applies to a seed. Re-invoking a COMPLETE
# job is still a no-op -- that is idempotence, not a restart.
set -uo pipefail

JOB_ARG="${1:-}"
[ -n "$JOB_ARG" ] || { echo "usage: run_seed.sh <job_dir> [--dry-run]" >&2; exit 64; }
DRY=0; OVER=0
for f in "${@:2}"; do
  case "$f" in
    --dry-run)      DRY=1;;
    --restart-over) OVER=1;;
    *) echo "unknown flag $f" >&2; exit 64;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB="$(cd "$JOB_ARG" && pwd)" || { echo "FATAL: no job dir $JOB_ARG" >&2; exit 66; }
JOB_REL="${JOB#$ROOT/}"
export FLUX_ROOT="$ROOT"
cd "$ROOT"

die(){ echo "FATAL: $*" >&2; exit 1; }
[ -f "$JOB/manifest.json" ] || die "no manifest.json in $JOB"
[ -f "$JOB/seed.in" ] || die "no seed.in in $JOB"

read -r NAME DT FRQ TOTAL WTH OUTBASE WALLMIN ZM ZK DXG < <(python3 - "$JOB/manifest.json" <<'PYMAN'
import json,sys
m=json.load(open(sys.argv[1])); r=m["run"]; g=m.get("gate", {})
print(m["job"], r["dt"], r["frqOutput"], r["steps_total"],
      m["target"]["wth_virtual"], r["outFileBase"], r["projected_wall_min"],
      g.get("zm", 10.0), g.get("k", 2), m.get("grid", {}).get("dx", 16.0))
PYMAN
) || die "manifest.json unreadable"

# ---- SEED_CEILING_H: a HARD ceiling in simulated hours, overriding the manifest --------
#
# === 2.0 SIMULATED HOURS, EVERY RUNG, EVERY REGIME -- set 2026-08-30 ====================
# This replaces the 1.0 h convective / 3.0 h neutral split the ninth pass ran under, and
# the manifests' own standing 3.0 h. The reasoning is per regime and lands on one number:
#
#   * CONVECTIVE rungs need 5-8 large-eddy turnovers, and T* = z_i/w* ~ 350 s here, so
#     band is reached in 30-45 min. 2.0 h is margin for the WEAKLY convective end, where
#     w* is small, T* is correspondingly longer, and the 1.0 h ceiling is genuinely tight.
#   * NEUTRAL rungs do not stabilise at ANY affordable length -- z_i in an Ekman layer with
#     no capping inversion grows for several inertial periods (17.6 h each here), and the
#     ninth pass measured seed_nbl-deep_a015 still climbing at +5.76 %/h after 2.917 h.
#     So hours past the point the flow is turbulent buy nothing that the gate can see, and
#     the extra 1.0 h of the old neutral ceiling was spent on a limit it cannot satisfy.
#
# One number for every rung also makes the library's cost a single figure -- 0.96 GPU-h per
# seed at 0.479 GPU-h/sim-h -- instead of a regime-dependent one.
#
# The ceiling is still a CEILING: bin/seed_watch.sh stops the run as soon as the
# oscillation-immune limits are in band, and a seed that has not entered band by 2.0 h
# stops there and THAT IS THE RESULT -- no extension, no respec.
SEED_CEILING_H="${SEED_CEILING_H:-2.0}"
if [ -n "$SEED_CEILING_H" ]; then
  # ROUNDED TO A WHOLE NUMBER OF DUMPS, WITH A TOLERANCE THE SIZE OF THE FAILURE IT IS
  # LOOKING FOR -- and the previous version was not, so it lost a dump at every ceiling.
  #
  # History, because this is the same bug twice. First a bare `int()`: at dt = 0.0308642
  # and frq = 9720, 1.0*3600/dt/frq evaluates to 11.999999040000077, so int() returned 11
  # and a 1.0 h ceiling silently became 0.917 sim-h. The ninth pass's run 1 is that run.
  # The fix was round()-then-verify -- and the VERIFY re-introduced it, because it rejected
  # any overshoot beyond 1e-6 SECONDS. MEASURED, at the production dt and cadence:
  #
  #     ceiling   round()    n*FRQ*dt - H*3600      old guard        cost
  #       1.0 h      12          +0.288 ms          -> 11 dumps      8.3%
  #       2.0 h      24          +0.576 ms          -> 23 dumps      4.2%
  #       3.0 h      36          +0.864 ms          -> 35 dumps      2.8%
  #
  # The overshoot is not a scheduling error, it is the .in's dt: the true value is
  # 5/162 = 0.030864197530864196 and the file carries 0.0308642, rounded UP at the 7th
  # decimal, so steps*dt always lands a fraction of a millisecond long. The guard exists
  # to catch "the rounding added a WHOLE DUMP", a 300-second event, and it was scoring
  # against one microsecond -- so it fired on an artifact 500,000 times smaller than the
  # thing it can act on, and paid for it in whole dumps.
  #
  # The tolerance is now expressed in DUMPS, which is the unit the decision is made in:
  # accept up to a thousandth of a dump of overshoot (0.3 s at the production cadence),
  # which is four orders of magnitude above the float artifact and three below one dump.
  _NEW=$(python3 -c "
import math
dumps = $SEED_CEILING_H*3600.0/($DT*$FRQ)
n = int(math.floor(dumps + 1e-3))
assert n*$FRQ*$DT <= $SEED_CEILING_H*3600.0 + 1e-3*$FRQ*$DT, 'ceiling overshoot'
print(n*$FRQ)")
  [ "$_NEW" -gt 0 ] || die "SEED_CEILING_H=$SEED_CEILING_H rounds to zero dumps"
  if [ "$_NEW" -lt "$TOTAL" ]; then
    WALLMIN=$(python3 -c "print(f'{$WALLMIN*$_NEW/$TOTAL:.1f}')")
    echo "  SEED_CEILING_H=$SEED_CEILING_H: ceiling $TOTAL -> $_NEW steps "\
"($(python3 -c "print(f'{$_NEW*$DT/3600:.3f}')") sim-h, $(python3 -c "print($_NEW//$FRQ)") dumps)"
    TOTAL="$_NEW"
  else
    echo "  SEED_CEILING_H=$SEED_CEILING_H is at or past the manifest's $TOTAL steps; "\
"keeping $TOTAL ($(python3 -c "print(f'{$TOTAL*$DT/3600:.3f}')") sim-h)"
  fi
fi

# ---- SCORE_H: the gate's window must fit STRICTLY INSIDE the run ------------------------
# --score-h defaults to 2.0 h in bin/seed_stationarity.py, chosen by a sweep inside a 3.0 h
# run. At a 2.0 h ceiling that width is the WHOLE RUN, so the window reaches step 0 -- the
# state the run was handed, which at a cold start has u* = 0 exactly. seed_stationarity.py
# now REFUSES that rather than reporting a trend through the spin-up, so the width has to be
# derived from the run that was actually produced, not inherited from the one it was swept
# on. Leave the first 0.5 h out: at a 2.0 h ceiling that gives a 1.5 h window (the width
# this project used before the sweep), and at any longer run it keeps the swept 2.0 h.
if [ -z "${SCORE_H:-}" ]; then
  SCORE_H=$(python3 -c "
sim_h = $TOTAL*$DT/3600.0
print(f'{min(2.0, max(0.5, sim_h - 0.5)):.3f}')")
  echo "  SCORE_H not set: scoring the last $SCORE_H h of a "\
"$(python3 -c "print(f'{$TOTAL*$DT/3600:.3f}')") sim-h run (the first 0.5 h is excluded)"
fi

echo "########## seed $NAME ##########"
date '+%F %H:%M:%S'
echo "  dt $DT, dump every $FRQ steps, $TOTAL steps in ONE invocation (proj ${WALLMIN} min wall)"

# THE STUB SHORT-CIRCUITS BEFORE THE GPU PREFLIGHT, DELIBERATELY.
# It sat after it at first, and the preflight -- which asks nvidia-smi about the device
# CUDA_VISIBLE_DEVICES names, and in native mode DIES if there is not one -- killed every
# stubbed job before the stub could run. The whole point of the stub is to exercise the
# scheduler on a box with fewer GPUs than workers, so it must not require the GPU it is
# pretending to use.
# ---- STUB_SEED: the whole driver, with the LES and the gate replaced ---------------
#
# THIS EXISTS TO TEST THE SCHEDULER, NOT THE PHYSICS, AND IT CAN NEVER BE MISTAKEN FOR A
# SEED. bin/run_seeds.py fans 30 jobs across N GPUs with a work queue, and the claim that
# a freed GPU picks up the next job is exactly the kind of claim this project does not
# accept as an assertion. Demonstrating it with real seeds costs ~29 GPU-h; demonstrating
# it with the LES stubbed costs seconds and exercises every line of the ORCHESTRATION --
# the queue, the per-GPU assignment, the failure path, the resume path, the summary.
#
# WHAT IT FABRICATES, AND WHAT IT DOES NOT. It writes return/seed_restart.nc,
# return/stationarity.json, return/run.log and return/manifest.json with the shapes the
# orchestrator reads, sleeps STUB_SEED_S seconds to occupy the worker, and exits. It runs
# NO FastEddy, NO stationarity gate and NO acceptance battery.
#
# `meta.stub = true` and `stub: true` go into BOTH json artifacts, the restart is a 1 kB
# text file rather than a 73 MB netCDF, and bin/run_seeds.py refuses to count a stub as
# accepted. That is the same discipline STUB_LES=1 already uses for corpus cases
# (docs/reference/standing-rules.md: "A STUBBED RECORD CAN NEVER MASQUERADE AS A CORPUS RECORD").
#
# STUB_SEED_FAIL=1 makes it fail instead, so "a failed seed frees its GPU rather than
# stranding it" is a thing that can be SHOWN rather than argued.
if [ "${STUB_SEED:-0}" = "1" ]; then
  echo "  *** STUB_SEED=1: no LES, no gate, no battery. This is a SCHEDULER test."
  mkdir -p "$JOB/output" "$JOB/return"
  sleep "${STUB_SEED_S:-2}"
  if [ "${STUB_SEED_FAIL:-0}" = "1" ]; then
    echo "  *** STUB_SEED_FAIL=1: failing deliberately, to show the GPU is released"
    exit 1
  fi
  : > "$JOB/output/$OUTBASE.$TOTAL"
  printf 'STUB. No FastEddy ran. %s steps at dt %s.\nsimulation is complete\n' \
    "$TOTAL" "$DT" > "$JOB/return/run.log"
  printf 'STUB SEED -- NOT A SEED. Produced by bin/run_seed.sh with STUB_SEED=1 to\n'\
'exercise the bin/run_seeds.py work queue. No LES ran. Do not use.\n' \
    > "$JOB/return/seed_restart.nc"
  python3 - "$JOB/manifest.json" "$JOB/return/manifest.json" \
            "$JOB/return/stationarity.json" "$TOTAL" "$DT" <<'PYSTUB'
import json, sys
src, dman, dstat, total, dt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
m = json.load(open(src))
m["stub"] = True
m.setdefault("meta", {})["stub"] = True
m.setdefault("run", {})["ceiling_steps"] = total
m["run"]["ceiling_sim_h"] = total * dt / 3600.0
m["achieved"] = {"stub": True}
json.dump(m, open(dman, "w"), indent=1)
json.dump({"label": m["job"], "stub": True, "pass": False,
           "gated": [], "drifting": [], "indeterminate": [],
           "final": {"stub": True}, "n_dumps": 0},
          open(dstat, "w"), indent=1)
PYSTUB
  echo "########## seed $NAME: STUB (not a seed) ##########"
  exit 0
fi

# ---- preflight: the GPU, before any GPU time is spent -----------------------------
#
# THIS CHECK USED TO ASK THE WRONG GPU A QUESTION WHOSE ANSWER IT THEN GOT WRONG.
# It read `nvidia-smi --query-gpu=compute_cap | head -1` -- GPU 0's capability, whatever
# card this seed was actually going to run on -- compared it against the literal "8.9",
# and, when they differed, reassured the operator that a newer architecture would "JIT
# from PTX, slower but correct".
#
# THAT REASSURANCE WAS FALSE, and on a 5090 it was the difference between a slow run and
# no run. docker/build_fasteddy.sh emitted `-arch=sm_89`, which is shorthand for
# `-gencode arch=compute_89,code=sm_89`: a cubin and NO PTX. There was nothing to JIT
# from. And it is worse than an omission -- MEASURED here, and it is why the deployable
# image carries real SASS for every architecture it supports rather than a PTX fallback:
# FastEddy is built with SEPARATE COMPILATION (-dc then -dlink), and `nvcc -dlink` DROPS
# every PTX image from the fatbin without a word. Adding `-gencode arch=compute_90,
# code=compute_90` puts PTX in the .o files and none of it survives into the executable.
#
# So the question is not "what capability is GPU 0" but "does the binary I am about to run
# contain SASS for the card THIS seed was given", and that is answerable exactly.
GPU_Q="${CUDA_VISIBLE_DEVICES:-0}"; GPU_Q="${GPU_Q%%,*}"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU_Q" 2>/dev/null | head -1 | tr -d ' ')
if [ -z "$CC" ]; then
  # FATAL IN NATIVE MODE, AND THE REASON IS TIMING RATHER THAN SILENCE.
  # MEASURED on the shipped image: a CUDA_VISIBLE_DEVICES naming a card this box does not
  # have makes cudaGetDeviceCount set cudaErrorNoDevice, and gpuErrchk at
  # fecuda_Device.cu:55 prints "GPUassert: no CUDA-capable device is detected" and exits
  # 100. So it fails LOUDLY -- the exit(0) branch at fecuda_Device.cu:75-79 is unreachable
  # on this stack, and an earlier version of this comment wrongly cited it as a silent
  # success. The check stays because failing HERE costs nothing and failing there costs a
  # container start, an mpirun and a confusing log on a machine running fifteen other
  # seeds.
  if [ "${FLUX_NATIVE:-0}" = "1" ]; then
    die "nvidia-smi reports no GPU $GPU_Q. FastEddy would abort in gpuAssert at
      fecuda_Device.cu:55 with 'no CUDA-capable device is detected' and exit 100, after a
      container start and an mpirun. Check CUDA_VISIBLE_DEVICES and that the container was
      started with --gpus all."
  fi
  echo "  WARNING: nvidia-smi gave no compute capability for GPU $GPU_Q" >&2
else
  WANT="sm_$(echo "$CC" | tr -d '.')"
  FE_BIN_CHK="${FE_BIN:-${ROOT}/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy}"
  if [ "${FLUX_NATIVE:-0}" = "1" ] && [ -x "$FE_BIN_CHK" ] && command -v cuobjdump >/dev/null 2>&1; then
    HAVE=$(cuobjdump --list-elf "$FE_BIN_CHK" 2>/dev/null | sed 's/.*\.\(sm_[0-9]*\)\.cubin/\1/' | sort -u | tr '\n' ' ')
    case " $HAVE " in
      *" $WANT "*) echo "  GPU $GPU_Q is $CC -> $WANT, and the binary carries it (has: $HAVE)";;
      *) die "GPU $GPU_Q is compute capability $CC ($WANT) and the FastEddy binary carries
      SASS for [$HAVE] and no PTX at all. It would fail at cuModuleLoad with 'no kernel
      image is available for execution on the device'. Rebuild the image with $WANT in
      FE_GENCODE.";;
    esac
  elif [ "$CC" != "8.9" ]; then
    echo "  WARNING: GPU $GPU_Q is compute capability $CC; this checkout's image is built" >&2
    echo "           for 8.9 (sm_89) with NO PTX, so it will not run there." >&2
  fi
fi
if [ "${FLUX_NATIVE:-0}" != "1" ]; then
  docker image inspect "${FLUX_IMAGE:-flux-fasteddy:cuda118}" >/dev/null 2>&1 \
    || die "image ${FLUX_IMAGE:-flux-fasteddy:cuda118} not present; build it with docker/build_fasteddy.sh"
fi

if [ "$DRY" = "1" ]; then echo "  --dry-run: preflight only, stopping here"; exit 0; fi

mkdir -p "$JOB/output" "$JOB/return"
LOG="$JOB/return/seed.log"

# ---- the run: ONE invocation ------------------------------------------------------
# Idempotent, not resumable. If the final dump is already on disk the LES is skipped and
# the gate is re-scored; anything short of that is re-run from the start, because there is
# no restart within a seed any more.
FINAL_DUMP="$JOB/output/$OUTBASE.$TOTAL"
if [ -f "$FINAL_DUMP" ]; then
  echo "  $OUTBASE.$TOTAL already on disk; skipping the LES and re-scoring the gate"
else
  # A PARTIAL RUN IS REFUSED, NOT SILENTLY WIPED. It cannot be resumed -- resuming is the
  # restart this design retires -- but it is also up to ~3 h of GPU that someone may want
  # to look at, and deleting it on their behalf is not this script's call to make.
  if ls -1 "$JOB/output/$OUTBASE".* >/dev/null 2>&1; then
    if [ "$OVER" = "1" ]; then
      echo "  --restart-over: discarding the partial run in $JOB/output/"
      rm -f "$JOB/output/$OUTBASE".*
    else
      P=$(ls -1 "$JOB/output/$OUTBASE".* | sort -t. -k2 -n | tail -1)
      die "a PARTIAL run is on disk (newest ${P##*/}, wanted step $TOTAL). A seed is one
      continuous invocation and cannot resume -- that restart is exactly what was retired.
      Re-run with --restart-over to discard it and start from step 0, or move it aside."
    fi
  fi
  # ---- Steinfeld spin-up accelerator (neutral rungs) -------------------------------
  # Steinfeld et al. (2008) start a neutral LES with a surface temperature flux of
  # ~0.05 K m/s for the first ~3000 s to trip the transition to resolved turbulence, then
  # remove it. A neutral boundary layer has no buoyant production to organise the initial
  # perturbation field, so it is the slowest regime to spin up -- h/u* ~ 1500 s here
  # against T* ~ 350 s convectively -- and the accelerator is aimed exactly at that.
  #
  # IT COSTS ONE RESTART, AND THE RESTART IS THE DANGEROUS PART. htFlux is IO-registered,
  # so the main invocation would inherit +0.05 from the burn-in dump whatever its .in says
  # (docs/reference/fasteddy-traps.md 17). bin/zero_htflux.py writes the zero into the FILE and reads it
  # back, and the existing per-run htFlux assertion below is the second lock.
  ACC_S="${SEED_ACCEL_S:-0}"
  if [ "$ACC_S" != "0" ]; then
    ACC_NT=$(python3 -c "print(int(round($ACC_S/$DT/$FRQ))*$FRQ)")
    ACC_WTH="${SEED_ACCEL_WTH:-0.05}"
    echo "  --- accelerator burn-in: $ACC_NT steps at surflayer_wth = $ACC_WTH"
    sed -e "s|^Nt = .*|Nt = $ACC_NT|" \
        -e "s|^inPath = .*|inPath = |" -e "s|^inFile = .*|inFile = |" \
        -e "s|^surflayer_wth = .*|surflayer_wth = $ACC_WTH|" \
        -e "s|^outFileBase = .*|outFileBase = ${OUTBASE}_ACC|" \
        "$JOB/seed.in" > "$JOB/accel.in"
    ./docker/run_case.sh "$JOB_REL" "accel.in" "$JOB/return/accel.log" || die "burn-in"
    ACC_LAST="$JOB/output/${OUTBASE}_ACC.$ACC_NT"
    [ -f "$ACC_LAST" ] || die "the burn-in wrote no dump at step $ACC_NT"
    cp -f "$ACC_LAST" "$JOB/FE_ACC.0" || die "staging the burn-in restart"
    ./docker/pyrun.sh bin/zero_htflux.py "${JOB_REL}/FE_ACC.0" --value "$WTH" \
      || die "could not clear the burn-in htFlux out of the restart"
    # ONE RUN PER DIRECTORY, OR IT IS NOT A SERIES (docs/reference/fasteddy-traps.md 18c): the burn-in's
    # dumps carry step numbers that overlap the main run's and would interleave into a
    # single sorted "history" with two states at the same time.
    rm -f "$JOB/output/${OUTBASE}_ACC".*
  fi
  # A COLD START, so inPath/inFile are empty and surflayer_wth is whatever the .in says.
  # With no restart to read, htFlux CANNOT be inherited -- which is the point of retiring
  # the chain. The assertion below is kept anyway: it costs seconds once per run, and
  # docs/reference/standing-rules.md's standing rule is to validate the state the model actually loaded, never
  # the config handed to it.
  if [ "$ACC_S" != "0" ]; then
    sed -e "s|^Nt = .*|Nt = $TOTAL|" \
        -e "s|^inPath = .*|inPath = ./|" -e "s|^inFile = .*|inFile = FE_ACC.0|" \
        -e "s|^surflayer_wth = .*|surflayer_wth = $WTH|" \
        "$JOB/seed.in" > "$JOB/run.in"
  else
    sed -e "s|^Nt = .*|Nt = $TOTAL|" \
        -e "s|^inPath = .*|inPath = |" -e "s|^inFile = .*|inFile = |" \
        -e "s|^surflayer_wth = .*|surflayer_wth = $WTH|" \
        "$JOB/seed.in" > "$JOB/run.in"
  fi
  echo "  --- single invocation -> step $TOTAL ($(python3 -c "print(f'{$TOTAL*$DT/60:.0f}')") min simulated)"
  date '+%F %H:%M:%S'
  rm -f "$JOB/output/.early_stop"
  WATCH_PID=""
  # SCORE_H IS EXPORTED, and it was not. The watcher is a CHILD PROCESS, so a plain shell
  # variable never reached it: bin/seed_watch.sh:30 fell back to its own `${SCORE_H:-2.0}`
  # while the final gate below used the value derived from the run that was actually
  # produced. The two therefore scored DIFFERENT WINDOW WIDTHS -- the watcher deciding
  # when to stop on one width, the verdict written down on another -- which is exactly the
  # kind of disagreement that looks like a physics result.
  export SCORE_H
  # And the ceiling, so the watcher reports the run's own rather than the manifest's.
  export SEED_TOTAL_STEPS="$TOTAL"
  if [ "${SEED_EARLY_STOP:-1}" = "1" ]; then
    # OPEN-ENDED WITH A CEILING. Nt is the ceiling; the watcher usually ends the run
    # sooner and stamps the step it stopped at. See bin/seed_watch.sh for why the
    # criterion is the oscillation-immune limits only.
    # setsid, so the watcher is a process GROUP LEADER. `kill $WATCH_PID` reaches only
    # the watcher's bash -- and the watcher spends most of its life blocked inside a
    # seed_stationarity.py it launched, so killing the shell ORPHANS that child. One per
    # seed is a curiosity on a workstation; sixteen per wave on a rented box is a leak
    # that outlives the run it belonged to.
    setsid ./bin/seed_watch.sh "$JOB" & WATCH_PID=$!
  fi
  ./docker/run_case.sh "$JOB_REL" "run.in" "$JOB/return/run.log"; RC_RUN=$?
  [ -n "$WATCH_PID" ] && { kill -TERM -- "-$WATCH_PID" 2>/dev/null || kill "$WATCH_PID" 2>/dev/null; }
  # A container the watcher stopped exits non-zero, which is not a failure -- but a run
  # that failed for any OTHER reason must still fail. Distinguish on the marker, and on
  # the artifact, never on the exit status alone (docs/reference/fasteddy-traps.md 12).
  if [ "$RC_RUN" != "0" ] && [ ! -f "$JOB/output/.early_stop" ]; then
    die "the seed run"
  fi
  cat "$JOB/return/run.log" >> "$LOG"
fi
LAST=$(ls -1 "$JOB/output/$OUTBASE".* | sort -t. -k2 -n | tail -1)
[ -n "$LAST" ] || die "the run wrote no dump"
# EARLY STOP. With SEED_EARLY_STOP=1 a watcher scores the trailing window every 30
# simulated minutes and stops the run as soon as the oscillation-immune limits are in
# band, so the newest dump is legitimately short of Nt -- and Nt is then a CEILING rather
# than a target. It stamps the step it stopped at, so "short" is only accepted when
# something actually decided to stop there; a crash still fails, which is the distinction
# that matters.
if [ "${LAST##*.}" != "$TOTAL" ]; then
  if [ -f "$JOB/output/.early_stop" ] && \
     [ "$(cat "$JOB/output/.early_stop")" = "${LAST##*.}" ]; then
    echo "  EARLY STOP at step ${LAST##*.} of a $TOTAL ceiling "
    echo "    = $(python3 -c "print(f'{${LAST##*.}*$DT/3600:.2f}')") of "\
         "$(python3 -c "print(f'{$TOTAL*$DT/3600:.2f}')") simulated hours"
  else
    die "newest dump is step ${LAST##*.}, wanted $TOTAL and no .early_stop marker matches"
  fi
fi
# ASSERT ON THE ARTIFACT: the flux the run actually USED, not the one its .in asked for.
./docker/pyrun.sh - "${LAST#$ROOT/}" "$WTH" <<'PYCHK' || die "the run used the wrong surface flux"
import sys, numpy as np
from netCDF4 import Dataset
path, want = sys.argv[1], float(sys.argv[2])
with Dataset(path) as ds:
    got = float(np.asarray(ds["htFlux"][:]).mean())
if abs(got - want) > 1e-6:
    print(f"FATAL: the dump carries htFlux {got:+.6f}, the run asked for {want:+.6f}",
          file=sys.stderr)
    raise SystemExit(1)
print(f"  htFlux confirmed {got:+.6f}")
PYCHK

# ---- the gate, HERE, so the 300 s dumps never travel -------------------------------
# ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS (docs/reference/fasteddy-traps.md 12): the gate's stdout is
# tee'd, so $? would be tee's. The verdict is read back out of the JSON it wrote.
# The gate scores the LAST 1.5 h, which is past the warm-up, so the flux it needs for the
# Kljun terms is the TARGET one and not the zero the first segment ran under.
./docker/pyrun.sh bin/seed_stationarity.py "$JOB_REL/output" --dt "$DT" --wth "$WTH" \
    --zm "$ZM" --k "$ZK" --dx "$DXG" ${SCORE_H:+--score-h $SCORE_H} \
    --json "$JOB_REL/return/stationarity.json" --label "$NAME" \
    2>&1 | tee "$JOB/return/stationarity.txt"
[ -s "$JOB/return/stationarity.json" ] \
  || { tail -20 "$JOB/return/stationarity.txt" >&2; die "the gate wrote no JSON"; }
VERDICT=$(python3 -c "import json;print('PASS' if json.load(open('$JOB/return/stationarity.json'))['pass'] else 'FAIL')")

# ---- what goes home ----------------------------------------------------------------
FINAL=$(ls -1 "$JOB/output/$OUTBASE".* | sort -t. -k2 -n | tail -1)
cp -f "$FINAL" "$JOB/return/seed_restart.nc" || die "could not stage the restart"
cp -f "$JOB/manifest.json" "$JOB/return/manifest.json"
python3 - "$JOB/return/manifest.json" "$JOB/return/stationarity.json" "$(basename "$FINAL")" \
        "$TOTAL" "$(python3 -c "print(f'{$TOTAL*$DT/3600:.6f}')")" <<'PY'
import json,sys
man=json.load(open(sys.argv[1])); st=json.load(open(sys.argv[2]))
man["achieved"]=st["final"]                 # LABEL THE SEED BY WHAT IT ACHIEVED, not by
man["achieved"]["pass"]=st["pass"]          # what it was asked for -- pick_seed.py reads
man["achieved"]["source_dump"]=sys.argv[3]  # these, and docs/les/seed-library.md says the same for
# THE CEILING THIS RUN WAS ACTUALLY HELD TO, STAMPED INTO THE ARTIFACT.
# steps_total in the manifest is the job's DESIGN ceiling (349920 = 3.0 sim-h for every
# seeds job); SEED_CEILING_H is applied inside this script and, until now, left no trace.
# bin/seed_accept.sh then reported "1.92 of 3.00 simulated hours" against a ceiling the run
# never had -- and could only avoid it if the operator happened to have SEED_CEILING_H set
# in their own shell. Assert on the artifact, not on an inherited environment variable.
man.setdefault("run",{})["ceiling_steps"]=int(sys.argv[4])
man["run"]["ceiling_sim_h"]=float(sys.argv[5])
json.dump(man,open(sys.argv[1],"w"),indent=1) # direction.
PY
SZ=$(du -sh "$JOB/return" | cut -f1)
echo
echo "########## seed $NAME: $VERDICT ##########"
echo "  return/ is $SZ  ($(ls "$JOB/return" | tr '\n' ' '))"
date '+%F %H:%M:%S'
# === THE EXIT STATUS ANSWERS "DID THE RUN PRODUCE A SEED", NOT "DID THE GATE PASS" =====
# CHANGED 2026-08-31. This was `[ "$VERDICT" = "PASS" ] || exit 1`, and on the 30-seed
# Blackwell library that returned 1 for ALL THIRTY -- every seed came back INDETERMINATE or
# DRIFTING and not one returned a clean PASS. A status that is identical for every possible
# outcome discriminates nothing, and it discriminates nothing in the DANGEROUS direction:
# anything downstream keying on it reads a complete, usable, fully-battery-tested seed as a
# failed run. bin/run_seeds.py survived only because it judges on the artifact.
#
# It is also the wrong question now. Seed selection uses the WHOLE library as of
# 2026-08-31 (bin/pick_seed.py, and the reasoning is in bin/run_corpus_case.sh): a seed is
# an initial condition, the case adjusts under its own forcing, and the ML inputs come from
# window_stats over the footprint's own window -- so a non-PASS verdict does not disqualify
# the artifact and must not be reported as an error.
#
# The verdict is NOT lost: it is in return/stationarity.json, in manifest.achieved.pass, on
# the machine manifest, and stamped onto every pair as seed.gate_state. That is docs/les/seed-library.md's
# standing rule -- assert on the ARTIFACT, not on the exit status -- applied to this
# script's own output rather than to FastEddy's.
#
# SEED_STRICT_EXIT=1 restores the old signal for a caller that genuinely wants PASS-or-fail.
if [ ! -s "$JOB/return/seed_restart.nc" ]; then
  echo "  EXIT 1: the run produced no seed_restart.nc" >&2
  exit 1
fi
if [ "${SEED_STRICT_EXIT:-0}" = "1" ] && [ "$VERDICT" != "PASS" ]; then
  echo "  EXIT 1: SEED_STRICT_EXIT=1 and the gate returned $VERDICT (the artifact is" >&2
  echo "          complete and usable; this exit is the caller's own strictness)" >&2
  exit 1
fi
[ "$VERDICT" = "PASS" ] || echo "  exit 0: the seed is complete. The gate returned" \
    "$VERDICT -- carried in return/stationarity.json and manifest.achieved.pass, not in" \
    "this exit status. SEED_STRICT_EXIT=1 to fail on it."
exit 0
