#!/usr/bin/env bash
# Run one FastEddy case and SCORE IT.
# usage: run_case.sh <case_dir> <case_file.in> [logfile]
#
# Always routes through check_run.sh, because FastEddy exits 0 on fully-NaN fields.
#
# TWO MODES, ONE SCRIPT.
#
#   FLUX_NATIVE unset (the host)  -- `docker run` a FastEddy container, as before, with a
#                                    MACHINE-GLOBAL "only one FastEddy" mutex. That is the
#                                    right rule for a one-GPU workstation.
#   FLUX_NATIVE=1  (inside the portable image) -- exec FastEddy directly, with a PER-GPU
#                                    mutex. On a 16-GPU box the machine-global rule would
#                                    serialise all sixteen cards onto one seed at a time,
#                                    turning a 2-hour job into a 30-hour one; what the rule
#                                    actually protects against is two runs sharing ONE GPU
#                                    and interleaving their dumps, and that is per-device.
#
# THE GPU IS SELECTED WITH CUDA_VISIBLE_DEVICES AND NOTHING ELSE.
# FastEddy picks its device at SRC/FECUDA/fecuda_Device.cu:59 as
# `cudaSetDevice(mpi_rank_world % numDevs)`, and every seed is `mpirun -np 1`, so the rank
# is always 0 and the expression is always device 0. That is not a limitation, because
# CUDA_VISIBLE_DEVICES masks the enumeration itself: MEASURED in this image, with
# CUDA_VISIBLE_DEVICES=0 cudaGetDeviceCount() returns 1 and device 0 is the physical card;
# unset, it returns all of them. So `0 % 1 = 0` lands on exactly the card the variable
# names, and no FastEddy source change is needed for per-GPU placement.
#
# AN OUT-OF-RANGE INDEX FAILS LOUDLY, AND AN EARLIER VERSION OF THIS COMMENT SAID THE
# OPPOSITE. It claimed cudaGetDeviceCount() returns 0 and fecuda_DeviceSetup takes its
# `else` branch -- which does exist (fecuda_Device.cu:75-79, printing "No CUDA devices
# found...exiting now!" and calling exit(0)) -- and concluded that a bad index is a silent
# success. MEASURED on the shipped image: it is not. cudaGetDeviceCount sets
# cudaErrorNoDevice, and `gpuErrchk(cudaPeekAtLastError())` at fecuda_Device.cu:55 fires
# FIRST -- gpuAssert (fecuda_Device_cu.h:28-36) prints
#     GPUassert: no CUDA-capable device is detected ../FECUDA/fecuda_Device.cu 55
# and exits 100, so the exit(0) branch is unreachable on this stack. bin/run_seeds.py and
# jobs/run_seed.sh still assert the device exists before spending anything, because failing
# at the preflight beats failing after the container has started -- but they are not
# guarding against silence.
set -uo pipefail
# Repo root is a variable so a seed job can run on a rented GPU whose checkout is
# somewhere else. Defaults to this machine, so nothing that already works changes.
FLUX_ROOT="${FLUX_ROOT:-/home/atyagi/Flux}"
FLUX_IMAGE="${FLUX_IMAGE:-flux-fasteddy:cuda118}"
NATIVE="${FLUX_NATIVE:-0}"
# Where the per-GPU flock files live. /var/lock is created in the image; a workstation
# running native mode by hand can point this anywhere writable.
FLUX_LOCKDIR="${FLUX_LOCKDIR:-/var/lock/flux}"
FE_BIN="${FE_BIN:-${FLUX_ROOT}/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy}"

CASE_DIR="$1"; CASE_FILE="$2"

# A CASE DIRECTORY MAY BE ABSOLUTE. On the host every caller passes a repo-relative path,
# because the host branch mounts the repo at /work and a host path does not exist inside
# the container. In native mode there is no mount and therefore no translation, and the
# orchestrator deliberately puts each seed's working directory on the MOUNTED OUTPUT
# VOLUME -- a 2 sim-h seed writes ~1.8 GB of dumps and 16 of them would otherwise land in
# the container's overlay layer, where they are slow to write and invisible afterwards.
abspath(){ case "$1" in /*) printf '%s' "$1";; *) printf '%s/%s' "${FLUX_ROOT}" "$1";; esac; }
CASE_ABS="$(abspath "$CASE_DIR")"

# ---- the mutex -------------------------------------------------------------------
# One GPU, one run. Two FastEddy processes writing the same output/ silently interleave
# their dumps and corrupt both -- and it looks like a mysteriously stalled run, not an
# error. Refuse rather than race.
if [ "$NATIVE" = "1" ]; then
  # PER-GPU. Scan /proc for another FastEddy whose CUDA_VISIBLE_DEVICES equals ours.
  # /proc/<pid>/environ is NUL-separated; this reads the value the process was STARTED
  # with, which is what CUDA actually honoured, rather than what some parent shell has now.
  # THE COMPARISON IS ON A NORMALISED DEVICE SET, NOT ON THE RAW STRING. Unset means EVERY
  # device (and FastEddy then lands on device 0 via `rank % numDevs`); "0,1" and a
  # "GPU-<uuid>" both also name device 0; none of the three is the string "0". A raw
  # comparison therefore lets `verify`, a standalone `seed`, or an `accept` run -- all of
  # which leave CUDA_VISIBLE_DEVICES unset -- start a second FastEddy on a card a worker is
  # already using. Normalise to the FIRST index each side actually lands on, and treat
  # unset as device 0, which is where FastEddy will go.
  _first_dev(){ local v="${1-}"; v="${v%%,*}"; printf '%s' "${v:-0}"; }
  MYDEV="${CUDA_VISIBLE_DEVICES-}"
  MYFIRST="$(_first_dev "$MYDEV")"
  # THE SCAN IS A FUNCTION, so the guard below and the self-test after the launch run
  # EXACTLY the same code. A mutex whose failure mode is silence needs a positive test,
  # and a second copy of the scan would be a second thing to get wrong.
  #
  # AND NOTE WHAT IS **NOT** USED HERE: `read -r _cmd < "$p/cmdline" || continue`.
  # /proc/<pid>/cmdline is NUL-separated with NO trailing delimiter, so `read` hits EOF and
  # RETURNS 1 even though it assigned the value correctly. MEASURED:
  #     sleep 5 & { read -r c < /proc/$!/cmdline; }; echo $? "[$c]"   ->   1 [sleep5]
  # Gating on that status `continue`s on every pid, `busy` is always empty, and the guard
  # silently permits exactly what it exists to forbid. It was written that way for one
  # revision here, to suppress a "/proc/196/cmdline: No such file" race message -- a
  # cosmetic fix that disabled a correctness guard. Read the file, then test the VALUE.
  # IT MATCHES THE PROCESS'S ACTUAL EXECUTABLE, NOT TEXT IN ITS COMMAND LINE.
  # The first version tested whether the whole cmdline CONTAINED "FEMAIN/FastEddy", and
  # that matches far more than FastEddy: any shell running a script whose text mentions the
  # path, a `bash -c` wrapper, a grep for it, an editor with the file open. MEASURED -- the
  # mutex race test refused all eight racers, and the pids it named were the test harness's
  # own shells, because the harness script contains the string. A guard that refuses
  # legitimate runs is worse than one that is merely loose: it would strand a GPU.
  #
  # /proc/<pid>/exe is the resolved binary and cannot be spoofed by text. argv[0] -- the
  # first NUL-separated cmdline field, which is the path mpirun execs FastEddy with -- is
  # the fallback for the case where exe is unreadable. Neither matches a shell.
  fe_on_gpu(){
    local want="$1" p pid exe a0 theirs out=""
    for p in /proc/[0-9]*; do
      pid="${p#/proc/}"
      [ "$pid" = "$$" ] && continue
      exe=$(readlink "$p/exe" 2>/dev/null)
      if [ -z "$exe" ]; then
        a0=$({ tr '\0' '\n' < "$p/cmdline"; } 2>/dev/null | head -1) || continue
        exe="$a0"
      fi
      case "$exe" in */FEMAIN/FastEddy|*/FEMAIN/FastEddy' (deleted)') ;; *) continue;; esac
      theirs=$({ tr '\0' '\n' < "$p/environ"; } 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
      [ "$(_first_dev "$theirs")" = "$want" ] && out="$out $pid"
    done
    printf '%s' "$out"
  }
  # THE MUTEX IS AN flock. THE /proc SCAN IS THE DIAGNOSTIC, NOT THE LOCK.
  #
  # A scan is a check-then-act: two processes can both scan, both see nothing, and both
  # launch. Under bin/run_seeds.py that cannot happen -- one worker thread per GPU -- but
  # the whole reason this guard exists is the runs that are NOT the orchestrator: a
  # hand-run `seed`, an `accept` whose Gate C2 needs the card, a second `verify`. Those
  # race by construction, and losing the race means two FastEddys interleaving dumps into
  # one output/, which looks like a stalled run rather than an error.
  #
  # flock(2) on a per-device file is atomic and needs no polling. fd 9 is held for the life
  # of this script -- including through the `exec` into check_run.sh at the end, so the
  # card stays claimed while its dump is being scored. The scan then runs only to NAME the
  # holder, which a bare "resource busy" cannot.
  #
  # SCOPE, STATED: flock and /proc are both per-container. Two CONTAINERS on one host share
  # neither, so neither form of this guard sees across them. On Vast there is one container
  # per instance and the question does not arise; on a workstation the host branch's
  # `docker ps` is what covers it.
  mkdir -p "$FLUX_LOCKDIR" 2>/dev/null || true
  _LOCKF="${FLUX_LOCKDIR}/fe.gpu${MYFIRST}.lock"
  exec 9>"$_LOCKF" || { echo "FATAL: cannot open the GPU lock $_LOCKF" >&2; exit 1; }
  if ! flock -n 9; then
    busy="$(fe_on_gpu "$MYFIRST")"
    echo "  REFUSED: a FastEddy run already holds GPU ${MYFIRST}${busy:+ (pid$busy)}" >&2
    echo "           CUDA_VISIBLE_DEVICES here is '${MYDEV:-<unset, which means device 0>}'" >&2
    exit 2
  fi
  # The lock is held. The scan is now a consistency check rather than the guard: if it
  # finds a FastEddy on this device while we hold the lock, something started outside this
  # container or outside this script, and that is worth saying out loud.
  busy="$(fe_on_gpu "$MYFIRST")"
  if [ -n "$busy" ]; then
    echo "  REFUSED: the GPU ${MYFIRST} lock was free but a FastEddy is running on it" >&2
    echo "           (pid$busy) -- it was started outside this script. Not racing it." >&2
    exit 2
  fi
else
  # MACHINE-GLOBAL, the host rule. Match on the FastEddy BINARY, not on the image:
  # analysis scripts share the image and only use the CPU, so an image-level filter blocks
  # the very runs it is meant to protect.
  busy=""
  for c in $(docker ps -q --filter ancestor="${FLUX_IMAGE}"); do
    if docker inspect -f '{{json .Config.Cmd}}' "$c" 2>/dev/null | grep -q 'FEMAIN/FastEddy'; then
      busy="$busy $c"
    fi
  done
  if [ -n "$busy" ]; then
    echo "  REFUSED: a FastEddy run is already in progress:" >&2
    # ONE --filter PER ID. `--filter "id=$busy"` with a space-joined list is one malformed
    # filter value that matches nothing, so the refusal named no container in exactly the
    # case -- more than one running -- where knowing which would matter.
    for c in $busy; do
      docker ps --format '    {{.Names}} {{.Status}} {{.Command}}' --filter "id=$c" >&2
    done
    exit 2
  fi
fi

# A MISSING RESTART FILE DOES NOT ABORT FastEddy. It prints "Error: No such file or
# directory", carries on with x,y,z dimensions of 0, and produces a run in which every
# cell of every field is NaN -- while still exiting 0. That cost a 30-minute segment once.
# Check the file exists before spending GPU time on it.
_cf="${CASE_ABS}/${CASE_FILE}"
# A LINE OF 256 CHARACTERS OR MORE IN THE .in SEGFAULTS FastEddy BEFORE IT STARTS.
# parameters.c:28 sets MAXLEN 256 and reads with fgets(strBuff, MAXLEN, ...), so a longer
# line is split. The first piece parses; the CONTINUATION is a fragment with no '=', and
# parameters.c:126-133 then calls str_trim(valueBuff) on the NULL that strchr returned.
# The result is "Signal: Segmentation fault ... Failing at address: (nil)" with a six-frame
# libc backtrace and no mention of the file or the line -- so the natural reading is that
# the model is broken, not that a COMMENT is too long. Cost: one acceptance run, and it
# would have cost every run of the pass, because the offending line was in the TEMPLATE.
if [ -f "$_cf" ]; then
  _long=$(awk 'length($0) >= 255 {print NR": "length($0)" chars"}' "$_cf" | head -3)
  if [ -n "$_long" ]; then
    echo "  REFUSED: ${CASE_FILE} has line(s) at or over 255 characters, which FastEddy's" >&2
    echo "           parameter parser cannot read (parameters.c:28 MAXLEN 256). It would" >&2
    echo "           segfault at address (nil) with no reference to the file:" >&2
    echo "$_long" | sed 's|^|             line |' >&2
    exit 4
  fi
fi
if [ -f "$_cf" ]; then
  _ip=$(grep -oP '^inPath\s*=\s*\K[^#[:space:]]*' "$_cf" || true)
  _if=$(grep -oP '^inFile\s*=\s*\K[^#[:space:]]*' "$_cf" || true)
  if [ -n "${_if:-}" ] && [ ! -f "${CASE_ABS}/${_ip}${_if}" ]; then
    echo "  REFUSED: restart file ${_ip}${_if} not found relative to ${CASE_DIR}" >&2
    echo "           (FastEddy would run to completion and write only NaN)" >&2
    exit 3
  fi
fi
LOG="${3:-/tmp/$(basename "$CASE_FILE" .in).log}"
mkdir -p "$(dirname "$LOG")"

if [ "$NATIVE" = "1" ]; then
  # ---- native: exec FastEddy here ------------------------------------------------
  [ -x "$FE_BIN" ] || { echo "FATAL: no FastEddy binary at $FE_BIN" >&2; exit 1; }
  # A PID FILE FROM A PREVIOUS, KILLED RUN IS A LIVE HAZARD, not stale clutter -- see the
  # start-time note below. Remove it before this run can be confused with that one, and
  # arrange to remove ours however this shell exits.
  rm -f "${CASE_ABS}/.fe.pid"
  trap 'rm -f "${CASE_ABS}/.fe.pid"' EXIT INT TERM
  # THE RUN GETS ITS OWN PROCESS GROUP, AND ITS PID IS WRITTEN DOWN.
  # jobs/seed_watch.sh has to be able to stop THIS run and no other. On the host it did
  # that with `docker stop` on a container filtered by IMAGE, which natively would mean
  # "kill all sixteen seeds on this box". setsid puts mpirun and its FastEddy child in one
  # group whose PGID is mpirun's PID, so a single `kill -- -PGID` reaches exactly this run.
  #
  # OpenMPI refuses to run as root without these two; the image declares no USER, so
  # inside it we ARE root. They are set in the Dockerfile as well -- here too, because a
  # bare `docker run --entrypoint bash` would not have them.
  export OMPI_ALLOW_RUN_AS_ROOT="${OMPI_ALLOW_RUN_AS_ROOT:-1}"
  export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="${OMPI_ALLOW_RUN_AS_ROOT_CONFIRM:-1}"
  export OMPI_MCA_plm="${OMPI_MCA_plm:-isolated}"
  # 16 single-rank jobs on one node: OpenMPI's default binds rank 0 to core 0 in EVERY
  # job, so all sixteen would fight for one core. --bind-to none lets the kernel place
  # them. The GPU does the work; the host thread is a driver-feeding loop.
  ( cd "$CASE_ABS" && exec setsid mpirun --bind-to none -np 1 "$FE_BIN" "./${CASE_FILE}" ) \
      > "$LOG" 2>&1 &
  FE_PID=$!
  # THE PID FILE CARRIES THE PROCESS START TIME, NOT JUST THE PID.
  # A container that is killed mid-seed leaves this file behind, and a FRESH container's
  # PID namespace starts at 1 -- so a stale low pid is very likely to name a live,
  # UNRELATED process on resume. jobs/seed_watch.sh would then latch onto it, find `kill
  # -0` succeeds, and on INBAND send `kill -TERM -- -<that pgid>` at something that is not
  # this run: plausibly the orchestrator, or another worker's FastEddy. Field 22 of
  # /proc/<pid>/stat is starttime in clock ticks since boot, which no later pid reuses.
  _st=$(awk '{print $22}' "/proc/$FE_PID/stat" 2>/dev/null || echo 0)
  printf '%s %s\n' "$FE_PID" "$_st" > "${CASE_ABS}/.fe.pid"
  # THE MUTEX PROVES ITSELF, ONCE PER RUN. Its failure mode is silence -- it permits what
  # it should refuse and says nothing -- so the same scan is run against the process just
  # started. If it cannot see a FastEddy that is demonstrably running on this device, the
  # guard is inoperative and every later run on this GPU would be allowed to collide.
  # A warning rather than a refusal, because THIS run is legitimate and already going;
  # what is being reported is that the protection for the NEXT one is not there.
  ( sleep 5
    if [ -z "$(fe_on_gpu "$MYFIRST")" ]; then
      echo "  WARNING: the per-GPU mutex cannot see its own FastEddy on GPU ${MYFIRST}." >&2
      echo "           Concurrent runs on this device would NOT be refused. Check that" >&2
      echo "           /proc is readable in this container and that CUDA_VISIBLE_DEVICES" >&2
      echo "           is inherited by the mpirun child." >&2
    fi ) &
  wait "$FE_PID"; RC=$?
  rm -f "${CASE_ABS}/.fe.pid"
  # AND ON EVERY EXIT PATH, not only the normal one -- a trap, because the shell can be
  # killed between the launch and the wait.
  trap - EXIT
else
  # ---- host: one container per case ----------------------------------------------
  # A NAME, SO THE CONTAINER CAN BE STOPPED BY SOMETHING OTHER THAN LUCK.
  # The in-process hand-off runs this in the BACKGROUND while the LPDM consumes in the
  # foreground, and a driver that has to abandon the case kills the shell it launched --
  # which does NOT stop the container. Measured: a killed smoke run left FastEddy holding
  # the GPU, and the next run was refused by the guard above with no hint of why.
  # `:+` AND NOT `+` on the forward below. `${VAR+...}` fires on set-but-EMPTY, and
  # `CUDA_VISIBLE_DEVICES=` is the conventional way to hide every GPU from a shell -- so
  # `+` would forward an empty value into the container and give the run no device at all,
  # on a host path that previously never forwarded the variable and always saw every GPU.
  FE_NAME="${FE_CONTAINER_NAME:-flux-fe-$(echo "$CASE_DIR" | tr -c 'A-Za-z0-9_.-' '-')}"
  docker rm -f "$FE_NAME" >/dev/null 2>&1 || true
  FLUX_RINGROOT="${FLUX_RINGROOT:-/dev/shm/flux}"
  mkdir -p "${FLUX_RINGROOT}" 2>/dev/null || true
  docker run --gpus all --rm --name "$FE_NAME" --user "$(id -u):$(id -g)" -e HOME=/tmp \
    ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"} \
    -v ${FLUX_ROOT}:/work -v ${FLUX_RINGROOT}:${FLUX_RINGROOT} -w "/work/${CASE_DIR}" "${FLUX_IMAGE}" \
    mpirun -np 1 /work/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy "./${CASE_FILE}" > "$LOG" 2>&1
  RC=$?
fi

# Score the log AND the newest dump (accuracy-CFL k0/k1 check) in one place.
# Read outPath from the CASE FILE rather than assuming ./output/. Sampling windows write
# to ./window/, and assuming ./output/ silently scored the leftover adjustment dump
# instead -- so the standing accuracy check was passing on a file the run never touched.
_op=$(grep -oP '^outPath\s*=\s*\K[^#[:space:]]*' "$_cf" 2>/dev/null || true)
_op="${_op:-./output/}"
LAST=$(ls -t "${CASE_ABS}/${_op}"/*.[0-9]* 2>/dev/null | head -1)
# The scorers run through pyrun.sh, which mounts the repo at /work on the host -- so a
# path under the repo must be handed over repo-relative there. Natively there is no mount
# and the absolute path is correct as it stands.
if [ "$NATIVE" = "1" ]; then LAST_REL="$LAST"; else LAST_REL="${LAST#${FLUX_ROOT}/}"; fi
exec "$(dirname "$0")/check_run.sh" "$LOG" "$RC" ${LAST_REL:+"$LAST_REL"}
