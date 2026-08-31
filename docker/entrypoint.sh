#!/usr/bin/env bash
# Entry point for the portable seed image. Prints what this image IS before it does
# anything, then dispatches.
#
# WHY THE BANNER IS NOT OPTIONAL. The whole point of baking the code in is that the tag
# pins a commit -- so the first question anyone debugging a rented box will ask is "which
# commit is this". `docker inspect` reads the labels, but not from INSIDE the container and
# not from a log someone pasted into a message. The provenance is therefore both a set of
# labels and a file, and the file is printed at every start.
set -uo pipefail
export FLUX_ROOT="${FLUX_ROOT:-/flux}"
export FLUX_NATIVE=1
export FE_BIN="${FE_BIN:-${FLUX_ROOT}/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy}"
cd "${FLUX_ROOT}"

banner(){
  echo "=============================================================================="
  echo " Flux seed generator -- FastEddy v5.0.1 (kegonsa fork) + Kegonsa LPDM pipeline"
  [ -f "${FLUX_ROOT}/IMAGE_PROVENANCE.txt" ] && sed 's/^/ /' "${FLUX_ROOT}/IMAGE_PROVENANCE.txt"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo " visible GPUs:"
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null | sed 's/^/   /' \
      || echo "   (nvidia-smi failed -- was --gpus all passed?)"
  else
    echo " visible GPUs: nvidia-smi ABSENT (this container has no GPU access)"
  fi
  echo "=============================================================================="
}

usage(){
cat <<'EOF'

USAGE
  docker run --gpus all -v /out:/out <image> run_seeds --gpu-count 16

COMMANDS
  run_seeds [opts]   Generate the seed library across every visible GPU. This is the one
                     command. It sweeps the CUDA thread-block shape on the first GPU, fans
                     the 30 seeds out one per GPU with a work queue, runs the full
                     acceptance battery per seed, writes each seed and a machine manifest
                     to the mounted output, and never lets one failed seed abort the box.
                     30 seeds over 16 GPUs needs NO --pass: it is a queue, not two passes.
                     --pass N/M exists for splitting the library across SEVERAL machines.
                       --gpu-count N     use the first N visible GPUs (0 = all)
                       --gpus 0,3,7      explicit indices
                       --only <job,...>  a smoke test on named seeds
                       --ceiling-h 2.0   simulated-hour hard ceiling per seed
                       --no-sweep        keep the .in's Ada-measured 1x2x64
                       --prune-dumps     delete each seed's output/ on success (~1.8 GB)
                       --dry-run         list what would run

  seed <job_dir>     One seed, this GPU (honours CUDA_VISIBLE_DEVICES). jobs/run_seed.sh.
  accept <job_dir>   The acceptance battery alone. bin/seed_accept.sh.
  verify             Self-check: SASS in the binary vs the visible GPUs, and a 200-step run.
  provenance         Print IMAGE_PROVENANCE.txt and exit.
  python3 ...        Any project script, e.g. python3 bin/threadblock_sweep.py --help
  shell              An interactive bash.
  <anything else>    Executed as given.

MOUNTS
  -v /out:/out       REQUIRED. Everything the run produces lands here.
  -v /data:/flux/data/hrrr:ro   ONLY for corpus CASES. A SEED NEEDS NO DATA AT ALL:
                     seed.in carries an empty topoFile and an empty inFile, so it reads no
                     sounding, no terrain and no land cover. The library is self-contained
                     in this image.
EOF
}

CMD="${1:-help}"; shift 2>/dev/null || true
case "$CMD" in
  help|--help|-h|"")  banner; usage ;;
  provenance)         cat "${FLUX_ROOT}/IMAGE_PROVENANCE.txt" ;;
  run_seeds)          banner; exec python3 "${FLUX_ROOT}/bin/run_seeds.py" "$@" ;;
  seed)               banner; exec "${FLUX_ROOT}/jobs/run_seed.sh" "$@" ;;
  accept)             banner; exec "${FLUX_ROOT}/bin/seed_accept.sh" "$@" ;;
  verify)             banner; exec "${FLUX_ROOT}/docker/verify_image.sh" "$@" ;;
  shell|bash)         exec bash "$@" ;;
  *)                  exec "$CMD" "$@" ;;
esac
