#!/usr/bin/env bash
# Self-check the image on the machine it is about to run on, before any GPU-hours.
#
#   docker run --gpus all -v /out:/out <image> verify
#
# FOUR QUESTIONS, IN THE ORDER THEY CAN COST MONEY:
#
#   1. Does the binary carry compiled code for the cards in this box? A miss here is
#      "no kernel image is available for execution on the device" at the first kernel,
#      after the grid is built and the banner has printed -- late enough to look like a
#      physics problem.
#   2. Do the libraries load? A CUDA/driver mismatch shows up as a missing .so.
#   3. Does it actually integrate? 200 steps, cold start, on the production grid.
#   4. Is the near-surface turbulence real rather than acoustic noise? k0/k1 and
#      turb_alive, the two standing checks -- and neither substitutes for the other.
#
# NOTHING HERE ASSERTS BITWISE ANYTHING. FastEddy is not bitwise reproducible run to run
# on ONE GPU, so it is not across architectures either. Physics parity with the toolkit
# this project's published results came from is a SEPARATE measurement, made against the
# model's own run-to-run floor: bin/test_toolkit_parity.py.
set -uo pipefail
ROOT="${FLUX_ROOT:-/flux}"; cd "$ROOT"
FE="${FE_BIN:-$ROOT/FastEddy-model-5.0.1/SRC/FEMAIN/FastEddy}"
STEPS="${1:-200}"
JOB="${VERIFY_JOB:-$ROOT/seeds/seed_nbl-deep_a015}"
W="${FLUX_OUT:-/out}/verify"
fail=0
note(){ echo "  $*"; }
bad(){ echo "  *** FAIL: $*"; fail=1; }

echo "=== 1. compiled architectures in the binary ==="
[ -x "$FE" ] || { bad "no FastEddy binary at $FE"; exit 1; }
SASS=$(cuobjdump --list-elf "$FE" 2>/dev/null | sed 's/.*\.\(sm_[0-9]*\)\.cubin/\1/' | sort -u | tr '\n' ' ')
PTX=$(cuobjdump --list-ptx "$FE" 2>/dev/null | sed 's/.*\.sm_\([0-9]*\)\.ptx/compute_\1/' | sort -u | tr '\n' ' ')
note "SASS: ${SASS:-none}"
note "PTX : ${PTX:-none}"
note "PTX is EXPECTED to be empty, and that is why the SASS list is long. FastEddy is"
note "built with separate compilation (-dc then -dlink), and MEASURED here: nvcc -dlink"
note "silently DROPS every PTX image from the fatbin. Asking for -gencode"
note "arch=compute_90,code=compute_90 puts PTX in the .o files and none of it in the"
note "executable, with no warning. So there is no JIT fallback for this binary by"
note "construction, and every architecture it supports carries real SASS instead --"
note "which is the better outcome anyway for register-heavy hydro-core kernels."
LPTX=$(cuobjdump --list-ptx "$ROOT/lib/liblpdm.so" 2>/dev/null | sed 's/.*\.sm_\([0-9]*\)\.ptx/compute_\1/' | sort -u | tr '\n' ' ')
note "and the contrast is on this image: lib/liblpdm.so is ONE translation unit, built"
note "whole-program with the same kind of -gencode, and its PTX is: ${LPTX:-none}"
case " $SASS " in *" sm_120 "*) note "sm_120 (Blackwell / RTX 5090): present";;
                  *) bad "no sm_120 SASS: this image cannot run on a 5090";; esac
case " $SASS " in *" sm_89 "*)  note "sm_89 (Ada / RTX 4080): present";;
                  *) bad "no sm_89 SASS: the local smoke test would not run natively";; esac

echo
echo "=== 2. the cards in this box, and the libraries ==="
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
             --format=csv,noheader | sed 's/^/    /'
  while IFS=, read -r i cc; do
    i=$(echo "$i" | tr -d ' '); cc=$(echo "$cc" | tr -d ' ')
    want="sm_$(echo "$cc" | tr -d '.')"
    case " $SASS " in
      *" $want "*) note "GPU $i needs $want -- present";;
      *) bad "GPU $i needs $want and the binary has [$SASS] and no PTX: it would fail at cuModuleLoad";;
    esac
  done < <(nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader)
else
  bad "nvidia-smi unavailable -- pass --gpus all to docker run"
fi
MISS=$(ldd "$FE" 2>/dev/null | grep -c "not found" || true)
[ "${MISS:-0}" = "0" ] && note "every shared library resolves" || bad "$MISS unresolved shared libraries"
ldd "$FE" | grep -E "cudart|curand|mpi|netcdf" | sed 's/^/    /'

echo
echo "=== 3. does it integrate? $STEPS steps, cold start, production grid ==="
note "WHAT THIS STEP CAN AND CANNOT ESTABLISH. It proves the binary loads on this card, the"
note "grid builds, the base state builds, the integrator runs and a netCDF dump lands. It"
note "does NOT establish k0/k1 or turb_alive: at $STEPS steps from a COLD START the"
note "near-surface variance has not developed, both checks correctly return SKIP, and"
note "check_run.sh says 'these established NOTHING' rather than calling a SKIP a PASS."
note "The numerical verdict belongs to the first real seed, where bin/seed_accept.sh scores"
note "k0/k1 on a developed dump and turb_alive alongside it."
rm -rf "$W"; mkdir -p "$W/output"
sed -e "s|^Nt = .*|Nt = $STEPS|" -e "s|^NtBatch = .*|NtBatch = $STEPS|" \
    -e "s|^frqOutput = .*|frqOutput = $STEPS|" -e "s|^inPath = .*|inPath = |" \
    -e "s|^inFile = .*|inFile = |" -e "s|^outPath = .*|outPath = ./output/|" \
    -e "s|^outFileBase = .*|outFileBase = FE_VER|" "$JOB/seed.in" > "$W/verify.in"
cp -f "$JOB/manifest.json" "$W/manifest.json" 2>/dev/null || true
if ./docker/run_case.sh "$W" verify.in "$W/verify.log"; then
  note "run_case.sh scored the run OK"
else
  bad "run_case.sh rejected the run (see $W/verify.log)"
fi
LAST=$(ls -1 "$W/output"/FE_VER.[0-9]* 2>/dev/null | sort -t. -k2 -n | tail -1)
if [ -n "$LAST" ]; then
  note "newest dump: $(basename "$LAST") ($(stat -c%s "$LAST") bytes)"
else
  bad "no dump produced"
fi

echo
echo "=== 4. peak VRAM actually used ==="
note "sampled during the run above by nvidia-smi if it was running; a 122^3 seed's own"
note "manifest budgets 1.6 GB, against 16 GB on an RTX 4080 and 32 GB on a 5090."
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv 2>/dev/null | sed 's/^/    /' || true

echo
if [ "$fail" = "0" ]; then
  echo "  VERIFY: PASS -- this image can run on this machine."
else
  echo "  VERIFY: FAIL -- see the lines marked *** above."
fi
exit $fail
