#!/usr/bin/env bash
# Does the per-GPU mutex hold under a REAL simultaneous launch? MEASURED, not asserted.
#
#   bash bin/test_gpu_mutex.sh [n_racers]
#
# WHY THIS IS NOT THE TWO-PROCESS TEST. Starting one run, waiting, and then starting a
# second proves the guard REJECTS a run that is already established. It says nothing about
# the case that actually matters: N processes reaching the guard at the same instant, all
# scanning before any of them has launched. A check-then-act guard passes the sequential
# test and fails the simultaneous one, and the failure is silent -- two FastEddys
# interleaving dumps into one output/ looks like a stalled run, not an error.
#
# docker/run_case.sh therefore holds an flock(2) on a per-device file for the life of the
# run, and the /proc scan only NAMES the holder. This races N of them at once and requires
# EXACTLY ONE to win.
#
# NO GPU IS USED. FE_BIN is overridden with a stub AT a path ending in FEMAIN/FastEddy --
# so the mutex, the setsid process group, the .fe.pid file and its start-time validation
# all run exactly as they do in production, against a process that costs nothing. What is
# NOT tested here is CUDA itself, which the guard never touches.
#
# THE STUB IS COMPILED, NOT A SHELL SCRIPT, AND THAT IS NOT FUSSINESS. The guard resolves
# /proc/<pid>/exe (see docker/run_case.sh for why it no longer greps the cmdline), and a
# `#!/bin/sh` script's exe is /bin/dash -- so a script stub would be invisible to the very
# mechanism under test and the test would pass without testing anything.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
N="${1:-8}"
IMAGE="${FLUX_IMAGE_TEST:-flux-seeds:latest}"
W="${TMPDIR:-/tmp}/flux-mutex-test"
rm -rf "$W"; mkdir -p "$W"

echo "=== per-GPU mutex: $N simultaneous launches at device 0, no GPU used ==="

docker run --rm -v "$W":/w -e FLUX_NATIVE=1 -e FLUX_ROOT=/flux "$IMAGE" bash -c '
set -u
N='"$N"'
# A REAL ELF AT A PATH ENDING IN FEMAIN/FastEddy, so /proc/<pid>/exe resolves the way it
# does for the production binary. Everything else about the launch path is production.
mkdir -p /w/fake/SRC/FEMAIN
cat > /w/stub.c <<EOS
#include <stdio.h>
#include <unistd.h>
int main(void){ printf("stub FastEddy, pid %d\n", getpid()); fflush(stdout);
                sleep(8); printf("simulation is complete\n"); return 0; }
EOS
gcc -O0 -o /w/fake/SRC/FEMAIN/FastEddy /w/stub.c || { echo "FATAL: cannot build the stub"; exit 1; }

# One case dir per racer, so a winner cannot be mistaken for a collision on shared output.
for i in $(seq 1 $N); do
  mkdir -p /w/case$i/output
  printf "Nt = 1\noutPath = ./output/\noutFileBase = FE_X\n" > /w/case$i/t.in
done

# RELEASED FROM A COMMON BARRIER. Backgrounding them in a loop staggers the starts by
# milliseconds, which is exactly the window a check-then-act guard survives. They all block
# on the same fifo and are released by one write.
mkfifo /w/gate
for i in $(seq 1 $N); do
  (
    read -r _ < /w/gate
    CUDA_VISIBLE_DEVICES=0 FE_BIN=/w/fake/SRC/FEMAIN/FastEddy FLUX_NATIVE=1 FLUX_ROOT=/flux \
      /flux/docker/run_case.sh /w/case$i t.in /w/case$i/run.log > /w/case$i/outer.log 2>&1
    echo "$? $i" >> /w/rc.txt
  ) &
done
sleep 1
for i in $(seq 1 $N); do echo go > /w/gate; done
wait
' 2>&1 | tail -3

echo
echo "=== verdict ==="
python3 - "$W" "$N" <<'PY'
import os, sys, re
W, N = sys.argv[1], int(sys.argv[2])
rc = {}
try:
    for ln in open(os.path.join(W, "rc.txt")):
        code, idx = ln.split()
        rc[int(idx)] = int(code)
except FileNotFoundError:
    print("  [FAIL] no rc.txt -- the racers did not run"); sys.exit(1)

refused = [i for i, c in rc.items() if c == 2]
# A winner is one that actually LAUNCHED: its run.log carries the stub's own banner.
launched = []
for i in rc:
    p = os.path.join(W, f"case{i}", "run.log")
    if os.path.isfile(p) and "stub FastEddy" in open(p, errors="replace").read():
        launched.append(i)

ok = True
def check(label, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not cond: ok = False

check("all racers returned a status", len(rc) == N, f"{len(rc)} of {N}")
check("EXACTLY ONE launched FastEddy", len(launched) == 1,
      f"launched: {sorted(launched)}")
check("every other racer was REFUSED with exit 2", len(refused) == N - 1,
      f"{len(refused)} refused, {sorted(set(rc.values()))} distinct exit codes")
check("no racer both refused and launched", not (set(refused) & set(launched)))

# The refusal must SAY something usable, not just fail.
named = 0
for i in refused:
    t = open(os.path.join(W, f"case{i}", "outer.log"), errors="replace").read()
    if "already holds GPU 0" in t or "already on GPU 0" in t:
        named += 1
check("the refusals name the device they were refused on", named == len(refused),
      f"{named} of {len(refused)}")
print()
print(f"  PER-GPU MUTEX: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
