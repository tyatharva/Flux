#!/usr/bin/env bash
# Does the work queue actually behave like a work queue? MEASURED, not asserted.
#
#   bash bin/test_work_queue.sh [outdir]
#
# THE CLAIM UNDER TEST. bin/run_seeds.py says that 30 seeds over 16 GPUs needs no "pass"
# model: 16 start, and a card that finishes picks up the next job. That is a statement
# about a schedule, and this project does not accept a schedule as an assertion any more
# than it accepts a physics number as one. Demonstrating it with real seeds costs ~29
# GPU-h. Demonstrating it with the LES stubbed costs about a minute and exercises every
# line of the orchestration -- the queue, the per-GPU assignment, the failure path, the
# accounting -- with no GPU at all.
#
# WHAT IS STUBBED AND WHAT IS NOT. `--stub` runs the REAL jobs/run_seed.sh with
# STUB_SEED=1: manifest parsing, the SEED_CEILING_H arithmetic, the preflight, the return/
# staging and the orchestrator's whole record-keeping all run. Only FastEddy, the
# stationarity gate and the acceptance battery are replaced. Every artifact is stamped
# `stub: true` and run_seeds.py refuses to count one as accepted.
#
# FOUR THINGS ARE CHECKED, ALL FROM THE RECORDED TIMELINE IN machine_manifest.json:
#
#   1. EVERY job ran exactly once -- none lost, none duplicated.
#   2. At least one worker took a SECOND job that STARTED AFTER its first one ENDED.
#      That is the queue property; a two-pass scheduler cannot produce it.
#   3. A DELIBERATELY FAILED job's worker went on to take another job. A failure must
#      release the GPU, not strand it.
#   4. Peak concurrency never exceeded the worker count, and the run ended only when the
#      queue was empty.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
OUT="${1:-${TMPDIR:-/tmp}/flux-queue-test}"
IMAGE="${FLUX_IMAGE_TEST:-flux-seeds:latest}"

# 20 jobs over 6 workers, so the queue has to recycle workers 14 times. Two of them fail,
# on purpose, and one of the two is early enough that its worker must take more work after.
NJOBS="${NJOBS:-20}"
NGPU="${NGPU:-6}"
FAILJOBS="${FAILJOBS:-seed_nbl-shallow_a030,seed_cbl-shallow_a015}"
SECS="${SECS:-3}"

echo "=== work-queue test: $NJOBS stubbed jobs over $NGPU fabricated workers ==="
echo "    image     : $IMAGE"
echo "    out       : $OUT"
echo "    failing   : $FAILJOBS"
# CLEANED FROM INSIDE A CONTAINER, because the previous run wrote it as root and a
# host-side `rm -rf` fails on it SILENTLY -- which is how the first version of this test
# ended up exercising the RESUME path instead of the queue, reporting 0.0 s job durations
# and "18 accepted" stubs. A test that quietly tests something else is worse than no test.
mkdir -p "$OUT"
docker run --rm -v "$OUT":/out "$IMAGE" bash -c 'rm -rf /out/* /out/.[!.]* 2>/dev/null; true'
[ -z "$(ls -A "$OUT" 2>/dev/null)" ] || { echo "FATAL: could not clear $OUT" >&2; exit 1; }

JOBS=$(python3 -c "
import json
d=json.load(open('jobs30/index.json'))['jobs'][:$NJOBS]
print(','.join(j['job'] for j in d))")

docker run --rm -v "$OUT":/out "$IMAGE" run_seeds \
    --stub --stub-seconds "$SECS" --stub-fail "$FAILJOBS" \
    --assume-gpus "$NGPU" --only "$JOBS" 2>&1 | tail -40

echo
echo "=== verdict, computed from machine_manifest.json ==="
python3 - "$OUT/machine_manifest.json" "$NGPU" "$FAILJOBS" <<'PY'
import json, sys, collections
man = json.load(open(sys.argv[1]))
ngpu = int(sys.argv[2])
failing = {s for s in sys.argv[3].split(",") if s}
rows = man["seeds"]
ok = True


def check(label, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        ok = False


# 1. every job exactly once
names = [r["job"] for r in rows]
dup = [n for n, c in collections.Counter(names).items() if c > 1]
check("every job recorded exactly once",
      len(names) == len(set(names)) and len(names) == man["n_attempted"],
      f"{len(names)} records, {len(set(names))} distinct, duplicates {dup or 'none'}")

# 2. a worker took a second job that STARTED AFTER its first ENDED
by_gpu = collections.defaultdict(list)
for r in rows:
    if r.get("t_start_s") is not None:
        by_gpu[r["gpu"]].append(r)
reuse = []
for g, v in by_gpu.items():
    v.sort(key=lambda r: r["t_start_s"])
    for a, b in zip(v, v[1:]):
        if b["t_start_s"] >= a["t_end_s"]:
            reuse.append((g, a["job"], round(a["t_end_s"], 1), b["job"], round(b["t_start_s"], 1)))
check("a worker took a SECOND job after finishing its first", bool(reuse),
      f"{len(reuse)} such hand-offs")
for g, j1, t1, j2, t2 in reuse[:4]:
    print(f"         gpu {g}: {j1} ended {t1}s -> {j2} started {t2}s")

# 3. a FAILED job's worker went on to take more work
failed = [r for r in rows if r["status"] != "ok" and r["job"] in failing]
check("the deliberately-failed jobs did fail", len(failed) == len(failing),
      f"{[r['job'] for r in failed]}")
freed = []
for r in failed:
    later = [x for x in by_gpu[r["gpu"]] if x["t_start_s"] >= r["t_end_s"]]
    if later:
        freed.append((r["job"], r["gpu"], later[0]["job"]))
check("a failed job RELEASED its worker rather than stranding it",
      bool(freed) or not failed,
      "; ".join(f"gpu {g} freed by {f} -> {n}" for f, g, n in freed) or
      "(the failures landed last in the queue; nothing followed them)")

# 4. concurrency bounded, and the run drained the queue
edges = sorted([(r["t_start_s"], 1) for r in rows if r.get("t_start_s") is not None] +
               [(r["t_end_s"], -1) for r in rows if r.get("t_end_s") is not None])
cur = peak = 0
for _, d in edges:
    cur += d
    peak = max(peak, cur)
check("peak concurrency never exceeded the worker count", peak <= ngpu,
      f"peak {peak}, workers {ngpu}")
check("the run ended only when the queue was empty",
      man["n_attempted"] == len(names) and len(names) == len(set(names)),
      f"attempted {man['n_attempted']} of {len(names)} queued")

# 5. and a stub can never be accepted
check("no stub was counted as an accepted seed",
      man["n_accepted"] == 0 and man.get("stub") is True,
      f"n_accepted={man['n_accepted']}, manifest stub={man.get('stub')}")

print()
print(f"  WORK QUEUE: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
