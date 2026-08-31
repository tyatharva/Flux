#!/usr/bin/env bash
# Does a corpus machine actually do what it claims? MEASURED, with no GPU and no HRRR.
#
#   bash bin/test_corpus_machine.sh [outdir]
#
# WHAT IS UNDER TEST. Not the physics -- the LES, the LPDM, the ring and the footprint are
# validated elsewhere and none of them runs here. What runs here is the ORCHESTRATION, and
# every claim it makes is a claim about scheduling or bookkeeping:
#
#   1. the 64-month partition covers every month exactly once across --machine 0..7
#   2. all 8 of a machine's months are walked, and every calendar day is accounted for
#   3. the shared queue REBALANCES -- workers take uneven numbers of days and all stay busy
#   4. the progress file updates and is readable by the separate viewer
#   5. resume skips completed cases and does not redo them
#   6. the manifest accounts for every day as case / missing / failed / skipped
#   7. every written record passes the npz schema check
#
# THE STUB IS DELIBERATELY NOT INSTANT. bin/stub_case.py sleeps a few milliseconds, varied
# per case from the case's own hash. With an instant stub every worker finishes at the same
# moment, the queue hands out work in lockstep, and a rigid month-per-GPU assignment would
# produce an identical timeline -- so the rebalancing claim would be untested while looking
# tested. A few ms is enough to make workers finish unevenly; the whole run still takes
# seconds.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
OUT="${1:-${TMPDIR:-/tmp}/flux-corpus-dry}"
MACH="${MACH:-2}"
NGPU="${NGPU:-8}"
PY="${PY:-python3}"

echo "=== corpus machine dry run: machine $MACH, $NGPU fabricated workers, no GPU ==="
rm -rf "$OUT"; mkdir -p "$OUT"

# ---- 1. the partition, across all eight machines -----------------------------------------
echo
echo "--- 1. the 64-month partition over --machine 0..7 ---"
$PY - <<'PY'
import sys; sys.path.insert(0, ".")
from lpdm.partition import MONTHS, N_MACHINES, months_for, month_str
seen = {}
ok = True
for m in range(N_MACHINES):
    for ym in months_for(m):
        if ym in seen:
            print(f"  [FAIL] {month_str(ym)} claimed by machines {seen[ym]} and {m}"); ok = False
        seen[ym] = m
missing = [month_str(x) for x in MONTHS if x not in seen]
extra = [month_str(x) for x in seen if x not in set(MONTHS)]
print(f"  [{'PASS' if not missing else 'FAIL'}] every corpus month is owned   "
      f"({len(seen)} owned, {len(MONTHS)} in corpus, missing {missing or 'none'})")
print(f"  [{'PASS' if not extra else 'FAIL'}] no month outside the corpus is owned   "
      f"({extra or 'none'})")
print(f"  [{'PASS' if ok else 'FAIL'}] no month is owned twice")
counts = {m: len(months_for(m)) for m in range(N_MACHINES)}
even = len(set(counts.values())) == 1
print(f"  [{'PASS' if even else 'FAIL'}] every machine carries the same number of months "
      f"({sorted(set(counts.values()))})")
raise SystemExit(0 if (ok and not missing and not extra and even) else 1)
PY
P1=$?

# ---- 2. a full machine, stubbed ----------------------------------------------------------
echo
echo "--- 2. machine $MACH end to end: 8 months, shared queue over $NGPU workers ---"
$PY bin/run_corpus_machine.py --machine "$MACH" --out "$OUT" \
    --stub --assume-gpus "$NGPU" --early-n 5 --max-hours 12 2>&1 | tail -22
RUN1=$?

# ---- 3. resume ---------------------------------------------------------------------------
echo
echo "--- 3. resume: the same command again must do no work ---"
$PY bin/run_corpus_machine.py --machine "$MACH" --out "$OUT" \
    --stub --assume-gpus "$NGPU" --early-n 5 2>&1 | grep -E 'DONE|case |missing |failed |skipped ' | tail -8

# ---- 4. the verdict, computed from the artifacts -----------------------------------------
echo
echo "=== verdict, computed from manifest.json and the records on disk ==="
$PY - "$OUT" "$MACH" "$NGPU" <<'PY'
import json, os, subprocess, sys, collections
out, mach, ngpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, ".")
from lpdm.partition import months_for, days_in, month_str

man = json.load(open(os.path.join(out, "manifest.json")))
ok = True
def check(label, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not cond: ok = False

mine = months_for(mach)
want_days = sum(days_in(*x) for x in mine)
days = man["days"]

# 2. all eight months walked, every day accounted for
walked = sorted({d[:7] for d in days})
check("all 8 owned months appear in the manifest",
      walked == sorted(month_str(x) for x in mine), f"{walked}")
check("every calendar day is accounted for exactly once",
      len(days) == want_days, f"{len(days)} entries, month lengths sum to {want_days}")
bad_status = {d: v["status"] for d, v in days.items()
              if v["status"] not in ("case", "missing", "failed", "not reached")}
check("every day is one of case/missing/failed", not bad_status, str(bad_status)[:80])
noreason = [d for d, v in days.items()
            if v["status"] in ("missing", "failed") and not v.get("reason")]
check("every missing/failed day carries a reason", not noreason, f"{noreason[:5]}")

# 3. the queue rebalanced
gpus = collections.Counter(v["gpu"] for v in days.values() if v.get("gpu") is not None)
load = collections.defaultdict(float)
for v in days.values():
    if v.get("gpu") is not None and v.get("wall_s"):
        load[v["gpu"]] += v["wall_s"]
check("every worker took work", len(gpus) == ngpu, f"{len(gpus)} of {ngpu} workers used")
spread = (max(gpus.values()) / min(gpus.values())) if gpus else 0
check("workers took UNEVEN day counts (the queue handed out work as they freed)",
      len(set(gpus.values())) > 1,
      f"per-worker days {sorted(gpus.values())}, spread {spread:.2f}x")
if load:
    busiest, mean = max(load.values()), sum(load.values()) / len(load)
    # SANITY BEFORE VERDICT. `imb < 25` is satisfied by a NEGATIVE imbalance, which can
    # only mean the timing is broken -- and it was: t_start and t_end were rounded to
    # different precisions and sub-100 ms days came out with negative durations, which
    # this check happily passed. A tolerance must reject nonsense as well as excess.
    sane = min(load.values()) > 0 and busiest >= mean > 0
    imb = 100 * (busiest - mean) / busiest if sane else float("nan")
    check("worker busy time is a sane measurement at all", sane,
          f"per-worker busy {sorted(round(v,3) for v in load.values())}")
    check("the queue kept workers within 25% of each other in BUSY TIME",
          sane and imb < 25.0,
          f"busiest {busiest:.3f}s vs mean {mean:.3f}s -> {imb:.1f}% imbalance")

# what a rigid month-per-GPU would have cost, from the SAME timeline
bymonth = collections.defaultdict(float)
for d, v in days.items():
    if v.get("wall_s"):
        bymonth[d[:7]] += v["wall_s"]
if len(bymonth) > 1 and min(bymonth.values()) > 0:
    rigid, queue_ = max(bymonth.values()), sum(bymonth.values()) / ngpu
    print(f"  [INFO] RIGID month-per-GPU would finish at {rigid:.2f}s (its busiest month); "
          f"the shared queue at {queue_:.2f}s -> {100*(rigid-queue_)/rigid:.0f}% of wall "
          f"time saved, MEASURED on this run's own yields")
    print(f"         per-month work: " +
          " ".join(f"{k}:{v:.1f}s" for k, v in sorted(bymonth.items())))

# 5. resume
# RESUME: after the second pass every day must be marked resumed, AND must still carry
# its RESOLVED status -- a manifest that says "skipped" everywhere has lost the corpus.
res = sum(1 for v in days.values() if v.get("resumed"))
check("the resume pass marked every day resumed", res == len(days), f"{res}/{len(days)}")
check("and the manifest still says what each day IS, not that a pass skipped it",
      not any(v["status"] == "skipped" for v in days.values())
      and sum(1 for v in days.values() if v["status"] == "case") > 0,
      f"case {sum(1 for v in days.values() if v['status']=='case')}, "
      f"missing {sum(1 for v in days.values() if v['status']=='missing')}")

# 6/7. every record on disk is schema valid
npz = sorted(f for f in os.listdir(os.path.join(out, "pairs_npz")) if f.endswith(".npz"))
ncase = sum(1 for v in days.values() if v["status"] in ("case", "skipped") and v.get("tag"))
check("one record per accepted day, and no others",
      len(npz) == len({v["tag"] for v in days.values()
                       if v.get("tag") and v["status"] == "case"}),
      f"{len(npz)} .npz on disk")
r = subprocess.run([sys.executable, "bin/check_npz.py"]
                   + [os.path.join(out, "pairs_npz", f) for f in npz]
                   + ["--quiet", "--allow-stub"], capture_output=True, text=True)
_out = (r.stdout + r.stderr).strip().splitlines()
check("every record passes the npz schema check", r.returncode == 0 and bool(npz),
      (_out[-1][:100] if _out else f"{len(npz)} records, no complaints"))
r2 = subprocess.run([sys.executable, "bin/check_npz.py",
                     os.path.join(out, "pairs_npz", npz[0]), "--quiet"],
                    capture_output=True, text=True)
check("and every record is REFUSED as a corpus record (meta.stub)", r2.returncode != 0)

# splits are the partition's, not re-derived
sp = {v["split"] for v in days.values()}
check("the manifest carries a split for every day", sp <= {"train", "val", "test"}, str(sp))
check("the manifest names the commit and the grid",
      man.get("git_commit") is not None and man["grid"]["dx_m"] == 30.0,
      f"commit {str(man.get('git_commit'))[:12]}, dx {man['grid']['dx_m']} m")

# 4. the progress file is readable by the viewer
pr = subprocess.run([sys.executable, "bin/corpus_progress.py", "--out", out, "--once"],
                    capture_output=True, text=True)
check("the progress file renders in the separate viewer", pr.returncode == 0,
      pr.stdout.splitlines()[0][:80] if pr.stdout else pr.stderr[:80])

print()
print(f"  CORPUS MACHINE: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
V=$?
echo
[ "$P1" = "0" ] && [ "$V" = "0" ] && echo "ALL CHECKS PASS" || echo "SOME CHECKS FAILED"
exit $(( P1 || V ))
