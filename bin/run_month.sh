#!/usr/bin/env bash
# One GPU, one month, one case per day. Resumable, and a failed day never stops the month.
#
#   usage: bin/run_month.sh 2023-07
#          DRY=1 bin/run_month.sh 2023-07        select the hours; run no cases
#          STUB_LES=1 bin/run_month.sh 2023-07   the whole path with the LES stubbed
#   env:   NPZ_DIR=pairs_npz   MAXDAYS=N   LOGDIR=...
#
# WHAT IT PRODUCES. One `<NPZ_DIR>/case_YYYYMMDDHH.npz` per successful day, and ONE
# manifest.json beside them carrying the case list, the split, the code commit, the grid,
# and every MISSING DAY with the reason it is missing. Nothing else survives a day --
# bin/get_case.sh deletes its own scratch, including when it fails.
#
# A FAILED CASE MUST NOT ABORT THE MONTH. Every day is run in its own subshell, its exit
# status is recorded, and the loop continues. Over ~30 days on a rented box the expected
# number of transient failures -- a dropped HRRR fetch, a container that did not start -- is
# not zero, and a month that stops at the first one wastes the rest of the rental.
#
# EVERY DAY IS ACCOUNTED FOR. A day is exactly one of: a case, or a MISSING DAY with a
# reason. The manifest's `days` block has one entry per calendar day of the month, so
# "how many days did this month yield" is answerable from the artifact rather than by
# counting files and hoping the difference is explicable.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"

MONTH="${1:-}"
[[ "$MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]] || {
  echo "usage: run_month.sh <YYYY-MM>" >&2; exit 64; }
NPZ_DIR="${NPZ_DIR:-pairs_npz}"
LOGDIR="${LOGDIR:-${TMPDIR:-/tmp}/flux-logs}"; mkdir -p "$LOGDIR" "$NPZ_DIR" results/hours
MAN="$NPZ_DIR/manifest.json"

# ---- the split, before anything else ----------------------------------------------------
# A month outside the corpus is refused here, once, rather than 30 times inside the loop.
SPLIT=$(./docker/pyrun.sh - "$MONTH-01" <<'PY' 2>&1
import sys, datetime as dt
from lpdm.corpus import split_of
try:
    print(split_of(dt.date.fromisoformat(sys.argv[1])))
except ValueError as e:
    print(f"REFUSED {e}"); raise SystemExit(3)
PY
) || { echo "FATAL: ${SPLIT#REFUSED }" >&2; exit 3; }
SPLIT="$(echo "$SPLIT" | tr -d '\r' | tail -1)"
case "$SPLIT" in train|val|test) ;; *) echo "FATAL: split '$SPLIT'" >&2; exit 1;; esac

Y="${MONTH%-*}"; M="${MONTH#*-}"
NDAYS=$(date -d "$MONTH-01 +1 month -1 day" +%d)
echo "########## $MONTH  split=$SPLIT  $NDAYS days ##########"
date '+%F %H:%M:%S'

ok=0; missing=0; failed=0
for dd in $(seq -w 1 "$NDAYS"); do
  DAY="$MONTH-$dd"
  [ -n "${MAXDAYS:-}" ] && [ "$((ok + missing + failed))" -ge "$MAXDAYS" ] && break
  echo; echo "===== $DAY  [$ok cases, $missing missing, $failed failed] ====="

  # ---- the hour, drawn without replacement (bin/pick_hour.py) --------------------------
  HJ="results/hours/$DAY.json"
  TS=$(./docker/pyrun.sh bin/pick_hour.py "$DAY" --json "$HJ" \
         2> >(grep -E "ACCEPTED|DAY MISSING|rejected|missing" >&2) | tail -1)
  if [ -z "$TS" ]; then
    missing=$((missing + 1))
    echo "  MISSING DAY (see $HJ)"
    continue
  fi

  # ---- the case ------------------------------------------------------------------------
  # In its own subshell so a `die` inside get_case.sh cannot take the month with it.
  ( NPZ_DIR="$NPZ_DIR" bash bin/get_case.sh "$TS" ) \
      > "$LOGDIR/$DAY.log" 2>&1
  rc=$?
  TAGN="${TS//[-:]/}"; TAGN="${TAGN/T/}"; TAG="case_${TAGN:0:10}"
  if [ -s "$NPZ_DIR/$TAG.npz" ]; then
    ok=$((ok + 1)); echo "  OK  $TAG"
  elif [ "$rc" = "3" ]; then
    # The hour passed the HRRR screen but the FITTED sounding put z_i outside the band --
    # a different and later test than pick_hour's, on the profile rather than on HPBL.
    missing=$((missing + 1))
    echo "  MISSING DAY: the fitted sounding is not representable at $TS"
    ./docker/pyrun.sh - "$HJ" "$TS" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
r["missing_reason"] = (f"the drawn hour {sys.argv[2]} passed the HRRR screen but its fitted "
                       f"sounding put z_i outside what the domain supports (stage 2)")
r["accepted"] = None
json.dump(r, open(sys.argv[1], "w"), indent=1, default=float)
PY
  else
    failed=$((failed + 1))
    echo "  FAILED (rc=$rc) -- see $LOGDIR/$DAY.log"
    tail -4 "$LOGDIR/$DAY.log" | sed 's/^/      /'
  fi
done

# ---- the manifest: every day accounted for ---------------------------------------------
./docker/pyrun.sh - "$MAN" "$MONTH" "$SPLIT" "$NDAYS" "$NPZ_DIR" <<'PY'
import datetime as dt, json, os, sys
import numpy as np
man, month, split, ndays, npz_dir = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]

m = {"format": "flux-footprint-manifest/1", "cases": {}}
if os.path.exists(man):
    try:
        m = json.load(open(man)); m.setdefault("cases", {})
    except (OSError, json.JSONDecodeError):
        print(f"  WARNING: {man} was unreadable and is being rewritten")
m.setdefault("days", {})
m.setdefault("months", {})

def commit():
    """Read .git/HEAD rather than shelling out to git.

    The analysis runs inside the container, which has no git binary -- so the subprocess
    call this replaced returned None on every run, and the manifest recorded no commit at
    all. Reading the files is what bin/make_pair.py already does, and it works anywhere the
    repo is mounted.
    """
    try:
        with open(".git/HEAD") as f:
            ref = f.read().strip()
        if ref.startswith("ref: "):
            with open(os.path.join(".git", ref[5:])) as f:
                return f.read().strip()
        return ref
    except OSError:
        return None

nok = nmiss = 0
for d in range(1, ndays + 1):
    day = f"{month}-{d:02d}"
    hj = f"results/hours/{day}.json"
    rec = {"split": split}
    if os.path.exists(hj):
        r = json.load(open(hj))
        acc = r.get("accepted")
        rec["hours_drawn"] = r.get("drawn")
        rec["hours_rejected"] = r.get("rejected")
        rec["n_hrrr_fetches"] = r.get("n_hrrr_fetches")
        if acc:
            tag = "case_" + acc["timestamp"].replace("-", "").replace("T", "")[:10]
            p = os.path.join(npz_dir, tag + ".npz")
            if os.path.exists(p):
                rec.update(status="case", tag=tag, timestamp=acc["timestamp"],
                           hrrr_zi_m=acc.get("zi_m"), hrrr_shtfl_wm2=acc.get("shtfl_wm2"),
                           hrrr_dzidt_rel_per_h=acc.get("dzidt_rel_per_h"),
                           hrrr_wdir_deg=acc.get("wdir_deg"))
                nok += 1
            else:
                rec.update(status="missing",
                           reason=r.get("missing_reason")
                           or f"the hour {acc['timestamp']} was drawn but no record exists")
                nmiss += 1
        else:
            rec.update(status="missing", reason=r.get("missing_reason", "no hour accepted"))
            nmiss += 1
    else:
        rec.update(status="missing", reason="the day was never drawn (the month stopped early)")
        nmiss += 1
    m["days"][day] = rec

# THE ACHIEVED NUMBERS COME OUT OF THE RECORDS THEMSELVES, not out of the HRRR screen.
# The screen's z_i is HRRR's HPBL at the analysis hour; the case's is what the LES realised
# over the window. They are different quantities and the manifest carries both.
for tag, c in m["cases"].items():
    p = os.path.join(npz_dir, c.get("file", tag + ".npz"))
    if not os.path.exists(p):
        continue
    with np.load(p, allow_pickle=True) as z:
        md = json.loads(str(z["meta"]))
    if md.get("stub"):
        c["STUB"] = True
    c.update(split=md.get("split"), zi_achieved_m=md.get("zi_achieved_m"),
             zi_accepted_drifting=md.get("zi_accepted_drifting"),
             gate_state=md.get("gate_state"),
             zol=(md.get("receptor", {}).get("z_m") * md.get("inv_L")
                  if md.get("inv_L") is not None
                  and md.get("receptor", {}).get("z_m") is not None else None),
             wdir_deg=md.get("wdir_deg"))

m["git_commit"] = commit()
m["host"] = os.uname().nodename
m["months"][month] = {"split": split, "days": ndays, "cases": nok, "missing": nmiss}
tmp = man + f".tmp.{os.getpid()}"
json.dump(m, open(tmp, "w"), indent=1, default=float)
os.replace(tmp, man)

stub = [t for t, c in m["cases"].items() if c.get("STUB")]
print(f"\n  manifest {man}: {len(m['cases'])} case(s) total, "
      f"{len(m['days'])} day(s) accounted for")
print(f"  {month}: {nok} cases, {nmiss} missing of {ndays} days ({100*nok/ndays:.0f}% yield)")
if stub:
    print(f"  *** {len(stub)} STUBBED record(s) in this manifest: {', '.join(sorted(stub))}")
    print(f"      These are NOT corpus cases. Delete them before shipping the corpus.")
PY

echo
echo "########## $MONTH DONE: $ok cases, $missing missing, $failed failed ##########"
date '+%F %H:%M:%S'
[ "$failed" -gt 0 ] && echo "  $failed day(s) failed for a reason that is not a screen; see $LOGDIR/"
exit 0
