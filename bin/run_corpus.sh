#!/usr/bin/env bash
# The corpus: one case per day, resumable, skipping what the domain cannot hold.
#
#   usage: bin/run_corpus.sh <start YYYY-MM-DD> <end YYYY-MM-DD> [hour-policy]
#          bin/run_corpus.sh 2021-01-01 2025-12-31 rotate
#   env:   DRY=1        select and cost the cases without spending GPU time
#          HOURS=...    comma-separated UTC hours to choose from (default 15,16,17,18,19,20)
#          MAXCASES=N   stop after N successful cases
#
# ONE CASE PER DAY, at an hour that advances with the day so the corpus walks the whole
# diurnal cycle rather than sampling one hour of it 1825 times. `fixed` pins the first hour
# instead; HOURS= restricts the list (HOURS=15,16,17,18,19,20 is convective midday only,
# which is 09-14 CST / 10-15 CDT).
#
# WHY ONE PER DAY RATHER THAN MANY. Consecutive hours of the same day are not independent
# -- the same synoptic state, the same soil moisture, a boundary layer that remembers the
# morning. PROJECT_BRIEF.md's split rule already says the effective sample size for
# generalisation is the number of RUNS; drawing 6 hours from one day inflates the count
# without inflating that.
#
# RESUMABLE AND IDEMPOTENT: a day whose pair already exists is skipped, and a day the
# domain cannot hold is recorded in the skip ledger so it is not re-fetched every restart.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export FLUX_ROOT="$ROOT"; cd "$ROOT"

START="${1:?usage: run_corpus.sh <start YYYY-MM-DD> <end YYYY-MM-DD> [rotate|fixed]}"
END="${2:?usage: run_corpus.sh <start YYYY-MM-DD> <end YYYY-MM-DD> [rotate|fixed]}"
POLICY="${3:-rotate}"
# THE WHOLE DIURNAL CYCLE, not just the afternoon. The tower measures 24/7 and a stable
# boundary layer on 1 February at 04 UTC is a different state from one on 2 September at
# 11 UTC -- both belong in the corpus. With `rotate` the hour advances with the day, so a
# 24-hour list walks the full cycle every 24 days and 1825 days give ~76 samples per hour.
#
# It also decides how many days SURVIVE. The 1952 m box holds z_i <= 976 m, and a summer
# afternoon boundary layer routinely runs 1000-2600 m -- a 12-day June afternoon sample was
# rejected 9 times out of 12. Nights and winter are shallow, and are accepted.
IFS=',' read -r -a HRS <<< "${HOURS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}"
# LOGDIR is discovered, not hardcoded: /tmp/flux-logs exists on this machine and
# nowhere else, and these scripts are meant to run on rented ones.
LOGDIR="${LOGDIR:-${TMPDIR:-/tmp}/flux-logs}"; mkdir -p "$LOGDIR"
LEDGER=results/corpus_skipped.tsv
LOG=results/corpus_progress.tsv
mkdir -p results pairs
[ -f "$LOG" ] || printf 'tag\tvalid_time\tstatus\tseed\trot\tzi\twth\n' > "$LOG"
[ -f "$LEDGER" ] || printf 'tag\tvalid_time\treason\n' > "$LEDGER"

d="$START"; n=0; ok=0; skip=0; fail=0
while [ "$(date -d "$d" +%Y%m%d)" -le "$(date -d "$END" +%Y%m%d)" ]; do
  if [ "$POLICY" = "rotate" ]; then H=${HRS[$(( n % ${#HRS[@]} ))]}; else H=${HRS[0]}; fi
  TS="${d}T$(printf '%02d' "$H"):00"
  TAG="case_$(date -d "$d" +%Y%m%d)$(printf '%02d' "$H")"
  n=$((n+1))
  d=$(date -d "$d + 1 day" +%Y-%m-%d)

  # ASSERT ON THE ARTIFACT: a pair on disk is the only evidence a case is done.
  if [ -s "pairs/$TAG.json" ]; then ok=$((ok+1)); continue; fi
  if grep -q "^$TAG	" "$LEDGER" 2>/dev/null; then skip=$((skip+1)); continue; fi

  if [ "${DRY:-0}" = "1" ]; then
    SKIP_LES=1 bin/run_corpus_case.sh "$TS" "$TAG" >"$LOGDIR/corpus_dry.log" 2>&1
    rc=$?
    if [ "$rc" = "3" ]; then
      R=$(python3 -c "
import json;print('; '.join(json.load(open('results/forcing/$TAG.json'))['warnings'])[:160])" 2>/dev/null || echo "not representable")
      printf '%s\t%s\t%s\n' "$TAG" "$TS" "$R" >> "$LEDGER"; skip=$((skip+1))
    elif [ "$rc" = "0" ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    continue
  fi

  echo; echo "===== $TAG  ($TS)  [$ok done, $skip skipped, $fail failed] ====="
  bin/run_corpus_case.sh "$TS" "$TAG"
  rc=$?
  if [ "$rc" = "3" ]; then
    R=$(python3 -c "
import json;print('; '.join(json.load(open('results/forcing/$TAG.json'))['warnings'])[:160])" 2>/dev/null || echo "not representable")
    printf '%s\t%s\t%s\n' "$TAG" "$TS" "$R" >> "$LEDGER"
    printf '%s\t%s\tSKIP\t\t\t\t\n' "$TAG" "$TS" >> "$LOG"
    skip=$((skip+1))
  elif [ -s "pairs/$TAG.json" ]; then
    read -r SEED ROT ZI WTH < <(python3 -c "
import json;r=json.load(open('pairs/$TAG.json'));s=r.get('seed',{})
print(s.get('job','-'), s.get('rot','-'), round(r['inputs']['h'],1), round(r['inputs']['ustar'],4))")
    printf '%s\t%s\tOK\t%s\t%s\t%s\t%s\n' "$TAG" "$TS" "$SEED" "$ROT" "$ZI" "$WTH" >> "$LOG"
    ok=$((ok+1))
  else
    printf '%s\t%s\tFAIL\t\t\t\t\n' "$TAG" "$TS" >> "$LOG"
    fail=$((fail+1))
    echo "  FAILED (rc=$rc); continuing to the next day"
  fi
  [ -n "${MAXCASES:-}" ] && [ "$ok" -ge "$MAXCASES" ] && break
done

echo
echo "########## CORPUS: $ok pairs, $skip skipped, $fail failed, of $n days ##########"
echo "  ledger  $LEDGER"
echo "  progress $LOG"
if [ "$skip" -gt 0 ]; then
  echo
  echo "  THE SKIPS ARE NOT A NEUTRAL TRIM. z_i and surface heat flux correlate at +0.43"
  echo "  at this site, so the deep-boundary-layer hours the 1952 m box cannot hold carry"
  echo "  1.51x the heat flux and 1.58x the w* of the ones it can. State the exclusion"
  echo "  wherever the corpus is described (PROJECT_BRIEF.md, Known limitations)."
  cut -f3 "$LEDGER" | tail -n +2 | sed 's/[0-9]\+/N/g' | sort | uniq -c | sort -rn | head -5
fi
