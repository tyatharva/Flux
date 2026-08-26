#!/usr/bin/env bash
# Run ONE sampling window as a SINGLE CONTINUOUS FastEddy invocation. No chaining.
#
# CHAINING IS RETIRED (2026-08-26). A window used to be an automatically chained series of
# sub-wall-cap segments, each restarting from the last. That is gone, and with it the
# entire class of failure FASTEDDY_TRAPS.md 17 describes: a restart READ overwrites every
# IO-registered field -- htFlux, z0m, z0t, tskin, topoPos, zPos -- with whatever the
# restart file holds, so a chained run silently inherits state its own .in does not
# describe. It cost a whole segment of a stable seed running at zero surface flux while
# its .in asked for -0.012. The assertion that caught it is kept; the mechanism that
# needed it is now absent by construction. THE ONLY RESTART LEFT IN THE PROJECT IS
# SEED -> TARGET.
#
# What that costs, stated: a run is now as long as it needs to be, and the one-hour wall
# cap no longer applies to it. At 122^3 @ 16 m a target case is 4200 s simulated
# (30 min adjustment + 600 s t_back + 30 min releases) = 287,280 steps = ~74 min wall, and
# a seed is 3.0 simulated hours = ~2.9 h wall. Both exceed the old cap; neither can be
# split any more. That is the trade this change makes and it is deliberate.
#
# A production window is (averaging period + t_back) long, because a backward trajectory
# needs t_back seconds of history behind it before it can be released -- the first t_back
# of any window yields no releases at all. t_back = 600 s here, MEASURED
# (results/g16_tback.txt: converged at 500 s, x1.25 margin, rounded to 50 s).
#
# SKIP_S -- discard the first SKIP_S seconds of output after the run. This is what lets the
# 30-minute adjustment and the sampling window be ONE invocation: FastEddy has a single
# frqOutput and writes from step 0, so the adjustment-period dumps are produced and then
# DELETED, and the earliest surviving field is asserted to be exactly the adjustment end.
# Without that, lpdm/fields.py:dump_series globs the whole directory and compute_footprint
# starts releasing at t[0] + t_back -- which would put the entire 30-minute averaging
# period inside the adjustment, using fields that were still settling.
#
# usage: run_window.sh <dir> <restart> <dt> <window_s> <topofile|-> <Ug> <Vg> [extra.in]
#        env: SKIP_S=0  seconds of leading adjustment to run and then discard
set -uo pipefail
cd "${FLUX_ROOT:-/home/atyagi/Flux}"
D="$1"; RST="$2"; DT="$3"; WIN="$4"; TOPO="$5"; UG="$6"; VG="$7"; EXTRA="${8:-}"
BASE="${BASE:-runs/g16_base/base.in}"
L=/tmp/flux-logs
SKIP_S="${SKIP_S:-0}"           # leading seconds to run and then discard (the adjustment)
SPS="${SPS:-0.0155}"            # measured s/step at 122x122x122, block 1x2x64, with IO
CAD="${CAD:-5.0}"               # output cadence, s

die(){ echo "FATAL: $*" >&2; exit 1; }
[ -f "$RST" ] || die "restart $RST not found"
mkdir -p "$D/window" || die "cannot make $D/window"

read -r FRQ TOT SKIP_NT < <(python3 -c "
dt=$DT; win=$WIN; cad=$CAD; skip=$SKIP_S
frq=int(round(cad/dt));            assert abs(frq*dt-cad)<2e-4, 'cadence not an integer step count'
tot=int(round(win/dt/frq))*frq     # rounded to a whole number of dumps
skip_nt=int(round(skip/dt/frq))*frq
assert skip_nt < tot, 'SKIP_S covers the whole run'
print(frq, tot, skip_nt)")
echo "### window $D: ${TOT} steps = $(python3 -c "print(f'{$TOT*$DT:.0f}')") s, ONE continuous invocation"
echo "###   projected $(python3 -c "print(f'{$TOT*$SPS/60:.1f}')") min wall (no cap: chaining is retired)"
echo "###   frqOutput = $FRQ ($CAD s), $(python3 -c "print($TOT//$FRQ+1)") dumps"
[ "$SKIP_NT" -gt 0 ] && echo "###   discarding the first $SKIP_NT steps = $(python3 -c "print(f'{$SKIP_NT*$DT:.0f}')") s (adjustment)"

# RESUMABILITY, and the reason it earns its keep. A window is 42 minutes of GPU; a crash
# or a kill anywhere in the analysis that follows used to mean recomputing all of it, and
# the fields are on disk the whole time. On success the exact configuration is stamped
# into the window directory, and an invocation whose configuration matches a stamp is a
# no-op. The stamp is computed HERE, by the same code on both paths, so a window produced
# by hand ahead of a campaign and one produced by the campaign are only ever reused when
# they are the same window -- a different dt, length, restart, terrain or forcing does not
# match and the LES runs.
STAMP="$TOT|$DT|$WIN|$TOPO|$UG|$VG|$SKIP_NT|$(basename "$RST")|$(stat -c%s "$RST")"
if [ -f "$D/window/.window_complete" ] && \
   [ "$(cat "$D/window/.window_complete")" = "$STAMP" ] && \
   [ "$(ls -1 "$D"/window/*.[0-9]* 2>/dev/null | wc -l)" -eq "$(((TOT-SKIP_NT)/FRQ + 1))" ]; then
  echo "--- window already complete and identically configured; reusing $(ls -1 $D/window/*.[0-9]* | wc -l) dumps"
  echo "---   ($D/window/.window_complete matches; delete it to force a re-run)"
  exit 0
fi

rm -f "$D"/window/*
# THE RESTART MAY ALREADY BE THE FILE WE ARE ABOUT TO WRITE. bin/run_corpus_case.sh stages
# the seed-derived restart at $D/FE_RST.0 and then passes that path in, so the stale-clean
# below would delete the very file the copy then reads -- `cp -f X X` with X already gone.
# It exits 1, the die fires, and the case dies before FastEddy is ever launched. This is a
# consequence of retiring the chain: the restart used to be the adjustment run's final
# dump, in another directory, and could not collide with the destination.
if [ "$(readlink -f "$RST")" = "$(readlink -f "$D/FE_RST.0")" ]; then
  echo "--- restart is already $D/FE_RST.0; staging is a no-op"
else
  rm -f "$D"/FE_RST.*
  cp -f "$RST" "$D/FE_RST.0" || die "copy restart"
fi
[ -s "$D/FE_RST.0" ] || die "no restart at $D/FE_RST.0 after staging"
sed -e "s|^dt = .*|dt = $DT|" -e "s|^Nt = .*|Nt = $TOT|" \
    -e "s|^NtBatch = .*|NtBatch = $FRQ|" -e "s|^frqOutput = .*|frqOutput = $FRQ|" \
    -e "s|^inPath = .*|inPath = ./|" -e "s|^inFile = .*|inFile = FE_RST.0|" \
    -e "s|^topoFile = .*|topoFile = $([ "$TOPO" = "-" ] && echo "" || echo "$TOPO")|" \
    -e "s|^U_g = .*|U_g = $UG|" -e "s|^V_g = .*|V_g = $VG|" \
    -e "s|^outPath = .*|outPath = ./window/|" \
    -e "s|^outFileBase = .*|outFileBase = FE_WIN|" \
    "$BASE" > "$D/win1.in"
# ioLPDMfullFrq, AND IT IS LOAD-BEARING WHEN SKIP_S IS SET.
#
# Under ioLPDMmode the STATIC GEOMETRY -- xPos, yPos, zPos, topoPos, lat, lon -- is written
# to the FIRST FILE OF THE RUN ONLY (io_netcdf.c lpdmSkipWrite: `lpdmIsGeometry(n) &&
# lpdmFileCount > 0` -> skip). lpdm/fields.py reads geometry from paths[0]. So if the first
# file of the run is an ADJUSTMENT dump and SKIP_S deletes it, the surviving series has no
# zPos at all and FieldSet cannot be built. That is a pipeline-stopping bug and it is
# created by unchaining, not by anything older.
#
# The escape is in the same function, one line earlier: `if(ioLPDMmode == 0 ||
# lpdmFullThisFile){ return 0; }` -- a FULL file writes everything, geometry included,
# whatever lpdmFileCount says. So setting ioLPDMfullFrq = SKIP_NT makes every multiple of
# SKIP_NT full-form, and the first surviving dump IS step SKIP_NT. Costs two extra 73 MB
# dumps in a 4200 s case against 18 MB lean ones; the assertion below is what proves it.
#
# With no SKIP (a plain window) the first file is kept, so TOT is enough: one full dump at
# the end, which is all that a restartable final state needs.
# AND CHECK THE SED ACTUALLY LANDED. `sed s|^key = .*|key = v|` on a template that has no
# such key is a silent no-op: the .in keeps whatever the template said, FastEddy runs, and
# the run is simply a different run than the one asked for. That is the same shape as every
# other failure this project has paid for -- a plausible wrong number instead of an error --
# so the written file is scored against the values it was supposed to receive.
TOPOVAL="$([ "$TOPO" = "-" ] && echo "" || echo "$TOPO")"
for kv in "dt|$DT" "Nt|$TOT" "NtBatch|$FRQ" "frqOutput|$FRQ" "inPath|./" \
          "inFile|FE_RST.0" "U_g|$UG" "V_g|$VG" "outPath|./window/" \
          "outFileBase|FE_WIN" "topoFile|$TOPOVAL"; do
  k="${kv%%|*}"; v="${kv#*|}"
  n=$(grep -c "^$k = " "$D/win1.in")
  [ "$n" -eq 1 ] || die "win1.in carries $n '$k' lines, wanted exactly 1 -- does $BASE define it?"
  got=$(grep -m1 "^$k = " "$D/win1.in" | sed "s|^$k = ||")
  [ "$got" = "$v" ] || die "win1.in has $k = '$got', asked for '$v' (the sed did not land)"
done
FULLFRQ="$TOT"; [ "$SKIP_NT" -gt 0 ] && FULLFRQ="$SKIP_NT"
printf 'ioLPDMmode = 1\nioLPDMfullFrq = %d\n' "$FULLFRQ" >> "$D/win1.in"
[ -n "$EXTRA" ] && cat "$EXTRA" >> "$D/win1.in"
echo "--- single invocation: 0 -> $TOT"
./docker/run_case.sh "$D" "win1.in" "$L/$(basename $D)_win1.log" \
    || die "window failed (see $L/$(basename $D)_win1.log)"
rm -f "$D"/FE_RST.*

# ---- discard the adjustment period, STRUCTURALLY ----------------------------------
# lpdm/fields.py:dump_series globs the whole directory and compute_footprint releases from
# t[0] + t_back, so a field written while the flow was still settling is indistinguishable
# from one written after. Deleting them is the guarantee; asserting on what SURVIVED is
# how we know the deletion happened.
if [ "$SKIP_NT" -gt 0 ]; then
  for f in "$D"/window/FE_WIN.*; do
    n="${f##*.}"
    case "$n" in (*[!0-9]*) continue;; esac
    [ "$n" -lt "$SKIP_NT" ] && rm -f "$f"
  done
  EARLIEST=$(ls -1 "$D"/window/FE_WIN.[0-9]* 2>/dev/null | sed 's/.*\.//' | sort -n | head -1)
  [ "$EARLIEST" = "$SKIP_NT" ] || die "earliest surviving dump is step ${EARLIEST:-none}, wanted $SKIP_NT -- a backward trajectory could reach into the adjustment"
  # AND IT MUST CARRY THE GEOMETRY, or nothing downstream can build a field cache. Under
  # ioLPDMmode only the first file of the run and the ioLPDMfullFrq multiples have zPos,
  # and the first file of the run was just deleted. Assert on the artifact.
  ./docker/pyrun.sh - "$D/window/FE_WIN.$EARLIEST" <<'PYGEO' || die "the earliest surviving dump has no geometry -- lpdm/fields.py cannot build a FieldSet from it"
import sys
from netCDF4 import Dataset
with Dataset(sys.argv[1]) as ds:
    have = [v for v in ("xPos", "yPos", "zPos", "topoPos") if v in ds.variables]
if len(have) < 4:
    print(f"FATAL: {sys.argv[1]} carries only {have}; ioLPDMfullFrq must make this dump "
          f"full-form (io_netcdf.c writes geometry to the first file of the run only)",
          file=sys.stderr)
    raise SystemExit(1)
print(f"      geometry confirmed in the earliest surviving dump ({len(have)} of 4)")
PYGEO
  echo "--- discarded $(python3 -c "print($SKIP_NT//$FRQ)") adjustment dumps; earliest field is now step $EARLIEST = $(python3 -c "print(f'{$SKIP_NT*$DT:.0f}')") s"
fi
NW=$(ls -1 "$D"/window/*.[0-9]* | wc -l)
[ "$NW" -eq "$(((TOT-SKIP_NT)/FRQ + 1))" ] \
  || die "window holds $NW dumps, expected $(((TOT-SKIP_NT)/FRQ + 1))"
printf '%s' "$STAMP" > "$D/window/.window_complete"
echo "--- window complete: $NW dumps, $(du -sh $D/window | cut -f1)"
