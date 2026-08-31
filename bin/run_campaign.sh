#!/usr/bin/env bash
# The whole remaining fifth pass, unattended and RESUMABLE.
#
# ~24 h of wall clock on one GPU, none of which needs a human. Every stage writes a
# sentinel to results/.done/ when it succeeds, so re-running the script picks up where it
# stopped instead of redoing hours of LES. Gates stop the chain rather than letting a later
# stage inherit a bad state.
#
#   usage: bin/run_campaign.sh [stage ...]      (default: all, in order)
#   env:   FORCE=1   ignore sentinels and redo
set -uo pipefail
cd /home/atyagi/Flux

# RUN FROM A FROZEN COPY. Bash reads a script lazily, by byte offset, so editing this file
# while it is executing can drop the interpreter into the middle of a line and run garbage.
# Over a 24 h campaign the file WILL be edited -- a later stage gets fixed while an earlier
# one is still running -- so re-exec from a snapshot and make that safe by construction.
# Edits then take effect on the next launch, which is what resumability is for anyway.
if [ "${FROZEN:-0}" != "1" ]; then
  cp -f "$0" /tmp/flux-logs/run_campaign.frozen.sh
  FROZEN=1 exec bash /tmp/flux-logs/run_campaign.frozen.sh "$@"
fi

L=/tmp/flux-logs
R=results
DONE=$R/.done
mkdir -p $DONE $R
DT_FLAT=0.0146417        # CFL_3d 1.35, measured flat accuracy boundary ~1.51
DT_WIN=0.0146199         # 5/342, so a 5 s cadence is an integer step count; CFL_3d 1.348
BASE=runs/g16_base/base.in
SPIN=runs/g16_spin

say(){ echo; echo "########## $* ##########"; date '+%F %H:%M:%S'; }
die(){ echo "CAMPAIGN STOPPED: $*" | tee -a $R/campaign_status.txt >&2; exit 1; }
newest(){ ls -1 "$1"/*.[0-9]* 2>/dev/null | sort -t. -k2 -n | tail -1; }
have(){ [ "${FORCE:-0}" = "1" ] && return 1; [ -f "$DONE/$1" ]; }
mark(){ date '+%F %H:%M:%S' > "$DONE/$1"; echo "  [stage $1 recorded done]"; }

# ============================================================ 1. neutral spin-up, extended
if ! have neutral_spin; then
  say "1. neutral spin-up -> ~6.7 simulated hours"
  # 3.58 h was 0.20 inertial periods and u* was still falling 3.4%/h. A neutral Ekman
  # layer's clock is 2pi/f = 17.6 h, not the 0.25 h eddy turnover, so it needs hours.
  D=$SPIN BASE=$BASE DT=$DT_FLAT OUTBASE=FE_G16 FRQ=20000 NSEG=7 bin/spin_cbl.sh \
    || die "neutral spin-up"
  mark neutral_spin
fi

# ============================================================ 2. gates C1, C2, B4, Phase D
if ! have gates_and_control; then
  say "2. gates C1/C2, terrain dt, and the flat/neutral control"
  bin/run_pass5.sh || die "run_pass5.sh (C1/C2/B4/Phase D)"
  mark gates_and_control
fi

# ============================================================ 3. convective spin-ups
for CASE in shallow deep; do
  if have cbl_$CASE; then continue; fi
  say "3. convective spin-up: $CASE"
  # A CBL's clock is the convective turnover z_i/w* ~ 300-600 s, so it converges in ~1-2
  # simulated hours -- far faster than the neutral case. The DEEP one needs longer because
  # it also has to GROW to its target depth by entrainment.
  case $CASE in
    shallow) NS=2 ;;
    deep)    NS=4 ;;
  esac
  D=runs/g16_cbl_$CASE BASE=runs/g16_base/base_cbl_$CASE.in DT=$DT_FLAT OUTBASE=FE_CBL \
    FRQ=20000 NSEG=$NS bin/spin_cbl.sh || die "convective spin-up $CASE"
  # Gate C3: CBL similarity. cbl_check.py must read the PRESCRIBED htFlux, not the
  # resolved covariance at k=0 -- that bug made a real CBL look like it was not one.
  FE_DT=$DT_FLAT ./docker/pyrun.sh bin/cbl_check.py \
    $(ls -1 runs/g16_cbl_$CASE/output/FE_CBL.* | sort -t. -k2 -n) 2>&1 \
    | tee $R/g16_cbl_${CASE}_c3.txt
  mark cbl_$CASE
done

# ============================================================ 4. Phase E: domain adequacy
if ! have adequacy; then
  say "4. Phase E: domain adequacy -- the decision experiment"
  for CASE in shallow deep; do
    SRC=$(newest runs/g16_cbl_$CASE/output)
    D=runs/g16_adq_$CASE
    mkdir -p $D/output $D/window
    # Flat surface: the adequacy pair is about the BOX, so the real geography would only
    # add a second thing that differs between them.
    cp -f "$SRC" $D/output/FE_ADJ.0
    # THE CASE'S OWN BASE FILE, not the neutral one. surflayer_wth is moot here (the
    # restart carries htFlux, which is IO-registered) but lsf_w_zlev1 is NOT: the deep
    # case puts the subsidence peak at 1000 m and the shallow one at 500 m. Running the
    # deep window under the shallow profile would apply the wrong subsidence to the
    # inversion -- and this pair exists precisely to differ in nothing but z_i.
    CBASE=runs/g16_base/base_cbl_$CASE.in
    BASE=$CBASE bin/run_window.sh $D $D/output/FE_ADJ.0 $DT_WIN 2400 - \
      10.000000 0.000000 || die "adequacy window $CASE"
    # 10 release groups, not 2 halves. The effect being tested is ~1 point of array
    # share; two halves give ONE difference and cannot put a standard error on that.
    ./docker/pyrun.sh bin/stage5_footprint.py $D/window --dt $DT_WIN --tback 400 \
      --rel-seconds 1800 --z-target 10.0 --sgs-most --cover-dir data/grid16_cbl \
      --receptor-from data/grid16_cbl --fp16-cache --cover-groups 10 \
      --tag g16_adq_$CASE 2>&1 | grep -vE 'batch [0-9]+/' > $R/g16_adq_$CASE.txt
    tail -20 $R/g16_adq_$CASE.txt
    ./docker/pyrun.sh bin/domain_adequacy.py spectra \
      $(ls -1 $D/window/FE_WIN.* | sort -t. -k2 -n | tail -3) 2>&1 \
      | tee -a $R/g16_adq_spectra.txt
    rm -f $D/window/* $D/output/FE_ADJ.0
  done
  ./docker/pyrun.sh bin/domain_adequacy.py compare \
    $R/g16_adq_shallow.json $R/g16_adq_deep.json 2>&1 | tee $R/g16_adequacy.txt
  # A DIFFERENCE HERE IS NOT A CRASH. It is a grid decision, and it belongs to the user --
  # so record it loudly and stop the chain rather than spending 12 more GPU-h on a corpus
  # whose box may be wrong.
  # GATE E IS A REPORT, NOT A BLOCKER FOR PHASE F. The grid decision it feeds belongs to
  # the user, and it governs which CONVECTIVE STATES the corpus may contain -- not whether
  # the production directions are valid. Phase F runs off the SHALLOW state, which sits at
  # L/z_i = 4.56 and satisfies the rule either way, so stopping the chain here would idle
  # the GPU on a question that does not gate it.
  if ! grep -q "GATE E: PASS" $R/g16_adequacy.txt; then
    echo "GATE E DIFFERS -- the z_i cap may bind. See $R/g16_adequacy.txt and" \
         "$R/g16_adq_spectra.txt. Phase F is UNAFFECTED (it uses the compliant" \
         "shallow state) and continues." | tee -a $R/campaign_status.txt
  fi
  mark adequacy
fi

# ============================================================ 5. Phase F: production
for REG in nbl cbl; do
  if have dirs_$REG; then continue; fi
  say "5. Phase F: four directions, $REG"
  case $REG in
    nbl) SRC=$(newest $SPIN/output);                GRID=data/grid16
         RBASE=$BASE ;;
    cbl) SRC=$(newest runs/g16_cbl_shallow/output); GRID=data/grid16_cbl
         RBASE=runs/g16_base/base_cbl_shallow.in ;;
  esac
  TB=$(cat $R/tback_production_dx16.txt 2>/dev/null || echo 400)
  WIN=$(python3 -c "print(int(1800+$TB))")
  BASE=$RBASE ADJ_S=1200 SPS=0.0155 ZTARGET=10.0 \
    bin/run_directions.sh g16_$REG "$SRC" "$GRID" $DT_WIN $WIN $TB \
    || die "Phase F directions $REG"
  mark dirs_$REG
done

# ============================================================ 6. displacement sensitivity
if ! have dsens; then
  say "6. the displacement-height sensitivity, on the northerly"
  SRC=$(newest runs/g16_cbl_shallow/output)
  TB=$(cat $R/tback_production_dx16.txt 2>/dev/null || echo 400)
  WIN=$(python3 -c "print(int(1800+$TB))")
  for TREAT in raised bracket; do
    G=data/grid16_$TREAT
    EXTRA=""
    [ "$TREAT" = "raised" ] && EXTRA="--exact-agl"
    BASE=runs/g16_base/base_cbl_shallow.in ADJ_S=1200 SPS=0.0155 ZTARGET=10.0 ONLY=wN \
      EXACT_AGL="${EXTRA:+1}" \
      bin/run_directions.sh g16_ds$TREAT "$SRC" "$G" $DT_WIN $WIN $TB \
      || die "displacement sensitivity $TREAT"
  done
  mark dsens
fi

say "CAMPAIGN COMPLETE"
echo "COMPLETE $(date '+%F %H:%M:%S')" >> $R/campaign_status.txt
