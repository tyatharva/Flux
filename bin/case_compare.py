#!/usr/bin/env python3
"""Compare corpus cases: what each one produced, and how they differ from each other.

ONE CASE IS AN ANECDOTE. Eight defects surfaced in the first corpus case alone, every one
of them producing a plausible number rather than an error, and none of them detectable
from that case's own output -- there was nothing to compare it against. A second and third
case are the first evidence of a per-case failure RATE, and of which quantities are stable
across cases and which are not. This assembles that comparison from artifacts that survive
the run: the footprint JSON, the pair record, and the seed pick.

WHAT IT DELIBERATELY PUTS SIDE BY SIDE, and why each one is here:

  peak / centroid / A80 / integral   the footprint, and the two SHARP fault detectors
                                     among them are the peak and the integral
  array share WITH ITS SE            the scientific result. Never read without the SE:
                                     a 6x error in h moved it 0.8 points against 3.66
  floor factor range, peak height,   the closure. The h defect read fac = 1.000 AT THE
  and the value AT THE RECEPTOR      RECEPTOR while running at 9e4 aloft, so both are shown
  seed mismatch, requested->achieved what the 30-minute adjustment actually absorbed. The
                                     design assumed direction CLOSES and depth barely
                                     moves; measured, direction WIDENED and depth
                                     over-closed at 3x the budgeted rate
  sigma_w at 10 m vs the tower       the only EXTERNAL check in the project -- one year of
                                     half-hourly eddy covariance at the real receptor,
                                     never used for training, tuning or forcing

usage: case_compare.py case_A case_B ...        (tags, not paths)
       case_compare.py --all
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

FPDIR = "results/corpus"
PAIRDIR = "pairs"
PICKDIR = "results/pick"
FORCEDIR = "results/forcing"


def load(tag):
    def j(p):
        return json.load(open(p)) if os.path.exists(p) else {}
    return dict(tag=tag, fp=j(f"{FPDIR}/{tag}.json"), pair=j(f"{PAIRDIR}/{tag}.json"),
                pick=j(f"{PICKDIR}/{tag}.json"), frc=j(f"{FORCEDIR}/{tag}.json"))


def tower_sigma_w(H, band=10.0, csv_path="data/raw/H_and_sigma_w.csv"):
    """The measured sigma_w distribution at this sensible heat flux. A WEAK check."""
    if not os.path.exists(csv_path):
        return None
    Hs, S = [], []
    for row in csv.DictReader(open(csv_path)):
        try:
            h_, s_ = float(row["H"]), float(row["sigma_w"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(h_) and np.isfinite(s_) and s_ > 0:
            Hs.append(h_); S.append(s_)
    Hs, S = np.asarray(Hs), np.asarray(S)
    m = np.abs(Hs - H) <= band
    if m.sum() < 30:
        return None
    q = np.percentile(S[m], [25, 50, 75])
    return dict(n=int(m.sum()), p25=q[0], p50=q[1], p75=q[2])


def row_of(c):
    fp, pair, pick, frc = c["fp"], c["pair"], c["pick"], c["frc"]
    les, klj = fp.get("les") or {}, fp.get("kljun") or {}
    hl = ((fp.get("floor") or {}).get("health")) or {}
    st = fp.get("stats") or {}
    lab = (frc.get("labels") or {})
    # THE PAIR IS AUTHORITATIVE, THE PICK IS THE FALLBACK. make_pair.py writes the seed
    # block that actually shipped with the training record, and a pair can be re-stamped
    # (case_2023031014's gate_state was backfilled after the gate learned to say
    # INDETERMINATE) without the pick being regenerated. Reading the pick first would show
    # the stale value and call it "unjudged".
    ch = dict(pick.get("chosen") or {})
    ch.update({k: v for k, v in (pair.get("seed") or {}).items() if v is not None})
    amr = ((pair.get("forcing") or {}).get("achieved_minus_requested")) or {}
    share = (fp.get("cover_share") or {}).get("solar array")
    se = (fp.get("cover_share_se") or {}).get("solar array")
    H_w = None
    if lab.get("wth_sensible") is not None:
        H_w = float(lab["wth_sensible"]) * 1.2 * 1005.0
    elif lab.get("shtfl_wm2") is not None:
        H_w = float(lab["shtfl_wm2"])
    return dict(
        tag=c["tag"], seed=ch.get("job"), rot=ch.get("rot"),
        gate_state=ch.get("gate_state"),
        gate_indet=", ".join(ch.get("gate_indeterminate") or []),
        dwdir_window=fp.get("dwdir_dt_window_deg_per_h"),
        seed_drift=ch.get("dwdir_dt_deg_per_h"),
        peak_x=les.get("peak_x"), peak_klj=klj.get("peak_x"),
        centroid=les.get("centroid_dist"), bearing=les.get("centroid_bearing"),
        a80=les.get("area80_ha"), a80_klj=klj.get("area80_ha"),
        integral=fp.get("integral_les"), integral_klj=fp.get("integral_kljun"),
        share=share, share_se=se, overlap=fp.get("overlap_kljun"),
        fac_min=hl.get("fac_min"), fac_max=hl.get("fac_max"),
        z_fac_max=hl.get("z_fac_max"), fac_recept=hl.get("fac_at_receptor"),
        fsgs_recept=hl.get("f_sgs_at_receptor"), fsgs_peak=hl.get("f_sgs_at_peak"),
        sw_over_us=hl.get("sigma_w_over_ustar_at_receptor"),
        h=st.get("h"), ustar=st.get("ustar"), sigma_w=st.get("sigma_w"),
        wdir=st.get("wdir"), H_w=H_w,
        d_zi=amr.get("zi_m"), d_dir=amr.get("dir_deg"),
        req_zi=lab.get("zi_m"), req_dir=lab.get("predicted_10m_dir_deg"),
        seed_zi=ch.get("seed_zi_m"), seed_dir=ch.get("seed_dir_deg"),
        seed_dir_frozen=ch.get("seed_dir_frozen_deg"),
        gap_dir=ch.get("d_dir_deg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    tags = a.tags or ([os.path.basename(p)[:-5] for p in sorted(glob.glob(f"{FPDIR}/*.json"))
                       if not p.endswith(".monitor.json")] if a.all else [])
    if not tags:
        print("usage: case_compare.py <tag> [<tag> ...]  |  --all", file=sys.stderr)
        return 2
    rows = [row_of(load(t)) for t in tags]

    f = lambda v, s="{:.3f}": ("n/a" if v is None else s.format(v))
    for r in rows:
        gs = r.get("gate_state") or "unjudged"
        print(f"\n=== {r['tag']}  (seed {r['seed']} rot {r['rot']}, gate {gs}) ===")
        if gs == "INDETERMINATE":
            print(f"  ^ the seed's stationarity is UNESTABLISHED on {r['gate_indet']} -- "
                  f"the library's normal state, not an exception: those two limits cannot "
                  f"be resolved in a 3.0 h spin-up at any window width.")
        print(f"  footprint   peak {f(r['peak_x'],'{:.0f}')} m (Kljun {f(r['peak_klj'],'{:.0f}')}),"
              f"  centroid {f(r['centroid'],'{:.0f}')} m at {f(r['bearing'],'{:.1f}')} deg,"
              f"  A80 {f(r['a80'],'{:.2f}')} ha (Kljun {f(r['a80_klj'],'{:.2f}')}),"
              f"  integral {f(r['integral'])} (Kljun {f(r['integral_klj'])})")
        sh = ("n/a" if r["share"] is None else
              f"{100*r['share']:.2f}%" + (f" +/- {100*r['share_se']:.2f}" if r["share_se"] else " (no SE)"))
        print(f"  array share {sh}   -- REPORTED, not gated; against 1.03% of the box by area")
        sg = ("n/a" if r["fsgs_recept"] is None
              else f"{100*r['fsgs_recept']:.1f}% sub-grid")
        print(f"  closure     floor {f(r['fac_min'])}-{f(r['fac_max'],'{:.4g}')} peaking at "
              f"z={f(r['z_fac_max'],'{:.0f}')} m;  AT THE RECEPTOR factor "
              f"{f(r['fac_recept'])}, {sg};  f_sgs at the floor's peak "
              f"{f(r['fsgs_peak'])};  sigma_w/u* {f(r['sw_over_us'],'{:.2f}')}")
        print(f"  seed        z_i {f(r['seed_zi'],'{:.0f}')} -> requested {f(r['req_zi'],'{:.0f}')} "
              f"-> ACHIEVED {f(r['h'],'{:.0f}')} m   (achieved-requested {f(r['d_zi'],'{:+.0f}')} m)")
        print(f"              dir {f(r['seed_dir'],'{:.1f}')} -> requested {f(r['req_dir'],'{:.1f}')} "
              f"-> ACHIEVED {f(r['wdir'],'{:.1f}')} deg  (achieved-requested {f(r['d_dir'],'{:+.1f}')} deg,"
              f" pick gap was {f(r['gap_dir'],'{:.1f}')})")
        print(f"              drift: seed at freeze {f(r['seed_drift'],'{:+.2f}')} deg/h;"
              f"  ACROSS THIS WINDOW {f(r['dwdir_window'],'{:+.2f}')} deg/h"
              f"   -- does the case inherit the seed's rate?")
        if r["sigma_w"] is not None and r["H_w"] is not None:
            tw = tower_sigma_w(r["H_w"])
            if tw:
                inside = tw["p25"] <= r["sigma_w"] <= tw["p75"]
                print(f"  EXTERNAL    LES sigma_w(10 m) {r['sigma_w']:.3f} m/s vs the tower at "
                      f"H = {r['H_w']:+.0f} W/m2: median {tw['p50']:.3f}, IQR "
                      f"[{tw['p25']:.3f}, {tw['p75']:.3f}] over {tw['n']} half-hours "
                      f"-> {'INSIDE' if inside else 'OUTSIDE'} the IQR "
                      f"({r['sigma_w']/tw['p50']:.2f}x the median). An order-of-magnitude "
                      f"check only: the file carries no wind speed and the IQR spans "
                      f"{tw['p75']/tw['p25']:.2f}x.")

    if len(rows) > 1:
        print(f"\n=== case-to-case spread over {len(rows)} cases ===")
        keys = [("peak_x", "peak (m)"), ("centroid", "centroid (m)"), ("a80", "A80 (ha)"),
                ("integral", "integral"), ("share", "array share"),
                ("fac_max", "floor max"), ("fsgs_recept", "f_sgs at receptor"),
                ("fsgs_peak", "f_sgs at floor peak"), ("h", "achieved h (m)"),
                ("dwdir_window", "drift in window (deg/h)"),
                ("ustar", "u*"), ("sigma_w", "sigma_w(10 m)")]
        print(f"  {'metric':22}{'min':>11}{'median':>11}{'max':>11}{'max/min':>10}")
        for k, nm in keys:
            v = np.array([r[k] for r in rows if r[k] is not None], float)
            if v.size < 2:
                continue
            sp = v.max() / v.min() if v.min() > 0 else float("nan")
            print(f"  {nm:22}{v.min():11.4g}{np.median(v):11.4g}{v.max():11.4g}{sp:10.2f}x")
        print("\n  READ THE SHARE SPREAD AGAINST ITS OWN SE, NOT AS A RANGE. Each case's")
        print("  array share carries a 3-4 point standard error over 10 release groups, so")
        print("  a several-point difference between cases is not necessarily a difference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
