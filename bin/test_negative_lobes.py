#!/usr/bin/env python3
"""Negative footprint values are PHYSICAL. Assert that nothing clips them away.

WHY THIS IS A TEST AND NOT A COMMENT. A flux footprint is a signed quantity, and both of
its negative mechanisms are real:

  * an ELEVATED CONCENTRATION MAXIMUM in a convective boundary layer. Air arriving at the
    receptor from above (w_release < 0) has been in contact with the surface too, and its
    touchdown density is not the same as that of air arriving from below. The signed
    difference IS the flux footprint (lpdm/footprint.py:16-23), and near the source the
    downward branch can dominate.
  * WIND TURNING WITH HEIGHT in neutral and stable air. A parcel descending through an
    Ekman layer arrives from a bearing rotated relative to the receptor-level wind, so its
    touchdowns land to one side; the crosswind-integrated profile stays positive while the
    2-D field goes negative on the other flank.

Clipping them would be a silent bias, not a cosmetic one: the estimator would no longer be
an unbiased estimate of <w'c'>, and the ML target would be a different quantity from the
one the LPDM computes. The audit found the production path clean -- the clips that exist
(`np.maximum(f, 0)` in the overlap masks and the positive-part moments) are metric-side and
deliberate -- so this test exists to keep it that way, and to MEASURE what is being
preserved rather than merely assert it.

Two levels:
  1. UNIT -- drive lpdm.footprint.FootprintGrid with a hand-built ensemble whose signs are
     known, and check the sign survives deposition, the CIC spreading and normalisation.
  2. ARTIFACT -- score real persisted footprints: negative-magnitude share, where the
     negatives sit, and (for a touchdowns file) that the subsample kept them too.

usage: bin/test_negative_lobes.py [results/cbl_flat.npz ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.footprint import FootprintGrid            # noqa: E402

NEG_SHARE_MIN = 0.005      # 0.5% of |flux|. Production convective runs measure 5.9-9.5%;
# this is a floor an order of magnitude below them, so it fires on "clipped to zero" and
# not on "this particular case happened to be weakly signed".


def unit_test():
    """A two-particle ensemble with opposite release velocities, at separated locations."""
    g = FootprintGrid(-100.0, 100.0, -100.0, 100.0, 20.0)
    res = dict(
        n=2,
        w_release=np.array([+0.5, -0.5]),
        td_particle=np.array([0, 1]),
        td_x=np.array([30.0, -30.0]),
        td_y=np.array([0.0, 0.0]),
        td_w=np.array([0.4, 0.4]),
        td_t=np.array([10.0, 10.0]),
    )
    g.add(res, 0.0, 0.0)
    f = g.normalised("flux")
    ok = []
    ok.append(("a positive-w release deposits POSITIVE flux", bool(f.max() > 0)))
    ok.append(("a negative-w release deposits NEGATIVE flux", bool(f.min() < 0)))
    # The two are equal and opposite by construction, so they must cancel exactly.
    ok.append(("equal and opposite releases cancel in the integral",
               bool(abs(g.integral()) < 1e-12)))
    # And the CONCENTRATION footprint, which uses |w_td| only, must stay strictly positive:
    # if it went negative the sign had leaked into the wrong estimator.
    ok.append(("the CONCENTRATION footprint stays non-negative",
               bool(g.normalised("conc").min() >= 0.0)))
    return ok


def score_artifact(path):
    d = np.load(path, allow_pickle=True)
    if "les" not in d:
        return None
    f = np.asarray(d["les"])
    if not np.isfinite(f).all():           # isfinite FIRST: inf passes every > (docs/reference/standing-rules.md)
        return dict(path=path, error="non-finite values in the footprint")
    tot = np.abs(f).sum()
    if tot <= 0:
        return dict(path=path, error="empty footprint")
    neg = f < 0
    xc = np.asarray(d["xc"]) if "xc" in d else np.arange(f.shape[1], dtype=float)
    yc = np.asarray(d["yc"]) if "yc" in d else np.arange(f.shape[0], dtype=float)
    X, Y = np.meshgrid(xc, yc)
    r = np.hypot(X, Y)
    jn, inx = np.unravel_index(int(np.argmin(f)), f.shape)
    jp, ipx = np.unravel_index(int(np.argmax(f)), f.shape)
    return dict(
        path=path,
        neg_cell_frac=float(neg.mean()),
        neg_mag_share=float(np.abs(f[neg]).sum() / tot),
        min_over_max=float(f.min() / f.max()),
        r_most_negative=float(r[jn, inx]),
        r_peak=float(r[jp, ipx]),
        # A distance-weighted centroid of the negative lobe says which mechanism it is:
        # an elevated-maximum lobe sits close in, a wind-turning lobe sits out with the
        # positive field and beside it.
        r_neg_centroid=float((np.abs(f[neg]) * r[neg]).sum() / np.abs(f[neg]).sum()),
        r_pos_centroid=float((f[~neg] * r[~neg]).sum() / f[~neg].sum()))


def score_touchdowns(path):
    d = np.load(path, allow_pickle=True)
    wt = np.asarray(d["wt"])
    meta = json.loads(str(d["meta"][0])) if "meta" in d else {}
    neg = wt < 0
    return dict(path=path, n=int(wt.size),
                neg_frac=float(neg.mean()),
                neg_mag_share=float(np.abs(wt[neg]).sum() / np.abs(wt).sum()),
                signed=bool(meta.get("signed", True)),
                unfolded=bool(meta.get("unfolded", True)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="*", help="footprint .npz files (default: convective ones)")
    ap.add_argument("--touchdowns", nargs="*", default=None)
    ap.add_argument("--min-share", type=float, default=NEG_SHARE_MIN)
    a = ap.parse_args()

    print("=== 1. UNIT: does the estimator carry a sign at all? ===")
    fails = 0
    for name, ok in unit_test():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        fails += (not ok)

    paths = a.npz or sorted(glob.glob("results/cbl_*.npz")) + sorted(
        glob.glob("results/g16_cbl_*.npz"))
    print(f"\n=== 2. ARTIFACT: {len(paths)} persisted footprint(s) ===")
    print(f"  {'file':<34} {'neg cells':>10} {'|neg| share':>12} {'min/max':>9} "
          f"{'r(neg centroid)':>16} {'r(pos centroid)':>16}")
    n_scored = 0
    for p in paths:
        r = score_artifact(p)
        if r is None:
            continue
        if "error" in r:
            print(f"  {os.path.basename(p):<34}  {r['error']}")
            fails += 1
            continue
        n_scored += 1
        flag = "" if r["neg_mag_share"] >= a.min_share else "   <-- CLIPPED?"
        fails += (r["neg_mag_share"] < a.min_share)
        print(f"  {os.path.basename(p):<34} {100*r['neg_cell_frac']:>9.1f}% "
              f"{100*r['neg_mag_share']:>11.2f}% {r['min_over_max']:>9.3f} "
              f"{r['r_neg_centroid']:>15.0f} m {r['r_pos_centroid']:>15.0f} m{flag}")

    tds = a.touchdowns if a.touchdowns is not None else sorted(
        glob.glob("results/*_touchdowns.npz"))
    if tds:
        print(f"\n=== 3. TOUCHDOWN SAMPLES: does the ML target keep the sign? ===")
        for p in tds:
            r = score_touchdowns(p)
            ok = r["neg_frac"] > 0 and r["signed"]
            fails += (not ok)
            print(f"  [{'PASS' if ok else 'FAIL'}] {os.path.basename(p):<34} "
                  f"n={r['n']:,}  {100*r['neg_frac']:.1f}% negative weights, "
                  f"|neg| share {100*r['neg_mag_share']:.2f}%, "
                  f"signed={r['signed']} unfolded={r['unfolded']}")
    else:
        print("\n=== 3. TOUCHDOWN SAMPLES: none on disk yet (run stage5 with "
              "--keep-touchdowns) ===")

    print(f"\n  {n_scored} footprint(s) scored, {fails} failure(s)")
    if not n_scored and not a.npz:
        print("  NOTE: no convective footprint was found, so the ARTIFACT half of this "
              "test did not run. A neutral footprint carries a much weaker negative lobe "
              "and is not the right specimen -- point this at a convective one.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
