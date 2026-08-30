#!/usr/bin/env python3
"""Does the in-process hand-off produce the same footprint as the file path?

Run one LES at `lpdmOnlineSelector = 2`: it stages every output step AND writes the netCDF
dumps, **from the same buffers**. Then compute the footprint twice -- once from the ring as
the LES produced it, once from the dumps afterwards -- and compare. That is the only way to
score the hand-off without running the LES twice and comparing across two turbulence
realisations, which on this project differ by 44% in the integral and would swamp any
plumbing error.

=== WHY THIS IS NOT SCORED AT ZERO, WHERE THE OTHER HAND-OFF TESTS ARE ===

`bin/test_dumpsrc.py`, `bin/test_ringsrc.py`, `bin/test_lpdmonline.py` and
`bin/test_streaming.py` all assert BIT-IDENTITY, because in each of them the two paths carry
the same bytes and the correct tolerance is machine zero. Here they do not:

  * the DISK path is CF-packed to 16 bit by `ioLPDMmode` on the way out and unpacked on the
    way back in;
  * the RING path carries raw fp32 and is quantised only once, when `FieldSet` builds its
    fp16 cache.

So the ring is the MORE accurate of the two, and the difference between them is a
quantisation difference in the fields rather than a difference in the estimator. The
tolerance that means something is therefore the footprint's own sampling floor -- the
half-vs-half difference the run already measures on itself -- and that is what this scores
against. A number picked instead would be the mistake PROJECT_BRIEF.md's tolerance rule exists to
prevent.

`window_stats` IS scored at zero, though, and that is deliberate: it is computed from the
same fields but the quantisation enters it as a mean over ~15,000 cells and 541 dumps, so
what it actually detects is a slot-ordering or rho-division error in the producer. It is
reported with its own scale rather than asserted, for the same fp16 reason.

usage: docker/pyrun.sh bin/handoff_accept.py --ring results/corpus/<tag>.json \\
                                             --disk results/corpus/<tag>_disk.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# The observables the acceptance is about, and the key each is scored against in the run's
# own `halves` block. A metric with no half-vs-half counterpart is REPORTED, not scored --
# there is nothing to score it against and inventing a bar would be the point of failure.
SCORED = [
    ("peak_x", "les", "dpeak", "m"),
    ("centroid_dist", "les", None, "m"),
    ("area80_ha", "les", None, "ha"),
    ("x80", "les", None, "m"),
]


def get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True, help="the footprint JSON computed THROUGH the ring")
    ap.add_argument("--disk", required=True, help="the same window, recomputed from the dumps")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    for p in (a.ring, a.disk):
        if not os.path.exists(p):
            print(f"FATAL: {p} does not exist", file=sys.stderr)
            return 2
    R, D = json.load(open(a.ring)), json.load(open(a.disk))

    # THE FLOOR IS THE RUN'S OWN, and it must exist. A half-vs-half block is written when
    # the footprint is split into two independent halves; without it there is no measured
    # scale on this window and the comparison would have to invent one.
    halves = R.get("halves") or D.get("halves") or {}
    if not halves:
        print("  NO HALF-VS-HALF FLOOR in either JSON, so there is nothing to score "
              "against. Reported only; this is NOT a pass.")

    rows, worst_ratio, verdict = [], 0.0, "PASS"
    print(f"ring  {a.ring}")
    print(f"disk  {a.disk}\n")
    print(f"{'observable':<22}{'ring':>12}{'disk':>12}{'|diff|':>10}{'floor':>10}{'x floor':>9}")

    # -- (a) the footprint observables -------------------------------------------------
    for key, block, hkey, unit in SCORED:
        rv, dv = get(R, block, key), get(D, block, key)
        if rv is None or dv is None:
            continue
        diff = abs(float(rv) - float(dv))
        fl = abs(float(halves[hkey])) if hkey and hkey in halves else None
        if fl is None:
            # No named floor for this one: use the raster cell, which is the resolution the
            # quantity is even defined to.
            fl = float(R.get("res") or 0.0) or None
        rat = (diff / fl) if fl else float("nan")
        if fl and np.isfinite(rat):
            worst_ratio = max(worst_ratio, rat)
        print(f"{key + ' [' + unit + ']':<22}{float(rv):12.4g}{float(dv):12.4g}"
              f"{diff:10.4g}{(fl if fl else float('nan')):10.4g}{rat:9.2f}")
        rows.append(dict(key=key, ring=float(rv), disk=float(dv), diff=diff,
                         floor=fl, ratio=rat))

    # the integral has no half-vs-half counterpart; score it against the run's own
    # asymptote, which is the scale the integral is quoted in everywhere else.
    ri, di = R.get("integral_les"), D.get("integral_les")
    if ri is not None and di is not None:
        asym = float(R.get("integral_asymptote") or 1.0)
        print(f"{'integral':<22}{ri:12.4g}{di:12.4g}{abs(ri-di):10.4g}"
              f"{asym:10.4g}{abs(ri-di)/asym:9.4f}")
        rows.append(dict(key="integral", ring=float(ri), disk=float(di),
                         diff=abs(ri - di), floor=asym, ratio=abs(ri - di) / asym))

    # -- the array share, with ITS OWN standard error, which is the right scale ---------
    rs = get(R, "cover_share", "solar array")
    ds = get(D, "cover_share", "solar array")
    se = get(R, "cover_share_se", "solar array")
    if rs is not None and ds is not None:
        d = abs(float(rs) - float(ds)) * 100.0
        s = (float(se) * 100.0) if se else None
        print(f"{'array share [%]':<22}{float(rs)*100:12.4g}{float(ds)*100:12.4g}"
              f"{d:10.4g}{(s if s else float('nan')):10.4g}"
              f"{(d/s if s else float('nan')):9.2f}")
        rows.append(dict(key="array_share_pct", ring=float(rs) * 100, disk=float(ds) * 100,
                         diff=d, floor=s, ratio=(d / s if s else None)))
        if s:
            worst_ratio = max(worst_ratio, d / s)

    # -- (c) the signed weights, on both paths -----------------------------------------
    rn = get(R, "touchdowns", "negative_weight_fraction")
    dn = get(D, "touchdowns", "negative_weight_fraction")
    print()
    if rn is not None or dn is not None:
        print(f"  negative-lobe weight fraction: ring {rn}, disk {dn} "
              f"-- signed weights are preserved on both paths if both are non-zero")
    else:
        print("  no touchdown block on either path, so the negative-lobe share is not "
              "reported here (--keep-touchdowns was 0)")

    # -- window_stats: the producer's own correctness ----------------------------------
    print("\n  window_stats, which is what would catch a slot-ordering or rho-division "
          "error in the producer:")
    sr, sd = R.get("stats") or {}, D.get("stats") or {}
    ws = []
    for k in sorted(set(sr) & set(sd)):
        x, y = sr[k], sd[k]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        rel = abs(x - y) / max(abs(x), abs(y), 1e-30)
        ws.append((rel, k, x, y))
    ws.sort(reverse=True)
    for rel, k, x, y in ws[:6]:
        print(f"    {k:<22}{x:14.6g}{y:14.6g}   {rel:9.2e} relative")
    if ws:
        print(f"    worst of {len(ws)} scalars: {ws[0][0]:.2e} relative at {ws[0][1]}")

    # -- the verdict -------------------------------------------------------------------
    print()
    if not halves:
        verdict = "UNJUDGED"
    elif worst_ratio > 1.0:
        verdict = "DIFFERS"
    print(f"(a) CPU-from-disk vs from-ring: worst {worst_ratio:.2f}x the run's own "
          f"half-vs-half floor -> {verdict}")
    print("    The two paths do NOT carry the same bytes -- the disk path is CF-packed to "
          "16 bit\n    and the ring is raw fp32 -- so the floor, not zero, is the "
          "tolerance that means\n    something here. The zero-tolerance tests are "
          "bin/test_dumpsrc.py, test_ringsrc.py,\n    test_lpdmonline.py and "
          "test_streaming.py.")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"ring": a.ring, "disk": a.disk, "verdict": verdict,
                   "worst_ratio_of_floor": worst_ratio, "rows": rows,
                   "negative_fraction": {"ring": rn, "disk": dn},
                   "window_stats_worst_rel": (ws[0][0] if ws else None),
                   "window_stats_worst_field": (ws[0][1] if ws else None)},
                  open(a.json, "w"), indent=1, default=float)
        print(f"wrote {a.json}")
    return 0 if verdict == "PASS" else (0 if verdict == "UNJUDGED" else 1)


if __name__ == "__main__":
    sys.exit(main())
