#!/usr/bin/env python3
"""How much of the footprint was the sigma_w floor's artifact?

The retired floor manufactured a maximum in sigma_w^2 at the taper's inner edge. Thomson's
drift points inward from both sides of a maximum, so particles converged on it instead of
descending -- and a particle that lingers near the release column touches down close to the
tower. The solar array sits within 250 m of the tower in every direction, so a spurious
near-field concentration inflates the array share directly. That is measurable, and this
measures it: the same LES fields, the same releases, the same seed, three closures.

  usage: floor_bias.py <label>=<tag> [<label>=<tag> ...]
         floor_bias.py new=g16_cbl_wN legacy=g16_cbl_wN_legacy none=g16_cbl_wN_nofloor

Reports the radial CDF of footprint weight about the tower, the share inside the array's
reach, the array cover share, and the integral -- differenced against the FIRST label,
which is the reference.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.sgs_floor import check_monotone

RINGS = (50.0, 100.0, 150.0, 250.0, 400.0, 600.0, 900.0)


def load(tag, outdir="results"):
    with open(os.path.join(outdir, f"{tag}.json")) as f:
        j = json.load(f)
    z = np.load(os.path.join(outdir, f"{tag}.npz"))
    return j, z


def radial(z, res):
    """Cumulative share of |footprint weight| inside radius r, and the weighted median."""
    x, y = np.meshgrid(z["xc"], z["yc"])
    r = np.hypot(x, y)
    f = np.asarray(z["les"], dtype=np.float64)
    tot = f.sum()
    cum = {R: float(f[r <= R].sum() / tot) for R in RINGS}
    o = np.argsort(r.ravel())
    c = np.cumsum(f.ravel()[o]) / tot
    med = float(r.ravel()[o][np.searchsorted(c, 0.5)])
    return cum, med, tot


def main():
    items = []
    for arg in sys.argv[1:]:
        if "=" not in arg:
            print(__doc__)
            return 2
        lab, tag = arg.split("=", 1)
        if not os.path.exists(f"results/{tag}.json"):
            print(f"  (missing: results/{tag}.json -- skipped)")
            continue
        items.append((lab, tag) + load(tag))
    if len(items) < 2:
        print("  need at least two closures to difference")
        return 2

    ref = items[0]
    print(f"\n=== sigma_w floor: what the closure was worth ===")
    print(f"  reference closure: {ref[0]} ({ref[1]})")
    print(f"  {'closure':<10} " + " ".join(f"{'<'+str(int(R)):>7}" for R in RINGS)
          + f" {'r50':>7} {'array%':>8} {'integral':>9} {'peak':>6} {'x80':>7}")
    rows = []
    for lab, tag, j, z in items:
        cum, med, tot = radial(z, j["res"])
        arr = 100.0 * float(j.get("cover_share", {}).get("solar array", np.nan))
        rows.append((lab, tag, cum, med, arr, j["integral_les"], j["les"]["peak_x"],
                     j["les"].get("x80", np.nan), j))
        print(f"  {lab:<10} " + " ".join(f"{100*cum[R]:7.2f}" for R in RINGS)
              + f" {med:7.0f} {arr:8.2f} {j['integral_les']:9.3f} "
                f"{j['les']['peak_x']:6.0f} {j['les'].get('x80', float('nan')):7.0f}")

    r0 = rows[0]
    print(f"\n  differences from {r0[0]} (points of share, absolute for the rest):")
    for row in rows[1:]:
        d = {R: 100.0 * (row[2][R] - r0[2][R]) for R in RINGS}
        print(f"  {row[0]:<10} " + " ".join(f"{d[R]:+7.2f}" for R in RINGS)
              + f" {row[3]-r0[3]:+7.0f} {row[4]-r0[4]:+8.2f} "
                f"{row[5]-r0[5]:+9.3f} {row[6]-r0[6]:+6.0f} {row[7]-r0[7]:+7.0f}")

    # The closure profile behind each, if the run persisted it.
    print("\n  closure profiles at the receptor:")
    for lab, tag, j, z in items:
        fl = j.get("floor")
        if not fl:
            print(f"  {lab:<10} (no floor: sgs_most off)")
            continue
        zl = np.asarray(fl["zl"]); fac = np.asarray(fl["fac"])
        sig2 = np.asarray(fl["sig2"]); base = np.asarray(fl["base"])
        k = int(np.argmin(np.abs(zl - j.get("z_target", 10.0))))
        # FLOOR-INDUCED turnovers only. sigma_w^2 also falls from the first model level
        # through the lowest tens of metres in the model's OWN profile, because the
        # sub-grid closure piles energy against the wall -- that is not the floor's doing
        # and counting it made the retired taper look better than the corrected one.
        # check_monotone() differences the two profiles, which is the whole question.
        top = float(zl[fl["kpk"]]) if fl.get("kpk", -1) > 0 else 0.2 * float(fl["h"])
        n_new, worst = check_monotone(
            dict(zl=zl, sig2=sig2, base=np.asarray(fl["base"]), kpk=fl.get("kpk", -1)),
            z_top=top)
        print(f"  {lab:<10} factor {fac[k]:6.3f} at the receptor, max {fac.max():6.2f} at "
              f"z={zl[int(np.argmax(fac))]:5.0f} m; sigma_w^2 {base[k]:.4f} -> {sig2[k]:.4f}; "
              f"{n_new} FLOOR-INDUCED turnover(s) below z={top:.0f} m (worst {worst:+.2%})"
              f"{'  [LEGACY TAPER]' if fl.get('legacy') else ''}")

    # The one number the exercise exists to produce.
    leg = [r for r in rows if "legacy" in r[0].lower()]
    if leg and np.isfinite(r0[4]) and np.isfinite(leg[0][4]):
        d = leg[0][4] - r0[4]
        print(f"\n  ARRAY SHARE: {leg[0][0]} {leg[0][4]:.2f}%  vs  {r0[0]} {r0[4]:.2f}%  "
              f"-> the retired closure was worth {d:+.2f} points "
              f"({100*d/max(r0[4],1e-9):+.1f}% relative)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
