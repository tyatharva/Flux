#!/usr/bin/env python3
"""Is the turbulence still ALIVE? The physics companion to the k0/k1 accuracy check.

WHY THIS EXISTS. docker/k0k1_check.py reads the ratio of resolved w variance between the
first two model levels and fails when dt is above FastEddy's accuracy CFL. It is a good
check and it is a dt CHECK. It read **0.442 -- a comfortable pass -- on a stable boundary
layer whose turbulence had entirely collapsed**: u* 0.236 -> 0.098 m/s, z/L +2.67, the
flow above 66 m exactly geostrophic. The ratio stayed healthy because it is a ratio: both
levels went quiet together.

So k0/k1 answers "is dt small enough", and nothing in this project answered "is there
still a boundary layer". This does, and it runs wherever k0/k1 runs.

TWO CONDITIONS, because either alone is fooled:

  1. NON-TRIVIAL RESOLVED TURBULENCE, scaled against THE FORCING:

         max_k e_res(k) / U_ref^2,   U_ref = max_k |<u>,<v>|,  e_res = 0.5<u'^2+v'^2+w'^2>

     AND THE OBVIOUS METRIC IS THE WRONG ONE, WHICH IS THE WHOLE LESSON REPEATING ITSELF.
     The natural choice is e_res/u*^2 -- surface-layer units, near 5 in a neutral layer,
     regime- and wind-independent. Measured on the collapsed seed's own 37-dump series it
     reads 11.72 at the healthy peak and **4.71 after the collapse**, squarely inside the
     healthy band. It cannot see the death because u* dies WITH the turbulence and the
     ratio is preserved: precisely k0/k1's failure mode, one level up. A ratio of two
     quantities that collapse together is blind to the collapse.

     U_ref does not collapse. It is the geostrophic wind, which is prescribed and constant
     -- the collapsed layer's flow went EXACTLY geostrophic, which is the symptom, not a
     confound. On the same series the metric separates cleanly:

         healthy peak   e_max 5.97e-1 / U_ref^2   ->  9.3e-3
         collapsed      e_max 4.56e-2 / U_ref^2   ->  7.1e-4      a factor of 13

     e_res/u*^2 is still REPORTED, because it is the standing stationarity gate's own
     quantity and it is informative. It is just not what decides.

     NOT the sub-grid FRACTION either: PROJECT_BRIEF.md retired that gate because this grid cannot
     clear it at any affordable spacing, and a check nobody can pass is not a check.

  2. u* NOT COLLAPSING, scored against the run's OWN history rather than an absolute
     value, because u* legitimately moves 18% over hours on the 17.6 h inertial
     oscillation (PROJECT_BRIEF.md). Two ways to fail: the final u* falls below half its own
     running maximum, or it is trending down faster than 20 %/h over the scored half.

THE THRESHOLDS ARE MEASURED, NOT PICKED. Calibrated on this project's own runs -- the
neutral spin-up, the shallow CBL, and the collapsed stable seed, whose 37-dump series
walks from healthy to dead inside ONE run and therefore separates the two by itself:

    run                              state              e_max/U_ref^2   u* final/peak   u* trend
    g16_spin        (neutral)        healthy              4.6-5.6e-3          73%          +2.8 %/h
    g16_cbl_shallow (convective)     healthy              8.0-9.1e-3          66%         -16.0 %/h
    seed_sbl_a030   (stable, early)  healthy              5.9-9.3e-3           --             --
    seed_sbl_a030   (stable, final)  COLLAPSED            6.3-7.1e-4          25%         -75.5 %/h
    g16_flatsbl     (stable window)  COLLAPSED                7.1e-4          77%        -388.5 %/h

  E_FLOOR   = 2.0e-3   healthy >= 4.6e-3, dead <= 7.1e-4, NOTHING between them -- a factor
                       of 6.5 gap. 2.0e-3 is its geometric midpoint (1.81e-3), so the floor
                       sits 2.8x above the dead state and 2.3x below the weakest healthy one.
  UST_FRAC  = 0.50     healthy 66-77%, dead 25%.
  E_DECAY   = 0.25     final e_res / peak e_res: healthy 0.87 and 0.88, collapsed 0.076.
                       This one exists to close the SKIP hole: k0k1_check.py returns SKIP
                       below its floor and check_run.sh counts SKIP as a pass, so a layer
                       dead enough to fall under a floor escapes with no verdict at all.
                       A single dump cannot tell "not yet developed" from "already dead";
                       a series can, and this is how.
  UST_TREND = -35 %/h  healthy worst -16.0, dead -75.5 and -388.5; -35 is the geometric
                       midpoint of -16.0 and -75.5. A cold-start transient legitimately
                       falls at 16 %/h, so this must NOT be tightened toward it -- it would
                       start failing healthy spin-ups, and this is not the stationarity gate.

Note what g16_flatsbl is and is NOT: it is a 2400 s sampling window launched from the
collapsed seed's own restart, so its death is INHERITED, not independent evidence. It is
kept in the table for a different reason -- it shows the check firing on a WINDOW (lean
ioLPDMmode output, no zPos, no htFlux) as well as on spin-up dumps, and it shows u*
final/peak at 77% while the layer is plainly dead, which is why that test alone is not
enough and the e_max/U_ref^2 floor carries it.

The measured numbers are printed by --calibrate, so the margin can be re-checked whenever
the grid or the regime changes rather than trusted from a comment.

SKIP, not FAIL, before turbulence has developed. A cold start has e_res ~ 0 and u* rising
from zero; scoring that would fail every run's first dump. The rule is the same shape as
k0/k1's: below the laminar floor the metric carries no information, so it is skipped.

usage: turb_alive.py [--calibrate] [--json out.json] <dump.nc> [<dump.nc> ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from netCDF4 import Dataset

E_FLOOR = 2.0e-3         # max e_res/U_ref^2 below which the layer is not turbulent
UST_FRAC = 0.50          # final u* must exceed this fraction of the run's own maximum
UST_TREND = -35.0        # %/h over the scored half; see the calibration below
LAMINAR = 1.0e-4         # m2/s2; below this e_res carries no information -> SKIP
E_DECAY = 0.25           # final e_res as a fraction of the run's own peak
SCORE_FRAC = 0.5         # score u* over the last half of the series


def _prof(path):
    """(u*, resolved TKE profile, sub-grid TKE profile, n_bad, U_ref) from one dump.

    Perturbations are taken about the HORIZONTAL mean at each level, which is the same
    decomposition window_stats and the k0/k1 check use, so the three are comparable.
    """
    with Dataset(path) as ds:
        u = np.squeeze(np.asarray(ds["u"][:], dtype=np.float64))
        v = np.squeeze(np.asarray(ds["v"][:], dtype=np.float64))
        w = np.squeeze(np.asarray(ds["w"][:], dtype=np.float64))
        ust = float(np.asarray(ds["fricVel"][:], dtype=np.float64).mean())
        esgs = (np.squeeze(np.asarray(ds["TKE_0"][:], dtype=np.float64)).mean(axis=(-2, -1))
                if "TKE_0" in ds.variables else None)
    nbad = int((~np.isfinite(u)).sum() + (~np.isfinite(v)).sum() + (~np.isfinite(w)).sum())
    e = np.zeros(u.shape[0], dtype=np.float64)
    for f in (u, v, w):
        fp = f - f.mean(axis=(-2, -1), keepdims=True)
        e += (fp ** 2).mean(axis=(-2, -1))
    # U_ref: the fastest horizontal-mean wind in the column, i.e. the geostrophic wind
    # above the boundary layer. It is set by the forcing and does not die with the
    # turbulence, which is exactly why it and not u* is the denominator.
    uref = float(np.nanmax(np.hypot(u.mean(axis=(-2, -1)), v.mean(axis=(-2, -1)))))
    return ust, 0.5 * e, esgs, nbad, uref


def _time_h(path):
    """Simulated hours, from the dump's own time variable; None if it is absent."""
    try:
        with Dataset(path) as ds:
            if "time" in ds.variables:
                return float(np.asarray(ds["time"][:]).ravel()[0]) / 3600.0
    except (OSError, KeyError, IndexError):
        pass
    return None


def scan(paths):
    """Metrics for every dump, in the order given."""
    rows = []
    for p in paths:
        ust, e, esgs, nbad, uref = _prof(p)
        emax = float(np.nanmax(e)) if e.size else 0.0
        kmax = int(np.nanargmax(e)) if e.size else 0
        # THE metric: scaled against the forcing, which cannot collapse.
        ratio = emax / max(uref, 1e-6) ** 2
        # reported alongside, because it is the standing gate's quantity -- and because
        # seeing it stay healthy through a collapse is the point of keeping it visible.
        ratio_ust = emax / max(ust, 1e-6) ** 2
        f_res = None
        if esgs is not None and esgs.size == e.size:
            with np.errstate(divide="ignore", invalid="ignore"):
                f_res = float(e[kmax] / max(e[kmax] + esgs[kmax], 1e-30))
        rows.append(dict(path=p, t_h=_time_h(p), ustar=ust, e_max=emax, k_max=kmax,
                         u_ref=uref, e_over_uref2=ratio, e_over_ust2=ratio_ust,
                         f_res=f_res, n_bad=nbad, e_col=float(np.nansum(e))))
    return rows


def verdict(rows):
    """OK / FAIL / SKIP plus the message. Series tests only run with >= 3 dumps."""
    if not rows:
        return "SKIP", "  turb-alive SKIP: no dumps given"
    last = rows[-1]
    if last["n_bad"] or not np.isfinite(last["ustar"]) or not np.isfinite(last["e_max"]):
        return "FAIL", (f"  FAIL: NON-FINITE velocity in {last['path']}\n"
                        f"        {last['n_bad']:,} bad cells. inf is not NaN and FastEddy "
                        f"prints no CORRUPTED banner for it (FASTEDDY_TRAPS.md #1).")
    tag = (f"max e_res/U_ref^2={last['e_over_uref2']:.2e} at k={last['k_max']}  "
           f"u*={last['ustar']:.4f}  U_ref={last['u_ref']:.2f}  "
           f"(e_res/u*^2={last['e_over_ust2']:.2f}, reported only)")
    emax = np.array([r["e_max"] for r in rows], dtype=np.float64)
    # DECAY FRACTION, and it is what closes the hole SKIP would otherwise leave.
    # docker/k0k1_check.py returns SKIP below its variance floor, and check_run.sh treats
    # SKIP as a pass -- so a boundary layer dead enough to fall under the floor gets NO
    # VERDICT rather than a failure. The same shape of hole exists here, and a single dump
    # genuinely cannot tell "not yet developed" from "already dead". A SERIES can: e_res
    # rising is developing, e_res collapsed off its own peak is dying. Measured, final/peak
    # e_res: healthy 0.87 (neutral) and 0.88 (convective), collapsed 0.076.
    decayed = (len(rows) >= 3 and emax.max() > LAMINAR
               and emax[-1] < E_DECAY * emax.max())
    if last["e_max"] < LAMINAR and not decayed:
        return "SKIP", f"  turb-alive SKIP (undeveloped, e_res < {LAMINAR:g}): {tag}"
    msgs = []
    if decayed:
        msgs.append(f"resolved TKE has fallen to {100*emax[-1]/emax.max():.0f}% of the "
                    f"run's own peak, under {100*E_DECAY:.0f}% -- the turbulence is dying, "
                    f"not developing")
    if last["e_over_uref2"] < E_FLOOR:
        msgs.append(f"resolved TKE {last['e_over_uref2']:.2e} U_ref^2 is below the "
                    f"{E_FLOOR:.1e} floor -- the layer is not turbulent")
    ust = np.array([r["ustar"] for r in rows], dtype=np.float64)
    trend = frac = None
    if len(rows) >= 3:
        t = np.array([r["t_h"] if r["t_h"] is not None else i
                      for i, r in enumerate(rows)], dtype=np.float64)
        half = t >= (t[0] + SCORE_FRAC * (t[-1] - t[0]))
        if half.sum() >= 3 and np.ptp(t[half]) > 0:
            sl = float(np.polyfit(t[half], ust[half], 1)[0])
            trend = 100.0 * sl / max(abs(ust[half].mean()), 1e-30)
            if trend < UST_TREND:
                msgs.append(f"u* trending {trend:+.1f} %/h, past the {UST_TREND:.0f} %/h "
                            f"limit -- u* is collapsing")
        frac = float(ust[-1] / max(ust.max(), 1e-30))
        if frac < UST_FRAC:
            msgs.append(f"final u* is {100*frac:.0f}% of the run's own peak, under "
                        f"{100*UST_FRAC:.0f}% -- u* has collapsed")
    extra = ""
    if trend is not None:
        extra += f"  u* trend {trend:+.1f} %/h"
    if frac is not None:
        extra += f", final/peak {100*frac:.0f}%"
    if msgs:
        return "FAIL", ("  FAIL: TURBULENCE IS NOT ALIVE in " + last["path"] + "\n"
                        + "".join(f"        - {m}\n" for m in msgs)
                        + f"        {tag}{extra}\n"
                        + "        NOTE k0/k1 can still PASS here: it is a dt check, not a\n"
                        + "        physics check. It read 0.442 on a fully collapsed SBL.")
    return "OK", f"  turb-alive OK: {tag}{extra}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--calibrate", action="store_true",
                    help="print every dump's metrics instead of a verdict")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    paths = []
    for d in a.dumps:
        paths.extend(sorted(glob.glob(d), key=lambda q: (len(q), q)) if any(
            c in d for c in "*?[") else [d])
    paths = [p for p in paths if os.path.exists(p)]
    try:
        paths.sort(key=lambda q: int(q.split(".")[-1]))
    except (ValueError, IndexError):
        pass
    if not paths:
        print("  turb-alive SKIP: no dumps matched", file=sys.stderr)
        return 0

    rows = scan(paths)
    if a.calibrate:
        print(f"  {'dump':<32}{'t_h':>7}{'u*':>9}{'U_ref':>8}{'e_max':>11}"
              f"{'e/Uref^2':>11}{'e/u*^2':>9}{'k':>4}{'f_res':>8}")
        for r in rows:
            th = f"{r['t_h']:.2f}" if r["t_h"] is not None else "-"
            fr = f"{r['f_res']:.3f}" if r["f_res"] is not None else "-"
            print(f"  {os.path.basename(r['path']):<32}{th:>7}{r['ustar']:9.4f}"
                  f"{r['u_ref']:8.2f}{r['e_max']:11.3e}{r['e_over_uref2']:11.3e}"
                  f"{r['e_over_ust2']:9.2f}{r['k_max']:4d}{fr:>8}")
    status, msg = verdict(rows)
    print(msg)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(dict(status=status, rows=rows), fh, indent=1)
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
