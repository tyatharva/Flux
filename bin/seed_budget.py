#!/usr/bin/env python3
"""When WOULD this seed have been done? The library budget, measured retrospectively.

WHY THIS EXISTS RATHER THAN THE LIVE WATCHER ALONE. `jobs/seed_watch.sh` stops a run as
soon as the oscillation-immune limits are in band, and its scoring window has to be a
trailing FRACTION of the elapsed time (you cannot score two hours of a run that is one hour
old). With the trailing-half rule and a 3.0 simulated-hour ceiling the widest window it can
ever reach is 1.5 h -- and at 16 m the gate needed **2.0 h** before those trends resolved
against their own standard errors. So on a 3 h run the live criterion is structurally
unreachable, the watcher runs to the ceiling, and reporting "the measured stop time is 3.0
h" would be reporting the ceiling and calling it a measurement.

THIS DOES THE MEASUREMENT PROPERLY, and for free: after the run, score a FIXED-WIDTH
trailing window at a sequence of end times and report the earliest end time at which every
oscillation-immune limit is both RESOLVABLE and inside its threshold. The width is held
fixed across end times, because a verdict compared across different window widths is
comparing different estimators -- the same reason the gate's own window was swept rather
than inherited.

    end time      1.5 h   2.0 h   2.5 h   3.0 h
    window        [-2,-0] [-2,-0] ...              (width fixed, both ends slide)

Two things it can say, and they are different results:
  * an end time where the limits enter band -> THAT is the rung's budget, and a shorter
    ceiling would have served it.
  * none of them -> the rung needs more than the ceiling, and the honest report is the
    margin at the ceiling rather than a stop time.

usage: seed_budget.py jobs24/seed_nbl-deep_a000 [--width 2.0] [--step 0.25]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

IMMUNE = {"U/u* (Kljun Pi_4)", "sigma_v/u*", "sigma_w/u* at the receptor",
          "Kljun x_peak", "Kljun x90"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--width", type=float, default=2.0,
                    help="the FIXED scoring-window width, h. Held fixed across end times "
                         "on purpose: a verdict compared across widths compares estimators.")
    ap.add_argument("--step", type=float, default=0.25, help="end-time increment, h")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.job, "manifest.json")))
    dt = float(man["run"]["dt"]); base = man["run"]["outFileBase"]
    zm = float(man.get("gate", {}).get("zm", 10.0))
    k = int(man.get("gate", {}).get("k", 2))
    wth = float(man["target"]["wth_virtual"])
    paths = sorted(glob.glob(os.path.join(a.job, "output", base + ".[0-9]*")),
                   key=lambda p: int(p.rsplit(".", 1)[1]))
    if len(paths) < 6:
        print(f"FATAL: only {len(paths)} dumps in {a.job}/output", file=sys.stderr)
        return 2
    t_end_max = int(paths[-1].rsplit(".", 1)[1]) * dt / 3600.0
    print(f"=== {man['job']}: when would it have been done? ===")
    print(f"  {len(paths)} dumps, run reached {t_end_max:.2f} simulated hours; "
          f"fixed scoring window {a.width:.2f} h")
    print(f"  criterion: EVERY oscillation-immune limit resolvable AND inside its "
          f"threshold, and\n  nothing DRIFTING. TKE_BL/u*^2 and z_i are excluded -- they "
          f"decorrelate on the eddy\n  turnover and cannot be resolved at any width in a "
          f"3 h run.")
    print(f"\n  {'end (h)':>8} {'n dumps':>8}  {'immune ok':>10} {'indet':>6} "
          f"{'drift':>6}  verdict")

    rows, first_ok = [], None
    t = a.width
    while t <= t_end_max + 1e-9:
        sel = [p for p in paths if int(p.rsplit(".", 1)[1]) * dt / 3600.0 <= t + 1e-9]
        if len(sel) < 5:
            t += a.step; continue
        with tempfile.TemporaryDirectory() as td:
            for p in sel:
                os.symlink(os.path.abspath(p), os.path.join(td, os.path.basename(p)))
            js = os.path.join(td, "gate.json")
            subprocess.run([sys.executable,
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "seed_stationarity.py"), td,
                            "--dt", str(dt), "--wth", str(wth), "--zm", str(zm),
                            "--k", str(k), "--score-h", str(a.width), "--json", js],
                           capture_output=True)
            if not os.path.exists(js):
                t += a.step; continue
            d = json.load(open(js))
        gated = d.get("gated", [])
        imm = [r for r in gated if r["name"] in IMMUNE]
        n_ok = sum(1 for r in imm if r["ok"] is True)
        n_ind = sum(1 for r in imm if r["ok"] is None)
        drift = [r["name"] for r in gated if r["ok"] is False]
        ok = bool(imm) and n_ok == len(imm) and not drift
        if ok and first_ok is None:
            first_ok = t
        print(f"  {t:8.2f} {len(sel):8d}  {n_ok:>4d}/{len(imm):<5d} {n_ind:>6d} "
              f"{len(drift):>6d}  {'IN BAND' if ok else 'wait'}"
              + ("  [" + ", ".join(drift) + "]" if drift else ""))
        rows.append(dict(t_end_h=t, n_dumps=len(sel), immune_ok=n_ok,
                         immune_indeterminate=n_ind, drifting=drift, in_band=ok))
        t += a.step

    print()
    if first_ok is not None:
        print(f"  MEASURED BUDGET for this rung: {first_ok:.2f} simulated hours "
              f"(scored on a fixed {a.width:.2f} h window).")
        print(f"  A ceiling of {first_ok:.2f} h would have served it; the run went to "
              f"{t_end_max:.2f} h.")
    else:
        print(f"  NOT IN BAND ANYWHERE up to {t_end_max:.2f} h. The rung needs more than "
              f"the ceiling, and\n  the honest report is the margin at the ceiling rather "
              f"than a stop time. This is a\n  RESULT, not a failure of the measurement.")
    out = a.out or os.path.join(a.job, "return", "budget.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(dict(job=man["job"], width_h=a.width, t_end_max_h=t_end_max,
                   first_in_band_h=first_ok, rows=rows), open(out, "w"), indent=1)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
