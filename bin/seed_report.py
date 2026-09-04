#!/usr/bin/env python3
"""What one seed job actually cost and achieved. The numbers the other 14 get rented on.

A seed job is projected before it runs -- 0.0149 s/step, ~1.02 GPU-h per simulated hour,
73.3 MB home, all seven stationarity limits passing by 3.0 h. This measures each of those
against the run, because the remaining seventeen are budgeted on them and a 10% error in
the wall-to-sim ratio is a 5-hour error across the library.

Reported and not assumed:

  wall-to-sim        measured s/step against the 0.0149 the planner used
  the seven limits   which PASS, and WHICH BINDS LAST -- the one closest to its threshold
                     is the one that sets the 3.0 h budget, and it is worth knowing which
  laminarisation     a stable LES can stop being turbulent. TKE trend over the run, and
                     the resolved fraction at the receptor. This is the failure mode that
                     looks like a converged run.
  achieved direction against the base angle after Ekman backing -- the forcing angle is
                     not the achieved one, and the library is labelled by the achieved
  k0/k1              the accuracy-CFL check; ~9 means dt past the boundary
  artifact size      against 73.3 MB

usage: seed_report.py seeds/seed_sbl_a030 [--wall-seconds N] [--out FILE]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--wall-seconds", type=float, default=None)
    ap.add_argument("--sps-planned", type=float, default=0.0149)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ret = os.path.join(a.job, "return")
    # THE RETURN MANIFEST IS THE ONE WITH THE ANSWER. bin/run_seed.sh stamps `achieved`
    # into return/manifest.json, not into the job's own; reading the job manifest reported
    # "the manifest carries no `achieved` block" on a run that had measured every one of
    # those numbers and written them down. Prefer the return copy, fall back to the job's.
    _rm = os.path.join(ret, "manifest.json")
    man = json.load(open(_rm if os.path.exists(_rm)
                         else os.path.join(a.job, "manifest.json")))
    st_path = os.path.join(ret, "stationarity.json")
    if not os.path.exists(st_path):
        print(f"FATAL: {st_path} does not exist", file=sys.stderr)
        return 2
    st = json.load(open(st_path))
    run = man["run"]

    o = []
    p = o.append
    p(f"=== seed {man['job']}: {man['rung']} ({man['regime']}), base angle "
      f"{man['base_angle_deg']:.0f} deg ===")

    # ---- cost --------------------------------------------------------------------
    steps = run["steps_total"]
    sim_h = steps * run["dt"] / 3600.0
    p(f"\n--- cost, against the projection the other 14 are budgeted on ---")
    p(f"  {steps} steps x dt {run['dt']} = {sim_h:.3f} simulated hours")
    # MEASURE THE RATE OFF THE ARTIFACTS, NOT OFF A NUMBER PASSED IN BY HAND.
    #
    # This block used to depend entirely on --wall-seconds, and on a RESUMED job that is
    # wrong: the elapsed time between the log's first and last timestamps covers only the
    # segments this invocation ran, while `steps` is the whole chain. It reported
    # 10.115 ms/step against a planned 14.9, i.e. "32% faster than projection", and the
    # library was costed at 37.4 GPU-h instead of 52 on the strength of it. The dumps
    # themselves say otherwise, and they cannot be mis-passed.
    #
    # The MEDIAN inter-dump interval is what is used, not the total span, so a pause or a
    # resume gap cannot contaminate it -- and any interval more than 3x the median is
    # reported as exactly that, a pause, rather than silently averaged in.
    dumpz = sorted(glob.glob(os.path.join(a.job, "output", "*.[0-9]*")),
                   key=lambda q: int(q.rsplit(".", 1)[1]))
    sps_meas, note = None, ""
    if len(dumpz) >= 3:
        stp = np.array([int(q.rsplit(".", 1)[1]) for q in dumpz], dtype=np.float64)
        mt = np.array([os.path.getmtime(q) for q in dumpz], dtype=np.float64)
        dstep, dt_s = np.diff(stp), np.diff(mt)
        good = dstep > 0
        per = dt_s[good] / dstep[good]
        med = float(np.median(per))
        npause = int((per > 3.0 * med).sum())
        sps_meas = med
        note = (f"  from {len(dumpz)} dump mtimes; median inter-dump rate, "
                f"{npause} pause(s) excluded" if npause else
                f"  from {len(dumpz)} dump mtimes, no pauses")
    if a.wall_seconds:
        sps = a.wall_seconds / steps
        p(f"  --wall-seconds {a.wall_seconds/3600:.3f} h  ->  {sps*1000:.3f} ms/step "
          f"(HAND-PASSED; trust the artifact line below instead)")
    if sps_meas is not None:
        p(f"  MEASURED off the dumps: {sps_meas*1000:.3f} ms/step")
        p(f"{note}")
        p(f"  planned                 {a.sps_planned*1000:.3f} ms/step"
          f"   ({100*(sps_meas/a.sps_planned-1):+.1f}%)")
        wall_h = sps_meas * steps / 3600.0
        p(f"  implied wall for the whole chain {wall_h:.3f} h")
        # TWO REFERENCES, because they are different numbers and quoting one hides the
        # other. The manifest's own projection is a CONSERVATIVE 0.0149 s/step (ratio
        # 1.019); the SANCTIONED RUN CLASS is 3.0 sim-h in ~2.86 h wall, ratio 0.953. A
        # run is in class when it lands at or under the class figure with margin, and the
        # planner is simply pessimistic by design.
        p(f"  wall-to-sim ratio {wall_h/sim_h:.3f} GPU-h per simulated hour")
        p(f"    vs the manifest projection 1.019  ({100*(wall_h/sim_h/1.019-1):+.1f}%)")
        p(f"    vs the sanctioned SEED CLASS 0.953 ({100*(wall_h/sim_h/0.953-1):+.1f}%)"
          f"  -> {'IN CLASS' if wall_h <= 3.15 else 'OUT OF CLASS'}"
          f"  ({wall_h:.2f} h against a 3.0 sim-h class of ~2.86 h)")
        p(f"  ONE invocation, {sps_meas*steps/60:.1f} min wall "
          f"(planner projected {run.get('projected_wall_min', 0):.1f}; chaining is retired, "
          f"there is no per-run cap)")
        p(f"  the library at this rate: 15 x {wall_h:.2f} h = "
          f"{15*wall_h:.1f} GPU-h  (projected 43)")
    elif not a.wall_seconds:
        p("  (fewer than 3 dumps on disk; cost not measurable)")

    # ---- the seven limits --------------------------------------------------------
    p(f"\n--- the seven stationarity limits, scored on the last {st['score_h']} h ---")
    p(f"  {'quantity':<30}{'mean':>10}{'trend':>11}{'limit':>8}{'used':>8}  verdict")
    rows = sorted(st["gated"], key=lambda r: -abs(r["trend_pct_per_h"]) / r["limit"])
    for r in rows:
        frac = abs(r["trend_pct_per_h"]) / r["limit"]
        p(f"  {r['name']:<30}{r['mean']:10.4f}{r['trend_pct_per_h']:+10.2f}%"
          f"{r['limit']:8.1f}{100*frac:7.0f}%  {r.get('verdict', 'ok' if r['ok'] else 'DRIFTING')}")
    # INDETERMINATE IS NOT DRIFTING. r['ok'] is None for a limit the gate cannot resolve,
    # and `'ok' if r['ok'] else 'DRIFTING'` printed None as DRIFTING -- asserting a limit
    # was moving when the gate's whole point was that it could not say.
    _ind = [r["name"] for r in rows if r.get("ok") is None]
    _dft = [r["name"] for r in rows if r.get("ok") is False]
    p(f"  ALL SEVEN: {'PASS' if st['pass'] else 'FAIL'}"
      + (f"  -- DRIFTING: {', '.join(_dft)}" if _dft else "")
      + (f"  -- INDETERMINATE: {', '.join(_ind)}" if _ind else ""))
    tight = rows[0]
    p(f"  BINDS LAST: {tight['name']} at {100*abs(tight['trend_pct_per_h'])/tight['limit']:.0f}% "
      f"of its limit ({tight['trend_pct_per_h']:+.2f} %/h against {tight['limit']:.1f})")
    # THE REPORTED ROWS ARE NOT ALL IN %/h. Wind direction is a BEARING and is carried
    # in deg/h -- a percentage of a bearing is unreadable and wraps through north -- so
    # each row names its own unit and this formats what it finds. Assuming one unit here
    # is what crashed this report the first time the direction row changed.
    p(f"  reported, not gated: " + ", ".join(
        f"{r['name']} "
        f"{r.get('trend_pct_per_h', r.get('trend_deg_per_h', float('nan'))):+.2f} "
        f"{r.get('unit', '%/h')}" for r in st["reported"]))

    # ---- laminarisation ----------------------------------------------------------
    dumps = sorted(glob.glob(os.path.join(a.job, "output", "*.[0-9]*")),
                   key=lambda q: int(q.rsplit(".", 1)[1]))
    p(f"\n--- is the stable boundary layer still turbulent? ---")
    if len(dumps) < 3:
        p(f"  only {len(dumps)} dumps on disk; cannot judge")
    else:
        from netCDF4 import Dataset
        ts, tkes, usts, ww0 = [], [], [], []
        for q in dumps:
            with Dataset(q) as ds:
                g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
                u, v, w = g("u"), g("v"), g("w")
                z = g("zPos")[:, 0, 0]
                e = np.maximum(g("TKE_0"), 0.0)
                usts.append(float(g("fricVel").mean()))
            pr = lambda arr: arr - arr.mean(axis=(-2, -1), keepdims=True)
            tk = 0.5 * ((pr(u) ** 2 + pr(v) ** 2 + pr(w) ** 2).mean(axis=(-2, -1)))
            tkes.append(float(np.trapezoid(tk, z)))
            ww = (pr(w) ** 2).mean(axis=(-2, -1))
            ww0.append(float(ww[0] / max(ww[1], 1e-30)))
            ts.append(int(q.rsplit(".", 1)[1]) * run["dt"] / 3600.0)
        ts, tkes, usts = np.array(ts), np.array(tkes), np.array(usts)
        half = ts >= ts[-1] / 2
        sl = np.polyfit(ts[half], tkes[half], 1)[0]
        p(f"  column-integrated TKE {tkes[0]:.3f} -> {tkes[-1]:.3f} m3/s2;"
          f" trend over the last half {100*sl/max(abs(tkes[half].mean()),1e-30):+.1f} %/h")
        p(f"  u* {usts[0]:.4f} -> {usts[-1]:.4f} m/s   (min over the run {usts.min():.4f})")
        # u* IS THE TEST, NOT COLUMN TKE. The first version keyed on the TKE integral and
        # said "no" on a run whose u* had fallen 58% and whose z/L had reached 2.67 --
        # because that integral is dominated by gravity-wave variance aloft, which GROWS
        # as the turbulence dies (docs/FASTEDDY_TRAPS.md 15). It is kept as a REPORTED line for
        # exactly that reason: seeing it rise while u* falls is the diagnosis.
        #
        # THE VERDICT ITSELF IS docker/turb_alive.py's, imported rather than reimplemented.
        # This function used to carry its own copy of the collapse test, which is the same
        # shape of mistake as stage4_wellmixed.py carrying its own copy of the sigma_w
        # floor -- the gate drifts from the thing it is supposed to be scoring. PROJECT_BRIEF.md
        # is explicit: gates import the production function.
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "docker"))
        import turb_alive                                          # noqa: E402
        status, msg = turb_alive.verdict(turb_alive.scan(dumps))
        for ln in msg.rstrip("\n").split("\n"):
            p("  " + ln.strip())
        p(f"  TURBULENCE ALIVE: {status}"
          + ("" if status == "OK" else "  -- this state is not a usable seed"))
        p(f"  k0/k1 at the last dump {ww0[-1]:.3f}   (must be < 1; ~9 means dt too large)")
        p(f"    ^ note this passed at 0.442 on a FULLY COLLAPSED SBL. k0/k1 is a dt check;")
        p(f"      the line above it is the physics check. Read both.")

    # ---- achieved vs asked -------------------------------------------------------
    ach = man.get("achieved", {})
    tgt = man["target"]
    p(f"\n--- achieved vs asked ---")
    if ach:
        gd = tgt["G_dir_from_deg"]
        wd = ach.get("wdir")
        p(f"  geostrophic forcing FROM {gd:.1f} deg (base angle {man['base_angle_deg']:.0f})")
        if wd is not None:
            back = ((gd - wd + 180.0) % 360.0) - 180.0
            p(f"  achieved receptor wind FROM {wd:.1f} deg  ->  Ekman backing "
              f"{back:+.1f} deg  (measured 12-24 deg neutrally, 7-13 convectively)")
        p(f"  z_i {ach.get('zi', float('nan')):.0f} m against a {tgt['zi_m']:.0f} m target")
        p(f"  u* {ach.get('ustar', float('nan')):.4f},  U {ach.get('U', float('nan')):.3f} m/s,"
          f"  sigma_w {ach.get('sigma_w', float('nan')):.4f}")
    else:
        p("  the manifest carries no `achieved` block")

    # ---- what comes home ---------------------------------------------------------
    p(f"\n--- what comes home ---")
    tot = 0
    for f in sorted(glob.glob(os.path.join(ret, "*"))):
        sz = os.path.getsize(f)
        tot += sz
        p(f"  {os.path.basename(f):<26}{sz/1e6:9.2f} MB")
    rst = os.path.join(ret, "seed_restart.nc")
    if os.path.exists(rst):
        sz = os.path.getsize(rst) / 1e6
        p(f"  restart {sz:.1f} MB against the 73.3 MB estimate ({100*(sz/73.3-1):+.1f}%)")
    p(f"  return/ total {tot/1e6:.1f} MB")

    txt = "\n".join(o)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
