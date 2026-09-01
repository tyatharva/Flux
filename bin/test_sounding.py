#!/usr/bin/env python3
"""Validate stages 1-2 offline, across regimes, before any GPU time is spent.

Four timestamps chosen to span what the corpus will actually contain -- summer convective
midday, a summer nocturnal stable layer, winter midday, and an autumn morning transition --
because a sounding reader that works on one convective July afternoon has been tested on
the easiest hour of the year. All four sit inside HRRR v4 (2020-12-02 onward), so the
corpus does not straddle a model-version change.

WHAT IS ASSERTED, and why each one is here rather than eyeballed:

  monotone z, physical theta        a silently reversed hybrid-level order would produce a
                                    plausible profile with the stratification upside down
  z_i diagnostics bracket HPBL      the theta-gradient pick used to land on the FREE
                                    TROPOSPHERE on any profile without a sharp capping
                                    inversion (2041 m against HPBL's 1648 m)
  meridian convergence ~ 5.1 deg    HRRR winds are GRID-relative; the rotation is invisible
                                    in the speed and is worth most of a direction bin
  the six stabilityScheme numbers   an out-of-range value does NOT abort FastEddy -- it
    inside FastEddy's own ranges    prints one line and silently uses the compiled-in
                                    default (parameters.c:309-315, FastEddy.c:96)
  base-state fit rms < 0.5 K        the base state is what the damping layer relaxes toward
  cadence lands on an integer       bin/run_window.sh asserts |frq*dt - 5| < 2e-4

ASSERT ON THE ARTIFACT, NOT THE EXIT STATUS (docs/FASTEDDY_TRAPS.md 12). Every check below reads
the JSON that was supposed to be written, not the return code of the thing that wrote it.

usage: test_sounding.py [--times ...] [--keep] [--outdir results/soundings]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_TIMES = [
    ("2023-07-15T19:00", "summer convective midday (14 CDT)"),
    ("2023-07-16T08:00", "summer nocturnal stable (03 CDT)"),
    ("2023-01-18T18:00", "winter midday (12 CST)"),
    ("2023-10-05T12:00", "autumn morning transition (07 CDT)"),
]

# hydro_core.c:640-650 -- zStableBottom* over [0, FLT_MAX], stableGradient* over
# [FLT_MIN, FLT_MAX], i.e. STRICTLY POSITIVE. surflayer_wth over [-5, 5] (:378),
# surflayer_z0 over [1e-12, 1] (:369), thetaAmplitude over [0, 2] (:668).
RANGES = {
    "zStableBottom": (0.0, 3.4e38), "zStableBottom2": (0.0, 3.4e38),
    "zStableBottom3": (0.0, 3.4e38),
    "stableGradient": (1.18e-38, 3.4e38), "stableGradient2": (1.18e-38, 3.4e38),
    "stableGradient3": (1.18e-38, 3.4e38),
    "surflayer_wth": (-5.0, 5.0), "surflayer_z0": (1e-12, 1.0),
    "temp_grnd": (1.18e-38, 3.4e38), "pres_grnd": (1.18e-38, 3.4e38),
}


class Checks:
    def __init__(self):
        self.rows = []

    def __call__(self, ok, name, detail=""):
        self.rows.append((bool(ok), name, detail))
        print(f"    {'ok  ' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
        return bool(ok)

    @property
    def passed(self):
        return all(r[0] for r in self.rows)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def check_sounding(path, c, tag):
    s = json.load(open(path))
    z = np.asarray(s["profile"]["z_agl_m"], float)
    th = np.asarray(s["profile"]["theta_k"], float)
    u = np.asarray(s["profile"]["u_ms"], float)
    v = np.asarray(s["profile"]["v_ms"], float)
    sf = s["surface"]

    c(np.all(np.diff(z) > 0), f"[{tag}] z strictly increasing",
      f"{z.size} levels, {z[0]:.1f} to {z[-1]:.0f} m")
    # Bound theta WHERE IT MATTERS. The profile runs to ~27 km, where theta really is
    # 700-730 K (p ~ 20 hPa, T ~ 230 K -> 230*(1000/20)^0.286 = 701) -- a blanket
    # "theta < 400 K" check calls a correct stratospheric profile broken. The LES column
    # is 0-2500 m and that is the only part FastEddy is ever given.
    col = th[z <= 2500.0]
    c(np.isfinite(th).all() and col.size >= 8
      and (col > 200).all() and (col < 350).all(),
      f"[{tag}] theta physical in the LES column",
      f"{col.min():.1f}-{col.max():.1f} K over {col.size} levels below 2500 m")
    c(np.isfinite(th).all() and (th > 180).all() and (th < 2000).all(),
      f"[{tag}] theta physical aloft",
      f"{th.min():.1f}-{th.max():.1f} K to {z[-1]/1000:.0f} km "
      f"(stratospheric theta is genuinely ~700 K)")
    c(np.isfinite(u).all() and np.isfinite(v).all()
      and np.hypot(u, v).max() < 120.0, f"[{tag}] winds finite and sane",
      f"max {np.hypot(u,v).max():.1f} m/s")
    # theta must increase with height somewhere aloft or the profile is upside down
    c(th[-1] > th[0], f"[{tag}] stratification right way up",
      f"theta {th[0]:.1f} -> {th[-1]:.1f} K")

    gam = s["geostrophic"]["meridian_convergence_deg"]
    c(3.0 < abs(gam) < 8.0, f"[{tag}] meridian convergence applied",
      f"{gam:+.2f} deg (Lambert n*(lon-lon0) = 5.11 here)")

    hp = sf.get("hpbl_m")
    ri, pa = sf.get("zi_bulk_ri_m"), sf.get("zi_parcel_m")
    if hp and np.isfinite(hp):
        got = [x for x in (ri, pa) if x is not None and np.isfinite(x)]
        c(bool(got), f"[{tag}] a profile z_i diagnostic exists")
        if got:
            # HPBL is a TKE-threshold depth and normally sits above a theta-based one;
            # a factor of 3 either way means one of them is measuring something else.
            r = max(got) / hp
            c(0.33 < r < 3.0, f"[{tag}] z_i diagnostics agree to a factor of 3",
              f"HPBL {hp:.0f} m, bulk-Ri {ri:.0f} m, parcel {pa:.0f} m")

    b = sf.get("bowen")
    if b is not None and np.isfinite(b) and b > 0:
        c(abs(sf["wth_virtual"]) >= abs(sf["wth_sensible"]) - 1e-12,
          f"[{tag}] virtual flux >= sensible in magnitude",
          f"B={b:.2f}, {sf['wth_sensible']:+.4f} -> {sf['wth_virtual']:+.4f} K m/s")
    return s


def check_forcing(path, snd, c, tag):
    f = json.load(open(path))
    p, fit, g = f["params"], f["fit"], f["grid"]

    bad = [k for k, (lo, hi) in RANGES.items()
           if k in p and not (lo <= float(p[k]) <= hi)]
    c(not bad, f"[{tag}] every parameter inside FastEddy's own range",
      f"checked {len(RANGES)} against hydro_core.c" if not bad else f"out of range: {bad}")
    c(p["zStableBottom"] <= p["zStableBottom2"] <= p["zStableBottom3"],
      f"[{tag}] stable-layer bases ordered",
      f"{p['zStableBottom']:.0f} <= {p['zStableBottom2']:.0f} <= {p['zStableBottom3']:.0f}")
    c(fit["rms_k"] < 0.5, f"[{tag}] base-state fit within 0.5 K rms",
      f"rms {fit['rms_k']:.3f} K, max |resid| {fit['max_abs_k']:.3f} K "
      f"over {fit['n_levels']} LES levels")

    n = g["frqOutput_per_cadence"]
    c(abs(n * p["dt"] - g["cadence_s"]) < 2e-4,
      f"[{tag}] the 5 s cadence is an integer step count",
      f"{n} x {p['dt']:.7f} = {n*p['dt']:.6f} s")
    c(g["CFL_3d"] <= 1.35 + 1e-9, f"[{tag}] CFL_3d at or under the production 1.35",
      f"{g['CFL_3d']:.4f} (accuracy boundary 1.51, measured)")

    # the fit must reproduce the sounding, not merely be self-consistent
    z = np.asarray(snd["profile"]["z_agl_m"], float)
    th = np.asarray(snd["profile"]["theta_k"], float)
    zz = np.linspace(0.0, 1500.0, 200)
    ramp = lambda lo, hi: np.clip(zz, lo, hi) - lo
    # FastEddy's own constants (hydro_core.c:1574-1580), because this check exists to
    # reproduce what FastEddy will build, not what a textbook would.
    fe_kappa = 287.04 / (287.04 + 718.0)
    model = (p["temp_grnd"] * (1.0e5 / p["pres_grnd"]) ** fe_kappa
             + p["stableGradient"] * ramp(p["zStableBottom"], p["zStableBottom2"])
             + p["stableGradient2"] * ramp(p["zStableBottom2"], p["zStableBottom3"])
             + p["stableGradient3"] * np.maximum(zz - p["zStableBottom3"], 0.0))
    resid = model - np.interp(zz, z, th)
    c(np.abs(resid).max() < 1.5,
      f"[{tag}] the base state reproduces the sounding below 1.5 km",
      f"max |resid| {np.abs(resid).max():.3f} K, rms {np.sqrt((resid**2).mean()):.3f} K")

    L = f["labels"].get("L_estimate")
    c(f["labels"]["zi_m"] > 0, f"[{tag}] z_i positive", f"{f['labels']['zi_m']:.0f} m")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", nargs="*", default=None)
    ap.add_argument("--outdir", default="results/soundings")
    ap.add_argument("--forcedir", default="results/forcing")
    ap.add_argument("--grid", default="data/grid16")
    a = ap.parse_args()

    times = ([(t, "") for t in a.times] if a.times else DEFAULT_TIMES)
    c = Checks()
    summary = []
    for ts, why in times:
        tag = ts.replace("-", "").replace(":", "").replace("T", "")[:10]
        print(f"\n=== {ts}  {why}")
        sp = os.path.join(a.outdir, f"case_{tag}.json")
        if not os.path.exists(sp):
            rc, out = run([os.path.join(HERE, "hrrr_sounding.py"), ts, "--out", sp])
            if not os.path.exists(sp):
                c(False, f"[{tag}] sounding written", out.strip().splitlines()[-1:] and
                  out.strip().splitlines()[-1] or "no output")
                continue
        c(True, f"[{tag}] sounding written", os.path.basename(sp))
        snd = check_sounding(sp, c, tag)

        fp = os.path.join(a.forcedir, f"case_{tag}.json")
        rc, out = run([os.path.join(HERE, "sounding_to_forcing.py"), sp,
                       "--out", fp, "--grid", a.grid])
        if not os.path.exists(fp):
            c(False, f"[{tag}] forcing written", out.strip().splitlines()[-1] if out else "")
            continue
        c(True, f"[{tag}] forcing written", os.path.basename(fp))
        f = check_forcing(fp, snd, c, tag)
        summary.append((ts, snd, f))

    print(f"\n=== the four cases side by side ===")
    print(f"  {'valid time':<18}{'z_i':>7}{'w th_v':>9}{'B':>7}{'G':>7}{'dir':>7}"
          f"{'fit rms':>9}{'repr':>6}")
    for ts, snd, f in summary:
        l = f["labels"]
        print(f"  {ts:<18}{l['zi_m']:>7.0f}{l['wth_virtual']:>9.4f}"
              f"{(l['bowen'] if l['bowen'] is not None else float('nan')):>7.2f}"
              f"{l['G_speed']:>7.2f}{l['G_dir_from_deg']:>7.0f}"
              f"{f['fit']['rms_k']:>9.3f}{str(f.get('representable')):>6}")

    n_ok = sum(1 for r in c.rows if r[0])
    print(f"\n  {n_ok}/{len(c.rows)} checks passed")
    print(f"  STAGES 1-2: {'PASS' if c.passed else 'FAIL'}")
    return 0 if c.passed else 1


if __name__ == "__main__":
    sys.exit(main())
