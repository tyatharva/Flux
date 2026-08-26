#!/usr/bin/env python3
"""Is this spun-up state stationary ENOUGH to be a seed? The portable form of Gate C1.

THE GATE IS ON U/u*, NOT ON u*. A doubly-periodic neutral Ekman layer forced by a constant
geostrophic wind does not settle to a fixed u* on any affordable timescale: f = 9.94e-5
here, so the inertial period is 17.6 h and u* falls for a quarter of it and then rises.
Measured on g16_spin, u* moved -27% over 6.26 simulated hours while U/u* was within 0.31%
of its final value by 3.01 h. Gating on u* alone failed this project's spin-ups TWICE for
a reason that was not a modelling error, and PROJECT_BRIEF.md now records why.

Kljun's Pi_4 = U(z_m)/u* is the only channel through which the wind enters the streamwise
footprint shape, and both of its terms ride the oscillation together -- so the RATIO is
stationary while its numerator and denominator each move at +6.3 %/h.

AND z_i IS THE ONE GATED QUANTITY THAT IS NOT A RATIO, which is why it needed a fix the
others did not. Diagnosed as a fraction of its own TKE peak it inherited the oscillation
through the THRESHOLD instead of through the value, and failed the first full-length seed
at +11.67 %/h while three independent depths put the layer at +1.71 to +2.33. It is now a
FIXED threshold (ZI_ABS). See the block above LIMITS for the measurements. The seven limits
below score the footprint's controlling parameters, and they are far tighter in footprint
terms than the u* test they replace.

WHY THIS FILE EXISTS RATHER THAN A COPY. bin/run_pass5.sh scored the fifth pass's neutral
spin-up from an inline heredoc with L hardwired to infinity. Seeds span stable, neutral
and convective, so the Kljun terms need a real L -- and this project has already shipped
one wrong result from a gate that carried its own drifted COPY of a production function
(stage4_wellmixed.py's sigma_w floor). So the limits and the scoring live HERE, once, and
run_pass5.sh imports them.

The gate runs INSIDE the seed job. The 300 s stationarity dumps are ~73 MB each and there
are ~36 of them; scoring them where they are written means the verdict travels back as a
few kB of JSON and the dumps never leave the rented machine.

usage: seed_stationarity.py <dump-glob-or-dir> --dt DT [--wth W] [--score-h 1.5]
                            [--json FILE] [--zm 10.0] [--k 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

G = 9.81
VONK = 0.4

# THE GATED z_i IS A FIXED TKE THRESHOLD, NOT A FRACTION OF THE RUNNING PEAK -- changed
# 2026-08-26 after the first full-length seed failed on this quantity alone.
#
# The old definition was "the height where resolved TKE falls below 5% of its own peak".
# That threshold MOVES WITH THE PEAK, and in a neutral Ekman layer the peak is not steady:
# u* falls through the first quarter of the 17.6 h inertial period, TKE ~ u*^2 goes with
# it, and a falling threshold pushes the crossing height up while the layer holds still.
# Measured on seed_nbl-shallow_a000 over its scored window (SEED_NBL_SHALLOW_RESULT.md):
#
#   z_i, 5% of the running peak   364.4 m   +11.67 %/h   <- FAILED a 3 %/h limit
#   z_i, fixed 0.01 m2/s2         389.3 m    +1.87 %/h
#   z_i, 5% of the settled peak   370.8 m    +1.71 %/h
#   z_i, theta-gradient           336.5 m    +2.33 %/h
#   peak resolved TKE             0.3308     -15.67 %/h  <- the normaliser
#
# The gated depth correlated -0.885 with the peak it was normalised by; the fixed-threshold
# depth, -0.379. And Kljun's x_peak and x90 CONSUME z_i and are gated ten times tighter --
# they read -0.21 and -0.17 %/h, which a layer moving at 11.67 %/h cannot produce.
#
# THIS IS THE ONE EXCEPTION TO THE RATIO RULE, AND THAT IS WHY IT NEEDED FIXING. Every
# other limit here is a RATIO whose numerator and denominator ride the inertial oscillation
# together, which is the whole design of this gate. z_i is not a ratio -- so with a
# peak-normalised threshold it inherited the oscillation anyway, through the threshold
# rather than through the value. A fixed threshold is what makes it behave like the others.
#
# Checked across regimes before adopting, on this project's own runs (the value must not
# run off the top of the column, and it does not):
#
#   rung          peak TKE   0.01 as % of peak   z_i 5%-peak   z_i fixed   domain top
#   nbl-shallow      0.331         3.02%             364 m       389 m       2500 m
#   neutral spin     0.487         2.05%             414 m       455 m       2500 m
#   cbl-shallow      1.084         0.92%             508 m       598 m       2500 m
#   cbl-deep         1.430         0.70%             976 m      1186 m       2500 m
#
# It runs 7-21% deeper than the peak fraction and the offset grows with regime intensity.
# That is a change of DEFINITION, not an error, and it is why the peak-fraction depth is
# still computed and still reported -- lpdm/les_stats.py:window_stats produces the corpus
# input `h` on the peak fraction, and bin/pick_seed.py matches seeds against cases in that
# same currency. The gate measures a TREND and needs a threshold that does not move; the
# matcher compares a VALUE and needs the definition the corpus inputs use.
ZI_ABS = 0.01            # m2/s2 of resolved TKE; the gated depth's fixed threshold


# THE SEARCH IS BOUNDED BY THE DECAY MINIMUM, and that lives in lpdm/les_stats.py because
# the corpus input `h` is computed there and the two must not drift apart. Imported, never
# reimplemented -- the rule this project already paid for twice (stage4_wellmixed.py's copy
# of the sigma_w floor; bin/run_pass5.sh's copy of this very estimator).
from lpdm.les_stats import bl_depth                                        # noqa: E402


def zi_fixed(tk, z, thresh=ZI_ABS):
    """Depth from a FIXED resolved-TKE threshold. The gated definition."""
    return bl_depth(tk, z, thresh=thresh)


def zi_peak_fraction(tk, z, frac=0.05):
    """Depth from a fraction of the profile's OWN peak. Reported, never gated.

    This is what lpdm/les_stats.py:window_stats produces as the corpus input `h`, so the
    library's depth axis stays commensurable with it. FASTEDDY_TRAPS.md 16 is the whole
    story of why it must not be trended.
    """
    return bl_depth(tk, z, frac=frac)


# Percent-per-hour trend limits, scored over the last SCORE_H hours. Single definition;
# bin/run_pass5.sh imports this dict rather than restating it.
LIMITS = {
    "U/u* (Kljun Pi_4)": 1.0,
    "sigma_v/u*": 3.0,
    "sigma_w/u* at the receptor": 2.0,
    "TKE/u*^2": 5.0,
    "z_i": 3.0,
    "Kljun x_peak": 1.0,
    "Kljun x90": 1.0,
}


def series(paths, dt, k):
    """Per-dump receptor-level moments and the derived Kljun geometry inputs."""
    from netCDF4 import Dataset
    out = {n: [] for n in ("t", "ustar", "tke", "zi", "zi_peakfrac", "sw", "sv", "U",
                          "wdir", "th0", "tkepeak")}
    for p in paths:
        with Dataset(p) as ds:
            g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
            u, v, w = g("u"), g("v"), g("w")
            z = g("zPos")[:, 0, 0]
            e = np.maximum(g("TKE_0"), 0.0)
            out["ustar"].append(float(g("fricVel").mean()))
            out["th0"].append(float(g("theta")[0].mean()))
        # inf is not NaN and NaN passes every > comparison (PLAN.md working agreement),
        # so the finiteness test comes FIRST and is on every field that feeds a moment.
        for nm, a in (("u", u), ("v", v), ("w", w), ("TKE_0", e)):
            if not np.isfinite(a).all():
                raise ValueError(f"{os.path.basename(p)}: {nm} is not finite")
        pr = lambda a: a - a.mean(axis=(-2, -1), keepdims=True)
        tk = 0.5 * ((pr(u) ** 2 + pr(v) ** 2 + pr(w) ** 2).mean(axis=(-2, -1)))
        out["tke"].append(float(tk.mean()))
        out["tkepeak"].append(float(tk[int(np.argmax(tk))]))
        out["zi"].append(zi_fixed(tk, z))                 # GATED: fixed threshold
        out["zi_peakfrac"].append(zi_peak_fraction(tk, z))  # reported; the corpus currency
        out["sw"].append(float(np.sqrt((pr(w)[k] ** 2).mean() + (2 / 3) * e[k].mean())))
        out["sv"].append(float(np.sqrt(((pr(u)[k] ** 2 + pr(v)[k] ** 2).mean()) / 2
                                       + (2 / 3) * e[k].mean())))
        out["U"].append(float(np.hypot(u[k].mean(), v[k].mean())))
        out["wdir"].append(float((270 - np.degrees(np.arctan2(v[k].mean(),
                                                              u[k].mean()))) % 360))
        out["t"].append(int(p.rsplit(".", 1)[1]) * dt / 3600.0)
    return {n: np.asarray(a, float) for n, a in out.items()}


def kljun_geometry(s, zm, wth):
    """x_peak and x90 per dump, with a REAL Obukhov length.

    run_pass5.sh could pass L = inf because it scored a neutral spin-up. A convective seed
    at w'th_v' = 0.16 K m/s and u* = 0.3 has L = -14 m, i.e. z_m/L = -0.7 -- treating that
    as neutral would score the wrong footprint entirely and call the seed stationary on a
    geometry it does not have.
    """
    from lpdm import kljun
    n = s["t"].size
    xp = np.empty(n)
    x90 = np.empty(n)
    xx = np.linspace(0.5, 3000.0, 4000)
    for i in range(n):
        if abs(wth) > 1e-6:
            L = -s["ustar"][i] ** 3 * s["th0"][i] / (VONK * G * wth)
        else:
            L = np.inf
        xp[i] = kljun.peak_distance(zm, s["zi"][i], s["ustar"][i],
                                    umean=s["U"][i], L=L)
        fy, _ = kljun.crosswind_integrated(xx, zm, s["zi"][i], s["ustar"][i],
                                           umean=s["U"][i], L=L)
        c = np.cumsum(fy)
        c /= c[-1]
        x90[i] = float(np.interp(0.90, c, xx))
    return xp, x90


def score(s, xp, x90, score_h):
    t = s["t"]
    sel = t >= t[-1] - score_h
    if sel.sum() < 4:
        raise ValueError(f"only {int(sel.sum())} dumps in the last {score_h} h; "
                         "the trend would have no degrees of freedom")

    def trend(y):
        A = np.vstack([t[sel], np.ones(int(sel.sum()))]).T
        slope = np.linalg.lstsq(A, y[sel], rcond=None)[0][0]
        return 100.0 * slope / max(abs(y[sel].mean()), 1e-30)

    us = s["ustar"]
    quantities = (("U/u* (Kljun Pi_4)", s["U"] / us),
                  ("sigma_v/u*", s["sv"] / us),
                  ("sigma_w/u* at the receptor", s["sw"] / us),
                  ("TKE/u*^2", s["tke"] / us ** 2),
                  ("z_i", s["zi"]),
                  ("Kljun x_peak", xp),
                  ("Kljun x90", x90))
    rows, ok = [], True
    for nm, y in quantities:
        v = trend(y)
        g_ = bool(abs(v) < LIMITS[nm])
        ok &= g_
        r = {"name": nm, "mean": float(y[sel].mean()),
             "trend_pct_per_h": float(v), "limit": LIMITS[nm], "ok": g_}
        # A LINEAR TREND THROUGH A STAIRCASE REPORTS THE STAIRCASE. z_i can only land on a
        # model level, so over a short window it takes a handful of discrete values and a
        # least-squares slope through them is as much an artifact of WHICH levels were
        # visited as of any drift. The count is reported with the trend so the number can
        # be read correctly -- two or three levels is a staircase, not a rate.
        if nm == "z_i":
            r["n_levels"] = int(np.unique(np.round(y[sel], 6)).size)
            r["level_span_m"] = float(y[sel].max() - y[sel].min())
        rows.append(r)
    reported = [{"name": nm, "mean": float(y[sel].mean()),
                 "trend_pct_per_h": float(trend(y))}
                for nm, y in (("u*", us), ("U(10 m)", s["U"]),
                              ("wind direction", s["wdir"]), ("domain TKE", s["tke"]),
                              ("peak resolved TKE", s["tkepeak"]),
                              ("z_i, 5% of the running peak", s["zi_peakfrac"]))]
    return bool(ok), rows, reported, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="a directory, or a glob of dumps")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--wth", type=float, default=0.0,
                    help="the PRESCRIBED surface virtual heat flux, for L in the Kljun "
                         "terms. The resolved covariance at k=0 is not it (cbl_check.py).")
    ap.add_argument("--score-h", type=float, default=1.5)
    ap.add_argument("--zm", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    pat = a.target
    if os.path.isdir(pat):
        pat = os.path.join(pat, "*.[0-9]*")
    paths = sorted((p for p in glob.glob(pat) if p.rsplit(".", 1)[-1].isdigit()),
                   key=lambda p: int(p.rsplit(".", 1)[1]))
    # ONE RUN PER DIRECTORY, OR THIS IS NOT A SERIES. FastEddy names a dump
    # <outFileBase>.<step>, so a directory that has held two runs holds two families with
    # OVERLAPPING step numbers -- and sorting on the step alone interleaves them into a
    # series that has two different states at the same time and announces nothing. It is
    # the same shape as every other failure in this project that produced a plausible wrong
    # number rather than an error, so it is refused here rather than diagnosed later.
    fams = sorted({os.path.basename(p).rsplit(".", 1)[0] for p in paths})
    if len(fams) > 1:
        print(f"FATAL: {len(fams)} dump families matched {pat}: {', '.join(fams)}. "
              f"Pass one family's glob (e.g. '<dir>/{fams[0]}.*'), or move the others "
              f"aside; interleaving two runs by step number is silently wrong.",
              file=sys.stderr)
        return 2
    if len(paths) < 6:
        print(f"FATAL: {len(paths)} dumps matched {pat}; need at least 6", file=sys.stderr)
        return 2

    s = series(paths, a.dt, a.k)
    xp, x90 = kljun_geometry(s, a.zm, a.wth)
    ok, rows, reported, sel = score(s, xp, x90, a.score_h)

    f = 2 * 7.292e-5 * math.sin(math.radians(42.957160))
    period = 2 * math.pi / f / 3600.0
    print(f"  {a.label or os.path.basename(os.path.dirname(paths[0]))}: {len(paths)} dumps "
          f"to {s['t'][-1]:.2f} simulated hours = {s['t'][-1]/period:.2f} inertial periods "
          f"(2pi/f = {period:.1f} h)")
    print(f"\n  z_i is the height where resolved TKE falls below a FIXED "
          f"{ZI_ABS} m2/s2 (gated); the 5%-of-running-peak depth is reported beside it "
          f"and is the currency lpdm/les_stats.py:window_stats uses for the corpus input.")
    print(f"\n  {'window':>12}{'u*':>9}{'U(10)':>8}{'U/u*':>8}{'sw/u*':>8}"
          f"{'TKE/u*^2':>10}{'z_i':>7}{'dir':>7}")
    t = s["t"]
    for lo in np.arange(0.0, t[-1] - 0.5 + 1e-9, 1.0):
        m = (t >= lo) & (t < lo + 1.0)
        if m.sum() < 3:
            continue
        print(f"  {lo:4.1f}-{lo+1:<7.1f}{s['ustar'][m].mean():9.4f}{s['U'][m].mean():8.3f}"
              f"{(s['U']/s['ustar'])[m].mean():8.3f}{(s['sw']/s['ustar'])[m].mean():8.3f}"
              f"{(s['tke']/s['ustar']**2)[m].mean():10.3f}{s['zi'][m].mean():7.0f}"
              f"{s['wdir'][m].mean():7.1f}")
    print(f"\n  === GATED: the footprint's controlling parameters, last {a.score_h:.1f} h ===")
    for r in rows:
        print(f"  {r['name']:<28}{r['mean']:10.4f}{r['trend_pct_per_h']:+9.2f} %/h  "
              f"(limit {r['limit']:.0f})   {abs(r['trend_pct_per_h'])*40/60:5.2f}% per "
              f"40-min window   {'ok' if r['ok'] else 'DRIFTING'}")
        if "n_levels" in r:
            print(f"    ^ on {r['n_levels']} distinct model level(s) spanning "
                  f"{r['level_span_m']:.0f} m across the window"
                  + ("  -- A STAIRCASE: a linear trend through this many levels reports "
                     "the staircase as much as any drift"
                     if r["n_levels"] <= 4 else ""))
    print(f"\n  === REPORTED, not gated: the mean flow rides the inertial oscillation ===")
    for r in reported:
        print(f"  {r['name']:<28}{r['mean']:10.4f}{r['trend_pct_per_h']:+9.2f} %/h")
    print(f"  x_peak spans {xp[sel].min():.1f}-{xp[sel].max():.1f} m across the scored "
          f"window, against a 16 m raster cell.")
    verdict = ("PASS" if ok else
               "FAIL -- a footprint-controlling parameter is still drifting")
    print(f"\n  SEED STATIONARITY: {verdict}")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"label": a.label, "pass": ok, "dt": a.dt, "wth": a.wth,
                   "score_h": a.score_h, "n_dumps": len(paths),
                   "t_end_h": float(s["t"][-1]), "gated": rows, "reported": reported,
                   "final": {"ustar": float(s["ustar"][-1]), "U": float(s["U"][-1]),
                             "zi": float(s["zi"][sel].mean()),
                             "zi_definition": f"fixed resolved-TKE threshold "
                                              f"{ZI_ABS} m2/s2 (gated)",
                             "zi_peakfrac": float(s["zi_peakfrac"][sel].mean()),
                             "tke_peak": float(s["tkepeak"][sel].mean()),
                             "wdir": float(s["wdir"][-1]),
                             "sigma_v": float(s["sv"][-1]),
                             "sigma_w": float(s["sw"][-1]),
                             "theta0": float(s["th0"][-1]),
                             "x_peak": float(xp[-1]), "x90": float(x90[-1])},
                   "last_dump": os.path.basename(paths[-1])},
                  open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
