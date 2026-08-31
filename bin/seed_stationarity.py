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


def tke_bl_average(tk, z, zi):
    """Resolved TKE averaged over the BOUNDARY LAYER, not over the whole column.

    THE ONE DEFINITION. bin/run_pass5.sh imports this rather than restating it: that file
    already carried an inline 5%-of-peak copy of the depth while importing LIMITS from
    here, and the same file then broke when the TKE key changed. A gate with a private
    copy of a definition is how stage4_wellmixed.py came to score a closure the footprints
    did not compute.

    The column mean divides by the whole 2500 m box, so it rises mechanically as z_i rises
    even in an equilibrated layer. This form does not: measured, it has nearly the same
    value across rungs (1.4814 nbl-deep vs 1.4262 nbl-shallow) where the column mean
    differs by 44%.
    """
    tk = np.asarray(tk, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    top = max(float(zi), float(z[2]))
    m = z <= top
    return float(np.trapezoid(tk[m], z[m]) / top)


def zi_peak_fraction(tk, z, frac=0.05):
    """Depth from a fraction of the profile's OWN peak. Reported, never gated.

    This is what lpdm/les_stats.py:window_stats produces as the corpus input `h`, so the
    library's depth axis stays commensurable with it. FASTEDDY_TRAPS.md 16 is the whole
    story of why it must not be trended.
    """
    return bl_depth(tk, z, frac=frac)


# Percent-per-hour trend limits, scored over the last SCORE_H hours. Single definition;
# bin/run_pass5.sh imports this dict rather than restating it.
# HOW MANY STANDARD ERRORS OF SEPARATION A VERDICT NEEDS. 3 SE is ~99.7% under a normal
# approximation; the estimator here is a least-squares slope over 19-25 correlated dumps,
# so the normal approximation is itself rough and 3 is chosen to be comfortably clear of
# the 1-2 SE band where both neutral seeds actually landed rather than to hit a p-value.
RESOLVE_SE = 3.0

LIMITS = {
    "U/u* (Kljun Pi_4)": 1.0,
    "sigma_v/u*": 3.0,
    "sigma_w/u* at the receptor": 2.0,
    "TKE_BL/u*^2": 5.0,
    "z_i": 3.0,
    "Kljun x_peak": 1.0,
    "Kljun x90": 1.0,
}


def series(paths, dt, k):
    """Per-dump receptor-level moments and the derived Kljun geometry inputs."""
    from netCDF4 import Dataset
    out = {n: [] for n in ("t", "ustar", "tke", "tke_bl", "zi", "zi_peakfrac", "sw",
                          "sv", "U", "wdir", "th0", "tkepeak")}
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
        _zi = zi_fixed(tk, z)
        out["tke"].append(float(tk.mean()))               # column mean; REPORTED only
        # === THE GATED TKE IS THE BOUNDARY-LAYER AVERAGE, NOT THE COLUMN MEAN =========
        # The column mean divides by the whole 2500 m box, so it rises MECHANICALLY as
        # z_i rises even in a layer that is otherwise in equilibrium -- it is not a
        # scale-free quantity and it was never a fair thing to trend. Measured on the two
        # neutral seeds, the boundary-layer average has nearly the same VALUE across
        # rungs (1.4814 at nbl-deep against 1.4262 at nbl-shallow) while the column mean
        # differs by 44% (1.1430 against 0.7955) -- which is the signature of a quantity
        # carrying the depth rather than the turbulence.
        #
        # This is wrong even when it PASSES. nbl-shallow passed the column-mean form, and
        # part of what it passed on was its own shallower depth.
        out["tke_bl"].append(tke_bl_average(tk, z, _zi))
        out["tkepeak"].append(float(tk[int(np.argmax(tk))]))
        out["zi"].append(_zi)                             # GATED: fixed threshold
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
    # === A SCORING WINDOW THAT REACHES STEP 0 IS SCORING THE RESTART, NOT THE FLOW ======
    # Step 0 is the state the run was HANDED: a cold start there has u* = 0 exactly, no
    # resolved turbulence and z_i undefined, so every trend it enters is dominated by the
    # spin-up transient and reports it as drift. It is not a small contamination -- it is
    # the largest excursion in the series, at the far end of the lever arm.
    #
    # This became reachable rather than theoretical when the seed ceiling moved to 2.0
    # sim-h: --score-h defaults to 2.0, so the window became the WHOLE RUN. MEASURED on
    # seed_cbl-mid_a015 (ninth pass, run 1, a 0.917 sim-h run scored over 2.0 h): the gate
    # returned an unscoreable verdict built on step 0 rather than saying so.
    #
    # Refuse rather than clamp. Clamping would silently score a different window than the
    # one asked for, and the caller -- jobs/run_seed.sh -- is what should choose the width
    # against the run it actually got.
    if bool(sel[0]) and float(t[0]) < 1e-9:
        raise ValueError(
            f"the {score_h:.2f} h scoring window reaches the FIRST dump of the series "
            f"(t = {t[0]:.3f} h, and the run ends at {t[-1]:.3f} h), so it is scoring the "
            f"state the run was handed. At a cold start that dump has u* = 0 and no "
            f"resolved turbulence, and every trend through it reports the spin-up. Score "
            f"a window strictly inside the run: --score-h below {t[-1]:.2f}.")

    def slope_of(y):
        A = np.vstack([t[sel], np.ones(int(sel.sum()))]).T
        return float(np.linalg.lstsq(A, y[sel], rcond=None)[0][0])

    def trend(y):
        return 100.0 * slope_of(y) / max(abs(y[sel].mean()), 1e-30)

    def trend_se(y):
        """The trend's OWN standard error, in %/h, and the effective sample size.

        A TREND IS AN ESTIMATE AND HAS A SAMPLING ERROR; a limit it is compared against
        is meaningless without one. This is the same rule PROJECT_BRIEF.md already states twice
        -- score a second moment against its own sampling spread, and never quote a
        tolerance without saying how many independent realisations went into it -- applied
        to the estimator rather than to the quantity.

        THE CORRECTION FOR AUTOCORRELATION IS NOT OPTIONAL HERE. Dumps are 300 s apart and
        the eddy turnover at these depths is 1300-1500 s, so consecutive dumps are not
        independent: measured, the residuals of TKE/u*^2 carry rho = +0.43 to +0.66, and
        19 dumps are worth n_eff = 4 to 8. The naive least-squares SE understates the
        spread by sqrt(n/n_eff), which is a factor of 1.6-2.2. Bartlett's AR(1) form is
        used because it needs only one lag and the series is short.
        """
        tt, yy = t[sel], y[sel]
        n = int(tt.size)
        A = np.vstack([tt, np.ones(n)]).T
        b = np.linalg.lstsq(A, yy, rcond=None)[0]
        r = yy - A @ b
        if n <= 3:
            return float("nan"), float(n)
        s2 = float((r ** 2).sum()) / (n - 2)
        sxx = float(((tt - tt.mean()) ** 2).sum())
        se = np.sqrt(s2 / max(sxx, 1e-30))
        with np.errstate(invalid="ignore"):
            rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        if not np.isfinite(rho):
            rho = 0.0
        # CLAMPED TO [3, n]. Negative residual autocorrelation genuinely does make a
        # mean more precise than independent sampling would, so Bartlett's formula
        # can return n_eff > n -- measured 40.4 from 19 dumps on U/u*. Reporting more
        # independent samples than dumps reads as an error whatever the algebra says,
        # and clamping only ever makes the SE more conservative, which is the safe
        # direction for something a gate is read against.
        neff = float(np.clip(n * (1.0 - rho) / (1.0 + rho), 3.0, n))
        return (100.0 * se * np.sqrt(n / neff) / max(abs(yy.mean()), 1e-30), neff)

    def trend_deg(y):
        """d(bearing)/dt in DEG PER HOUR, on an unwrapped series.

        A BEARING HAS NO PERCENTAGE. Reported as %/h it was 100*slope/mean, so the same
        physical backing read -3.15 %/h at a mean of 258 deg and would read -8.1 %/h at a
        mean of 100 -- and nobody can convert either back to a rate without also knowing
        the mean. Worse, the series is modular: a run drifting through north takes the
        mean of {359, 1} as 180 and the slope through the wrap is meaningless. Unwrapping
        first and reporting deg/h fixes both, and deg/h is the unit the projection in
        bin/pick_seed.py actually consumes.
        """
        return float(np.degrees(slope_of(np.unwrap(np.radians(np.asarray(y))))))

    us = s["ustar"]
    quantities = (("U/u* (Kljun Pi_4)", s["U"] / us),
                  ("sigma_v/u*", s["sv"] / us),
                  ("sigma_w/u* at the receptor", s["sw"] / us),
                  ("TKE_BL/u*^2", s["tke_bl"] / us ** 2),
                  ("z_i", s["zi"]),
                  ("Kljun x_peak", xp),
                  ("Kljun x90", x90))
    rows, ok, n_indet = [], True, 0
    for nm, y in quantities:
        v = trend(y)
        se, neff = trend_se(y)
        # === RESOLVABILITY IS PART OF THE VERDICT, NOT A FOOTNOTE ===================
        # A limit whose threshold sits within RESOLVE_SE standard errors of the
        # measurement cannot separate PASS from FAIL: measured on the two neutral seeds,
        # the ACCEPTED one read +4.32 +/- 3.46 %/h against a 5.0 limit (0.2 SE of margin)
        # and would have read +6.51 -- a FAIL -- had its run stopped fifteen minutes
        # earlier, while the REJECTED one read +8.13 +/- 3.09 (1.0 SE). The two seeds were
        # never distinguishable by that limit, and calling one PASS and the other FAIL
        # asserted a difference the data does not contain.
        #
        # So such a limit returns INDETERMINATE. This is the same move as
        # docker/turb_alive.py refusing to let a SKIP read as a PASS: a check that could
        # not run is not a check that passed. It does NOT loosen anything -- an
        # INDETERMINATE limit fails the run just as a DRIFTING one does, because a seed
        # whose stationarity is unestablished is not a seed. What changes is that the run
        # is refused for the honest reason.
        resolvable = bool(np.isfinite(se) and se > 0
                          and abs(abs(v) - LIMITS[nm]) > RESOLVE_SE * se)
        if not resolvable:
            g_ = None
            n_indet += 1
        else:
            g_ = bool(abs(v) < LIMITS[nm])
        ok &= bool(g_)                      # None -> False: neither PASS nor DRIFTING
        r = {"name": nm, "mean": float(y[sel].mean()),
             "trend_pct_per_h": float(v), "limit": LIMITS[nm], "ok": g_,
             "trend_se_pct_per_h": float(se), "n_eff": float(neff),
             "resolvable": resolvable,
             "margin_se": (float(abs(abs(v) - LIMITS[nm]) / se)
                           if np.isfinite(se) and se > 0 else float("nan")),
             "verdict": ("INDETERMINATE" if g_ is None
                         else ("ok" if g_ else "DRIFTING"))}
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
                 "trend_pct_per_h": float(trend(y)), "unit": "%/h"}
                for nm, y in (("u*", us), ("U(10 m)", s["U"]),
                              ("TKE_column/u*^2 (retired form)", s["tke"] / us ** 2),
                              ("domain TKE", s["tke"]),
                              ("peak resolved TKE", s["tkepeak"]),
                              ("z_i, 5% of the running peak", s["zi_peakfrac"]))]
    # DIRECTION DRIFT AT FREEZE -- the number bin/pick_seed.py projects forward.
    # MEASURED, and it inverted the design assumption. pick_seed used to assume the
    # 30-minute adjustment CLOSES a direction gap by ~2.7 deg. On the first corpus case
    # it did the opposite: the seed was frozen mid-backing and simply kept turning, so
    # the gap WIDENED from 11.3 to 21.8 deg. 30 min is 2.8% of a 17.6 h inertial period,
    # which is far too little for the case's own forcing to assert itself; what the case
    # inherits is the seed's angular momentum, not the seed's angle.
    dwdir = trend_deg(s["wdir"])
    reported.insert(2, {"name": "wind direction", "mean": float(s["wdir"][sel].mean()),
                        "trend_deg_per_h": dwdir, "unit": "deg/h"})
    return bool(ok), rows, reported, sel, dwdir, n_indet


def sweep(s, xp, x90, widths):
    """Trend, SE and n_eff for every limit at several scoring-window widths.

    THE WINDOW LENGTH IS A GATE PARAMETER AND IT WAS NEVER MEASURED. 1.5 h was inherited
    from the run that first passed at 3.0 h, not chosen against the estimators' own
    resolution. A trend's SE falls roughly as T^(-3/2) for independent samples, so a wider
    window buys resolution fast -- but it also reaches back into the cold-start transient,
    which biases the trend. This prints both sides of that trade so the width is picked on
    evidence and not on inheritance.
    """
    rows = []
    for w in widths:
        try:
            ok, gated, _rep, sel, _d, _ni = score(s, xp, x90, w)
        except ValueError as e:
            rows.append((w, None, str(e)))
            continue
        rows.append((w, dict(ok=ok, n=int(sel.sum()), gated=gated), None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="a directory, or a glob of dumps")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--wth", type=float, default=0.0,
                    help="the PRESCRIBED surface virtual heat flux, for L in the Kljun "
                         "terms. The resolved covariance at k=0 is not it (cbl_check.py).")
    # === 2.0 h, MEASURED, NOT INHERITED ==========================================
    # 1.5 h was carried over from the first run that passed at 3.0 h; it was never chosen
    # against the estimators' own resolution. Swept over 1.0-2.5 h on both neutral seeds
    # (--sweep-score-h), 2.0 h is the best width available in a 3.0 h run:
    #
    #   * the four oscillation-immune limits improve by ~50% in margin: U/u* goes from
    #     4.7 to 7.2 SE (nbl-shallow) and 11.1 to 10.2 (nbl-deep); x90 5.6 -> 8.7 and
    #     13.5 -> 13.8; sigma_w/u* 3.4 -> 5.3 and 6.9 -> 6.3;
    #   * sigma_v/u* CROSSES into resolvable on nbl-shallow (2.6 -> 3.6 SE);
    #   * and 2.25-2.5 h reaches back into the cold-start transient, where the z_i trend
    #     blows up to +9.0 and +9.6 %/h on the two seeds. 2.0 h does not.
    #
    # WHAT IT DOES NOT DO IS FIX THE TWO THAT MATTER, and that is the finding rather than
    # a shortcoming of the width. TKE_BL/u*^2 holds n_eff = 3.0-5.5 at EVERY width from
    # 1.0 h to 2.5 h, and z_i holds 3.0-9.6: adding dumps adds no independent information,
    # because both decorrelate on the EDDY TURNOVER (h/u* = 1330-1470 s here), not on the
    # 300 s dump interval. A 2.0 h window is 4.9-5.4 turnovers, so it can hold about five
    # independent samples of them however finely it is sampled. Those two limits are
    # bounded by RUN LENGTH, and nothing about the scoring can substitute for it.
    ap.add_argument("--score-h", type=float, default=2.0)
    ap.add_argument("--zm", type=float, default=10.0)
    ap.add_argument("--dx", type=float, default=16.0,
                    help="raster cell size, for reporting the x_peak span against it")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--sweep-score-h", default=None,
                    help="comma-separated window widths to report trend/SE/n_eff at, "
                         "e.g. '1.0,1.5,2.0,2.5'. Diagnostic: the verdict still comes "
                         "from --score-h.")
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
    # A REFUSED SCORING WINDOW IS A NAMED FATAL, NOT A TRACEBACK. The gate's stdout is
    # tee'd by jobs/run_seed.sh and its verdict is read back out of the JSON, so a traceback
    # here surfaces to the driver only as "the gate wrote no JSON" -- true, and silent about
    # why. Say why, and exit non-zero (FASTEDDY_TRAPS.md 12).
    try:
        ok, rows, reported, sel, dwdir, n_indet = score(s, xp, x90, a.score_h)
    except ValueError as e:
        print(f"FATAL: the scoring window is not usable -- {e}", file=sys.stderr)
        return 2

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
              f"40-min window   {r['verdict']}")
        if np.isfinite(r["trend_se_pct_per_h"]):
            print(f"    ^ trend SE {r['trend_se_pct_per_h']:.2f} %/h "
                  f"(AR(1)-corrected, n_eff {r['n_eff']:.1f} of {int(sel.sum())} dumps); "
                  f"|trend| is {abs(r['trend_pct_per_h'])/max(r['trend_se_pct_per_h'],1e-9):.1f} "
                  f"SE from zero and the limit is "
                  f"{abs(abs(r['trend_pct_per_h'])-r['limit'])/max(r['trend_se_pct_per_h'],1e-9):.1f} "
                  f"SE away"
                  + ("" if r["resolvable"] else
                     f"  -- INDETERMINATE: {RESOLVE_SE:.0f} SE of separation is required "
                     f"and there is {r['margin_se']:.1f}. This limit cannot tell PASS "
                     f"from FAIL on this window, so it asserts neither"))
        if "n_levels" in r:
            print(f"    ^ on {r['n_levels']} distinct model level(s) spanning "
                  f"{r['level_span_m']:.0f} m across the window"
                  + ("  -- A STAIRCASE: a linear trend through this many levels reports "
                     "the staircase as much as any drift"
                     if r["n_levels"] <= 4 else ""))
    print(f"\n  === REPORTED, not gated: the mean flow rides the inertial oscillation ===")
    for r in reported:
        v = r.get("trend_pct_per_h", r.get("trend_deg_per_h"))
        print(f"  {r['name']:<28}{r['mean']:10.4f}{v:+9.2f} {r['unit']}")
    print(f"  ^ the direction row is the DRIFT AT FREEZE. bin/pick_seed.py projects it "
          f"forward to the window's own midpoint rather than assuming the adjustment "
          f"closes a gap; measured, the adjustment WIDENS one.")
    # THE CELL SIZE IS THE GRID'S, NOT A CONSTANT. It read "a 16 m raster cell" on a 24 m
    # grid -- the same defect as FASTEDDY_TRAPS.md 19, in a print rather than in a
    # calculation, which makes it worse rather than better: a wrong number in prose is
    # copied into a result file and never recomputed.
    print(f"  x_peak spans {xp[sel].min():.1f}-{xp[sel].max():.1f} m across the scored "
          f"window, against a {a.dx:.0f} m raster cell.")
    if a.sweep_score_h:
        widths = [float(x) for x in a.sweep_score_h.split(",")]
        print(f"\n  === SCORING-WINDOW SWEEP: does the gate resolve at all, and where? ===")
        for w, r, err in sweep(s, xp, x90, widths):
            if r is None:
                print(f"  {w:.2f} h: {err}")
                continue
            print(f"\n  --- window {w:.2f} h ({r['n']} dumps) ---")
            print(f"    {'quantity':28}{'trend':>9}{'SE':>8}{'n_eff':>7}{'limit':>7}"
                  f"{'margin/SE':>11}  verdict")
            for g in r["gated"]:
                se = g["trend_se_pct_per_h"]
                marg = (abs(abs(g["trend_pct_per_h"]) - g["limit"]) / se
                        if np.isfinite(se) and se > 0 else float("nan"))
                print(f"    {g['name']:28}{g['trend_pct_per_h']:+9.2f}{se:8.2f}"
                      f"{g['n_eff']:7.1f}{g['limit']:7.1f}{marg:11.1f}  {g['verdict']}")

    drifting = [r["name"] for r in rows if r["ok"] is False]
    indet = [r["name"] for r in rows if r["ok"] is None]
    if ok:
        verdict = "PASS"
    elif drifting:
        verdict = ("FAIL -- still drifting: " + ", ".join(drifting)
                   + (f"; and INDETERMINATE: {', '.join(indet)}" if indet else ""))
    else:
        # NOT A PASS AND NOT A DRIFT. The run is refused because its stationarity is
        # UNESTABLISHED, which is a different thing to report and a different thing to
        # fix: a drifting seed needs different physics or more time, an indeterminate one
        # needs a longer scoring window or more dumps.
        verdict = ("FAIL -- INDETERMINATE: " + ", ".join(indet)
                   + f". No limit is drifting; the gate cannot resolve these at "
                     f"{a.score_h:.2f} h. DUMPING MORE OFTEN WILL NOT HELP -- measured, "
                     f"n_eff for these saturates at 3-5 at every window width from 1.0 to "
                     f"2.5 h, because they decorrelate on the eddy turnover and not on "
                     f"the dump interval. What is short is the RUN. Do NOT loosen the "
                     f"threshold and do NOT read this as a pass.")
    print(f"\n  SEED STATIONARITY: {verdict}")

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"label": a.label, "pass": ok,
                   "n_indeterminate": int(n_indet),
                   "indeterminate": [r["name"] for r in rows if r["ok"] is None],
                   "drifting": [r["name"] for r in rows if r["ok"] is False],
                   "resolve_se": RESOLVE_SE,
                   "dt": a.dt, "wth": a.wth,
                   "score_h": a.score_h, "n_dumps": len(paths),
                   "t_end_h": float(s["t"][-1]), "gated": rows, "reported": reported,
                   "final": {"ustar": float(s["ustar"][-1]), "U": float(s["U"][-1]),
                             "zi": float(s["zi"][sel].mean()),
                             "zi_definition": f"fixed resolved-TKE threshold "
                                              f"{ZI_ABS} m2/s2 (gated)",
                             "zi_peakfrac": float(s["zi_peakfrac"][sel].mean()),
                             "tke_peak": float(s["tkepeak"][sel].mean()),
                             "wdir": float(s["wdir"][-1]),
                             "dwdir_dt_deg_per_h": dwdir,
                             "dwdir_scored_over_h": float(a.score_h),
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
