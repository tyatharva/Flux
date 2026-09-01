#!/usr/bin/env python3
"""Did this short cold start actually work? Validation item 3, for each new regime config.

A 5-minute cold start cannot tell you a seed is converged -- nothing at 5 minutes can. It
CAN tell you the configuration is not broken, which is the only question worth asking
before committing 3.1 h of GPU per job to it:

  finiteness      np.isfinite(...).all(), never isnan().any(). `inf` is not NaN, NaN passes
                  every `>` comparison, and FastEddy exits 0 on fully-NaN fields.
  k0/k1 < 1       the accuracy-CFL check. ~0.27 is right, ~9 means dt is past the accuracy
                  boundary and the lowest levels are grid-scale acoustic noise rather than
                  turbulence -- which otherwise looks completely fine (PROJECT_BRIEF.md).
  turbulence      AND k0/k1 IS NOT THAT CHECK. It read 0.442 -- a pass -- on a boundary
     alive        layer whose turbulence had entirely collapsed, because it is a ratio
                  between two levels and both went quiet together. docker/turb_alive.py
                  asks the separate question, and this imports it rather than
                  reimplementing it (PROJECT_BRIEF.md: gates import the production function).
  the receptor    k = 2 must sit at 10.000 m, or every footprint is computed at the wrong
                  height with nothing in the output to say so.
  z_i in range    the capping inversion is the CONTROL on z_i; if the achieved depth is
                  nowhere near the rung's target the control is not working.
  the log         grepped for `outside limits` as well as CORRUPTED -- an out-of-range
                  parameter does not stop FastEddy, it silently uses the compiled-in
                  default (docs/FASTEDDY_TRAPS.md 13).

usage: smoke_check.py <dump> [--manifest jobs/<job>/manifest.json] [--log FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--case-in", default=None,
                    help="the .in this run used; enables the base-state check")
    ap.add_argument("--receptor", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()

    from netCDF4 import Dataset
    if not os.path.exists(a.dump):
        print(f"FATAL: {a.dump} does not exist", file=sys.stderr)
        return 2

    ok = True
    def chk(good, name, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'ok  ' if good else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")

    with Dataset(a.dump) as ds:
        g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
        z = g("zPos")[:, 0, 0]
        flds = {v: g(v) for v in ("u", "v", "w", "theta", "TKE_0", "fricVel")}

    bad = [v for v, arr in flds.items() if not np.isfinite(arr).all()]
    chk(not bad, "every field finite",
        f"{len(flds)} fields" if not bad else f"non-finite: {bad}")
    if bad:
        return 1

    pr = lambda arr: arr - arr.mean(axis=(-2, -1), keepdims=True)
    u, v, w, th, e = (flds[k] for k in ("u", "v", "w", "theta", "TKE_0"))
    e = np.maximum(e, 0.0)
    ww = (pr(w) ** 2).mean(axis=(-2, -1))
    ratio = ww[0] / max(ww[1], 1e-30)
    # A cold start can have ww[1] below the floor docker/k0k1_check.py uses, in which case
    # the ratio is a quotient of two nearly-zero numbers and says nothing. Report that
    # rather than passing or failing on noise.
    if ww[1] < 1e-8:
        print(f"  SKIP  k0/k1  -- ww[1] = {ww[1]:.2e} is below the floor; too early to say")
    else:
        chk(ratio < 1.0, "k0/k1 resolved w variance ratio < 1",
            f"{ratio:.3f}  (~0.27 is right, ~9 means dt past the accuracy boundary)")

    # The physics companion. One dump, so only the resolved-TKE half of turb_alive can
    # fire here -- the u* series tests need a series and belong in the seed report.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "docker"))
    import turb_alive                                              # noqa: E402
    # verdict_for(), not a locally-written comparison: a 5 min cold start is legitimately
    # below the floor while CLIMBING, and only the series can tell that from a collapse.
    ta_status, ta_msg = turb_alive.verdict_for(a.dump)
    tr = turb_alive.scan([a.dump])[0]
    if ta_status == "SKIP":
        print(f"  SKIP  turbulence alive -- {ta_msg.strip()}")
    else:
        chk(ta_status == "OK", "turbulence alive",
            f"max e_res/U_ref^2 = {tr['e_over_uref2']:.2e} vs floor "
            f"{turb_alive.E_FLOOR:.1e}  (e_res/u*^2 = {tr['e_over_ust2']:.2f}, which is "
            f"the metric that CANNOT see a collapse -- reported, never gated)")
        if ta_status == "FAIL":
            print(ta_msg)

    # TOLERANCE 1e-4 m, NOT ZERO, and the reason is the file rather than the grid.
    # bin/vgrid.py solves the receptor onto k=2 at 10.000000000 m in fp64, but FastEddy is
    # hardwired fp32 and writes zPos as NC_FLOAT -- so what comes back is 10.000011 m, a
    # relative error of 1.1e-6, which is fp32 roundoff accumulated through the cubic. A
    # tolerance tighter than the file's own precision fails a correct grid; 0.1 mm against
    # a 3.99 m first layer is far below anything a footprint can resolve.
    chk(abs(z[a.k] - a.receptor) < 1e-4, f"receptor on cell centre k={a.k}",
        f"z = {z[a.k]:.6f} m, wanted {a.receptor:.6f} "
        f"(fp32 zPos; {abs(z[a.k]-a.receptor)*1e3:.3f} mm)")

    tk = 0.5 * ((pr(u) ** 2 + pr(v) ** 2 + pr(w) ** 2).mean(axis=(-2, -1)))
    kmax = int(np.argmax(tk))
    ab = np.where(tk[kmax:] < 0.05 * tk[kmax])[0]
    zi = float(z[kmax + ab[0]]) if len(ab) else float(z[-1])
    print(f"        u* {flds['fricVel'].mean():.4f} m/s, U {np.hypot(u[a.k].mean(), v[a.k].mean()):.3f} m/s "
          f"at the receptor, theta {th.min():.2f}-{th.max():.2f} K")

    # === THE BASE STATE FastEddy ACTUALLY BUILT, against the .in that asked for it ===
    # This is the one check that closes the loop. bin/test_sounding.py verifies the fit
    # arithmetically, but "my formula reproduces my formula" is not evidence that FastEddy
    # read the six numbers, inverted temp_grnd into theta_grnd with ITS gas constants, and
    # integrated the hydrostatic profile the way the source says it does. The dump is.
    # Compared ABOVE the boundary layer, where the surface flux has not yet moved the
    # profile after a few minutes: below z_i a correct run SHOULD differ from the base
    # state, so scoring there would fail the physics rather than the plumbing.
    if a.case_in and os.path.exists(a.case_in):
        par = {}
        for ln in open(a.case_in):
            if "=" in ln and not ln.strip().startswith("#"):
                k, val = ln.split("=", 1)
                val = val.split("#", 1)[0].strip()
                try:
                    par[k.strip()] = float(val)
                except ValueError:
                    pass
        need = ("temp_grnd", "pres_grnd", "zStableBottom", "stableGradient",
                "zStableBottom2", "stableGradient2", "zStableBottom3", "stableGradient3")
        if all(k in par for k in need):
            fe_kappa = 287.04 / (287.04 + 718.0)      # hydro_core.c:1574-1580
            th_g = par["temp_grnd"] * (1.0e5 / par["pres_grnd"]) ** fe_kappa
            ramp = lambda lo, hi: np.clip(z, lo, hi) - lo
            base = (th_g
                    + par["stableGradient"] * ramp(par["zStableBottom"],
                                                   par["zStableBottom2"])
                    + par["stableGradient2"] * ramp(par["zStableBottom2"],
                                                    par["zStableBottom3"])
                    + par["stableGradient3"] * np.maximum(z - par["zStableBottom3"], 0.0))
            thbar = th.mean(axis=(-2, -1))
            tk_ = 0.5 * ((pr(u) ** 2 + pr(v) ** 2 + pr(w) ** 2).mean(axis=(-2, -1)))
            km = int(np.argmax(tk_))
            abv = np.where(tk_[km:] < 0.05 * tk_[km])[0]
            zi_ = float(z[km + abv[0]]) if len(abv) else float(z[-1])
            m_ = (z > zi_ + 200.0) & (z < 2000.0)      # above the BL, below the damping layer
            if m_.sum() >= 5:
                d_ = float(np.abs(thbar[m_] - base[m_]).max())
                # SUBSIDENCE MOVES THE PROFILE ABOVE THE BL, AND THAT IS THE POINT OF IT.
                # lsfSelector = 1 with lsf_horMnSubTerms = 1 advects the SLAB MEAN against
                # its own gradient, so a convective rung's free atmosphere warms by
                # (descent) x (local base-state gradient). Scoring it against zero would
                # fail a correctly working scheme -- and it did, at 0.1596 K on cbl-mid.
                # The tolerance is therefore the PREDICTED warming, which turns the
                # nuisance into a second confirmation of Gate B7.
                tol, why = 0.02, "no subsidence"
                wsub = abs(par.get("lsf_w_lev1", 0.0))
                if (wsub > 0 and par.get("lsfSelector", 0) == 1
                        and par.get("lsf_horMnSubTerms", 0) == 1):
                    step = int(a.dump.rsplit(".", 1)[1])
                    tsim = step * par.get("dt", 0.0)
                    drop = wsub / 3600.0 * tsim              # m; the kernel divides by 3600
                    grad = float(np.max(np.abs(np.gradient(base, z))[m_]))
                    tol = 0.02 + 1.3 * drop * grad           # 30% headroom on the taper
                    why = (f"{wsub:.0f} m/h x {tsim:.0f} s = {drop:.2f} m of descent "
                           f"through a {grad:.3f} K/m gradient = {drop*grad:.3f} K expected")
                chk(d_ < tol, "FastEddy's base state matches the .in above the BL",
                    f"max |theta_LES - theta_base| = {d_:.4f} K vs tolerance {tol:.4f} "
                    f"({why}) over {int(m_.sum())} levels, {z[m_][0]:.0f}-{z[m_][-1]:.0f} m; "
                    f"theta_grnd {th_g:.4f} K")
            else:
                print(f"  SKIP  base state -- only {int(m_.sum())} levels above z_i")

    if a.manifest and os.path.exists(a.manifest):
        m = json.load(open(a.manifest))
        tgt = float(m["target"]["zi_m"])
        # A 5-minute cold start is nowhere near equilibrium, so this is a WIDE bracket: it
        # is asking whether the capping inversion is holding the depth at all, not whether
        # the seed has converged.
        chk(0.4 * tgt <= zi <= 2.5 * tgt, "z_i within a factor of ~2 of the rung target",
            f"{zi:.0f} m against a {tgt:.0f} m target ({m['rung']}, {m['regime']})")
    else:
        print(f"        z_i (5% of peak TKE) = {zi:.0f} m")

    if a.log and os.path.exists(a.log):
        txt = open(a.log, errors="replace").read()
        for pat, why in (("CORRUPTED", "the field integrity check"),
                         ("outside limits", "an out-of-range parameter (TRAPS 13)"),
                         ("Segmentation", "a segfault"),
                         ("illegal memory", "an illegal memory access")):
            chk(pat not in txt, f"log clean of '{pat}'", why)

    print(f"\n  SMOKE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
