#!/usr/bin/env python3
"""One HRRR pseudo-sounding -> the FastEddy .in parameters that force that case.

Stage 2 of the corpus pipeline: bin/hrrr_sounding.py -> HERE -> bin/pick_seed.py.

WHAT THIS PRODUCES. The six stabilityScheme = 2 numbers that reproduce the sounding's
potential-temperature profile over the LES column, the geostrophic wind, the surface
virtual heat flux (both as the flat-spin-up scalar and as the cropland reference that
prep_surface.py's per-cell map is built from), the ground state, and a dt that lands the
5 s output cadence on an integer step count.

=== THE BASE-STATE FIT, AND THE THREE CONSTRAINTS FastEddy PUTS ON IT ===

Read out of SRC/HYDRO_CORE/hydro_core.c:1776-1822, not out of documentation. The base
state is CONTINUOUS PIECEWISE-LINEAR IN THETA with four segments:

    z <= b1              theta = theta_grnd                       (FORCED NEUTRAL)
    b1 < z <= b2         theta = theta_grnd + g1 (z - b1)
    b2 < z <= b3         theta = ... + g2 (z - b2)
    z  > b3              theta = ... + g3 (z - b3)

  (1) THE LOWEST SEGMENT HAS NO FREE GRADIENT. It is exactly theta_grnd. For a CBL that
      is what you want -- b1 is the mixed-layer top. For a STABLE case there is no neutral
      layer to give it, so the fit drives b1 towards 0 and the first gradient segment
      starts at the ground. Nothing special is needed; it falls out of leaving b1 free.

  (2) ALL THREE GRADIENTS MUST BE STRICTLY POSITIVE. hydro_core.c:642,646,650 query them
      over [FLT_MIN, FLT_MAX], so zero and negative are both rejected. A residual layer
      with a genuinely zero gradient cannot be expressed; it is floored at GRAD_FLOOR,
      which is 0.1 K/km and therefore physically negligible over any segment the LES
      column contains.

  (3) AND A REJECTED VALUE DOES NOT STOP THE RUN. parameters.c:309-315 prints
      "ERROR: parameter '<name>' value <v> is outside limits", increments numErrors, and
      LEAVES THE VARIABLE AT ITS COMPILED-IN DEFAULT -- and FastEddy.c:96 never checks the
      return code of hydro_coreGetParams(). So an out-of-range stableGradient silently
      runs the case with 0.1 K/m: a 10 K capping inversion where the sounding wanted 0.4.
      The only trace is one line in a log that is otherwise grepped for CORRUPTED.
      This script therefore GUARANTEES the ranges rather than hoping, and --check-log
      greps a finished log for that exact error string.

  The pressure integral carries (1/g)*log(1 + g*dz/theta), which looks like it would lose
  the neutral limit as g -> 0. It does not: the literal 1.0 promotes the expression to
  double, so the term is accurate to ~1e-13 relative at GRAD_FLOOR. The positivity
  constraint is a parameter-range rule, not a numerical one.

THE FIT IS DONE ON THE LES's OWN CELL CENTRES, weighted by layer thickness. Fitting on
HRRR's 13-or-so levels below the ceiling under-resolves the inversion; fitting on the LES
levels unweighted over-resolves the surface layer, where 55 of 122 levels sit below 400 m
and the LES's own dynamics -- not the base state -- decide the answer anyway. Thickness
weighting makes the residual an integral over height, so it is invariant to how the grid
is stretched, and one deliberate tier weight below 1.5 km is then the only thumb on it.

usage: sounding_to_forcing.py results/soundings/<case>.json [--out FILE] [--in-out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

G = 9.80665
RD = 287.05
CP = 1004.5
P0 = 100000.0
KAPPA_P = RD / CP

# FASTEDDY'S OWN GAS CONSTANTS, for quantities FastEddy will reconstruct.
# hydro_core.c:1574-1580 -- R_gas = 287.04, cv_gas = 718.0, cp_gas = R + cv = 1005.04,
# so R_cp = 0.285600 against the meteorological 287.05/1004.5 = 0.285765. The gap is
# 0.06% in the exponent and 0.001 K on temp_grnd, so nothing here turns on it -- but
# temp_grnd exists ONLY to be turned back into theta_grnd by
# `theta_grnd = temp_grnd*pow(pres_grnd/refPressure, -R_cp)` (hydro_core.c:1759), and
# inverting that with the model's own constants makes the round trip exact rather than
# nearly exact. Theta itself is computed from HRRR with the meteorological values,
# because that is a meteorological quantity, not a FastEddy one.
FE_R = 287.04
FE_CV = 718.0
FE_KAPPA = FE_R / (FE_R + FE_CV)
VONK = 0.4

C_SOUND = 347.2          # bin/vgrid.py; the constant CFL_3d is defined with
GRAD_FLOOR = 1.0e-4      # K/m = 0.1 K/km. See constraint (2) above.
GRAD_CEIL = 0.5          # K/m; 500 K/km is far above any real inversion
CADENCE = 5.0            # s between LPDM dumps


# --------------------------------------------------------------------------- grid
def les_levels(nz=122, zceiling=2500.0, c1=0.194059):
    """FastEddy cell-centre heights over flat ground. grid.c:1114-1127, via bin/vgrid.py."""
    d_zeta = zceiling / (nz - 0.5)
    zeta = (np.arange(nz) + 0.5) * d_zeta
    zC = (nz - 0.5) * d_zeta
    return ((1.0 - c1) / zC ** 2) * zeta ** 3 + c1 * zeta


def derive_dt(dx, dz_sfc, cfl_max=1.35, cadence=CADENCE):
    """Largest dt at or below the CFL target that makes the cadence an integer step count.

    CFL_3d = c dt sqrt(2/dx^2 + 1/dz_sfc^2), the form that reproduces the retired 24 m
    grid's stated 1.4946 to four digits (docs/les/configuration.md). run_window.sh asserts
    |frq*dt - cadence| < 2e-4, so rounding dt and hoping is not an option.
    """
    fac = C_SOUND * np.sqrt(2.0 / dx ** 2 + 1.0 / dz_sfc ** 2)
    dt_max = cfl_max / fac
    n = int(np.ceil(cadence / dt_max))          # steps per dump, rounded UP -> dt down
    dt = cadence / n
    return float(dt), int(n), float(dt * fac)


# --------------------------------------------------------------- base-state theta fit
def _design(z, b1, b2, b3):
    """[1, ramp(b1,b2), ramp(b2,b3), ramp(b3,inf)] -- hydro_core.c:1776-1810 exactly."""
    return np.column_stack([
        np.ones_like(z),
        np.clip(z, b1, b2) - b1,
        np.clip(z, b2, b3) - b2,
        np.maximum(z - b3, 0.0),
    ])


def _solve(A, y, w):
    """Weighted LS with gradients bounded below. Unconstrained first: it almost always
    lands in range on a real profile, and lsq_linear is ~50x slower."""
    sw = np.sqrt(w)[:, None]
    Aw, yw = A * sw, y * sw[:, 0]
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    if (coef[1:] >= GRAD_FLOOR).all() and (coef[1:] <= GRAD_CEIL).all():
        return coef, float(np.sum(w * (A @ coef - y) ** 2))
    from scipy.optimize import lsq_linear
    lo = np.array([150.0, GRAD_FLOOR, GRAD_FLOOR, GRAD_FLOOR])
    hi = np.array([400.0, GRAD_CEIL, GRAD_CEIL, GRAD_CEIL])
    r = lsq_linear(Aw, yw, bounds=(lo, hi))
    return r.x, float(np.sum(w * (A @ r.x - y) ** 2))


def fit_base_state(zf, thf, wts, zceiling):
    """Breakpoints by search, amplitudes by bounded LS at each candidate triple.

    The breakpoints are the only non-convex part, and there are three of them on a bounded
    interval, so a coarse sweep followed by a local refine is both exhaustive enough and
    cheap. Nothing here is iterative-with-a-tolerance: the reported rms is the answer.
    """
    def score(b1, b2, b3):
        A = _design(zf, b1, b2, b3)
        return _solve(A, thf, wts)

    coarse = np.unique(np.concatenate([
        np.array([0.0, 25.0, 50.0, 100.0]),
        np.linspace(150.0, zceiling, 24),
    ]))
    best = None
    for i, b1 in enumerate(coarse):
        for j in range(i, len(coarse)):
            b2 = coarse[j]
            for k in range(j, len(coarse)):
                b3 = coarse[k]
                coef, sse = score(b1, b2, b3)
                if best is None or sse < best[0]:
                    best = (sse, b1, b2, b3, coef)
    # refine on a 25 m lattice around the winner, order preserved
    step = float(coarse[2] - coarse[1])
    for _ in range(3):
        sse, b1, b2, b3, coef = best
        span = max(step, 25.0)
        cand = lambda b: np.clip(np.arange(b - span, b + span + 1e-9, span / 4.0),
                                 0.0, zceiling)
        for nb1 in cand(b1):
            for nb2 in cand(b2):
                if nb2 < nb1:
                    continue
                for nb3 in cand(b3):
                    if nb3 < nb2:
                        continue
                    c, s = score(nb1, nb2, nb3)
                    if s < best[0]:
                        best = (s, nb1, nb2, nb3, c)
        step = span / 4.0
    sse, b1, b2, b3, coef = best
    th_g, g1, g2, g3 = (float(v) for v in coef)
    g1, g2, g3 = (float(np.clip(g, GRAD_FLOOR, GRAD_CEIL)) for g in (g1, g2, g3))
    model = _design(zf, b1, b2, b3) @ np.array([th_g, g1, g2, g3])
    resid = model - thf
    rms = float(np.sqrt(np.sum(wts * resid ** 2) / wts.sum()))
    return dict(theta_grnd=th_g,
                zStableBottom=float(b1), stableGradient=g1,
                zStableBottom2=float(b2), stableGradient2=g2,
                zStableBottom3=float(b3), stableGradient3=g3,
                rms_k=rms, max_abs_k=float(np.abs(resid).max()),
                model=model, resid=resid)


# ------------------------------------------------------------------------- forcing
def ekman_backing_deg(zoL):
    """Expected surface-to-geostrophic turning, from THIS project's own measurements.

    docs/les/seed-library.md: 22-25 deg in the neutral cases, 7-13 deg in the CBL. Interpolated on
    z_i/L rather than invented, and used only as a LABEL and as the optional
    --match-10m pre-compensation; the LES's achieved direction is what the corpus records.
    """
    if not np.isfinite(zoL):
        return 23.5
    if zoL <= -10.0:
        return 10.0
    if zoL >= 0.0:
        return 23.5
    f = (zoL + 10.0) / 10.0              # -10 -> 0 maps to 0 -> 1
    return 10.0 + f * 13.5


def _prep_surface_tables():
    """The class tables and virtual_factor, taken from prep_surface.py's OWN source.

    prep_surface.py imports rasterio at module scope and the analysis container has no
    rasterio (it is only needed to READ the GeoTIFFs, which this stage does not). Rather
    than duplicate the tables -- which would then drift, and the drift would be a silent
    few-percent error in every case's surface flux -- the module-level assignments and
    function definitions are executed and the imports are skipped. One definition, still.
    """
    import ast
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prep_surface.py")
    tree = ast.parse(open(p).read(), filename=p)
    body = [n for n in tree.body
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.FunctionDef))]
    ns = {"np": np, "__name__": "_prep_surface_tables"}
    exec(compile(ast.Module(body=body, type_ignores=[]), p, "exec"), ns)
    need = ("WORLDCOVER_WTH", "WORLDCOVER_BOWEN", "WTH_ARRAY", "BOWEN_ARRAY",
            "WTH_FALLBACK", "BOWEN_FALLBACK", "virtual_factor")
    miss = [k for k in need if k not in ns]
    if miss:
        raise KeyError(f"prep_surface.py no longer defines {miss} at module scope")
    return ns


def flux_split(grid_dir):
    """mean(f) over the real map: the factor between the cropland reference and the
    domain mean of the per-cell virtual flux.

    HRRR's SHTFL is a 3 km grid-cell average over a heterogeneous surface, so it is a
    DOMAIN MEAN, not a cropland value. prep_surface.py --wth takes the cropland
    REFERENCE. Dividing by mean(f) is what makes the two consistent, and getting it
    backwards spins the seed up at the wrong z_i (docs/les/seed-library.md, 0.1363 vs 0.1290).

    The grid's saved htFlux.npy is NOT used even when it is non-zero: it carries whichever
    reference flux that build happened to use, and data/grid16 is in fact a neutral build
    where the whole map is 0.0 -- so deriving the ratio from it would divide by zero on
    the grid this project actually ships.
    """
    ps = _prep_surface_tables()
    cls = np.load(os.path.join(grid_dir, "lcclass.npy"))
    array = np.load(os.path.join(grid_dir, "array.npy")) > 0.5
    bc = ps["WORLDCOVER_BOWEN"][40]
    vf = ps["virtual_factor"]
    fc = float(vf(bc))
    f = np.full(cls.shape, ps["WTH_FALLBACK"], dtype=float)
    for k, sr in ps["WORLDCOVER_WTH"].items():
        b = bc if k == 40 else ps["WORLDCOVER_BOWEN"].get(k, ps["BOWEN_FALLBACK"])
        f[cls == k] = sr * float(vf(b)) / fc
    f[array] = ps["WTH_ARRAY"] * float(vf(ps["BOWEN_ARRAY"])) / fc
    return float(f.mean())



def _grid_receptor(grid_dir):
    """Receptor height from the grid's own meta.npy, falling back loudly."""
    import os
    p = os.path.join(grid_dir or "", "meta.npy")
    if os.path.exists(p):
        return float(np.load(p, allow_pickle=True).item().get("receptor", 10.0))
    print(f"  WARNING: no {p}; assuming a 10 m receptor for the representability screen")
    return 10.0


def _grid_z0(grid_dir):
    """Geometric-mean z0 of the grid's own roughness map."""
    import os
    p = os.path.join(grid_dir or "", "z0m.npy")
    if os.path.exists(p):
        return float(np.exp(np.log(np.load(p)).mean()))
    # REFUSE RATHER THAN GUESS. 0.1435 was the 122^2 @ 16 m value and it is 0.0615 at
    # 30 m -- the box takes in more lake. A grid constant used on the wrong grid is
    # docs/reference/fasteddy-traps.md 19, and a WARNING is not enough: the run continues, the Obukhov
    # label comes out wrong by a plausible amount, and it reaches the pair. The caller
    # always has a grid; if the grid has no z0m.npy then the surface was never built and
    # nothing downstream should proceed on a literal.
    raise SystemExit(
        f"FATAL: no {p}. The geometric-mean z0 is a property of the grid this case runs "
        f"on and there is no safe default -- it is 0.1435 m at 16 m and 0.0615 m at 30 m. "
        f"Build the surface with bin/prep_surface.py, or pass --z0 explicitly.")


def build(snd, grid_dir, nz, zceiling, c1, dx, cfl, match_10m, w_low, z_low,
          nx=122, zi_l_ratio=2.0, z_recept=None, z0_dom=None, zi_max_abs=None):
    z = np.asarray(snd["profile"]["z_agl_m"], float)
    th = np.asarray(snd["profile"]["theta_k"], float)
    sfc, geo = snd["surface"], snd["geostrophic"]

    zc = les_levels(nz, zceiling, c1)
    dz_sfc = float(2.0 * zc[0])
    domain_l = nx * dx
    dt, frq, cfl_ach = derive_dt(dx, dz_sfc, cfl)

    # fit on the LES cell centres, weighted by layer thickness x one tier weight
    edges = np.concatenate([[0.0], 0.5 * (zc[1:] + zc[:-1]), [zceiling]])
    thick = np.diff(edges)
    inside = zc <= zceiling
    zf, thick = zc[inside], thick[inside]
    thf = np.interp(zf, z, th)
    wts = thick * np.where(zf <= z_low, w_low, 1.0)
    fit = fit_base_state(zf, thf, wts, zceiling)

    psfc = float(sfc["psfc_pa"])
    temp_grnd = fit["theta_grnd"] * (psfc / P0) ** FE_KAPPA

    ug, vg = float(geo["U_g"]), float(geo["V_g"])
    gspd = float(np.hypot(ug, vg))
    gdir = float((270.0 - np.degrees(np.arctan2(vg, ug))) % 360.0)

    zi = float(sfc.get("zi_used_m") or sfc.get("hpbl_m"))
    wthv = float(sfc["wth_virtual"])
    # THE RECEPTOR HEIGHT AND THE DOMAIN z0 ARE PROPERTIES OF THE GRID, NOT CONSTANTS.
    # Both were hardcoded at their 16 m values (10.0 m and 0.1435 m). At 24 m the receptor
    # is 30 m and the geometric-mean z0 of the bigger box is 0.0832 -- and every one of
    # z_i_min, the Obukhov label and the Ekman-angle choice is built on them, so carrying
    # the old numbers would have refused the wrong cases and picked the wrong seeds while
    # printing perfectly reasonable output. Read them off the grid the case will run on.
    zm_recept = float(z_recept if z_recept is not None else _grid_receptor(grid_dir))

    # WHAT THE BOX CAN HOLD, AS A RATIO. The constraint is L >= R z_i, and R is the
    # thing the evidence is about -- writing the cap as a height hides which experiment
    # licences it. Phase E measured the 10 m footprint indistinguishable between
    # L/z_i = 4.56 and 2.28 (p ~ 0.54), which licences R = 2 and a 976 m cap. Any smaller
    # R is a claim beyond Phase E and needs its own run of Phase E's experiment. A sounding above the cap is not a case this domain can represent,
    # and it must be REJECTED at selection time rather than run and quietly mis-labelled --
    # the excluded hours are also the strongest ones (z_i and heat flux correlate at +0.43,
    # and the excluded hours carry 1.51x the flux), so the exclusion is a stated bias, not
    # a neutral trim.
    zi_max = domain_l / zi_l_ratio
    # AND A FLOOR, which is the mirror of the cap and just as real. The receptor must sit
    # in the surface layer for any of this to mean anything, and the surface layer is the
    # bottom ~10% of the boundary layer -- so a 10 m receptor needs z_i >= 100 m. Below
    # that the receptor is a mixed-layer measurement, not a surface-layer one, MOST does
    # not apply, and the whole boundary layer spans about 25 model levels at dz_sfc = 4 m.
    # HRRR gives z_i = 36 m on 2023-10-05 12Z, so this is not hypothetical.
    zi_min = 10.0 * zm_recept
    # AND A SECOND CEILING, from the DOMAIN HEIGHT rather than its width. The width
    # constraint L >= R z_i is about horizontal lock-in; a boundary layer also has to fit
    # under the Rayleigh damping layer with room for the entrainment zone. At zCeiling
    # 3000 m with a 500 m damping layer the clean column ends at 2500 m, and a z_i above
    # about half of that is being held down by the sponge rather than by its inversion.
    # Whichever ceiling is lower is the one that binds, and which one it was is recorded.
    if zi_max_abs is not None:
        zi_max = min(zi_max, float(zi_max_abs))
    representable = bool(zi_min <= zi <= zi_max)
    # L from HRRR's own u* proxy is not available; use the surface-layer relation with the
    # 10 m wind and the domain z0 -- a LABEL for choosing the Ekman angle and the seed
    # rung, never a corpus input. The corpus input is the LES's achieved L.
    z0 = float(z0_dom if z0_dom is not None else _grid_z0(grid_dir))
    u10 = float(sfc.get("wspd10_ms") or np.nan)
    ust_est = VONK * u10 / np.log(10.0 / z0) if np.isfinite(u10) else np.nan
    th_s = fit["theta_grnd"]
    L_est = (-ust_est ** 3 * th_s / (VONK * G * wthv)
             if np.isfinite(ust_est) and abs(wthv) > 1e-6 else np.inf)
    zoL = zi / L_est if np.isfinite(L_est) and L_est != 0 else 0.0

    back = ekman_backing_deg(zoL)
    pred10 = (gdir - back) % 360.0
    rot = 0.0
    if match_10m and sfc.get("wdir10_from_deg") is not None:
        want = float(sfc["wdir10_from_deg"])
        rot = ((want - pred10 + 180.0) % 360.0) - 180.0
        a = np.radians(-rot)         # rotating the FROM-direction by +rot turns the
        ug, vg = (ug * np.cos(a) - vg * np.sin(a),   # vector by -rot
                  ug * np.sin(a) + vg * np.cos(a))
        gdir = (gdir + rot) % 360.0
        pred10 = (pred10 + rot) % 360.0

    fmean = flux_split(grid_dir) if grid_dir and os.path.isdir(grid_dir) else float("nan")
    wth_ref = wthv / fmean if np.isfinite(fmean) and fmean > 0 else wthv

    # Subsidence opposes entrainment at the inversion. The knee follows the case's own
    # z_i instead of the 500 m the static base.in carries; -25 m/h is the fair-weather
    # value docs/les/case-generation.md settled on (the kernel divides by 3600).
    lsf = dict(lsf_w_surf=0.0, lsf_w_lev1=-25.0, lsf_w_lev2=0.0,
               lsf_w_zlev1=float(round(max(zi, 200.0), 1)),
               lsf_w_zlev2=float(round(min(max(2.0 * zi, 1000.0), 2000.0), 1)))

    params = {
        "stabilityScheme": 2,
        "temp_grnd": round(float(temp_grnd), 4),
        "pres_grnd": round(psfc, 1),
        "zStableBottom": round(fit["zStableBottom"], 2),
        "stableGradient": float(f"{fit['stableGradient']:.6g}"),
        "zStableBottom2": round(fit["zStableBottom2"], 2),
        "stableGradient2": float(f"{fit['stableGradient2']:.6g}"),
        "zStableBottom3": round(fit["zStableBottom3"], 2),
        "stableGradient3": float(f"{fit['stableGradient3']:.6g}"),
        "U_g": round(float(ug), 6),
        "V_g": round(float(vg), 6),
        "z_Ug": 10000.0, "z_Vg": 10000.0, "Ug_grad": 0.0, "Vg_grad": 0.0,
        "surflayer_wth": float(f"{wthv:.6g}"),
        "surflayer_z0": z0,
        "coriolisLatitude": float(snd["tower"]["lat"]),
        "dt": float(f"{dt:.7g}"),
        **lsf,
    }
    # Constraint (3): guarantee, do not hope.
    bad = [k for k in ("stableGradient", "stableGradient2", "stableGradient3")
           if not (GRAD_FLOOR <= params[k] <= GRAD_CEIL)]
    if bad:
        raise ValueError(f"gradients out of FastEddy's range after rounding: {bad}")
    for k in ("zStableBottom", "zStableBottom2", "zStableBottom3"):
        if params[k] < 0.0:
            raise ValueError(f"{k} negative")
    if not (params["zStableBottom"] <= params["zStableBottom2"]
            <= params["zStableBottom3"]):
        raise ValueError("stable-layer bases are not ordered; the middle branch of "
                         "hydro_core.c:1786-1800 would be unreachable")

    warn = list(snd.get("warnings", []))
    if zi > zi_max:
        warn.append(
            f"z_i = {zi:.0f} m exceeds the {zi_max:.0f} m this {domain_l:.0f} m box "
            f"supports at L >= {zi_l_ratio:g} z_i. NOT a usable corpus case; skip the date or accept "
            f"a domain-constrained boundary layer and say so.")
    elif zi < zi_min:
        warn.append(
            f"z_i = {zi:.0f} m puts the {zm_recept:.0f} m receptor at "
            f"z/z_i = {zm_recept/max(zi,1e-6):.2f}, outside the surface layer. MOST does "
            f"not apply there and the library's shallowest rung is 150 m. NOT a usable "
            f"corpus case.")

    return {
        "provenance": dict(snd["provenance"], stage="sounding_to_forcing"),
        "representable": bool(representable),
        "params": params,
        "grid": {"nz": nz, "zceiling": zceiling, "verticalDeformFactor": c1,
                 "dx": dx, "dz_sfc": round(dz_sfc, 5),
                 "frqOutput_per_cadence": frq, "cadence_s": CADENCE,
                 "CFL_3d": round(cfl_ach, 5)},
        "fit": {"rms_k": round(fit["rms_k"], 4),
                "max_abs_k": round(fit["max_abs_k"], 4),
                "n_levels": int(zf.size),
                "weight_below_m": z_low, "weight_factor": w_low},
        "labels": {"zi_m": zi, "wth_virtual": wthv, "wth_sensible": sfc["wth_sensible"],
                   "bowen": sfc["bowen"], "wth_cropland_reference": float(wth_ref),
                   "flux_map_mean_factor": fmean,
                   "G_speed": round(gspd, 4), "G_dir_from_deg": round(gdir, 3),
                   "ekman_backing_deg": round(back, 2),
                   "predicted_10m_dir_deg": round(pred10, 3),
                   "hrrr_10m_dir_deg": sfc.get("wdir10_from_deg"),
                   # HRRR's own 10 m direction minus the Ekman prediction. NEGATIVE means
                   # the real surface wind is backed MORE than Ekman alone; POSITIVE means
                   # it is veered relative to the layer aloft, which Ekman cannot do and
                   # which is therefore baroclinic shear. Measured +19.3 deg on
                   # 2023-07-15 19Z, where the profile backs 9 deg with height through a
                   # z_i/L = -38 boundary layer -- so at this site the thermal wind can
                   # exceed the 10 deg convective Ekman angle outright. And +18.7 deg on
                   # 2023-01-18 18Z, a winter case with a quite different profile: the same
                   # SIGN twice is suggestive, but two cases are not a pattern, which is
                   # exactly why this is recorded per case rather than corrected for. If a
                   # corpus-wide bias is real it will be visible in ~1825 of these; if it
                   # is not, nothing has been baked in.
                   # The height-gradient geostrophic estimate is a DIAGNOSTIC, never the
                   # forcing -- but when it disagrees badly with the above-BL wind that is
                   # worth carrying per case rather than leaving in the sounding file. On
                   # 2023-01-18 18Z the two differ by 140 deg at a speed ratio of 1.01:
                   # a weak height gradient has a poorly determined DIRECTION, which is
                   # exactly the regime where a plane fit says least.
                   "geo_cross_check_deg": (
                       round(float(geo["cross_check"]["dir_diff_deg"]), 2)
                       if "cross_check" in geo else None),
                   "geo_cross_check_flag": (bool(geo["cross_check"]["flag"])
                                            if "cross_check" in geo else None),
                   "dir10_residual_deg": (
                       round(float(((float(sfc["wdir10_from_deg"]) - pred10 + 180.0)
                                    % 360.0) - 180.0), 3)
                       if sfc.get("wdir10_from_deg") is not None else None),
                   "forcing_rotation_deg": round(rot, 3),
                   "ustar_estimate": (round(ust_est, 4)
                                      if np.isfinite(ust_est) else None),
                   "L_estimate": (round(L_est, 2) if np.isfinite(L_est) else None),
                   "zi_over_L": round(float(zoL), 3),
                   "zi_max_m": round(zi_max, 1), "zi_min_m": round(zi_min, 1),
                   "zi_l_ratio": zi_l_ratio,
                   "domain_length_m": domain_l},
        "warnings": warn,
    }


def write_in(template, params, out, comments=None, stamp="set per case"):
    """Rewrite an existing .in in place-by-key.

    Keys absent from the template are an ERROR, not an append: a silently-added key would
    be a parameter FastEddy never reads, and it would look exactly like one it does.

    THE TEMPLATE'S COMMENT IS DISCARDED ON ANY KEY THAT CHANGES, not carried over. Carrying
    it over produced lines like `zStableBottom = 700.0  # mixed layer 0-400 m` and
    `dt = 0.01461988  # = 5/316 s` -- a stale explanation attached to a new number, which
    is worse than no explanation at all, and precisely the sort of comment this project has
    been bitten by trusting. Pass `comments` to say something true instead.
    """
    comments = comments or {}
    lines = open(template).read().splitlines(keepends=True)
    seen = set()
    for i, ln in enumerate(lines):
        s = ln.split("=", 1)
        if len(s) != 2:
            continue
        key = s[0].strip()
        if key in params:
            old_val = s[1].split("#", 1)[0].strip()
            new_val = f"{params[key]}"
            cmt = comments.get(key)
            if cmt is None:
                # unchanged values keep the template's comment; changed ones lose it
                tail = s[1].split("#", 1)
                cmt = (tail[1].strip() if len(tail) > 1 and old_val == new_val
                       else stamp)
            lines[i] = f"{key} = {new_val}  # {cmt}\n" if cmt else f"{key} = {new_val}\n"
            seen.add(key)
    missing = sorted(set(params) - seen)
    if missing:
        raise KeyError(f"template {template} has no line for: {missing}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sounding")
    ap.add_argument("--out", default=None, help="forcing JSON (default: alongside)")
    ap.add_argument("--in-out", default=None, help="also write a FastEddy .in here")
    ap.add_argument("--template", default="runs/g16_base/base.in")
    ap.add_argument("--grid", default="data/grid16")
    ap.add_argument("--nz", type=int, default=122)
    ap.add_argument("--nx", type=int, default=122)
    ap.add_argument("--zi-max-abs", type=float, default=None,
                    help="a second z_i ceiling, from the DOMAIN HEIGHT rather than its "
                         "width. The lower of the two binds and the record says which.")
    ap.add_argument("--zm", type=float, default=None,
                    help="receptor height for the representability screen; default is "
                         "read off <grid>/meta.npy")
    ap.add_argument("--z0", type=float, default=None,
                    help="domain geometric-mean z0 for the Obukhov label; default is "
                         "read off <grid>/z0m.npy")
    ap.add_argument("--zi-l-ratio", type=float, default=2.0,
                    help="the domain constraint L >= R z_i. R = 2 caps z_i at 976 m and is "
                         "what Phase E licences. Lowering R raises the cap and is a claim "
                         "BEYOND Phase E until its experiment is re-run at the new R.")
    ap.add_argument("--zceiling", type=float, default=2500.0)
    ap.add_argument("--deform", type=float, default=0.194059)
    ap.add_argument("--dx", type=float, default=16.0)
    ap.add_argument("--cfl", type=float, default=1.35)
    ap.add_argument("--weight-below", type=float, default=1500.0)
    ap.add_argument("--weight-factor", type=float, default=3.0)
    ap.add_argument("--match-10m", action="store_true",
                    help="rotate the forcing so the PREDICTED 10 m direction matches "
                         "HRRR's. Off by default: forcing with the real geostrophic wind "
                         "and letting the LES find its own Ekman turning over the real "
                         "Kegonsa roughness is the more faithful of the two.")
    a = ap.parse_args()

    snd = json.load(open(a.sounding))
    rec = build(snd, a.grid, a.nz, a.zceiling, a.deform, a.dx, a.cfl,
                a.match_10m, a.weight_factor, a.weight_below, a.nx,
                a.zi_l_ratio, z_recept=a.zm, z0_dom=a.z0,
                zi_max_abs=a.zi_max_abs)

    out = a.out or os.path.join("results/forcing",
                                os.path.basename(a.sounding).replace(".json", "") + ".json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(rec, open(out, "w"), indent=1, sort_keys=True)

    p, l, f, g = rec["params"], rec["labels"], rec["fit"], rec["grid"]
    print(out)
    print(f"  base state: theta_grnd {p['temp_grnd']:.3f} K at {p['pres_grnd']:.0f} Pa")
    print(f"    neutral to {p['zStableBottom']:.0f} m, then "
          f"{1000*p['stableGradient']:.2f} K/km to {p['zStableBottom2']:.0f} m, "
          f"{1000*p['stableGradient2']:.2f} to {p['zStableBottom3']:.0f} m, "
          f"{1000*p['stableGradient3']:.2f} above")
    print(f"    fit over {f['n_levels']} LES levels: rms {f['rms_k']:.3f} K, "
          f"max |resid| {f['max_abs_k']:.3f} K")
    print(f"  forcing: G = {l['G_speed']:.2f} m/s from {l['G_dir_from_deg']:.1f} deg "
          f"-> (U_g, V_g) = ({p['U_g']:.3f}, {p['V_g']:.3f})")
    print(f"    predicted 10 m dir {l['predicted_10m_dir_deg']:.1f} deg "
          f"(Ekman {l['ekman_backing_deg']:.1f}); HRRR says "
          f"{l['hrrr_10m_dir_deg']:.1f}; rotation applied {l['forcing_rotation_deg']:+.1f}")
    print(f"  flux: virtual {p['surflayer_wth']:.4f} K m/s domain mean "
          f"-> cropland reference {l['wth_cropland_reference']:.4f} "
          f"(map mean factor {l['flux_map_mean_factor']:.4f})")
    print(f"  z_i {l['zi_m']:.0f} m (usable {l['zi_min_m']:.0f}-{l['zi_max_m']:.0f} m "
          f"for a {l['domain_length_m']:.0f} m box), z_i/L {l['zi_over_L']:+.1f}, "
          f"subsidence knee {p['lsf_w_zlev1']:.0f} m")
    if not rec["representable"]:
        print("  NOT REPRESENTABLE: this sounding's boundary layer is deeper than the "
              "domain supports.")
    print(f"  dt {p['dt']:.7f} s, {g['frqOutput_per_cadence']} steps per {CADENCE:.0f} s "
          f"dump, CFL_3d {g['CFL_3d']:.4f}")
    for w in rec["warnings"]:
        print(f"  WARNING: {w}")

    if a.in_out:
        write_in(a.template, p, a.in_out)
        print(f"  wrote {a.in_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
