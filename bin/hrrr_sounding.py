#!/usr/bin/env python3
"""One HRRR pseudo-sounding at the Kegonsa tower, for one valid time.

WHY HRRR AND NOT CONUS404. CONUS404 hourly carries no time-varying atmospheric
profiles at all -- checked in the store's own .zmetadata, recorded in PROJECT_BRIEF.md -- so it
cannot force a per-case run however convenient it would be. HRRR supplies 3 km analyses on
~50 HYBRID levels, which is the part that matters: pressure-level products put 3-4 levels
in the whole boundary layer, and this project's receptor is at 10 m.

WHAT IT IS FOR. The output is the forcing for ONE LES case: the base-state theta profile,
the geostrophic wind, and the surface fluxes. It is never a boundary condition -- FastEddy
runs doubly periodic with constant forcing and each case is one quasi-stationary state.

THE AVERAGING CONVENTION IS PERIOD-ENDING, matching the tower's own record
(data/raw/H_and_sigma_w.csv runs 00:30 -> 00:00 next day, exactly 365*48 rows). A footprint
stamped 01:00 UTC is the average over 00:30-01:00 UTC, whose midpoint 00:45 is nearest the
01:00 analysis -- so THE ANALYSIS HOUR EQUALS THE FOOTPRINT TIMESTAMP.

TWO TRAPS, BOTH OF WHICH PRODUCE PLAUSIBLE WRONG NUMBERS RATHER THAN ERRORS:

  1. HRRR GRIB WINDS ARE GRID-RELATIVE, not earth-relative. On the Lambert grid at this
     longitude that is a ~5 degree rotation -- comparable to a whole direction bin in a
     12-direction corpus, and invisible in the wind SPEED, which is rotation-invariant.
     Rotated here with pyproj's meridian convergence rather than a hand-rolled Lambert
     formula. (bin/conus404_dist.py hits the identical issue and quotes 5.5 deg for
     CONUS404's grid; agreement to a few tenths is the cross-check.)

  2. The geostrophic wind needs a HORIZONTAL GRADIENT, so a single gridpoint cannot give
     it. A box of points is pulled and a plane is least-squares fitted to geopotential
     height. The wind just above the boundary layer is computed too, as an independent
     cross-check -- when the two disagree badly the case is flagged, not silently used.

usage: hrrr_sounding.py 2023-07-15T19:00 [--out FILE] [--box-km 45] [--level 850]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

# The surveyed tower coordinate lives in exactly one place in this repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

G = 9.80665
RD = 287.05
CP = 1004.5
P0 = 100000.0
KAPPA = RD / CP
LV = 2.501e6          # latent heat of vaporisation, J/kg
OMEGA = 7.2921159e-5

TOWER_LAT = 42.957160
TOWER_LON = -89.292362
FCOR = 2.0 * OMEGA * np.sin(np.radians(TOWER_LAT))     # 9.94e-5 s^-1


def _tower_from_prep_stage6():
    """Prefer the single source of truth if it is importable; fall back to the copy."""
    try:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prep_stage6.py")
        spec = importlib.util.spec_from_file_location("_ps6", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return float(m.TOWER_LAT), float(m.TOWER_LON)
    except Exception:
        return TOWER_LAT, TOWER_LON


def norm_lon(lon):
    """HRRR GRIB longitudes run 0..360. Everything else here is -180..180.

    Left unnormalised, the geostrophic box test `|lon - lon0| <= box` matches ZERO
    points and the code falls back to the above-BL proxy without saying so. That is
    how this was actually found.
    """
    return ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0


def crs_of(ds):
    """Projection from the CF grid-mapping coordinate cfgrib attaches.

    NOT ds.herbie.crs: that accessor imports metpy, which is a herbie-data[extras]
    dependency we do not carry, and it raises rather than returning None -- so the
    caller's `except: pass` left the rotation angle at exactly 0.0 with no warning
    and a ~5 degree direction error in every sounding.
    """
    from pyproj import CRS
    for name in ("gribfile_projection", "crs", "projection"):
        if name in getattr(ds, "coords", {}) or name in getattr(ds, "variables", {}):
            attrs = ds[name].attrs
            if attrs.get("grid_mapping_name") or attrs.get("crs_wkt"):
                return CRS.from_cf(attrs)
    raise KeyError("no CF grid-mapping coordinate on this dataset")


def meridian_convergence(crs, lon, lat):
    """Angle from grid north to true north, degrees, via pyproj's own map factors.

    Hand-rolling the Lambert formula n*(lon - lon0) is easy to get sign-wrong and the
    error is a few degrees -- large enough to matter, small enough to look fine.
    """
    from pyproj import Proj
    fac = Proj(crs).get_factors(lon, lat)
    return float(np.asarray(fac.meridian_convergence).ravel()[0])


def rotate_to_earth(u, v, gamma_deg):
    """Grid-relative -> earth-relative. gamma is the meridian convergence in degrees."""
    a = np.radians(gamma_deg)
    return u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a)


def theta_from(t_k, p_pa):
    return np.asarray(t_k) * (P0 / np.asarray(p_pa)) ** KAPPA


def zi_bulk_ri(z, th, u, v, rb_crit=0.25, zmax=4000.0):
    """Boundary-layer depth by the bulk Richardson criterion, the regime-independent one.

    Rb(z) = (g/th_s)(th(z) - th_s)(z - z_s) / (|U(z) - U_s|^2), first crossing of 0.25
    (Vogelezang & Holtslag 1996; Seibert et al. 2000), linearly interpolated between the
    bracketing levels. Unlike a theta-gradient pick it works in a STABLE layer, where
    there is no inversion to find, and in a CBL with no sharp cap, where the strongest
    gradient below 4 km is the free-troposphere lapse rate rather than the BL top.

    The run is dry and the profile carries no qv, so this is theta, not theta_v -- named,
    because at a summer Bowen ratio of 0.4 the difference is a few tenths of a Kelvin.
    """
    if z.size < 4:
        return float("nan")
    th_s, u_s, v_s, z_s = float(th[0]), float(u[0]), float(v[0]), float(z[0])
    du2 = (u - u_s) ** 2 + (v - v_s) ** 2
    rb = (G / th_s) * (th - th_s) * (z - z_s) / np.maximum(du2, 0.1)
    m = (z > z_s + 20.0) & (z < zmax)
    idx = np.nonzero(m & (rb >= rb_crit))[0]
    if idx.size == 0:
        return float("nan")
    k = int(idx[0])
    if k == 0:
        return float(z[0])
    r0, r1 = float(rb[k - 1]), float(rb[k])
    if not np.isfinite(r0) or r1 <= r0:
        return float(z[k])
    f = (rb_crit - r0) / (r1 - r0)
    return float(z[k - 1] + f * (z[k] - z[k - 1]))


def zi_parcel(z, th, d_theta=0.5, z_ml=200.0, zmax=4000.0):
    """Mixed-layer top: the first level whose theta exceeds the mixed-layer value by 0.5 K.

    The classic convective diagnostic, and the one that matches what a CBL looks like on
    this site's profiles -- well mixed to some height, then rising steadily with no cap.
    MEANINGLESS IN A STABLE LAYER (theta rises from the ground up, so it returns the first
    level), which is exactly why it is reported alongside zi_bulk_ri and never instead.
    """
    if z.size < 4:
        return float("nan")
    m = z <= max(z_ml, float(z[0]) + 1.0)
    th_ml = float(th[m].mean()) if m.sum() >= 1 else float(th[0])
    tgt = th_ml + d_theta
    idx = np.nonzero((z < zmax) & (th >= tgt))[0]
    idx = idx[idx > 0]
    if idx.size == 0:
        return float("nan")
    k = int(idx[0])
    t0, t1 = float(th[k - 1]), float(th[k])
    if t1 <= t0:
        return float(z[k])
    f = (tgt - t0) / (t1 - t0)
    return float(z[k - 1] + f * (z[k] - z[k - 1]))


def zi_max_gradient(z, th, zmax=4000.0):
    """Strongest theta gradient below zmax. RECORDED ONLY -- it is not a BL-depth estimator.

    Kept because it is the obvious thing to reach for and it is quietly wrong: on a summer
    profile with no capping inversion the free troposphere runs 3.8-4.9 K/km all the way
    up, so the maximum lands at 2041 m on a boundary layer HRRR itself puts at 1648 m and
    whose mixed layer ends near 1250 m. Deleting it would only invite it back.
    """
    m = (z > 50.0) & (z < zmax)
    if m.sum() < 4:
        return float("nan")
    zz, tt = z[m], th[m]
    return float(zz[int(np.argmax(np.gradient(tt, zz)))])


def _flatten(ds):
    """H.xarray() returns a Dataset or a list of them; give back a list."""
    return list(ds) if isinstance(ds, (list, tuple)) else [ds]


def _pick(ds, lat, lon, k=1):
    import pandas as pd
    pts = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
    return ds.herbie.pick_points(pts, method="nearest", k=k)


def fetch(ts, box_km, level_mb, save_dir=None, keep_grib=False, nlev=20):
    from herbie import Herbie

    out = {"provenance": {"model": "hrrr", "fxx": 0,
                          "valid_time": ts.strftime("%Y-%m-%dT%H:%M:%SZ")}}
    try:
        import herbie as _h
        out["provenance"]["herbie_version"] = getattr(_h, "__version__", "?")
    except Exception:
        pass

    lat0, lon0 = _tower_from_prep_stage6()
    out["tower"] = {"lat": lat0, "lon": lon0}

    # ---- native (hybrid) levels: the profile ------------------------------------
    # SPFH IS NOT REQUESTED, because nothing downstream reads it: the run is dry, the
    # profile carries only z/theta/u/v/p/T, and the ONE thing moisture would change --
    # buoyancy -- is absorbed by prescribing htFlux as the VIRTUAL flux (PROJECT_BRIEF.md). It is
    # a sixth of the download for a field that is thrown away.
    #
    # AND THE GRIB IS DELETED AFTER EXTRACTION unless --keep-grib. GRIB byte-range
    # subsetting works by MESSAGE, not by area, so 5 variables x ~50 hybrid levels is 300
    # full CONUS fields = ~263 MB per timestamp -- and this project wants ~1825 of them,
    # which is 470 GB of cache for 15 MB of JSON. Measured, not estimated: the first six
    # cached `nat` subsets averaged 315 MB each. The durable artifact is the 8 kB sounding.
    # ONLY THE LOWEST `nlev` HYBRID LEVELS. Verified against the file's own inventory
    # rather than assumed: HRRR numbers hybrid level 1 at the MODEL BOTTOM (level 1 sits at
    # 289 m ASL over this tower, level 20 at 6413 m, level 50 at 27176 m). Levels 1-20
    # therefore reach ~6.1 km AGL, which contains everything downstream needs -- the
    # 2500 m LES column, the 4 km ceiling on the z_i searches, and the above-BL
    # geostrophic layer, which tops out at z_i + 550 <= 1526 m for the deepest
    # representable case. The remaining 30 levels are stratosphere the LES never sees.
    #
    # This is 40% of the messages, and it is the corpus's largest data cost: 228 MB per
    # case becomes ~91 MB, and 1825 cases go from ~416 GB of transfer to ~166 GB.
    lv = "|".join(str(i) for i in range(1, int(nlev) + 1))
    Hn = Herbie(ts, model="hrrr", product="nat", fxx=0, save_dir=save_dir)
    dsn = _flatten(Hn.xarray(rf":(?:TMP|UGRD|VGRD|HGT|PRES):(?:{lv}) hybrid level:",
                             remove_grib=not keep_grib))
    prof = {}
    crs = None
    for d in dsn:
        if crs is None:
            try:
                crs = crs_of(d)
            except Exception as e:
                out.setdefault("warnings", []).append(f"CRS from native levels: {e}")
        pt = _pick(d, lat0, lon0)
        for name in ("t", "u", "v", "gh", "pres", "q", "unknown"):
            if name in pt.variables:
                prof[name] = np.asarray(pt[name].values).ravel()
        # cfgrib sometimes names them by their GRIB shortName instead
        for name in ("TMP", "UGRD", "VGRD", "HGT", "PRES"):
            if name in pt.variables:
                prof[name.lower()] = np.asarray(pt[name].values).ravel()
        if "hybrid" in pt.coords:
            prof["hybrid"] = np.asarray(pt["hybrid"].values).ravel()
        for c in ("latitude", "longitude"):
            if c in pt.coords:
                out.setdefault("grid", {})[c] = float(np.asarray(pt[c].values).ravel()[0])

    # ---- surface fields ---------------------------------------------------------
    Hs = Herbie(ts, model="hrrr", product="sfc", fxx=0, save_dir=save_dir)
    sfc = {}
    for pat, key in ((r":HPBL:surface:", "hpbl"),
                     (r":SHTFL:surface:", "shtfl"),
                     (r":LHTFL:surface:", "lhtfl"),
                     (r":PRES:surface:", "psfc"),
                     (r":HGT:surface:", "zsfc"),
                     (r":TMP:2 m above ground:", "t2m"),
                     (r":UGRD:10 m above ground:", "u10"),
                     (r":VGRD:10 m above ground:", "v10")):
        try:
            for d in _flatten(Hs.xarray(pat, remove_grib=not keep_grib)):
                pt = _pick(d, lat0, lon0)
                for vn in pt.data_vars:
                    val = np.asarray(pt[vn].values).ravel()
                    if val.size:
                        sfc[key] = float(val[0])
                        break
                if key in sfc:
                    break
        except Exception as e:            # a missing optional field is not fatal
            sfc.setdefault(key + "_error", str(e))
    out["surface_raw"] = sfc

    # ---- pressure level box: the geostrophic gradient ---------------------------
    Hp = Herbie(ts, model="hrrr", product="prs", fxx=0, save_dir=save_dir)
    box = {}
    try:
        for d in _flatten(Hp.xarray(rf":HGT:{level_mb} mb:",
                                    remove_grib=not keep_grib)):
            lat = np.asarray(d["latitude"].values)
            lon = norm_lon(d["longitude"].values)
            gh = np.asarray(d["gh"].values) if "gh" in d.variables else \
                 np.asarray(d[list(d.data_vars)[0]].values)
            # crude great-circle distances are fine over a 45 km box
            dx = (lon - lon0) * 111320.0 * np.cos(np.radians(lat0))
            dy = (lat - lat0) * 110540.0
            m = (np.abs(dx) <= box_km * 1000) & (np.abs(dy) <= box_km * 1000)
            if m.sum() >= 12:
                box = {"dx": dx[m], "dy": dy[m], "gh": gh[m], "n": int(m.sum())}
            else:
                out.setdefault("warnings", []).append(
                    f"geostrophic box empty: {int(m.sum())} points within {box_km} km "
                    f"(lon range {lon.min():.1f}..{lon.max():.1f}) -- falling back to the "
                    f"above-BL proxy")
            if crs is None:
                try:
                    crs = crs_of(d)
                except Exception as e:
                    out.setdefault("warnings", []).append(f"CRS from prs: {e}")
            break
    except Exception as e:
        out.setdefault("warnings", []).append(f"prs HGT box failed: {e}")
    out["_box"] = box
    out["_crs"] = crs
    out["_prof"] = prof
    return out


def build(raw, level_mb):
    """Turn the raw pulls into the sounding record the forcing stage consumes."""
    prof, sfc, box = raw["_prof"], raw["surface_raw"], raw["_box"]
    lat0, lon0 = raw["tower"]["lat"], raw["tower"]["lon"]

    def need(*names):
        for n in names:
            if n in prof and np.size(prof[n]):
                return np.asarray(prof[n], float)
        raise KeyError(f"none of {names} in the native-level pull; got {sorted(prof)}")

    t = need("t", "tmp")
    p = need("pres")
    gh = need("gh", "hgt")
    ug = need("u", "ugrd")
    vg = need("v", "vgrd")

    gamma = 0.0
    if raw["_crs"] is None:
        raw.setdefault("warnings", []).append(
            "NO PROJECTION: winds left grid-relative. At this longitude that is a ~5 deg "
            "direction error, which is comparable to a direction bin and invisible in the "
            "wind speed. Do not use this sounding for a production case.")
    else:
        try:
            gamma = meridian_convergence(raw["_crs"], lon0, lat0)
        except Exception as e:
            raw.setdefault("warnings", []).append(f"meridian convergence failed: {e}")
    # sin(38.5 deg) * (lon - 262.5) = 5.11 deg here; anything far from that is wrong.
    if abs(gamma) > 30.0:
        raw.setdefault("warnings", []).append(
            f"meridian convergence {gamma:.2f} deg is implausible for this grid")
    u, v = rotate_to_earth(ug, vg, gamma)

    zsfc = float(sfc.get("zsfc", 0.0))
    z = gh - zsfc                                   # height above the model surface
    th = theta_from(t, p)

    order = np.argsort(z)                           # HRRR hybrid levels run top-down
    z, th, u, v, p, t = (a[order] for a in (z, th, u, v, p, t))
    keep = np.isfinite(z) & np.isfinite(th) & np.isfinite(u) & np.isfinite(v) & (z > -50)
    z, th, u, v, p, t = (a[keep] for a in (z, th, u, v, p, t))
    if z.size < 10:
        raise ValueError(f"only {z.size} usable levels in the profile")

    # ---- geostrophic wind, two independent ways ---------------------------------
    geo = {"meridian_convergence_deg": gamma, "f": FCOR}
    if box:
        A = np.column_stack([box["dx"], box["dy"], np.ones(box["n"])])
        coef, *_ = np.linalg.lstsq(A, box["gh"], rcond=None)
        dzdx, dzdy = float(coef[0]), float(coef[1])
        # the plane is fitted in EARTH axes (dx east, dy north), so no rotation here
        ugeo = -(G / FCOR) * dzdy
        vgeo = +(G / FCOR) * dzdx
        geo["gradient"] = {"level_mb": level_mb, "dzdx": dzdx, "dzdy": dzdy,
                           "n_points": box["n"],
                           "u": ugeo, "v": vgeo,
                           "speed": float(np.hypot(ugeo, vgeo)),
                           "dir_from_deg": float((270.0 - np.degrees(
                               np.arctan2(vgeo, ugeo))) % 360.0)}

    # ---- boundary-layer depth: FOUR diagnostics, one of them primary ------------
    # HPBL is primary because it is HRRR's OWN PBL-scheme depth, so it is the same
    # diagnostic in every one of the ~1825 cases -- consistency across the corpus beats
    # any per-case improvement here. The profile-based estimates are the cross-check, and
    # a large disagreement is a WARNING on the case rather than a silent substitution.
    # The LES will measure a fifth thing again (cbl_check.py takes the height of the
    # MINIMUM buoyancy flux), and the corpus label comes from THAT, not from any of these.
    zi_hp = float(sfc.get("hpbl", np.nan))
    zi_ri = zi_bulk_ri(z, th, u, v)
    zi_pa = zi_parcel(z, th)
    zi_mg = zi_max_gradient(z, th)
    zi = zi_hp
    if not np.isfinite(zi):
        zi = zi_ri if np.isfinite(zi_ri) else zi_pa
        raw.setdefault("warnings", []).append(
            f"HPBL missing; z_i fell back to the profile estimate {zi:.0f} m")
    zi_ref = zi_ri if np.isfinite(zi_ri) else zi_pa
    if np.isfinite(zi) and np.isfinite(zi_ref) and zi > 0:
        if abs(zi_ref - zi) / zi > 0.5:
            raw.setdefault("warnings", []).append(
                f"z_i diagnostics disagree by more than 50%: HPBL {zi:.0f} m vs "
                f"bulk-Ri/parcel {zi_ref:.0f} m. Check the profile before using this case.")
    if np.isfinite(zi):
        # A HEIGHT average, not a level average. HRRR's hybrid levels thin out with
        # height -- above a 1648 m boundary layer the whole 600 m slab held exactly ONE
        # level, so a level-mean was a single sample taken wherever that level happened
        # to fall, in a layer where the direction wobbles by ~10 deg. Interpolating onto
        # a uniform height grid first makes the estimate independent of where the model
        # put its levels, which is the only reason the number is stable case to case.
        lo, hi = zi + 50.0, min(zi + 550.0, float(z[-1]))
        if hi > lo:
            zs = np.linspace(lo, hi, 25)
            ua = float(np.interp(zs, z, u).mean())
            va = float(np.interp(zs, z, v).mean())
            # shear ACROSS the averaging layer, reported so a badly sheared case is
            # visible rather than averaged into a plausible-looking single vector
            d_lo = (270.0 - np.degrees(np.arctan2(np.interp(lo, z, v),
                                                  np.interp(lo, z, u)))) % 360.0
            d_hi = (270.0 - np.degrees(np.arctan2(np.interp(hi, z, v),
                                                  np.interp(hi, z, u)))) % 360.0
            geo["above_bl"] = {
                "z_lo": float(lo), "z_hi": float(hi),
                "n_levels": int(((z >= lo) & (z <= hi)).sum()),
                "n_samples": 25, "u": ua, "v": va,
                "speed": float(np.hypot(ua, va)),
                "dir_from_deg": float((270.0 - np.degrees(np.arctan2(va, ua))) % 360.0),
                "dir_shear_deg": float(abs(((d_hi - d_lo + 180.0) % 360.0) - 180.0))}
            if geo["above_bl"]["dir_shear_deg"] > 30.0:
                raw.setdefault("warnings", []).append(
                    f"the above-BL layer turns {geo['above_bl']['dir_shear_deg']:.0f} deg "
                    f"across {hi-lo:.0f} m; the geostrophic proxy is a poor summary of it")

    # WHICH ONE FORCES THE RUN, and why -- decided, not defaulted.
    #
    # `above_bl` is primary. FastEddy runs DOUBLY PERIODIC on a 1952 m box, so it can
    # represent neither synoptic curvature nor a horizontal height gradient; forcing it
    # with the height-gradient geostrophic wind would drive a boundary layer much
    # stronger than the one HRRR actually has. Measured on 2023-07-15 19Z: the actual
    # wind is 6.2 m/s where the 850 mb gradient says 10.7, and the profile shows exactly
    # why -- 850 mb sits INSIDE the BL (z_i = 1648 m), so its wind is sub-geostrophic and
    # backed 28 deg, which is Ekman balance behaving correctly. Above the BL the two
    # DIRECTIONS agree to a few degrees; the residual speed gap is flow curvature.
    #
    # `gradient` is kept as a recorded diagnostic -- it carries the baroclinicity and
    # curvature information -- and the disagreement is reported rather than hidden.
    if "above_bl" in geo:
        geo["chosen"] = "above_bl"
    elif "gradient" in geo and geo["gradient"]["speed"] < 60.0:
        geo["chosen"] = "gradient"
    else:
        raise ValueError("no usable geostrophic wind estimate")
    ch = geo[geo["chosen"]]
    geo["U_g"], geo["V_g"] = ch["u"], ch["v"]
    geo["speed"], geo["dir_from_deg"] = ch["speed"], ch["dir_from_deg"]
    if "gradient" in geo and "above_bl" in geo:
        d = abs(((geo["gradient"]["dir_from_deg"] - geo["above_bl"]["dir_from_deg"]
                  + 180) % 360) - 180)
        geo["cross_check"] = {
            "dir_diff_deg": float(d),
            "speed_ratio": float(geo["gradient"]["speed"]
                                 / max(geo["above_bl"]["speed"], 0.1)),
            "flag": bool(d > 45.0)}

    # ---- surface fluxes: sign convention and the per-case Bowen ratio -----------
    # HRRR reports SHTFL/LHTFL positive UPWARD out of the surface in the analysis
    # files this project uses; that is already the sign the LES wants for w'th'.
    # The 10 m wind is grid-relative like every other GRIB wind and needs the same
    # rotation. It is a cross-check on the forcing, so an unrotated copy would be worse
    # than none at all.
    u10, v10 = rotate_to_earth(float(sfc.get("u10", np.nan)),
                               float(sfc.get("v10", np.nan)), gamma)
    sh = float(sfc.get("shtfl", np.nan))
    lh = float(sfc.get("lhtfl", np.nan))
    rho_s = float(sfc.get("psfc", 96000.0)) / (RD * float(sfc.get("t2m", 288.0)))
    wth = sh / (rho_s * CP) if np.isfinite(sh) else np.nan
    bowen = sh / lh if (np.isfinite(sh) and np.isfinite(lh) and abs(lh) > 1.0) else np.nan
    # Virtual, because the run is DRY and buoyancy is what htFlux is for (PROJECT_BRIEF.md).
    # With a per-case Bowen ratio this conversion is exact instead of class-table-derived.
    th_s = float(sfc.get("t2m", 288.0)) * (P0 / float(sfc.get("psfc", 96000.0))) ** KAPPA
    # The conversion needs a POSITIVE Bowen ratio. At night SHTFL < 0 and LHTFL is small
    # or negative (dew), so the ratio is negative or near-zero and w'th'(1 + 0.0735/B)
    # would flip sign or blow up -- a stable case would come back convective. The
    # conversion is skipped there and the skip is RECORDED, because "virtual = sensible"
    # is an approximation, not an identity: with dew forming, w'q' < 0 makes the true
    # virtual flux slightly MORE negative than the sensible one, so the stability is
    # marginally understated. That is a few percent of a nocturnal flux and it is the
    # conservative direction; it is written down rather than left to be rediscovered.
    if np.isfinite(bowen) and bowen > 0.02:
        wthv = wth * (1.0 + 0.61 * th_s * CP / (bowen * LV))
    else:
        wthv = wth
        raw.setdefault("warnings", []).append(
            f"Bowen ratio {bowen:.3f} is not positive (H={sh:.1f}, LE={lh:.1f} W/m2); the "
            f"sensible-to-virtual conversion was SKIPPED and w'th_v' = w'th' = "
            f"{wth:+.4f} K m/s. Typical at night; the true virtual flux is slightly more "
            f"negative, so the stability is marginally understated.")
    return {
        "provenance": raw["provenance"],
        "tower": raw["tower"],
        "grid": raw.get("grid", {}),
        "surface": {"zsfc_m": zsfc, "psfc_pa": sfc.get("psfc"), "t2m_k": sfc.get("t2m"),
                    "rho": rho_s, "shtfl_wm2": sh, "lhtfl_wm2": lh, "bowen": bowen,
                    "wth_sensible": wth, "wth_virtual": wthv,
                    "hpbl_m": sfc.get("hpbl"),
                    "zi_bulk_ri_m": zi_ri, "zi_parcel_m": zi_pa,
                    "zi_max_gradient_m": zi_mg, "zi_used_m": zi,
                    "u10_ms": u10, "v10_ms": v10,
                    "wspd10_ms": (float(np.hypot(u10, v10))
                                  if np.isfinite(u10) and np.isfinite(v10) else None),
                    "wdir10_from_deg": (float((270.0 - np.degrees(np.arctan2(v10, u10)))
                                              % 360.0)
                                        if np.isfinite(u10) and np.isfinite(v10)
                                        else None)},
        "geostrophic": geo,
        "profile": {"z_agl_m": z.tolist(), "theta_k": th.tolist(),
                    "u_ms": u.tolist(), "v_ms": v.tolist(),
                    "pres_pa": p.tolist(), "t_k": t.tolist()},
        "warnings": raw.get("warnings", []),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("timestamp", help="analysis valid time, UTC, e.g. 2023-07-15T19:00")
    ap.add_argument("--out", default=None)
    ap.add_argument("--box-km", type=float, default=45.0)
    ap.add_argument("--level", type=int, default=850, help="pressure level (mb) for dZ/dx")
    ap.add_argument("--cache", default="data/hrrr")
    ap.add_argument("--nlev", type=int, default=20,
                    help="how many hybrid levels from the MODEL BOTTOM to fetch. 20 "
                         "reaches ~6.1 km AGL and is 40%% of the download; the rest is "
                         "stratosphere the LES never sees.")
    ap.add_argument("--keep-grib", action="store_true",
                    help="keep the downloaded GRIB. OFF by default: GRIB byte-range "
                         "subsetting works per MESSAGE, not per area, so a `nat` subset "
                         "is ~263 MB (measured 315 MB with SPFH) and a 1825-case corpus "
                         "would cache ~470 GB for 15 MB of JSON.")
    a = ap.parse_args()

    ts = dt.datetime.fromisoformat(a.timestamp.replace("Z", ""))
    os.makedirs(a.cache, exist_ok=True)
    raw = fetch(ts, a.box_km, a.level, save_dir=a.cache, keep_grib=a.keep_grib,
                nlev=a.nlev)
    rec = build(raw, a.level)

    out = a.out or os.path.join("results", "soundings",
                                ts.strftime("%Y%m%d%H") + ".json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)

    # ASSERT ON THE ARTIFACT, not on the exit status (docs/FASTEDDY_TRAPS.md 12).
    assert os.path.getsize(out) > 512, f"{out} is suspiciously small"
    g, s = rec["geostrophic"], rec["surface"]
    print(f"{out}")
    print(f"  levels {len(rec['profile']['z_agl_m'])}, "
          f"z {rec['profile']['z_agl_m'][0]:.1f} -> {rec['profile']['z_agl_m'][-1]:.0f} m AGL")
    print(f"  G = {g['speed']:.2f} m/s from {g['dir_from_deg']:.1f} deg "
          f"[{g['chosen']}], convergence {g['meridian_convergence_deg']:+.2f} deg")
    if "cross_check" in g:
        c = g["cross_check"]
        print(f"  cross-check: dir diff {c['dir_diff_deg']:.1f} deg, "
              f"speed ratio {c['speed_ratio']:.2f}"
              + ("   <-- FLAGGED" if c["flag"] else ""))
    print(f"  z_i: HPBL {s['hpbl_m']:.0f} m [used], bulk-Ri {s['zi_bulk_ri_m']:.0f} m, "
          f"parcel {s['zi_parcel_m']:.0f} m, max-grad {s['zi_max_gradient_m']:.0f} m")
    if s.get("wspd10_ms"):
        print(f"  10 m wind {s['wspd10_ms']:.2f} m/s from {s['wdir10_from_deg']:.1f} deg")
    print(f"  H {s['shtfl_wm2']:.1f} W/m2, "
          f"Bowen {s['bowen']:.2f}, w'th' {s['wth_sensible']:.4f} -> "
          f"virtual {s['wth_virtual']:.4f} K m/s")
    for w in rec["warnings"]:
        print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
