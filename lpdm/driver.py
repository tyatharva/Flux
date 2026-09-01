"""Release / rotate / accumulate: turn a FieldSet into a wind-aligned flux footprint."""
from __future__ import annotations

import time

import numpy as np

from .footprint import FootprintGrid
from .les_stats import window_stats
from .model import LPDM
from .sgs_floor import (FSGS_AT_PEAK_MIN, check_monotone, floor_health,
                        most_floor)


def receptor_indices(fs, z_target=10.0, x_frac=0.75, y_frac=0.5, ij=None,
                     exact_agl=False):
    """Receptor cell. `ij` pins it to a specific column and overrides the fractions.

    The fractions date from the wind-ALIGNED elongated domain, where the tower sat 3/4
    along x to buy upwind fetch. The static domain centres the tower instead, so the
    fractions would put the receptor 1128 m east of it -- harmless over flat uniform
    ground, and completely wrong over real geography, where it would sample the wrong
    surface entirely. Pass the tower cell from data/grid/meta.npy.

    `exact_agl` returns a FRACTIONAL level whose height above the local ground is exactly
    z_target, instead of the nearest cell centre. The production grid puts a cell centre
    at 10.000000 m so the two agree exactly there and this costs nothing -- but the
    instrument is 10 m above BARE GROUND, and a surface built with --raise-topo lifts the
    model ground over the array by the displacement height. Snapping to the nearest level
    there would put the receptor 10 m above the PANELS, i.e. 11.5 m above bare ground,
    which is a 15% error in exactly the quantity this pass exists to get right.
    """
    if ij is not None:
        i, j = int(ij[0]), int(ij[1])
    else:
        i = int(round(x_frac * fs.nx))
        j = int(round(y_frac * fs.ny))
    if exact_agl:
        one = lambda v: np.array([float(v)])
        zg = float(fs.ground(one(i), one(j))[0])
        return i, j, float(fs.kindex(one(zg + z_target), one(i), one(j))[0])
    return i, j, int(np.argmin(np.abs(fs.zk - z_target)))


def make_releases(fs, n_per_release, t_first, t_last, dt_release, xr, yr, zr):
    times = np.arange(t_first, t_last + 1e-9, dt_release)
    t = np.repeat(times, n_per_release)
    n = len(t)
    return (np.full(n, xr), np.full(n, yr), np.full(n, zr), t, times)


def compute_footprint(fs, paths, z_target=10.0, n_per_release=700, dt_release=4.0,
                      t_back=900.0, c0=3.0, z_touch=2.0, grid_res=20.0,
                      grid_x=(-600.0, 4500.0), grid_y=(-1500.0, 1500.0),
                      seed=0, split_halves=True, batch_releases=12, w_floor=0.02,
                      max_disp=None, cover=None, aniso=None, sgs_scale=1.0, sgs_most=False,
                      receptor_ij=None, tback_marks=(), rel_seconds=None, sgs_most_mode="surface",
                      exact_agl=False, n_cover_groups=2, sgs_most_legacy=False,
                      sgs_most_form="multiplicative", sgs_subgrid_weight=True,
                      sgs_eps_consistent=True, require_rel_seconds=False,
                      keep_touchdowns=0, verbose=True, stats=None):
    """Release, integrate backward, accumulate on the STATIC north-up raster.

    THE RASTER IS THE LES GRID. Touchdowns are binned by their LES column index, folded
    modulo the periodic domain, so raster cell (j,i) is LES column (j,i) and nothing is
    ever resampled or rotated. That matters twice over: the land-cover masks and the
    roughness map live on those indices, and this is the raster the emulator consumes, so
    the target it trains on is the array the estimator actually produced.

    The wind frame has not been abandoned -- it is where Kljun lives and where "upwind
    distance" means anything -- but it is now a 1-D histogram of the touchdowns' upwind
    coordinate, accumulated from the touchdowns themselves. That is exact, whereas
    rotating a finished raster blurs precisely the near field the peak sits in.

    Releases are processed in batches of `batch_releases` release times rather than as one
    ensemble. The field cache is several GB and every integrator step does scattered
    4-D interpolation into it; a batch whose release times span ~1 minute touches a narrow
    slab of the time axis and stays resident, while one ensemble spanning the whole window
    touches all of it on every step. Same answer, far less memory traffic.
    """
    i_r, j_r, k_r = receptor_indices(fs, z_target, ij=receptor_ij, exact_agl=exact_agl)
    xr = fs.x0 + i_r * fs.dx
    yr = fs.y0 + j_r * fs.dy
    # ABSOLUTE height of that cell centre. Over terrain this is NOT fs.zk[k_r]: the
    # vertical coordinate is terrain-following, so the level sits at zg + ~z_target above
    # the local ground. Releasing at the flat-column height would put the receptor 20 m
    # underground on the hill.
    zr = float(fs.height(np.array([float(k_r)]), np.array([float(i_r)]),
                         np.array([float(j_r)]))[0])
    zg_r = float(fs.ground(np.array([float(i_r)]), np.array([float(j_r)]))[0])
    d_r = float(fs.displacement(np.array([float(i_r)]), np.array([float(j_r)]))[0])
    # THE WINDOW STATISTICS, EITHER RE-READ HERE OR HANDED IN ALREADY ACCUMULATED.
    #
    # Re-reading is the original behaviour and stays the default: with netCDF handles it
    # costs a second open per dump and nothing else. It is not available at all when the
    # window was STREAMED -- the snapshots were released as they were consumed, so there is
    # nothing left to re-read -- so `stats` lets the caller hand in the dict that
    # `WindowAccumulator` built during the single pass. Same arithmetic, same order,
    # bit-identical (bin/test_streaming.py).
    #
    # The receptor level is asserted rather than assumed: an accumulator built at a
    # different k than this function just derived would produce a plausible footprint from
    # the wrong level's turbulence, which is precisely the failure mode this project keeps
    # paying for.
    if stats is None:
        st = window_stats(paths, k_r)
    else:
        st = stats
        k_given = float(st.get("k_recept", float("nan")))
        if not (abs(k_given - float(k_r)) < 1e-9):
            raise ValueError(
                f"the window statistics handed in were accumulated at receptor level "
                f"k = {k_given}, but this window's geometry puts the receptor at "
                f"k = {float(k_r)}. Those are different heights and the difference would "
                f"be invisible in the output.")
    # Effective aerodynamic height: what every similarity relation, and Kljun, actually
    # take as z_m. Over the flat control d is ~0.1 m and this is a 1% correction; over the
    # array at a 10 m receptor it is 15%.
    st["d_recept"] = d_r
    st["z_eff"] = (zr - zg_r) - d_r

    # A window is (averaging period + t_back) long: the first t_back seconds of it produce
    # no releases at all, because a backward trajectory needs that much history behind it.
    # `rel_seconds` then holds the RELEASE period to exactly the averaging period an eddy
    # covariance system reports -- 30 minutes -- rather than letting it grow with whatever
    # window happened to be run. Without it a 45-minute window would silently produce a
    # 45-minute footprint and be compared against 30-minute observations.
    t_last = float(fs.t[-1])
    t_first = float(fs.t[0]) + t_back
    if rel_seconds is not None:
        t_first = max(t_first, t_last - float(rel_seconds))
    if t_last <= t_first:
        raise ValueError(f"window too short: need > {t_back:.0f} s of history before the "
                         f"first release (have {fs.t[-1]-fs.t[0]:.0f} s)")
    # A SHORT WINDOW DOES NOT FAIL, IT SILENTLY SHORTENS THE AVERAGING PERIOD, and the
    # production configuration sits at exactly zero margin. With a 4200 s case the two
    # terms of the max() above are EQUAL to the microsecond -- fs.t[0] + t_back = 1800 +
    # 600 = 2400 and t_last - rel_seconds = 4200 - 1800 = 2400 -- so anything that shortens
    # the head by one dump moves t_first up and produces a 29-minute footprint that is then
    # compared against 30-minute observations, with nothing in the output to say so. This
    # is the only place that can see it.
    if rel_seconds is not None:
        got = t_last - t_first
        # THE TOLERANCE MUST BE THE SCALE OF THE FAILURE, NOT MACHINE EPSILON. What this
        # guard exists to catch is losing a DUMP off the head of the window -- 5 s at the
        # production cadence. What it must not trip on is the rounding slack in dt: the
        # .in carries dt to 8 decimals (0.01461988), so frqOutput*dt is 5 s only to
        # 1.04e-6, and over 840 dumps that accumulates to a 4.99e-4 s shortfall in a
        # release period that is otherwise EXACTLY 1800 s. A 1e-6 tolerance therefore
        # failed the production configuration on half a millisecond -- after 74 minutes of
        # GPU, at stage 7, with the fields already on disk. One lost dump exceeds that
        # deficit by a factor of 10,016, so a tolerance of one tenth of an output interval
        # separates the two cleanly and leaves the guard as sharp as it was meant to be.
        cad = float(np.median(np.diff(fs.t))) if len(fs.t) > 2 else 0.0
        tol = max(1e-6, 0.1 * cad)
        if got < float(rel_seconds) - tol:
            msg = (
                f"the release period came out {got:.1f} s, not the {float(rel_seconds):.0f} s "
                f"asked for: the window has only {fs.t[0]:.1f}-{t_last:.1f} s of fields and "
                f"t_back is {t_back:.0f} s, so releases cannot start before "
                f"{fs.t[0]+t_back:.1f} s. Lengthen the window or shorten t_back -- do NOT "
                f"accept a short averaging period, it is compared against 30-minute "
                f"eddy-covariance observations. (Scored against a {tol:.4g} s tolerance, "
                f"one tenth of the {cad:.3g} s output interval -- this is a missing dump, "
                f"not rounding.)")
            # STRICT FOR THE CORPUS, LOUD EVERYWHERE ELSE. Five retired fourth-pass
            # windows legitimately released for 900 s against the 1800 s default, and
            # raising on those would break drivers that are the historical record rather
            # than production. What must never happen again is a SHORT averaging period
            # going unremarked, so the warning is unmissable and bin/run_corpus_case.sh
            # passes --strict-rel.
            if require_rel_seconds:
                raise ValueError(msg)
            print("\n  *** WARNING: " + msg + "\n")
        elif verbose:
            # REPORT THE MARGIN ON SUCCESS TOO. A guard that only speaks when it fires
            # leaves no evidence of how close the run came, and this configuration is
            # designed to sit at zero margin.
            print(f"  release period {got:.4f} s against {float(rel_seconds):.0f} s asked "
                  f"for (margin {got - float(rel_seconds):+.2e} s, tolerance {tol:.4g} s "
                  f"= 0.1 x the {cad:.3g} s output interval)")
    times = np.arange(t_first, t_last + 1e-9, dt_release)
    tmid = 0.5 * (times[0] + times[-1])
    if verbose:
        print(f"  receptor  (i,j,k)=({i_r},{j_r},"
              f"{k_r:.4f}" + (" fractional" if not float(k_r).is_integer() else "") +
              f")  x={xr:.0f} y={yr:.0f}  "
              f"z={zr:.3f} m ASL = {zr-zg_r:.3f} m AGL (ground {zg_r:.2f} m)")
        print(f"  displacement height at the receptor d={d_r:.3f} m  ->  effective "
              f"aerodynamic height z-d = {st['z_eff']:.3f} m")
        print(f"  releases  {len(times)} times over {t_first:.0f}-{t_last:.0f} s, "
              f"{n_per_release} each = {len(times)*n_per_release:,} particles; "
              f"t_back={t_back:.0f} s")

    ang = np.arctan2(st["V"], st["U"])
    ca, sa = np.cos(ang), np.sin(ang)
    # STATIC north-up raster on the LES columns, tower-centred. Cell k spans LES
    # fractional index [k-0.5, k+0.5), which is exactly one LES column.
    xe_m = (np.arange(fs.nx + 1) - 0.5 - i_r) * fs.dx
    ye_m = (np.arange(fs.ny + 1) - 0.5 - j_r) * fs.dy
    mkgrid = lambda: FootprintGrid.from_edges(xe_m, ye_m)
    full, h1, h2 = mkgrid(), mkgrid(), mkgrid()
    # 1-D wind-frame crosswind-integrated footprint, at the same 24 m resolution as the
    # raster. Bins span the full wrap cap in both directions so nothing falls off the end.
    nfy = int(np.ceil(fs.Lx / fs.dx))
    fy_e = (np.arange(-nfy, nfy + 1) + 0.5) * fs.dx
    fy_c = 0.5 * (fy_e[:-1] + fy_e[1:])
    fy_h = np.zeros(len(fy_c)); fy_h1 = np.zeros(len(fy_c)); fy_h2 = np.zeros(len(fy_c))
    # Capture-vs-t_back curve: the same estimator truncated at a shorter backward time.
    # Free -- it is a mask on the touchdown ages already in hand -- and it is what sizes
    # every production window, since a window must be (averaging period + t_back) long.
    marks = tuple(sorted(float(m) for m in tback_marks if 0 < float(m) <= t_back))
    cap_w = {m: 0.0 for m in marks}
    cap_fy = {m: np.zeros(len(fy_c)) for m in marks}
    # Integral decomposed by how far the trajectory had TRAVELLED when it touched down.
    # This is the direct test for periodic double counting: the domain repeats every Lx,
    # so influence attributed to touchdowns beyond that distance is turbulence the
    # trajectory had already sampled once. On a folded raster nothing falls off the edge
    # any more, so the contamination shows up in the integral instead of being silently
    # truncated -- which means it can finally be measured.
    # The ladder must extend PAST the cap, or it can only ever report the cap. When the
    # cap is raised for the wrap diagnostic the ladder follows it, so the curve is sampled
    # where the question actually lives.
    disp_edges = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]) * fs.Lx
    disp_w = np.zeros(len(disp_edges))

    # ---------------------------------------------------------------- reference frame
    # A real eddy-covariance system over a slope is DOUBLE ROTATED (Wilczak et al. 2001;
    # Kaimal & Finnigan 1994): first about z so the mean crosswind vanishes, then about the
    # new y so the mean vertical vanishes. The reported flux is w' c' in that streamline
    # frame, not in the model's frame. Weighting by the model-frame w is a different
    # quantity whenever the terrain tilts the mean flow.
    #
    #   theta = atan2(V, U)                        (yaw)
    #   phi   = atan2(W, sqrt(U^2 + V^2))          (pitch)
    #   w_sf  = w cos(phi) - (u cos(theta) + v sin(theta)) sin(phi)
    #
    # The pitch term matters more than its size suggests: sin(phi) is only ~0.014 here, but
    # it multiplies the HORIZONTAL fluctuation, whose flux <u'c'> is far larger than <w'c'>.
    # Rotating also makes the mean vanish BY CONSTRUCTION, so the estimator needs no
    # separate mean subtraction -- and subtracting a mean is exactly what was numerically
    # dangerous, because it adds w_bar times the unbounded concentration integral.
    sel = (fs.t >= t_first - 1e-6) & (fs.t <= t_last + 1e-6)
    # k_r is FRACTIONAL under exact_agl -- the receptor holds a fixed height above bare
    # ground, so over ground raised by the displacement height it sits between two model
    # levels. Interpolate the same way window_stats does, or these three lines raise
    # "only integers, slices ... are valid indices" and only on the raised treatment.
    _k0 = int(np.floor(k_r)); _k1 = min(_k0 + 1, fs.nz - 1)
    _fk = float(k_r) - _k0
    _lev = lambda A: ((1.0 - _fk) * A[sel, _k0, j_r, i_r] + _fk * A[sel, _k1, j_r, i_r])
    Ub = float(_lev(fs.u).mean())
    Vb = float(_lev(fs.v).mean())
    Wb = float(_lev(fs.w).mean())
    theta = np.arctan2(Vb, Ub)
    phi = np.arctan2(Wb, np.hypot(Ub, Vb))
    if verbose:
        print(f"  receptor means over the averaging period: U={Ub:+.4f} V={Vb:+.4f} "
              f"W={Wb:+.4f} m/s")
        print(f"  double rotation: yaw {np.degrees(theta):+.2f} deg, "
              f"pitch {np.degrees(phi):+.3f} deg (sin={np.sin(phi):+.5f})")

    def streamline_w(r):
        """Streamline-frame vertical velocity of each released particle."""
        return (r["rel_w"] * np.cos(phi)
                - (r["rel_u"] * np.cos(theta) + r["rel_v"] * np.sin(theta)) * np.sin(phi))

    # PERIODIC WRAP-AROUND. A backward trajectory that travels more than one domain length
    # re-enters the turbulence it already sampled, and its extra touchdowns are not new
    # information -- they are the same eddies counted twice. Measured, over the flat window:
    #
    #   t_back    wrapped    integral (uncapped)    integral (capped at one domain length)
    #      300       0.0%          0.643                        0.643
    #      600       0.0%          0.800                        0.800
    #      900       8.2%          0.791                        0.896
    #     1500      31.8%          1.064                        0.961
    #
    # Uncapped, the integral sails past 1 as soon as trajectories start wrapping; capped, it
    # converges to 1 from below, which is what a flux footprint must do. Default the cap to
    # one streamwise domain length. Crosswind wrap is secondary: sigma_y at that distance is
    # a few hundred metres against a 750 m half-width.
    if max_disp is None:
        max_disp = fs.Lx
    if verbose:
        print(f"  wrap-around cap: retiring trajectories past {max_disp:.0f} m of "
              f"displacement (domain {fs.Lx:.0f} x {fs.Ly:.0f} m)")

    floor_diag = None
    sgs_offset = None
    if sgs_most:
        # ---- MOST-anchored sub-grid variance floor ---------------------------------
        # Measured on this pipeline: the LES delivers sigma_w/u* well below the
        # surface-layer 1.25 at the receptor, because at z/Delta ~ 1 the eddies that
        # carry w are at or below the filter scale. A low sigma_w makes backward
        # particles descend too slowly, so they travel too far before touching down.
        # This is a FLOOR: where the model already resolves enough, the factor is 1.
        #
        # The construction lives in lpdm/sgs_floor.py so that the well-mixed GATE and
        # the footprints use the same code rather than two drifting copies -- see the
        # module docstring for why the factor-taper form was retired.
        fl = most_floor(st, d_r=d_r, mode=sgs_most_mode, legacy=sgs_most_legacy,
                        subgrid_weight=sgs_subgrid_weight)
        zl, fac = fl["zl"], fl["fac"]
        # THE INVARIANT, SCORED ON EVERY CASE. See lpdm/sgs_floor.py:floor_health for the
        # derivation and the margins. It is computed before anything is printed so the
        # alarm cannot be lost in the ordinary diagnostics above it.
        fl["health"] = floor_health(fl, z_receptor=z_target)
        floor_diag = fl
        n_new, worst = check_monotone(fl)
        if n_new and not sgs_most_legacy:
            raise RuntimeError(
                f"the restructured floor introduced {n_new} new turnover(s) in "
                f"sigma_w^2 (worst {worst:+.3%}) -- this is the defect the "
                f"restructure exists to make impossible; do not run footprints on it")
        # DELIVERY. MEASURED, and the answer was not the expected one. Additive delivery
        # was built to remove a real inconsistency -- the field's own dsig2dz is a central
        # difference that multiplicative delivery enters with weight sc (10.2x
        # convectively) while the product rule's two terms nearly cancel. It does remove
        # it, and the two deliveries are bit-identical on sigma^2. It did NOT fix the
        # convective well-mixed gate, and it made the BACKWARD direction -- the one
        # footprints use -- worse: lowest-three-bins 1.014 -> 1.152. So multiplicative
        # stays the default on measurement, additive is kept as a recorded negative
        # result, and neither is trusted convectively until the gate passes. See
        # docs/results/SIXTH_PASS_RESULTS.md.
        if sgs_most_legacy or sgs_most_form == "multiplicative":
            sgs_scale = (zl, fac)
        else:
            sgs_offset = (zl, fl["delta"])
        if verbose:
            kk = int(np.argmin(np.abs(zl - z_target)))
            print(f"  SGS floor mode '{sgs_most_mode}'"
                  f"{' [LEGACY TAPER]' if sgs_most_legacy else ''}: surface-layer target "
                  f"{fl['tgt_sfc'][kk]/fl['ustar']:.2f} u*, mixed-layer target "
                  f"{fl['tgt_mix'][kk]/fl['ustar']:.2f} u* (w*={fl['wstar']:.2f} m/s)")
            print(f"  SGS MOST floor: receptor d={d_r:.3f} m; factor {fac[kk]:.3f} at the "
                  f"receptor ({fac.min():.2f}-{fac.max():.2f} over the column); "
                  f"active below z={zl[fl['kpk']]:.0f} m (the model's own sigma_w^2 peak); "
                  f"sigma_w/u* {np.sqrt(fl['base'][kk])/fl['ustar']:.2f} -> "
                  f"{np.sqrt(fl['sig2'][kk])/fl['ustar']:.2f} (surface-layer target 1.25)")
            hh = fl["health"]
            # LOG THE RANGE AND WHERE IT PEAKS, ON SUCCESS TOO. A configuration that sits
            # at zero margin leaves no evidence of how close it came unless the passing
            # path says so.
            print(f"  floor health: factor {hh['fac_min']:.3f}-{hh['fac_max']:.4g} "
                  f"(peak at z={hh['z_fac_max']:.0f} m), inflates sigma_w^2 by at most "
                  f"{100*hh['inflation_max']:.1f}% at z={hh['z_delta_max']:.0f} m where "
                  f"f_sgs = {hh['f_sgs_at_peak']:.3f} (must be >= "
                  f"{FSGS_AT_PEAK_MIN}); "
                  f"inert above z={hh['z_inert']:.0f} m = {hh['z_inert_over_h']:.2f} h; "
                  f"{100*hh['share_resolved']:.0f}% of the added variance lands where the "
                  f"LES already resolves >=80%")
            for _a in hh["alarms"]:
                print(f"\n  *** FLOOR ALARM: {_a}\n")
            print(f"  monotonicity: {n_new} floor-induced turnover(s) in sigma_w^2 below "
                  f"the peak (must be 0); sub-grid weighting "
                  f"{'ON' if sgs_subgrid_weight and not sgs_most_legacy else 'off'}, "
                  f"eps-consistent {'ON' if sgs_eps_consistent else 'off'}; delivery "
                  f"{'MULTIPLICATIVE (retired)' if sgs_offset is None else 'additive'}")

    lp = LPDM(fs, c0=c0, z_touch=z_touch, seed=seed, aniso=aniso,
              sgs_scale=sgs_scale, sgs_offset=sgs_offset,
              sgs_eps_consistent=sgs_eps_consistent)
    t_start = time.time()
    n_td = 0
    wsf_bar, wsf_n = [], 0
    # Footprint-weighted share of each land-cover class. Accumulated from the touchdown
    # positions in LES INDEX space rather than by overlaying a mask on the rotated
    # footprint grid: the rotation and the 60 m footprint cells would both blur a 30 m
    # patch, and the water/array shares are exactly the numbers that must not be blurred.
    cover_w = {k: 0.0 for k in (cover or {})}
    cover_tot = 0.0
    # Land-cover attribution, computed TWICE.
    #
    # The raster is periodic and touchdowns fold into it, which is right for the tiled
    # world the LES actually simulates and right for the emulator's target. It is NOT
    # right for attributing flux to real geography: a touchdown 3 km upwind folds to
    # 1.5 km on the far side of the tower, and the land cover there is a different lake
    # and a different wood. (Terrain is tapered to a constant near the seams; land cover
    # deliberately is not, because tapering it would erase the water from exactly the
    # easterly cases meant to sample it -- so the folded cells carry real, specific, wrong
    # classes.)
    #
    # A touchdown is "wrapped" if the fold moved it, which is exact and needs no threshold.
    # The unwrapped shares are the ones to quote for the site; the difference between them
    # is the size of the ambiguity.
    cover_nw = {k: 0.0 for k in (cover or {})}
    cover_tot_nw = 0.0
    wrapped_w = 0.0
    # Per-HALF cover shares. The land-cover share is the observable the domain-adequacy
    # gate turns on, and a gate needs a sampling floor measured the same way its quantity
    # is -- otherwise the tolerance is an opinion. The halves split by RELEASE TIME, the
    # same split the peak and centroid floors already use, so the three floors are
    # commensurable.
    # N-WAY SPLIT OF THE RELEASE PERIOD. Two halves give a share ONE difference and hence
    # essentially one degree of freedom -- enough for a rough floor, not enough to put a
    # standard error on a 1-point effect. Splitting into N independent release groups costs
    # nothing (the touchdowns are already labelled by release time) and turns the floor into
    # a real sampling distribution. N = 2 reproduces the halves exactly.
    NG = max(2, int(n_cover_groups))
    cover_h = [{k: 0.0 for k in (cover or {})} for _ in range(NG)]
    cover_tot_h = [0.0] * NG
    # Cover share as a function of t_back, on the same age mask the capture curve uses.
    # The INTEGRAL is the slowest thing to converge, because it keeps collecting far-field
    # tail; the array sits within 250 m of the tower, so its share may settle far sooner --
    # and the array share, not the integral, is what the corpus is being built to predict.
    # Sizing t_back on the integral alone would buy backward time the observable does not
    # need. Free, like the rest of the curve: it is a mask on touchdowns already in hand.
    cap_cov = {m: {k: 0.0 for k in (cover or {})} for m in marks}
    cap_cov_tot = {m: 0.0 for m in marks}
    # ---- RAW TOUCHDOWNS, for the ML target ------------------------------------------
    # The 122^2 raster is not the training target: binning is what makes the per-cell L1
    # ~92% between two realisations of the SAME conditions, so a loss computed on it is
    # mostly fitting Monte-Carlo noise. The target is a continuous density fitted to the
    # touchdowns themselves (docs/ML_TARGETS.md), and the fields are DELETED at the end of every
    # case -- so a touchdown not captured here is gone for good.
    #
    # UNFOLDED coordinates. The raster folds modulo the domain because the LES world is
    # tiled and the land-cover attribution folds the same way; the ML target wants the true
    # displacement, which is what the wrap cap is measured against and what a density in
    # map coordinates needs. Both are available: fold is a modulo away, unfolding is not.
    #
    # BOTTOM-k SUBSAMPLING, not "take the first k". Each touchdown draws an independent
    # uniform key and the k smallest keys are kept, which is an exactly uniform sample
    # without replacement, streams in one pass, and needs no advance knowledge of how many
    # touchdowns there will be. Taking a prefix would sample the first release times only.
    td_keep = int(keep_touchdowns)
    td_res = dict(key=np.empty(0), dx=np.empty(0, np.float32), dy=np.empty(0, np.float32),
                  wt=np.empty(0, np.float32), grp=np.empty(0, np.int16),
                  age=np.empty(0, np.float32))
    td_rng = np.random.default_rng(seed + 977)
    td_sum_w = 0.0
    for b0 in range(0, len(times), batch_releases):
        tb = times[b0:b0 + batch_releases]
        n = len(tb) * n_per_release
        t = np.repeat(tb, n_per_release)
        res = lp.run(np.full(n, xr), np.full(n, yr), np.full(n, zr), t,
                     direction=-1, t_limit=t_back, reflect_touchdown=True,
                     record_touchdown=True, max_disp=max_disp)
        dx = res["td_x"] - xr
        dy = res["td_y"] - yr
        X = -(dx * ca + dy * sa)          # upwind-positive, wind frame
        wsf = streamline_w(res)
        wsf_bar.append(wsf.mean() * len(wsf)); wsf_n += len(wsf)
        # STATIC frame: fold the touchdown into the periodic domain by LES index. A
        # touchdown one domain length upwind is over the SAME surface as its image, and
        # the cover attribution below already folds the identical way -- so folding here
        # is what keeps the raster and the attribution describing one thing.
        fi_t, fj_t = fs.hindex(res["td_x"], res["td_y"])
        ii = np.clip(np.round(fi_t).astype(int) % fs.nx, 0, fs.nx - 1)
        jj = np.clip(np.round(fj_t).astype(int) % fs.ny, 0, fs.ny - 1)
        Xm = (((fi_t + 0.5) % fs.nx) - 0.5 - i_r) * fs.dx
        Ym = (((fj_t + 0.5) % fs.ny) - 0.5 - j_r) * fs.dy
        r = dict(res); r["td_x"] = Xm; r["td_y"] = Ym
        r["w_release"] = wsf
        full.add(r, 0.0, 0.0, w_floor=w_floor)
        wt_c = 2.0 / np.maximum(res["td_w"], w_floor)
        wt = wsf[res["td_particle"]] * wt_c
        fy_h += np.histogram(X, bins=fy_e, weights=wt)[0]
        dsp = np.hypot(res["td_x"] - xr, res["td_y"] - yr)     # UNWRAPPED, so it is the
        for m_ in range(len(disp_edges)):                      # true path displacement
            disp_w[m_] += wt[dsp <= disp_edges[m_]].sum()
        if marks:
            age = res.get("td_t")
            if age is not None:
                for m in marks:
                    sel_m = age <= m
                    cap_w[m] += wt[sel_m].sum()
                    cap_fy[m] += np.histogram(X[sel_m], bins=fy_e,
                                              weights=wt[sel_m])[0]
        wrap = (np.abs(res["td_x"] - xr) > 0.5 * fs.Lx) | \
               (np.abs(res["td_y"] - yr) > 0.5 * fs.Ly)
        wrapped_w += wt[wrap].sum()
        rel_t = t[res["td_particle"]]          # release time of each touchdown
        if td_keep:
            _span = max(float(times[-1] - times[0]), 1e-9)
            _g = np.clip(((rel_t - times[0]) / _span * NG).astype(np.int16), 0, NG - 1)
            _k = td_rng.random(dx.size)
            td_sum_w += float(wt.sum())
            for _key, _val in (("key", _k), ("dx", dx.astype(np.float32)),
                               ("dy", dy.astype(np.float32)),
                               ("wt", wt.astype(np.float32)), ("grp", _g),
                               ("age", (res.get("td_t") if res.get("td_t") is not None
                                        else np.zeros(dx.size)).astype(np.float32))):
                td_res[_key] = np.concatenate([td_res[_key], _val])
            if td_res["key"].size > td_keep:
                sel = np.argpartition(td_res["key"], td_keep)[:td_keep]
                for _key in td_res:
                    td_res[_key] = td_res[_key][sel]
        if cover:
            cover_tot += wt.sum()
            cover_tot_nw += wt[~wrap].sum()
            # group index from the release time, evenly over [times[0], times[-1]]
            span = max(float(times[-1] - times[0]), 1e-9)
            gi = np.clip(((rel_t - times[0]) / span * NG).astype(int), 0, NG - 1)
            hmask = tuple(gi == hh for hh in range(NG))
            for hh in range(NG):
                cover_tot_h[hh] += wt[hmask[hh] & (~wrap)].sum()
            age_c = res.get("td_t")
            for m in marks:
                if age_c is None:
                    break
                am = (age_c <= m) & (~wrap)
                cap_cov_tot[m] += wt[am].sum()
            for nm, msk in cover.items():
                sel_c = msk[jj, ii]
                cover_w[nm] += wt[sel_c].sum()
                cover_nw[nm] += wt[sel_c & (~wrap)].sum()
                for hh in range(NG):
                    cover_h[hh][nm] += wt[sel_c & (~wrap) & hmask[hh]].sum()
                for m in marks:
                    if age_c is None:
                        break
                    cap_cov[m][nm] += wt[sel_c & (~wrap) & (age_c <= m)].sum()
        n_td += len(X)
        if split_halves:
            rel = t[res["td_particle"]]
            n_h1 = int((tb <= tmid).sum()) * n_per_release
            for g, m, nn, acc in ((h1, rel <= tmid, n_h1, "1"),
                                  (h2, rel > tmid, n - n_h1, "2")):
                rr = dict(res)
                rr["w_release"] = wsf
                rr["td_particle"] = res["td_particle"][m]
                rr["td_w"] = res["td_w"][m]
                rr["td_x"] = Xm[m]; rr["td_y"] = Ym[m]
                rr["n"] = nn
                g.add(rr, 0.0, 0.0, w_floor=w_floor)
                hh = np.histogram(X[m], bins=fy_e, weights=wt[m])[0]
                if acc == "1":
                    fy_h1 += hh
                else:
                    fy_h2 += hh
        if verbose:
            print(f"    batch {b0//batch_releases+1}/"
                  f"{-(-len(times)//batch_releases)}  {n_td:,} touchdowns  "
                  f"{time.time()-t_start:.0f} s", flush=True)

    w_sf_mean = float(sum(wsf_bar) / max(wsf_n, 1))
    if verbose:
        print(f"  mean streamline-frame w over all releases: {w_sf_mean:+.5f} m/s "
              f"(model-frame mean was {Wb:+.5f}); rotation removed "
              f"{100*(1-abs(w_sf_mean)/max(abs(Wb),1e-12)):.1f}% of it")
    npart = max(full.n_particles, 1)
    out = dict(stats=st, grid=full, floor=floor_diag, receptor=(xr, yr, zr), z_agl=zr - zg_r, k_recept=k_r,
               fy=dict(xe=fy_e, xc=fy_c, f=fy_h / npart / fs.dx,
                       f1=fy_h1 / max(h1.n_particles, 1) / fs.dx,
                       f2=fy_h2 / max(h2.n_particles, 1) / fs.dx),
               capture={m: dict(integral=float(cap_w[m] / npart),
                                fy=cap_fy[m] / npart / fs.dx,
                                cover={k: (v / cap_cov_tot[m] if cap_cov_tot[m] else np.nan)
                                       for k, v in cap_cov[m].items()})
                        for m in marks},
               by_disp=[dict(max_disp=float(d), frac_of_Lx=float(d / fs.Lx),
                             integral=float(v / npart))
                        for d, v in zip(disp_edges, disp_w)],
               w_bar=Wb, w_sf_mean=w_sf_mean, yaw=float(theta), pitch=float(phi),
               cover_share={k: (v / cover_tot if cover_tot else np.nan)
                            for k, v in cover_w.items()},
               cover_share_nowrap={k: (v / cover_tot_nw if cover_tot_nw else np.nan)
                                   for k, v in cover_nw.items()},
               cover_share_halves=[{k: (v / cover_tot_h[hh] if cover_tot_h[hh] else np.nan)
                                    for k, v in cover_h[hh].items()}
                                   for hh in range(NG)],
               wrapped_fraction=float(wrapped_w / max(full.sum_flux_all, 1e-30)),
               n_particles=full.n_particles, n_touchdown=n_td,
               wind_angle=float(np.degrees(ang)))
    if td_keep:
        # The sample is a FRACTION of the touchdowns; the exact pre-subsample totals are
        # carried with it so the density can be renormalised to what the full ensemble
        # produced rather than to the sample. `weight_scale` is what a sample weight must
        # be multiplied by to reproduce the full estimator: sum(wt_sample)*scale/n_particles
        # is the footprint integral over all space.
        _ns = int(td_res["key"].size)
        out["touchdowns"] = dict(
            dx=td_res["dx"], dy=td_res["dy"], wt=td_res["wt"], grp=td_res["grp"],
            age=td_res["age"], n_sample=_ns, n_touchdown_total=int(n_td),
            n_particles=int(full.n_particles), n_release_groups=int(NG),
            sum_wt_total=float(td_sum_w),
            sum_wt_sample=float(td_res["wt"].sum()),
            weight_scale=float(n_td / _ns) if _ns else float("nan"),
            unfolded=True, signed=True,
            domain=dict(Lx=float(fs.Lx), Ly=float(fs.Ly), dx=float(fs.dx), dy=float(fs.dy)),
            receptor=dict(x=float(xr), y=float(yr), z_agl=float(zr - zg_r)),
            wind_angle_deg=float(np.degrees(ang)))
    if split_halves:
        out["halves"] = [h1, h2]
    return out
