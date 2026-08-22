"""Release / rotate / accumulate: turn a FieldSet into a wind-aligned flux footprint."""
from __future__ import annotations

import time

import numpy as np

from .footprint import FootprintGrid
from .les_stats import window_stats
from .model import LPDM


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
                      exact_agl=False, verbose=True):
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
    st = window_stats(paths, k_r)
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
    disp_edges = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0]) * fs.Lx
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
    Ub = float(fs.u[sel, k_r, j_r, i_r].mean())
    Vb = float(fs.v[sel, k_r, j_r, i_r].mean())
    Wb = float(fs.w[sel, k_r, j_r, i_r].mean())
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

    if sgs_most:
        # ---- MOST-anchored sub-grid variance floor -------------------------------------
        # Measured on this pipeline: the LES delivers sigma_w/u* = 1.09 at the 30 m
        # receptor against the neutral surface-layer value of ~1.25, because at
        # z/Delta ~ 1.5 the eddies that carry w are at or below the filter scale. A low
        # sigma_w makes backward particles descend too slowly, so they travel too far
        # before touching down -- which is exactly the observed error (peak 390 m against
        # Kljun's 210 m). Verified by direct test: scaling the sub-grid variance so
        # sigma_w matches similarity moves the peak to 270 m and lifts the 80% source-area
        # overlap from 36.9% to 47.6%.
        #
        # This is a FLOOR, not a replacement. Where the LES already resolves enough, the
        # factor is 1 and nothing happens; the correction only supplies what similarity
        # says is missing, and it switches itself off above the surface layer because
        # (1 - z/h)^(3/4) decays and the resolved fraction grows.
        #
        # It is a calibration against Monin-Obukhov similarity, valid where MOST is:
        # flat, uniform, surface layer. Calibrate there and apply the RULE -- never the
        # number -- everywhere else.
        zl = np.asarray(st["zlev"], dtype=np.float64)
        wwp = np.asarray(st["ww_prof"], dtype=np.float64)
        esp = np.asarray(st["esgs_prof"], dtype=np.float64)
        h = float(st["h"])
        # STABILITY DEPENDENCE. The neutral surface layer has sigma_w/u* = 1.25; under
        # free convection it is larger and grows as (-z/L)^(1/3), and in stable conditions
        # it is slightly larger per unit u* as well. Using the neutral constant in a CBL
        # would anchor the floor to the wrong target -- and the CBL is the modal daytime
        # state at this site (CONUS404: 57% of quality-controlled hours are unstable).
        #
        #   phi_w = (1 - 3 z/L)^(1/3)   z/L < 0    (Panofsky et al. 1977)
        #         = 1 + 0.2 z/L         z/L > 0    (Kaimal & Finnigan 1994)
        #
        # phi_w(0) = 1 exactly, so every neutral result is unchanged bit for bit; this is a
        # strict generalisation of the relation the floor was calibrated on, not a retune.
        # It is still a SURFACE-LAYER relation, and the taper below still switches it off
        # above 0.2h -- which in a CBL is where sigma_w stops scaling with u* and starts
        # scaling with w*.
        Lv = float(st["L"])
        # DISPLACEMENT HEIGHT. phi_w is a function of (z - d)/L, not z/L.
        #
        # The floor is one 1-D profile applied to every particle, so d has to be a single
        # number, and the RECEPTOR COLUMN's is the right one rather than the domain mean.
        # The floor exists to repair sigma_w at the receptor -- that is what it is
        # calibrated against and what the taper below confines it to -- so it should take
        # the receptor's own aerodynamic height. The domain mean would be dominated by
        # something else entirely: WorldCover puts 28.5% of the wider domain under tree
        # cover, whose d ~ 0.7 h_c is metres, none of which the LES resolves and none of
        # which is near the tower.
        #
        # The (1 - z/h) taper below keeps z: it is about boundary-layer depth, not about
        # surface-layer similarity.
        zl_eff = np.maximum(zl - d_r, 1e-3)
        zeta = zl_eff / Lv if np.isfinite(Lv) and abs(Lv) > 1e-6 else np.zeros_like(zl)
        phi_w = np.where(zeta < 0.0, np.maximum(1.0 - 3.0 * zeta, 1.0) ** (1.0 / 3.0),
                         1.0 + 0.2 * np.minimum(zeta, 2.0))
        tgt_sfc = (1.25 * phi_w * float(st["ustar"])
                   * np.maximum(1.0 - zl / max(h, 1.0), 0.0) ** 0.75)
        # MIXED-LAYER TARGET, and why a convective case needs both.
        #
        # The surface-layer relation above and Lenschow et al. (1980)'s mixed-layer
        # relation are both standard, both valid in their own regime, and at the
        # measurement height of a convective boundary layer they DISAGREE: at
        # z/L = -0.4, z/z_i = 0.033 the surface-layer form asks for sigma_w/u* = 1.62
        # while Lenschow asks for about 1.26. Anchoring a floor to the larger of two
        # relations that disagree by 30% is how a correction becomes a fudge.
        #
        # THE DEFAULT REMAINS THE SURFACE-LAYER FORM, for a reason worth writing down.
        # In the free-convection limit the two agree to 1%: Panofsky asymptotes to
        # 1.803 (kappa z g/theta w'th')^(1/3) and Lenschow to 1.34 (z g/theta w'th')^(1/3),
        # and 1.803 kappa^(1/3) / 1.34 = 0.991. They differ only in the TRANSITION, where
        # the surface-layer form retains a neutral 1.25 u* term that carries shear
        # production and the mixed-layer form has none. At z/L = -0.36 the shear term is
        # not negligible, so the surface-layer form is the more complete of the two.
        # 'blend' and 'mixed' exist to MEASURE how much that choice is worth, not to
        # quietly replace it -- and 'blend' must never be the default, because as w* -> 0
        # the mixed-layer target -> 0 and the minimum would switch the floor off entirely
        # just short of neutral, where it is needed most.
        wth = float(st.get("htFlux", 0.0) or 0.0)
        th0 = float(st.get("theta0", 300.0) or 300.0)
        wstar = (9.81 / th0 * max(wth, 0.0) * max(h, 1.0)) ** (1.0 / 3.0)
        zz = np.clip(zl / max(h, 1.0), 1e-4, 1.5)
        tgt_mix = 1.34 * wstar * zz ** (1.0 / 3.0) * np.maximum(1.0 - 0.8 * zz, 0.0)
        if sgs_most_mode == "surface" or wstar <= 0.0:
            tgt = tgt_sfc
        elif sgs_most_mode == "mixed":
            tgt = tgt_mix
        else:                                     # "blend": the conservative floor
            tgt = np.minimum(tgt_sfc, tgt_mix)
        tgt2 = tgt ** 2
        need = np.maximum(tgt2 - wwp, 0.0)
        have = np.maximum((2.0 / 3.0) * esp, 1e-9)
        # MOST is a SURFACE-LAYER relation. Applying it through the whole boundary layer
        # over-corrects: between roughly 0.1h and h the resolved variance is legitimately
        # below the surface-layer extrapolation and there is nothing to repair. Measured:
        # the untapered floor reached 3.3x aloft and gave a WORSE 80% overlap (40.4%) than
        # a plain scalar (47.6%), despite an identical peak. Taper the correction off
        # across 0.1h - 0.2h so it acts only where the relation it is anchored to holds.
        taper = np.clip((0.2 * h - zl) / (0.1 * h), 0.0, 1.0)
        fac = 1.0 + taper * np.maximum(need / have - 1.0, 0.0)
        sgs_scale = (zl, fac)
        if verbose:
            kk = int(np.argmin(np.abs(zl - z_target)))
            print(f"  SGS floor mode '{sgs_most_mode}': surface-layer target "
                  f"{tgt_sfc[kk]/st['ustar']:.2f} u*, mixed-layer target "
                  f"{tgt_mix[kk]/st['ustar']:.2f} u* (w*={wstar:.2f} m/s), "
                  f"used {tgt[kk]/st['ustar']:.2f} u*")
            print(f"  SGS MOST floor: receptor d={d_r:.3f} m; factor {fac[kk]:.3f} at the receptor "
                  f"({fac.min():.2f}-{fac.max():.2f} over the column); "
                  f"phi_w={phi_w[kk]:.3f} at z/L={zeta[kk] if np.ndim(zeta) else 0:+.3f}; "
                  f"sigma_w/u* {np.sqrt(wwp[kk] + (2/3)*esp[kk])/st['ustar']:.2f} -> "
                  f"{np.sqrt(wwp[kk] + fac[kk]*(2/3)*esp[kk])/st['ustar']:.2f} "
                  f"(surface-layer target 1.25)")
    lp = LPDM(fs, c0=c0, z_touch=z_touch, seed=seed, aniso=aniso,
              sgs_scale=sgs_scale)
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
    cover_h = [{k: 0.0 for k in (cover or {})}, {k: 0.0 for k in (cover or {})}]
    cover_tot_h = [0.0, 0.0]
    # Cover share as a function of t_back, on the same age mask the capture curve uses.
    # The INTEGRAL is the slowest thing to converge, because it keeps collecting far-field
    # tail; the array sits within 250 m of the tower, so its share may settle far sooner --
    # and the array share, not the integral, is what the corpus is being built to predict.
    # Sizing t_back on the integral alone would buy backward time the observable does not
    # need. Free, like the rest of the curve: it is a mask on touchdowns already in hand.
    cap_cov = {m: {k: 0.0 for k in (cover or {})} for m in marks}
    cap_cov_tot = {m: 0.0 for m in marks}
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
        if cover:
            cover_tot += wt.sum()
            cover_tot_nw += wt[~wrap].sum()
            hmask = (rel_t <= tmid, rel_t > tmid)
            for hh in (0, 1):
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
                for hh in (0, 1):
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
    out = dict(stats=st, grid=full, receptor=(xr, yr, zr), z_agl=zr - zg_r, k_recept=k_r,
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
                                    for k, v in cover_h[hh].items()} for hh in (0, 1)],
               wrapped_fraction=float(wrapped_w / max(full.sum_flux_all, 1e-30)),
               n_particles=full.n_particles, n_touchdown=n_td,
               wind_angle=float(np.degrees(ang)))
    if split_halves:
        out["halves"] = [h1, h2]
    return out
