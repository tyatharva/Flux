"""Release / rotate / accumulate: turn a FieldSet into a wind-aligned flux footprint."""
from __future__ import annotations

import time

import numpy as np

from .footprint import FootprintGrid
from .les_stats import window_stats
from .model import LPDM


def receptor_indices(fs, z_target=30.0, x_frac=0.75, y_frac=0.5, ij=None):
    """Receptor cell. `ij` pins it to a specific column and overrides the fractions.

    The fractions date from the wind-ALIGNED elongated domain, where the tower sat 3/4
    along x to buy upwind fetch. The static domain centres the tower instead, so the
    fractions would put the receptor 1128 m east of it -- harmless over flat uniform
    ground, and completely wrong over real geography, where it would sample the wrong
    surface entirely. Pass the tower cell from data/grid/meta.npy.
    """
    k = int(np.argmin(np.abs(fs.zk - z_target)))
    if ij is not None:
        return int(ij[0]), int(ij[1]), k
    i = int(round(x_frac * fs.nx))
    j = int(round(y_frac * fs.ny))
    return i, j, k


def make_releases(fs, n_per_release, t_first, t_last, dt_release, xr, yr, zr):
    times = np.arange(t_first, t_last + 1e-9, dt_release)
    t = np.repeat(times, n_per_release)
    n = len(t)
    return (np.full(n, xr), np.full(n, yr), np.full(n, zr), t, times)


def compute_footprint(fs, paths, z_target=30.0, n_per_release=700, dt_release=4.0,
                      t_back=900.0, c0=3.0, z_touch=2.0, grid_res=20.0,
                      grid_x=(-600.0, 4500.0), grid_y=(-1500.0, 1500.0),
                      seed=0, split_halves=True, batch_releases=12, w_floor=0.02,
                      max_disp=None, cover=None, aniso=None, sgs_scale=1.0, sgs_most=False,
                      receptor_ij=None, tback_marks=(), rel_seconds=None, verbose=True):
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
    i_r, j_r, k_r = receptor_indices(fs, z_target, ij=receptor_ij)
    xr = fs.x0 + i_r * fs.dx
    yr = fs.y0 + j_r * fs.dy
    # ABSOLUTE height of that cell centre. Over terrain this is NOT fs.zk[k_r]: the
    # vertical coordinate is terrain-following, so the level sits at zg + ~z_target above
    # the local ground. Releasing at the flat-column height would put the receptor 20 m
    # underground on the hill.
    zr = float(fs.height(np.array([float(k_r)]), np.array([float(i_r)]),
                         np.array([float(j_r)]))[0])
    zg_r = float(fs.ground(np.array([float(i_r)]), np.array([float(j_r)]))[0])
    st = window_stats(paths, k_r)

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
        print(f"  receptor  (i,j,k)=({i_r},{j_r},{k_r})  x={xr:.0f} y={yr:.0f}  "
              f"z={zr:.3f} m ASL = {zr-zg_r:.3f} m AGL (ground {zg_r:.2f} m)")
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
        zeta = zl / Lv if np.isfinite(Lv) and abs(Lv) > 1e-6 else np.zeros_like(zl)
        phi_w = np.where(zeta < 0.0, np.maximum(1.0 - 3.0 * zeta, 1.0) ** (1.0 / 3.0),
                         1.0 + 0.2 * np.minimum(zeta, 2.0))
        tgt2 = (1.25 * phi_w * float(st["ustar"])
                * np.maximum(1.0 - zl / max(h, 1.0), 0.0) ** 0.75) ** 2
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
            print(f"  SGS MOST floor: factor {fac[kk]:.3f} at the receptor "
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
        if marks:
            age = res.get("td_t")
            if age is not None:
                for m in marks:
                    sel_m = age <= m
                    cap_w[m] += wt[sel_m].sum()
                    cap_fy[m] += np.histogram(X[sel_m], bins=fy_e,
                                              weights=wt[sel_m])[0]
        if cover:
            cover_tot += wt.sum()
            for nm, msk in cover.items():
                cover_w[nm] += wt[msk[jj, ii]].sum()
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
                                fy=cap_fy[m] / npart / fs.dx) for m in marks},
               w_bar=Wb, w_sf_mean=w_sf_mean, yaw=float(theta), pitch=float(phi),
               cover_share={k: (v / cover_tot if cover_tot else np.nan)
                            for k, v in cover_w.items()},
               n_particles=full.n_particles, n_touchdown=n_td,
               wind_angle=float(np.degrees(ang)))
    if split_halves:
        out["halves"] = [h1, h2]
    return out
