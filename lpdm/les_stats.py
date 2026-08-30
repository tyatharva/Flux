"""Diagnose, from the LES itself, exactly the scalars Kljun et al. (2015) takes as input.

This is the comparison's whole point: FFP and the emulator must be driven by the SAME
scalars, so they are read off the LES rather than assumed. Kljun needs
z_m, h, u*, sigma_v, and either u(z_m) or (z0, L).

**Variances are resolved PLUS sub-grid.** At 30 m spacing the receptor sits 1.5 cells above
the surface and most of the velocity variance there is sub-grid: the resolved sigma_w at
30 m is 0.074 m/s against a surface-layer expectation of ~1.25 u* = 0.43. Kljun's sigma_v is
the total turbulent fluctuation, and the LPDM itself moves particles with resolved plus
modelled sub-grid motion, so feeding FFP the resolved part alone would make its footprint
spuriously narrow and the comparison meaningless. The sub-grid contribution is isotropic in
FastEddy's closure, so each component gets (2/3) e_sgs.

Wind direction is defined by the mean wind AT THE RECEPTOR HEIGHT, not at the surface or
the geostrophic level -- in a neutral Ekman layer those differ by ~10-25 degrees, and using
the wrong one rotates the LES footprint against the FFP one and turns a matching pair into
an apparent disagreement.
"""
from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

KAPPA = 0.4
G = 9.81


# THE PROFILE DOES NOT DECAY MONOTONICALLY, AND BOTH THRESHOLD DEFINITIONS ASSUMED IT DID.
#
# z_i was "search upward from the TKE peak for the first level below a threshold". On the
# first real corpus case that returned **2500 m -- the domain top** -- and the corpus input
# `h` went into the training record as the fallback rather than as a measurement. The
# profile is why (results/corpus/case_2023031014.json:profiles):
#
#     z (m)      2     18     52    236    428    559    907   1395   1699   2447
#     TKE     0.279  0.515  0.343  0.141  0.067  0.058  0.105  0.293  0.327  0.286
#     ww      0.007  0.033  0.095  0.044  0.019  0.053  0.103  0.284  0.329  0.031
#     e_sgs   0.221  0.168  0.032  0.009  0.001  0.000  0.000  0.012  0.017  0.045
#
# The layer decays to a clean minimum at ~560 m and then the variance RISES AGAIN to 0.33
# at 1700 m -- and that variance is essentially all RESOLVED w with no sub-grid part, i.e.
# internal-wave activity in the stable free atmosphere, not boundary-layer turbulence. A
# first-crossing search walks straight through the boundary layer, fails to cross before
# the wave layer lifts the profile back above the threshold, and falls through to z[-1].
# bin/seed_report.py already had the observation in a comment -- "that integral is
# dominated by gravity-wave variance aloft, which GROWS as the turbulence dies" -- but no
# estimator acted on it.
#
# THE FIX IS TO BOUND THE SEARCH BY THE DECAY MINIMUM, which is the classical definition of
# a boundary-layer top when there is wave activity above it: the depth is where the
# turbulence stops decaying, and a threshold crossing only counts if it happens on the way
# down. Everything above the minimum is a different fluid.
DAMP_FRAC = 0.8    # search below this fraction of the column; the production configuration
                   # is zCeiling 2500 m with dampingLayerDepth 500 m, so 0.8 IS the sponge
                   # base, and a sponge's variance is a numerical boundary condition.


def bl_depth(tk, z, thresh=None, frac=0.05, damp_frac=DAMP_FRAC):
    """Boundary-layer depth from a TKE profile that need not decay monotonically.

    `thresh` -- an absolute TKE threshold, m2/s2. If None, `frac` x the profile's own peak
    is used instead (the definition lpdm/les_stats.py has always produced as the corpus
    input `h`, and the one bin/pick_seed.py matches seeds in).

    The search runs from the peak DOWNWARD in TKE and stops at the decay minimum. Returns
    the crossing height if the threshold is met on the way down, and the minimum's height
    if it is not -- never the top of the domain, which is not a measurement.
    """
    tk = np.asarray(tk, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    top = np.searchsorted(z, damp_frac * z[-1])
    top = int(np.clip(top, 3, len(z)))
    k_pk = int(np.argmax(tk[:top]))
    k_min = k_pk + int(np.argmin(tk[k_pk:top]))
    if k_min <= k_pk:
        return float(z[k_pk])
    t = float(thresh) if thresh is not None else float(frac) * float(tk[k_pk])
    seg = tk[k_pk:k_min + 1]
    ab = np.where(seg < t)[0]
    return float(z[k_pk + ab[0]]) if len(ab) else float(z[k_min])


def window_stats(paths, k_recept):
    """Ensemble statistics over a series of dumps, at the receptor level.

    `k_recept` may be FRACTIONAL. It has to be, once the surface build can raise topoPos
    by the displacement height over the array: the receptor is pinned to a fixed height
    above BARE GROUND, so over a raised patch it sits between two model levels rather than
    on one. The receptor-level moments are then taken from the linearly interpolated
    field, which is exactly what the LPDM's own 4-D interpolation does at that height --
    interpolating the finished variances instead would be a different quantity.
    """
    kf = float(k_recept)
    k0 = int(np.floor(kf))
    fr = kf - k0
    U = V = 0.0
    uu = vv = ww = uv = 0.0
    esgs = 0.0
    tke_prof = None
    ww_prof = None          # resolved sigma_w^2(z), horizontal variance per level
    esgs_prof = None        # mean sub-grid TKE(z)
    zlev = None
    ust = hfx = th0 = 0.0
    n = 0
    # PER-DUMP DIRECTION, SO THE DRIFT INSIDE THE WINDOW CAN BE MEASURED AT ALL.
    # The seed library's dominant skill axis is direction, and the one thing measured
    # about it -- that the 30-minute adjustment WIDENS a gap rather than closing it --
    # rests on comparing a window MEAN against a requested value. That says the gap moved
    # but not how fast, and the window's own fields are deleted at the end of every case,
    # so it cannot be recovered afterwards. Two floats per dump make the rate recoverable
    # from the training record instead of from fields that no longer exist.
    uv_series, step_series = [], []
    # Level heights are STATIC, so they are read ONCE rather than from every dump -- but
    # NOT necessarily from paths[0]. Under ioLPDMmode the coordinate geometry goes to the
    # first file of a RUN and to the ioLPDMfullFrq multiples; a target case runs the
    # adjustment and the window as one invocation and DELETES the adjustment dumps, so the
    # run's first file is routinely gone by the time this reads. bin/run_window.sh sets
    # ioLPDMfullFrq so the first SURVIVING dump is full-form and asserts that it is, which
    # makes paths[0] correct in production -- but lpdm/fields.py already had to learn to
    # search rather than assume, and a caller that subsamples the series
    # (bin/stage4_wellmixed.py takes paths[::n]) should not depend on which end it kept.
    zsrc = None
    for _p in paths:
        with Dataset(_p) as _ds:
            if "zPos" in _ds.variables:
                zsrc = _p
                break
    if zsrc is None:
        raise KeyError(
            "no dump in this series carries zPos, so the receptor level cannot be "
            "located. Under ioLPDMmode only the first file of a run and the "
            "ioLPDMfullFrq multiples do -- see bin/run_window.sh.")
    with Dataset(zsrc) as ds0:
        z = np.squeeze(np.asarray(ds0["zPos"][:], dtype=np.float64))[:, 0, 0]
    nz = len(z)
    k0 = int(np.clip(k0, 0, nz - 1))
    k1 = min(k0 + 1, nz - 1)
    f1 = fr if k1 > k0 else 0.0
    lev = lambda a: (1.0 - f1) * a[k0] + f1 * a[k1]      # receptor-level 2-D slice
    for p in paths:
        with Dataset(p) as ds:
            g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
            u, v, w = g("u"), g("v"), g("w")
            e = np.maximum(g("TKE_0"), 0.0)
            ust += float(g("fricVel").mean())
            # htFlux is not written by the fork's ioLPDMmode (the LPDM never reads it), so
            # fall back to invOblen, which IS written and is the quantity L is wanted for
            # anyway. Deriving hfx back out of 1/L keeps the returned dict unchanged for
            # every caller.
            if "htFlux" in ds.variables:
                hfx += float(g("htFlux").mean())
            else:
                # PER CELL, THEN AVERAGE -- NEVER THE OTHER WAY ROUND.
                #
                # invOblen is 1/L = -kappa g htFlux/(u*^3 theta) (cuda_surfaceLayerDevice.cu
                # :426). Averaging THAT and then multiplying by a mean u*^3 is the mean of a
                # RATIO whose denominator is u*^3, so the average is dominated by whichever
                # cells happen to have the smallest friction velocity -- and over a
                # heterogeneous surface with a strong flux there are always some.
                #
                # MEASURED, on case_2023052519 (convective, w'th_v' = 0.291 K m/s over the
                # raised surface): the mean-of-the-ratio form returned hfx = 43.09 K m/s and
                # therefore L = -0.17 m, against a true -25 m. That is a factor of 148, it
                # went straight into Kljun's x_peak and into the sigma_w floor's zeta, and
                # NOTHING complained -- every downstream number was finite and plausible.
                # The earlier 10 m cases hid it because their fluxes were near zero, where
                # the two forms agree.
                #
                # The per-cell product is exact by construction: -u*_c^3 theta iL_c/(kappa g)
                # IS htFlux_c, so its mean is the mean surface flux and nothing else.
                iL_ = g("invOblen")
                us_ = g("fricVel")
                th_ = g("theta")[0]
                hfx += float((-(us_ ** 3) * th_ * iL_ / (KAPPA * G)).mean())
            th0 += float(g("theta")[0].mean())
        ur, vr, wr = lev(u), lev(v), lev(w)
        Uk, Vk = ur.mean(), vr.mean()
        U += Uk; V += Vk
        uu += ((ur - Uk) ** 2).mean()
        vv += ((vr - Vk) ** 2).mean()
        uv += ((ur - Uk) * (vr - Vk)).mean()
        esgs += float(lev(e).mean())
        ww += ((wr - wr.mean()) ** 2).mean()
        pr = lambda a: a - a.mean(axis=(-2, -1), keepdims=True)
        t = 0.5 * ((pr(u) ** 2) + (pr(v) ** 2) + (pr(w) ** 2)).mean(axis=(-2, -1))
        tke_prof = t if tke_prof is None else tke_prof + t
        wp = (pr(w) ** 2).mean(axis=(-2, -1))
        ep = e.mean(axis=(-2, -1))
        ww_prof = wp if ww_prof is None else ww_prof + wp
        esgs_prof = ep if esgs_prof is None else esgs_prof + ep
        zlev = z
        n += 1
        uv_series.append((float(Uk), float(Vk)))
        # `p`, THE MAIN LOOP'S VARIABLE -- not `_p`, which belongs to the zsrc search
        # above and stays bound to the FIRST file carrying zPos because that loop breaks.
        # Using it here stamped every dump with step 123120, so the time axis had zero
        # span and the fitted drift came out +19.3 and +59.9 deg/h on two cases whose
        # direction was actually BACKING by 14.9 and 7.2 deg. A plausible number from a
        # stale variable, with nothing complaining -- the house failure mode.
        try:
            step_series.append(int(str(p).rsplit(".", 1)[1]))
        except (ValueError, IndexError):
            step_series.append(n - 1)
    U /= n; V /= n; uu /= n; vv /= n; ww /= n; uv /= n; esgs /= n
    sgs = (2.0 / 3.0) * esgs        # isotropic sub-grid variance per component
    ust /= n; hfx /= n; th0 /= n
    tke_prof /= n
    ww_prof /= n
    esgs_prof /= n

    wdir = (270.0 - np.degrees(np.arctan2(V, U))) % 360.0     # meteorological
    spd = float(np.hypot(U, V))
    # rotate the (co)variances into the mean-wind frame: sigma_v is the CROSSWIND one
    ang = np.arctan2(V, U)
    ca, sa = np.cos(ang), np.sin(ang)
    # Kljun's sigma_v is the CROSSWIND fluctuation, so the full rotation is needed --
    # dropping the u'v' cross term biases it whenever the stress tensor is not aligned
    # with the grid, which in an Ekman layer it never quite is.
    sig_v_res = max(sa * sa * uu - 2.0 * sa * ca * uv + ca * ca * vv, 0.0)
    sig_v = float(np.sqrt(sig_v_res + sgs))
    # boundary-layer height: highest level with resolved TKE above 5% of its maximum
    # 5% of the profile's own peak, bounded by the decay minimum -- see bl_depth above for
    # why the bound is not optional. This is the corpus input `h` and the currency
    # bin/pick_seed.py matches seeds in; the seed GATE uses a fixed threshold instead,
    # because it scores a trend and a peak-normalised threshold moves with the peak
    # (FASTEDDY_TRAPS.md 16).
    h = bl_depth(tke_prof, z, frac=0.05)
    # AND IT MUST NEVER BE THE TOP OF THE COLUMN. `h` is a corpus INPUT and it also sets
    # the sigma_w floor's mixed-layer blend, so a fallback value does not announce itself
    # anywhere downstream -- it just makes a plausible footprint out of the wrong closure.
    # bl_depth cannot return z[-1] any more; this asserts that it did not.
    if h >= 0.98 * float(z[-1]):
        raise ValueError(
            f"h came out {h:.0f} m against a column top of {z[-1]:.0f} m. That is the "
            f"estimator failing to find a boundary layer, not a 2.5 km one: it would go "
            f"into the training record as a feature and into the sigma_w floor as the "
            f"mixed-layer blend height. See lpdm/les_stats.py:bl_depth.")
    L = (-ust ** 3 * th0 / (KAPPA * G * hfx)) if abs(hfx) > 1e-6 else np.inf
    return dict(z=z, z_recept=float(lev(z)), k_recept=kf, u_mean=spd, wdir=float(wdir),
                sigma_u=float(np.sqrt(uu + sgs)), sigma_v=sig_v,
                sigma_w=float(np.sqrt(ww + sgs)),
                sigma_v_resolved=float(np.sqrt(sig_v_res)),
                sigma_w_resolved=float(np.sqrt(ww)), e_sgs=float(esgs),
                ustar=float(ust), htFlux=float(hfx), theta0=float(th0),
                L=float(L), h=h, tke_prof=tke_prof, n_dumps=n,
                wdir_per_dump=[float((270.0 - np.degrees(np.arctan2(v_, u_))) % 360.0)
                               for (u_, v_) in uv_series],
                step_per_dump=[int(x) for x in step_series],
                ww_prof=ww_prof, esgs_prof=esgs_prof, zlev=zlev,
                U=float(U), V=float(V))
