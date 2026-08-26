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
                iLm = float(g("invOblen").mean())
                us_ = float(g("fricVel").mean())
                th_ = float(g("theta")[0].mean())
                hfx += (-(us_ ** 3) * th_ * iLm / (KAPPA * G)) if abs(iLm) > 1e-12 else 0.0
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
    kmax = int(np.argmax(tke_prof))
    above = np.where(tke_prof[kmax:] < 0.05 * tke_prof[kmax])[0]
    h = float(z[kmax + above[0]]) if len(above) else float(z[-1])
    L = (-ust ** 3 * th0 / (KAPPA * G * hfx)) if abs(hfx) > 1e-6 else np.inf
    return dict(z=z, z_recept=float(lev(z)), k_recept=kf, u_mean=spd, wdir=float(wdir),
                sigma_u=float(np.sqrt(uu + sgs)), sigma_v=sig_v,
                sigma_w=float(np.sqrt(ww + sgs)),
                sigma_v_resolved=float(np.sqrt(sig_v_res)),
                sigma_w_resolved=float(np.sqrt(ww)), e_sgs=float(esgs),
                ustar=float(ust), htFlux=float(hfx), theta0=float(th0),
                L=float(L), h=h, tke_prof=tke_prof, n_dumps=n,
                ww_prof=ww_prof, esgs_prof=esgs_prof, zlev=zlev,
                U=float(U), V=float(V))
