"""Diagnose, from the LES itself, exactly the scalars Kljun et al. (2015) takes as input.

This is the comparison's whole point: FFP and the emulator must be driven by the SAME
scalars, so they are read off the LES rather than assumed. Kljun needs
z_m, h, u*, sigma_v, and either u(z_m) or (z0, L).

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
    """Ensemble statistics over a series of dumps, at the receptor level."""
    U = V = 0.0
    uu = vv = ww = uv = 0.0
    tke_prof = None
    ust = hfx = th0 = 0.0
    n = 0
    for p in paths:
        with Dataset(p) as ds:
            g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
            u, v, w = g("u"), g("v"), g("w")
            z = g("zPos")[:, 0, 0]
            ust += float(g("fricVel").mean())
            hfx += float(g("htFlux").mean())
            th0 += float(g("theta")[0].mean())
        Uk, Vk = u[k_recept].mean(), v[k_recept].mean()
        U += Uk; V += Vk
        uu += ((u[k_recept] - Uk) ** 2).mean()
        vv += ((v[k_recept] - Vk) ** 2).mean()
        uv += ((u[k_recept] - Uk) * (v[k_recept] - Vk)).mean()
        ww += ((w[k_recept] - w[k_recept].mean()) ** 2).mean()
        pr = lambda a: a - a.mean(axis=(-2, -1), keepdims=True)
        t = 0.5 * ((pr(u) ** 2) + (pr(v) ** 2) + (pr(w) ** 2)).mean(axis=(-2, -1))
        tke_prof = t if tke_prof is None else tke_prof + t
        n += 1
    U /= n; V /= n; uu /= n; vv /= n; ww /= n; uv /= n
    ust /= n; hfx /= n; th0 /= n
    tke_prof /= n

    wdir = (270.0 - np.degrees(np.arctan2(V, U))) % 360.0     # meteorological
    spd = float(np.hypot(U, V))
    # rotate the (co)variances into the mean-wind frame: sigma_v is the CROSSWIND one
    ang = np.arctan2(V, U)
    ca, sa = np.cos(ang), np.sin(ang)
    # Kljun's sigma_v is the CROSSWIND fluctuation, so the full rotation is needed --
    # dropping the u'v' cross term biases it whenever the stress tensor is not aligned
    # with the grid, which in an Ekman layer it never quite is.
    sig_v = float(np.sqrt(max(sa * sa * uu - 2.0 * sa * ca * uv + ca * ca * vv, 0.0)))
    # boundary-layer height: highest level with resolved TKE above 5% of its maximum
    kmax = int(np.argmax(tke_prof))
    above = np.where(tke_prof[kmax:] < 0.05 * tke_prof[kmax])[0]
    h = float(z[kmax + above[0]]) if len(above) else float(z[-1])
    L = (-ust ** 3 * th0 / (KAPPA * G * hfx)) if abs(hfx) > 1e-6 else np.inf
    return dict(z=z, z_recept=float(z[k_recept]), u_mean=spd, wdir=float(wdir),
                sigma_u=float(np.sqrt(uu)), sigma_v=sig_v, sigma_w=float(np.sqrt(ww)),
                ustar=float(ust), htFlux=float(hfx), theta0=float(th0),
                L=float(L), h=h, tke_prof=tke_prof, n_dumps=n,
                U=float(U), V=float(V))
