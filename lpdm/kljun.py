"""Kljun et al. (2015) FFP flux-footprint parameterisation.

Reference: N. Kljun, P. Calanca, M. W. Rotach, H. P. Schmid, "A simple two-dimensional
parameterisation for Flux Footprint Prediction (FFP)", Geosci. Model Dev. 8, 3695-3713,
2015.  Equation numbers below are that paper's.

This is the model the emulator has to beat, and the reference the Stage 5 gate compares
the LES+LPDM footprint against over flat uniform terrain -- where FFP is valid, so
disagreement there is a pipeline bug, not a result.

    (6)  X* = (x/z_m) (1 - z_m/h) [ (u(z_m)/u*) k ]^-1
    (7)  X* = (x/z_m) (1 - z_m/h) [ ln(z_m/z0) - Psi_M ]^-1
    (8)  F_y* = f_y z_m (1 - z_m/h)^-1 (u(z_m)/u*) k
    (14) F_y*(X*) = a (X* - d)^b exp( -c / (X* - d) ),  X* > d
    (17) a = 1.4524, b = -1.9914, c = 1.4622, d = 0.1359
    (18) sigma_y* = a_c [ b_c X*^2 / (1 + c_c X*) ]^(1/2)
    (19) a_c = 2.17, b_c = 1.66, c_c = 20.0
    (13) sigma_y* = p_s1 sigma_y u* / (z_m sigma_v),
         p_s1 = min(1, |z_m/L|^-1 1e-5 + p),  p = 0.8 (L<=0), 0.55 (L>0)
    (10) f(x,y) = f_y(x) exp(-y^2/(2 sigma_y^2)) / (sqrt(2 pi) sigma_y)
    (20) X*_max = -c/b + d = 0.8701
    (5)  Psi_M = -5.3 z_m/L                                        (L > 0)
              = ln[(1+chi^2)/2] + 2 ln[(1+chi)/2] - 2 atan(chi) + pi/2   (L < 0),
         chi = (1 - 19 z_m/L)^(1/4)

Note the MINUS sign inside the exponential of Eq. (14); the integral constraint
Eq. (15-16), int F_y* dX* = a c^(b+1) Gamma(-b-1) = 1, only closes with that sign, and it
is the check `verify_normalisation()` performs.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf as _erf
from scipy.special import gamma as _gamma

A, B, C, D = 1.4524, -1.9914, 1.4622, 0.1359
AC, BC, CC = 2.17, 1.66, 20.0
K = 0.4
XSTAR_MAX = -C / B + D          # 0.8701 (Eq. 20)


def psi_m(zm, L):
    """Integrated stability correction, Hogstrom (1996) as used by FFP (Eq. 5)."""
    if L is None or not np.isfinite(L) or abs(L) > 1e5:
        return 0.0
    if L > 0:
        return -5.3 * zm / L
    chi = (1.0 - 19.0 * zm / L) ** 0.25
    return (np.log((1.0 + chi ** 2) / 2.0) + 2.0 * np.log((1.0 + chi) / 2.0)
            - 2.0 * np.arctan(chi) + np.pi / 2.0)


def _pi4(zm, ustar, umean=None, z0=None, L=None):
    """Pi_4 = u(zm)/u* * k = ln(zm/z0) - Psi_M  (Eq. 4). Prefer the measured ratio."""
    if umean is not None:
        return umean / ustar * K
    return np.log(zm / z0) - psi_m(zm, L)


def crosswind_integrated(x, zm, h, ustar, umean=None, z0=None, L=None):
    """f_y(x) [m^-1], the crosswind-integrated flux footprint."""
    p4 = _pi4(zm, ustar, umean, z0, L)
    scale = zm / (1.0 - zm / h) * p4         # dx/dX*, from Eqs. (6) and (8)
    xs = np.asarray(x, dtype=float) / scale
    fy = np.zeros_like(xs)
    m = xs > D
    fy[m] = A * (xs[m] - D) ** B * np.exp(-C / (xs[m] - D))
    return fy / scale, xs


def sigma_y(x, zm, h, ustar, sigmav, umean=None, z0=None, L=None):
    """Crosswind standard deviation [m] at upwind distance x (Eqs. 13, 18)."""
    p4 = _pi4(zm, ustar, umean, z0, L)
    scale = zm / (1.0 - zm / h) * p4
    xs = np.maximum(np.asarray(x, dtype=float) / scale, 0.0)
    sy_star = AC * np.sqrt(BC * xs ** 2 / (1.0 + CC * xs))
    if L is None or not np.isfinite(L) or abs(L) > 1e5:
        ps1 = 0.8
    else:
        p = 0.8 if L <= 0 else 0.55
        ps1 = min(1.0, abs(zm / L) ** -1 * 1e-5 + p)
    return sy_star * zm * sigmav / (ustar * ps1)


def footprint_2d(xc, yc, zm, h, ustar, sigmav, umean=None, z0=None, L=None,
                 y_edges=None):
    """2-D footprint f(x,y) [m^-2] on a grid; x is upwind distance from the receptor.

    The crosswind Gaussian is INTEGRATED over each cell rather than sampled at its centre.
    Close to the receptor sigma_y is a few metres -- narrower than a footprint cell -- so
    point-sampling drops most of the near-field mass between grid points and moves the
    apparent peak tens of metres downwind. Pass `y_edges` to get the exact cell average;
    without it the centre-sample form is used.
    """
    fy, _ = crosswind_integrated(xc, zm, h, ustar, umean, z0, L)
    sy = np.maximum(sigma_y(xc, zm, h, ustar, sigmav, umean, z0, L), 1e-6)
    if y_edges is None:
        Y = yc[:, None]
        Dy = np.exp(-Y ** 2 / (2.0 * sy[None, :] ** 2)) / (np.sqrt(2 * np.pi) * sy[None, :])
    else:
        e = np.asarray(y_edges, dtype=float)[:, None]
        cdf = 0.5 * (1.0 + _erf(e / (np.sqrt(2.0) * sy[None, :])))
        dy = np.diff(np.asarray(y_edges, dtype=float))[:, None]
        Dy = np.diff(cdf, axis=0) / dy
    return fy[None, :] * Dy


def peak_distance(zm, h, ustar, umean=None, z0=None, L=None):
    """x_max, distance of maximum contribution (Eqs. 21-22)."""
    return XSTAR_MAX * zm / (1.0 - zm / h) * _pi4(zm, ustar, umean, z0, L)


def verify_normalisation():
    """Eq. (15-16): int_d^inf F_y*(X*) dX* = a c^(b+1) Gamma(-b-1) must equal 1."""
    return float(A * C ** (B + 1.0) * _gamma(-B - 1.0))


def footprint_on_static(xe, ye, ang, zm, h, ustar, sigmav, umean=None, z0=None, L=None,
                        nsub=8):
    """FFP evaluated on a NORTH-UP static raster whose axes are east and north.

    The LES+LPDM footprint is accumulated on the static LES columns, so the reference it
    is scored against has to live on the same cells. Rotating a wind-frame Kljun raster
    onto them would interpolate, which is exactly what this pass set out to remove -- and
    it would do it to the analytic field, where interpolation is gratuitous: FFP is a
    closed-form function of (upwind distance, crosswind distance) and can simply be
    evaluated at the cells' own coordinates.

    Each cell is sub-sampled `nsub` x `nsub` and averaged. That is not cosmetic. Near the
    receptor sigma_y is a few metres against a 24 m cell, so a centre sample would miss
    most of the near-field mass and push the apparent peak downwind -- the same failure the
    `y_edges` argument of `footprint_2d` exists to avoid on a rectilinear grid.

    `ang` is the direction the mean wind BLOWS, in radians measured from east
    (i.e. atan2(V, U)), matching lpdm.driver.
    """
    xe = np.asarray(xe, float); ye = np.asarray(ye, float)
    ca, sa = np.cos(ang), np.sin(ang)
    dxc, dyc = np.diff(xe), np.diff(ye)
    off = (np.arange(nsub) + 0.5) / nsub - 0.5
    xs = (xe[:-1, None] + dxc[:, None] * (off[None, :] + 0.5)).ravel()   # sub-cell easts
    ys = (ye[:-1, None] + dyc[:, None] * (off[None, :] + 0.5)).ravel()   # sub-cell norths
    EX, NY = np.meshgrid(xs, ys)
    X = -(EX * ca + NY * sa)
    Y = -EX * sa + NY * ca
    fy, _ = crosswind_integrated(X.ravel(), zm, h, ustar, umean, z0, L)
    sy = np.maximum(sigma_y(X.ravel(), zm, h, ustar, sigmav, umean, z0, L), 1e-6)
    f = fy * np.exp(-Y.ravel() ** 2 / (2.0 * sy ** 2)) / (np.sqrt(2 * np.pi) * sy)
    f = f.reshape(len(ye) - 1, nsub, len(xe) - 1, nsub)
    return f.mean(axis=(1, 3))
