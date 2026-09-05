"""The Kljun (2015) FFP channel, evaluated by the OFFICIAL implementation.

`third_party/FFP/calc_footprint_FFP.py` is Natascha Kljun's own code, vendored unmodified
(v1.42; see `third_party/FFP/PROVENANCE.md` for the URL, the date and the hashes). This
module is a THIN ADAPTER around it and **reimplements no formula**: every number that
depends on the parameterisation -- `f_ci`, `sigma_y`, `x_ci_max`, `Psi_M`, the scale
constant -- comes out of the official code.

=== WHY THE ADAPTER EXISTS AT ALL ===

The official `FFP()` builds its footprint on ITS OWN grid: a uniform axis in upwind
distance, mirrored in crosswind distance, optionally rotated at the end. The LES+LPDM
footprint is accumulated on the STATIC LES columns, so the reference it is scored against
-- and the `kljun` input channel every training pair carries -- has to live on those same
cells. Resampling the official raster onto them would interpolate in the crosswind
direction, where the field is a narrow Gaussian and interpolation loses mass near the
receptor.

So the adapter does the one thing that is not a formula: it re-evaluates the official's
own two factors at our cell coordinates.

    f(x, y)  =  f0(x) * exp( -y^2 / (2 sigma_y(x)^2) )

    f0(x)        = the official f_2d column at y = 0, i.e. f_ci(x)/(sqrt(2 pi) sigma_y(x))
    sigma_y(x)   = f_ci(x) / (sqrt(2 pi) f0(x))            -- exact algebra on the same two
                                                              official arrays

Both are taken straight off the official output. `f0` is interpolated in x (LOG-linearly,
because it spans ~120 decades across the near field and linear interpolation there is a
statement about the wrong quantity); `sigma_y` is interpolated linearly, which it deserves
-- it is smooth and monotone. **The crosswind direction is never interpolated**: the
Gaussian is evaluated analytically at each sub-cell's own y, which is the property
`lpdm/kljun.py:footprint_on_static` was written to preserve and this keeps.

=== WHAT THIS REPLACED, AND THE ONE DIVERGENCE THAT MATTERED ===

`lpdm/kljun.py` is a reimplementation from the paper's equations. It is not retired -- the
already-validated gates (`bin/corpus_monitor.py`, the seed stationarity battery) still call
it -- but it is no longer what a training pair carries. Writing this adapter surfaced a
real divergence, and it is in the regime the project's standing regression lives in:

**The near-neutral crosswind width was 25% too wide.** The official resets `ol = -1e6`
whenever `|ol| > oln = 5000` and evaluates `scale_const = 1e-5 |ol/z_m| + 0.80`, which at
`z_m = 30 m` is **1.133 and is then CLIPPED TO 1.0**. `lpdm/kljun.py` short-circuits
`|L| > 1e5` straight to `ps1 = 0.8` and never reaches the clip, so it divides `sigma_y` by
0.8 and comes out **1.25x** wide. At `|L| < 5000` the two agree to roundoff. See
`bin/test_kljun_adapter.py`, which measures the gap on corpus-representative inputs and
PRINTS it rather than asserting it away.

=== THE OFFICIAL VALIDITY CHECKS ARE CAPTURED, NOT SWALLOWED ===

`raise_ffp_exception` only RAISES for its 'fatal' codes; 'error' codes (z_m/L below -15.5,
u* <= 0.1, h <= 10, z_m > h) are PRINTED and execution continues. A printed warning inside
a redirected log is this project's standing failure mode (docs/reference/fasteddy-traps.md 12), so
`ffp_validity()` evaluates the official's own conditions up front and returns them as data.
The caller decides; nothing is silently fine.
"""
from __future__ import annotations

import importlib.util
import os
import threading

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
FFP_PATH = os.path.join(_HERE, os.pardir, "third_party", "FFP", "calc_footprint_FFP.py")
FFP_PATH = os.path.normpath(FFP_PATH)

# nx SETS THE OFFICIAL'S OWN x RESOLUTION, AND IT IS A MEMORY/ACCURACY TRADE.
# The official allocates the full 2-D field, nx x ~1.5 nx doubles, even though the adapter
# reads one column of it: nx = 2000 is ~48 MB transient and gives dx ~ 3.3 m at this site's
# scale factor, finer than the 3.75 m sub-cell an 8x8 subdivision of a 30 m cell produces.
# nx = 1000 (the official default) would be 6.6 m, coarser than the sub-cells, which is the
# wrong way round. The official refuses nx < 600.
NX_DEFAULT = 2000
SQRT_2PI = float(np.sqrt(2.0 * np.pi))
OLN = 5000.0        # the official's own neutral cutoff, quoted for ffp_validity only

_ffp_mod = None
_ffp_lock = threading.Lock()


def _ffp():
    """Import the vendored official module by PATH, once, without touching sys.path."""
    global _ffp_mod
    if _ffp_mod is None:
        with _ffp_lock:
            if _ffp_mod is None:
                if not os.path.exists(FFP_PATH):
                    raise FileNotFoundError(
                        f"the official FFP is not vendored at {FFP_PATH}. It is the Kljun "
                        f"channel of every training pair, so this is not something to work "
                        f"around -- see third_party/FFP/PROVENANCE.md for where it came "
                        f"from and re-fetch it.")
                spec = importlib.util.spec_from_file_location(
                    "_kljun_official_ffp", FFP_PATH)
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                _ffp_mod = m
    return _ffp_mod


def ffp_validity(zm, h, L, ustar, sigmav, z0=None, umean=None):
    """The official's own input conditions, returned as data instead of printed.

    Mirrors the checks in `calc_footprint_FFP.FFP` lines 86-98 EXACTLY, including the ones
    whose exception type is 'error' and therefore only print. Returns a list of strings;
    empty means the official would emit nothing.
    """
    bad = []
    if not (zm > 0.0):
        bad.append(f"zm = {zm} must be > 0 (FFP code 2)")
    if z0 is not None and umean is None and not (z0 > 0.0):
        bad.append(f"z0 = {z0} must be > 0 (FFP code 3)")
    if not (h > 10.0):
        bad.append(f"h = {h} must be > 10 m (FFP code 4)")
    if zm > h:
        bad.append(f"zm = {zm} must be < h = {h} (FFP code 5)")
    if z0 is not None and umean is None and zm <= 12.5 * z0:
        bad.append(f"zm = {zm} is inside the roughness sublayer 12.5*z0 = {12.5 * z0} "
                   f"(FFP code 12)")
    # zm/L <= -15.5 is the official's convective validity floor. L = inf is exactly
    # neutral and gives 0.0, which passes -- that is the official's behaviour too.
    zol = (zm / L) if (L is not None and np.isfinite(L) and L != 0.0) else 0.0
    if zol <= -15.5:
        bad.append(f"zm/L = {zol:.2f} must be >= -15.5 (FFP code 7)")
    if not (sigmav > 0.0):
        bad.append(f"sigmav = {sigmav} must be > 0 (FFP code 8)")
    if ustar <= 0.1:
        bad.append(f"ustar = {ustar} must be > 0.1 (FFP code 9)")
    return bad


def ffp_profile(zm, h, L, ustar, sigmav, umean=None, z0=None, nx=NX_DEFAULT):
    """Call the OFFICIAL FFP once and return its two separable factors on its own x axis.

    Returns a dict with

        x     (n,)  upwind distance [m], uniform, ascending
        f0    (n,)  f(x, y = 0) [m^-2]  -- the official f_2d column at y = 0
        sigy  (n,)  sigma_y(x) [m]
        f_ci  (n,)  crosswind-integrated footprint [m^-1]
        x_peak      x_ci_max [m]

    Exactly one of `umean` and `z0` may be given; passing both makes the official print an
    alert and silently prefer z0, so it is refused here instead.
    """
    if (umean is None) == (z0 is None):
        raise ValueError(
            "give exactly one of umean or z0. The official FFP prints an alert and "
            "silently prefers z0 when both are passed, which is a choice that would then "
            "live only in a log.")
    if nx < 600:
        raise ValueError(f"the official FFP refuses nx < 600 (got {nx})")

    # L = inf is a legitimate exactly-neutral value and the official handles it: |ol| > oln
    # sends it to -1e6 before sigma_y. It reaches the code as a float either way.
    ol = float(L) if L is not None else float("inf")

    # rs=None AND crop=False are load-bearing, not tidiness: the contour path imports
    # matplotlib and calls plt.contour, which is a display dependency the container does
    # not need and a cost per case the adapter never uses.
    out = _ffp().FFP(zm=float(zm), z0=(None if z0 is None else float(z0)),
                     umean=(None if umean is None else float(umean)),
                     h=float(h), ol=ol, sigmav=float(sigmav), ustar=float(ustar),
                     wind_dir=None, rs=None, rslayer=1, nx=int(nx),
                     crop=False, fig=False)
    if out.get("flag_err"):
        raise ValueError(f"the official FFP set flag_err for zm={zm} h={h} L={L} "
                         f"ustar={ustar} umean={umean} z0={z0}")

    x = np.asarray(out["x_ci"], dtype=np.float64)
    f_ci = np.asarray(out["f_ci"], dtype=np.float64)
    f_2d = np.asarray(out["f_2d"], dtype=np.float64)
    y_ax = np.asarray(out["y_2d"], dtype=np.float64)[0]

    # THE y = 0 COLUMN IS FOUND, NOT ASSUMED. The official builds y by concatenating a
    # mirrored negative half onto y_pos, so the zero sits at len(y_pos) - 1 -- but that is
    # an implementation detail of code this project does not own and must not depend on.
    iy0 = int(np.argmin(np.abs(y_ax)))
    if y_ax[iy0] != 0.0:
        raise ValueError(
            f"the official FFP's crosswind axis has no exact zero (nearest {y_ax[iy0]!r}); "
            f"the separable reconstruction below assumes the y = 0 column IS the Gaussian "
            f"prefactor and that is no longer true.")
    f0 = f_2d[:, iy0].astype(np.float64)

    # sigma_y BY EXACT ALGEBRA ON THE OFFICIAL'S OWN TWO ARRAYS, not by re-deriving Eq. 18:
    #   f0 = f_ci / (sqrt(2 pi) sigma_y)   =>   sigma_y = f_ci / (sqrt(2 pi) f0)
    # Where f0 underflows to zero the ratio is undefined -- and there f_ci is zero too, so
    # the footprint is zero whatever sigma_y is. sigy is set to 1.0 there purely so the
    # exponent is finite; the prefactor f0 makes the product exactly 0.
    good = np.isfinite(f0) & (f0 > 0.0) & np.isfinite(f_ci)
    sigy = np.ones_like(f0)
    sigy[good] = f_ci[good] / (SQRT_2PI * f0[good])
    f0 = np.where(good, f0, 0.0)

    return {"x": x, "f0": f0, "sigy": sigy, "f_ci": np.where(good, f_ci, 0.0),
            "x_peak": float(out["x_ci_max"]), "nx": int(nx)}


def _interp_factors(prof, X):
    """f0 and sigma_y at arbitrary upwind distances X, from the official's own axis.

    f0 is interpolated in LOG space. Across the near field it climbs through ~120 decades
    -- at this site f0 is 1e-126 at x = 31 m and 1e-3 near the 190 m peak -- and a linear
    interpolant there is fitting the wrong function. In log space the same samples describe
    exp(-c/(X*-d)) faithfully. Outside the official's axis the footprint is zero: below its
    first point because FFP is identically zero for X* <= d, above its last because X* = 30
    is the official's own end of domain.
    """
    x = prof["x"]
    f0 = prof["f0"]
    inside = (X >= x[0]) & (X <= x[-1])
    out = np.zeros_like(X)
    if not inside.any():
        return out, np.ones_like(X)
    tiny = 1e-300
    lf = np.log(np.maximum(f0, tiny))
    lo = np.interp(X[inside], x, lf)
    out[inside] = np.where(lo <= np.log(tiny) + 1.0, 0.0, np.exp(lo))
    sy = np.maximum(np.interp(X, x, prof["sigy"]), 1e-6)
    return out, sy


def footprint_on_static(xe, ye, ang, zm, h, ustar, sigmav, umean=None, z0=None, L=None,
                        nsub=8, nx=NX_DEFAULT, prof=None):
    """OFFICIAL FFP evaluated on a NORTH-UP static raster whose axes are east and north.

    A drop-in replacement for `lpdm.kljun.footprint_on_static` -- identical signature,
    identical geometry, identical return shape -- differing only in that every value comes
    out of the vendored official code.

    `xe`, `ye` are cell EDGES relative to the receptor (metres east, metres north).
    `ang` is the direction the mean wind BLOWS, radians measured from east (atan2(V, U)),
    matching `lpdm.driver`.

    Each cell is sub-sampled `nsub` x `nsub` and averaged. That is not cosmetic: near the
    receptor sigma_y is tens of metres against a 30 m cell, so a centre sample would misplace
    the near-field mass and push the apparent peak downwind.

    Pass `prof` to reuse a profile already computed by `ffp_profile` for the same scalars
    (the official call is the expensive part; the geometry is not).
    """
    xe = np.asarray(xe, dtype=np.float64)
    ye = np.asarray(ye, dtype=np.float64)
    if prof is None:
        prof = ffp_profile(zm, h, L, ustar, sigmav, umean=umean, z0=z0, nx=nx)

    ca, sa = np.cos(ang), np.sin(ang)
    dxc, dyc = np.diff(xe), np.diff(ye)
    off = (np.arange(nsub) + 0.5) / nsub - 0.5
    xs = (xe[:-1, None] + dxc[:, None] * (off[None, :] + 0.5)).ravel()   # sub-cell easts
    ys = (ye[:-1, None] + dyc[:, None] * (off[None, :] + 0.5)).ravel()   # sub-cell norths
    EX, NY = np.meshgrid(xs, ys)
    X = -(EX * ca + NY * sa)            # upwind distance
    Y = -EX * sa + NY * ca              # crosswind distance

    f0, sy = _interp_factors(prof, X.ravel())
    f = f0 * np.exp(-Y.ravel() ** 2 / (2.0 * sy ** 2))
    f = f.reshape(len(ye) - 1, nsub, len(xe) - 1, nsub)
    return f.mean(axis=(1, 3))


def peak_distance(zm, h, ustar, umean=None, z0=None, L=None, nx=NX_DEFAULT):
    """x_ci_max from the official code (its Eqs. 21-22), for parity with lpdm.kljun."""
    return ffp_profile(zm, h, L, ustar, 1.0, umean=umean, z0=z0, nx=nx)["x_peak"]


def crosswind_integrated_at(prof, x):
    """`f_ci(x)` [1/m] from an official profile, at arbitrary upwind distances.

    Linear in x, which `f_ci` deserves and `f0` does not: `f_ci` is the smooth
    crosswind-INTEGRATED curve, where `f0` carries the 1/sigma_y prefactor and spans ~120
    decades across the near field. Zero outside the official's own axis -- below its first
    point FFP is identically zero (X* <= d), above its last is X* = 30, its end of domain.
    """
    return np.interp(np.asarray(x, dtype=np.float64), prof["x"], prof["f_ci"],
                     left=0.0, right=0.0)
