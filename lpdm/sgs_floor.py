"""The MOST-anchored sub-grid variance floor, in ONE place.

WHY THIS FILE EXISTS AT ALL. The floor used to be written out twice -- once in
`lpdm/driver.py` for the footprints and once in `bin/stage4_wellmixed.py` for the gate that
is supposed to VALIDATE the footprints. The two copies had already drifted: the gate's copy
never got the displacement-height correction, so it was scoring a closure the production
path does not use. A correctness gate that reimplements the thing it is gating is not a
gate. There is now one function, and both call it.

WHAT THE FLOOR IS. At `z/Delta ~ 1` the LES resolves only a small part of `sigma_w` at the
receptor, so backward particles descend too slowly and touch down too far away. The floor
supplies what Monin-Obukhov similarity says is missing and nothing more: where the model
already carries enough variance the factor is exactly 1 and nothing happens.

WHAT WENT WRONG WITH THE FIRST VERSION, AND WHY THE SHAPE HERE IS DIFFERENT.
The first version tapered the FACTOR off between 0.1h and 0.2h. That is a multiplier
applied to a quantity that is itself falling fast with height, and the product of a
rising-then-collapsing factor with a falling sub-grid energy manufactured a MAXIMUM in the
transported `sigma_w^2` at the taper's inner edge -- measured at 9.45x and z = 52 m in the
convective control, with `d(sigma_w^2)/dz < 0` at 10 of the 26 levels below 120 m.
`sigma_w^2` must not decrease away from an impermeable wall: where it does, Thomson's drift
points inward from both sides and particles pile up on the artificial maximum. The gate saw
it as a forward/backward asymmetry (backward rms 7.51% PASS, forward lowest-three-bins
1.258 FAIL), and the footprints saw it as convective integrals saturating ABOVE 1.

The fix is structural, not another taper:

  1. Build the TARGET `sigma_w^2(z)`, not a factor.
  2. Make the target NON-DECREASING with height by a running maximum upward. A floor that
     never decreases cannot introduce a maximum that was not already in the model's own
     profile -- `max(base, monotone)` has a local maximum only where `base` has one.
  3. Bound the target by the model's own peak variance, and switch the floor off at and
     above that peak. Above its own maximum the profile is legitimately decreasing and
     there is nothing for a surface-layer relation to repair. This replaces the arbitrary
     0.1h-0.2h taper with the profile's own structure, and it costs no tuning constant.

The two properties that follow are asserted, not hoped for: `fac >= 1` everywhere (it is a
floor), and the floor adds no interior maximum to `sigma_w^2`.

NOTE ON WHICH PROFILE MUST BE MONOTONE. The model transports the SUB-GRID variance
`sc(z) (2/3) e`; the resolved part rides the interpolated LES field. The sub-grid part is
SUPPOSED to fall with height -- that is what a sub-filter energy profile does, and forcing
it upward would be wrong. It is the TOTAL `wwp + sc (2/3) e` that must not turn over, and
that is what is constrained here.
"""
import numpy as np

__all__ = ["most_floor", "phi_w_of_zeta"]


def phi_w_of_zeta(zeta):
    """sigma_w/u* stability function, normalised to 1 at neutral.

    (1 - 3 zeta)^(1/3) unstable (Panofsky et al. 1977), 1 + 0.2 zeta stable
    (Kaimal & Finnigan 1994). phi_w(0) = 1 exactly, so the neutral case is untouched.
    """
    zeta = np.asarray(zeta, dtype=np.float64)
    return np.where(zeta < 0.0, np.maximum(1.0 - 3.0 * zeta, 1.0) ** (1.0 / 3.0),
                    1.0 + 0.2 * np.minimum(zeta, 2.0))


def most_floor(st, d_r=0.0, mode="surface", legacy=False):
    """Height-dependent multiplier on the sub-grid variance.

    st      window_stats() dict: zlev, ww_prof, esgs_prof, ustar, L, h, htFlux, theta0
    d_r     displacement height at the RECEPTOR column, m. phi_w is a function of
            (z - d)/L; the receptor's own d is the right one because the floor exists to
            repair sigma_w at the receptor, not in the domain-mean column (23.5% of this
            box is tree cover whose d is metres and none of it is near the tower).
    mode    'surface' (default, Panofsky), 'mixed' (Lenschow), 'blend' (the smaller).
    legacy  reproduce the retired 0.1h-0.2h factor taper, for measuring what it cost.

    Returns a dict with the profiles and a one-line summary. `fac` is what the LPDM takes
    as sgs_scale = (zl, fac).
    """
    zl = np.asarray(st["zlev"], dtype=np.float64)
    wwp = np.asarray(st["ww_prof"], dtype=np.float64)
    esp = np.asarray(st["esgs_prof"], dtype=np.float64)
    h = float(st["h"])
    Lv = float(st["L"])
    ust = float(st["ustar"])
    wth = float(st.get("htFlux", 0.0) or 0.0)
    th0 = float(st.get("theta0", 300.0) or 300.0)

    # ---- the target -----------------------------------------------------------------
    zl_eff = np.maximum(zl - d_r, 1e-3)
    zeta = (zl_eff / Lv if np.isfinite(Lv) and abs(Lv) > 1e-6
            else np.zeros_like(zl))
    phi = phi_w_of_zeta(zeta)
    tgt_sfc = 1.25 * phi * ust * np.maximum(1.0 - zl / max(h, 1.0), 0.0) ** 0.75
    # Lenschow et al. (1980) mixed-layer form. In the free-convection limit the two agree
    # to 1% (1.803 kappa^(1/3)/1.34 = 0.991); they differ only in the transition, where the
    # surface-layer form keeps a neutral 1.25 u* term carrying shear production and the
    # mixed-layer form has none. 'surface' stays the default for that reason; 'mixed' and
    # 'blend' exist to MEASURE the choice, and 'blend' must never be default because as
    # w* -> 0 the mixed target -> 0 and the floor would switch off just short of neutral.
    wstar = (9.81 / th0 * max(wth, 0.0) * max(h, 1.0)) ** (1.0 / 3.0)
    zz = np.clip(zl / max(h, 1.0), 1e-4, 1.5)
    tgt_mix = 1.34 * wstar * zz ** (1.0 / 3.0) * np.maximum(1.0 - 0.8 * zz, 0.0)
    if mode == "surface" or wstar <= 0.0:
        tgt = tgt_sfc
    elif mode == "mixed":
        tgt = tgt_mix
    elif mode == "blend":
        tgt = np.minimum(tgt_sfc, tgt_mix)
    else:
        raise ValueError(f"unknown sgs_most_mode {mode!r}")
    tgt2 = tgt ** 2

    extra = {}
    have = np.maximum((2.0 / 3.0) * esp, 1e-9)
    base = wwp + have                       # what the model transports with fac = 1

    if legacy:
        # THE RETIRED FORM. Kept only so the bias it introduced can be measured against
        # the same LES fields; never a production path.
        need = np.maximum(tgt2 - wwp, 0.0)
        taper = np.clip((0.2 * h - zl) / (0.1 * h), 0.0, 1.0)
        fac = 1.0 + taper * np.maximum(need / have - 1.0, 0.0)
        kpk = -1
    else:
        # THE RESOLVED PROFILE'S OWN PEAK, searched inside the boundary layer. Above it
        # sigma_w^2 is legitimately falling and a surface-layer relation has nothing to
        # repair, so the floor stops there.
        #
        # It has to be the RESOLVED variance's peak, not the total's. The total
        # wwp + (2/3)e has its global maximum at the FIRST MODEL LEVEL, where the
        # sub-grid closure piles up energy against the wall -- taking that as "the peak"
        # would switch the floor off above 2 m and leave the receptor uncorrected, which
        # is the opposite of what the floor is for. The resolved profile is the one with
        # the physical shape: zero at the wall by impermeability, rising to a maximum at
        # roughly 0.3-0.4 h, falling above it.
        inbl = zl <= max(h, float(zl[2]))
        kpk = int(np.argmax(np.where(inbl, wwp, -np.inf)))
        # DEGENERATE FALLBACK, and it must be loud. If the resolved profile has no peak
        # in the interior -- a window with essentially no resolved vertical motion -- the
        # argmax lands in the bottom few levels and would switch the floor off exactly
        # where it is needed. Fall back to the conventional surface-layer top, 0.2h, and
        # say so; a silent fallback here would look identical to a working floor.
        degenerate = zl[kpk] < 0.05 * max(h, 1.0)
        if degenerate:
            kpk = int(np.argmin(np.abs(zl - 0.2 * max(h, 1.0))))
        kpk = max(kpk, 1)
        cap = float(base[kpk])
        tgt2m = np.maximum.accumulate(np.minimum(tgt2, cap))   # monotone, bounded
        tgt2m[kpk:] = 0.0                                      # inactive at/above the peak
        sig2 = np.maximum(base, tgt2m)
        fac = sig2 / have - wwp / have
        extra["peak_degenerate"] = bool(degenerate)

    fac = np.maximum(fac, 1.0)              # it is a FLOOR; roundoff must not lower it
    sig2 = wwp + fac * have
    return dict(zl=zl, fac=fac, sig2=sig2, base=base, tgt2=tgt2, tgt_sfc=tgt_sfc,
                tgt_mix=tgt_mix, wwp=wwp, have=have, phi=phi, zeta=zeta, kpk=kpk,
                wstar=float(wstar), ustar=ust, h=h, L=Lv, d_r=float(d_r),
                mode=mode, legacy=bool(legacy), **extra)


def check_monotone(fl, z_top=None):
    """Does the FLOOR introduce a turnover the unmodified model did not have?

    Compares the count of decreasing steps in the transported sigma_w^2 against the same
    count in the model's own profile, below z_top (default: the floor's active range).
    Returns (n_new_drops, worst_relative_drop). Both must be 0 for the restructured form.
    """
    zl, sig2, base = fl["zl"], fl["sig2"], fl["base"]
    top = z_top if z_top is not None else (zl[fl["kpk"]] if fl["kpk"] > 0 else zl[-1])
    m = zl <= top
    ds, db = np.diff(sig2[m]), np.diff(base[m])
    new = (ds < -1e-12) & ~(db < -1e-12)
    worst = float(np.min(ds[new] / np.maximum(sig2[m][:-1][new], 1e-12))) if new.any() else 0.0
    return int(new.sum()), worst
