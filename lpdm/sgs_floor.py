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

__all__ = ["most_floor", "phi_w_of_zeta", "floor_health", "FSGS_AT_PEAK_MIN"]


def phi_w_of_zeta(zeta):
    """sigma_w/u* stability function, normalised to 1 at neutral.

    (1 - 3 zeta)^(1/3) unstable (Panofsky et al. 1977), 1 + 0.2 zeta stable
    (Kaimal & Finnigan 1994). phi_w(0) = 1 exactly, so the neutral case is untouched.
    """
    zeta = np.asarray(zeta, dtype=np.float64)
    return np.where(zeta < 0.0, np.maximum(1.0 - 3.0 * zeta, 1.0) ** (1.0 / 3.0),
                    1.0 + 0.2 * np.minimum(zeta, 2.0))


def most_floor(st, d_r=0.0, mode="surface", legacy=False, subgrid_weight=True):
    """Height-dependent multiplier on the sub-grid variance.

    st      window_stats() dict: zlev, ww_prof, esgs_prof, ustar, L, h, htFlux, theta0
    d_r     displacement height at the RECEPTOR column, m. phi_w is a function of
            (z - d)/L; the receptor's own d is the right one because the floor exists to
            repair sigma_w at the receptor, not in the domain-mean column (23.5% of this
            box is tree cover whose d is metres and none of it is near the tower).
    mode    'surface' (default, Panofsky), 'mixed' (Lenschow), 'blend' (the smaller).
    legacy  reproduce the retired 0.1h-0.2h factor taper, for measuring what it cost.
    subgrid_weight
            scale the correction by the sub-grid fraction (production; see below).

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

    # ---- SUB-GRID-FRACTION WEIGHTING -------------------------------------------------
    # The floor's entire justification is unresolved sub-filter variance, so it must scale
    # with how much of the variance is actually sub-filter:
    #
    #     sc_eff = 1 + (sc - 1) f_sgs,     f_sgs = (2/3)e / (ww_resolved + (2/3)e)
    #
    # Without it the factor is LARGEST exactly where the LES resolves the MOST -- measured
    # 10.1 at z = 52 m where 92% of ww is resolved -- which is backwards.  What the floor
    # was repairing up there is not an LES deficit but the error in extrapolating a
    # surface-layer anchor to 0.1 z_i, and it was repairing it by inflating the small
    # remaining sub-grid part tenfold.  Measured: a CONSTANT inflation of 10 fails the
    # convective well-mixed gate forward at 1.370 while a constant 1.673 passes at 1.130,
    # so the magnitude is the thing that has to be bounded, and this bounds it by the only
    # quantity that carries the physical argument.
    #
    # The weighting is applied to the FACTOR, then monotonicity is re-imposed, because the
    # two do not commute: f_sgs falls with height faster than the raw factor rises, so
    # their product dips (measured: sigma_w^2 0.3876 at 18 m against 0.3565 at 35 m) and a
    # dip is the defect the running maximum exists to forbid.  Re-running it costs a
    # slightly larger factor at the dip and keeps the structural guarantee.
    if subgrid_weight and not legacy:
        f_sgs = have / np.maximum(base, 1e-12)
        facw = 1.0 + (fac - 1.0) * f_sgs
        cap2 = float(base[kpk])
        # Monotonise the weighted TARGET, never the model's own profile: `base` has a
        # near-wall maximum from the sub-grid closure piling energy against the ground,
        # and running a cumulative maximum through THAT would drag the wall value up the
        # column and leave the floor active above the peak.
        # Where the floor asserts nothing the target must CONTRIBUTE nothing, or the
        # cumulative maximum picks up `base`'s near-wall value and spreads it up the
        # column -- a floor switched on by a profile it was not correcting.
        # The tolerance is load-bearing, not cosmetic: sig2 = base recovers fac = 1 only
        # to roundoff, and a fac of 1 + 1e-16 would otherwise read as "active" and let the
        # cap propagate up the column.
        tgt_w = np.where(facw > 1.0 + 1e-9,
                         np.minimum(wwp + facw * have, cap2), 0.0)
        tgt_wm = np.maximum.accumulate(tgt_w[:kpk + 1])
        sig2w = base.copy()
        sig2w[:kpk + 1] = np.maximum(base[:kpk + 1], tgt_wm)
        fac = np.maximum((sig2w - wwp) / have, 1.0)
        extra["f_sgs"] = f_sgs
        extra["fac_unweighted_max"] = float(facw.max())

    sig2 = wwp + fac * have
    # ---- ADDITIVE DELIVERY, and why it is not cosmetic ------------------------------
    # The model transports sigma^2_sgs.  Delivered as a FACTOR it is sc(z) (2/3)e(x,z),
    # whose z-derivative the drift needs, and the product rule gives
    #     sc * d[(2/3)e]/dz  +  (2/3)e * dsc/dz.
    # Two things go wrong there and BOTH are amplified by the floor.  The first term's
    # d[(2/3)e]/dz comes from lpdm/fields.py as a CENTRAL DIFFERENCE that is then
    # 4-D interpolated -- it is not the derivative of the interpolant that samples e --
    # and the floor multiplies that inconsistency by sc, which reaches 10 convectively.
    # The second is that the two terms nearly cancel (measured -0.0257 and +0.0232 at
    # z = 25 m, summing to -0.001), so a few-percent error in either is most of the
    # answer.  Measured consequence: the base model passes the well-mixed gate in both
    # directions convectively and the SAME model with a multiplicative floor fails
    # forward at lowest-three-bins 1.236.
    #
    # Delivered as an OFFSET, sigma^2_sgs = (2/3)e(x,z) + delta(z), the derivative is
    #     d[(2/3)e]/dz  +  d(delta)/dz,
    # the field term keeps weight 1 -- exactly the configuration that passes with no
    # floor at all -- and d(delta)/dz is the exact derivative of a 1-D profile.  Same
    # target, same monotone construction, same floor semantics (delta >= 0); only the
    # delivery differs.
    delta = np.maximum(sig2 - base, 0.0)
    return dict(zl=zl, fac=fac, sig2=sig2, base=base, delta=delta, tgt2=tgt2, tgt_sfc=tgt_sfc,
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


# =====================================================================================
# THE INVARIANT: THE FLOOR MUST DO ITS WORK WHERE THE LES HAS NOT RESOLVED THE VARIANCE
# =====================================================================================
# WHY THIS EXISTS. On the first corpus case `h` fell through to the domain top (2500 m),
# and because `h` sets the mixed-layer blend the floor ran at 3-20x between 35 and 200 m
# where it should have been 1.0, peaking near 9e4. NOTHING COMPLAINED. The receptor factor
# read 1.000, the driver printed "1.00-381935.02 over the column" as an ordinary range,
# and the footprint came out plausible -- the near-field peak was wrong by a full raster
# cell while the array share moved 0.8 points against a 3.66-point SE. The 100x warning
# added afterwards catches that particular case and would NOT catch a 3-20x version of it.
#
# WHAT TO TEST, AND WHY IT IS NOT THE FACTOR. `fac` is a multiplier on the SUB-GRID part,
# whose denominator collapses with height, so a large `fac` aloft can mean nothing at all:
# in g16r_cbl_wE the running maximum holds the TOTAL sigma_w^2 at ~0.655 from 18 m to
# 52 m, and `fac` reads 8.6 at 52 m purely because `have` has fallen to 0.032 there. Gating
# `fac` would fail four validated, Gate-D1-passing convective production cases.
#
# The premise the sub-grid weighting rests on is the testable thing. The floor's whole
# justification is unresolved sub-filter variance, so THE LEVEL AT WHICH IT ADDS THE MOST
# VARIANCE MUST BE A LEVEL THE LES HAS NOT RESOLVED. Measured on every production record
# on disk -- 12 of them, neutral and convective, four directions each, plus two corpus
# cases -- the sub-grid fraction at that level sits in 0.368-0.564, straddling the 0.5
# crossover where the resolved and sub-grid parts are equal. That band is not a
# coincidence: it is the f_sgs weighting placing the correction at the crossover, which is
# what it was built to do.
#
# With `h` broken the same quantity is 0.008 -- the floor's hardest work lands at 414 m,
# where the LES resolves 99.2% of sigma_w^2 and there is nothing to repair.
#
# THE THRESHOLD IS DERIVED, NOT PICKED. The crossover is f_sgs = 0.5. Half of it means
# "the floor's peak influence sits where the LES already resolves three quarters of the
# variance", which contradicts the premise outright. That is 1.47x below the lowest
# production value and 31x above the defect, and the h-sweep in both regimes shows the
# statistic is FLAT across every plausible h and only moves when h is grossly wrong -- so
# it does not fire on ordinary depth uncertainty. Margins, on this evidence:
#
#     production (n=14)   f_sgs at the floor's peak      0.368 - 0.564
#     h -> 800..2500 m    same quantity                  0.008
#     alarm                                              < 0.25
FSGS_AT_PEAK_MIN = 0.25
DELTA_ACTIVE_MIN = 1e-3      # below this the floor asserts nothing; use the inert arm
FAC_ABSURD = 100.0           # the coarse arm, kept: it costs nothing and it is unambiguous


def floor_health(fl):
    """Is this floor repairing a deficit, or repairing a broken input?

    Takes a most_floor() result. Returns a dict of diagnostics plus `ok` and `alarms`.
    Cheap, pure, and safe to call on every case -- which is the point: the defect it
    exists to catch produced a plausible footprint and no error at all.
    """
    zl = np.asarray(fl["zl"], float)
    base = np.asarray(fl["base"], float)
    wwp = np.asarray(fl["wwp"], float)
    fac = np.asarray(fl["fac"], float)
    delta = np.maximum(np.asarray(fl["sig2"], float) - base, 0.0)
    f_sgs = 1.0 - wwp / np.maximum(base, 1e-30)

    active = fac > 1.0 + 1e-9
    k_fac = int(np.argmax(fac))
    tot = float(delta.sum())
    inflation = delta / np.maximum(base, 1e-30)

    d = dict(fac_min=float(fac.min()), fac_max=float(fac.max()),
             z_fac_max=float(zl[k_fac]), h=float(fl["h"]),
             z_inert=float(zl[active].max()) if active.any() else 0.0,
             inflation_max=float(inflation.max()),
             n_active=int(active.sum()))
    d["z_inert_over_h"] = d["z_inert"] / max(d["h"], 1.0)

    alarms = []
    if tot <= 0.0 or d["inflation_max"] < DELTA_ACTIVE_MIN:
        # THE INERT ARM. A floor that asserts nothing at z/Delta ~ 1 is not a floor that
        # was not needed -- at this receptor the LES resolves 4-10% of sigma_w^2 at 10 m
        # and the correction is never zero in a healthy window. Measured: a neutral
        # production case given h = 2000 m switches the floor OFF entirely, because the
        # resolved-variance peak search runs past the boundary layer and lands above
        # everything the floor would have corrected. Same broken input, opposite symptom.
        d.update(f_sgs_at_peak=float("nan"), z_delta_max=float("nan"),
                 share_resolved=float("nan"))
        alarms.append(f"the floor is INERT (max inflation {d['inflation_max']:.2e} of "
                      f"sigma_w^2). At z/Delta ~ 1 a healthy window always needs some "
                      f"correction, so this is an input fault, not an easy case -- most "
                      f"often h ({d['h']:.0f} m) placing the resolved-variance peak "
                      f"outside the boundary layer.")
    else:
        k_d = int(np.argmax(delta))
        hi = wwp / np.maximum(base, 1e-30) >= 0.8
        d.update(f_sgs_at_peak=float(f_sgs[k_d]), z_delta_max=float(zl[k_d]),
                 share_resolved=float(delta[hi].sum() / tot))
        if d["f_sgs_at_peak"] < FSGS_AT_PEAK_MIN:
            alarms.append(
                f"the floor adds the most variance at z = {d['z_delta_max']:.0f} m, where "
                f"the LES already resolves {100*(1-d['f_sgs_at_peak']):.1f}% of "
                f"sigma_w^2 (f_sgs {d['f_sgs_at_peak']:.3f} < {FSGS_AT_PEAK_MIN}). The "
                f"floor's justification is UNRESOLVED variance, so its peak influence "
                f"cannot sit where the model has resolved the field. Production runs put "
                f"this at 0.37-0.56; the h-fell-through-to-the-domain-top defect put it "
                f"at 0.008. Check st['h'] = {d['h']:.0f} m before trusting this "
                f"footprint.")
    if d["fac_max"] > FAC_ABSURD:
        alarms.append(f"the floor reaches {d['fac_max']:.3g} somewhere in the column; a "
                      f"correction to a sub-grid variance does not have a factor of "
                      f"order 100.")
    d["alarms"] = alarms
    d["ok"] = not alarms
    return d
