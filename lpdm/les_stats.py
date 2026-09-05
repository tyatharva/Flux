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

from .dumpsrc import MemDump, open_dump, step_of

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


def surface_layer_top(tk, z, damp_frac=DAMP_FRAC):
    """Index of the minimum terminating the SURFACE-ATTACHED turbulent layer.

    Returns `(k_sfc, k_sfc_min, top)`. `k_sfc_min` is `top - 1` -- i.e. bounds nothing away
    -- unless the column holds a SECOND turbulent layer that out-energises the first, which
    is the only structure a boundary-layer depth cannot be measured through.

    WHY THE TEST IS "OUT-ENERGISES" AND NOT "FIRST LOCAL MINIMUM". The first strict local
    minimum above the surface peak is not a layer boundary; on a real profile it is usually
    noise. Measured: taking it as the search ceiling moved h on 15 of the 47 footprint
    profiles on disk, by up to 331 m, on profiles with no wave layer at all -- it was
    cutting ordinary monotone decay short at the first wiggle.

    What actually distinguishes a wave layer is the thing that made it a bug in the first
    place: it carries MORE resolved TKE than the boundary layer under it, so the column's
    global maximum lands in it. So the structure this looks for is exactly that -- a
    surface-attached peak, a trough, and then a maximum that exceeds the surface peak --
    and the boundary is the trough between the two. When the global maximum IS the
    surface-attached peak, or the profile between them is monotone (the two are the same
    feature seen through smoothing), there is one layer and nothing is bounded away.
    """
    tk = np.asarray(tk, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    top = int(np.clip(np.searchsorted(z, damp_frac * z[-1]), 3, len(z)))
    sm = np.convolve(tk[:top], np.ones(3) / 3.0, mode="same")
    if top > 2:
        sm[0], sm[-1] = tk[0], tk[top - 1]
    k_sfc = int(np.argmax(sm[:max(3, top // 3)]))
    k_gmax = int(np.argmax(tk[:top]))
    if k_gmax <= k_sfc:
        return k_sfc, top - 1, top          # the strongest turbulence IS the surface layer
    k_tr = k_sfc + int(np.argmin(tk[k_sfc:k_gmax + 1]))
    if k_tr == k_sfc or k_tr == k_gmax:
        return k_sfc, top - 1, top          # monotone between them: one layer, not two
    return k_sfc, k_tr, top


def bl_depth(tk, z, thresh=None, frac=0.05, damp_frac=DAMP_FRAC, return_info=False):
    """Boundary-layer depth from a TKE profile that need not decay monotonically.

    `thresh` -- an absolute TKE threshold, m2/s2. If None, `frac` x the profile's own peak
    is used instead (the definition lpdm/les_stats.py has always produced as the corpus
    input `h`, and the one bin/pick_seed.py matches seeds in).

    THE DEPTH IS THE SURFACE-ATTACHED BOUNDARY LAYER, AND THE SEARCH CANNOT LEAVE IT.
    `surface_layer_top` bounds the whole estimate at the first minimum terminating the
    surface-attached layer; the peak is the largest value at or below that bound, and the
    threshold search runs from it down to the layer's own decay minimum. Returns the
    crossing height if the threshold is met on the way down, and the minimum's height if it
    is not -- never the top of the domain, and never a level in the free atmosphere.

    With `return_info=True` returns `(h, info)`, `info` carrying the surface layer's
    geometry and whether the profile's GLOBAL maximum was outside it.
    """
    tk = np.asarray(tk, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    k_sfc, k_sfc_min, top = surface_layer_top(tk, z, damp_frac=damp_frac)

    # === THE SEARCH BAND IS THE SURFACE-ATTACHED LAYER, NOT THE WHOLE COLUMN ============
    #
    # This used to take the GLOBAL argmax over [0, top) and search down from there, and on
    # case_2023112120 -- a NEUTRAL case -- the global argmax was in the free atmosphere.
    # Resolved TKE peaks at 1.01 m2/s2 at 39 m, decays monotonically to 0.28 at 448 m --
    # an entirely ordinary neutral boundary layer -- and then RISES to 2.42 at 1887 m, more
    # than twice anything in the boundary layer, with e_sgs an order of magnitude smaller.
    # That is internal-wave activity in the stable free atmosphere. Searching down from it
    # returned h = 2372 m, which set the sigma_w floor's mixed-layer blend and drove it to
    # a factor of 1.2e+04. Nothing downstream went non-finite; every number stayed
    # plausible, which is why bin/corpus_monitor.py's G1 was what caught it.
    #
    # The previous fix REFUSED such a profile. That was right as a stop-gap and wrong as an
    # answer: it blocked the entire neutral half of the corpus over a quantity the profile
    # does in fact determine. A neutral boundary layer under a wave layer still has a
    # depth -- it is the surface-attached layer, and the level where that layer stops
    # decaying is exactly where it ends. So this DECIDES rather than refusing, and it
    # decides structurally: no height threshold, no constant picked rather than derived.
    #
    # On case_2023112120 the answer is 448 m (the k=42 minimum), not 760 m -- which is a
    # point on the way back UP through the wave layer's lower flank -- and not 2372 m.
    #
    # ORDINARY PROFILES ARE UNTOUCHED, and that is asserted rather than argued: the global
    # maximum of a profile with no wave layer already lies inside the surface-attached
    # layer, so the band bounds nothing away and every step below is the arithmetic that
    # was there before. bin/test_bl_depth.py re-derives h for all 47 footprint profiles on
    # disk -- 16, 24 and 30 m grids, neutral and convective -- and requires EXACT equality
    # with the h each run stored.
    k_pk = int(np.argmax(tk[:k_sfc_min + 1]))
    aloft = int(np.argmax(tk[:top])) > k_sfc_min

    k_min = k_pk + int(np.argmin(tk[k_pk:k_sfc_min + 1]))
    if k_min <= k_pk:
        h = float(z[k_pk])
    else:
        t = float(thresh) if thresh is not None else float(frac) * float(tk[k_pk])
        seg = tk[k_pk:k_min + 1]
        ab = np.where(seg < t)[0]
        h = float(z[k_pk + ab[0]]) if len(ab) else float(z[k_min])
    if not return_info:
        return h
    return h, {
        "h": h,
        "k_peak": k_pk, "z_peak_m": float(z[k_pk]), "tke_peak": float(tk[k_pk]),
        "k_surface_layer_top": int(k_sfc_min),
        "z_surface_layer_top_m": float(z[k_sfc_min]),
        "tke_surface_layer_top": float(tk[k_sfc_min]),
        "k_surface_peak": int(k_sfc),
        "global_max_above_surface_layer": bool(aloft),
        "z_global_max_m": float(z[int(np.argmax(tk[:top]))]),
        "tke_global_max": float(np.max(tk[:top])),
    }


def zlevels_of(paths):
    """The static level-height column, from the first dump in the series that carries it.

    Level heights are STATIC, so they are read ONCE rather than from every dump -- but NOT
    necessarily from paths[0]. Under ioLPDMmode the coordinate geometry goes to the first
    file of a RUN and to the ioLPDMfullFrq multiples; a target case runs the adjustment and
    the window as one invocation and DELETES the adjustment dumps, so the run's first file
    is routinely gone by the time this reads. bin/run_window.sh sets ioLPDMfullFrq so the
    first SURVIVING dump is full-form and asserts that it is, which makes paths[0] correct
    in production -- but lpdm/fields.py already had to learn to search rather than assume,
    and a caller that subsamples the series (bin/stage4_wellmixed.py takes paths[::n])
    should not depend on which end it kept.
    """
    for _p in paths:
        with open_dump(_p) as _ds:
            if "zPos" in _ds.variables:
                return np.squeeze(np.asarray(_ds["zPos"][:], dtype=np.float64))[:, 0, 0]
    raise KeyError(
        "no dump in this series carries zPos, so the receptor level cannot be "
        "located. Under ioLPDMmode only the first file of a run and the "
        "ioLPDMfullFrq multiples do -- see bin/run_window.sh.")


class WindowAccumulator:
    """`window_stats` as a ONE-PASS accumulator, so a window need not be held in RAM.

    WHY THIS SHAPE. `window_stats(paths, k)` opened every dump a SECOND time, after
    `FieldSet` had already opened every one to build its cache. With netCDF handles that
    costs a re-read; with the in-process hand-off it costs 19.7 GB, because the snapshots
    are `MemDump`s and a second pass means nothing can be released until both passes are
    done. Streaming needs the two passes fused into one.

    **The estimator is not re-implemented and is not re-ordered.** `window_stats` below is
    now a thin loop over this class, so there is exactly one implementation of the
    arithmetic and the batch and streamed results are bit-identical rather than
    approximately equal -- which `bin/test_streaming.py` asserts, at zero tolerance,
    because there is no physics between the two.

    `k_recept` may be FRACTIONAL. It has to be, once the surface build can raise topoPos by
    the displacement height over the array: the receptor is pinned to a fixed height above
    BARE GROUND, so over a raised patch it sits between two model levels rather than on
    one. The receptor-level moments are taken from the linearly interpolated FIELD, which
    is exactly what the LPDM's own 4-D interpolation does at that height -- interpolating
    the finished variances instead would be a different quantity, which is also why this
    class needs `k_recept` UP FRONT and cannot defer the level choice to `finish()`.
    """

    def __init__(self, z, k_recept):
        self.z = np.asarray(z, dtype=np.float64)
        kf = float(k_recept)
        k0 = int(np.floor(kf))
        fr = kf - k0
        nz = len(self.z)
        self.kf = kf
        self.k0 = int(np.clip(k0, 0, nz - 1))
        self.k1 = min(self.k0 + 1, nz - 1)
        self.f1 = fr if self.k1 > self.k0 else 0.0
        self.U = self.V = 0.0
        self.uu = self.vv = self.ww = self.uv = 0.0
        self.esgs = 0.0
        self.tke_prof = None
        self.ww_prof = None       # resolved sigma_w^2(z), horizontal variance per level
        self.esgs_prof = None     # mean sub-grid TKE(z)
        self.zlev = None
        self.ust = self.hfx = self.th0 = 0.0
        self.n = 0
        # PER-DUMP DIRECTION, SO THE DRIFT INSIDE THE WINDOW CAN BE MEASURED AT ALL.
        # The seed library's dominant skill axis is direction, and the one thing measured
        # about it -- that the 30-minute adjustment WIDENS a gap rather than closing it --
        # rests on comparing a window MEAN against a requested value. That says the gap
        # moved but not how fast, and the window's own fields are deleted at the end of
        # every case, so it cannot be recovered afterwards. Two floats per dump make the
        # rate recoverable from the training record instead of from fields that no longer
        # exist.
        self.uv_series, self.step_series = [], []

    def _lev(self, a):
        """The receptor-level 2-D slice of a 3-D field."""
        return (1.0 - self.f1) * a[self.k0] + self.f1 * a[self.k1]

    def add(self, ds, step=None):
        """Accumulate one OPEN dump. `ds` is whatever `open_dump` returned."""
        z = self.z
        g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float64))
        u, v, w = g("u"), g("v"), g("w")
        e = np.maximum(g("TKE_0"), 0.0)
        self.ust += float(g("fricVel").mean())
        # THE SURFACE FLUX IS DERIVED PER CELL FOR EVERY DUMP, AND THE BRANCH THAT
        # USED htFlux WHEN IT HAPPENED TO BE PRESENT IS GONE.
        #
        # It read "htFlux is not written by the patched ioLPDMmode, so fall back to
        # invOblen", and that premise was stale: ioLPDMfullFrq writes a FULL dump at
        # every multiple of its setting, and a full dump carries htFlux. So a lean
        # window silently mixed TWO estimators -- measured on case_2023052519, 2 of 12
        # sampled dumps took the htFlux branch and 10 took the derived one -- and the
        # window mean was a mean of neither. The two agree to 1.3e-7 here, so nothing
        # downstream was ever wrong by a visible amount; what was wrong is that the
        # estimator depended on the IO MODE, which is the "a diagnostic is only as
        # scale-free as its reference" rule wearing a different hat. Found by
        # bin/test_ringsrc.py, because the in-process ring carries no htFlux and so
        # could not reproduce the mixture.
        #
        # Deriving everywhere is not a compromise: as the note below says, the
        # per-cell product IS htFlux_c, so this is the same quantity computed the same
        # way for every dump under every output mode.
        #
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
        self.hfx += float((-(us_ ** 3) * th_ * iL_ / (KAPPA * G)).mean())
        self.th0 += float(g("theta")[0].mean())
        lev = self._lev
        ur, vr, wr = lev(u), lev(v), lev(w)
        Uk, Vk = ur.mean(), vr.mean()
        self.U += Uk
        self.V += Vk
        self.uu += ((ur - Uk) ** 2).mean()
        self.vv += ((vr - Vk) ** 2).mean()
        self.uv += ((ur - Uk) * (vr - Vk)).mean()
        self.esgs += float(lev(e).mean())
        self.ww += ((wr - wr.mean()) ** 2).mean()
        pr = lambda a: a - a.mean(axis=(-2, -1), keepdims=True)
        t = 0.5 * ((pr(u) ** 2) + (pr(v) ** 2) + (pr(w) ** 2)).mean(axis=(-2, -1))
        self.tke_prof = t if self.tke_prof is None else self.tke_prof + t
        wp = (pr(w) ** 2).mean(axis=(-2, -1))
        ep = e.mean(axis=(-2, -1))
        self.ww_prof = wp if self.ww_prof is None else self.ww_prof + wp
        self.esgs_prof = ep if self.esgs_prof is None else self.esgs_prof + ep
        self.zlev = z
        self.n += 1
        self.uv_series.append((float(Uk), float(Vk)))
        # THE STEP IS PASSED IN, NOT PARSED HERE, and the reason is a bug that produced a
        # plausible wrong number. The batch loop below used to parse it off the handle
        # inside this body, and an earlier version of that used `_p` -- the variable the
        # zPos SEARCH left bound to the first file it found -- so every dump was stamped
        # with the same step, the time axis had zero span, and the fitted direction drift
        # came out +19.3 and +59.9 deg/h on two cases whose direction was actually BACKING
        # by 14.9 and 7.2 deg.
        #
        # AND A HANDLE IS NOT ALWAYS A PATH: the in-process ring hands these readers
        # MemDump objects, whose str() has no trailing ".<step>", so a rsplit raised, the
        # except swallowed it, and the series silently became 0,1,2,... -- a time axis
        # spanning 9 STEPS instead of 82,134, and every rate off by four orders of
        # magnitude. bin/test_dumpsrc.py is what caught it. The fallback for a handle that
        # genuinely has no step is kept, and it SAYS SO instead of quietly returning an
        # index.
        if step is None:
            if self.n == 1:
                print(f"  WARNING: this dump handle carries no timestep; the step series "
                      f"falls back to the dump INDEX, so any rate derived from it is in "
                      f"units of dumps, not steps.")
            self.step_series.append(self.n - 1)
        else:
            self.step_series.append(int(step))

    def finish(self):
        """The window's statistics. Identical to what the batch loop always returned."""
        n = self.n
        if not n:
            raise ValueError("no dumps were accumulated; there is no window to describe")
        z = self.z
        U, V = self.U / n, self.V / n
        uu, vv, ww, uv = self.uu / n, self.vv / n, self.ww / n, self.uv / n
        esgs = self.esgs / n
        sgs = (2.0 / 3.0) * esgs        # isotropic sub-grid variance per component
        ust, hfx, th0 = self.ust / n, self.hfx / n, self.th0 / n
        tke_prof = self.tke_prof / n
        ww_prof = self.ww_prof / n
        esgs_prof = self.esgs_prof / n

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
        # 5% of the profile's own peak, bounded by the decay minimum -- see bl_depth above
        # for why the bound is not optional. This is the corpus input `h` and the currency
        # bin/pick_seed.py matches seeds in; the seed GATE uses a fixed threshold instead,
        # because it scores a trend and a peak-normalised threshold moves with the peak
        # (docs/reference/fasteddy-traps.md 16).
        h, h_info = bl_depth(tke_prof, z, frac=0.05, return_info=True)
        # AND WHEN A WAVE LAYER WAS BOUNDED AWAY, SAY SO IN THE RECORD. h is then the
        # surface-attached depth and is correct, but the column also holds a second, more
        # energetic turbulent layer -- a fact about that window which a consumer filtering
        # or weighting the corpus should be able to see without re-deriving the profile.
        if h_info["global_max_above_surface_layer"]:
            print(f"  h = {h:.0f} m is the SURFACE-ATTACHED depth: this column's global "
                  f"resolved-TKE maximum is {h_info['tke_global_max']:.2f} m2/s2 at "
                  f"{h_info['z_global_max_m']:.0f} m, above the surface layer's own "
                  f"{h_info['tke_peak']:.2f} at {h_info['z_peak_m']:.0f} m. That is wave "
                  f"activity in the free atmosphere and it is excluded from h.")
        # AND IT MUST NEVER BE THE TOP OF THE COLUMN. `h` is a corpus INPUT and it also
        # sets the sigma_w floor's mixed-layer blend, so a fallback value does not announce
        # itself anywhere downstream -- it just makes a plausible footprint out of the
        # wrong closure. bl_depth cannot return z[-1] any more; this asserts that it did
        # not.
        if h >= 0.98 * float(z[-1]):
            raise ValueError(
                f"h came out {h:.0f} m against a column top of {z[-1]:.0f} m. That is the "
                f"estimator failing to find a boundary layer, not a 2.5 km one: it would "
                f"go into the training record as a feature and into the sigma_w floor as "
                f"the mixed-layer blend height. See lpdm/les_stats.py:bl_depth.")
        L = (-ust ** 3 * th0 / (KAPPA * G * hfx)) if abs(hfx) > 1e-6 else np.inf
        return dict(z=z, z_recept=float(self._lev(z)), k_recept=self.kf, u_mean=spd,
                    wdir=float(wdir),
                    sigma_u=float(np.sqrt(uu + sgs)), sigma_v=sig_v,
                    sigma_w=float(np.sqrt(ww + sgs)),
                    sigma_v_resolved=float(np.sqrt(sig_v_res)),
                    sigma_w_resolved=float(np.sqrt(ww)), e_sgs=float(esgs),
                    ustar=float(ust), htFlux=float(hfx), theta0=float(th0),
                    L=float(L), h=h, h_info=h_info, tke_prof=tke_prof, n_dumps=n,
                    wdir_per_dump=[float((270.0 - np.degrees(np.arctan2(v_, u_))) % 360.0)
                                   for (u_, v_) in self.uv_series],
                    step_per_dump=[int(x) for x in self.step_series],
                    ww_prof=ww_prof, esgs_prof=esgs_prof, zlev=self.zlev,
                    U=float(U), V=float(V))


def window_stats(paths, k_recept):
    """Ensemble statistics over a series of dumps, at the receptor level.

    A THIN LOOP OVER `WindowAccumulator`, so the batch and the streamed paths are the same
    arithmetic in the same order rather than two implementations that agree. The public
    signature and the returned dict are unchanged; every existing caller and gate keeps
    working, and `bin/test_streaming.py` asserts the two routes are bit-identical.
    """
    acc = WindowAccumulator(zlevels_of(paths), k_recept)
    for p in paths:
        with open_dump(p) as ds:
            try:
                step = step_of(p)
            except (AttributeError, ValueError, IndexError):
                step = None
            acc.add(ds, step=step)
    return acc.finish()
