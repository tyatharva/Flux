"""Touchdown weighting -> 2-D flux footprint (Flesch, Wilson & Yee 1995; Flesch 1996).

The estimator, and why it has the form it does.

Backward trajectories are released at the receptor and each surface touchdown is
recorded with the vertical speed |w_td| it had on arrival. For a surface source of
strength Q [kg m^-2 s^-1] the concentration at the receptor is the Monte-Carlo mean

    C/Q = (1/N) sum_i sum_{touchdowns of i} 2/|w_td|

The 2/|w_td| weight is a residence-time contribution; the 2 is the reflection.

The FLUX at the receptor is <w' c'>, so the same ensemble is reweighted by the
vertical velocity each trajectory had AT THE RECEPTOR when it was released:

    F/Q = (1/N) sum_i w_release,i * [ sum_{touchdowns of i} 2/|w_td| ]

That is an unbiased estimate of E[w * (c/Q)] because w_release is drawn from the
actual joint distribution at the receptor (resolved LES value plus a sub-grid draw),
not from an assumed Gaussian. Trajectories arriving with w > 0 came from below and
carry surface-influenced air; those arriving with w < 0 came from aloft. Their
touchdown densities differ, and the signed difference IS the flux footprint -- so
the estimator must NOT use |w_release|.

Splitting the touchdowns by location gives the footprint per unit area. Its integral
over all space must be 1 for a horizontally uniform source: every bit of surface flux
crosses the measurement height in steady state. The shortfall below 1 is the fraction
of influence truncated by the finite backward integration time, and is reported rather
than normalised away -- normalising would hide a too-short t_limit.
"""
from __future__ import annotations

import numpy as np


class FootprintGrid:
    def __init__(self, x0, x1, y0, y1, res):
        self.res = float(res)
        self.xe = np.arange(x0, x1 + res, res)
        self.ye = np.arange(y0, y1 + res, res)
        self._finish()

    @classmethod
    def from_edges(cls, xe, ye):
        """Build on explicit cell edges.

        Used for the STATIC raster, whose cells must coincide exactly with the LES
        columns -- the surface masks, the roughness map and the CNF target all live on
        that indexing, and an independently constructed grid that is merely close would
        smear a 24 m array patch across two cells.
        """
        g = cls.__new__(cls)
        g.xe = np.asarray(xe, dtype=float)
        g.ye = np.asarray(ye, dtype=float)
        g.res = float(g.xe[1] - g.xe[0])
        g._finish()
        return g

    def _finish(self):
        self.xc = 0.5 * (self.xe[:-1] + self.xe[1:])
        self.yc = 0.5 * (self.ye[:-1] + self.ye[1:])
        self.area = float(np.diff(self.xe).mean() * np.diff(self.ye).mean())
        self.flux = np.zeros((len(self.yc), len(self.xc)))
        self.conc = np.zeros((len(self.yc), len(self.xc)))
        self.n_particles = 0
        self.sum_flux_all = 0.0     # includes touchdowns outside the grid
        self.sum_conc_all = 0.0
        self.n_td = 0
        self.top_w = []          # largest per-touchdown |flux| weights seen

    def add(self, res, x_recept, y_recept, w_floor=0.02):
        """Accumulate one LPDM ensemble, in receptor-relative coordinates.

        `w_floor` bounds the 2/|w_td| weight. The physical estimator has no floor, but the
        weight's tail is heavy: a particle that barely crosses the touchdown level gets an
        arbitrarily large weight, and the estimator's variance is formally infinite. The
        default 0.02 m/s is ~5% of sigma_w here, so it leaves essentially every physical
        touchdown untouched while capping a single one at 100 s/m. `tail_concentration()`
        reports what the tail is still worth, so the choice is measured rather than assumed.
        """
        w_rel = res["w_release"]
        if len(res["td_x"]) == 0:
            self.n_particles += res["n"]
            return
        wt_c = 2.0 / np.maximum(res["td_w"], w_floor)
        wt_f = w_rel[res["td_particle"]] * wt_c
        dx = res["td_x"] - x_recept
        dy = res["td_y"] - y_recept
        self.sum_flux_all += wt_f.sum()
        self.sum_conc_all += wt_c.sum()
        self.n_td += len(wt_f)
        for arr, wt in ((self.flux, wt_f), (self.conc, wt_c)):
            h, _, _ = np.histogram2d(dy, dx, bins=(self.ye, self.xe), weights=wt)
            arr += h
        self.n_particles += res["n"]
        # The 2/|w_td| weight has a heavy tail: a particle that barely crosses the
        # touchdown level contributes 2/|w| with |w| near zero. If a handful of
        # touchdowns carried most of the footprint the estimate would be meaningless,
        # so the largest weights are kept and reported rather than assumed harmless.
        k = min(len(wt_f), 2000)
        self.top_w = sorted(self.top_w + list(np.abs(np.partition(wt_f, -k)[-k:])),
                            reverse=True)[:2000]

    def tail_concentration(self):
        """Fraction of the total |flux| weight carried by the largest touchdowns."""
        if not self.n_td:
            return {}
        tw = np.array(self.top_w)
        tot = np.abs(self.flux).sum() or 1.0
        n01 = max(1, int(0.001 * self.n_td))
        return dict(max_weight=float(tw[0]),
                    top0p1pct_share=float(tw[:n01].sum() / tot),
                    n_touchdown=self.n_td)

    # ------------------------------------------------------------------ results
    def normalised(self, which="flux"):
        """Footprint density [m^-2]: integrates to the captured fraction of influence."""
        a = (self.flux if which == "flux" else self.conc)
        return a / max(self.n_particles, 1) / self.area

    def integral(self, which="flux"):
        """Integral over the FOOTPRINT GRID only."""
        return float(self.normalised(which).sum() * self.area)

    def integral_all(self, which="flux"):
        """Integral including touchdowns that fell outside the grid."""
        s = self.sum_flux_all if which == "flux" else self.sum_conc_all
        return float(s / max(self.n_particles, 1))

    def crosswind_integrated(self, which="flux"):
        return self.normalised(which).sum(axis=0) * self.res

    def metrics_map(self, which="flux"):
        """Metrics for a NORTH-UP static raster, where "upwind distance" has no meaning.

        The wind-frame quantities (peak upwind distance, crosswind-integrated profile) are
        accumulated separately as a 1-D histogram of the touchdowns' upwind coordinate --
        exactly, from the touchdowns themselves, rather than by rotating this raster, which
        would blur the near field it is meant to resolve.
        """
        f = self.normalised(which)
        tot = f.sum()
        if tot <= 0:
            return dict(centroid_e=np.nan, centroid_n=np.nan, centroid_dist=np.nan,
                        centroid_bearing=np.nan, area80_cells=0, area80_ha=np.nan,
                        peak_e=np.nan, peak_n=np.nan, degenerate=True)
        e = float((f.sum(axis=0) * self.xc).sum() / tot)
        n = float((f.sum(axis=1) * self.yc).sum() / tot)
        m80 = source_area_mask(f, 0.80)
        m50 = source_area_mask(f, 0.50)
        jp, ip = np.unravel_index(int(np.argmax(f)), f.shape)
        return dict(centroid_e=e, centroid_n=n,
                    centroid_dist=float(np.hypot(e, n)),
                    # bearing FROM the tower TO the centroid, degrees clockwise from north
                    centroid_bearing=float(np.degrees(np.arctan2(e, n)) % 360.0),
                    area80_cells=int(m80.sum()),
                    area80_ha=float(m80.sum() * self.area / 1e4),
                    area50_ha=float(m50.sum() * self.area / 1e4),
                    peak_e=float(self.xc[ip]), peak_n=float(self.yc[jp]),
                    peak_value=float(f[jp, ip]))

    def metrics(self, which="flux"):
        f = self.normalised(which)
        tot = f.sum()
        if tot <= 0:                       # pathological (e.g. a runaway weight) -- say so
            return dict(peak_x=np.nan, centroid_x=np.nan, centroid_y=np.nan,
                        x80_near=np.nan, x80_far=np.nan, area80_cells=0, thr80=np.nan,
                        degenerate=True)
        fy = f.sum(axis=0)
        peak = self.xc[int(np.argmax(fy))]
        xbar = float((f.sum(axis=0) * self.xc).sum() / tot)
        ybar = float((f.sum(axis=1) * self.yc).sum() / tot)
        # upwind extent containing 80% of the (positive) source area
        flat = np.sort(f.ravel())[::-1]
        cum = np.cumsum(flat) / tot
        thr = flat[np.searchsorted(cum, 0.80)] if cum[-1] >= 0.80 else flat[-1]
        mask = f >= thr
        xs = self.xc[np.where(mask.any(axis=0))[0]]
        return dict(peak_x=float(peak), centroid_x=xbar, centroid_y=ybar,
                    x80_near=float(xs.min()) if len(xs) else np.nan,
                    x80_far=float(xs.max()) if len(xs) else np.nan,
                    area80_cells=int(mask.sum()), thr80=float(thr))


def source_area_mask(f, frac=0.80):
    """Boolean mask of the smallest set of cells holding `frac` of the footprint."""
    tot = f.sum()
    if tot <= 0:
        return np.zeros(f.shape, dtype=bool)
    flat = np.sort(f.ravel())[::-1]
    cum = np.cumsum(flat) / tot
    thr = flat[np.searchsorted(cum, frac)] if cum[-1] >= frac else flat[-1]
    return f >= thr


def source_area_overlap(a, b, frac=0.80):
    """Fraction overlap of the two footprints' `frac` source areas (Jaccard index)."""
    def mask(f):
        tot = f.sum()
        flat = np.sort(f.ravel())[::-1]
        cum = np.cumsum(flat) / tot
        thr = flat[np.searchsorted(cum, frac)] if cum[-1] >= frac else flat[-1]
        return f >= thr
    ma, mb = mask(a), mask(b)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / max(union, 1))
