"""Backward Lagrangian particle dispersion model driven by FastEddy LES fields.

THE ONE FORMULA THAT MATTERS -- the reverse-time drift.

Naively reversing a Langevin model by substituting (u, t) -> (-u, -t) gives an
ANTI-damped velocity equation that blows up. That substitution is wrong: Thomson
(1987, section 5) shows the time-reversed diffusion picks up an extra term from the
stationary density. For a diffusion dX = A dt + B dW with stationary density p, the
reverse process has drift

    A_hat = -A + (B B^T) grad_X ln p

Applied to the velocity block, with p Gaussian in u with variance sigma^2(x):

    A_hat_i = -a_i - C0 eps u_i / sigma^2
            = -[ -(C0 eps/(2 sigma^2)) u_i + (1/2)(dsigma^2/dx_i + u_i u_j/sigma^2 dsigma^2/dx_j) ]
              - C0 eps u_i / sigma^2
            = -(C0 eps/(2 sigma^2)) u_i - (1/2)(dsigma^2/dx_i + (u_i u_j/sigma^2) dsigma^2/dx_j)

So the reverse-time model keeps the SAME (stable, negative) damping as the forward
model, and ONLY the sigma^2-gradient drift term changes sign. Position and model time
run backward. That is the entire difference. Getting this wrong produces a model that
either diverges (naive reversal) or accumulates particles near the surface (dropping the
gradient term) -- and the second failure is silent and looks like a plausible footprint.

The Langevin model here acts on the SUB-GRID velocity only; the resolved velocity comes
from interpolated LES fields (Weil, Sullivan & Moeng 2004). sigma_s^2 = (2/3) e_sgs and
eps are FastEddy's own, recomputed in lpdm/fields.py with FastEddy's own constants.

SUB-GRID ANISOTROPY (optional, default OFF).
The (2/3) e_sgs split is isotropic, which is the standard WSM04 assumption and is wrong
near a wall: blocking suppresses w relative to u and v. The neutral surface layer has
sigma_u : sigma_v : sigma_w ~ 2.5 : 1.9 : 1.25 (times u*), i.e. variance ratios of
6.25 : 3.61 : 1.5625, so relative to the isotropic (2/3) e the per-component factors are

    r_u = 1.6417,  r_v = 0.9482,  r_w = 0.4102        (sum exactly 3, so e is conserved)

Pass aniso=SURFACE_LAYER_ANISO to switch it on. With CONSTANT ratios the horizontal
sigma^2-gradient drift is unchanged -- r_i cancels between d(sigma_i^2)/dz and 1/sigma_i^2
-- and only the vertical term and the per-component OU timescales change.

Particle state is fp64 throughout (PROJECT_BRIEF.md convention): trajectories integrate for
thousands of steps and fp32 roundoff accumulates into a spurious drift.
"""
from __future__ import annotations

import numpy as np

C0_DEFAULT = 3.0        # Weil, Sullivan & Moeng (2004) for the LES sub-grid Langevin term
KAPPA = 0.4

# Neutral surface-layer variance ratios, normalised so they sum to 3 (isotropic = 1,1,1).
# From sigma_u:sigma_v:sigma_w = 2.5:1.9:1.25 u* (Panofsky & Dutton 1984).
SURFACE_LAYER_ANISO = (6.25 / 3.807500, 3.61 / 3.807500, 1.5625 / 3.807500)


class LPDM:
    def __init__(self, fs, c0=C0_DEFAULT, z_touch=2.0, dt_frac=0.05,
                 dt_min=0.01, dt_max=1.0, seed=0, aniso=None, sgs_scale=1.0):
        self.fs = fs
        self.c0 = float(c0)
        # (3,1) column so it broadcasts against the (3,n) sub-grid velocity block.
        r = np.ones(3) if aniso is None else np.asarray(aniso, dtype=np.float64)
        if abs(r.sum() - 3.0) > 1e-6:
            raise ValueError(f"aniso must sum to 3 (energy conservation), got {r.sum()}")
        self.aniso = r.reshape(3, 1)
        # Uniform multiplier on the sub-grid VARIANCE. Not a physical parameter -- a
        # diagnostic lever, to test whether the near-field footprint error is driven by
        # the LES under-producing sigma_w at the receptor (measured 1.09 u* against the
        # surface-layer 1.25) rather than by how the sub-grid energy is partitioned.
        # Either a scalar or a (z_levels, scale_at_level) pair -- a HEIGHT-DEPENDENT
        # multiplier, interpolated at the particle's height above ground.
        if isinstance(sgs_scale, tuple):
            self.sgs_scale = None
            self.sgs_scale_z, self.sgs_scale_f = (np.asarray(v, dtype=np.float64)
                                                  for v in sgs_scale)
            # d(scale)/dz. THE MODEL IS NOT WELL MIXED WITHOUT IT. The Thomson drift
            # contains d(sigma^2)/dz, and if the variance is rescaled by a HEIGHT-DEPENDENT
            # factor then the scaled variance's gradient is
            #     d/dz [ sc(z) (2/3) e ]  =  sc dsig2dz  +  (2/3) e dsc/dz
            # Using the unscaled dsig2dz leaves the drift too weak to balance the extra
            # mixing the factor introduces, so backward particles get unopposed vertical
            # motion, touch down too often, and the flux-footprint integral climbs past 1.
            # Measured on the flat neutral control: 1.089 at t_back = 900 s, where the
            # answer for a horizontally homogeneous surface is exactly 1.
            self.sgs_scale_dfdz = np.gradient(self.sgs_scale_f, self.sgs_scale_z)
        else:
            self.sgs_scale = float(sgs_scale)
            self.sgs_scale_z = self.sgs_scale_f = None
        self.z_touch = float(z_touch)
        self.dt_frac, self.dt_min, self.dt_max = dt_frac, dt_min, dt_max
        self.rng = np.random.default_rng(seed)
        self.z_ref = float(fs.zk[0])        # lowest LES cell centre
        self.z_top = float(fs.zk[-1])

    # ------------------------------------------------------------------ sampling
    def _local(self, x, y, z, t):
        """Resolved velocity and SGS statistics at the particle positions.

        Below the lowest LES cell centre the LES carries no information, so the
        horizontal resolved wind is continued by the neutral log law anchored at that
        cell, the resolved w is taken to zero linearly at the ground (impermeability),
        and eps follows surface-layer scaling eps ~ 1/z. sigma_s^2 is held at its
        z_ref value, which makes its gradient zero there -- so the sub-layer is
        trivially well mixed and cannot manufacture the near-surface accumulation the
        Stage 4 gate exists to detect.
        """
        fs = self.fs
        fi, fj = fs.hindex(x, y)
        zg = fs.ground(fi, fj)
        # Heights are AGL for the sub-layer treatment but the vertical index map wants
        # absolute height; with terrain the two differ by zg, so keep them separate.
        zagl = np.clip(z - zg, self.z_touch * 0.5, self.z_top - 1.0)
        fk = fs.kindex(zg + np.maximum(zagl, self.z_ref), fi, fj)
        ft = fs.tindex(t)
        u, v, w, e, eps, ds2z = fs.sample(
            ("u", "v", "w", "e", "eps", "dsig2dz"), ft, fk, fj, fi)
        ustar, z0 = fs.sample2d(("ustar", "z0m"), ft, fj, fi)

        below = zagl < self.z_ref
        if below.any():
            z0b = np.maximum(z0[below], 1e-4)
            scal = (np.log(np.maximum(zagl[below], 1.001 * z0b) / z0b)
                    / np.log(self.z_ref / z0b))
            scal = np.clip(scal, 0.0, 1.0)
            u[below] *= scal
            v[below] *= scal
            w[below] *= zagl[below] / self.z_ref
            eps[below] *= self.z_ref / zagl[below]
            ds2z[below] = 0.0

        # Floors, for numerics not physics. Above the boundary layer FastEddy's SGS TKE
        # goes to zero; the Langevin timescale 2 sigma^2/(C0 eps) then becomes 0/0 and the
        # adaptive step collapses to dt_min, burning iterations on particles that are not
        # moving. The floors are far below any turbulent value, so they change nothing
        # where there is turbulence.
        if self.sgs_scale is None:
            sc = np.interp(zagl, self.sgs_scale_z, self.sgs_scale_f)
            dscdz = np.interp(zagl, self.sgs_scale_z, self.sgs_scale_dfdz)
        else:
            sc = self.sgs_scale
            dscdz = 0.0
        # Product rule, so that ds2z is the gradient of the variance the model ACTUALLY
        # uses. With sc = 1 this reduces to the unscaled field exactly.
        ds2z = sc * ds2z + (2.0 / 3.0) * e * dscdz
        if below.any():
            # Same reasoning as above: inside the sub-layer sigma^2 is held at its z_ref
            # value, so its gradient is zero -- including the rescaling's contribution.
            ds2z[below] = 0.0
        sig2 = np.maximum(sc * (2.0 / 3.0) * e, 1e-6)
        eps = np.maximum(eps, 1e-8)
        return u, v, w, sig2, eps, ds2z, ustar

    # ------------------------------------------------------------------ integrator
    def run(self, x, y, z, t, direction=-1, t_limit=600.0, max_iter=200_000,
            reflect_touchdown=True, record_touchdown=True, z_ceil=None,
            max_disp=None, progress=None):
        """Integrate an ensemble. direction=-1 backward (footprints), +1 forward.

        Returns a dict of arrays. Touchdowns are recorded as (particle, x, y, |w|)
        in UNWRAPPED horizontal coordinates so a footprint can extend past one
        periodic domain length.
        """
        fs = self.fs
        n = len(x)
        x = np.asarray(x, dtype=np.float64).copy()
        y = np.asarray(y, dtype=np.float64).copy()
        z = np.asarray(z, dtype=np.float64).copy()
        t = np.full(n, float(t), dtype=np.float64) if np.isscalar(t) else \
            np.asarray(t, dtype=np.float64).copy()
        idx = np.arange(n)
        elapsed = np.zeros(n, dtype=np.float64)
        z_ceil = self.z_top if z_ceil is None else float(z_ceil)
        # Horizontal domains are periodic, so a backward trajectory that travels more than
        # one domain length re-enters the SAME turbulence it already sampled. max_disp
        # retires a particle once its unwrapped displacement from the release point exceeds
        # a limit, which is how wrap-around double counting is tested for and, if present,
        # removed.
        x0r, y0r = x.copy(), y.copy()
        # Final state of EVERY particle, not just the survivors. Particles retire at
        # different iterations (each carries its own adaptive step), so the working
        # arrays are compacted as they go and finished states are written back here.
        fx = np.full(n, np.nan); fy_ = np.full(n, np.nan); fz = np.full(n, np.nan)
        ft_ = np.full(n, np.nan); fe = np.full(n, np.nan)

        # SGS velocity initialised from the local, per-component SGS variance
        _, _, _, sig2, _, _, _ = self._local(x, y, z, t)
        us = self.rng.normal(0.0, 1.0, (3, n)) * np.sqrt(self.aniso * sig2)

        td_p, td_x, td_y, td_w, td_t = [], [], [], [], []
        rel_uvw = None
        sgn = float(direction)

        for it in range(max_iter):
            if len(idx) == 0:
                break
            U, V, W, sig2, eps, ds2z, ustar = self._local(x, y, z, t)
            # Per-component variance and Langevin timescale. With aniso=None these are
            # three identical rows and the model is bit-identical to the isotropic one.
            sig2_i = self.aniso * sig2                      # (3, n)
            sig_i = np.sqrt(sig2_i)
            TL_i = 2.0 * sig2_i / (self.c0 * eps)
            sig = sig_i[2]                                  # kept for the touchdown logic
            # the adaptive step must respect the FASTEST-decorrelating component
            TL = TL_i.min(axis=0)
            dt = np.clip(self.dt_frac * TL, self.dt_min, self.dt_max)
            dt = np.minimum(dt, np.maximum(t_limit - elapsed, 0.0))

            if it == 0 and rel_uvw is None:
                # All THREE components at release. The flux weight is frame-dependent:
                # a real EC tower double-rotates into the streamline frame, and over a
                # slope the vertical there is a mix of model-frame w and horizontal wind.
                rel_uvw = np.vstack([(U + us[0]).copy(), (V + us[1]).copy(),
                                     (W + us[2]).copy()])

            # exact Ornstein-Uhlenbeck update of the linear (damping + noise) part,
            # per component -- each has its own variance and hence its own timescale
            a_i = np.exp(-dt / TL_i)
            b_i = sig_i * np.sqrt(np.maximum(1.0 - a_i * a_i, 0.0))
            xi = self.rng.standard_normal((3, len(idx)))
            us = us * a_i + b_i * xi

            # sigma^2-gradient drift. Sign flips under time reversal; see module docstring.
            # Vertical component only: the flat spin-up is horizontally homogeneous and
            # in the terrain case the vertical gradient still dominates by ~dx/dz.
            # d(sigma_w^2)/dz carries the aniso factor; the horizontal terms do not,
            # because r_i cancels between d(sigma_i^2)/dz and 1/sigma_i^2 (see docstring).
            drift_w = 0.5 * self.aniso[2, 0] * ds2z * (1.0 + us[2] * us[2] / sig2_i[2])
            drift_u = 0.5 * ds2z * (us[0] * us[2] / sig2)
            drift_v = 0.5 * ds2z * (us[1] * us[2] / sig2)
            us[0] += sgn * drift_u * dt
            us[1] += sgn * drift_v * dt
            us[2] += sgn * drift_w * dt
            # cap at 5 sigma_i: a rare large-|u|/sigma^2 excursion in the cross term can
            # otherwise run away between steps without changing the statistics we want.
            np.clip(us, -5.0 * sig_i, 5.0 * sig_i, out=us)

            wtot = W + us[2]
            utot, vtot = U + us[0], V + us[1]
            # Rate of change of height ABOVE GROUND, which is what a touchdown is. Over
            # sloping terrain a particle can lose height-above-surface purely because the
            # ground rises under it, with w near zero -- and the 2/|w| touchdown weight
            # then explodes. Measured on the flat case this changes nothing (slope is 0);
            # over the Stage 6 terrain it is the difference between a sane footprint and
            # one whose integral goes NEGATIVE because a single touchdown carried a weight
            # of 2e5.
            sx, sy = fs.ground_slope(*fs.hindex(x, y))
            w_agl = wtot - utot * sx - vtot * sy
            x += sgn * (U + us[0]) * dt
            y += sgn * (V + us[1]) * dt
            znew = z + sgn * wtot * dt
            t += sgn * dt
            elapsed += dt

            zg = fs.ground(*fs.hindex(x, y))
            hit = znew - zg < self.z_touch
            if hit.any():
                if record_touchdown:
                    td_p.append(idx[hit]); td_x.append(x[hit]); td_y.append(y[hit])
                    td_w.append(np.abs(w_agl[hit])); td_t.append(elapsed[hit].copy())
                if reflect_touchdown:
                    znew[hit] = 2.0 * (zg[hit] + self.z_touch) - znew[hit]
                    us[2][hit] = -us[2][hit]
                else:
                    znew[hit] = zg[hit] + self.z_touch
            top = znew > z_ceil
            if top.any():
                znew[top] = 2.0 * z_ceil - znew[top]
                us[2][top] = -us[2][top]
            z = znew

            disp = np.hypot(x - x0r[idx], y - y0r[idx])
            done = elapsed >= t_limit
            if max_disp is not None:
                done |= disp > max_disp
            if not reflect_touchdown:
                done |= hit
            if done.any():
                fin = idx[done]
                fx[fin] = x[done]; fy_[fin] = y[done]; fz[fin] = z[done]
                ft_[fin] = t[done]; fe[fin] = elapsed[done]
                keep = ~done
                x, y, z, t, elapsed, idx = (x[keep], y[keep], z[keep], t[keep],
                                            elapsed[keep], idx[keep])
                us = us[:, keep]
            if progress and it % 500 == 0:
                progress(it, len(idx))

        if len(idx):                      # hit max_iter before retiring
            fx[idx] = x; fy_[idx] = y; fz[idx] = z; ft_[idx] = t; fe[idx] = elapsed
        cat = lambda L: np.concatenate(L) if L else np.zeros(0)
        return dict(n=n, rel_u=rel_uvw[0], rel_v=rel_uvw[1], rel_w=rel_uvw[2],
                    w_release=rel_uvw[2],
                    td_particle=cat(td_p).astype(np.int64), td_x=cat(td_x),
                    td_y=cat(td_y), td_w=cat(td_w), td_t=cat(td_t),
                    x=fx, y=fy_, z=fz, t=ft_, elapsed=fe,
                    n_unfinished=len(idx), iters=it + 1)
