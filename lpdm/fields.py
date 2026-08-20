"""Load a FastEddy dump series into a time-indexed field cache for the LPDM.

Design notes that are load-bearing:

* **FastEddy's netCDF output is collocated and primitive.** Every prognostic field is
  written at the cell centre on the same (z,y,x) index space as `zPos`, and the writer
  emits primitive variables, not density-weighted ones -- verified against the config
  (`theta` = 300.000 K at the surface where `temp_grnd = 300.0`, `u` = 10.0000 m/s where
  `U_g = 10.0`). So no de-densifying, and no staggered-grid offsets.

* **The vertical coordinate is terrain-following and analytically invertible.** FastEddy's
  `zDeform` (SRC/GRID/grid.c) is
      z(zeta) = F(zeta) * (zCeiling - zGround)/zCeiling + zGround
  with F a fixed cubic of the *computational* coordinate. F therefore depends only on k,
  so a particle's fractional k index comes from one 1-D interpolation against the flat
  column F_k after removing the terrain stretch -- exact, and no per-column search.

* **SGS quantities are recomputed exactly as the LES computes them**, not with textbook
  constants: `eps = c_e * e^(3/2) / l` with `l = min(0.76 sqrt(e)/N, Delta)` for N^2 > 0
  else `Delta`, and `Delta = (dx dy dz J)^(1/3)` (SRC/HYDRO_CORE/CUDA/cuda_sgstkeDevice.cu).
  A Langevin model driven by an inconsistent epsilon fails the well-mixed test for reasons
  that look like a bug in the integrator.

* Horizontal directions are periodic. Arrays are padded by one cell in i and j with a copy
  of the opposite edge so that linear interpolation is exact across the seam, and particle
  indices are wrapped into [0, n) before lookup.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import map_coordinates

KAPPA = 0.4
C_E = 0.93          # FastEddy default dissipation constant (l_corr_ce path reduces to this
                    # in neutral conditions, where l = Delta)
G = 9.81


def _step_of(path: str) -> int:
    return int(re.search(r"\.(\d+)$", path).group(1))


def dump_series(outdir: str, base: str = None) -> list[str]:
    """All dumps in outdir, sorted by absolute timestep."""
    pat = os.path.join(outdir, (base or "*") + ".[0-9]*")
    return sorted(glob.glob(pat), key=_step_of)


class FieldSet:
    """A window of FastEddy dumps held in RAM, with 4-D (t,z,y,x) linear interpolation."""

    def __init__(self, paths, dt_model, verbose=True, store_dtype=None,
                 cache_dtype=np.float32):
        # store_dtype emulates FastEddy writing the LPDM's four fields at reduced
        # precision (PLAN.md Stage 3: fp16 on write takes a 37.5-min window from 82 GB to
        # 16 GB). None = full fp32, as written today.
        self.store_dtype = store_dtype
        # RAM dtype of the field cache.
        #
        # float16 halves it and the QUANTISATION is harmless (bin/fp16_test.py: 75.7%
        # source-area overlap with the fp32 result, against a 59.2% overlap between two
        # halves of the same window). But scipy.ndimage.map_coordinates REFUSES float16 --
        # "RuntimeError: data type not supported" -- and that is the interpolator the entire
        # trajectory integration runs on. So float16 here needs the interpolator replaced
        # too; at 361 dumps the fp32 cache is 37 GB and fits, so it is not worth doing.
        # Kept because the storage argument still holds for the FILES on disk.
        self.cache_dtype = cache_dtype
        self.paths = list(paths)
        steps = np.array([_step_of(p) for p in self.paths], dtype=np.float64)
        self.t = steps * dt_model                     # seconds of model time
        self.dt_dump = float(np.diff(self.t).mean()) if len(self.t) > 1 else 0.0
        nt = len(self.paths)

        with Dataset(self.paths[0]) as ds:
            zpos = np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))
            xpos = np.squeeze(np.asarray(ds["xPos"][:], dtype=np.float64))
            ypos = np.squeeze(np.asarray(ds["yPos"][:], dtype=np.float64))
            topo = np.squeeze(np.asarray(ds["topoPos"][:], dtype=np.float64))
        nz, ny, nx = zpos.shape
        self.nz, self.ny, self.nx = nz, ny, nx
        self.dx = float(xpos[0, 0, 1] - xpos[0, 0, 0])
        self.dy = float(ypos[0, 1, 0] - ypos[0, 0, 0])
        self.x0 = float(xpos[0, 0, 0])
        self.y0 = float(ypos[0, 0, 0])
        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy
        self.zg = np.ascontiguousarray(topo if topo.ndim == 2 else np.zeros((ny, nx)))
        self.zpos = zpos
        # Terrain slope, for the surface-normal approach rate at touchdown. Periodic in
        # both horizontal directions, so the gradient wraps.
        self.zg_dx = np.gradient(np.pad(self.zg, 1, mode="wrap"), self.dx, axis=1)[1:-1, 1:-1]
        self.zg_dy = np.gradient(np.pad(self.zg, 1, mode="wrap"), self.dy, axis=0)[1:-1, 1:-1]

        # --- invert the terrain-following map: recover the flat-column shape F_k ---
        # z = F_k*(zC - zg)/zC + zg  =>  F_k = (z - zg) * zC/(zC - zg).  Taken from a
        # column, then cross-checked against every column so a wrong assumption is loud.
        zc_guess = zpos[-1].max()
        col = zpos[:, 0, 0]
        zg00 = self.zg[0, 0]
        self.zC = float((col[-1] - zg00) / (1.0 - zg00 / zc_guess)) if zg00 else float(col[-1])
        self.Fk = (col - zg00) * self.zC / max(self.zC - zg00, 1e-12)
        recon = self.Fk[:, None, None] * (self.zC - self.zg) / self.zC + self.zg
        err = np.abs(recon - zpos).max()
        if err > 1e-3:
            raise ValueError(f"terrain-following inversion off by {err:.4g} m -- "
                             "the vertical map is not the assumed zDeform form")
        self.zk = self.Fk.copy()          # flat-terrain level heights

        shape = (nt, nz, ny + 1, nx + 1)
        alloc = lambda: np.empty(shape, dtype=self.cache_dtype)
        self.u, self.v, self.w = alloc(), alloc(), alloc()
        self.e, self.eps, self.dsig2dz = alloc(), alloc(), alloc()
        self.ustar = np.empty((nt, ny + 1, nx + 1), dtype=np.float32)
        self.z0m = np.empty((nt, ny + 1, nx + 1), dtype=np.float32)
        self.invL = np.empty((nt, ny + 1, nx + 1), dtype=np.float32)

        # dz between cell centres of the flat column, used for Delta and d/dz
        dzc = np.gradient(self.zk)
        delta = (self.dx * self.dy * dzc[:, None, None]) ** (1.0 / 3.0)

        for n, p in enumerate(self.paths):
            with Dataset(p) as ds:
                g = lambda v: np.squeeze(np.asarray(ds[v][:], dtype=np.float32))
                uu, vv, ww = g("u"), g("v"), g("w")
                ee = np.maximum(g("TKE_0"), 0.0)
                th = g("theta").astype(np.float64)
                us, z0, iL = g("fricVel"), g("z0m"), g("invOblen")
            if self.store_dtype is not None:
                # Emulate a reduced-precision WRITE. This must happen HERE, before eps and
                # dsig2dz are derived, because FastEddy would store TKE_0 at the reduced
                # precision and the LPDM would derive them from the stored value -- not
                # from a full-precision one. Quantising afterwards would flatter the test.
                q = lambda arr: arr.astype(self.store_dtype).astype(arr.dtype)
                uu, vv, ww, ee = q(uu), q(vv), q(ww), q(ee)
                th = q(th)
            # SGS length scale and dissipation, exactly as cuda_sgstkeDevice.cu does it
            dthdz = np.gradient(th, self.zk, axis=0)
            n2 = (G / th) * dthdz
            ed = ee.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                len1 = 0.76 * np.sqrt(np.maximum(ed, 0.0)) / np.sqrt(np.maximum(n2, 1e-12))
            ell = np.where(n2 > 0.0, np.maximum(np.minimum(len1, delta), 1e-2), delta)
            eps = C_E * ed ** 1.5 / np.maximum(ell, 1e-2)
            if self.cache_dtype == np.float16:
                # fp16 subnormals bottom out near 6e-8; the model floors eps at 1e-8
                # anyway, so clip well above the representable limit rather than
                # letting it flush silently to zero.
                eps = np.maximum(eps, 1e-6)
            sig2 = (2.0 / 3.0) * ed
            ds2 = np.gradient(sig2, self.zk, axis=0)
            for dst, src in ((self.u, uu), (self.v, vv), (self.w, ww), (self.e, ee),
                             (self.eps, eps), (self.dsig2dz, ds2)):
                dst[n] = self._pad(src)
            for dst, src in ((self.ustar, us), (self.z0m, z0), (self.invL, iL)):
                dst[n] = self._pad2(src)
            if verbose and (n % 25 == 0 or n == nt - 1):
                print(f"    loaded {n+1}/{nt}  {os.path.basename(p)}", flush=True)

        self.mem_gb = sum(a.nbytes for a in
                          (self.u, self.v, self.w, self.e, self.eps, self.dsig2dz)) / 1e9

    @staticmethod
    def _pad(a):
        """Wrap-pad by one cell in y and x so linear interpolation crosses the seam."""
        out = np.empty((a.shape[0], a.shape[1] + 1, a.shape[2] + 1), dtype=np.float32)
        out[:, :-1, :-1] = a
        out[:, -1, :-1] = a[:, 0, :]
        out[:, :-1, -1] = a[:, :, 0]
        out[:, -1, -1] = a[:, 0, 0]
        return out

    @staticmethod
    def _pad2(a):
        out = np.empty((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float32)
        out[:-1, :-1] = a
        out[-1, :-1] = a[0, :]
        out[:-1, -1] = a[:, 0]
        out[-1, -1] = a[0, 0]
        return out

    # ------------------------------------------------------------------ indexing
    def hindex(self, x, y):
        """Fractional, wrapped i and j indices."""
        fi = np.mod((x - self.x0) / self.dx, self.nx)
        fj = np.mod((y - self.y0) / self.dy, self.ny)
        return fi, fj

    def kindex(self, z, fi, fj):
        """Fractional k index from physical height, undoing the terrain stretch."""
        zg = self.ground(fi, fj)
        F = (z - zg) * self.zC / np.maximum(self.zC - zg, 1e-12)
        return np.interp(F, self.Fk, np.arange(self.nz, dtype=np.float64))

    def height(self, fk, fi, fj):
        """Physical height from fractional k (inverse of kindex)."""
        zg = self.ground(fi, fj)
        F = np.interp(fk, np.arange(self.nz, dtype=np.float64), self.Fk)
        return F * (self.zC - zg) / self.zC + zg

    def ground(self, fi, fj):
        if not self.zg.any():
            return np.zeros_like(np.asarray(fi, dtype=np.float64))
        return map_coordinates(self.zg, [np.mod(fj, self.ny), np.mod(fi, self.nx)],
                               order=1, mode="grid-wrap")

    def ground_slope(self, fi, fj):
        """(d zg/dx, d zg/dy) at the particle, bilinearly interpolated."""
        if not self.zg.any():
            z = np.zeros_like(np.asarray(fi, dtype=np.float64))
            return z, z.copy()
        c = [np.mod(fj, self.ny), np.mod(fi, self.nx)]
        return (map_coordinates(self.zg_dx, c, order=1, mode="grid-wrap"),
                map_coordinates(self.zg_dy, c, order=1, mode="grid-wrap"))

    def tindex(self, t):
        return np.clip((t - self.t[0]) / max(self.dt_dump, 1e-12), 0.0, len(self.t) - 1.0)

    def sample(self, names, ft, fk, fj, fi):
        """Linear 4-D interpolation of the named fields at (t,k,j,i) fractional indices."""
        coords = np.vstack([ft, fk, fj, fi])
        return [map_coordinates(getattr(self, n), coords, order=1, mode="nearest")
                for n in names]

    def sample2d(self, names, ft, fj, fi):
        coords = np.vstack([ft, fj, fi])
        return [map_coordinates(getattr(self, n), coords, order=1, mode="nearest")
                for n in names]
