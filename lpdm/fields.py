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

from .dumpsrc import MemDump, open_dump, step_of

KAPPA = 0.4
C_E = 0.93          # FastEddy default dissipation constant (l_corr_ce path reduces to this
                    # in neutral conditions, where l = Delta)
G = 9.81


def _step_of(path) -> int:
    """Absolute timestep of a dump HANDLE -- a netCDF path or an in-RAM MemDump.

    Delegated to lpdm/dumpsrc.py so the path and ring sources cannot drift apart on how a
    time axis is built; that axis sets every release time and every interpolation weight.
    """
    return step_of(path)


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

        # GEOMETRY IS NOT NECESSARILY IN paths[0]. Under ioLPDMmode the static geometry
        # (xPos/yPos/zPos/topoPos) is written to the FIRST FILE OF THE RUN and to any
        # ioLPDMfullFrq multiple (io_netcdf.c: lpdmSkipWrite returns 0 for a full file
        # before it ever tests lpdmFileCount). A target case now runs the adjustment and
        # the window as ONE invocation and DELETES the adjustment dumps, so the run's
        # first file is routinely gone by the time this reads. Search the series rather
        # than assuming; bin/domain_adequacy.py already had to learn this.
        geom = None
        for _p in self.paths:
            with open_dump(_p) as _ds:
                if "zPos" in _ds.variables and "xPos" in _ds.variables:
                    geom = _p
                    break
        if geom is None:
            raise KeyError(
                "no dump in this series carries xPos/zPos. Under ioLPDMmode only the "
                "first file of a run and the ioLPDMfullFrq multiples do -- see "
                "bin/run_window.sh, which sets ioLPDMfullFrq so the first SURVIVING "
                "dump is full-form when the adjustment period is discarded.")
        with open_dump(geom) as ds:
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
        # Per-cell displacement height. FastEddy has no d -- surflayer_z0 and surflayer_z0t
        # are the only surface-layer length scales it carries -- so d never appears in a
        # dump and has to be supplied from the surface directory that built the run.
        # None means zero everywhere, which is what every flat/uniform case wants.
        self.dmap = None

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
            with open_dump(p) as ds:
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

    def set_displacement(self, d):
        """Attach a per-cell displacement height (ny, nx), in metres.

        MUST BE MEASURED FROM THE MODEL'S GROUND, not from bare earth. Where the surface
        build raised topoPos by the displacement height (--raise-topo, over the array), the
        model ground already IS the effective surface, so d there is zero and passing the
        canopy value would count it twice; everywhere else d is unchanged.
        bin/prep_surface.py writes dmap.npy under exactly that rule -- it subtracts
        whatever it put into the terrain -- so the pairing cannot be got wrong by hand.
        """
        if d is None:
            self.dmap = None
            return
        d = np.ascontiguousarray(np.asarray(d, dtype=np.float64))
        if d.shape != (self.ny, self.nx):
            raise ValueError(f"displacement map is {d.shape}, grid is {(self.ny, self.nx)}")
        self.dmap = d

    def displacement(self, fi, fj):
        """d at the particle, bilinear and periodic. Zero unless a map is attached."""
        if self.dmap is None:
            return np.zeros_like(np.asarray(fi, dtype=np.float64))
        return map_coordinates(self.dmap, [np.mod(fj, self.ny), np.mod(fi, self.nx)],
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

    @staticmethod
    def _corner(c, n):
        """Clamped base index and weight for one axis (matches mode='nearest')."""
        c = np.clip(c, 0.0, n - 1.0)
        i0 = np.clip(np.floor(c).astype(np.intp), 0, max(n - 2, 0))
        return i0, (c - i0)

    def _weights4(self, ft, fk, fj, fi):
        """Flat corner indices (16, N) and their weights, shared by every 4-D field.

        Built once per call and reused across fields, so each field costs ONE gather
        instead of sixteen. That matters: the integrator calls this at every step for six
        fields, and the naive form spends most of its time re-deriving the same indices.
        """
        nt_, nz_, ny_, nx_ = self.u.shape
        t0, wt = self._corner(ft, nt_)
        k0, wk = self._corner(fk, nz_)
        j0, wj = self._corner(fj, ny_)
        i0, wi = self._corner(fi, nx_)
        t1 = np.minimum(t0 + 1, nt_ - 1); k1 = np.minimum(k0 + 1, nz_ - 1)
        j1 = np.minimum(j0 + 1, ny_ - 1); i1 = np.minimum(i0 + 1, nx_ - 1)
        idx = np.empty((16, len(t0)), dtype=np.intp)
        w = np.empty((16, len(t0)), dtype=np.float64)
        n = 0
        for tt, ct in ((t0, 1.0 - wt), (t1, wt)):
            for kk, ck in ((k0, 1.0 - wk), (k1, wk)):
                for jj, cj in ((j0, 1.0 - wj), (j1, wj)):
                    for ii, ci in ((i0, 1.0 - wi), (i1, wi)):
                        idx[n] = ((tt * nz_ + kk) * ny_ + jj) * nx_ + ii
                        w[n] = ct * ck * cj * ci
                        n += 1
        return idx, w

    def sample(self, names, ft, fk, fj, fi):
        """Linear 4-D interpolation of the named fields at (t,k,j,i) fractional indices.

        Written out by hand rather than handed to scipy.ndimage.map_coordinates, for one
        reason: map_coordinates refuses a float16 array ("data type not supported"), and
        the cache HAS to be float16 for a long window to fit in memory. A 30-minute
        averaging period needs a (30 min + t_back) window -- 541 dumps at 5 s -- which is
        55 GB at fp32 against 28 GB at fp16 on a 62 GB machine. Gathering the 16 corners
        and promoting them to float64 costs nothing and removes the dtype constraint.

        Identical to order=1, mode='nearest' otherwise: coordinates are clamped to the
        array, not wrapped (x and y are already wrap-padded by one cell at load time).
        """
        idx, w = self._weights4(ft, fk, fj, fi)
        return [(getattr(self, nm).reshape(-1)[idx].astype(np.float64) * w).sum(axis=0)
                for nm in names]

    def sample2d(self, names, ft, fj, fi):
        nt_, ny_, nx_ = self.ustar.shape
        t0, wt = self._corner(ft, nt_)
        j0, wj = self._corner(fj, ny_)
        i0, wi = self._corner(fi, nx_)
        t1 = np.minimum(t0 + 1, nt_ - 1); j1 = np.minimum(j0 + 1, ny_ - 1)
        i1 = np.minimum(i0 + 1, nx_ - 1)
        idx = np.empty((8, len(t0)), dtype=np.intp)
        w = np.empty((8, len(t0)), dtype=np.float64)
        n = 0
        for tt, ct in ((t0, 1.0 - wt), (t1, wt)):
            for jj, cj in ((j0, 1.0 - wj), (j1, wj)):
                for ii, ci in ((i0, 1.0 - wi), (i1, wi)):
                    idx[n] = (tt * ny_ + jj) * nx_ + ii
                    w[n] = ct * cj * ci
                    n += 1
        return [(getattr(self, nm).reshape(-1)[idx].astype(np.float64) * w).sum(axis=0)
                for nm in names]
