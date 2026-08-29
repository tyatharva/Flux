"""ctypes front-end to the GPU-resident LPDM (lib/liblpdm.so).

WHAT THIS IS FOR. The production path for the GPU LPDM is inside FastEddy: the ring buffer
is filled from the live device fields and no field ever reaches disk. That path cannot be
compared against the CPU LPDM, because there is nothing saved to compare on. So the same
translation unit is also built as a shared library and driven from here, fed with the
arrays `lpdm/fields.py` already holds -- which makes the acceptance comparison a comparison
of the INTEGRATOR and of nothing else. Both paths see bit-identical fp16 fields, the same
floor table, the same release points and the same t_back.

The one difference that cannot be removed is the random stream: curand's Philox on the GPU
against numpy's PCG64 on the host. Agreement is therefore statistical, within the sampling
floors this project already measures, never bitwise -- which is the same standard every
other comparison here is held to, because FastEddy itself is not bitwise reproducible.
"""
from __future__ import annotations

import ctypes as ct
import os

import numpy as np

_LIB = None
_SO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "lib", "liblpdm.so")


class LpdmGrid(ct.Structure):
    _fields_ = [("nx", ct.c_int), ("ny", ct.c_int), ("nz", ct.c_int), ("nt", ct.c_int),
                ("dx", ct.c_float), ("dy", ct.c_float),
                ("x0", ct.c_float), ("y0", ct.c_float),
                ("zC", ct.c_float), ("dt_dump", ct.c_float)]


class LpdmOpts(ct.Structure):
    _fields_ = [("c0", ct.c_double), ("z_touch", ct.c_double), ("dt_frac", ct.c_double),
                ("dt_min", ct.c_double), ("dt_max", ct.c_double),
                ("t_limit", ct.c_double), ("max_disp", ct.c_double),
                ("w_floor", ct.c_double), ("reflect", ct.c_int), ("direction", ct.c_int),
                ("max_iter", ct.c_int), ("td_prob", ct.c_double),
                ("yaw", ct.c_double), ("pitch", ct.c_double),
                ("seed", ct.c_ulonglong)]


def _lib():
    global _LIB
    if _LIB is None:
        if not os.path.exists(_SO):
            raise RuntimeError(f"{_SO} not built -- run docker/build_lpdm.sh")
        L = ct.CDLL(_SO)
        f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
        u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
        i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
        L.lpdmDeviceInit.argtypes = [ct.POINTER(LpdmGrid), ct.c_int, ct.c_longlong]
        L.lpdmDeviceSetStatic.argtypes = [f32p, f32p, f32p, f32p, ct.c_void_p]
        L.lpdmDeviceSetFloor.argtypes = [ct.c_int, f64p, f64p, ct.c_int]
        L.lpdmDeviceSetCover.argtypes = [ct.c_int, u8p]
        L.lpdmDevicePushPaddedHalf.argtypes = ([ct.c_int, ct.c_double]
                                               + [ct.c_void_p] * 6 + [f32p, f32p])
        L.lpdmDeviceDeriveCheck.argtypes = [f32p, f32p, f32p, f32p]
        L.lpdmDeviceRelease.argtypes = [ct.POINTER(LpdmOpts), ct.c_int,
                                        f64p, f64p, f64p, f64p,
                                        ct.c_double, ct.c_double,
                                        ct.c_void_p, ct.c_void_p, ct.c_void_p]
        L.lpdmDeviceFetch.argtypes = [f64p, f64p, f64p, f64p, i64p, i64p, f64p, f64p]
        L.lpdmDeviceFetchTouchdowns.argtypes = [ct.c_longlong, f32p, f32p, f32p, f32p,
                                                i64p, ct.POINTER(ct.c_int)]
        L.lpdmDeviceFetchTouchdowns.restype = ct.c_longlong
        _LIB = L
    return _LIB


class GpuLPDM:
    """Drive the device LPDM from a FieldSet, one window at a time."""

    def __init__(self, fs, td_capacity=4_000_000):
        L = _lib()
        self.fs = fs
        self.L = L
        nt, nz, nyp, nxp = fs.u.shape
        self.nt, self.nz = nt, nz
        g = LpdmGrid(nx=fs.nx, ny=fs.ny, nz=nz, nt=nt,
                     dx=float(fs.dx), dy=float(fs.dy),
                     x0=float(fs.x0), y0=float(fs.y0),
                     zC=float(fs.zC), dt_dump=float(fs.dt_dump))
        if L.lpdmDeviceInit(ct.byref(g), 0, int(td_capacity)) != 0:
            raise RuntimeError("lpdmDeviceInit failed")
        self.td_capacity = int(td_capacity)
        zg = np.ascontiguousarray(fs.zg, dtype=np.float32)
        sx = np.ascontiguousarray(getattr(fs, "zg_dx", np.zeros_like(fs.zg)), np.float32)
        sy = np.ascontiguousarray(getattr(fs, "zg_dy", np.zeros_like(fs.zg)), np.float32)
        dm = (np.ascontiguousarray(fs.dmap, np.float32)
              if getattr(fs, "dmap", None) is not None else None)
        zk = np.ascontiguousarray(fs.zk, dtype=np.float32)
        if L.lpdmDeviceSetStatic(zk, zg, sx, sy,
                                 dm.ctypes.data if dm is not None else None) != 0:
            raise RuntimeError("lpdmDeviceSetStatic failed")
        self._dm = dm                    # keep alive
        # THE RING IS FILLED FROM THE CACHE AS-IS. fs.u..fs.dsig2dz are already padded and
        # already fp16 when cache_dtype is float16; anything else is cast here, which is
        # what makes a float32 cache comparable too.
        names = ("u", "v", "w", "e", "eps", "dsig2dz")
        for s in range(nt):
            bufs = []
            for nm in names:
                a = np.ascontiguousarray(getattr(fs, nm)[s], dtype=np.float16)
                bufs.append(a)
            us = np.ascontiguousarray(fs.ustar[s], dtype=np.float32)
            z0 = np.ascontiguousarray(fs.z0m[s], dtype=np.float32)
            rc = L.lpdmDevicePushPaddedHalf(s, float(fs.t[s]),
                                            *[b.ctypes.data for b in bufs], us, z0)
            if rc != 0:
                raise RuntimeError(f"push of snapshot {s} failed")
        self._cover_names = []

    def set_floor(self, z, f, mode=1):
        """mode 1 = multiplicative sc(z) (production), 2 = additive, 0 = off."""
        z = np.ascontiguousarray(z, dtype=np.float64)
        f = np.ascontiguousarray(f, dtype=np.float64)
        if self.L.lpdmDeviceSetFloor(len(z), z, f, int(mode)) != 0:
            raise RuntimeError("lpdmDeviceSetFloor failed")

    def set_cover(self, cover):
        """cover: dict name -> (ny,nx) boolean mask, in LES index space."""
        self._cover_names = list(cover)
        if not self._cover_names:
            return
        m = np.ascontiguousarray(
            np.stack([np.asarray(cover[k], dtype=bool) for k in self._cover_names]),
            dtype=np.uint8)
        self._cover_buf = m
        if self.L.lpdmDeviceSetCover(len(self._cover_names), m) != 0:
            raise RuntimeError("lpdmDeviceSetCover failed")

    def reset(self):
        self.L.lpdmDeviceReset()

    def release(self, x, y, z, t_rel, *, c0=3.0, z_touch=2.0, dt_frac=0.05,
                dt_min=0.01, dt_max=1.0, t_limit=600.0, max_disp=0.0, w_floor=0.02,
                reflect=True, direction=-1, max_iter=200_000, td_prob=1.0,
                yaw=0.0, pitch=0.0, seed=0, want_final=False,
                x_ref=None, y_ref=None):
        t_rel = np.ascontiguousarray(t_rel, dtype=np.float64)
        n = len(t_rel)
        bc = lambda v: np.ascontiguousarray(
            np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, float))
        xa, ya, za = bc(x), bc(y), bc(z)
        x_ref = float(xa[0] if x_ref is None else x_ref)
        y_ref = float(ya[0] if y_ref is None else y_ref)
        fx = np.zeros(n, np.float64) if want_final else None
        fy = np.zeros(n, np.float64) if want_final else None
        fz = np.zeros(n, np.float64) if want_final else None
        o = LpdmOpts(c0=c0, z_touch=z_touch, dt_frac=dt_frac, dt_min=dt_min,
                     dt_max=dt_max, t_limit=t_limit, max_disp=max_disp, w_floor=w_floor,
                     reflect=int(bool(reflect)), direction=int(direction),
                     max_iter=int(max_iter), td_prob=float(td_prob),
                     yaw=float(yaw), pitch=float(pitch),
                     seed=ct.c_ulonglong(int(seed)))
        if self.L.lpdmDeviceRelease(
                ct.byref(o), n, xa, ya, za, t_rel, x_ref, y_ref,
                fx.ctypes.data if want_final else None,
                fy.ctypes.data if want_final else None,
                fz.ctypes.data if want_final else None) != 0:
            raise RuntimeError("lpdmDeviceRelease failed")
        if want_final:
            return dict(x=fx, y=fy, z=fz)

    def fetch(self):
        ny, nx = self.fs.ny, self.fs.nx
        flux = np.zeros((ny, nx), np.float64)
        conc = np.zeros((ny, nx), np.float64)
        sf = np.zeros(1, np.float64); sc = np.zeros(1, np.float64)
        npart = np.zeros(1, np.int64); ntd = np.zeros(1, np.int64)
        nc = max(len(self._cover_names), 1)
        cnum = np.zeros(nc, np.float64); cden = np.zeros(1, np.float64)
        if self.L.lpdmDeviceFetch(flux.ravel(), conc.ravel(), sf, sc, npart, ntd,
                                  cnum, cden) != 0:
            raise RuntimeError("lpdmDeviceFetch failed")
        share = {}
        if self._cover_names and cden[0]:
            share = {k: float(cnum[i] / cden[0])
                     for i, k in enumerate(self._cover_names)}
        return dict(flux=flux, conc=conc, sum_flux_all=float(sf[0]),
                    sum_conc_all=float(sc[0]), n_particles=int(npart[0]),
                    n_touchdown=int(ntd[0]), cover_share=share)

    def fetch_touchdowns(self, cap=None):
        cap = int(cap or self.td_capacity)
        dx = np.zeros(cap, np.float32); dy = np.zeros(cap, np.float32)
        wt = np.zeros(cap, np.float32); ag = np.zeros(cap, np.float32)
        tot = np.zeros(1, np.int64); ovf = ct.c_int(0)
        n = self.L.lpdmDeviceFetchTouchdowns(cap, dx, dy, wt, ag, tot, ct.byref(ovf))
        return dict(dx=dx[:n], dy=dy[:n], wt=wt[:n], age=ag[:n],
                    n_kept=int(n), n_drawn=int(tot[0]), overflowed=bool(ovf.value))

    def derive_check(self, e, theta):
        """Run kDerive on one snapshot and return (eps, dsig2dz) as interior fp32."""
        e = np.ascontiguousarray(e, dtype=np.float32)
        th = np.ascontiguousarray(theta, dtype=np.float32)
        eps = np.zeros_like(e); ds2 = np.zeros_like(e)
        if self.L.lpdmDeviceDeriveCheck(e.ravel(), th.ravel(), eps.ravel(),
                                        ds2.ravel()) != 0:
            raise RuntimeError("lpdmDeviceDeriveCheck failed")
        return eps, ds2

    def close(self):
        self.L.lpdmDeviceFree()
