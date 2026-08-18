#!/usr/bin/env python3
"""Stage 5: first footprint, flat and neutral -- and the irreducible error floor.

Gate 1: over flat uniform terrain in neutral conditions the LES+LPDM footprint must agree
        with Kljun et al. (2015). A homogeneous surface is where FFP is valid, so
        disagreement here is a pipeline bug and not a result.
Gate 2: the same configuration run twice gives two turbulence realisations (FastEddy is
        not bitwise reproducible). The difference between their footprints is the floor
        below which no emulator can be scored.

usage: stage5_footprint.py <window_dir> [<window_dir_2>] [--dt ...] [--tag ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm import kljun
from lpdm.driver import compute_footprint
from lpdm.fields import FieldSet, dump_series
from lpdm.footprint import source_area_overlap


def kljun_on(grid, st, zm):
    return kljun.footprint_2d(grid.xc, grid.yc, zm, st["h"], st["ustar"],
                              st["sigma_v"], umean=st["u_mean"], L=st["L"],
                              y_edges=grid.ye)


def describe(name, g, which="flux"):
    m = g.metrics(which)
    print(f"  {name:<26} peak_x={m['peak_x']:7.1f} m  centroid=({m['centroid_x']:7.1f},"
          f"{m['centroid_y']:6.1f}) m  80% area={m['area80_cells']*g.area/1e4:7.2f} ha"
          f"  x80=[{m['x80_near']:.0f},{m['x80_far']:.0f}] m")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--dt", type=float, default=0.0625)
    ap.add_argument("--tback", type=float, default=900.0)
    ap.add_argument("--nrel", type=int, default=700)
    ap.add_argument("--dtrel", type=float, default=4.0)
    ap.add_argument("--c0", type=float, default=3.0)
    ap.add_argument("--res", type=float, default=20.0)
    ap.add_argument("--tag", default="stage5")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    runs = []
    for d in a.dirs:
        paths = dump_series(d)
        print(f"\n=== {d}: {len(paths)} dumps ===")
        t0 = time.time()
        fs = FieldSet(paths, a.dt, verbose=False)
        print(f"  cache {fs.mem_gb:.2f} GB, window {fs.t[0]:.0f}-{fs.t[-1]:.0f} s "
              f"(cadence {fs.dt_dump:.1f} s), loaded in {time.time()-t0:.0f} s")
        r = compute_footprint(fs, paths, n_per_release=a.nrel, dt_release=a.dtrel,
                              t_back=a.tback, c0=a.c0, grid_res=a.res,
                              seed=len(runs))
        st = r["stats"]
        print(f"  LES scalars at z={st['z_recept']:.2f} m: U={st['u_mean']:.2f} m/s "
              f"dir={st['wdir']:.1f} deg  u*={st['ustar']:.3f}  sigma_v={st['sigma_v']:.3f}"
              f"  sigma_w={st['sigma_w']:.3f}  h={st['h']:.0f} m  1/L={1/st['L']:.2e}")
        print(f"  integral of f_flux over grid = {r['grid'].integral():.3f} "
              f"(all touchdowns {r['grid'].integral_all():.3f}; shortfall = influence "
              f"truncated by t_back={a.tback:.0f} s)")
        runs.append((d, fs, r))
        del fs.u, fs.v, fs.w, fs.e, fs.eps, fs.dsig2dz

    print("\n=== GATE 1: LES+LPDM vs Kljun et al. (2015) ===")
    d0, fs0, r0 = runs[0]
    g0, st = r0["grid"], r0["stats"]
    zm = st["z_recept"]
    kl = kljun_on(g0, st, zm)
    kl_int = kl.sum() * g0.area
    les = g0.normalised("flux")
    m_les = describe("LES+LPDM", g0)
    kg = type(g0)(g0.xe[0], g0.xe[-1], g0.ye[0], g0.ye[-1], g0.res)
    kg.flux = kl * kg.area * 1.0
    kg.n_particles = 1
    m_kl = describe("Kljun FFP", kg)
    print(f"  Kljun x_max (analytic)     {kljun.peak_distance(zm, st['h'], st['ustar'], umean=st['u_mean']):7.1f} m")
    print(f"  Kljun integral over grid   {kl_int:.3f}")
    ov = source_area_overlap(np.maximum(les, 0), np.maximum(kl, 0))
    dpeak = m_les["peak_x"] - m_kl["peak_x"]
    dcent = m_les["centroid_x"] - m_kl["centroid_x"]
    print(f"\n  peak-location difference   {dpeak:+7.1f} m "
          f"({100*dpeak/max(m_kl['peak_x'],1e-9):+.0f}%)")
    print(f"  centroid difference        {dcent:+7.1f} m "
          f"({100*dcent/max(m_kl['centroid_x'],1e-9):+.0f}%)")
    print(f"  80% source-area overlap    {ov*100:6.1f}%")

    out = dict(dirs=a.dirs, zm=zm, stats={k: (float(v) if np.isscalar(v) else None)
                                          for k, v in st.items()},
               les=m_les, kljun=m_kl, overlap_kljun=ov,
               dpeak=dpeak, dcentroid=dcent,
               integral_les=g0.integral(), integral_les_all=g0.integral_all(),
               integral_kljun=float(kl_int))

    print("\n=== GATE 2: irreducible error floor ===")
    if "halves" in r0:
        h1, h2 = r0["halves"]
        a1, a2 = h1.normalised("flux"), h2.normalised("flux")
        m1, m2 = describe("window half 1", h1), describe("window half 2", h2)
        ovh = source_area_overlap(np.maximum(a1, 0), np.maximum(a2, 0))
        print(f"  half-vs-half  80% overlap {ovh*100:6.1f}%   "
              f"peak difference {m1['peak_x']-m2['peak_x']:+.1f} m   "
              f"centroid difference {m1['centroid_x']-m2['centroid_x']:+.1f} m")
        out["halves"] = dict(overlap=ovh, dpeak=m1["peak_x"] - m2["peak_x"],
                             dcentroid=m1["centroid_x"] - m2["centroid_x"])
    if len(runs) > 1:
        g1 = runs[1][2]["grid"]
        b0, b1 = g0.normalised("flux"), g1.normalised("flux")
        mb = describe("second realisation", g1)
        ovr = source_area_overlap(np.maximum(b0, 0), np.maximum(b1, 0))
        num = np.abs(b0 - b1).sum()
        den = 0.5 * (np.abs(b0).sum() + np.abs(b1).sum())
        print(f"  realisation-vs-realisation  80% overlap {ovr*100:6.1f}%   "
              f"peak difference {m_les['peak_x']-mb['peak_x']:+.1f} m   "
              f"centroid difference {m_les['centroid_x']-mb['centroid_x']:+.1f} m")
        print(f"  normalised L1 difference {num/den*100:.1f}%")
        out["floor"] = dict(overlap=ovr, dpeak=m_les["peak_x"] - mb["peak_x"],
                            dcentroid=m_les["centroid_x"] - mb["centroid_x"],
                            l1_rel=float(num / den))

    plot(a, runs, kl, out)
    with open(os.path.join(a.outdir, f"{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  wrote {a.outdir}/{a.tag}.json and {a.outdir}/{a.tag}.png")
    return 0


def plot(a, runs, kl, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g0 = runs[0][2]["grid"]
    les = g0.normalised("flux")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ext = [g0.xe[0], g0.xe[-1], g0.ye[0], g0.ye[-1]]
    vmax = max(les.max(), kl.max())
    for k, (arr, name) in enumerate(((les, "LES + backward LPDM"), (kl, "Kljun et al. (2015)"))):
        im = ax[k].imshow(arr, origin="lower", extent=ext, aspect="equal",
                          vmin=0, vmax=vmax, cmap="magma")
        for lev, ls in ((0.5, "-"), (0.8, "--")):
            tot = arr.sum()
            flat = np.sort(arr.ravel())[::-1]
            cum = np.cumsum(flat) / tot
            thr = flat[np.searchsorted(cum, lev)]
            ax[k].contour(g0.xc, g0.yc, arr, levels=[thr], colors="c", linestyles=ls,
                          linewidths=1.0)
        ax[k].plot(0, 0, "w*", ms=12)
        ax[k].set_title(f"{name}\n(cyan: 50% solid, 80% dashed)")
        ax[k].set_xlabel("upwind distance (m)"); ax[k].set_ylabel("crosswind (m)")
        fig.colorbar(im, ax=ax[k], label="f  (m$^{-2}$)")
    ax[2].plot(g0.xc, g0.crosswind_integrated("flux"), lw=2, label="LES + LPDM")
    ax[2].plot(g0.xc, kl.sum(axis=0) * g0.res, lw=2, ls="--", label="Kljun 2015")
    if len(runs) > 1:
        ax[2].plot(g0.xc, runs[1][2]["grid"].crosswind_integrated("flux"), lw=1,
                   alpha=0.7, label="LES realisation 2")
    ax[2].set_xlabel("upwind distance (m)"); ax[2].set_ylabel("$f_y$  (m$^{-1}$)")
    ax[2].legend(); ax[2].grid(alpha=0.3)
    ax[2].set_title("crosswind-integrated footprint")
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, f"{a.tag}.png"), dpi=130)


if __name__ == "__main__":
    sys.exit(main())
