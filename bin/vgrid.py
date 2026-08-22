#!/usr/bin/env python3
"""Solve FastEddy's vertical grid: d_zeta and verticalDeformFactor, from the receptor up.

WHY THIS EXISTS. The vertical grid is over-determined by hand: three things you want
(a domain ceiling, a receptor sitting exactly on a cell centre, a first level clear of the
canopy) against two free parameters. Every previous pass solved it with arithmetic in a
comment, and the comment then had to be trusted. This solves it from FastEddy's own
formula and prints the level table, so a grid change is one command and the receptor
height is an assertion rather than a claim.

THE FORMULA, read out of SRC/GRID/grid.c:1114-1127 (zDeform) and :470-486 (where it is
applied), not out of documentation:

    zeta_k = (k + 1/2) * d_zeta          k = 0 .. Nz-1     the rectilinear coordinate
    zC     = (Nz - 1/2) * d_zeta                           ztop, grid.c:470
    c2     = fCoeff*(1 - c1)/zC                            fCoeff = verticalDeformQuadCoeff
    c3     = (1 - c2*zC - c1)/zC^2
    z      = (c3 zeta^3 + c2 zeta^2 + c1 zeta) * (zC - zg)/zC + zg

With verticalDeformQuadCoeff = 0 (the pure-cubic form this project uses) c2 vanishes and
z(zeta) = ((1-c1)/zC^2) zeta^3 + c1 zeta, which is LINEAR IN c1 -- so pinning a cell centre
to an exact height is a division, not a root-find. Verified against the retired 24 m grid:
d_zeta = 24.691358, c1 = 0.346601 puts k=3 at 30.0001 m, which is the number that grid was
built on.

usage: vgrid.py [--nz 122] [--zceiling 2500] [--receptor 10.0] [--k 2]
                [--dx 16.0] [--cfl 1.461] [--nx 122]
"""
import argparse
import sys

import numpy as np

C_SOUND = 347.2          # sqrt(gamma R T) at 300 K; the constant CFL_3d is defined with
NS_PER_CELL_STEP = 8.51  # measured, PROJECT_BRIEF.md "Cost and thread blocks"


def z_of_zeta(zeta, c1, zC, fcoeff=0.0):
    """FastEddy's zDeform, over flat ground (zg = 0). See the module docstring."""
    c2 = fcoeff * (1.0 - c1) / zC
    c3 = (1.0 - c2 * zC - c1) / zC ** 2
    return c3 * zeta ** 3 + c2 * zeta ** 2 + c1 * zeta


def solve_c1(zeta_target, z_target, zC):
    """c1 placing z(zeta_target) exactly at z_target. Pure-cubic only, where it is linear."""
    # z = (1-c1) zeta^3/zC^2 + c1 zeta  =>  z - zeta^3/zC^2 = c1 (zeta - zeta^3/zC^2)
    cube = zeta_target ** 3 / zC ** 2
    denom = zeta_target - cube
    if abs(denom) < 1e-12:
        raise SystemExit("degenerate: the target level sits at the ceiling")
    return (z_target - cube) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz", type=int, default=122)
    ap.add_argument("--zceiling", type=float, default=2500.0)
    ap.add_argument("--receptor", type=float, default=10.0, help="receptor height, m AGL")
    ap.add_argument("--k", type=int, default=2, help="cell index the receptor must land on")
    ap.add_argument("--nx", type=int, default=122)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--dx", type=float, default=16.0)
    ap.add_argument("--cfl", type=float, default=1.461,
                    help="target CFL_3d; production sits ~10%% below the ~1.64 accuracy limit")
    ap.add_argument("--cadence", type=float, default=5.0, help="output cadence, s")
    ap.add_argument("--levels", type=int, default=8, help="how many levels to print")
    a = ap.parse_args()
    ny = a.ny if a.ny is not None else a.nx

    d_zeta = a.zceiling / (a.nz - 0.5)
    zeta = (np.arange(a.nz) + 0.5) * d_zeta
    c1 = solve_c1(zeta[a.k], a.receptor, a.zceiling)
    if not 0.0 <= c1 <= 1.0:
        raise SystemExit(f"verticalDeformFactor = {c1:.6f} is outside FastEddy's [0,1] "
                         f"bound (grid.c:106) -- change Nz, zCeiling or k")
    z = z_of_zeta(zeta, c1, a.zceiling)
    err = abs(z[a.k] - a.receptor)
    dz = np.gradient(z)
    dz_sfc = 2.0 * z[0]        # the surface is at z=0, so the first cell is twice its centre

    print(f"=== vertical grid: Nz = {a.nz}, zCeiling = {a.zceiling:.1f} m ===")
    print(f"  d_zeta                  = {d_zeta:.6f}      "
          f"(zCeiling = d_zeta*(Nz-0.5) = {d_zeta*(a.nz-0.5):.4f} m)")
    print(f"  verticalDeformFactor    = {c1:.6f}")
    print(f"  verticalDeformQuadCoeff = 0.0        (pure cubic)")
    print(f"  dz_sfc                  = {dz_sfc:.4f} m")
    print(f"  RECEPTOR: k = {a.k} centre at {z[a.k]:.9f} m  (target {a.receptor:.6f}, "
          f"error {err:.2e} m)  {'OK' if err < 1e-6 else 'FAIL'}")
    print(f"  cells below the receptor: {a.k}   "
          f"({'LPDM interpolation needs >= 2' if a.k >= 2 else 'TOO FEW for the LPDM'})")

    print(f"\n  {'k':>4} {'z (m)':>12} {'dz (m)':>10}")
    for k in range(min(a.levels, a.nz)):
        mark = "  <-- receptor" if k == a.k else ""
        print(f"  {k:>4} {z[k]:12.6f} {dz[k]:10.4f}{mark}")
    for frac, lab in ((400.0, "400 m"), (1000.0, "1000 m")):
        k_ = int(np.searchsorted(z, frac))
        if k_ < a.nz:
            print(f"  {k_:>4} {z[k_]:12.6f} {dz[k_]:10.4f}   first level above {lab}"
                  f"  ({k_} levels below)")
    print(f"  {a.nz-1:>4} {z[-1]:12.6f} {dz[-1]:10.4f}   top")

    # ---- horizontal, cost, and the CFL that sets dt --------------------------------
    dz_r = float(dz[a.k])
    delta = (a.dx * a.dx * dz_r) ** (1.0 / 3.0)
    ncell = a.nx * ny * a.nz
    s_per_step = ncell * NS_PER_CELL_STEP * 1e-9
    # CFL_3d = c dt sqrt(1/dx^2 + 1/dy^2 + 1/dz^2), with dz the SURFACE spacing (the
    # tightest). Reproduces the 24 m grid's stated 1.4946 at dt = 0.0328947 to 4 digits.
    kfac = C_SOUND * np.sqrt(2.0 / a.dx ** 2 + 1.0 / dz_sfc ** 2)
    dt_raw = a.cfl / kfac
    # Round dt DOWN to a value whose cadence is an integer step count: run_window.sh
    # asserts frq*dt == cadence, and a non-integer count silently shifts every dump.
    nstep = int(np.ceil(a.cadence / dt_raw))
    dt = a.cadence / nstep
    print(f"\n=== horizontal and time step: {a.nx} x {ny} @ {a.dx:.1f} m "
          f"= {a.nx*a.dx:.0f} x {ny*a.dx:.0f} m ===")
    print(f"  (N+6) = {a.nx+6}, {ny+6}, {a.nz+6}   "
          f"1x2x64 legal: {all((n+6) % t == 0 for n, t in ((a.nx,1),(ny,2),(a.nz,64)))}")
    print(f"  cells = {ncell/1e6:.3f} M    s/step = {s_per_step:.5f} "
          f"(at {NS_PER_CELL_STEP} ns/cell/step)")
    print(f"  dx/dz_sfc = {a.dx/dz_sfc:.3f}   "
          f"(terrain amplifies CFL by sqrt(1+(slope*dx/dz)^2) -- re-bisect over terrain)")
    print(f"  Delta at the receptor = {delta:.2f} m   z/Delta = {a.receptor/delta:.2f}")
    print(f"  dt for CFL_3d = {a.cfl:.3f}:  {dt_raw:.7f} s  ->  {dt:.7f} s = "
          f"{a.cadence:.0f}/{nstep} (CFL_3d = {dt*kfac:.4f})")
    print(f"  cadence {a.cadence:.0f} s = {nstep} steps exactly")
    gpuh = 3600.0 / dt * s_per_step / 3600.0
    print(f"  cost = {gpuh:.3f} GPU-h per simulated hour")
    for cap_h, lab in ((1.0, "1 h wall cap"),):
        steps = cap_h * 3600.0 / s_per_step
        print(f"  {lab}: {steps:,.0f} steps = {steps*dt/3600.0:.2f} simulated hours "
              f"per segment")
    print(f"\n  doubly-periodic CBL support: L >= 4 z_i caps z_i at {a.nx*a.dx/4:.0f} m, "
          f"L >= 2 z_i at {a.nx*a.dx/2:.0f} m")
    return 0 if err < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
