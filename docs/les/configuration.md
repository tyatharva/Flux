# Configuration: 122³ at 30 m, receptor at 30 m

Every corpus case runs on one grid with one template. This page is the record of that
configuration and of the measurements that fixed it.

## The grid

| | |
|---|---|
| grid | 122 × 122 × 122 at `dx = dy = 30 m`, domain 3660 m. `(N + 6) = 128 = 2⁷` in all three directions. |
| vertical | `d_zeta` 24.691358, `verticalDeformFactor` 0.346601, `zCeiling` 3000 m, `dz_sfc` 8.5583 m, **level `k = 3` at exactly 30.000000000 m** |
| receptor | 30 m above bare ground. The array surface is raised 1.5 m, so the aerodynamic height is 28.5 m, and that is what every record carries. |
| `dt` | **0.0308642 s = 5/162 s**, `CFL_3d` 1.3502, 10.0% below the measured accuracy boundary |
| thread block | 1 × 2 × 64 (from a sweep of nine legal shapes, `bin/threadblock_sweep.py`) |
| cost | 0.479 GPU-h per simulated hour measured; a case is about 0.36 GPU-h at 8-way concurrency |
| `Δ`, `z/Δ` | 19.78 m, 1.52 |
| geometric-mean `z0` | 0.0615 m; water 13.61%, array 0.30% (44 cells) |
| taper knee | pad 12: real geography to 1470 m |

`bin/vgrid.py --dx 30 --nx 122 --receptor 30 --k 3 --zceiling 3000` re-derives the grid.
`runs/g30_base/base.in` is the template every case is built from, and `data/grid30_raised/`
is the production surface. Both are asserted present when the deployment image is built,
because losing the template once produced 81 cases and 0 records.

## Why 30 m and not 10 m

The real instrument is at about 10 m. The model receptor is at 30 m as a resolution
decision. At a 10 m receptor the footprint peak did not respond to meteorology at all: 48 m
in all three validation targets, max/min 1.00×, because the near field there was closure
output, not LES output. Say so wherever the emulator is described.

## A case is 1.25 simulated hours; the footprint is its last 30 minutes

145,800 steps = 4500.000 s, verified arithmetically.

| clock | event | step |
|---|---|---|
| T − 1.25 h | restart from the seed; adjustment begins | 0 |
| T − 0.75 h | adjustment complete (`ADJ_S` 1800 s) | 58,320 |
| T − 0.50 h | first release (needs `t_back` = 900 s) | 87,480 |
| T | last release; window closes | 145,800 |

The earliest field a backward trajectory may reach is the adjustment end. This is enforced
twice: `bin/run_window.sh` deletes the adjustment's dumps and refuses unless the earliest
survivor is step `A_NT`, and `bin/stage5_footprint.py --t-min` refuses independently.

`N_WINDOWS = 1`. A second window was measured and cut: on both validation cases the two
windows were near-duplicates in shape (median `|w0 − w1|` over the within-footprint floor
0.19 and 0.33, where independent draws give about √2). `N_WINDOWS = 2` stays supported for a
spread-estimating model.

## `dt` is set by the acoustic CFL, and the accuracy limit is below the stability limit

FastEddy is fully compressible with RK3, with no acoustic sub-stepping and no CFL machinery
at all. `dt` is a mandatory user constant, never computed or checked. Tutorial values are
hand-picked and mutually inconsistent; never copy them.

$$
\mathrm{CFL}_{3d} = c\,\Delta t\,\sqrt{\frac{2}{\Delta x^2} + \frac{1}{\Delta z_{sfc}^2}},
\qquad c = 347.2\ \mathrm{m\,s^{-1}}
$$

| | `CFL_3d` | behaviour |
|---|---|---|
| stability limit | about 1.79 | NaN, `CORRUPTED` |
| **accuracy limit** | **grid dependent** | above it: *silent* grid-scale acoustic noise |
| production, 122³ at 30 m | **1.3502** | 10% margin |

The accuracy boundary is a property of the grid. It must be re-measured on every grid, and
it does not interpolate with anisotropy: 122³ at 16 m (`dx/dz` 4.007) gave about 1.51; at
24 m (2.804) 1.55–1.60; **at 30 m (3.505) 1.50–1.55**. This grid's anisotropy sits between
the other two and its boundary at the bottom of their range. The transition is sharp:
`k0/k1` is 0.130 at CFL 1.50 and 8.857 at 1.55, a factor of 68 across 0.05 of CFL, and
`turb_alive` reads OK at every rung, so `k0/k1` is the only check that sees it. The runs
that measured it are `runs/s30_cfl160`, `s30_cfl170`, `s30_cfl180`, `s30_gap`, and the
record is `results/g30_bringup.txt`.

Verify every run with `docker/diag_near_surface.py`: the first-level `w` variance ratio
`k0/k1` must be below 1 (about 0.27 when correct). Near 9 means `dt` is too large.
`docker/k0k1_check.py` is a domain mean and is structurally blind to terrain-driven local
noise; `bin/k0k1_by_slope.py` conditions on slope and is the terrain-aware form.

Terrain amplifies the effective CFL as `CFL_3d · sqrt(1 + (slope · dx/dz)²)`, but measured at
122³ at 16 m it did not lower the boundary. Re-measure; never carry the number. Vertical
stretching is not a speed lever: with `dx` fixed, even an infinitely coarse vertical relaxes
the 3-D CFL by at most `sqrt(3/2)`.

## Restart overwrites grid and surface fields: the trap and the lever

`hydro_coreInit()` runs before the restart read, which walks the entire registered variable
list, including `xPos`, `yPos`, `zPos`, `topoPos`, `z0m`.

- **Trap.** Restarting a flat spin-up with a `topoFile` set leaves correct terrain-following
  metrics but silently overwrites the *diagnostic* `zPos` and `topoPos` with flat values. The
  LES is right and the output coordinates are wrong, so the LPDM places every particle at the
  wrong height with nothing to indicate it.
- **Lever.** The same mechanism is the only way to give FastEddy v5.0.1 spatially varying
  roughness or heat flux. `z0m` is a 2-D field with no input path; writing it into the
  restart file works with no source change. `bin/prep_restart.py` does this for every case:
  it rotates the seed's flow and injects the static surface (terrain, `z0`, `htFlux`).

## FastEddy capabilities the configuration relies on

Every capability claim carries a file and line in the FastEddy source; keep it that way.

- **Geostrophic forcing takes a linear vertical gradient** (`z_Ug`, `z_Vg`, `Ug_grad`,
  `Vg_grad`), all mandatory even when zero.
- **`stabilityScheme = 2` gives a 4-segment piecewise-linear base-state theta profile**
  (`zStableBottom{,2,3}`, `stableGradient{,2,3}`). `bin/sounding_to_forcing.py` fits a
  per-case HRRR profile to those six numbers, 0.04–0.27 K rms over the LES column.
- **Subsidence needs `lsfSelector = 1` and `lsf_horMnSubTerms = 1`.** Inputs are per hour.
  It is an advection tendency on U, V and theta against the slab-mean gradient; there is no
  `W` term, so `w` never acquires it. With `moistureSelector = 0` upstream v5.0.1 dies with an
  illegal memory access here, because the qv slab mean and `Frhs_qv` are unconditional while
  the arrays are allocated only when moisture is on. Patch 0004 fixes it; see
  [FastEddy and the patches](fasteddy-and-patches.md).
- **FastEddy is fp32** (confirmed in source) and **not bitwise reproducible**: two runs
  differ about 1e-4 relative in velocity and about 7e-4 K in theta after 200 steps. Any "did
  my change matter?" test compares against that floor, not against zero.
- **Restart is a true bit-for-bit state resume.** It requires netCDF output;
  `ioOutputMode = 1` is not restartable.
