# Fourth pass — true 30-minute footprints, a static raster, and the convective regime

Supersedes `THIRD_PASS_RESULTS.md` for everything about the estimator, the window length
and the corpus design. The grid, the surface and the sub-grid closure are unchanged from
the third pass and are not re-litigated here.

Four things the third pass left open are closed, and one new regime is added.

---

## 1. A window is (30 min + `t_back`), and the fork had to change to allow it

**The third pass's footprints were 15-minute footprints wearing a 30-minute label.** A
backward trajectory cannot be released until it has `t_back` seconds of stored field
behind it, so the first `t_back` of any window produces nothing. With `t_back = 900 s` a
30-minute window yields 15 minutes of releases — and 15 minutes is below what the
ensemble-convergence curve says the centroid needs, which is why the terrain cases showed
27-43% half-vs-half overlap against 53.6% flat.

The averaging period stays 30 minutes. That is what eddy covariance means and it is not
ours to move; the **LES window** is what has to grow.

**That ran straight into the 45-minute per-run ceiling.** At `186 x 186 x 122` the terrain
`dt` gives 1.28 s of wall clock per simulated second, so a 2700 s window is 57.5 minutes —
and it could not be split, because lean `ioLPDMmode` output is deliberately not restartable
(`rho` and `pressure` are absent by construction).

`ioLPDMfullFrq` on the `kegonsa` fork fixes exactly that: any output whose **absolute**
timestep is a multiple of it is written in full upstream form — every registered variable,
fp32, coordinate geometry included — while every other dump stays lean and 16-bit packed.
Setting it to the segment length puts a restartable dump at each end of a chain link.

Verified before use, on 400 steps:

| | result |
|---|---|
| variable set and dtypes vs a mode-0 dump | identical |
| values vs an independent mode-0 run | `u` 3.4e-5, `w` 3.4e-4 relative — the nondeterminism floor |
| second segment restarts from it | yes, clean, no missing-variable errors |
| cost | one 170 MB dump per chain link against 42.7 MB lean, i.e. ~0.5% of a window |

`bin/run_window.sh` is the driver: it takes a window length, works out the segmentation
that keeps every segment under the ceiling, checks the projection before launching, and
preserves each boundary dump outside `window/` so a failed segment does not take the chain
point with it.

---

## 2. The footprint raster is now the LES grid

The third pass accumulated on a 60 m wind-aligned raster and then interpolated onto the map
for figures. Both halves of that were wrong for what comes next.

- The resample fell hardest on the **near field**, which is where the footprint peak sits
  and where the solar array is. It blurred exactly the part the result depends on.
- The emulator will consume a fixed north-up raster. Training it on a rotated-then-
  resampled array means training it on the resample.

Touchdowns are now binned by their **LES column index**, folded modulo the periodic domain,
so a footprint cell IS an LES column — the same indexing the land-cover masks use. Nothing
is rotated and nothing is interpolated.

The wind frame has not been abandoned; it has been made exact. The crosswind-integrated
footprint is a 1-D histogram of the touchdowns' own upwind coordinate at 24 m, and Kljun is
evaluated at the static cells' own coordinates (`lpdm.kljun.footprint_on_static`, 8x8
sub-sampled per cell because near the receptor `sigma_y` is smaller than one cell) rather
than rotated onto them. FFP is a closed-form function; interpolating it was gratuitous.

**A side effect worth stating: the integral now includes the wrapped touchdowns.** They
fold onto the same surface they came from, so they belong in it — where the old
4500 x 3000 m wind-frame grid simply dropped everything beyond its edge.

One implementation consequence. A 541-dump window is 55 GB of field cache at fp32 and
28 GB at fp16, on a 62 GB machine, so the cache had to become float16 — and
`scipy.ndimage.map_coordinates` refuses float16. The 4-D linear interpolation is therefore
written out by hand in `lpdm/fields.py`, gathering the 16 corners once and reusing the
index set across all six fields. It matches `map_coordinates` to float32 roundoff and is
marginally faster per field.

---

## 3. CONUS404 as a climatology, never as forcing

`bin/conus404_site.py` streams a stratified 45-year hourly sample at the tower cell
directly off the USGS Open Storage Network pod — anonymous S3 over plain HTTPS, no cloud
SDK, no credentials, no egress charge. 274 of 2740 time-chunks (6 contiguous days out of
every 60, so every month of every year is represented and the diurnal cycle is complete
within each block), 39,456 hourly records, ~30 GB streamed and discarded, 8 minutes.

**Nothing here forces a run.** No per-case sounding, no projection matching, no
time-varying boundary conditions. Each LES case stays one idealised quasi-stationary state.
CONUS404 only decides which states are worth GPU time, and how many of each.

Quality-controlled at `u* >= 0.15 m/s` (65.2% of hours):

| | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| `z_i` | 80 m | 267 m | 493 m | 835 m | 1475 m |
| `w'theta'` | -0.027 | -0.006 | +0.015 | +0.076 | +0.164 K m/s |
| `u*` | 0.17 | 0.24 | 0.32 | 0.44 | 0.65 m/s |
| `U(30 m)` | 2.4 | 3.9 | 5.2 | 6.8 | 10.0 m/s |

**Three things follow directly.**

1. **The site is unstable more than half the time** — 27.2% very unstable (`z/L < -0.5`),
   30.3% unstable, 13.3% near-neutral, 20.4% stable, 8.8% very stable. A neutral-only
   corpus misses the modal daytime state entirely, which is why this pass adds one.
2. **`z_i` must be swept, over a factor of 18.** Kljun takes `z_i` as an input; a corpus at
   one `z_i` leaves that input channel untrained and the emulator cannot learn what it does.
3. **The wind rose and the array signal point in different directions.** The rose is
   S 16.0%, NW 14.5%, W 14.4%, SW 14.3% against N 10.6%, E 10.4%, NE 10.2%, SE 9.8%. But
   the array is upwind only on northerlies, so N/NE/NW is where the site-specific skill
   has to come from. Direction sampling needs a **floor**, not pure rose weighting.

Derivations are all standard surface-layer relations and are written out in the script:
`ACSHFLSM` (kJ m^-2 accumulated over the prior hour) to `H` to `w'theta'` through
`rho = PSFC/(R_d T)`; `u*` and `L` solved jointly from `U10` with `z0 = 0.05 m`. The
grid-relative to earth-relative wind rotation (`COSALPHA`/`SINALPHA`, -5.55 deg here) is a
whole direction bin and is applied.

---
