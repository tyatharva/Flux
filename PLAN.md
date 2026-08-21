# Staged Plan — Fifth Pass, 10 m receptor

> **STATUS: planned, not run.** Written 2026-08-21, after the receptor was corrected from
> 30 m to ~10 m. Stages 0-6 have all passed at least once at 24-30 m and a 30 m receptor
> (`FOURTH_PASS_RESULTS.md`); this pass rebuilds the configuration around the real
> instrument height and re-runs them. **Absolute distances from earlier passes do not carry
> over. Methodology, traps and closure findings do.**

Goal of this pass: **one validated configuration at a 10 m receptor, producing one flat/neutral
control footprint and then the production directions.**

**Validate the configuration as ONE thing.** Every change below lands together, gets a batch of
short smoke runs, and then one full window. No one-at-a-time experiments — the fourth pass
established that the expensive failures are interactions (a scoring bug hidden by a path
assumption, a closure rescaling hidden by a gate run in the wrong mode), and those only show up
when the whole configuration runs end to end.

**HARD RULE: no single run over 45 minutes wall.** Project before launching. Everything long is
a chain of sub-45-minute segments driven by `bin/run_window.sh` / `bin/run_directions.sh`, which
compute the projection and refuse rather than ask.

---

## What changed, and why each change is here

| # | change | reason |
|---|---|---|
| 1 | **Receptor 30 m -> 10 m** | the instrument height. Everything else follows from it. |
| 2 | **Grid 186x186x122 @ 10 m, `dz_sfc` 3.9933 m** | receptor lands on a cell centre at exactly 10.000000 m at `k = 2`; domain 1860 m contains `x90` for every stability class |
| 3 | **Sub-grid 40% gate retired** | unreachable by ~2 orders of magnitude at a 10 m receptor. Replaced by measure-and-report + the well-mixed gate + a stated sensitivity band. |
| 4 | **24 m vs 12 m convergence test dropped** | the grid is changing anyway |
| 5 | **RSL caveat adopted** | panels 2-3 m, RSL 5-15 m, receptor 10 m. Kljun and the MOST floor are both weakened over the array. |
| 6 | **`lsfSelector` subsidence** | the physical fix for Stage 2 stationarity, and the control on `z_i` |
| 7 | **`stabilityScheme = 2` fitted to CONUS404** | 4-segment piecewise-linear theta, fitted, not injected |
| 8 | **`Ug_grad`/`Vg_grad`** | geostrophic forcing takes a linear vertical gradient; use it if the sounding wants one |
| 9 | **Dry, virtual heat flux** | explicit call. Moisture rejected; buoyancy deficit absorbed into `htFlux`. |
| 10 | **`surflayer_idealsine` rejected** | it overwrites the per-cell `htFlux` map, which is load-bearing |
| 11 | **Online footprint rejected** | IO is 3% of compute; forward tile-resolved aux scalars are a worse estimator |
| 12 | **Array `z0` 0.10 m** | the first model level is at 2.0 m; a larger `z0` leaves the log law no room |

---

## Phase A — offline, no GPU (~1 day)

Everything here runs on fields and rasters already on disk. **Do it all before any GPU time**,
because two of the items can still change the domain size.

1. **Re-evaluate the footprint geometry on the REAL map at `z_m = 10 m`.** Kljun on the actual
   WorldCover/3DEP grid, not the idealised estimates in CLAUDE.md. Report: fraction of the
   footprint inside the array reach by direction and stability; fraction beyond 930 m; the
   water share. **If the water share at 10 m exceeds ~10% in any direction, the 1860 m domain
   is too small and the fallback is `N = 234 @ 8 m` or a return to a larger box.** This is the
   last cheap chance to change the domain.
2. **Re-measure `t_back` at 10 m** by masking one existing release ensemble on touchdown age —
   the same method as the fourth pass, no new LES. It cannot be done at the new grid until a
   window exists, so do it first on the 24 m fields with the receptor moved to the nearest cell
   to 10 m. That gives the *scaling*, which is what sizes the window. Expect ~150-250 s
   convergence and a **35-minute window**; confirm before writing the drivers.
3. **Add displacement height to the LPDM's similarity functions.** Every MOST argument becomes
   `(z - d)/L` with `d` from the per-cell land cover (array 1.5 m, tree 0.7 x height, crop/grass
   ~0.1 m). This is a 10-15% correction at a 10 m receptor and it was negligible at 30 m.
4. **Fit `stabilityScheme = 2` to the CONUS404 mean sounding.** Six parameters
   (`zStableBottom{,2,3}`, `stableGradient{,2,3}`) by least squares on the lowest 1.5 km. Report
   the RMS misfit in K. **Only if that misfit is demonstrably inadequate** — say, more than
   ~0.5 K through the inversion — fall back to injecting a 3-D theta field via the restart file.
   Also fit `Ug_grad`/`Vg_grad`/`z_Ug`/`z_Vg` to the mean wind profile in the same pass.
5. **Rebuild the surface** with `bin/prep_surface.py` on the 186 x 186 @ 10 m box: 3DEP at native
   resolution (**measure the slope distribution and decide on a light smooth**), WorldCover by
   mode, array rectangle override at `z0 = 0.10 m`, per-cell `htFlux` as the **virtual** flux.

**Gate A:** the array-share and water-share table at `z_m = 10 m` on the real map, and a
`t_back` number. Both are inputs to the domain and window size. Commit before Phase B.

---

## Phase B — smoke runs, one batch (~2 h GPU)

Short runs only. Nothing here exceeds a few minutes. Run them as one batch and score them
together.

1. **Grid launches.** 200 steps at `186 x 186 x 122`. No `GRID_CUDA_DECOMPOSE_FAIL`, no
   `too many resources requested for launch`.
2. **Thread-block sweep.** `1x2x64` is the incumbent and `(N+6) = 192/192/128` leaves every
   candidate legal. 200 steps each, ~2 minutes total. Pick the fastest, record s/step.
3. **`dt` bisection, FLAT.** Start at `1/68 = 0.0147059 s` (`CFL_3d = 1.468`) and bisect with
   `docker/diag_near_surface.py`. Pass condition `k0/k1 < 1`. Find the accuracy boundary, then
   take ~10% margin below it.
4. **`dt` bisection, TERRAIN.** Separately, on the real surface. `dx/dz = 2.504`, but the 3DEP
   slopes are steeper at native resolution — the amplification must be measured, not scaled from
   the 24 m grid. Expect a lower `dt` than the flat case; the two are not interchangeable.
5. **Restart injection.** Write terrain, terrain-following `zPos`, `z0m`, `z0t`, `htFlux` into a
   restart file and confirm the read is a no-op: dump the fields back out and compare.
6. **90-degree equivariance.** Re-index a short flat run by 90 deg, run 200 steps, compare
   profiles against the unrotated run. Must agree at the ~1e-4 nondeterminism floor.
7. **`lsfSelector` + `lsf_horMnSubTerms` smoke.** 200 steps with subsidence on, confirm the run
   is clean and that `w` acquires the prescribed slab-mean subsidence.
8. **Halo check.** `ncdump -h` the first dump: dimensions must be `186 x 186 x 122`, interior
   only. *(Confirmed on the 24 m grid 2026-08-21; re-confirm here because it fixes the CNF
   raster shape.)*

**Gate B:** all eight clean, and two `dt` values recorded — flat and terrain. **Every script must
grep output for `CORRUPTED`/NaN and must never trust the exit code alone**, and every check must
test `np.isfinite(...).all()` FIRST — `inf` is not NaN, and a NaN passes every `>` comparison.

---

## Phase C — spin-up (12.2 GPU-h per base state, ~26 chained segments)

Flat, uniform, doubly periodic. One base state per `(stability, wind speed)` bin; 90-degree
re-indexing gives four directions from each.

- Neutral: `stabilityScheme = 2` with the fitted capping inversion, `lsfSelector = 1` +
  `lsf_horMnSubTerms = 1`. **The capping inversion plus subsidence is the physical fix for the
  Stage 2 stationarity failure** — an idealised neutral BL with no inversion has no equilibrium
  depth and `u*` drifts forever.
- Convective: same, plus `surflayer_wth` as the virtual flux, with the inversion and subsidence
  chosen to hold `z_i` **below ~465 m** (`L >= 4 z_i`). Record the achieved `z_i` per segment.

**Gate C1:** turbulence statistically stationary — domain TKE plateau and `u*` trend within
~2 sigma. This is the gate that failed at -8.4 sigma in the second pass and passed in the third;
subsidence is here to make it pass robustly rather than by luck of sampling time.

**Gate C2:** the saved restart restarts bit-for-bit. Already verified at three grids; re-verify
once at this one, it costs minutes.

**Gate C3 (convective only):** CBL similarity — `w*/u*`, entrainment ratio ~0.2, `sigma_w/w*`
against Lenschow. `bin/cbl_check.py`. **It must read the prescribed `htFlux`, not the resolved
covariance at `k = 0`** — that bug made a real CBL look like it was not one.

---

## Phase D — the flat/neutral control, first and standing (~2.5 GPU-h)

**This is the first full window, and it is the standing regression.** Re-run it at every
configuration change, not once. It is the only place Kljun is diagnostic — uniform `z0 = 0.03 m`
puts the RSL top at 0.3-0.5 m, well below a 10 m receptor — and it is the canary for silent
geometric bugs. It earned that on its first run by finding two.

1. One (30 min + `t_back`) window, `--rel-seconds 1800`, 5 s cadence.
2. **Well-mixed test in the production closure configuration**: `stage4_wellmixed.py --sgs-most`,
   with the displacement-height correction active. Backward rms against the counting-noise floor.
3. Footprint, and Kljun on the identical cells.
4. Half-vs-half error floor: peak, centroid, 80% source-area overlap.
5. Sub-grid fraction of `sigma_w^2` at the receptor — **reported, not gated**.

**Gate D1 — well-mixed.** Backward rms within the counting-noise floor, lowest three bins within
tolerance. Non-negotiable: if particles accumulate near the surface, every footprint computed
afterward is wrong in exactly the near field where the whole signal now lives.

**Gate D2 — the integral.** It must converge **from below** with the wrap cap on, and it must be
compared against Kljun evaluated on the same box, never against 1. At 10 m only 7-9% of the
footprint lies beyond 930 m, so expect the domain shortfall to be small — if the integral lands
far from Kljun's value on the same cells, that is a real inconsistency, not truncation.

**Gate D3 — error floor.** Half-vs-half 80% overlap must sit clearly above the LES-vs-Kljun
overlap, or the metric is at its own noise floor and cannot score the emulator. It separated at
59.2% vs 36.9% in the second pass; re-measure.

**Also record here, once:** the ensemble-convergence curve at the new receptor height — peak and
centroid p90 against sampling time, from sub-windows of one integration. It is the corpus design
parameter and the absolute metres will have changed with the footprint.

---

## Phase E — production directions (~2.24 GPU-h each, 3-4 chained segments)

Four directions per base state, by 90-degree re-index, then restart onto the real surface with
~20 min of adjustment before sampling.

**Gate E — explicable difference.** The footprint must differ from Kljun in a direction you can
point at. **This gate is weaker at 10 m and the plan should say so up front**: the array's
N-vs-E/W footprint share ratio falls from ~370x measured at 30 m to about **3.7x**, because at a
10 m receptor the array is 15-24% of the footprint even crosswise. What replaces the swing as the
discriminator:

- **Absolute array share by direction**, predicted from the array chord and the LES's own `f_y`,
  against measured. At 30 m these agreed to 0.6% — that is the real test, and it is sharper than
  a ratio.
- **The near-field peak position** against the upwind roughness transect (`bin/upwind_transect.py`).
- **Terrain response**, which is now a larger fraction of what is left once the array is
  subtracted.

**Report agreement to fewer significant figures, and state which parts are constrained and which
are free.** Over terrain and over the array, Kljun is descriptive, not a target. The near-field
peak is constrained by the MOST-anchored `sigma_w` floor, which is anchored to the same theory
Kljun rests on — and inside the array's RSL that anchoring is an extrapolation. The 80% area, the
tail, and the land-cover shares are free. Quote the measured anchor-sensitivity band (**46-66%
shape L1**, against a 38% sampling floor) alongside any near-field number.

---

## After Phase E

Only then: corpus design, wind-rose stratification with a directional floor, CNF implementation.

Do not start ML work before Phase E passes. A trained model on a broken target pipeline looks
exactly like a trained model on a correct one.

**The CNF raster is `186 x 186`** — the LES interior, no halos, confirmed by `ncdump`.

---

## Known limitations to state wherever the corpus is described

1. **The receptor may be inside the roughness sublayer over the array.** MOST does not hold
   there, so Kljun is not a reference over the array and the `sigma_w` floor is extrapolated.
2. **The first model level is at 2.0 m, at or below panel top.** The array's surface exchange is
   parameterised, not resolved. Dominant known modelling uncertainty.
3. **Deep convective boundary layers are out of reach.** `L >= 4 z_i` caps `z_i` at ~465 m in an
   1860 m box; the site's convective-midday median is 859 m.
4. **The lake is outside the domain.** At a 10 m receptor it carries ~3% of the footprint rather
   than the ~35% it carried at 30 m, but that is a Kljun estimate — confirm it in Phase A.
5. **The near field is closure-dominated** at `z/Delta = 1.36`, worse than the 1.76 of the
   previous configuration, because the eddy scale shrinks with height and `Delta` does not.
6. **Real geography extends only to the taper ring**, ~730 m from the tower for terrain and
   ~930 m for land cover; beyond that the surface is uniform.

---

## Working agreement

- One phase per session where possible.
- Report the gate result explicitly before moving on.
- If a phase reveals the plan is wrong, say so and stop. Do not work around it silently.
- Prefer reading FastEddy's own source over inferring behaviour from documentation. Every
  capability claim in CLAUDE.md now carries a file and line number; keep it that way.
- Commit at every passed gate. FastEddy source edits go to the fork on `kegonsa`; everything else
  to the main repo.
- Stop early only if a gate fails three times, a fix needs FastEddy source changes beyond the
  existing fork, or a segment projects over 45 minutes and cannot be broken into a chain.
