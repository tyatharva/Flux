# Ruled out

Things that were considered and rejected, with the reason. Do not propose them again without
new evidence.

| idea | why not |
|---|---|
| STILT | Replaced by this project's own backward LPDM. |
| Mesoscale coupling (`hydroBCs = 1`, GenICBCs, cell perturbation) | The fetch requirement would consume most of the domain. The domain is periodic instead. |
| LES-to-LES nesting, NSCBC, 512³ domains | Schedule, unnecessary, infeasible, respectively. |
| Running FastEddy backwards in time | Mathematically impossible, not a code limitation. Reversing `t` and `u` flips the sign of the SGS stress term, giving negative eddy viscosity and the backward heat equation. The backward LPDM steps *particles* backward through *forward-stored* fields. |
| Multiple virtual tower locations | Would add unexplained variance. One fixed tower. |
| Surface fields as ML inputs | Out of scope. The emulator takes Kljun's six scalars only. |
| `surflayer_idealsine` or a diurnal cycle within a run | Both branches assign a scalar to `htFlux`, overwriting the per-cell map that gives the array its enhancement. |
| Moisture (`moistureSelector = 1`) | Run dry and prescribe the virtual heat flux instead. |
| Sub-grid fraction < 40% as a gate | Retired. Unreachable at any affordable grid (reaching it at `z = 10 m` needs `dx ≈ 3–4 m`, about 22× this configuration). |
| A neutral well-mixed PASS as evidence about the convective closure | The floor is nearly inert neutrally. It passed a closure with nine turnovers. |
| Stable corpus cases at this grid | `L_O/Δ = 3.57` at the receptor against 318 neutral. GABLS1 runs that regime at `dx = 6.25 m`, 17× the cells. See [limitations](../limitations-and-future-work.md). |
| A post-hoc wraparound mask as a *correction* to the footprint integral | Built twice (half-plane, then cone), measured both times, refuted both times by the sign of its own correlation. The cone stays as an operational cleanup only. |
| A downwind half-plane as the wraparound cut | It cannot see single-axis wrap, which lands back upwind. Superseded by the cone. The clean fix, recorded rather than done because it needs a corpus regeneration, is to deposit the *unfolded* displacement at generation time. |
| Online footprint calculation inside FastEddy | IO is ~3% of compute, so it solves a problem that does not exist, and it would be a worse estimator. Forward tracers resolve source *tiles*, so the footprint's resolution becomes the number of tracers you can afford and the near field would be the coarsest part. |
| Fitting `stabilityScheme = 2` to a CONUS404 *mean* sounding | `conus404_hourly` has no time-varying atmospheric profiles at all. A per-case fit to a real HRRR profile is the corpus mechanism and is different. |
| CONUS404 as a forcing | It sets sweep ranges and is the 45-year climatology that characterises the site. HRRR forces the runs. |
| A U-Net baseline beside the FNO | A second architecture is a second thing to validate, and the FNO's resolution-independence is the property that matters. (The CFM later used a U-Net as its velocity network for a different purpose: sample spread.) |
| A conditional normalizing flow on the touchdown point process | The 2026-08-29 target design. Reversed on 2026-08-30. The target is the raster, touchdowns are not saved, and the model is an FNO residual on Kljun. See [targets and architecture](../emulator/targets-and-architecture.md). |
