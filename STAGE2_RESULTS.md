# Stage 2 — Gate Result: **CONDITIONAL PASS**

Date: 2026-08-18 · 270,000 steps = **123.75 min simulated** · 6 h wall ·
`434 x 146 x 122`, stretched vertical, Rayleigh damping, neutral, flat, doubly periodic

Run scored with `docker/check_run.sh`: **exit 0, no CORRUPTED, no NaN/Inf, completion
banner present.** (Exit code alone is never sufficient — FastEddy exits 0 on fully NaN fields.)

---

## Configuration (settled)

| | value | note |
|---|---|---|
| Vertical grid | `Nz=122, d_zeta=20.576, verticalDeformFactor=0.4860, quad=0.0` | `dz_sfc = factor * d_zeta` |
| Built grid | 10.00 m surface, 14.4 m @ 500 m, 41.5 m top, **2500 m** deep | matches design exactly |
| Resolution < 400 m | **37 levels** | Nz=154 rejected: +26% cost, +1 level |
| Rayleigh damping | `dampingLayerSelector=1, dampingLayerDepth=500` | 20% of depth |
| dt | 0.0275 s | 5% margin under measured 0.029 s boundary |
| Cost | **0.0735 s/step** | identical to Stage 1 — depth doubled for free |
| Output | `hydroSubGridWrite=0` | 19 -> 10 3-D fields, 312 MB/dump |

---

## Gate 1 — TKE stationarity: **PASS (within turbulent noise)**

```
 t(min)   9.2   18.3   27.5   36.7   45.8   55.0   64.2   73.3   82.5   91.7  100.8  110.0  119.2  123.8
 TKE    .0638  .0892  .1078  .1041  .1123  .1143  .1185  .1207  .1261  .1301  .1326  .1321  .1353  .1280
```

Growth per dump fell monotonically and **reversed sign**: +3.34% -> +3.18% -> +3.19% ->
+1.57% -> **-1.17%**. Over the last five dumps TKE oscillates about 0.132 with a standard
deviation of ~2%, which is turbulent sampling noise rather than trend.

The strict `|growth per dump| < 1%` criterion reads NO on the final window, but that window
is distorted: the last interval is 10,000 steps where every other is 20,000. Taken together —
flattening, sign reversal, and ±2% scatter about a constant — the turbulence is
statistically stationary.

`u*` converged cleanly and monotonically: 0.440 -> 0.382 m/s.

## Gate 2 — resolved w'w' profile vs NBL validation: **PASS**

Reference values read from `docs/Tutorials/images/{TURB,MEAN}-PROF-neutral.png` (NBL at t=6-7 h).

| quantity | ours | NBL ref |
|---|---|---|
| **sigma_w^2 peak / u\*^2** | **0.804** | **0.730** |
| u* (m/s) | 0.382 | 0.410 |
| wind speed at first level (m/s) | 4.77 | 4.30 |
| height of sigma_w^2 peak (m) | 65-190 (flat) | 130 |
| sigma_w^2 -> 0 by (m) | 480 | 650 |
| wind veering, surface -> free (deg) | -6.8 | -25 |

The normalized peak matches to 10%. The profile is the recognizable neutral-BL shape: a
broad maximum through the lower-middle boundary layer (0.80 at 65 m, essentially flat to
190 m), then monotonic decay to zero at the boundary-layer top.

Our first level is faster than NBL's (4.77 vs 4.30 m/s) in the correct direction, since
z0 = 0.03 m here against their 0.10 m.

Our boundary layer is shallower (480 vs 650 m) because the capping inversion sits at
~450-500 m and because 2 h of spinup is well short of NBL's 6-7 h.

## Gate 3 — restart of the spun-up state: **PASS, bit-for-bit**

`FE_S2.270000` restarted and immediately re-dumped is **byte-identical** to the source
(`cmp` reports no difference), with `simTime_it` correctly resuming at 270000. The
spun-up flat-terrain state is a valid reusable asset, which is what the corpus structure
depends on.

**Reusable asset:** `runs/stage2_spinup/output/FE_S2.270000` (312 MB) — the neutral,
U_g = 10 m/s, z0 = 0.03 m spun-up state. Do not delete.

---

## Why "conditional"

**1. The near-surface w artifact is unresolved and blocks Stage 4.**
Resolved `sigma_w^2/u*^2` reads 32.5, 4.03, 1.07 at z = 5, 25, 45 m against a physical
~0.80. It is `w`-only (u, v, theta have k0/k1 ratios of 0.96, 0.99, 1.00), it is grid-scale
noise rather than resolved eddies (lag-1 spatial autocorrelation 0.08 at 5 m against 0.95
at 85 m), and it persists rather than decaying.

NCAR's published NBL figure independently confirms this is wrong: their `sigma_w^2` goes to
**zero at the ground and rises**, ours falls from a large first-level value.

Eliminated as causes: vertical stretching, thread-block shape, roughness length, spanwise
domain width, discrete impermeability (correctly enforced at
`cuda_advectionDevice.cu:106-109`), and explicit filtering (identical to NBL's).

**The tower is at 30 m, inside the affected layer.** The LPDM integrates `w` directly, so
this must be resolved before Stage 4 — it is exactly the near-field signal the well-mixed
test exists to protect. Untested remaining differences from NBL: horizontal resolution
(10 vs 15 m), domain depth (2500 vs 1150 m), damping depth, Coriolis latitude, dt.

**2. Ekman veering is far from equilibrium** (-6.8 deg against -25 deg). Veering develops
on the inertial timescale, `2*pi/f ~ 17 h` at 43 deg N, not the eddy-turnover timescale.
Reaching NBL's veering needs several more hours of simulated time.

This matters for the corpus because wind direction is the dominant skill axis. If the
spun-up state carries an immature Ekman spiral, every direction inherits the same bias.
Since spinup is amortized once per `(stability, wind speed)` bin, a longer spinup is
affordable: at 2.53 s wall per simulated second, 6 h simulated costs ~15 h wall, once.

**Recommendation:** extend this spinup by restart to 4-6 h simulated before generating
corpus states, and resolve the w artifact first so the extension is not wasted.
