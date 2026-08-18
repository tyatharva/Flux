# Stage 1 — Gate Result: **PASS**

Date: 2026-08-18 · Grid `434 x 146 x 122` @ 10 m · thread block `4 x 4 x 16` ·
flat, uniform roughness, neutral, doubly periodic · 1 rank / 1 GPU

---

## Gate

| Check | Result |
|---|---|
| Clean exit | **PASS** — exit 0 |
| `too many resources requested for launch` | **absent** — `4 x 4 x 16` works |
| `GRID_*_FAIL`, `FECUDA_THREADS_PER_BLOCK_FAILURE`, `MPI_ERR` | 0 occurrences |
| Field corruption (FastEddy's own NaN/Inf detector) | 0 fields corrupted |
| **Wall clock per simulated second** | **2.53 s/s** (at dt = 0.029 s) |
| GPU memory high-water | **2,757 MiB** of 16,376 MiB (17%) |

Divisibility, per-rank and halo-inclusive: `(434+6)%4=0`, `(146+6)%4=0`, `(122+6)%16=0`;
256 threads/block. Domain 4.34 x 1.46 x 1.22 km, 7,730,408 cells.

### Stage 0a extrapolation was accurate

| Quantity | Predicted (Stage 0a) | Measured (Stage 1) | Error |
|---|---|---|---|
| Compute per step | 0.0725 s | 0.0733 s | **1.1 %** |
| NetCDF dump size | 0.592 GB | 0.590 GB | **0.3 %** |

The 9.37 ns/cell/step cost model holds across a 12x range in grid size and now at the
production grid. Treat it as reliable for corpus planning.

---

## dt: acoustically limited, and there is no headroom

**FastEddy contains no CFL machinery at all** — no Courant number, no stability check, no
adaptive timestep. `dt` is a mandatory user constant bounded only by `FLT_MIN..FLT_MAX`
(`SRC/TIME_INTEGRATION/time_integration.c:61`), broadcast and used as given. Every
tutorial's `dt` was therefore hand-picked, and the tutorials do **not** share a common CFL
(NBL sits at 3-D acoustic CFL 1.603, SBL at 0.904), so they are no guide.

The solver is fully compressible — pressure diagnosed from `rhoTheta` via the EOS, RK3
(Wicker-Skamarock 2002), **no acoustic sub-stepping**, optional divergence damping. Sound
waves are resolved explicitly, so the limit is acoustic, not advective. (Advective CFL here
is ~0.03 — two orders of magnitude from binding.)

### Measured stability boundary

Bisected directly on the production grid, 500 steps per trial, scored by FastEddy's own
`****CORRUPTED*** --- (#NaN, #Inf)` detector. **Note that detector logs but does not change
the exit code — FastEddy exits 0 on a fully NaN field**, so exit status alone is not a
stability test.

| dt (s) | acoustic CFL_3d | result |
|---|---|---|
| 0.0267 | 1.606 | STABLE |
| 0.0280 | 1.684 | STABLE |
| **0.0290** | **1.744** | **STABLE — max usable** |
| **0.0300** | **1.804** | **UNSTABLE** |
| 0.0310 | 1.864 | UNSTABLE |
| 0.0350 | 2.105 | UNSTABLE |
| 0.0600 | 3.608 | UNSTABLE |

with `CFL_3d = dt * c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)`, `c = 347.2 m/s` at 300 K.

**The 3-D combined metric governs, not 1-D minimum spacing.** The boundary at CFL_3d
between 1.744 and 1.804 brackets RK3's theoretical imaginary-axis limit **sqrt(3) = 1.7321**
to within a few percent. The 1-D min-spacing theory predicted dt_max = 0.050 s; that is
decisively wrong — 0.050 s is fully corrupted.

**Consequence: dt was NOT inherited-conservative, and the cost lever is not there.** The
Stage 0a working value of 0.0267 s was already within 8 % of the true limit. Adopting
dt = 0.029 s buys 8 %, nothing more.

~~Recommended production value: dt = 0.0275 s (5 % margin below the measured boundary).~~

**SUPERSEDED 2026-08-18 — this recommendation was wrong. Use `dt = 0.0250 s`.**

The boundary bisected above is the **stability** boundary (NaN / `CORRUPTED`). There is a
**lower accuracy boundary at CFL_3d ~ 1.64, i.e. dt ~ 0.0273 s**, above which the run
completes normally and exits 0 while resolved `w` in the lowest ~3 levels degenerates into
grid-scale acoustic noise. `dt = 0.0275` sat inside that window, which is what produced the
Stage 2 near-surface artifact. See `STAGE2_RESULTS.md` and CLAUDE.md.

---

## Runtime, and how the Stage 2 restructuring resolves it

At dt = 0.029 s: **2.53 s of wall clock per simulated second**.

| Run type | Simulated | Wall clock |
|---|---|---|
| Full 3 h run | 10,800 s | **7.58 h** |
| Spinup, 2 h | 7,200 s | 5.06 h |
| **Per-direction run** (20 min adjust + 30 min sample) | 3,000 s | **2.11 h** |

A monolithic 3 h run is **7.58 h**, still far above PLAN.md Stage 1's ~4 h flag threshold.
But under the Stage 2 restructuring — spin up over flat uniform terrain, then reuse that
state across all wind directions in a `(stability, speed)` bin — the *per-direction* cost is
**2.11 h, comfortably under the threshold**, with one 5.06 h spinup amortised across every
direction in the bin. The corpus arithmetic works; the monolithic framing was the problem.

---

## Correction to the Stage 0a output-field finding

Stage 0a recorded "FastEddy exposes no output field selection". **That was too absolute.**
Coarse, module-level output switches do exist:

- `hydroSubGridWrite` gates registration of the 9 SGS stress fields
  (`Tau11 Tau21 Tau31 Tau32 Tau22 Tau33 TauTH1 TauTH2 TauTH3`) at
  `SRC/HYDRO_CORE/hydro_core.c:1224` — `if((hydroSubGridWrite == 1) && (turbulenceSelector > 0))`
- `hydroForcingWrite` gates the `Frhs` tendency fields

Measured directly:

| Setting | 3-D fields | B/cell | Dump | 30-min window @ 5 s |
|---|---|---|---|---|
| `hydroSubGridWrite = 1` | 19 | 76.3 | 590 MB | **212 GB** |
| `hydroSubGridWrite = 0` | 10 | 40.3 | 312 MB | **112 GB** |
| + fp16 on write | 10 | 20.2 | 156 MB | **56 GB** |
| 4 LPDM fields only + fp16 | 4 | 8.0 | 62 MB | **22 GB** |

What remains at 10 fields is `xPos yPos zPos rho u v w theta pressure TKE_0`. The LPDM needs
only `u v w TKE_0`. Note `xPos/yPos/zPos` are 12 B/cell of **time-invariant grid geometry
rewritten identically in every dump** — the single largest pure waste in the output.

**So: configuration alone reaches 112 GB, and 56 GB with fp16 — still above Stage 3's ~30 GB
gate.** Clearing that gate requires restricting to the 4 LPDM fields, which is not
config-reachable and is a Stage 3 source change. `io_binary.c` remains untouched.

---

## Notes carried forward

- FastEddy's corruption detector does not affect exit status. **Every production run must be
  scored by grepping for `CORRUPTED`**, not by exit code alone.
- `verticalDeformFactor` compresses the vertical grid toward the surface, so the near-surface
  `dz` — not `d_zeta` — sets the acoustic limit. Stage 2's stretching must re-derive dt: with
  dx = dy = 10 m fixed, even an infinitely coarse vertical only relaxes `CFL_3d` by
  `sqrt(3/2)`, i.e. **dt at most 22 % larger**. Vertical stretching is for domain depth, not
  for speed.
