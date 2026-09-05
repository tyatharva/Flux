# The LPDM and the footprint estimator

`lpdm/` is this project's own backward Lagrangian particle dispersion model. It releases
particles at the receptor, integrates them backward through the stored LES fields plus a
sub-grid Langevin model, and turns their touchdowns into a signed flux footprint on the LES's
own cells. Everything below was validated on real fields; the gates are in
[gates and diagnostics](gates-and-diagnostics.md).

## The estimator

Flux footprint after Flesch, Wilson & Yee (1995) and Flesch (1996):

$$
\frac{F}{Q} = \frac{1}{N}\sum_i w_{\mathrm{release},i}\sum_{\text{touchdowns of } i}\frac{2}{|w_{td}|}
$$

- `w_release` is the vertical velocity the trajectory had at the receptor, resolved LES value
  plus a sub-grid draw. Trajectories arriving with `w > 0` came from below and carry
  surface-influenced air; those with `w < 0` came from aloft. Their touchdown densities differ,
  and the signed difference is the flux footprint, so the estimator never uses `|w_release|`.
- The weight is taken in the **streamline frame** (Wilczak double rotation): first about `z` so
  the mean crosswind vanishes, then about the new `y` so the mean vertical vanishes. It is the
  frame an eddy-covariance system reports in and it makes the mean vanish by construction. It
  removes 95–100% of the model-frame mean `w` from the weight; what it cannot remove is `w̄`
  from the *transport* (below).
- **The touchdown weight uses the surface-normal approach rate** `|d(z − z_ground)/dt|`, not `|w|`.
  Over sloping ground the `2/|w|` weight explodes; flat ground hides this completely.
- **Deposition is cloud-in-cell** onto the LES column index, exactly conservative and 0.67× the
  per-cell noise of nearest-grid-point. **The raster is the LES grid**: nothing is rotated or
  interpolated, and a footprint cell is an LES column, the same indexing the land-cover masks use.
- **Negative values are physical and nothing clips them.** The estimator is signed by
  construction, cloud-in-cell takes signed weights, the persisted array is unclipped, and the
  `np.maximum(f, 0)` calls that exist are metric-side (`bin/test_negative_lobes.py` asserts all
  of it). The lobe is 1.59% of |f| after the cone.
- **Trajectories are retired at one streamwise domain length** (`max_disp = fs.Lx` = 3660 m).
  A backward trajectory that travels further re-enters turbulence it already sampled and its
  later touchdowns are the same eddies counted again. Uncapped, the flat-control integral sailed
  past 1 exactly as wrapping set in; capped, it converges to 1 from below. What the cap still
  leaves, the per-axis fold of a single wrap, is what the [cone](../history/cone-mask.md) removes.
- **The integral asymptotes to `1 − z_m/z_i`, not to 1** (Steinfeld et al. 2008, after Horst &
  Weil 1992): the fraction `z_m/z_i` of the column lies below the receptor and its flux never
  crosses it. At 30 m in an 800 m layer that is 3.75%. Departure from the asymptote tracks `w̄`
  at the receptor with the right sign (subsidence 1.497×, updraft 0.916× on two cases of
  opposite sign): the advective non-closure that makes eddy covariance hard over complex terrain.
  An integral that crosses 1 and keeps climbing cannot be truncation and is always a model
  inconsistency; a saturating integral above 1 is `w̄` times the concentration integral.

**The constant was verified without the LES** (`bin/test_estimator.py`): in homogeneous
turbulence with a reflecting lid the surviving flux is `Q(1 − (z − z_td)/H)`, a lid-dependent
target that a wrong constant cannot hide behind; measured 0.530 and 0.753 against 0.600 and
0.778 within their standard errors.

## The reverse-time drift

Reversing a Langevin model by substituting `(u, t) → (−u, −t)` gives an anti-damped velocity
equation that diverges. Thomson (1987, §5): for `dX = A dt + B dW` with stationary density `p`,
the reverse drift is `Â = −A + (BBᵀ)∇ ln p`. With `p` Gaussian in `u`,

$$
\hat A_i = -\frac{C_0\,\varepsilon}{2\sigma^2}u_i \;-\; \frac{1}{2}\Big[\frac{\partial\sigma^2}{\partial x_i} + \frac{u_i u_j}{\sigma^2}\frac{\partial\sigma^2}{\partial x_j}\Big]
$$

so the damping keeps its sign and only the `σ²`-gradient term flips. Getting this wrong either
diverges (loud) or drops the gradient term and accumulates particles at the surface (silent, and
looks like a plausible footprint). Forward and backward well-mixed tests agreeing is what
confirms the sign.

## Fields and the sub-grid model

`lpdm/fields.py` loads the window into a time-indexed **float16 cache** (12.0 GB for a 541-dump
production window; a 4-D linear interpolation written by hand because `scipy.ndimage.map_coordinates`
refuses float16, matching it to float32 roundoff). fp16 was verified harmless on real fields:
peak identical, centroid +19 m, integral +0.015, overlap 75.7% against a 59% half-vs-half floor.

The sub-grid velocity is a Langevin model (Weil, Sullivan & Moeng 2004) driven by FastEddy's
own `TKE_0` and its own dissipation: `eps = c_e e^{3/2}/l` with `l = min(0.76√e/N, Δ)` when
`N² > 0` else `Δ`, `Δ = (dx dy dz J)^{1/3}`, `c_e = 0.93`, read out of `cuda_sgstkeDevice.cu` and
recomputed at load time. A Langevin model driven by an inconsistent `eps` fails the well-mixed
test in a way that looks like an integrator bug.

Below the lowest LES level the LES carries no information: the horizontal wind is continued by
the log law anchored at that level with the displacement height `d ≈ 1.5 m` over the array,
resolved `w` goes to zero linearly at the ground, `eps` follows surface-layer `1/z` scaling, and
`σ_s²` is held constant so its gradient is zero and the sub-layer cannot manufacture
accumulation. Particle state is fp64 throughout.

**The LPDM is forked into 16 fixed chunks with per-chunk seeds**, so worker count is a pure
performance knob and cannot change a result (asserted bit-identical, 1 worker vs 12,
`bin/test_parallel_lpdm.py`; 6.8× on 12 workers). Workers are forked so the field cache is
shared copy-on-write; do not switch to spawn.

## The `σ_w` closure

At `z/Δ = 1.52` the eddies that carry `w` sit at or below the filter scale, so the LES
under-resolves `σ_w` at the receptor, backward particles descend too slowly, and the footprint
lands too far out and too broad (the third pass measured `σ_w/u*` 1.09 against the
surface-layer 1.25; peak +86%). The fix is a **MOST-anchored floor** (`lpdm/sgs_floor.py:most_floor`,
in one place; the gates import it):

- target `σ_w = 1.25 u* φ_w(z/L)` with `φ_w = (1 − 3z/L)^{1/3}` unstable (Panofsky et al. 1977)
  and `1 + 0.2 z/L` stable; `φ_w(0) = 1`, so neutral results are unchanged;
- it supplies only what similarity says is missing, never reduces the LES's own variance, and
  tapers off across `0.1h–0.2h` because MOST is a surface-layer relation;
- **weighted by the sub-grid fraction**: `sc_eff = 1 + (sc − 1)·f_sgs`, `f_sgs = (2/3)e/(ww + (2/3)e)`,
  with monotonicity re-imposed afterwards;
- **`eps` scaled with `σ²`** so `T_L = 2σ²/(C0·eps)` is preserved, both sides of the ratio floored at
  the same value (a clip on one side inflated `eps` a millionfold above the boundary layer and
  pinned every particle at `dt_min`; the only symptom was a 4× slowdown);
- the drift carries the floor's gradient: `sc·dσ²/dz + (2/3)e·dsc/dz`, where the second term is
  the larger. Omitting it made the integral inflate with integration time and cross 1.

The cause of the old failure was the **magnitude** of the inflation, not its shape: a constant ×10
with no taper failed the well-mixed test at 1.370 while a constant ×1.673 passed at 1.130. Two
earlier diagnoses were wrong and each cost a rebuild ([sixth pass](../history/pass-6.md)). Gate
D1 passes in both regimes and both directions with 0 turnovers, within counting noise of the
unmodified model, while supplying 3.49× where the variance is genuinely sub-grid.

**The near field is closure-dominated, and that is a number**: the floor is worth +8.40 points
of convective array share and shortens `x80` from 400 to 227 m on the flat convective control
(+7.37 points on the 30 m convective target); the retired closure inflated that share by up to
+18.46 points. Sub-grid fraction at the receptor: 52.5% convective, 86.4% neutral (57.7–59.3%
and 90.4% measured on the 30 m targets); reaching 40% at a 10 m receptor would need `dx ≈ 3–4 m`.
The choice of anchor (Panofsky surface-layer, adopted, against Lenschow mixed-layer) moves the
crosswind-integrated shape by 46–66% L1 against a 38% sampling floor; quote that band with any
near-field number. **Quote the compaction ratio with its closure**: floor off 0.57× (convective
broader), floor on 1.33× (more compact), and both pass Gate D1, because well-mixedness tests
self-consistency, not whether `σ_w` has the right magnitude.

## `h`, the boundary-layer depth

`lpdm/les_stats.py:bl_depth` finds the surface-attached layer by walking up from the ground on a
smoothed resolved-TKE profile to its first minimum, and requires the column's global peak to lie
inside it. The test is "a second layer that out-energises the first", not "the first local
minimum": using the first strict minimum moved `h` on 15 of 47 stored profiles by up to 331 m on
profiles with no wave layer. What identifies a wave layer is that it carries more resolved TKE
than the boundary layer under it. The corpus input `h` is 5% of that layer's peak
(`h_estimator = tke_peak_fraction`); the seed gate uses a fixed 0.01 m²/s² threshold instead, and
the two differ by 7–21%. `bin/test_bl_depth.py` re-derives `h` for all 47 stored profiles and
requires exact equality (no physics between the two, only arithmetic; 47 of 47), and gives 448 m
on the neutral profile whose wave layer once produced 2372 m (traps §22).

`window_stats` (the same module) produces every corpus input over the footprint's own window:
`h`, `u*`, `σ_v`, `L` (from the per-cell flux, never from a mean of `invOblen`: the mean of a
ratio whose denominator is `u*³` was wrong by 148×, traps §19e), the mean wind and its
direction. It is an accumulator (`WindowAccumulator`) so the streamed hand-off and the disk
path run one implementation.

## The hand-off

Under `lpdmOnlineSelector = 1` FastEddy stages each snapshot of the window to host RAM
(`io_lpdmonline.c`, patch 0005) and `lpdm/ringsrc.py` consumes it, deleting each after reading,
which releases the producer's backpressure. The two paths agree to 0.00e+00 on every field and
every `window_stats` scalar, asserted rather than toleranced. Persisted per case: 3.6 MB against
19 GB of window dumps. Host residency floors at the 12.0 GB field cache because the CPU
integrator random-accesses all of it. Details and the eight bugs it cost: traps §20–21 and the
[eighth](../history/pass-8.md) and [ninth](../history/pass-9.md) passes.

## The GPU LPDM

`lpdm/cuda/cuda_lpdmDevice.cu`, built into `lib/liblpdm.so` and driven by `lpdm/gpu.py`: a VRAM
ring of `t_back` history at fp16, the backward ensemble integrated in-kernel with fp64 particle
state, the whole production closure transliterated. Accepted against the CPU path on a 900 s
convective window (`results/gpu_lpdm_acceptance.json`): footprint within the CPU path's own
half-vs-half floor, backward well-mixed lowest three 0.999 vs 0.995, negative lobe 7.7% vs 9.0%,
0.06 s against 9.3 s (153×). It is not the production integrator ([limitations](../limitations-and-future-work.md)).

## Kljun beside it

`lpdm/kljun_ffp.py` evaluates Natascha Kljun's own FFP v1.42 (`third_party/FFP/`, unmodified)
on the raster's cell edges and agrees with the code it wraps to 9.4e-16. It is the input channel
and the baseline. `lpdm/kljun.py`, the project's earlier reimplementation, survives for the gates
validated against it and is known to be 1.25× wide in `σ_y` at `|L| > 5000`
([ninth pass](../history/pass-9.md)).
