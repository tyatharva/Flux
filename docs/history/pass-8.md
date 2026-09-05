# Eighth pass: 122³ at 30 m, the in-process hand-off, two footprints per case

2026-08-30. The grid the corpus was generated on, and the hand-off that let it be generated
on rented machines without a filesystem.

## The grid

122 × 122 × 122 at `dx = dy = 30 m`, domain 3660 m. The vertical grid is identical to the
24 m solution (`d_zeta` 24.691358, `verticalDeformFactor` 0.346601, `dz_sfc` 8.5583 m,
`k = 3` at exactly 30.000 m), so only the horizontal spacing changed: `Δ` 19.78 m, `z/Δ` 1.52,
`dx/dz_sfc` 3.505.

| | 24 m (retired) | **30 m** |
|---|---|---|
| domain | 2928 m | 3660 m |
| cost, measured | 0.481 GPU-h/sim-h | 0.479 at the production `dt` (14.781 ms/step) |
| geometric-mean `z0` | 0.0832 m | 0.0615 m |
| water in the box | 8.78% | 13.61% (2026 cells) |
| array in the box | 0.50% | 0.30% (44 cells) |
| taper knee | pad 10 | pad 12, real geography to 1470 m |

**Why 3660 m and not 186² at 24 m.** The [containment gate](containment.md) failed for
neutral at 2928 m: the flat control's integral needed 1.5 domain lengths to stop growing and
the cap removed 6.1%. 186² at 24 m buys full containment at +132% (820 → 2150 GPU-h); 122³ at
30 m buys 25% more box for 3%. A relative claim against Kljun does not need full containment,
and the parity number is what made that defensible: LES 0.874 of its asymptote against Kljun's
0.867 on identical cells at 2928 m. The acceptance became "the neutral integral saturates by
2.5 L", not "it is complete". (The ninth pass then found the parity does not survive the move.)

**The flat `dt` accuracy boundary**, from a ladder branched off a developed state
(`results/g30_bringup.txt`): `k0/k1` 0.130 at CFL 1.30, 1.40, 1.45 and 1.50; **8.857 at 1.55**;
8.433, 8.078, 7.591 above. A factor of 68 across 0.05 of CFL, with `turb_alive` OK at every rung.
And it does not interpolate with anisotropy (16 m at `dx/dz` 4.007 gave about 1.51; 24 m at 2.804
gave 1.55–1.60; 30 m at 3.505 gives 1.50–1.55). Production `dt = 5/162 s = 0.0308642`, CFL
1.3502, 10.0% below the last clean rung, landing the 5 s cadence, the 300 s seed cadence, a
2.0 sim-h case (233,280 steps) and a 3.0 sim-h seed on integer step counts.

## Gate A1 fails, and it is the site

Worst case over all directions and stabilities 25.93% water (17.45% at 2928 m); over corpus
regimes 11.58% (neutral easterly; 7.38% before). Kljun's neutral `x90` moved only 1615 → 1665 m,
so the physical footprint did not change; what changed is that a 3660 m box holds it where a
2928 m box replaced the lake between 1464 and 1830 m east with a periodic re-sample of its own
land. The 2928 m PASS was truncation. Recorded as a site limitation.

## Coarsening costs less than `z/Δ` suggests

`bin/subgrid_apriori.py` (retired; the table is the record), with no GPU: the 2-D spectrum of `w`
at the receptor level of existing windows, split at each candidate grid's cutoff. Going from
24 to 30 m keeps 99.7% of resolved variance at a 2dx filter and 87.2% at 4dx, so the sub-grid
fraction at the receptor moves 52.5% → about 56% convective and 86.4% → about 87% neutral. A
lower bound on the degradation, and a prediction that the ninth pass re-measured (59.3% / 57.7%
convective).

## The LES hands its fields to the LPDM in RAM

`io_lpdmonline.c` (patch 0005) + `lpdm/ringsrc.py` + `lpdm/dumpsrc.py`, behind
`lpdmOnlineSelector` (default 0). About 20 GB of window scratch per case became about 3 MB;
over the corpus, about 14 TB → 4 GB. That was the deployment argument: IO was about 3% of
compute, never a speed problem, but a rented GPU's scratch performance and quota were the one
thing the plan could not measure in advance.

Two design points where the obvious reading was wrong:

- **The ring holds a full window (541 slots, 6 fields, 12.0 GB), not `t_back` (180 slots,
  4.0 GB)**, because the `σ_w` floor is built from whole-window statistics, and a shorter ring
  would force the floor onto partial-window statistics: an estimator change wearing a plumbing
  change's clothes. A slot is 122 × 123 × 123 wrap-padded, 3.691 MB per field at fp16.
- **The route into the ring is host memory over tmpfs, not CUDA IPC.** The push entry point takes
  host pointers by design, the hop costs about 4 ms of PCIe per 5 s of model time against 2.5 s of
  compute, and IPC would put a new packing kernel on a validated integrator's critical path.

| test | result |
|---|---|
| `bin/test_dumpsrc.py`, the reader indirection | PASS, bit-identical |
| `bin/test_ringsrc.py`, the consumer | PASS, bit-identical against an fp32 reference |
| `stage5 --ring` end to end on an identical 60-snapshot window | PASS, 0.00e+00 on integral, asymptote, wrapped fraction and every `window_stats` field |
| `bin/test_lpdmonline.py`, producer against consumer on a real LES | PASS, exact: 23 snapshots, 8 fields, max \|diff\| 0.000e+00 |

Bit-identity rather than a tolerance, because there is no physics between the two paths. The
producer↔consumer test is the half no CPU test could reach: one LES at `lpdmOnlineSelector = 2`
stages and writes from the same buffer at the same point in the writer, so the two artifacts are
comparable without differencing two turbulence realisations (44% apart in the integral). It
excludes a wrong field order, a transpose, a pre-rho-division snapshot, an off-by-one slot, a
2-D field read with 3-D extents, and a truncated write.

Five real bugs found by building it: `window_stats` mixing two surface-flux estimators inside
one window (traps §20c); a 504-character `.in` comment segfaulting the parser (§20f); `0` as
the disabled sentinel for a step (§20g); the consumer's drain loop as an unbounded accumulator
(§20h); and `window_stats` parsing the timestep with its own `rsplit` inside a `try/except`, so
a non-path handle fell through to the dump index and the time axis spanned 9 units instead of
82,134, caught by `test_dumpsrc.py` on its first run.

## Two footprints per case

1800 s adjustment plus two 2700 s windows in one invocation, window 2's releases beginning
`t_back` = 900 s after window 1's end so the field intervals are disjoint: 0.99 GPU-h per case,
0.50 per footprint against 0.60 for one window at 24 m. The motive: re-running an identical
case gave integral 1.463 → 1.019 and array share 5.65% → 1.07%, so every floor quoted so far
was a within-realisation floor and too small. Whether the second window was worth it was left
to `bin/window_independence.py`, which asks two ways that fail differently. The split rule
tightened: `<case>_w0` and `<case>_w1` share everything upstream of the window, so `split_key`
is the parent. (The ninth pass measured the two windows as near-duplicates in shape, and
`N_WINDOWS` went back to 1.)

## Still owed at the end of this pass

Gate D1 through the ring in both directions and both regimes; a footprint from a live LES rather
than a replayed window; the containment acceptance; the deciding test re-run at this grid; rung
re-spacing; `zCeiling`. All but the last two were closed by the [ninth pass](pass-9.md).

Removed from the tree on 2026-09-04 (at the `pre-cleanup-2026-09-04` tag): `bin/run_pass8.sh`,
`bin/subgrid_apriori.py`, `results/pass8_deciding_test.txt`, `results/subgrid_apriori_30m.txt`
(kept: `results/g30_bringup.txt`, the `runs/s30_cfl*` ladder, `results/g30_flat*`).
