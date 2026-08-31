# THE 30-SEED LIBRARY, SPUN ON 16x RTX 5090 — 2026-08-31

Thirty seeds, one command, one machine, **0.936 h wall and 13.24 GPU-h**. Every seed
returned complete and every acceptance-battery item passed on all thirty. The two things
worth reading are further down: the stationarity gate accepted 11 of 30 and **the split was
not measuring stationarity**, and the diagnostic it used is the fourth instance of a failure
class this project already has a standing rule about.

Artifacts: `vast-seeds/` as returned, mirrored byte-identical into `jobs30/<job>/return/`.
`results/seed_tke_rescore.{txt,json}` is the rescore. The image and the deployment
procedure are `DEPLOY.md`; the build is `PLAN.md`'s 2026-08-31 block.

---

## 1. Delivery — complete, and checked on the artifacts rather than on the exit status

| check | result |
|---|---|
| seeds returned | **30 of 30**, 376 files, 2.1 GB, **zero empty files** |
| `seed_restart.nc` | all present, all exactly **73,271,565 B**, **30 distinct md5 sums** |
| finiteness | `np.isfinite` on `u`, `v`, `w`, `theta` in all 30: **no NaN, no Inf** |
| `CORRUPTED` / `#NaN` / `#Inf` | **none in any run log** (the grep hits are `acceptance.txt`'s own section header, `CORRUPTED: 0`) |
| file structure | 18 convective x 12 files, 12 neutral x 13 — the extra one is `accel.log`, the Steinfeld accelerator leg. Consistent. |
| `achieved` block | present in **30 of 30** manifests, so `pick_seed` matches on measured state rather than falling back to targets |

Physics is sane and orders the way it must: `w_rms` runs **0.28–0.74** on the convective
rungs against **0.07–0.20** neutral, and within a rung the six base angles cluster tightly
(`U` spreads < 0.05 m/s), which is what a set of rotations of one forcing magnitude should
look like.

**The ceiling arithmetic held on the real path on Blackwell**: `run.ceiling_steps` =
**233,280 = 2.000 sim-h** in every manifest, against the job's 349,920-step design ceiling.
That is the `int()`-vs-`round()` defect from 2026-08-31 staying fixed on hardware it had
never run on.

## 2. The machine

16x RTX 5090 (`sm_120`, cc 12.0, 32,607 MiB each). Provenance stamped into the manifest:
flux `bb961bcfc77d`, FastEddy `0ce48d5dff06`, CUDA 13.0.1, gcc 11.4.0, `-std=c++17`,
OpenMPI 4.1.2, netCDF 4.8.1; FastEddy carrying SASS for `sm_75 … sm_120` and **no PTX**,
`liblpdm.so` carrying `compute_80`/`compute_120` PTX — the contrast `Dockerfile.blackwell`
asserts, reproduced on the target hardware.

**The work queue behaved as a work queue, measured from the recorded timeline**: 16 of 16
workers used, **14 of them took a second job**, peak concurrency exactly 16, queue drained,
zero failures. That is the property `bin/test_work_queue.sh` demonstrated with the LES
stubbed, now observed under 30 real seeds.

| | |
|---|---|
| peak VRAM | **904 MiB** attributed to the process, 913 MiB whole-device, of **32,607** — **2.8%** |
| host | peak container RSS **58.9 GiB**, summed FastEddy RSS **8.98 GiB** over 32 processes, machine 755 GiB, low-water `MemAvailable` 718 GiB |
| `k0/k1` | **0.124–0.144**, OK on all 30 — the Blackwell numerics check `PLAN.md` listed as owed |
| `turb_alive` | real OK on all 30, not a SKIP |
| Gate C2 restart | **bit-for-bit on all 30** |
| static rotation check | **PASS on all 30** |
| battery failure lines | **none, on any seed** |

### The cost number, and the boundary on what it licenses

**0.189 GPU-h per simulated hour under full 16-way load.** The LES leg is the same in both
regimes — 1365.8 s convective against 1361.9 s neutral — and the neutral rungs' higher
all-in figure of **0.268** is entirely the accelerator (+567.3 s = +0.158 GPU-h).

Against **0.469 GPU-h/sim-h measured single-GPU on the RTX 4080 from inside this same
image**, and 0.479 at Ada bring-up. So **16-way contention costs nothing measurable — the
number is 2.5x better, not worse**, which was the question that had to be answered before
renting again.

> **THIS IS A SEED NUMBER AND IT MUST NOT BE CARRIED TO THE CORPUS ESTIMATE.** A seed runs
> FastEddy and nothing else. A corpus case additionally runs the LPDM and the in-process
> ring hand-off, and the ring's host-side field cache — **12.0 GB per case**, which is not
> buildup but the window itself, random-accessed by a CPU integrator — has **never been
> exercised at 16-way**. Sixteen concurrent cases would ask for ~200 GB of host residency
> against this machine's 755 GiB, and the *contention* behaviour of that is unmeasured.
> The corpus estimate stays at its own measured ~0.49 GPU-h/sim-h until a real case runs
> through this path on rented hardware. `PROJECT_BRIEF.md`'s 12.45 GB figure is a CASE, not a
> seed, and the machine manifest says so in its own `host_memory.note`.

### The thread-block sweep changed its answer, and the answer is not established

`bin/threadblock_sweep.py` re-measured on the machine, as designed, and picked **`1x8x16`
at 0.00580 s/step** where Ada picks `1x2x64` (0.00590 here, **1.017x**).

**Do not read that as a Blackwell result.** FastEddy prints its timing to five decimals, so
**one quantum is 0.00010 s = 1.7% at this speed**. Three shapes tie exactly at 0.00580
(`1x8x16`, `1x2x32`, `1x8x8`), the top eight span two quanta, and the reported "repeat noise
0.00%" is quantisation rather than precision. The sweep's two-phase design exists precisely
to stop it choosing on noise — on Ada that fix reversed a call — and here it chose on the
last printed digit instead. **Worst case if the choice is wrong: 1.7%.** Recorded, not
fixed; the fix is a longer timing run or a finer timer, and it is not worth GPU time at
this size.

One thing that did carry: `tBx` is no longer the 17% penalty `PROJECT_BRIEF.md` records at 186^2 on
Ada. The best `tBx > 1` shape here, `2x2x32`, is **1.017x** the winner — inside the same
quantum.

## 3. The gate accepted 11 of 30, and the split was not measuring stationarity

All 19 refusals are stationarity **DRIFTING** verdicts. Not one is a run failure: every one
of the nineteen produced a complete, finite, battery-passing seed.

Drifting limits, counted per limit so a seed drifting in two appears twice: `TKE_BL/u*^2` **11**, `sigma_w/u*` 9, `sigma_v/u*` 4, `z_i` 3,
Kljun `x_peak` 2, `U/u*` 1.

**The drift is a rung-wide property, not realisation noise.** Mean `TKE_BL/u*^2` trend
against a 5 %/h limit, with the strict-gate acceptance beside it:

| rung | gated trend | accepted under the old refusal |
|---|---|---|
| cbl-deep | +3.5 %/h | 5 / 6 |
| cbl-mid | +12.5 | 2 / 6 |
| **cbl-shallow** | **+22.5** | **0 / 6** |
| nbl-deep | +19.4 | 3 / 6 |
| **nbl-shallow** | **+35.9** | **1 / 6** |

Every cbl-shallow seed is +18 to +25; every nbl-shallow seed is +28 to +44.

**AND THE ACCEPT/REFUSE SPLIT TRACKED THE STANDARD ERROR, NOT THE TREND.** The gate returns
INDETERMINATE when the threshold sits within 3 SE of the measurement, which is the correct
and deliberate behaviour — but the consequence at these drift magnitudes is that the
*better-measured* seed is the one refused:

| seed | trend | SE | n_eff | verdict |
|---|---|---|---|---|
| `cbl-shallow_a000` | **+23.5 %/h** | 7.37 | 3.0 | INDETERMINATE — **admitted** |
| `cbl-shallow_a030` | **+22.0 %/h** | 2.41 | 8.6 | DRIFTING — **refused** |
| `nbl-shallow_a000` | **+28.3 %/h** | 8.72 | 3.0 | INDETERMINATE — **admitted** |
| `nbl-shallow_a015` | **+30.3 %/h** | 2.62 | 4.4 | DRIFTING — **refused** |

So "11 accepted" reads as a quality statement and is not one. **The eleven are the seeds
whose drift could not be resolved at 3 SE**, and in cbl-shallow and nbl-shallow no seed is
anywhere near band.

## 4. `TKE_BL/u*^2` was measuring its own references — the fourth instance

`bin/seed_tke_rescore.py`, `results/seed_tke_rescore.txt`. **No GPU: this is arithmetic on
trends the seed runs already returned.**

`PROJECT_BRIEF.md` carries the standing rule that *a diagnostic whose denominator or reference
varies with anything other than the quantity being measured will report that variation as
signal*, with three recorded instances — `z_i` gated against a running TKE peak, `TKE`
averaged over the whole column, and `k0/k1` as a ratio of two levels that die together.
**`TKE_BL/u*^2` is the fix that was applied to the second of those**, and it is now the
fourth instance.

The absolute `TKE_BL` series was **not** returned — `stationarity.json` carries verdicts and
trends, not the per-dump series they were fitted to — but the numerator is recoverable from
what was, by two routes with **disjoint inputs**:

```
A.  trend(TKE_BL) = trend(TKE_BL/u*^2) + 2*trend(u*)      invert the ratio
B.  trend(TKE_BL) = trend(domain TKE)  -   trend(z_i)     de-normalise the column
```

A uses `u*`; B uses `domain TKE` and `z_i`. **Their agreement is the evidence**, and it is
also the check on the linearisation both perform.

| rung | GATED | u* | z_i | **ABSOLUTE** | \|A−B\| | reading |
|---|---|---|---|---|---|---|
| cbl-deep | +3.5 | −7.3 | +16.7 | **−13.5** | 4.9 | UNRESOLVED |
| cbl-mid | +12.5 | −8.8 | +6.4 | **−5.0** | 0.2 | STEADY |
| **cbl-shallow** | **+22.5** | −11.1 | −0.3 | **+0.5** | 0.5 | **STEADY** |
| nbl-deep | +19.4 | −2.4 | +4.0 | **+12.7** | 3.8 | UNRESOLVED |
| **nbl-shallow** | **+35.9** | −3.2 | +0.5 | **+29.4** | 0.1 | **RISING** |

**The gated ratio has three moving parts and only one is the turbulence.**

```
TKE_BL / u*^2  =  [ int_0^zi TKE dz / zi ]  /  u*^2
                    \_______  ______/          \__ falls through the first quarter of the
                            \/                     17.6 h inertial period -- PROJECT_BRIEF.md
                    the averaging DEPTH,           already records ~10 %/h -- and is not
                    which entrains upward          a statement about the turbulence
```

Moving the average inside the boundary layer removed the column mean's `z_i` dependence.
That fix was real. But it **exchanged one reference for two**: `u*^2` in the denominator,
and `z_i` again in the averaging depth. Every *other* gated limit is a ratio whose numerator
rides the oscillation *with* `u*` and cancels it — `U/u*`, `sigma_v/u*`, `sigma_w/u*`.
`TKE_BL` is an energy, not a velocity carried by the mean flow, so nothing cancels.

Two conclusions, and they point opposite ways:

- **The convective rungs were not drifting.** cbl-shallow's absolute BL TKE is **flat
  (+0.5 %/h)** while the gate reports +22.5; the whole reported drift is `u*` falling at
  −11.1 %/h. cbl-mid is **falling at −5.0** where the gate reports +12.5 *rising* — the
  wrong sign. cbl-deep is falling faster still (−13.5, both routes agreeing on the sign),
  which is its `z_i` growing at +16.7 %/h diluting a roughly fixed integrated TKE over a
  deepening layer — and **cbl-deep is the rung the old refusal accepted best, 5 of 6.**
- **The neutral rungs genuinely are still spinning up.** nbl-shallow's absolute BL TKE is
  **rising at +29.4 %/h** with both routes agreeing to 0.1, and `z_i` and `u*` are both
  quiet there, so the ratio *is* tracking the turbulence on those rungs. **A 2.0 sim-h
  ceiling is short for the neutral half**, and that is a real limitation of this library —
  not an artifact.

**WHAT IS OWED FROM FUTURE SEED RUNS, and it is small.** This recovery is first-order
arithmetic on trends, not a re-fit: the absolute trend cannot be given its own standard
error and no verdict here carries an `n_eff`. `stationarity.json` should return **the scored
series itself**, not only the verdicts and trends fitted to it. That is a few kB per seed
and it would have made this a measurement rather than a reconstruction.

## 5. Decision: seed selection uses the whole library

**Changed 2026-08-31.** `bin/pick_seed.py` ranks all 30. `--allow-drifting` defaults to
`any`, `--allow-indeterminate` to on, and **`--strict-gate` restores the old refusal**. The
corpus drivers (`bin/run_corpus.sh`, `bin/run_corpus_case.sh`, `bin/get_case.sh`) export
`ALLOW_DRIFTING=any`, superseding the 2026-08-30 `zi-neutral` narrow form.

**The reasoning generalises the one the `zi-neutral` concession was already granted on.** A
seed is an **initial condition**, not a corpus point: the case restarts from it, integrates
`ADJ_S` under its **own** sounding's forcing, and every ML input is then measured by
`window_stats` over **exactly the same window as the footprint**. The pair is
self-consistent whatever the seed's drift state — which is precisely what was argued for
`z_i`, and it never depended on the limit being `z_i`. Refusing a seed removes a **restart
point** without removing any error.

**What the narrow form cost, measured on this library:**

| | strict gate | whole library |
|---|---|---|
| seeds available | 11 of 30 | **30 of 30** |
| cbl-shallow | **0 of 6** — the weakly-convective rung had no restart point at all | 6 |
| neutral rungs | 4 of 12, i.e. **4 base angles** | **12** |
| Ekman backing calibration | n = 5 / 2 / 3 / 1, one rung absent | **n = 6 on every rung** |
| convective pick (`case_2023052519`) | `cbl-mid_a030`, cost **0.346**, `z_i` 766 vs 970 m | `cbl-deep_a030`, cost **0.268**, `z_i` 1011 vs 970 m |
| neutral pick (`case_2023112120`) | `nbl-deep_a000`, cost **0.983**, direction gap **14.5 deg**, and pick_seed's own **half-spacing warning fired** | `nbl-deep_a015`, cost **0.216**, gap **1.3 deg**, no warning |

The neutral case improves **4.6x in cost and 11x in direction gap**, and the guard that fires
when a base angle is missing stops firing — which is the library's own instrument agreeing
that the hole is closed.

**`gate_state` is still stamped on every pair** and `make_pair.py` still writes its warning
into the training record. Nothing here calls a seed stationary; it stops treating an
unestablished verdict as a disqualification. The per-seed notices are collapsed into **one**
policy line, because 30 identical paragraphs on each of ~1469 cases is not disclosure.

## 6. `run_seed.sh`'s exit status now answers the right question

It was `[ "$VERDICT" = "PASS" ] || exit 1`, and on this library it returned **1 for all
thirty** — no seed returned a clean PASS. A status identical for every outcome discriminates
nothing, and it does so in the dangerous direction: anything keying on it reads a complete,
usable, battery-passing seed as a failed run. `bin/run_seeds.py` survived only because it
judges on the artifact.

Now: **exit 0 when the run produced a seed**, exit 1 when it did not, and `SEED_STRICT_EXIT=1`
restores the old signal. The verdict is not lost — it is in `stationarity.json`,
`manifest.achieved.pass`, the machine manifest, and `seed.gate_state` on every pair. That is
`PROJECT_BRIEF.md`'s own rule — assert on the artifact, not the exit status — applied to this
script's output rather than to FastEddy's. All four branches exercised.

## 7. And `n_accepted` in the machine manifest was the count of NEUTRAL seeds

Found while re-deriving the table in §3. `bin/run_seeds.py` binds `acc` to the accepted
seeds, prints `SUMMARY 11 accepted` from it, and then **rebinds the same name** to the list
of accelerator GPU-h — which exists for exactly the 12 neutral seeds — before writing
`"n_accepted": len(acc)` into `machine_manifest.json`.

So the log said **11** and the JSON said **12**, and 12 is plausible enough that it was read
off the JSON and quoted before the log was checked. The true split is **11 accepted / 19
DRIFTING**; every per-seed `accepted` flag was correct throughout and nothing downstream
consumed the field.

Fixed by renaming the second binding to `accel_h` and by computing the field from the
records (`sum(1 for r in ok if r["accepted"])`) rather than from whichever list the name
happens to hold by then. It is the same shape as everything else in this file: **a number
came out, it was finite and plausible, and it was not the number on the label.**

## 8. What this leaves open

1. **The neutral rungs are short at 2.0 sim-h.** Absolute BL TKE rising at +29.4 %/h
   (nbl-shallow) and +12.7 (nbl-deep). A longer ceiling would help these and would *not*
   help the convective rungs, whose gated drift was the references. Costed: +1.0 sim-h on
   12 neutral seeds is **+2.3 GPU-h** at the measured rate. Not done; the library is in use.
2. **The gated form of `TKE_BL/u*^2` should be reconsidered**, now that it is known to be
   driven by `u*` and the averaging depth. Not changed here — changing a gate on the same
   pass that reinterprets its output is how a threshold gets tuned to a result.
3. **The scored series must be returned** from seed runs, so item 2 can be settled by a fit
   rather than by a first-order reconstruction.
4. **The corpus cost at 16-way is unmeasured**, and 0.189 does not answer it (§2).
5. **Nothing has run the ring or the LPDM on Blackwell.** The seeds exercise FastEddy only.
