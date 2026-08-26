#!/usr/bin/env python3
"""Generate the 18 portable seed-spin-up jobs. Stage 3 of the corpus pipeline.

WHAT A SEED IS, AND WHAT IT IS NOT. A seed is pre-spun flat, uniform, doubly-periodic
turbulence. It is NOT a corpus point and is never trained on. Its only job is to delete the
3 simulated hours of spin-up that a cold-started case would need: a case restarts from the
nearest seed, adjusts for 30 min under its OWN sounding's forcing, then samples 30 min.

    1825 cases x 1.07 sim-h                 ~1990 GPU-h
    18 seeds  x 3.0  sim-h                    ~55 GPU-h
    the same corpus cold-started            ~7574 GPU-h

55 GPU-h buys back about 5500. The library is rounding error beside the corpus.

=== WHY THESE AXES ===

Sized by what 30 MINUTES CANNOT ADJUST, which is the only criterion that matters when the
seed exists solely to be adjusted away:

  direction        NO  -- the mean flow backs at -5.4 deg/h (measured, g16_spin), so 30 min
                          closes 2.7 deg of a gap that can be 45.        AXIS: 3 base angles
  z_i              NO  -- entrainment runs +79 m/h (measured, g16_cbl_shallow), so 30 min
                          closes 40 m of a gap that can be 800.          AXIS: 6 real levels
  stability regime NO  -- a CBL needs ~8 T* ~ 1.2 h to turn over.        AXIS: in the rungs
  u* / wind speed  PARTLY -- but the receptor is at 10 m and the surface layer is ~0.1 z_i
                          deep, so it re-equilibrates in ~2 min.         no axis
  fine z/L         YES -- the surface flux is prescribed and the surface layer follows.  no axis

=== A STABLE RUNG'S DEPTH IS EMERGENT, NOT SET -- AND THE FIRST SPEC WAS INCONSISTENT ===

Every other rung sets z_i with a CAPPING INVERSION, which is a direct control: put
`zStableBottom` at the target and the boundary layer stops there (measured: 154 m against a
150 m target, 299 against 300). A STABLE rung has no such control. Its depth is an
equilibrium of the forcing,

    z_i = 0.4 sqrt(u* L / f)          (Zilitinkevich)

so the depth, the wind and the cooling cannot be chosen independently -- one of them has to
be solved for. The first spec chose all three (z_i = 150 m, G = 6 m/s, w'th' = -0.020) and
they were mutually inconsistent: at G = 6 the neutral warm-up settles to u* ~ 0.17, which
gives L ~ 20 m and **z_i ~ 76 m**, half the target -- and puts the 10 m receptor at
z/z_i = 0.13, OUTSIDE the surface layer and inside the band the corpus filter itself
rejects as too shallow.

Solved instead of chosen: a 150 m stable layer needs u* = 0.218 at w'th' = -0.012, and
u*/G ~ 0.027 in stable stratification puts G at **8 m/s**. That is GABLS1 (Beare et al.
2006: G = 8 m/s, cooling 0.25 K/h ~ -0.012 K m/s, observed z_i 175-200 m), which is the
canonical stable-LES benchmark and a reassuring place to land. The formula reproduces
GABLS1's own depth to 186 m against an observed 175-200.

-0.012 is also the more representative flux: CONUS404 at this tower puts w'th' p25 at
-0.006 and p5 at -0.027, so the original -0.020 sat near the 10th percentile -- an
unusually strong cooling night -- while -0.012 is an ordinary one.

=== AND THE STABLE RUNG IS RESTRICTED TO WEAK STABILITY, BY MEASUREMENT ===

The spec above (G = 8, w'th' = -0.012, z_i 150 m) IS GABLS1's regime, and at dx = 16 m it
does not survive. It ran healthy for 1.75 simulated hours -- u* 0.20-0.24, z/L 0.12-0.21,
Ri_g 0.03-0.05, a proper Ekman profile -- and then collapsed to u* 0.098 and z/L +2.67.

The cause is RESOLUTION and it was measured at the HEALTHY dump, not inferred from the
collapse: the Ozmidov scale L_O = sqrt(eps/N^3) -- the largest eddy stratification permits
to overturn -- was only 1.0-3.2 x Delta through the layer, and the resolved fraction of
sigma_w^2 at the receptor was 0.6% at 6 m and 4.0% at 14 m against 16-56% convectively.
GABLS1 (Beare et al. 2006) runs that regime at dx = 6.25 m: 2.5x finer, 16x the cells.

MOVING THE RUNG TO THE WEAKLY STABLE END WAS TRIED, AND IT FAILED. The reasoning below is
kept because it is why the attempt was worth making and because the measurements in it
stand; the OUTCOME is that the stable rung is deleted. Read to the end of this section.
Measured over three independent sources (bin/stable_fraction.py, results/stable_fraction.txt)
the median stable hour at this tower sits at z/L = 0.056 (the tower's own H and sigma_w),
0.063 (HRRR) and 0.071 (CONUS404), and 61-66% of QC'd stable hours are at z/L <= 0.10. The
rung is placed at that median, not at the edge of the band, so a PASS licenses the band and
a FAIL is unambiguous.

    G = 10 m/s, w'th' = -0.012 K m/s  ->  u* ~ 0.30, z/L(10 m) ~ 0.06

and the flux is left at -0.012 rather than weakened. RAISING G IS THE BETTER KNOB, and the
reason is the failure mode: z/L falls as u*^-3 while eps rises as u*^3, so more wind buys
weaker stratification AND a larger Ozmidov scale at the same time, where a weaker flux buys
only the first. It also deepens the layer, since z_i = 0.4 sqrt(u* L/f) and L ~ u*^3 make
z_i ~ u*^2:

    z_i = 0.4 sqrt(0.30 * 162 / 9.94e-5) = 280 m      ->   z_i/Delta = 27.7

against GABLS1's own 180 m / 6.25 m = 28.8. THE LAYER IS RESOLVED IN THE SAME RELATIVE
SENSE THE CANONICAL STABLE BENCHMARK IS. The collapsed 150 m rung had z_i/Delta = 14.9,
half of it. That ratio, not the absolute spacing, is what the grid has to buy.

=== AND THE MEASURED OUTCOME: IT COLLAPSED ANYWAY, ON THE SAME TIMELINE ===

seed_sbl-weak_a030, 3.0 simulated hours, one neutral warm-up segment, z/L 0.044 at the
receptor -- less than half the stratification of the run that collapsed before it:

  t_h     u*     z/L   backing   |U-G| aloft   Ri_g@20m   dth/dz@2m   zTKE95
  0.75  0.2794  0.074     8 deg      0.0001      -0.000        -0.0      92 m
  1.50  0.3334  0.044     7 deg      0.3440       0.012         7.1     559 m
  3.00  0.1848  0.253    21 deg      0.4670       0.043        12.4    1825 m

u* ends at 40% of its own peak and still falling at -40 %/h; resolved TKE is at 5% of its
peak. All seven stationarity limits fail.

**And it is NOT the cold-start failure this file's warm-up was written to fix.** Ri_g
peaked at 0.043 against a critical 0.25; the Ekman backing was normal and INCREASING; the
inversion reached an ordinary 12 K/km rather than thousands; and the flow aloft DEPARTED
from geostrophic instead of pinning to it. Every one of those says the surface layer was
healthy. What moved was WHERE the energy sat: the height holding 95% of the column TKE ran
92 m -> 1825 m. The turbulence was not destroyed by stratification at the surface; it
failed to be RESOLVED there. bin/sbl_diagnose.py scores both signatures and this is the
starvation one.

So halving z/L bought a slightly slower death and nothing else, which is what the Ozmidov
measurement predicted: 6.88 Delta at the receptor at z/L 0.044, against 3.57 at 0.12 and
318 neutrally -- better, and still an order of magnitude short of a resolved band.

**THE STABLE RUNG IS THEREFORE DELETED. The corpus contains no stable cases.** The library
is 5 rungs x 3 base angles = 15 seeds. bin/select_times.py defaults to --max-zol 0.0.
What it costs, measured: 44% of QC'd hours are stable, but only 5.4 points of DAY coverage
(80.4% -> 75.0%), because enumeration finds a neutral or unstable hour on almost every day.
The loss is a REGIME, not a sample size -- the emulator is undefined in stable conditions
and must not be extrapolated into them. 26% of retained cases still fall outside 06-18 LST.

=== AND A STABLE RUNG CANNOT BE COLD-STARTED EITHER, WHICH WAS MEASURED THE HARD WAY ===

`sbl` at G = 6 m/s with w'th' = -0.020 K m/s was cold-started for 1.25 simulated hours and
COLLAPSED. Measured at the end: u* fell 0.219 -> 0.043 m/s, z_i 209 -> 61 m, the receptor
reached **z/L = +34.8**, dtheta/dz at the first level hit **2551 K/km**, and above 66 m the
mean wind sat at EXACTLY the geostrophic 6.000 m/s with a gradient Richardson number of
~1e8 -- the boundary layer had decoupled from the flow entirely.

The forcing is not the fault. With turbulence already present, u* ~ 0.30 gives L ~ 100 m
and z/L ~ 0.10 at the receptor, which is weakly stable and perfectly sustainable. **The
COLD START is the fault**: at t = 0 there is no turbulence, so the prescribed cooling
builds a near-discontinuous surface inversion before any can develop, and the resulting
stratification then prevents it developing at all. Runaway surface cooling is a documented
LES failure mode and it is why GABLS1 prescribes a cooling RATE rather than a flux.

The fix keeps the flux boundary condition -- the corpus's whole surface treatment is built
on per-cell `htFlux` -- and removes the cold start instead: **a stable rung runs its first
segment NEUTRAL, so real turbulence exists before the cooling is switched on.** That is
also how a stable boundary layer forms in nature, out of the evening transition of a
neutral or convective one, so the seed is more physical rather than less. `warmup_segments`
carries it, it is 0 for every other rung, and the job stays self-contained -- no
cross-rung dependency and nothing shared between jobs.

THE RUNGS ARE COUPLED, not a product. A 150 m stable boundary layer cannot carry a 12 m/s
geostrophic wind -- shear that strong destroys the stratification that defines it -- so G
belongs to the rung rather than to an axis of its own. Six rungs walk the site's real joint
(z_i, flux, wind) distribution as CONUS404 measures it at this tower.

THREE BASE ANGLES, NOT FOUR. A square doubly-periodic flat uniform domain with dx = dy is
exactly equivariant under 90-degree rotation (Gate B6: exact to 1.2e-14), so each base
angle re-indexes into 4 directions and {0, 30, 60} covers the compass on a 30 deg grid.
Under a SMOOTH reconstruction -- which is what a CNF learns, not a piecewise-linear
interpolant -- {0,30,60} reconstructs the Kljun array-share-vs-bearing curve to 0.80
points, about 4x below the 3.03-point LES sampling standard error; an uneven four-angle set
scores WORSE (1.36) because it makes the spline ring.

=== PORTABILITY ===

Each job is one directory: one .in, one manifest, one entrypoint, no absolute paths, no
shared state. It needs a checkout of this repo (for the container image and the gate) and
an sm_89 GPU with ~1.6 GB free. It returns the final restart dump, the stationarity verdict
as JSON, and the log -- the ~36 intermediate 300 s dumps never leave the rented machine.

BITWISE REPRODUCIBILITY WILL NOT HOLD ACROSS DIFFERENT PHYSICAL GPUs. Stated, not fixed:
FastEddy is already non-reproducible run-to-run on ONE GPU (~1e-4 relative in velocity,
~7e-4 K in theta after 200 steps -- PROJECT_BRIEF.md). Seeds are turbulence realisations, so this
costs nothing; it is recorded so nobody later diffs two seeds expecting equality.

usage: make_seed_jobs.py [--outdir jobs] [--template runs/g16_base/base.in] [--sim-h 3.0]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from sounding_to_forcing import derive_dt, les_levels, write_in

# name, regime, z_i target (m), virtual w'th' (K m/s), G (m/s)
#
# Anchored on CONUS404 at the tower (39,456 hourly records, PROJECT_BRIEF.md): z_i p25/p50/p75 =
# 267/493/835 m, w'th' p25/p50/p75 = -0.006/+0.015/+0.076, U(30 m) p25/p50/p75 =
# 3.9/5.2/6.8 m/s, and the site is unstable more than half the time. The convective rungs
# straddle the convective-midday reference (z_i p50 859 m, w'th' p50 0.109 sensible ->
# 0.129 virtual at the cropland Bowen ratio).
# name, regime, z_i target (m), virtual w'th' (K m/s), G (m/s), warm-up segments
# THE STABLE RUNG IS GONE, BY MEASUREMENT. Two seeds were run and both collapsed:
#   sbl      G=8,  w'th'=-0.012 (GABLS1's own regime), z/L ~ 0.12-0.21 -> dead by 2.3 h
#   sbl-weak G=10, w'th'=-0.012, z/L ~ 0.044          -> dead by 3.0 h, same timeline
# Halving the stratification bought nothing, and the second run's diagnosis says why:
# Ri_g peaked at 0.043 against a critical 0.25, Ekman backing was normal and INCREASING,
# the inversion was an ordinary 12 K/km and the flow aloft was DEPARTING from geostrophic.
# The surface layer was healthy the whole way down. What failed is resolution -- the
# Ozmidov scale is 6.9 Delta at the receptor even at z/L 0.044, against 318 neutrally --
# so no forcing change reaches it. See STABLE_REGIME_RESULT.md.
RUNGS = [
    ("nbl-shallow", "neutral",    300.0,  0.000,  8.0, 0),
    ("nbl-deep",    "neutral",    550.0,  0.000, 12.0, 0),
    ("cbl-shallow", "convective", 450.0,  0.060,  7.0, 0),
    ("cbl-mid",     "convective", 700.0,  0.110,  9.0, 0),
    ("cbl-deep",    "convective", 950.0,  0.160, 11.0, 0),
]
BASE_ANGLES = (0.0, 30.0, 60.0)

CAP_GRADIENT = 0.08      # K/m across the capping inversion -- the CONTROL on z_i
CAP_DEPTH = 100.0        # m
FREE_LAPSE = 0.004       # K/m, the free-atmosphere lapse above the cap
SBL_GRADIENT = 0.010     # K/m, a typical nocturnal surface inversion
THETA_GRND = 300.0
PRES_GRND = 97700.0      # Pa, the site's mean surface pressure (HRRR)
Z0 = 0.1435              # geometric-mean z0 of the real map; the flat spin-up's scalar
SPS = 0.0149             # measured s/step at 122^3 with a spin-up IO cadence
WALLCAP = 3600.0
MARGIN = 0.93
CADENCE_SPINUP = 300.0   # s between stationarity dumps


def base_state(regime, zi):
    """The four stabilityScheme = 2 segments for a rung.

    hydro_core.c:1776-1810: the LOWEST segment has no free gradient, it is exactly
    theta_grnd. For neutral and convective rungs that is the mixed layer and
    zStableBottom IS the target z_i. A stable rung has no neutral layer to give it, so
    zStableBottom goes to 0 and the inversion starts at the ground.

    All three gradients must be strictly positive (queried over [FLT_MIN, FLT_MAX]) --
    and an out-of-range value does NOT stop the run, it silently reverts to the
    compiled-in 0.1 K/m default. Every value here is a literal chosen inside the range.
    """
    if regime == "stable":
        # NEUTRAL TO THE TARGET DEPTH, THEN STRATIFIED -- the same shape as every other
        # rung, and GABLS1's own initial profile (neutral to 100 m, 0.01 K/m above).
        #
        # The first version stratified from z = 0 on the reasoning that "a stable rung has
        # no neutral layer to give it". That confuses the INITIAL condition with the final
        # state. A stable boundary layer forms by cooling a neutral or residual layer from
        # below; it does not appear ready-made, so there has to be something to cool INTO.
        #
        # CORRECTION, same day: this change was originally justified by a z_i that fell
        # 154 -> 76 m under neutral forcing. That fall was a DIAGNOSTIC ARTIFACT -- z_i as
        # "5% of the peak TKE" shrinks when the surface peak grows, and it grew 25x while
        # the TKE at 150 m grew 8x (FASTEDDY_TRAPS.md 16). The layer was deepening. The
        # change is kept anyway because a neutral-below-stratified-above initial profile is
        # GABLS1's own shape and the right initial condition for a stable rung, but it was
        # not the fix it was claimed to be. The fix was the forcing.
        return dict(zStableBottom=round(zi, 1), stableGradient=SBL_GRADIENT,
                    zStableBottom2=round(max(3.0 * zi, 500.0), 1),
                    stableGradient2=FREE_LAPSE,
                    zStableBottom3=20000.0, stableGradient3=FREE_LAPSE)
    return dict(zStableBottom=round(zi, 1), stableGradient=CAP_GRADIENT,
                zStableBottom2=round(zi + CAP_DEPTH, 1), stableGradient2=FREE_LAPSE,
                zStableBottom3=20000.0, stableGradient3=FREE_LAPSE)


def plan_segments(sim_h, dt, frq):
    """Chain of segments, each a whole number of dumps and under the wall cap.

    The cap is a HARD rule (PLAN.md), so the driver projects before launching and refuses
    rather than asks. Rounding DOWN to whole dumps is what keeps the projection honest.
    """
    total = int(round(sim_h * 3600.0 / dt / frq)) * frq
    per = int(WALLCAP * MARGIN / SPS) // frq * frq
    if per <= 0:
        raise ValueError("a single dump already exceeds the wall cap")
    nseg = int(np.ceil(total / per))
    seg = int(np.ceil(total / nseg / frq)) * frq
    return total, nseg, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="jobs")
    ap.add_argument("--template", default="runs/g16_base/base.in")
    ap.add_argument("--sim-h", type=float, default=3.0)
    ap.add_argument("--dx", type=float, default=16.0)
    ap.add_argument("--nz", type=int, default=122)
    ap.add_argument("--zceiling", type=float, default=2500.0)
    ap.add_argument("--deform", type=float, default=0.194059)
    ap.add_argument("--cfl", type=float, default=1.35)
    a = ap.parse_args()

    zc = les_levels(a.nz, a.zceiling, a.deform)
    dz_sfc = float(2.0 * zc[0])
    dt, frq5, cfl = derive_dt(a.dx, dz_sfc, a.cfl)
    # ROUND ONCE, HERE. The .in can only carry what fits in a text field, and the manifest
    # is what the stationarity gate uses to build its time axis -- so if the manifest keeps
    # full precision and the .in gets 8 digits, the gate is scoring a slightly different
    # clock than the run kept. It is 3 ms over 3 simulated hours and would never show up as
    # anything; it is fixed because "the two files agree" is worth more than the 3 ms.
    dt = float(f"{dt:.7g}")
    # the spin-up cadence, in steps; an integer because dt was chosen to make 5 s integral
    frq = int(round(CADENCE_SPINUP / dt))
    if abs(frq * dt - CADENCE_SPINUP) > 2e-4:
        raise ValueError(f"spin-up cadence {frq*dt} s is not {CADENCE_SPINUP}")
    total, nseg, seg = plan_segments(a.sim_h, dt, frq)

    os.makedirs(a.outdir, exist_ok=True)
    index = []
    print(f"dt {dt:.7f} s (CFL_3d {cfl:.4f}), dump every {frq} steps = "
          f"{frq*dt:.1f} s, {total} steps = {total*dt/3600:.2f} sim-h")
    print(f"{nseg} segment(s) of {seg} steps = {seg*SPS/60:.1f} min wall each "
          f"(cap {WALLCAP/60:.0f} min)")
    if seg * SPS > WALLCAP:
        raise SystemExit(f"FATAL: segment projects {seg*SPS/60:.1f} min, over the cap")
    print(f"\n{'job':<22}{'regime':>12}{'z_i tgt':>9}{'wth_v':>9}{'G':>7}"
          f"{'angle':>7}{'U_g':>9}{'V_g':>9}{'warm':>6}")

    for name, regime, zi, wth, gmag, warm in RUNGS:
        for ang in BASE_ANGLES:
            job = f"seed_{name}_a{int(ang):03d}"
            d = os.path.join(a.outdir, job)
            os.makedirs(os.path.join(d, "output"), exist_ok=True)
            ug = gmag * np.cos(np.radians(ang))
            vg = gmag * np.sin(np.radians(ang))
            bs = base_state(regime, zi)
            # subsidence opposes entrainment, so it is only meaningful where there is
            # entrainment. A neutral or stable rung is held by its cap alone.
            wsub = -25.0 if regime == "convective" else 0.0
            params = {
                "Description": f"seed {job}: flat uniform {regime} spin-up, "
                               f"z_i target {zi:.0f} m, base angle {ang:.0f} deg",
                "stabilityScheme": 2,
                "temp_grnd": round(THETA_GRND * (PRES_GRND / 1.0e5) ** (287.05 / 1004.5), 4),
                "pres_grnd": PRES_GRND,
                **{k: v for k, v in bs.items()},
                "U_g": round(float(ug), 6), "V_g": round(float(vg), 6),
                "z_Ug": 10000.0, "z_Vg": 10000.0, "Ug_grad": 0.0, "Vg_grad": 0.0,
                # The .in carries the TARGET flux; jobs/run_seed.sh forces it to 0 for
                # the first `warmup_segments` segments so turbulence exists before the
                # cooling starts. See the docstring: a cold-started stable rung collapses.
                "surflayer_wth": wth, "surflayer_z0": Z0,
                "surflayer_idealsine": 0,
                "coriolisLatitude": 42.957160,
                "thetaPerturbationSwitch": 1,
                "thetaHeight": round(min(300.0, 0.8 * zi), 1),
                "thetaAmplitude": 0.10 if regime == "stable" else 0.25,
                "dt": dt,
                "Nt": seg, "NtBatch": frq, "frqOutput": frq,
                "inPath": "", "inFile": "", "topoFile": "",
                "outPath": "./output/", "outFileBase": "FE_SEED",
                "lsf_w_surf": 0.0, "lsf_w_lev1": wsub, "lsf_w_lev2": 0.0,
                "lsf_w_zlev1": round(zi, 1),
                "lsf_w_zlev2": round(min(max(2.0 * zi, 1000.0), 2000.0), 1),
            }
            cmt = {
                "zStableBottom": (f"neutral to {zi:.0f} m -- the layer the cooling turns "
                                  f"into an SBL (GABLS1 shape)" if regime == 'stable'
                                  else f"mixed layer 0-{zi:.0f} m: the CONTROL on z_i"),
                "stableGradient": (f"ambient stratification above the residual layer"
                                   if regime == 'stable'
                                   else f"capping inversion, +{CAP_GRADIENT*CAP_DEPTH:.0f} K "
                                        f"across {CAP_DEPTH:.0f} m"),
                "zStableBottom2": ("top of the stratified layer" if regime == "stable"
                                   else "top of the capping inversion"),
                "stableGradient2": "free-atmosphere lapse",
                "zStableBottom3": "above the domain; the third segment is unused",
                "stableGradient3": "free-atmosphere lapse",
                "surflayer_wth": (f"prescribed VIRTUAL kinematic heat flux ({regime})"
                                  + (f"; the first {warm} segment(s) run NEUTRAL first -- a "
                                     f"cold-started stable rung collapses" if warm else "")),
                "U_g": f"G = {gmag:.1f} m/s at base angle {ang:.0f} deg",
                "V_g": f"geostrophic wind FROM {(270.0-ang)%360.0:.0f} deg",
                "dt": f"= {CADENCE_SPINUP:.0f}/{frq} s; CFL_3d = {cfl:.4f}",
                "Nt": f"one segment; the driver chains {nseg} of them to {total} steps",
                "NtBatch": f"one {frq*dt:.0f} s dump per batch",
                "frqOutput": f"{frq*dt:.0f} s between stationarity dumps",
                "lsf_w_lev1": ("no subsidence: this rung is held by its cap alone"
                               if wsub == 0.0 else
                               f"{wsub:.0f} m/h at z_i; opposes entrainment "
                               f"(the kernel divides by 3600)"),
                "lsf_w_zlev1": "the inversion, where subsidence must oppose entrainment",
                "thetaHeight": "seed the turbulence inside the mixed layer only",
                "thetaAmplitude": ("small: a strong perturbation would destroy the "
                                   "stratification that defines a stable BL"
                                   if regime == "stable" else "cold-start seeding"),
                "temp_grnd": f"theta_grnd = {THETA_GRND:.1f} K at {PRES_GRND:.0f} Pa",
            }
            write_in(a.template, params, os.path.join(d, "seed.in"), comments=cmt,
                     stamp=f"seed {job}")
            man = {
                "job": job, "rung": name, "regime": regime,
                "base_angle_deg": ang,
                "target": {"zi_m": zi, "wth_virtual": wth, "G": gmag,
                           "G_dir_from_deg": float((270.0 - ang) % 360.0)},
                "run": {"dt": dt, "CFL_3d": cfl, "frqOutput": frq,
                        "cadence_s": frq * dt, "steps_total": total,
                        "n_segments": nseg, "steps_per_segment": seg,
                        "sim_hours": total * dt / 3600.0,
                        "warmup_segments": warm,
                        "projected_wall_min_per_segment": round(seg * SPS / 60.0, 1),
                        "outFileBase": "FE_SEED"},
                "requires": {"compute_capability": "8.9", "vram_gb": 1.6,
                             "image": "flux-fasteddy:cuda118"},
                "returns": ["return/seed_restart.nc", "return/stationarity.json",
                            "return/seed.log", "return/manifest.json"],
                "reproducibility": "FastEddy is not bitwise reproducible run-to-run on one "
                                   "GPU (~1e-4 relative in velocity), and will not be "
                                   "across different physical GPUs either. Seeds are "
                                   "turbulence realisations; do not diff two of them.",
            }
            json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
            index.append(man)
            print(f"{job:<22}{regime:>12}{zi:>9.0f}{wth:>9.3f}{gmag:>7.1f}"
                  f"{ang:>7.0f}{ug:>9.3f}{vg:>9.3f}{warm:>6d}")

    ent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jobs",
                       "run_seed.sh")
    json.dump({"n_jobs": len(index), "sim_hours_each": total * dt / 3600.0,
               "gpu_hours_estimate": round(len(index) * total * dt / 3600.0 * 0.97, 1),
               "base_angles_deg": list(BASE_ANGLES),
               "directions_covered_deg": sorted(
                   float((270.0 - ang - 90.0 * r) % 360.0)
                   for ang in BASE_ANGLES for r in range(4)),
               "jobs": index}, open(os.path.join(a.outdir, "index.json"), "w"), indent=1)
    print(f"\n{len(index)} jobs in {a.outdir}/  "
          f"({len(index)*total*dt/3600*0.97:.0f} GPU-h estimated)")
    print(f"  directions covered: " + ", ".join(
        f"{d:.0f}" for d in sorted(float((270.0 - ang - 90.0 * r) % 360.0)
                                   for ang in BASE_ANGLES for r in range(4))))
    print(f"  entrypoint: {os.path.relpath(os.path.normpath(ent))}")
    print(f"  index:      {os.path.join(a.outdir, 'index.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
