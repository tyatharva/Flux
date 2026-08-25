#!/usr/bin/env python3
"""Per-case surface: the same static geography, this case's surface heat flux.

WHY THIS IS NEEDED, and it is not optional. bin/prep_restart.py injects `htFlux` into the
restart file from `<grid>/htFlux.npy`, because the restart read is the only way to give
FastEddy v5.0.1 a spatially varying surface (PROJECT_BRIEF.md, the Stage 6 lever). The retired
per-bin campaign could therefore build ONE grid directory per regime and reuse it.

**A sounding-forced corpus cannot.** Every one of ~1825 cases has its own surface heat
flux, and `data/grid16` ships with `htFlux.npy` ALL ZEROS -- it is a neutral build. Point a
convective case at it and the restart read silently overwrites the `.in`'s
`surflayer_wth` with a zero map: the case runs NEUTRAL, exits 0, and nothing anywhere says
so. That is the same mechanism as the Stage 6 trap, pointed at the flux instead of at the
terrain.

WHAT IT DOES. The per-cell map is `wth_reference * f`, where `f` is a fixed field of
land-cover class ratios that does not depend on the case at all. So the static arrays are
HARDLINKED from the source grid -- identical bytes, no copy -- and only `htFlux.npy` is
written fresh. A case directory is ~116 kB of real data.

=== THE THREE REGIMES ARE NOT THE SAME PROBLEM ===

  wth > 0   CONVECTIVE. The per-class map, exactly as prep_surface.py builds it. The
            ratios (array 1.60, water 0.12, built 1.50 ...) are DAYTIME sensible-flux
            enhancements from field studies, converted to virtual with each class's own
            Bowen ratio. This is the case the array signal lives in.

  wth ~ 0   NEUTRAL. Zero everywhere, which is what neutral means. The array's signal is
            then purely aerodynamic, and whether it exists at all depends on the grid:
            data/grid16 has z0_array = 0.10, which is EXACTLY WorldCover's cropland value,
            so the override changes nothing and the signal is zero. data/grid16_raised --
            which is production -- has z0_array = 0.25 against cropland's 0.10, a 2.5x
            contrast. Point a neutral or stable case at the baseline grid and the array
            disappears entirely.

  wth < 0   STABLE, and this one is a DECISION, not a formula. The class table is a
            DAYTIME table. At night the physics inverts -- water holds heat, built-up
            surfaces release what they stored, vegetation cools fastest -- and this
            project has no nocturnal equivalent of that table. Applying daytime
            enhancement ratios to a negative flux would invent a nocturnal contrast that
            nothing measured. **So a stable case gets a UNIFORM negative flux**, and the
            absence of nocturnal per-class contrast is recorded rather than fabricated.

            The consequence, stated because it is easy to get backwards: on the
            PRODUCTION surface a stable case is a ROUGHNESS-ONLY array case. The thermal
            contrast is gone but the 2.5x z0 contrast remains, so stable cases are the one
            place in the corpus where the array's aerodynamic effect is ISOLATED from its
            heat-flux effect -- which every convective case confounds. On the retired flat
            baseline (z0_array = 0.10 = cropland) there would be no array signal at all,
            which is a reason not to build a corpus on that grid.

usage: case_surface.py --grid data/grid16 --wth-ref 0.1290 --out data/grid16_case_<tag>
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sounding_to_forcing import _prep_surface_tables

NEUTRAL_EPS = 1e-4          # |w'th_v'| below this is neutral, K m/s
STATIC = ("topo.npy", "topo.bin", "z0m.npy", "lcclass.npy", "array.npy", "water.npy",
          "dmap.npy", "meta.npy")


def class_ratio_field(grid_dir):
    """`f`, the per-cell multiplier on the cropland reference. Case-independent.

    Built from prep_surface.py's OWN tables, read out of its source, so there is exactly
    one definition of the class ratios in this repo. Verified against the campaign's
    data/grid16_cbl, whose mean/cropland ratio is 1.0565.
    """
    ps = _prep_surface_tables()
    cls = np.load(os.path.join(grid_dir, "lcclass.npy"))
    array = np.load(os.path.join(grid_dir, "array.npy")) > 0.5
    bc = ps["WORLDCOVER_BOWEN"][40]
    vf = ps["virtual_factor"]
    fc = float(vf(bc))
    f = np.full(cls.shape, float(ps["WTH_FALLBACK"]))
    for k, sr in ps["WORLDCOVER_WTH"].items():
        b = bc if k == 40 else ps["WORLDCOVER_BOWEN"].get(k, ps["BOWEN_FALLBACK"])
        f[cls == k] = sr * float(vf(b)) / fc
    f[array] = ps["WTH_ARRAY"] * float(vf(ps["BOWEN_ARRAY"])) / fc
    return f, array


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="data/grid16", help="the STATIC source grid")
    ap.add_argument("--wth-ref", type=float, required=True,
                    help="the CROPLAND REFERENCE virtual flux (not the domain mean)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for n in STATIC:
        if not os.path.exists(os.path.join(a.grid, n)):
            print(f"FATAL: {a.grid} has no {n}", file=sys.stderr)
            return 2
    if os.path.exists(a.out) and not a.force:
        # Rebuilding is cheap; silently reusing a directory built for a DIFFERENT case's
        # flux is the failure this whole file exists to prevent.
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True)

    for n in STATIC:
        src, dst = os.path.join(a.grid, n), os.path.join(a.out, n)
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)               # identical bytes, no copy
        except OSError:
            shutil.copyfile(src, dst)

    f, array = class_ratio_field(a.grid)
    w = float(a.wth_ref)
    if w > NEUTRAL_EPS:
        wth = w * f
        regime, note = "convective", "per-class daytime ratios"
    elif w < -NEUTRAL_EPS:
        wth = np.full(f.shape, w)
        regime = "stable"
        note = ("UNIFORM: the class table is a daytime table and there is no nocturnal "
                "equivalent, so there is no THERMAL array contrast. The aerodynamic one "
                "survives via z0 -- check it below.")
    else:
        wth = np.zeros_like(f)
        regime, note = "neutral", "zero everywhere"
    np.save(os.path.join(a.out, "htFlux.npy"), wth.astype(np.float64))

    print(f"{a.out}")
    print(f"  static geography hardlinked from {a.grid} ({len(STATIC)} arrays)")
    print(f"  regime {regime}: {note}")
    print(f"  cropland reference {w:+.4f} K m/s -> map {wth.min():+.4f} .. "
          f"{wth.max():+.4f}, DOMAIN MEAN {wth.mean():+.4f}")
    if regime == "convective":
        print(f"  array cells {int(array.sum())}: {wth[array].mean():+.4f} K m/s "
              f"({wth[array].mean()/max(w,1e-12):.3f}x the cropland reference)")
    else:
        z0 = np.load(os.path.join(a.out, "z0m.npy"))
        cls = np.load(os.path.join(a.out, "lcclass.npy"))
        crop = (cls == 40) & ~array
        r = float(z0[array].mean() / max(z0[crop].mean(), 1e-12)) if crop.any() else float("nan")
        print(f"  array cells {int(array.sum())}: {wth[array].mean():+.4f} K m/s -- the "
              f"same as every other cell, so NO thermal array signal")
        print(f"  the array's remaining signal is aerodynamic: z0 {z0[array].mean():.3f} m "
              f"vs cropland {z0[crop].mean():.3f} m = {r:.2f}x"
              + ("  <-- 1.00x means NO array signal at all on this grid; use the raised one"
                 if r < 1.01 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
