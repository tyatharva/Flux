#!/usr/bin/env python3
"""Footprint from a saved LES window, on the STATIC north-up raster.

The raster is the LES grid itself -- 186 x 186 at 24 m, tower-centred, folded modulo the
periodic domain -- so a footprint cell IS an LES column. Nothing is rotated and nothing is
resampled. That matters three times over: the land-cover masks live on those indices, the
emulator will consume this array, and the near field (where the peak sits) is the part a
resample damages most.

The wind frame survives as a 1-D histogram of the touchdowns' upwind coordinate,
accumulated directly from the touchdowns. It is where "upwind distance" means anything and
where Kljun's crosswind-integrated form can be compared like for like.

WHAT KLJUN IS FOR. Over flat, uniform ground in neutral conditions FFP is valid and this
pipeline is not yet trusted, so Kljun is diagnostic there: disagreement is a bug in us.
Over real terrain and real land cover the roles reverse -- FFP has no way to represent a
lake or a roughness patch, so it is DESCRIPTIVE, a reference curve, not a target. Numbers
below are reported to the precision they deserve and never as an "error against Kljun".

usage: stage5_footprint.py <window_dir> [<window_dir_2>] [--dt ...] [--tag ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm import kljun
from lpdm.driver import compute_footprint
from lpdm.fields import FieldSet, dump_series, _step_of
from lpdm.footprint import source_area_overlap


def fy_metrics(xc, f, res):
    """Peak and first moment of a crosswind-integrated footprint."""
    if f.sum() <= 0:
        return dict(peak_x=np.nan, mean_x=np.nan, x80=np.nan, integral=0.0)
    sm = np.convolve(f, np.ones(5) / 5.0, mode="same")     # peak of a 5-cell mean: one
    peak = float(xc[int(np.argmax(sm))])                   # 24 m cell is Monte-Carlo noisy
    pos = np.maximum(f, 0.0)
    mean = float((pos * xc).sum() / pos.sum())
    c = np.cumsum(pos) / pos.sum()
    x80 = float(np.interp(0.80, c, xc))
    return dict(peak_x=peak, mean_x=mean, x80=x80, integral=float(f.sum() * res))


def describe(name, g, fy, xc, res):
    m = g.metrics_map()
    q = fy_metrics(xc, fy, res)
    print(f"  {name:<24} peak {q['peak_x']:6.0f} m   80% of f_y within {q['x80']:6.0f} m"
          f"   80% area {m['area80_ha']:6.1f} ha"
          f"   centroid {m['centroid_dist']:6.0f} m at {m['centroid_bearing']:5.1f} deg")
    return {**m, **q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--dt", type=float, default=0.0158228)
    ap.add_argument("--tback", type=float, default=900.0)
    ap.add_argument("--t-min", type=float, default=None,
                    help="refuse any field written before this model time, in seconds. "
                         "The adjustment and the sampling window are ONE FastEddy "
                         "invocation now (chaining is retired), so both write into the "
                         "same directory and dump_series() cannot tell them apart. "
                         "bin/run_window.sh DELETES the adjustment dumps and asserts on "
                         "what survived; this is the second, independent guard, because a "
                         "backward trajectory that walks into the adjustment is served "
                         "fields that were still settling and NOTHING in the output would "
                         "say so.")
    ap.add_argument("--z-target", type=float, default=10.0,
                    help="receptor height in m AGL. THE DEFAULT USED TO BE 30.0 AND WAS "
                         "NEVER PASSED FROM HERE, so every footprint silently landed on "
                         "the level nearest 30 m. The instrument is at 10 m.")
    ap.add_argument("--exact-agl", action="store_true",
                    help="release at a FRACTIONAL level exactly --z-target above the local "
                         "ground, instead of the nearest cell centre. Needed when the "
                         "surface was built with --raise-topo, which lifts the model "
                         "ground over the array by the displacement height while the "
                         "instrument stays 10 m above bare ground.")
    ap.add_argument("--nrel", type=int, default=700)
    ap.add_argument("--dtrel", type=float, default=4.0)
    ap.add_argument("--c0", type=float, default=3.0)
    ap.add_argument("--tag", default="stage5")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--rel-seconds", type=float, default=1800.0,
                    help="release period in seconds; 1800 = the 30-min EC averaging period")
    ap.add_argument("--strict-rel", action="store_true",
                    help="FAIL rather than warn if the window is too short to deliver the "
                         "full release period. A short window does not error on its own -- "
                         "it silently shortens the averaging period, and the production "
                         "4200 s case sits at EXACTLY zero margin, so one dump of "
                         "shortening would produce a 29.9-minute footprint compared "
                         "against 30-minute observations. bin/run_corpus_case.sh passes "
                         "this; the retired fourth-pass drivers, which legitimately "
                         "released for 900 s, do not.")
    ap.add_argument("--tback-marks", default="",
                    help="comma-separated shorter t_back values to also score, e.g. "
                         "300,450,600,750 -- free, and it is what sizes a window")
    ap.add_argument("--receptor-from", default="data/grid",
                    help="surface dir whose meta.npy pins the receptor to the tower cell")
    ap.add_argument("--sgs-most", action="store_true",
                    help="MOST-anchored sub-grid variance floor (see lpdm/driver.py)")
    ap.add_argument("--sgs-most-legacy", action="store_true",
                    help="reproduce the RETIRED 0.1h-0.2h factor taper. Production never "
                         "uses it; it exists so the bias it introduced can be measured "
                         "against the SAME LES fields the corrected floor runs on.")
    ap.add_argument("--no-subgrid-weight", action="store_true",
                    help="DIAGNOSTIC: drop the sub-grid-fraction weighting, restoring the "
                         "factor that reached 10x where the LES resolved 92%.")
    ap.add_argument("--no-eps-consistent", action="store_true",
                    help="DIAGNOSTIC: raise sigma^2 without raising eps, so T_L inflates "
                         "with the floor as it used to.")
    ap.add_argument("--sgs-most-form", default="multiplicative",
                    choices=("additive", "multiplicative"),
                    help="how the floor reaches the drift; additive is production.")
    ap.add_argument("--sgs-most-mode", default="surface",
                    choices=("surface", "blend", "mixed"),
                    help="which similarity relation the sigma_w floor is anchored to. "
                         "'surface' (default) is Panofsky et al. (1977), which carries "
                         "both shear and buoyancy production and reduces to 1.25 u* at "
                         "neutral. 'mixed' is Lenschow et al. (1980), a free-convection "
                         "relation with no neutral term. 'blend' takes the minimum. The "
                         "two asymptote to within 1% of each other in free convection but "
                         "differ by ~25% in the transition, which is where a 30 m "
                         "receptor in a 900 m CBL sits -- so this is a real modelling "
                         "freedom and it is quantified rather than assumed away.")
    ap.add_argument("--sgs-scale", type=float, default=1.0,
                    help="multiplier on the sub-grid VARIANCE (diagnostic lever)")
    ap.add_argument("--aniso", action="store_true",
                    help="surface-layer anisotropic sub-grid split instead of isotropic (2/3)e")
    ap.add_argument("--cover-dir", default=None,
                    help="surface dir (data/grid); adds land-cover attribution")
    ap.add_argument("--fp16-cache", action="store_true")
    ap.add_argument("--cover-groups", type=int, default=2,
                    help="split the release period into N independent groups and report "
                         "each group's land-cover share. Two halves give one difference "
                         "and one degree of freedom; N groups give a sampling "
                         "DISTRIBUTION, which is what a 1-point effect needs.")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    marks = tuple(float(v) for v in a.tback_marks.split(",") if v.strip())

    runs = []
    for d in a.dirs:
        paths = dump_series(d)
        if a.t_min is not None:
            keep = [p for p in paths if _step_of(p) * a.dt >= a.t_min - 0.5 * a.dt]
            if len(keep) != len(paths):
                print(f"  --t-min {a.t_min:.0f} s dropped {len(paths)-len(keep)} of "
                      f"{len(paths)} dumps written before the adjustment completed")
            if not keep:
                print(f"FATAL: --t-min {a.t_min:.0f} s leaves no fields in {d}",
                      file=sys.stderr)
                return 2
            paths = keep
        print(f"\n=== {d}: {len(paths)} dumps ===")
        t0 = time.time()
        fs = FieldSet(paths, a.dt, verbose=False,
                      cache_dtype=np.float16 if a.fp16_cache else np.float32)
        print(f"  cache {fs.mem_gb:.2f} GB, window {fs.t[0]:.0f}-{fs.t[-1]:.0f} s "
              f"(cadence {fs.dt_dump:.1f} s), loaded in {time.time()-t0:.0f} s")
        cover = None
        if a.cover_dir:
            z0 = np.load(os.path.join(a.cover_dir, "z0m.npy"))
            wm = np.load(os.path.join(a.cover_dir, "water.npy")) > 0.5
            ap_ = os.path.join(a.cover_dir, "array.npy")
            am = (np.load(ap_) > 0.5) if os.path.exists(ap_) else (z0 > 0.1)
            cover = {"solar array": am, "water": wm}
            lcp = os.path.join(a.cover_dir, "lcclass.npy")
            if os.path.exists(lcp):
                lc = np.load(lcp)
                for nm, k in (("tree", 10), ("grassland", 30), ("cropland", 40),
                              ("built", 50), ("wetland", 90)):
                    m_ = (lc == k) & (~am)
                    if m_.any():
                        cover[nm] = m_
        # Displacement height: FastEddy has no d, so it comes from the surface directory
        # that built the run. prep_surface.py writes zeros into dmap.npy when --raise-topo
        # already put the displacement surface into topoPos, so attaching it is always the
        # right thing and can never double-count.
        dsrc = a.cover_dir or a.receptor_from
        if dsrc:
            dp = os.path.join(dsrc, "dmap.npy")
            if os.path.exists(dp):
                dm = np.load(dp)
                fs.set_displacement(dm)
                print(f"  displacement height map from {dp}: "
                      f"{dm.min():.2f}-{dm.max():.2f} m, mean {dm.mean():.3f} m")
            else:
                print(f"  no dmap.npy in {dsrc} -- displacement height treated as ZERO")
        rij = None
        if a.receptor_from and os.path.exists(os.path.join(a.receptor_from, "meta.npy")):
            m_ = np.load(os.path.join(a.receptor_from, 'meta.npy'), allow_pickle=True).item()
            rij = (m_['itower'], m_['jtower'])
            print(f"  receptor pinned to the TOWER cell (i,j) = {rij}")
        from lpdm.model import SURFACE_LAYER_ANISO
        r = compute_footprint(fs, paths, z_target=a.z_target, exact_agl=a.exact_agl,
                              n_per_release=a.nrel, dt_release=a.dtrel,
                              t_back=a.tback, c0=a.c0, seed=len(runs), cover=cover,
                              aniso=SURFACE_LAYER_ANISO if a.aniso else None,
                              sgs_scale=a.sgs_scale, sgs_most=a.sgs_most,
                              tback_marks=marks, rel_seconds=a.rel_seconds,
                              require_rel_seconds=bool(a.strict_rel),
                              sgs_most_mode=a.sgs_most_mode, receptor_ij=rij,
                              sgs_most_legacy=a.sgs_most_legacy,
                              sgs_most_form=a.sgs_most_form,
                              sgs_subgrid_weight=not a.no_subgrid_weight,
                              sgs_eps_consistent=not a.no_eps_consistent,
                              n_cover_groups=a.cover_groups)
        if a.sgs_scale != 1.0:
            print("  SUB-GRID VARIANCE SCALED by %.3f (diagnostic)" % a.sgs_scale)
        if a.aniso:
            print("  SUB-GRID SPLIT: surface-layer anisotropic "
                  "(r_u, r_v, r_w) = (%.4f, %.4f, %.4f)" % SURFACE_LAYER_ANISO)
        print(f"  flux weight from touchdowns the periodic fold MOVED: "
              f"{100*r.get('wrapped_fraction', np.nan):.1f}%")
        if cover:
            print("  footprint-weighted land-cover share (from touchdowns, unblurred):")
            print(f"    {'class':<12} {'folded':>8} {'unwrapped':>10} {'area':>8}")
            for nm, v in r["cover_share"].items():
                vn = r.get("cover_share_nowrap", {}).get(nm, np.nan)
                print(f"    {nm:<12} {v*100:7.2f}% {vn*100:9.2f}% {cover[nm].mean()*100:7.2f}%")
            print("    (unwrapped excludes touchdowns whose periodic fold moved them onto "
                  "different\n     real geography; quote that column for the site)")
        st = r["stats"]
        print(f"  LES scalars at z={st['z_recept']:.2f} m: U={st['u_mean']:.2f} m/s "
              f"dir={st['wdir']:.1f} deg  u*={st['ustar']:.3f}  h={st['h']:.0f} m  "
              f"1/L={1/st['L']:.2e}")
        print(f"    sigma_v {st['sigma_v']:.3f} (resolved {st['sigma_v_resolved']:.3f} "
              f"+ sub-grid, e_sgs={st['e_sgs']:.4f});  sigma_w {st['sigma_w']:.3f} "
              f"(resolved {st['sigma_w_resolved']:.3f});  sigma_w/u* = "
              f"{st['sigma_w']/st['ustar']:.2f}")
        tc = r["grid"].tail_concentration()
        print(f"  touchdowns {tc['n_touchdown']:,};  largest single weight "
              f"{tc['max_weight']:.1f};  top 0.1% carry {tc['top0p1pct_share']*100:.1f}% "
              f"of the total |weight|")
        print(f"  integral of f_flux over the raster = {r['grid'].integral():.3f} "
              f"(all touchdowns {r['grid'].integral_all():.3f})")
        runs.append((d, fs, r))
        del fs.u, fs.v, fs.w, fs.e, fs.eps, fs.dsig2dz

    d0, fs0, r0 = runs[0]
    g0, st = r0["grid"], r0["stats"]
    # Kljun's z_m is an aerodynamic height, so it is z - d, not z. Over the flat control
    # d ~ 0.1 m and the two agree to 1%; over the array at 10 m the difference moves the
    # E/W footprint share by ~39% relative, which is far larger than any error floor here.
    zm_agl = r0.get("z_agl", st["z_recept"])
    zm = float(st.get("z_eff", zm_agl))
    if abs(zm - zm_agl) > 1e-6:
        print(f"\n  Kljun evaluated at the EFFECTIVE height z-d = {zm:.3f} m "
              f"(geometric {zm_agl:.3f} m, d = {st.get('d_recept', 0.0):.3f} m)")
    ang = np.radians(r0["wind_angle"])
    fyc, fy = r0["fy"]["xc"], r0["fy"]["f"]
    res = float(fs0.dx)

    # ---- Kljun, evaluated on the SAME static cells (no rotation, no resample) --------
    kl = kljun.footprint_on_static(g0.xe, g0.ye, ang, zm, st["h"], st["ustar"],
                                   st["sigma_v"], umean=st["u_mean"], L=st["L"])
    kl_fy, _ = kljun.crosswind_integrated(fyc, zm, st["h"], st["ustar"],
                                          umean=st["u_mean"], L=st["L"])
    kg = type(g0).from_edges(g0.xe, g0.ye)
    kg.flux = kl * kg.area
    kg.n_particles = 1

    print("\n=== flat-and-neutral reference check (Kljun is DIAGNOSTIC here, "
          "DESCRIPTIVE over real surface) ===")
    m_les = describe("LES + LPDM", g0, fy, fyc, res)
    m_kl = describe("Kljun FFP", kg, kl_fy, fyc, res)
    ov80 = source_area_overlap(np.maximum(g0.normalised("flux"), 0), np.maximum(kl, 0), 0.80)
    ov50 = source_area_overlap(np.maximum(g0.normalised("flux"), 0), np.maximum(kl, 0), 0.50)
    print(f"  source-area overlap vs Kljun:  80% {ov80*100:.0f}%   50% {ov50*100:.0f}%")
    print(f"  peak   LES {m_les['peak_x']:.0f} m   Kljun {m_kl['peak_x']:.0f} m")
    print(f"  integral over the raster:  LES {g0.integral():.2f}   Kljun {kl.sum()*g0.area:.2f}")
    if a.sgs_most:
        print("  NOTE: the near-field peak is CONSTRAINED -- the MOST-anchored sigma_w "
              "floor is\n        anchored to surface-layer similarity, the same theory "
              "Kljun rests on. The\n        80% area, the tail and the land-cover shares "
              "are free.")

    out = dict(dirs=a.dirs, zm=zm, zm_agl=zm_agl, d_recept=st.get("d_recept", 0.0),
               z_target=a.z_target, exact_agl=bool(a.exact_agl), tback=a.tback, rel_seconds=a.rel_seconds, res=res,
               stats={k: (float(v) if np.isscalar(v) else None) for k, v in st.items()},
               les=m_les, kljun=m_kl, overlap_kljun=ov80, overlap50_kljun=ov50,
               integral_les=g0.integral(), integral_les_all=g0.integral_all(),
               integral_kljun=float(kl.sum() * g0.area),
               cover_share=r0.get("cover_share", {}),
               cover_share_nowrap=r0.get("cover_share_nowrap", {}),
               wrapped_fraction=r0.get("wrapped_fraction", None),
               sgs_most=bool(a.sgs_most), sgs_most_mode=a.sgs_most_mode,
               sgs_most_legacy=bool(a.sgs_most_legacy),
               sgs_most_form=a.sgs_most_form,
               sgs_subgrid_weight=not a.no_subgrid_weight,
               sgs_eps_consistent=not a.no_eps_consistent,
               wind_angle=r0["wind_angle"])
    # PERSIST THE CLOSURE PROFILES. Re-deriving the floor after the fact needs zlev,
    # ww_prof and esgs_prof, and until now none of the three survived the run -- the
    # window fields are deleted by design, so the sigma_w^2 profile that produced a
    # footprint was unrecoverable the moment the case finished. They are a few kB.
    _fl = r0.get("floor")
    if _fl is not None:
        out["floor"] = {k: ([float(v) for v in _fl[k]]
                            if isinstance(_fl[k], np.ndarray) else _fl[k])
                        for k in ("zl", "fac", "sig2", "base", "wwp", "have", "tgt2",
                                  "kpk", "wstar", "ustar", "h", "L", "d_r", "mode",
                                  "legacy", "delta")}
        # THE INVARIANT'S VERDICT TRAVELS WITH THE CASE. bin/run_corpus_case.sh asserts on
        # it, because PROJECT_BRIEF.md's standing rule is to assert on the artifact rather than
        # on an exit status -- and this analysis is piped into tee, so the exit status
        # belongs to tee.
        if _fl.get("health") is not None:
            out["floor"]["health"] = _fl["health"]
    for _k in ("zlev", "ww_prof", "esgs_prof", "tke_prof"):
        if st.get(_k) is not None:
            out.setdefault("profiles", {})[_k] = [float(v) for v in np.asarray(st[_k])]

    # ---- how much backward time the estimator actually needs ------------------------
    if r0.get("capture"):
        print("\n=== capture vs t_back (same window, same touchdowns, truncated) ===")
        full_i = r0["grid"].integral_all()
        rows = []
        for m in sorted(r0["capture"]):
            c = r0["capture"][m]
            q = fy_metrics(fyc, c["fy"], res)
            rows.append(dict(t_back=m, integral=c["integral"],
                             frac=c["integral"] / max(full_i, 1e-12),
                             peak_x=q["peak_x"], x80=q["x80"],
                             cover=c.get("cover", {})))
            print(f"   t_back {m:5.0f} s   integral {c['integral']:.3f}  "
                  f"= {100*c['integral']/max(full_i,1e-12):5.1f}% of the "
                  f"{a.tback:.0f} s value   peak {q['peak_x']:5.0f} m   "
                  f"80% within {q['x80']:6.0f} m")
        rows.append(dict(t_back=a.tback, integral=full_i, frac=1.0,
                         peak_x=m_les["peak_x"], x80=m_les["x80"],
                         cover=r0.get("cover_share_nowrap", {})))
        out["capture"] = rows
        ck = [k for k in (rows[0].get("cover") or {})]
        if ck:
            print("\n  land-cover share vs t_back (does the OBSERVABLE converge sooner")
            print("  than the integral? the array is within 250 m; the tail is not)")
            print("   " + "t_back".rjust(7) + "".join(f"{k:>13}" for k in ck))
            for r in rows:
                print(f"   {r['t_back']:7.0f}" +
                      "".join(f"{100*(r['cover'].get(k) or 0.0):12.2f}%" for k in ck))

    if r0.get("by_disp"):
        print("\n=== where the integral comes from, by trajectory displacement ===")
        print("   the domain repeats every %.0f m, so anything beyond one domain length is"
              % fs0.Lx)
        print("   turbulence the trajectory had already sampled once")
        for d in r0["by_disp"]:
            print(f"   within {d['frac_of_Lx']:4.2f} L  ({d['max_disp']:5.0f} m)   "
                  f"integral {d['integral']:.3f}")
        out["by_disp"] = r0["by_disp"]

    print("\n=== error floor: the two halves of this window ===")
    if "halves" in r0:
        h1, h2 = r0["halves"]
        a1, a2 = h1.normalised("flux"), h2.normalised("flux")
        m1 = describe("half 1", h1, r0["fy"]["f1"], fyc, res)
        m2 = describe("half 2", h2, r0["fy"]["f2"], fyc, res)
        ovh = source_area_overlap(np.maximum(a1, 0), np.maximum(a2, 0))
        print(f"  half-vs-half 80% overlap {ovh*100:.0f}%   "
              f"peak difference {m1['peak_x']-m2['peak_x']:+.0f} m   "
              f"centroid difference {np.hypot(m1['centroid_e']-m2['centroid_e'], m1['centroid_n']-m2['centroid_n']):.0f} m")
        out["halves"] = dict(overlap=ovh, dpeak=m1["peak_x"] - m2["peak_x"],
                             dx80=m1["x80"] - m2["x80"],
                             dcentroid=float(np.hypot(m1["centroid_e"] - m2["centroid_e"],
                                                      m1["centroid_n"] - m2["centroid_n"])))
        ch = r0.get("cover_share_halves") or [{}, {}]
        out["cover_share_halves"] = ch
        if ch[0]:
            ng = len(ch)
            print(f"  land-cover share over {ng} independent release groups "
                  f"(the sampling distribution of each share):")
            # PERSIST THE SAMPLING SPREAD, NOT JUST THE VALUE. Until now this block
            # printed the standard error and dropped it: the .txt held the only copy of
            # the number that says whether a share DIFFERENCE means anything, and across
            # 1370 cases nothing downstream could reach it. A share quoted without its
            # own SE is exactly the reporting failure PROJECT_BRIEF.md forbids -- "score a
            # second moment against its own sampling spread".
            out["cover_share_se"] = {}
            out["cover_share_groups_n"] = ng
            for nm in ch[0]:
                v = np.array([100*(g.get(nm) or np.nan) for g in ch], dtype=float)
                v = v[np.isfinite(v)]
                if v.size < 2:
                    continue
                se = v.std(ddof=1)/np.sqrt(v.size)
                out["cover_share_se"][nm] = float(se) / 100.0     # a FRACTION, like the
                                                                  # share it belongs to
                print(f"    {nm:<12} mean {v.mean():6.2f}%  sd {v.std(ddof=1):5.2f}  "
                      f"se {se:5.2f}  range {v.min():6.2f}-{v.max():6.2f}"
                      + (f"  [{' '.join(f'{x:.1f}' for x in v)}]" if ng <= 12 else ""))
    if len(runs) > 1:
        g1 = runs[1][2]["grid"]
        b0, b1 = g0.normalised("flux"), g1.normalised("flux")
        mb = describe("realisation 2", g1, runs[1][2]["fy"]["f"], fyc, res)
        ovr = source_area_overlap(np.maximum(b0, 0), np.maximum(b1, 0))
        num = np.abs(b0 - b1).sum(); den = 0.5 * (np.abs(b0).sum() + np.abs(b1).sum())
        print(f"  realisation-vs-realisation 80% overlap {ovr*100:.0f}%   "
              f"normalised L1 {num/den*100:.0f}%")
        out["floor"] = dict(overlap=ovr, l1_rel=float(num / den))

    np.savez_compressed(
        os.path.join(a.outdir, f"{a.tag}.npz"),
        xc=g0.xc, yc=g0.yc, xe=g0.xe, ye=g0.ye,
        les=g0.normalised("flux"), conc=g0.normalised("conc"), kljun=kl,
        fy_xc=fyc, fy=fy, fy_kljun=kl_fy,
        fy1=r0["fy"]["f1"], fy2=r0["fy"]["f2"],
        half1=r0["halves"][0].normalised("flux") if "halves" in r0 else np.zeros(1),
        half2=r0["halves"][1].normalised("flux") if "halves" in r0 else np.zeros(1),
        **({f"cap_{int(m)}": r0["capture"][m]["fy"] for m in r0.get("capture", {})}),
        **({"les2": runs[1][2]["grid"].normalised("flux")} if len(runs) > 1 else {}))
    with open(os.path.join(a.outdir, f"{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  wrote {a.outdir}/{a.tag}.npz and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
