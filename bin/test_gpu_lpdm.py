#!/usr/bin/env python3
"""ACCEPTANCE for the GPU-resident LPDM. The port ships only if it matches the CPU one.

THE BAR, and why it is set here. There is a validated CPU implementation. A CUDA rewrite
of a Langevin integrator with drift terms is exactly where well-mixedness breaks, and it
breaks SILENTLY -- the footprint still looks like a footprint. So the GPU path is not
"tested"; it is scored against the CPU path on the SAME saved fields, and a failure means
the CPU path stays in production and this is reported as not having worked.

Four tests, in the order they localise a fault:

  (d) INGEST      kDerive vs lpdm/fields.py on one snapshot's e and theta. No GPU physics
                  involved; if this fails, nothing downstream means anything. Also: the
                  fp16 round-trip, scored on the CPU path alone, which costs no GPU and
                  bounds what the ring's precision is worth.
  (a) FOOTPRINT   same fields, same releases, same floor table: peak, centroid, A80,
                  integral and array share within the CPU path's OWN half-vs-half floor.
                  Against that floor and not against zero, because the two paths draw from
                  different random streams (curand Philox vs numpy PCG64) and can only
                  ever agree statistically -- the same standard FastEddy itself is held to.
  (b) WELL-MIXED  Gate D1 run THROUGH the GPU integrator. This is the one that matters:
                  a well-mixed test is a test of the drift, and the drift is the part a
                  port gets wrong.
  (c) SIGN        negative lobes survive. The estimator is signed by construction and a
                  port that dropped w_release's sign would look almost right.

usage:
  bin/test_gpu_lpdm.py runs/g24_ctlcbl/window --dt 0.0295858 --z-target 30 [--tback 450]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm.fields import FieldSet, dump_series                       # noqa: E402
from lpdm.footprint import FootprintGrid, source_area_mask          # noqa: E402
from lpdm.driver import compute_footprint                           # noqa: E402
from lpdm import sgs_floor, les_stats                               # noqa: E402


def metrics(flux, n_particles, area, xc, yc):
    f = flux / max(n_particles, 1) / area
    tot = f.sum()
    if tot <= 0:
        return dict(integral=float(tot * area), degenerate=True)
    X, Y = np.meshgrid(xc, yc)
    m80 = source_area_mask(f, 0.80)
    jp, ip = np.unravel_index(int(np.argmax(f)), f.shape)
    return dict(integral=float(tot * area),
                peak_e=float(xc[ip]), peak_n=float(yc[jp]),
                peak_r=float(np.hypot(xc[ip], yc[jp])),
                centroid_e=float((f * X).sum() / tot),
                centroid_n=float((f * Y).sum() / tot),
                centroid_r=float(np.hypot((f * X).sum() / tot, (f * Y).sum() / tot)),
                area80_ha=float(m80.sum() * area / 1e4),
                neg_mag_share=float(np.abs(f[f < 0]).sum() / np.abs(f).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("window")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--z-target", type=float, default=30.0)
    ap.add_argument("--tback", type=float, default=450.0)
    ap.add_argument("--nrel", type=int, default=400)
    ap.add_argument("--dtrel", type=float, default=4.0)
    ap.add_argument("--rel-seconds", type=float, default=None)
    ap.add_argument("--cover-dir", default=None)
    ap.add_argument("--receptor-from", default=None)
    ap.add_argument("--no-sgs-most", action="store_true")
    ap.add_argument("--cover-groups", type=int, default=10,
                    help="release groups for the cover-share sampling floor. A TOLERANCE "
                         "MEASURED FROM ONE DIFFERENCE IS NOT A TOLERANCE (docs/reference/standing-rules.md): "
                         "the 2-group floor came out 0.00 and failed a 1.2-point "
                         "difference against nothing.")
    ap.add_argument("--wm-n", type=int, default=40000,
                    help="particles for the well-mixed gate; 0 skips it")
    ap.add_argument("--wm-score-top", type=float, default=400.0)
    ap.add_argument("--wm-release-top", type=float, default=1200.0)
    ap.add_argument("--json", default="results/gpu_lpdm_acceptance.json")
    a = ap.parse_args()

    paths = dump_series(a.window)
    if len(paths) < 10:
        print(f"FATAL: only {len(paths)} dumps in {a.window}", file=sys.stderr)
        return 2
    print(f"=== fields: {len(paths)} dumps from {a.window} ===")
    fs = FieldSet(paths, a.dt, verbose=False)
    print(f"  {fs.nx} x {fs.ny} x {fs.zk.size}, dx {fs.dx:.1f} m, "
          f"cadence {fs.dt_dump:.1f} s, cache {fs.mem_gb:.2f} GB")

    out = {"window": a.window, "n_dumps": len(paths), "z_target": a.z_target,
           "tback": a.tback}
    fails = []

    # ---------------------------------------------------------------- (d) INGEST -------
    print("\n=== (d) INGEST: kDerive vs lpdm/fields.py on one snapshot ===")
    from netCDF4 import Dataset
    with Dataset(paths[len(paths) // 2]) as ds:
        e_h = np.squeeze(np.asarray(ds["TKE_0"][:], dtype=np.float32))
        th_h = np.squeeze(np.asarray(ds["theta"][:], dtype=np.float32))
    e_h = np.maximum(e_h, 0.0)
    from lpdm.gpu import GpuLPDM
    g = GpuLPDM(fs, td_capacity=4_000_000)
    eps_g, ds2_g = g.derive_check(e_h, th_h)
    # the CPU reference is what FieldSet itself stored for this snapshot (interior part)
    s = len(paths) // 2
    eps_c = np.asarray(fs.eps[s][:, :fs.ny, :fs.nx], dtype=np.float64)
    ds2_c = np.asarray(fs.dsig2dz[s][:, :fs.ny, :fs.nx], dtype=np.float64)
    for nm, gg, cc, tol in (("eps", eps_g, eps_c, 2e-2), ("dsig2dz", ds2_g, ds2_c, 2e-2)):
        gg = np.asarray(gg, dtype=np.float64)
        den = np.maximum(np.abs(cc), np.percentile(np.abs(cc), 90) or 1e-12)
        rel = np.abs(gg - cc) / den
        ok = bool(np.isfinite(gg).all() and np.median(rel) <= tol
                  and np.percentile(rel, 99) <= 10 * tol)
        fails += [] if ok else [f"(d) {nm}"]
        print(f"  [{'PASS' if ok else 'FAIL'}] {nm}: median rel {np.median(rel):.2e}, "
              f"p99 {np.percentile(rel, 99):.2e} (tol {tol:.0e} / {10*tol:.0e})")
        out[f"derive_{nm}_median_rel"] = float(np.median(rel))
        out[f"derive_{nm}_p99_rel"] = float(np.percentile(rel, 99))
    print("  NOTE: both sides are stored fp16, so ~1e-3 relative is the representation "
          "floor;\n  the tolerance is set an order of magnitude above it and the p99 is "
          "reported so a\n  localised disagreement cannot hide behind a good median.")

    # ---------------------------------------------------------------- setup for (a)/(c)
    cover = None
    if a.cover_dir:
        z0 = np.load(os.path.join(a.cover_dir, "z0m.npy"))
        cover = {}
        for nm, fn in (("water", "water.npy"), ("solar array", "array.npy")):
            p = os.path.join(a.cover_dir, fn)
            if os.path.exists(p):
                cover[nm] = np.load(p) > 0.5
        del z0
    rij = None
    if a.receptor_from and os.path.exists(os.path.join(a.receptor_from, "meta.npy")):
        m_ = np.load(os.path.join(a.receptor_from, "meta.npy"), allow_pickle=True).item()
        rij = (m_["itower"], m_["jtower"])

    print(f"\n=== (a) FOOTPRINT: CPU vs GPU on identical fields ===")
    t0 = time.time()
    cpu = compute_footprint(fs, paths, z_target=a.z_target, n_per_release=a.nrel,
                            dt_release=a.dtrel, t_back=a.tback, cover=cover,
                            sgs_most=not a.no_sgs_most, receptor_ij=rij,
                            rel_seconds=a.rel_seconds, split_halves=True,
                            n_cover_groups=a.cover_groups, verbose=False)
    t_cpu = time.time() - t0
    st = cpu["stats"]
    grid_c = cpu["grid"]
    xc, yc, area = grid_c.xc, grid_c.yc, grid_c.area
    m_cpu = metrics(grid_c.flux, grid_c.n_particles, area, xc, yc)

    # THE SAME FLOOR TABLE, so the comparison is of the integrator and not of two floors.
    if not a.no_sgs_most:
        d_r = 0.0
        fl = sgs_floor.most_floor(st, d_r=d_r, mode="surface")
        g.set_floor(fl["zl"], fl["fac"], mode=1)
        print(f"  floor: multiplicative sc(z), {len(fl['zl'])} levels, "
              f"max factor {np.max(fl['fac']):.2f}")
    else:
        g.set_floor(np.array([0.0, 1.0]), np.array([1.0, 1.0]), mode=0)
    if cover:
        g.set_cover(cover)

    # THE SAME RELEASE TIMES THE CPU PATH USED -- rebuilt from lpdm/driver.py's own rule,
    # not re-derived here. The first version of this harness released over
    # [t0 + t_back, t0 + t_back + rel_seconds] while the driver releases over
    # [t_last - rel_seconds, t_last]: a different 300 s of a spinning-up convective layer,
    # which is a difference in the FLOW rather than in the integrator, and it showed up as
    # a 27% integral gap that looked like a port bug.
    xr, yr, zr = cpu["receptor"]
    t_last = float(fs.t[-1])
    t_first = float(fs.t[0]) + a.tback
    if a.rel_seconds:
        t_first = max(t_first, t_last - float(a.rel_seconds))
    times = np.arange(t_first, t_last + 1e-9, a.dtrel)
    t_rel = np.repeat(times, a.nrel)
    print(f"  {len(times)} release times x {a.nrel} particles = {len(t_rel):,}")
    g.reset()
    t0 = time.time()
    g.release(xr, yr, zr, t_rel, t_limit=a.tback, max_disp=fs.Lx,
              yaw=float(cpu["yaw"]), pitch=float(cpu["pitch"]),
              td_prob=1.0, seed=12345, x_ref=xr, y_ref=yr)
    t_gpu = time.time() - t0
    res = g.fetch()
    m_gpu = metrics(res["flux"], res["n_particles"], area, xc, yc)
    print(f"  CPU {t_cpu:6.1f} s   GPU {t_gpu:6.1f} s   speed-up {t_cpu/max(t_gpu,1e-9):.1f}x")

    # the CPU path's OWN half-vs-half floor is the tolerance
    floor = {}
    if "halves" in cpu:
        h1, h2 = cpu["halves"]
        m1 = metrics(h1.flux, h1.n_particles, area, xc, yc)
        m2 = metrics(h2.flux, h2.n_particles, area, xc, yc)
        for k in ("peak_r", "centroid_r", "area80_ha", "integral"):
            floor[k] = abs(m1.get(k, np.nan) - m2.get(k, np.nan))
    print(f"\n  {'metric':<14} {'CPU':>12} {'GPU':>12} {'|diff|':>10} "
          f"{'CPU floor':>10}  verdict")
    for k, unit in (("peak_r", "m"), ("centroid_r", "m"), ("area80_ha", "ha"),
                    ("integral", "")):
        c_, gv = m_cpu.get(k, np.nan), m_gpu.get(k, np.nan)
        d = abs(c_ - gv)
        fl_ = floor.get(k, np.nan)
        # A floor of exactly zero is not a tolerance -- the peak's own sampling p90 is
        # 0 m in a converged window, so fall back to one raster cell there.
        tol = max(fl_, fs.dx if k == "peak_r" else 0.0) if np.isfinite(fl_) else np.nan
        ok = bool(np.isfinite(tol) and d <= tol)
        fails += [] if ok else [f"(a) {k}"]
        print(f"  {k:<14} {c_:>12.4f} {gv:>12.4f} {d:>10.4f} {tol:>10.4f}  "
              f"{'PASS' if ok else 'FAIL'} {unit}")
        out[f"a_{k}"] = dict(cpu=float(c_), gpu=float(gv), diff=float(d),
                             tol=float(tol) if np.isfinite(tol) else None, ok=ok)
    if cover:
        print()
        nw = cpu.get("cover_share_nowrap", {})
        wrapped = cpu.get("wrapped_fraction", float("nan"))
        for nm in cover:
            c_ = cpu["cover_share"].get(nm, np.nan)
            gv = res["cover_share"].get(nm, np.nan)
            grp = [h.get(nm, np.nan) for h in cpu["cover_share_halves"]]
            se = np.nanstd(grp, ddof=1) / np.sqrt(max(len(grp), 2))
            # A TOLERANCE MEASURED AT ZERO IS NOT A TOLERANCE. Where the release groups
            # all report the same share the estimated floor is 0 and every difference
            # "fails" against nothing. That happens here for a real reason and it is worth
            # naming: cover_share FOLDS touchdowns into the periodic domain, and the
            # per-group floor is computed on the UNWRAPPED subset -- so a class whose
            # entire share comes from wrapped touchdowns (a fold onto a different lake,
            # exactly as lpdm/driver.py warns) has an unwrapped share of zero in every
            # group and hence no measurable spread. Return NO VERDICT there rather than a
            # failure, which is the same discipline the stationarity gate uses.
            if not np.isfinite(se) or se <= 0.0:
                verdict, ok = "NO VERDICT", None
            else:
                ok = bool(abs(c_ - gv) <= 3.0 * se)
                verdict = "PASS" if ok else "FAIL"
                fails += [] if ok else [f"(a) share {nm}"]
            print(f"  share {nm:<20} CPU {100*c_:6.2f}%  GPU {100*gv:6.2f}%  "
                  f"|diff| {100*abs(c_-gv):5.2f}  3 SE {100*3*se:5.2f}  "
                  f"(n={len(grp)} groups; unwrapped CPU share "
                  f"{100*nw.get(nm, float('nan')):5.2f}%)  {verdict}")
            out[f"a_share_{nm}"] = dict(cpu=float(c_), gpu=float(gv), three_se=float(3*se),
                                        cpu_nowrap=float(nw.get(nm, np.nan)), ok=ok)
        print(f"  {100*wrapped:.1f}% of the CPU footprint's |weight| came from touchdowns "
              f"the periodic fold moved.")

    # ---------------------------------------------------------------- (c) SIGN ---------
    print("\n=== (c) SIGN: negative lobes survive the port ===")
    for nm, m in (("CPU", m_cpu), ("GPU", m_gpu)):
        print(f"  {nm}: |negative| share of |flux| = {100*m['neg_mag_share']:.2f}%")
    ok = bool(m_gpu["neg_mag_share"] > 0.005
              and abs(m_gpu["neg_mag_share"] - m_cpu["neg_mag_share"]) < 0.5 *
              max(m_cpu["neg_mag_share"], 1e-9) + 0.01)
    fails += [] if ok else ["(c) sign"]
    print(f"  [{'PASS' if ok else 'FAIL'}] the GPU footprint is signed and its negative "
          f"lobe is the CPU one's size")
    out["c_neg_share"] = dict(cpu=m_cpu["neg_mag_share"], gpu=m_gpu["neg_mag_share"], ok=ok)


    # ---------------------------------------------------------------- (b) WELL-MIXED ---
    # THE ONE THAT MATTERS. A well-mixed test is a test of the DRIFT, and the drift is the
    # part a Langevin port gets wrong -- silently, because the footprint still looks like a
    # footprint. Run in BOTH directions: backward is what footprints use, forward is the
    # control that localises a sign error in the reverse-time term (lpdm/model.py).
    if a.wm_n:
        from lpdm import wellmixed
        from lpdm.model import LPDM
        print(f"\n=== (b) WELL-MIXED: Gate D1 through both integrators ===")
        sgs_scale = ((fl["zl"], fl["fac"]) if not a.no_sgs_most else 1.0)
        cpu_lp = LPDM(fs, c0=3.0, seed=1, sgs_scale=sgs_scale)
        rng = np.random.default_rng(1)
        z_lo = cpu_lp.z_touch
        nwm = int(a.wm_n)
        xs = rng.uniform(fs.x0, fs.x0 + fs.Lx, nwm)
        ys = rng.uniform(fs.y0, fs.y0 + fs.Ly, nwm)
        zs = rng.uniform(z_lo, a.wm_release_top, nwm)
        edges = np.linspace(z_lo, a.wm_score_top, 21)
        h0, _ = np.histogram(zs, bins=edges)
        noise = float(np.sqrt(2.0 / h0.mean()))
        print(f"  {nwm:,} particles over {z_lo:.0f}-{a.wm_release_top:.0f} m, scored "
              f"{z_lo:.0f}-{a.wm_score_top:.0f} m in 20 bins; counting noise "
              f"{100*noise:.2f}%")
        print(f"  {'direction':>10} {'path':>5} {'max|r-1|':>10} {'rms':>8} "
              f"{'lowest 3':>10}  verdict")
        for direction, tlab in ((-1, "backward"), (+1, "forward")):
            t_rel0 = float(fs.t[-1] if direction < 0 else fs.t[0])
            row = {}
            for lab in ("CPU", "GPU"):
                if lab == "CPU":
                    r = cpu_lp.run(xs, ys, zs, t_rel0, direction=direction,
                                   t_limit=a.tback, reflect_touchdown=True,
                                   record_touchdown=False)
                    zf = r["z"]
                else:
                    g.reset()
                    r = g.release(xs, ys, zs, np.full(nwm, t_rel0),
                                  t_limit=a.tback, direction=direction, max_disp=0.0,
                                  td_prob=0.0, seed=999, want_final=True,
                                  x_ref=float(fs.x0), y_ref=float(fs.y0))
                    zf = r["z"]
                h1, _ = np.histogram(zf, bins=edges)
                ratio = h1 / np.maximum(h0, 1)
                dev = np.abs(ratio - 1.0)
                lo3 = float(h1[:3].sum() / max(h0[:3].sum(), 1))
                ok = bool(dev.max() < max(0.10, 4 * noise)
                          and abs(lo3 - 1.0) < max(0.05, 3 * noise))
                se3 = float(np.sqrt(2.0 / max(h0[:3].sum(), 1)))
                row[lab] = dict(maxdev=float(dev.max()),
                                rms=float(np.sqrt((dev ** 2).mean())), lo3=lo3,
                                lo3_se=se3, ok=ok)
                print(f"  {tlab:>10} {lab:>5} {100*dev.max():9.2f}% "
                      f"{100*np.sqrt((dev**2).mean()):7.2f}% {lo3:10.3f} "
                      f"+/-{100*se3:4.2f}%  {'PASS' if ok else 'FAIL'}")
            # THE GPU MUST PASS. The CPU row is the control: if the CPU also fails, the
            # fields are the problem and not the port, and saying so is the whole reason
            # both are run on the same release ensemble.
            # TWO SEPARATE QUESTIONS, and conflating them is how a port gets blamed for
            # a property of the fields. (i) Does the GPU pass Gate D1 in absolute terms?
            # (ii) Does it AGREE with the CPU on the same release ensemble? A marginal
            # absolute failure that agrees with the CPU inside their combined counting
            # error is a statement about this window; a disagreement is a port defect.
            dlo3 = abs(row["GPU"]["lo3"] - row["CPU"]["lo3"])
            comb = float(np.hypot(row["GPU"]["lo3_se"], row["CPU"]["lo3_se"]))
            agree = bool(dlo3 <= 3.0 * comb)
            row["agree"] = dict(dlo3=dlo3, three_se=3 * comb, ok=agree)
            print(f"  {'':>10} {'diff':>5} lowest-3 CPU vs GPU {dlo3:.3f} against "
                  f"3 combined SE {3*comb:.3f}  {'AGREE' if agree else 'DISAGREE'}")
            if not agree:
                fails.append(f"(b) well-mixed {tlab}: GPU and CPU DISAGREE")
            elif not row["GPU"]["ok"] and row["CPU"]["ok"]:
                fails.append(f"(b) well-mixed {tlab}: GPU fails D1 where the CPU passes")
            elif not row["GPU"]["ok"]:
                print(f"  {'':>10} note: BOTH paths fail D1 {tlab} on this window -- a "
                      f"statement about the\n             fields, not the port. Not "
                      f"counted against the port.")
            out[f"b_{tlab}"] = row
        print("  the CPU row is the CONTROL. A GPU failure where the CPU also fails is a "
              "statement\n  about these fields, not about the port -- which is why both "
              "run on the same ensemble.")
        # the GPU path must reset its accumulators before anything else uses them
        g.reset()

    out["timing"] = dict(cpu_s=t_cpu, gpu_s=t_gpu, speedup=t_cpu / max(t_gpu, 1e-9))
    out["fails"] = fails
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    print(f"\n  wrote {a.json}")
    print(f"\n########## GPU LPDM ACCEPTANCE: "
          f"{'PASS' if not fails else 'FAIL (' + ', '.join(fails) + ')'} ##########")
    if fails:
        print("  The CPU path stays in production. Do not ship a GPU LPDM that has not "
              "matched it.")
    g.close()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
