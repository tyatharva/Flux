"""Stage 4 gate: the well-mixed condition.

Release particles uniformly and integrate. A correct Lagrangian stochastic model leaves
the distribution uniform forever (Thomson 1987). If particles accumulate near the surface,
the sub-grid closure violates well-mixedness, and every footprint computed afterwards is
wrong exactly in the near field where the signal lives.

Run in BACKWARD mode, because that is the mode the footprints use. Forward is the control:
a model that is well mixed forward but not backward means the reverse-time drift term (see
lpdm/model.py) has the wrong sign, which is the single most likely error in the pipeline.

**No artificial lid.** An earlier version closed the layer with a reflecting lid at 400 m
and produced a clean profile everywhere except a 2x pile-up in the bin touching the lid --
in BOTH time directions. That was the test's fault, not the model's: reflection flips the
particle's sub-grid velocity but not the RESOLVED w interpolated from the LES, so a particle
reflected at an arbitrary height keeps its resolved upward motion and gets pinned against
the boundary. At the real surface this does not arise, because resolved w is taken to zero
there by impermeability -- which is why the near-surface bins were uniform to 0.5% while the
lid bin was off by 95%.

So instead: release uniformly through a column that extends WELL ABOVE the boundary layer,
and score only the interior. Above the boundary layer FastEddy's SGS TKE is ~0, particles
there barely move, and that quiescent column acts as a reservoir with no artificial
boundary anywhere near the region being scored.
"""
from __future__ import annotations

import numpy as np


def run_test(lpdm, fs, n=40000, z_lo=None, z_score_top=400.0, z_release_top=1200.0,
             t_release=None, t_limit=900.0, direction=-1, nbins=20, seed=1):
    rng = np.random.default_rng(seed)
    z_lo = lpdm.z_touch if z_lo is None else z_lo
    t0 = float(fs.t[-1] if direction < 0 else fs.t[0]) if t_release is None else t_release
    x = rng.uniform(fs.x0, fs.x0 + fs.Lx, n)
    y = rng.uniform(fs.y0, fs.y0 + fs.Ly, n)
    z = rng.uniform(z_lo, z_release_top, n)

    res = lpdm.run(x, y, z, t0, direction=direction, t_limit=t_limit,
                   reflect_touchdown=True, record_touchdown=False)
    zf = res["z"]

    edges = np.linspace(z_lo, z_score_top, nbins + 1)
    h0, _ = np.histogram(z, bins=edges)
    h1, _ = np.histogram(zf, bins=edges)
    # Scored per-bin expectation is the INITIAL count in that bin, not n/nbins: the release
    # spans a deeper column than the scored region, so the reference is what was there.
    return dict(edges=edges, h0=h0, h1=h1, z_init=z, z_final=zf,
                n_scored=int(h0.sum()), n_final=len(zf), iters=res["iters"],
                z_release_top=z_release_top)


def report(out, label=""):
    edges, h0, h1 = out["edges"], out["h0"], out["h1"]
    zc = 0.5 * (edges[:-1] + edges[1:])
    ratio = h1 / np.maximum(h0, 1)
    lo3 = float(h1[:3].sum() / max(h0[:3].sum(), 1))
    dev = np.abs(ratio - 1.0)
    noise = np.sqrt(2.0 / h0.mean())          # two independent counts per bin
    print(f"\n  --- well-mixed test {label} ---")
    print(f"  released uniformly over {edges[0]:.0f}-{out['z_release_top']:.0f} m; "
          f"scored over {edges[0]:.0f}-{edges[-1]:.0f} m")
    print(f"  {'z (m)':>9} {'final/initial':>15}")
    for c, r in zip(zc, ratio):
        print(f"  {c:9.1f} {r:15.3f}   |{'#' * int(min(r, 2.0) / 2.0 * 40)}")
    print(f"  ~{h0.mean():.0f} particles per bin initially; "
          f"1-sigma counting noise {noise*100:.2f}%")
    print(f"  max |ratio-1| = {dev.max()*100:.2f}%   rms = {np.sqrt((dev**2).mean())*100:.2f}%"
          f"   lowest 3 bins = {lo3:.3f}")
    # A closure failure is a MONOTONE near-surface pile-up, not bin-to-bin scatter, so the
    # near-surface bins are held to a tighter bound than the profile as a whole.
    ok = dev.max() < max(0.10, 4 * noise) and abs(lo3 - 1.0) < max(0.05, 3 * noise)
    print(f"  GATE: {'PASS' if ok else 'FAIL'}"
          f"  (max deviation < max(10%, 4 sigma); lowest 3 bins within max(5%, 3 sigma))")
    return ok
