"""Stage 4 gate: the well-mixed condition.

Release particles uniformly through a closed layer and integrate. A correct Lagrangian
stochastic model leaves the distribution uniform forever (Thomson 1987). If particles
accumulate near the surface, the sub-grid closure violates well-mixedness, and every
footprint computed afterwards is wrong exactly in the near field where the signal lives.

This is run in BACKWARD mode, because that is the mode the footprints use. Forward mode
is also available and is checked as a control: a model that is well mixed forward but not
backward means the reverse-time drift term (see lpdm/model.py) has the wrong sign, which
is the single most likely error in the whole pipeline.

The layer is closed with a reflecting lid inside the boundary layer rather than at the
domain top, because above the boundary layer FastEddy's SGS TKE goes to zero, the
Lagrangian timescale diverges, and the test would be measuring the damping layer instead
of the closure.
"""
from __future__ import annotations

import numpy as np


def run_test(lpdm, fs, n=40000, z_lo=None, z_lid=500.0, t_release=None,
             t_limit=600.0, direction=-1, nbins=20, seed=1):
    rng = np.random.default_rng(seed)
    z_lo = lpdm.z_touch if z_lo is None else z_lo
    t0 = float(fs.t[-1] if direction < 0 else fs.t[0]) if t_release is None else t_release
    x = rng.uniform(fs.x0, fs.x0 + fs.Lx, n)
    y = rng.uniform(fs.y0, fs.y0 + fs.Ly, n)
    z = rng.uniform(z_lo, z_lid, n)
    z_init = z.copy()

    res = lpdm.run(x, y, z, t0, direction=direction, t_limit=t_limit,
                   reflect_touchdown=True, record_touchdown=False, z_ceil=z_lid)
    zf = res["z"]

    edges = np.linspace(z_lo, z_lid, nbins + 1)
    h0, _ = np.histogram(z_init, bins=edges)
    h1, _ = np.histogram(zf, bins=edges)
    exp = n / nbins
    return dict(edges=edges, h0=h0, h1=h1, expected=exp, z_final=zf,
                n_final=len(zf), iters=res["iters"])


def report(out, label=""):
    edges, h1, exp = out["edges"], out["h1"], out["expected"]
    zc = 0.5 * (edges[:-1] + edges[1:])
    ratio = h1 / exp
    lo3 = ratio[:3].mean()
    dev = np.abs(ratio - 1.0)
    # binomial counting noise on one bin, as a 1-sigma fraction of the expectation
    noise = np.sqrt(exp * (1 - 1 / len(ratio))) / exp
    print(f"\n  --- well-mixed test {label} ---")
    print(f"  {'z (m)':>9} {'count/expected':>16}   {'':<44}")
    for c, r in zip(zc, ratio):
        bar = "#" * int(min(r, 2.0) / 2.0 * 40)
        print(f"  {c:9.1f} {r:16.3f}   |{bar}")
    print(f"  expected per bin {exp:.0f}   1-sigma counting noise {noise*100:.2f}%")
    print(f"  max |ratio-1| = {dev.max()*100:.2f}%   rms = {np.sqrt((dev**2).mean())*100:.2f}%"
          f"   lowest 3 bins mean = {lo3:.3f}")
    # A real closure failure shows as a MONOTONE near-surface pile-up, not scatter.
    ok = dev.max() < max(0.10, 4 * noise) and abs(lo3 - 1.0) < max(0.05, 3 * noise)
    print(f"  GATE: {'PASS' if ok else 'FAIL'}"
          f"  (max deviation < max(10%, 4 sigma) and lowest-3-bin mean within 5%)")
    return ok
