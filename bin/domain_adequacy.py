#!/usr/bin/env python3
"""Is L >= 4 z_i actually binding for a 10 m footprint, or only for w*-scaling?

THE QUESTION. A doubly-periodic CBL is conventionally run with L >= 4 z_i, or the largest
thermals lock to the domain. In a 1952 m box that caps z_i at 488 m, which covers only
19.3% of this site's convective-midday hours -- and the cap is BIASED, because z_i and
surface heat flux are positively correlated, so it excludes preferentially the strongly
heated states where the solar array's flux enhancement is largest (bin/zi_coverage.py).
Accepting L >= 2 z_i instead would take coverage to 60.9%. Paying for it with a bigger box
costs 3.2x. So the question is worth measuring rather than assuming.

The 4 z_i rule was written for mixed-layer similarity -- w* scaling, entrainment ratio,
the shape of the vertical velocity spectrum at mid-depth. Whether it corrupts a footprint
at 10 m, a height where surface-layer scaling governs and where Kljun's own z_i channel
spans 1 percentage point of array share over h = 200-1200 m, is a SEPARATE question and
is unmeasured.

TWO INDEPENDENT LINES OF EVIDENCE, because the two cases are genuinely different physical
states and a footprint difference alone could not be attributed:

  1. LOCK-IN, DIAGNOSED DIRECTLY, with no reference to the footprint. The 2-D spectrum of
     w at mid-depth. An unconstrained CBL peaks near lambda ~ 1.5 z_i; a locked one snaps
     to the largest mode the box holds, so the peak sits at lambda = L and the k=1 mode
     carries an anomalous share of the variance. This is the artifact itself, and it
     either is or is not present.

  2. THE FOOTPRINT OBSERVABLES, against BOTH the half-vs-half sampling floor measured in
     the same window AND the Kljun null for this z_i pair. Kljun predicts +0.24 points of
     array share and -0.9% in x90 between z_i = 488 and 976 m at a 10 m receptor; a
     difference inside that is not evidence of anything.

usage:
  domain_adequacy.py spectra <dump.nc> [<dump.nc> ...]        lock-in diagnostic
  domain_adequacy.py compare <shallow.json> <deep.json>       footprint comparison
"""
import json
import os
import sys

import numpy as np


# ------------------------------------------------------------------ lock-in diagnostic
def zi_from_theta(th, z):
    """Inversion height as the level of maximum slab-mean d(theta)/dz.

    The standard CBL definition, and the right one here: the TKE-threshold estimate
    lpdm.les_stats uses is fine for a footprint's `h` but is noisy in exactly the
    mid-depth region this diagnostic samples.
    """
    prof = th.mean(axis=(-2, -1))
    dth = np.gradient(prof, z)
    k = int(np.argmax(dth[2:])) + 2
    return float(z[k]), k


def spectrum_2d(f, dx):
    """Radially binned 2-D power spectrum of one horizontal plane.

    Returns (wavelength, power) with the mean removed, normalised to unit total power,
    so two runs with different w variance are directly comparable in SHAPE.
    """
    a = f - f.mean()
    ny, nx = a.shape
    P = np.abs(np.fft.fft2(a)) ** 2
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.hypot(KX, KY)
    # bin on the fundamental 1/L, so bin index n IS the mode number
    dk = 1.0 / (nx * dx)
    idx = np.round(kr / dk).astype(int)
    nb = min(nx, ny) // 2
    pw = np.bincount(idx.ravel(), weights=P.ravel(), minlength=nb + 1)[:nb + 1]
    pw[0] = 0.0
    tot = pw.sum()
    if tot > 0:
        pw = pw / tot
    n = np.arange(len(pw))
    with np.errstate(divide="ignore"):
        lam = np.where(n > 0, nx * dx / np.maximum(n, 1), np.inf)
    return lam, pw, n


def _geometry(paths):
    """z column and dx, from whichever dump in the series actually carries them.

    Sampling windows are written in the fork's `ioLPDMmode`: lean, 16-bit, and holding
    ONLY the fields the LPDM reads. The coordinate geometry is written into the FIRST
    dump of a run and no other, so asking the last dump for `zPos` raises
    "IndexError: zPos not found in /" -- which is what this function exists to avoid.
    """
    from netCDF4 import Dataset
    for p in paths:
        with Dataset(p) as ds:
            if "zPos" in ds.variables and "xPos" in ds.variables:
                z = np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))[:, 0, 0]
                x = np.squeeze(np.asarray(ds["xPos"][:], dtype=np.float64))
                return z, float(x[0, 0, 1] - x[0, 0, 0])
    # Nothing in the series carries it, so fall back to the LOWEST-numbered dump in the
    # same directory -- which for a sampling window is the full-form step-0 dump the fork
    # writes before lean output begins. This is what makes the diagnostic usable on a
    # window without the caller having to know how ioLPDMmode lays its output out.
    import glob as _glob
    d = os.path.dirname(paths[0]) if paths else "."
    cand = sorted(_glob.glob(os.path.join(d, "*.[0-9]*")),
                  key=lambda q: int(q.rsplit(".", 1)[1]))
    for p in cand:
        with Dataset(p) as ds:
            if "zPos" in ds.variables and "xPos" in ds.variables:
                z = np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))[:, 0, 0]
                x = np.squeeze(np.asarray(ds["xPos"][:], dtype=np.float64))
                print(f"  (geometry taken from {os.path.basename(p)}: lean ioLPDMmode "
                      f"output writes zPos/xPos once, in the run's first dump)")
                return z, float(x[0, 0, 1] - x[0, 0, 0])
    raise SystemExit("  no dump in this directory carries zPos/xPos")


def spectra(paths, geom_from=None):
    from netCDF4 import Dataset
    print("=== lock-in diagnostic: the 2-D spectrum of w at mid-depth ===")
    print("An unconstrained CBL peaks near lambda ~ 1.5 z_i. A locked one snaps to the")
    print("largest mode the box holds (mode 1, lambda = L) and that mode takes an")
    print("anomalous share of the variance.\n")
    print(f"  {'dump':<26}{'z_i':>7}{'L/z_i':>7}{'z':>7}{'peak lam':>10}"
          f"{'lam/z_i':>9}{'mode1 %':>9}{'r(L/2)':>8}")
    rows = []
    z, dx = _geometry(list(geom_from or []) + list(paths))
    for p in paths:
        with Dataset(p) as ds:
            w = np.squeeze(np.asarray(ds["w"][:], dtype=np.float64))
            th = np.squeeze(np.asarray(ds["theta"][:], dtype=np.float64))
        if not (np.isfinite(w).all() and np.isfinite(th).all()):
            print(f"  {os.path.basename(p):<26}  NON-FINITE FIELD -- skipped")
            continue
        L = w.shape[-1] * dx
        zi, _ = zi_from_theta(th, z)
        km = int(np.argmin(np.abs(z - 0.5 * zi)))
        lam, pw, n = spectrum_2d(w[km], dx)
        ip = int(np.argmax(pw))
        # autocorrelation at half a domain length: a locked cell pattern is strongly
        # ANTI-correlated there, because one updraft and one downdraft fill the box.
        a = w[km] - w[km].mean()
        half = w.shape[-1] // 2
        r = float((a * np.roll(a, half, axis=1)).mean() / max(a.var(), 1e-30))
        rows.append(dict(dump=os.path.basename(p), zi=zi, L_over_zi=L / zi,
                         z=float(z[km]), peak_lambda=float(lam[ip]),
                         lam_over_zi=float(lam[ip] / zi), mode1=float(pw[1]),
                         r_half=r))
        print(f"  {os.path.basename(p):<26}{zi:6.0f}m{L/zi:7.2f}{z[km]:7.0f}"
              f"{lam[ip]:9.0f}m{lam[ip]/zi:9.2f}{100*pw[1]:8.1f}%{r:8.3f}")
    print("\n  READ: lam/z_i near 1.5 and a small mode-1 share = unconstrained.")
    print("        peak lambda pinned at L, a large mode-1 share, and r(L/2) strongly")
    print("        negative = the box is organising the thermals.")
    return rows


# ------------------------------------------------------------------ footprint compare
# Kljun's own z_i sensitivity at a 10 m receptor, measured in Phase A: over
# h = 488 -> 976 m the array share moves +0.24 points and x90 by -0.9%. A difference
# inside that is what the MODEL ITSELF predicts from the z_i change and is not evidence
# that the box did anything.
KLJUN_NULL = dict(array_share_pts=0.24, x90_rel=0.009, peak_m=1.0)


def compare(fa, fb, res=16.0):
    """Compare the two footprints against a floor MEASURED the same way each quantity is.

    Every tolerance here is the larger of two things that were both measured, never an
    opinion: the half-vs-half difference within each window (the sampling floor), and what
    Kljun itself predicts from the z_i change alone (the model's own null). A third floor,
    one grid cell, applies to the peak -- the peak is quantised to the raster, so a
    tolerance below 16 m is asking the measurement to be finer than its own resolution.
    """
    a, b = (json.load(open(f)) for f in (fa, fb))
    print("=== footprint observables: shallow (L=4 z_i) vs deep (L=2 z_i) ===")
    print(f"  A = {fa}\n  B = {fb}\n")
    fa_h = a.get("halves", {}) or {}
    fb_h = b.get("halves", {}) or {}
    flo = lambda k: max(abs(fa_h.get(k, 0.0) or 0.0), abs(fb_h.get(k, 0.0) or 0.0))
    floor_peak, floor_cent, floor_x80 = flo("dpeak"), flo("dcentroid"), flo("dx80")

    def share(d, cls="solar array"):
        return 100.0 * (d.get("cover_share_nowrap", {}).get(cls) or 0.0)

    def share_floor(d, cls="solar array"):
        ch = d.get("cover_share_halves") or [{}, {}]
        if not ch[0] or cls not in ch[0]:
            return 0.0
        return 100.0 * abs((ch[0][cls] or 0.0) - (ch[1][cls] or 0.0))
    floor_share = max(share_floor(a), share_floor(b))

    print("  sampling floors, from each window's own halves:")
    print(f"    peak {floor_peak:.0f} m (raster resolution {res:.0f} m)   "
          f"centroid {floor_cent:.0f} m   x80 {floor_x80:.0f} m   "
          f"array share {floor_share:.2f} points")
    print("  Kljun's own null for z_i 488 -> 976 m at a 10 m receptor:")
    print(f"    array share {KLJUN_NULL['array_share_pts']:.2f} points   "
          f"x90 {100*KLJUN_NULL['x90_rel']:.1f}%   peak {KLJUN_NULL['peak_m']:.0f} m")

    ok = True
    rows = []

    def cmp(name, va, vb, tol, unit=""):
        nonlocal ok
        d = vb - va
        good = abs(d) <= tol
        ok &= good
        rows.append(f"  {name:<22}{va:10.3f}{vb:10.3f}{d:+11.3f}{unit:<4}"
                    f"  tol {tol:7.2f}{unit:<4}  {'ok' if good else 'DIFFERS'}")

    print(f"\n  {'observable':<22}{'shallow':>10}{'deep':>10}{'diff':>11}"
          f"{'':4}  {'tolerance':>11}{'':4}")
    cmp("peak_x (m)", a["les"]["peak_x"], b["les"]["peak_x"],
        max(floor_peak, KLJUN_NULL["peak_m"], res), " m")
    cmp("centroid dist (m)", a["les"]["centroid_dist"], b["les"]["centroid_dist"],
        max(floor_cent, res), " m")
    cmp("x80 of f_y (m)", a["les"]["x80"], b["les"]["x80"],
        max(floor_x80, KLJUN_NULL["x90_rel"] * abs(a["les"]["x80"]), res), " m")
    sa, sb = share(a), share(b)
    cmp("array share (points)", sa, sb,
        max(floor_share, KLJUN_NULL["array_share_pts"]), " pt")
    print("\n".join(rows))

    if ok:
        verdict = ("PASS -- L >= 2 z_i is not binding for this observable. Convective-"
                   "midday\n          coverage goes 19.3% -> 60.9% and 122^3 covers "
                   "the corpus.")
    else:
        verdict = ("DIFFERS -- report the size of the error and STOP. The fix is ~218^2 "
                   "@ 16 m\n          (L = 3488 m, 3.2x cost, 53.0% coverage at "
                   "L >= 4 z_i), and that is a\n          grid decision, which belongs "
                   "to the user.")
    print(f"\n  GATE E: {verdict}")
    print("\n  This half of the gate cannot stand alone: the two cases are genuinely")
    print("  different physical states, so agreement here is necessary and not sufficient.")
    print("  Read it WITH the lock-in spectra, which detect the artifact directly.")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    mode = sys.argv[1]
    if mode == "spectra":
        # any --geom argument names a dump to take zPos/xPos from
        args = sys.argv[2:]
        geom = []
        if "--geom" in args:
            i = args.index("--geom")
            geom = [args[i + 1]]
            args = args[:i] + args[i + 2:]
        spectra(args, geom_from=geom)
        sys.exit(0)
    elif mode == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    print(__doc__)
    sys.exit(2)
