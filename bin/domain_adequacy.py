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


def spectra(paths):
    from netCDF4 import Dataset
    print("=== lock-in diagnostic: the 2-D spectrum of w at mid-depth ===")
    print("An unconstrained CBL peaks near lambda ~ 1.5 z_i. A locked one snaps to the")
    print("largest mode the box holds (mode 1, lambda = L) and that mode takes an")
    print("anomalous share of the variance.\n")
    print(f"  {'dump':<26}{'z_i':>7}{'L/z_i':>7}{'z':>7}{'peak lam':>10}"
          f"{'lam/z_i':>9}{'mode1 %':>9}{'r(L/2)':>8}")
    rows = []
    for p in paths:
        with Dataset(p) as ds:
            w = np.squeeze(np.asarray(ds["w"][:], dtype=np.float64))
            th = np.squeeze(np.asarray(ds["theta"][:], dtype=np.float64))
            z = np.squeeze(np.asarray(ds["zPos"][:], dtype=np.float64))[:, 0, 0]
            x = np.squeeze(np.asarray(ds["xPos"][:], dtype=np.float64))
        if not (np.isfinite(w).all() and np.isfinite(th).all()):
            print(f"  {os.path.basename(p):<26}  NON-FINITE FIELD -- skipped")
            continue
        dx = float(x[0, 0, 1] - x[0, 0, 0])
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


def compare(fa, fb):
    a, b = (json.load(open(f)) for f in (fa, fb))
    print("=== footprint observables: shallow (L=4 z_i) vs deep (L=2 z_i) ===")
    print(f"  A = {fa}\n  B = {fb}\n")
    fl_a = a.get("halves", {}) or {}
    fl_b = b.get("halves", {}) or {}
    # The sampling floor is the larger of the two windows' own half-vs-half differences.
    floor_peak = max(abs(fl_a.get("dpeak", 0.0)), abs(fl_b.get("dpeak", 0.0)))
    floor_cent = max(fl_a.get("dcentroid", 0.0), fl_b.get("dcentroid", 0.0))
    print(f"  sampling floor from the two windows' own halves: "
          f"peak {floor_peak:.0f} m, centroid {floor_cent:.0f} m")
    ok = True
    rows = []

    def cmp(name, va, vb, floor, null, unit="", rel=False):
        nonlocal ok
        d = vb - va
        tol = max(floor, null)
        good = abs(d) <= tol if not rel else abs(d) <= tol * max(abs(va), 1e-9)
        ok &= good
        lim = f"{tol:.3g}{unit}" if not rel else f"{100*tol:.1f}%"
        dd = f"{d:+.3g}{unit}" if not rel else f"{100*d/max(abs(va),1e-9):+.1f}%"
        rows.append(f"  {name:<22}{va:10.3f}{vb:10.3f}   {dd:>10}   tol {lim:>8}   "
                    f"{'ok' if good else 'DIFFERS'}")

    print(f"\n  {'observable':<22}{'shallow':>10}{'deep':>10}{'diff':>13}{'tolerance':>14}")
    cmp("peak_x (m)", a["les"]["peak_x"], b["les"]["peak_x"],
        floor_peak, KLJUN_NULL["peak_m"], " m")
    cmp("centroid dist (m)", a["les"]["centroid_dist"], b["les"]["centroid_dist"],
        floor_cent, 0.0, " m")
    cmp("x80 of f_y (m)", a["les"]["x80"], b["les"]["x80"], 0.0,
        KLJUN_NULL["x90_rel"], rel=True)
    sa = 100 * (a.get("cover_share_nowrap", {}).get("solar array") or 0.0)
    sb = 100 * (b.get("cover_share_nowrap", {}).get("solar array") or 0.0)
    cmp("array share (points)", sa, sb, 0.0, KLJUN_NULL["array_share_pts"], " pts")
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
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    mode = sys.argv[1]
    if mode == "spectra":
        spectra(sys.argv[2:])
        sys.exit(0)
    elif mode == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    print(__doc__)
    sys.exit(2)
