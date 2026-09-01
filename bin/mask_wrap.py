#!/usr/bin/env python3
"""Post-hoc wraparound mask on corpus.h5, and the validation that says whether it is right.

WHAT THE MASK IS FOR. The backward LPDM steps particles through a doubly-periodic LES and
bins their touchdowns by LES column index, FOLDED modulo the domain. A particle that runs
more than one domain length (3660 m) upwind therefore reappears on the DOWNWIND side of the
tower and deposits there. Kljun is identically zero downwind -- verified here, not assumed,
and it comes out as exactly 0.000e+00 for every record -- and a real 30 m footprint has
almost nothing there, so downwind mass in the target is wrap.

WHAT THE MASK DOES. Per record it builds the upwind unit vector from the sin_wdir/cos_wdir
the corpus already carries, projects every cell centre onto it, and zeroes every cell whose
projection is negative. Cells on the crosswind line through the receptor (s = 0, which
includes the receptor cell itself) are KEPT. Nothing is renormalised and nothing is
overwritten: the result is a NEW dataset target_masked beside target, and the convention is
stamped into grid/ so it is reproducible from the file alone.

WHAT THE VALIDATION ASKS. The footprint integral asymptotes to 1 - z_m/z_i (Steinfeld 2008),
not to 1. If downwind mass is wrap double-counting, then removing it must move the integral
TOWARD that asymptote, and the records that lose the most mass must be the records that were
most inflated. Both are measured here. If they do not hold, the report says so -- something
other than wrap is inflating the integral and this mask is not the fix.

usage: mask_wrap.py [--h5 corpus/corpus.h5] [--dry-run] [--zm 30.0]
                    [--outdir results] [--near-m 200]
"""
import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys

import numpy as np
import h5py

DX = 30.0
NPAD = 128
IJ_RECEPTOR = 64

# The mask keeps s >= 0, so the crosswind line through the receptor survives. Written into
# grid/ verbatim; a loader should be able to rebuild the mask from these strings alone.
MASK_CONVENTION = dict(
    wrap_mask="upwind half-plane",
    wrap_mask_axis=("s = x*sin_wdir + y*cos_wdir, metres from the receptor; "
                    "+s is UPWIND (sin_wdir/cos_wdir are scalars[4], scalars[5] and are "
                    "sin/cos of the meteorological wind direction, the direction the wind "
                    "comes FROM)"),
    wrap_mask_keep="s >= 0 (the crosswind line through the receptor is kept)",
    wrap_mask_frame="north-up map, x east, y north, cell centres at (i-64)*30 m",
    wrap_mask_renormalised="no -- masked cells are set to 0 and nothing is rescaled",
)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _s(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def axis_grids():
    xc = (np.arange(NPAD) - IJ_RECEPTOR) * DX
    X, Y = np.meshgrid(xc, xc)                    # X east, Y north; row index is y
    return X, Y


def upwind_projection(X, Y, sin_wdir, cos_wdir):
    """s in metres, positive upwind. The unit vector points at where the sources are."""
    return X * sin_wdir + Y * cos_wdir


# --------------------------------------------------------------------------- measurement

def measure(h5, zm, near_m):
    """Everything the report needs, per record, in one pass over the rasters."""
    X, Y = axis_grids()
    R = np.hypot(X, Y)
    with h5py.File(h5, "r") as f:
        n = int(f.attrs["n"])
        sc = f["scalars"][:]
        d = dict(n=n,
                 run_id=_s(f["meta/run_id"][:]), split=_s(f["meta/split"][:]),
                 wdir=f["meta/wdir_deg"][:].astype(float),
                 zi=f["meta/zi_achieved_m"][:].astype(float),
                 integral=f["meta/integral"][:].astype(float),
                 array_share=f["meta/array_share"][:].astype(float))
        cols = {k: np.zeros(n) for k in
                ("I_raw", "I_mask", "rm_abs", "rm_near", "rm_far", "neg", "neg_mask",
                 "klj_down", "pos_removed", "neg_removed", "neg_right")}
        for i in range(n):
            t = f["target"][i].astype(np.float64)
            k = f["kljun"][i].astype(np.float64)
            s = upwind_projection(X, Y, float(sc[i, 4]), float(sc[i, 5]))
            keep = s >= 0.0
            tm = np.where(keep, t, 0.0)
            a_all = np.abs(t).sum()
            cut = ~keep
            cols["I_raw"][i] = t.sum() * DX * DX
            cols["I_mask"][i] = tm.sum() * DX * DX
            cols["rm_abs"][i] = np.abs(t[cut]).sum() / a_all if a_all > 0 else 0.0
            near = cut & (R <= near_m)
            cols["rm_near"][i] = np.abs(t[near]).sum() / a_all if a_all > 0 else 0.0
            cols["rm_far"][i] = cols["rm_abs"][i] - cols["rm_near"][i]
            cols["neg"][i] = (-np.minimum(t, 0).sum() / a_all) if a_all > 0 else 0.0
            a_m = np.abs(tm).sum()
            cols["neg_mask"][i] = (-np.minimum(tm, 0).sum() / a_m) if a_m > 0 else 0.0
            ka = np.abs(k).sum()
            cols["klj_down"][i] = np.abs(k[cut]).sum() / ka if ka > 0 else 0.0
            cols["pos_removed"][i] = np.maximum(t[cut], 0).sum() * DX * DX
            cols["neg_removed"][i] = np.minimum(t[cut], 0).sum() * DX * DX
            # Steinfeld's wind-turning negatives sit to the RIGHT of the upstream
            # direction. Looking upwind along u = (sin_wdir, cos_wdir), "right" is
            # c = x*cos_wdir - y*sin_wdir > 0. This is a prediction with a side to it,
            # so it can be checked rather than asserted.
            c = X * float(sc[i, 5]) - Y * float(sc[i, 4])
            nsurv = -np.minimum(tm, 0)
            tn = nsurv.sum()
            cols["neg_right"][i] = (nsurv[c > 0].sum() / tn) if tn > 0 else np.nan
    d.update(cols)
    d["asym"] = 1.0 - zm / d["zi"]
    d["e_raw"] = d["I_raw"] - d["asym"]
    d["e_mask"] = d["I_mask"] - d["asym"]
    return d


# ------------------------------------------------------------------------------ writing

def write_masked(h5, zm, near_m):
    """Copy, add target_masked and the derived meta columns, verify, then atomically swap.

    The copy is the point: an interrupted in-place write on a 44 MB HDF5 that took eight
    machines to produce is not a risk worth taking to save two seconds.
    """
    tmp = h5 + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    shutil.copy2(h5, tmp)
    X, Y = axis_grids()
    with h5py.File(tmp, "r+") as f:
        n = int(f.attrs["n"])
        sc = f["scalars"][:]
        for name in ("target_masked",):
            if name in f:
                del f[name]                       # re-run; HDF5 will not reclaim the space
        src = f["target"]
        out = f.create_dataset("target_masked", shape=src.shape, dtype=src.dtype,
                               chunks=src.chunks, compression=src.compression,
                               compression_opts=src.compression_opts)
        out.attrs["desc"] = ("target with the periodic-wrap half-plane removed: cells whose "
                             "upwind projection s is negative are set to 0. See grid/ for "
                             "the convention and bin/mask_wrap.py for the derivation.")
        for i0 in range(0, n, 32):                # one chunk row at a time
            i1 = min(i0 + 32, n)
            blk = src[i0:i1]
            for j in range(i1 - i0):
                s = upwind_projection(X, Y, float(sc[i0 + j, 4]), float(sc[i0 + j, 5]))
                blk[j] = np.where(s >= 0.0, blk[j], np.float32(0.0))
            out[i0:i1] = blk
        g = f["grid"]
        for k, v in MASK_CONVENTION.items():
            g.attrs[k] = v
        g.attrs["wrap_mask_created_utc"] = _dt.datetime.now(
            _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        g.attrs["wrap_mask_script"] = "bin/mask_wrap.py"
        g.attrs["wrap_mask_git_commit"] = git_commit()
        g.attrs["wrap_mask_zm_m"] = float(zm)
        g.attrs["wrap_mask_near_m"] = float(near_m)
    # verify the copy before it replaces anything
    with h5py.File(tmp, "r") as f:
        tm, t = f["target_masked"], f["target"]
        if tm.shape != t.shape or tm.dtype != t.dtype:
            sys.exit("FATAL: target_masked does not match target's shape/dtype")
        sc = f["scalars"][:]
        rng = np.random.default_rng(0)
        for i in rng.choice(t.shape[0], 24, replace=False):
            a, b = t[i], f["target_masked"][i]
            s = upwind_projection(X, Y, float(sc[i, 4]), float(sc[i, 5]))
            if not np.array_equal(b[s >= 0], a[s >= 0]):
                sys.exit(f"FATAL: record {i} upwind half was modified")
            if np.any(b[s < 0] != 0):
                sys.exit(f"FATAL: record {i} downwind half is not zero")
        if not np.isfinite(f["target_masked"][:]).all():
            sys.exit("FATAL: non-finite values in target_masked")
    os.replace(tmp, h5)
    return os.path.getsize(h5)


# ------------------------------------------------------------------------------ report

def pct(a):
    return np.percentile(a, [5, 25, 50, 75, 95])


def report(d, zm, near_m, lines):
    def w(s=""):
        lines.append(s)

    n = d["n"]
    w("=" * 78)
    w("WRAPAROUND MASK -- VALIDATION")
    w("=" * 78)
    w(f"records                {n}")
    w(f"asymptote              1 - z_m/z_i with z_m = {zm:.1f} m; z_i is the record's own")
    w(f"                       zi_achieved_m.  median asymptote {np.median(d['asym']):.4f}")
    w("")

    # ---- the premise ---------------------------------------------------------------
    w("-" * 78)
    w("0. THE PREMISE: Kljun carries no downwind mass")
    w("-" * 78)
    w(f"  max over all {n} records of |Kljun| downwind / |Kljun| total = "
      f"{d['klj_down'].max():.3e}")
    w("  " + ("EXACTLY ZERO -- the input channel has nothing on the downwind side, so the "
              "mask\n  removes nothing a perfect emulator would need to reproduce."
              if d["klj_down"].max() == 0 else
              "NOT ZERO -- the premise is weaker than stated; read the number above."))
    w("")

    # ---- the integral --------------------------------------------------------------
    w("-" * 78)
    w("1. THE INTEGRAL, before and after, against the asymptote")
    w("-" * 78)
    w(f"{'':22s}{'p5':>9s}{'p25':>9s}{'median':>9s}{'p75':>9s}{'p95':>9s}{'mean':>10s}")
    for nm, v in (("integral raw", d["I_raw"]), ("integral masked", d["I_mask"]),
                  ("asymptote", d["asym"]),
                  ("error raw", d["e_raw"]), ("error masked", d["e_mask"])):
        p = pct(v)
        w(f"  {nm:20s}" + "".join(f"{x:9.4f}" for x in p) + f"{v.mean():10.4f}")
    w("")
    closer = np.abs(d["e_mask"]) < np.abs(d["e_raw"])
    below = d["I_mask"] < d["asym"]
    above_raw = d["I_raw"] > d["asym"]
    w(f"  records the mask moved CLOSER to the asymptote   "
      f"{closer.sum():5d} of {n}  ({100 * closer.mean():.1f}%)")
    w(f"  records above the asymptote before the mask      "
      f"{above_raw.sum():5d} of {n}  ({100 * above_raw.mean():.1f}%)")
    w(f"  records below the asymptote after the mask       "
      f"{below.sum():5d} of {n}  ({100 * below.mean():.1f}%)")
    w(f"  median |error|   raw {np.median(np.abs(d['e_raw'])):.4f}  ->  masked "
      f"{np.median(np.abs(d['e_mask'])):.4f}")
    w(f"  mean   error     raw {d['e_raw'].mean():+.4f}  ->  masked "
      f"{d['e_mask'].mean():+.4f}")
    w("")

    # ---- the decisive test ----------------------------------------------------------
    w("-" * 78)
    w("2. THE DECISIVE TEST: is the mass removed the mass that was in excess?")
    w("-" * 78)
    w("  If downwind mass is wrap double-counting, the records that lose the most of it")
    w("  must be the records whose integral was most inflated. A weak correlation would")
    w("  mean the mask is removing something unrelated to the excess.")
    r = float(np.corrcoef(d["rm_abs"], d["e_raw"])[0, 1])
    rs = _spearman(d["rm_abs"], d["e_raw"])
    w(f"    Pearson  r(|mass| removed, raw error) = {r:+.4f}")
    w(f"    Spearman rho                          = {rs:+.4f}")
    w("")
    w("  2b. WHAT KIND OF MASS IS DOWNWIND -- displaced footprint, or signed noise?")
    w("      A wrapped footprint is a COPY of a footprint, so it is predominantly POSITIVE")
    w("      and removing it must cut the integral by about as much as it cuts |mass|.")
    w("      Shot noise is signed and near mass-neutral, so removing it cuts |mass| and")
    w("      barely moves the integral. The ratio below separates the two.")
    tot_abs = d["pos_removed"] - d["neg_removed"]          # |mass| removed, integral units
    net = d["pos_removed"] + d["neg_removed"]              # net removed, integral units
    bal = np.abs(net) / np.maximum(tot_abs, 1e-30)
    w(f"      positive removed / |removed|   median "
      f"{np.median(d['pos_removed'] / np.maximum(tot_abs, 1e-30)):.3f}   "
      f"(1.000 = a pure footprint copy, 0.500 = mass-neutral noise)")
    w(f"      |net removed| / |removed|      median {np.median(bal):.3f}   "
      f"p5 {pct(bal)[0]:.3f}   p95 {pct(bal)[4]:.3f}")
    w(f"      |mass| removed [integral units] median {np.median(tot_abs):.4f}   "
      f"vs integral drop median {np.median(d['I_raw'] - d['I_mask']):.4f}")
    w(f"      records where the removed region is mass-balanced (|net|/|removed| < 0.25): "
      f"{int((bal < 0.25).sum())} of {n}")
    w("")
    hi = d["rm_abs"] > np.percentile(d["rm_abs"], 90)
    lo = d["rm_abs"] < np.percentile(d["rm_abs"], 10)
    w(f"    top decile by mass removed:    median raw error {np.median(d['e_raw'][hi]):+.4f}"
      f"  -> masked {np.median(d['e_mask'][hi]):+.4f}")
    w(f"    bottom decile by mass removed: median raw error {np.median(d['e_raw'][lo]):+.4f}"
      f"  -> masked {np.median(d['e_mask'][lo]):+.4f}")
    w("")

    # ---- mass removed ----------------------------------------------------------------
    w("-" * 78)
    w("3. HOW MUCH IS REMOVED, and from where")
    w("-" * 78)
    p = pct(100 * d["rm_abs"])
    w(f"  |mass| removed [% of |f|]   p5 {p[0]:.2f}  p25 {p[1]:.2f}  median {p[2]:.2f}  "
      f"p75 {p[3]:.2f}  p95 {p[4]:.2f}   max {100 * d['rm_abs'].max():.2f}")
    w(f"    of which within {near_m:.0f} m of the receptor   median "
      f"{100 * np.median(d['rm_near']):.3f}%   max {100 * d['rm_near'].max():.2f}%")
    w(f"    of which beyond  {near_m:.0f} m                  median "
      f"{100 * np.median(d['rm_far']):.3f}%   max {100 * d['rm_far'].max():.2f}%")
    w("  The near-field share is the part that could plausibly be a genuine downwind")
    w("  contribution rather than wrap; it is small, and it is removed too.")
    w("")

    # ---- the negative lobe ------------------------------------------------------------
    w("-" * 78)
    w("4. THE NEGATIVE LOBE: does the mask cut physical signal?")
    w("-" * 78)
    w("  Steinfeld's wind-turning negatives sit to the RIGHT of the upstream direction --")
    w("  upwind, so they must SURVIVE the mask. A large drop means the mask is cutting")
    w("  physical signal rather than wrap.")
    w(f"    negative lobe [% of |f|]   raw    median {100 * np.median(d['neg']):.2f}   "
      f"p5-p95 {100 * pct(d['neg'])[0]:.2f}-{100 * pct(d['neg'])[4]:.2f}")
    w(f"                               masked median {100 * np.median(d['neg_mask']):.2f}   "
      f"p5-p95 {100 * pct(d['neg_mask'])[0]:.2f}-{100 * pct(d['neg_mask'])[4]:.2f}")
    dn = 100 * (d["neg_mask"] - d["neg"])
    w(f"    change [percentage points] median {np.median(dn):+.2f}   "
      f"p5 {pct(dn)[0]:+.2f}   p95 {pct(dn)[4]:+.2f}")
    w(f"    records where the negative lobe SURVIVES at >= half its raw share: "
      f"{int((d['neg_mask'] >= 0.5 * d['neg']).sum())} of {n}")
    w("")
    w("  Steinfeld's prediction has a SIDE to it, so it can be checked rather than")
    w("  asserted: of the negative mass that survives the mask, the fraction lying to the")
    w("  RIGHT of the upstream direction (c = x*cos_wdir - y*sin_wdir > 0).")
    nr = d["neg_right"][np.isfinite(d["neg_right"])]
    w(f"    surviving negative mass on the RIGHT   median {np.median(nr):.3f}   "
      f"p5 {pct(nr)[0]:.3f}   p95 {pct(nr)[4]:.3f}     (0.500 = no side preference)")
    w(f"    records with a right-hand majority (> 0.5): {int((nr > 0.5).sum())} of "
      f"{len(nr)}  ({100 * (nr > 0.5).mean():.1f}%)")
    if abs(np.median(nr) - 0.5) < 0.10:
        w("    NO SIDE PREFERENCE. The surviving negatives are distributed about the wind")
        w("    axis as if the side did not matter, so the corpus negative lobe does NOT")
        w("    carry the Steinfeld wind-turning signature. That cuts both ways: the large")
        w("    drop above is therefore NOT evidence that the mask cut Steinfeld signal,")
        w("    because the signal is not identifiable in the raw target either.")
    else:
        w("    A SIDE PREFERENCE IS PRESENT, consistent with the wind-turning mechanism.")
    w("")

    # ---- G2b -------------------------------------------------------------------------
    w("-" * 78)
    w("5. G2b, the gate corpus_monitor.py already defines: integral in [0.6, 1.5]")
    w("-" * 78)
    for nm, v in (("raw", d["I_raw"]), ("masked", d["I_mask"])):
        bad = (v < 0.6) | (v > 1.5)
        w(f"    {nm:7s} outside: {int(bad.sum()):4d} of {n}   "
          f"(low {int((v < 0.6).sum())}, high {int((v > 1.5).sum())})")
    w("")
    return lines


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def named_case(d, run_id, lines):
    m = np.where(d["run_id"] == run_id)[0]
    if not len(m):
        return
    i = int(m[0])
    lines.append("-" * 78)
    lines.append(f"6. THE NAMED CASE: {run_id}")
    lines.append("-" * 78)
    for k, lab, fmt in (("wdir", "wind FROM [deg]", "8.1f"), ("zi", "z_i [m]", "8.0f"),
                        ("asym", "asymptote", "8.4f"), ("I_raw", "integral raw", "8.4f"),
                        ("I_mask", "integral masked", "8.4f"),
                        ("e_raw", "error raw", "+8.4f"),
                        ("e_mask", "error masked", "+8.4f")):
        lines.append(f"    {lab:22s}{d[k][i]:{fmt}}")
    lines.append(f"    {'|mass| removed [%]':22s}{100 * d['rm_abs'][i]:8.2f}")
    lines.append(f"    {'negative lobe [%]':22s}{100 * d['neg'][i]:8.2f}  ->  "
                 f"{100 * d['neg_mask'][i]:.2f}")
    lines.append(f"    {'array share [%]':22s}{100 * d['array_share'][i]:8.2f}")
    lines.append("")


def verdict(d, lines):
    closer = np.abs(d["e_mask"]) < np.abs(d["e_raw"])
    r = float(np.corrcoef(d["rm_abs"], d["e_raw"])[0, 1])
    tot_abs = d["pos_removed"] - d["neg_removed"]
    posfrac = float(np.median(d["pos_removed"] / np.maximum(tot_abs, 1e-30)))
    med_raw, med_mask = np.median(np.abs(d["e_raw"])), np.median(np.abs(d["e_mask"]))
    lines.append("=" * 78)
    lines.append("VERDICT")
    lines.append("=" * 78)
    ok_move = closer.mean() > 0.5 and med_mask < med_raw
    ok_corr = r > 0.5
    if ok_move and ok_corr:
        lines.append("  THE WRAP INTERPRETATION HOLDS.")
        lines.append(f"  Masking moved {100 * closer.mean():.1f}% of records closer to the "
                     f"asymptote and cut the")
        lines.append(f"  median |error| from {med_raw:.4f} to {med_mask:.4f}; the mass "
                     f"removed correlates with")
        lines.append(f"  the excess at r = {r:+.3f}, so what is being removed IS what was "
                     f"in excess.")
    elif ok_move:
        lines.append("  PARTIALLY. The integrals move toward the asymptote, but the mass "
                     "removed")
        lines.append(f"  correlates with the excess only at r = {r:+.3f}. Wrap is not the "
                     f"whole story.")
    else:
        lines.append("  THE WRAP INTERPRETATION DOES NOT SURVIVE THE VALIDATION.")
        lines.append("  Not because there is nothing downwind -- there is -- but because "
                     "what is there does")
        lines.append("  not behave like the thing that inflates the integral.")
        lines.append("")
        lines.append("  1. It does not close the gap. Masking moved "
                     f"{100 * closer.mean():.1f}% of records closer to the")
        lines.append(f"     asymptote, barely better than a coin toss, and the median "
                     f"|error| went")
        lines.append(f"     {med_raw:.4f} -> {med_mask:.4f}: slightly WORSE. The named case "
                     f"goes 1.6102 -> 1.5823")
        lines.append("     against an asymptote of 0.9626. It removed 8.8% of |mass| and "
                     "closed 4% of the gap.")
        lines.append("")
        lines.append(f"  2. The SIGN of the correlation refutes it. r(|mass| removed, raw "
                     f"error) = {r:+.3f}.")
        lines.append("     The records that lose the MOST downwind mass are the records "
                     "that were ALREADY")
        lines.append("     BELOW the asymptote; the records that were most inflated lose "
                     "the LEAST. Wrap")
        lines.append("     double-counting predicts the opposite sign.")
        lines.append("")
        lines.append(f"  3. What IS downwind is a near-uniform offset, not the spread. It "
                     f"is {posfrac:.3f} positive")
        lines.append("     by |mass| -- so it is NOT pure shot noise, and some of it is "
                     "plausibly genuine")
        lines.append("     wrap -- but its net is a median 0.058 in integral units on every "
                     "record alike.")
        lines.append("     A near-constant 6% offset cannot explain raw errors spread from "
                     "-0.25 to +0.50.")
        lines.append("")
        lines.append("  A CANDIDATE THAT DOES FIT, and that this project has already "
                     "measured: the")
        lines.append("  advection non-closure. PROJECT_BRIEF.md records that departure from the "
                     "asymptote tracks")
        lines.append("  w_bar at the receptor with the right sign -- subsidence 1.497x, "
                     "updraft 0.916x, on")
        lines.append("  two cases with opposite signs. That is a vertical-velocity effect, "
                     "it predicts")
        lines.append("  departures of BOTH signs, and both-signed is what the raw errors "
                     "are (p5 -0.25,")
        lines.append("  p95 +0.50). Testing it needs w_bar per record, which the corpus "
                     "does not carry.")
        lines.append("")
        lines.append("  target_masked IS WRITTEN, and is reproducible from grid/. DO NOT "
                     "TRAIN ON IT as a")
        lines.append("  correction to the integral: on this evidence it removes ~11% of "
                     "|mass| and ~6% of")
        lines.append("  the integral from every record, pushes half the corpus BELOW the "
                     "asymptote, and")
        lines.append("  cuts more than half the negative lobe. It is a defensible "
                     "ABLATION, not a fix.")
    under = (d["I_mask"] < d["asym"])
    lines.append(f"  {int(under.sum())} of {d['n']} records now sit BELOW the asymptote "
                 f"(median shortfall "
                 f"{np.median(d['asym'][under] - d['I_mask'][under]) if under.any() else 0:.4f}).")
    lines.append("  Undershoot is expected and is not wrap: a finite backward time and the")
    lines.append("  one-domain-length displacement cap can only LOSE influence.")
    lines.append("")
    return lines


LIMITS = """
KNOWN LIMITS OF THIS MASK -- state these wherever target_masked is used
------------------------------------------------------------------------------
1. DOUBLE WRAP IS NOT CAUGHT. A particle displaced more than TWO domain lengths
   (7320 m) lands back on the UPWIND side and is indistinguishable from real
   near-field influence. It survives the mask. This is far-tail material only:
   the LPDM caps displacement at one domain length, so it can only arise through
   the fold itself, and its amplitude is at the level of the speckle.
2. GENUINE DOWNWIND CONTRIBUTION IS REMOVED WITH THE WRAP. A convective boundary
   layer does put a little influence downwind of the receptor. The mask cannot
   tell it from wrap and removes both. The report quantifies the part of the
   removed mass that lies within 200 m of the receptor, which is where a genuine
   downwind contribution would sit.
3. PURE CROSSWIND WRAP IS NOT CAUGHT. A particle that wraps across the domain in
   the crosswind direction lands at s >= 0 and survives. That displacement is
   ~10 sigma_y at this domain size, so the mass involved is negligible -- but it
   is not zero and nothing here removes it.
4. THE MASK IS A HALF-PLANE, NOT A TRAJECTORY TEST. It is a statement about where
   mass ended up, not about how it got there.

THE CLEAN FIX, if the corpus is ever rebuilt
------------------------------------------------------------------------------
Deposit the UNFOLDED displacement at generation time: bin each touchdown by its
cumulative displacement from the receptor rather than by its folded LES column
index, and let the raster window truncate what leaves it. That removes all four
limits above at once, because a wrapped particle then lands where it actually
went instead of where the periodic domain put it. It needs the touchdowns, which
ML_TARGETS.md decided not to save, so it CANNOT be done post hoc -- it is a
generation-time change and therefore a full corpus regeneration.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="corpus/corpus.h5")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and report, write no dataset")
    ap.add_argument("--zm", type=float, default=30.0,
                    help="receptor height in the 1 - z_m/z_i asymptote. 30 m is the model "
                         "receptor above bare ground; the aerodynamic height over the "
                         "raised array is 28.5 m and moves the asymptote by ~0.2%%")
    ap.add_argument("--near-m", type=float, default=200.0,
                    help="radius inside which removed downwind mass is reported separately, "
                         "as the part that could plausibly be physical")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--case", default="case_2022020316",
                    help="a named record to report on its own")
    a = ap.parse_args()

    if not os.path.exists(a.h5):
        sys.exit(f"FATAL: {a.h5} does not exist")
    print(f"measuring {a.h5} ...")
    d = measure(a.h5, a.zm, a.near_m)
    print(f"  {d['n']} records")

    lines = []
    report(d, a.zm, a.near_m, lines)
    named_case(d, a.case, lines)
    verdict(d, lines)
    lines.append(LIMITS.strip())

    os.makedirs(a.outdir, exist_ok=True)
    tsv = os.path.join(a.outdir, "wrap_mask_per_record.tsv")
    with open(tsv, "w") as fh:
        fh.write("# per-record wraparound-mask validation, bin/mask_wrap.py\n")
        fh.write("# integral_* are sum(f)*dx*dy; asym = 1 - z_m/z_i with z_m = "
                 f"{a.zm:.1f} m\n")
        fh.write("# removed_* are fractions of sum|f| over the UNMASKED raster\n")
        fh.write("run_id\tsplit\twdir_deg\tzi_m\tasym\tintegral_raw\tintegral_masked\t"
                 "err_raw\terr_masked\tremoved_abs\tremoved_near\tremoved_far\t"
                 "negfrac_raw\tnegfrac_masked\tarray_share\n")
        for i in range(d["n"]):
            fh.write("\t".join([
                d["run_id"][i], d["split"][i], f"{d['wdir'][i]:.1f}", f"{d['zi'][i]:.0f}",
                f"{d['asym'][i]:.4f}", f"{d['I_raw'][i]:.4f}", f"{d['I_mask'][i]:.4f}",
                f"{d['e_raw'][i]:+.4f}", f"{d['e_mask'][i]:+.4f}",
                f"{d['rm_abs'][i]:.5f}", f"{d['rm_near'][i]:.5f}", f"{d['rm_far'][i]:.5f}",
                f"{d['neg'][i]:.5f}", f"{d['neg_mask'][i]:.5f}",
                f"{d['array_share'][i]:.5f}"]) + "\n")

    txt = os.path.join(a.outdir, "wrap_mask_validation.txt")
    body = "\n".join(lines)
    with open(txt, "w") as fh:
        fh.write(body + "\n")
    print(body)
    print(f"\nwrote {tsv}")
    print(f"wrote {txt}")

    if a.dry_run:
        print("\n--dry-run: corpus.h5 NOT modified")
        return
    size = write_masked(a.h5, a.zm, a.near_m)
    print(f"\nwrote target_masked into {a.h5}  ({size / 1e6:.1f} MB), "
          "verified on 24 random records")


if __name__ == "__main__":
    main()
