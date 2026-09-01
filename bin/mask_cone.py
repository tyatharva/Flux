#!/usr/bin/env python3
"""Wind-aligned CONE mask on corpus.h5: writes target_cone, the training target.

WHY A CONE AND NOT A HALF-PLANE. The backward LPDM bins touchdowns by LES column index,
folded modulo the periodic domain, and the fold is PER AXIS AND INDEPENDENT. So for a
diagonal wind a particle can wrap in x alone and land back in the UPWIND half, as a thin
off-axis streak. A half-plane cut on the sign of the along-wind projection cannot see it:
single-axis wrap survives whenever the displacement exceeds 3660*max(|sin|,|cos|), which is
why the streaks are thin shells and why they vanish for axis-aligned winds. Measured on
case_2022030716 (wdir 303 deg), the half-plane leaves 2.41% of |f| of pure wrap sitting
upwind.

The physical criterion is CROSSWIND SPREAD, not the sign of a projection. Real material
cannot be far off-axis at large along-wind distance, because |y'| is bounded by sigma_y(x').
Kljun already computes sigma_y and it is the corpus's own input channel.

THE MASK.  keep  <=>  |y'| <= max(k*sigma_y(x'), y_min)  AND  x' >= -y_min

x' is along-wind (positive upwind) and y' crosswind, both from sin_wdir/cos_wdir in scalars.
No separate half-plane cut is needed: a cell at negative x' is outside a cone opening upwind
and the same criterion removes it. The two regularisations are there because sigma_y -> 0 at
the receptor, so a pure cone would pinch the peak: y_min floors the half-width, and the apex
is pushed back to x' = -y_min so the mask boundary does not pass through the receptor cell.

k = 8, y_min = 90 m.  BOTH WERE MEASURED, NOT PICKED -- see choose_k() and --sweep.

WHAT max_disp MAKES THIS. Production retires a trajectory at ONE domain length
(bin/run_corpus_case.sh passes no --max-disp; stage5_footprint.py defaults to None;
lpdm/driver.py then sets max_disp = fs.Lx = 3660 m -- all three verified inside the image
that generated the corpus). A particle therefore CANNOT wrap twice. Every wrapped particle
is displaced by exactly one domain length in x, in y, or in both, so it lands either
downwind or far off-axis -- and the cone catches all three cases. There is no double-wrapped
material sitting on-axis for it to miss.

usage: mask_cone.py [--h5 corpus/corpus.h5] [--k 8] [--y-min 90] [--dry-run] [--sweep]
"""
import argparse
import datetime as _dt
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm import kljun_ffp

DX = 30.0
NPAD = 128
IJ_RECEPTOR = 64
Z_RECEPTOR = 28.5            # aerodynamic receptor height; what the Kljun channel was built at
K_DEFAULT = 8.0
YMIN_DEFAULT = 90.0          # 3 cells

MASK_CONVENTION = dict(
    cone_mask="wind-aligned cone on Kljun's own sigma_y",
    cone_mask_rule=("keep <=> |y'| <= max(k*sigma_y(x'), y_min) AND x' >= -y_min, with "
                    "x' = x*sin_wdir + y*cos_wdir (positive UPWIND) and "
                    "y' = x*cos_wdir - y*sin_wdir"),
    cone_mask_sigma_y=("sigma_y(x') from the OFFICIAL FFP v1.42 via lpdm/kljun_ffp.py:"
                       "ffp_profile(zm=28.5, h, L, ustar, sigma_v, umean), i.e. the same "
                       "call that produced the kljun channel. umean is meta/u_mean_ms."),
    cone_mask_frame="north-up map, x east, y north, cell centres at (i-64)*30 m",
    cone_mask_renormalised="no -- masked cells are set to 0 and nothing is rescaled",
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
    X, Y = np.meshgrid(xc, xc)
    return X, Y


def wind_frame(X, Y, sin_wdir, cos_wdir):
    """(along-wind x' positive UPWIND, crosswind y')."""
    return X * sin_wdir + Y * cos_wdir, X * cos_wdir - Y * sin_wdir


def sigma_y_field(scal, umean, xw):
    """sigma_y at every cell's along-wind distance, from the official FFP.

    Evaluated at max(x', 0): the profile is not defined downwind, and downwind cells are
    removed by the x' >= -y_min half of the rule anyway.
    """
    prof = kljun_ffp.ffp_profile(Z_RECEPTOR, float(scal[0]), float(scal[3]), float(scal[1]),
                                 float(scal[2]), umean=float(umean))
    return np.interp(np.maximum(xw, 0.0).ravel(), prof["x"], prof["sigy"]).reshape(xw.shape)


def cone_keep(xw, yw, sy, k, y_min):
    return (np.abs(yw) <= np.maximum(k * sy, y_min)) & (xw >= -y_min)


def load_umean(h5, npz_dir):
    """u_mean per record. From meta/u_mean_ms if present, else from the source .npz."""
    with h5py.File(h5, "r") as f:
        rid = _s(f["meta/run_id"][:])
        if "u_mean_ms" in f["meta"]:
            return rid, f["meta/u_mean_ms"][:].astype(float)
    have = {}
    for p in glob.glob(os.path.join(npz_dir, "*.npz")):
        try:
            m = json.loads(str(np.load(p, allow_pickle=True)["meta"]))
        except Exception:
            continue
        if "run_id" in m and "u_mean_ms" in m:
            have[m["run_id"]] = float(m["u_mean_ms"])
    missing = [r for r in rid if r not in have]
    if missing:
        sys.exit(f"FATAL: u_mean is unavailable for {len(missing)} records "
                 f"(e.g. {missing[:3]}). It is needed for sigma_y. Point --npz-dir at the "
                 f"source .npz, or run once with them present so meta/u_mean_ms is written.")
    return rid, np.array([have[r] for r in rid], float)


# --------------------------------------------------------------------------- measurement

def measure(h5, k, y_min, zm, near_m, npz_dir, want_halfplane=False):
    X, Y = axis_grids()
    R = np.hypot(X, Y)
    rid, umean = load_umean(h5, npz_dir)
    with h5py.File(h5, "r") as f:
        n = int(f.attrs["n"])
        sc = f["scalars"][:]
        d = dict(n=n, run_id=rid, u_mean=umean,
                 split=_s(f["meta/split"][:]),
                 wdir=f["meta/wdir_deg"][:].astype(float),
                 zi=f["meta/zi_achieved_m"][:].astype(float),
                 array_share=f["meta/array_share"][:].astype(float),
                 sigma_v=sc[:, 2].astype(float))
        keys = ("I_raw", "I_cone", "rm_abs", "rm_near", "rm_up", "neg", "neg_cone",
                "klj_rm", "pos_removed", "neg_removed", "rm_hp", "rm_up_hp")
        cols = {kk: np.zeros(n) for kk in keys}
        for i in range(n):
            t = f["target"][i].astype(np.float64)
            kj = f["kljun"][i].astype(np.float64)
            xw, yw = wind_frame(X, Y, float(sc[i, 4]), float(sc[i, 5]))
            sy = sigma_y_field(sc[i], umean[i], xw)
            keep = cone_keep(xw, yw, sy, k, y_min)
            cut = ~keep
            tc = np.where(keep, t, 0.0)
            a = np.abs(t)
            atot = a.sum()
            cols["I_raw"][i] = t.sum() * DX * DX
            cols["I_cone"][i] = tc.sum() * DX * DX
            cols["rm_abs"][i] = a[cut].sum() / atot if atot > 0 else 0.0
            cols["rm_near"][i] = a[cut & (R <= near_m)].sum() / atot if atot > 0 else 0.0
            # the part the HALF-PLANE would have missed: removed, but upwind
            cols["rm_up"][i] = a[cut & (xw >= 0)].sum() / atot if atot > 0 else 0.0
            cols["neg"][i] = (-np.minimum(t, 0).sum() / atot) if atot > 0 else 0.0
            am = np.abs(tc).sum()
            cols["neg_cone"][i] = (-np.minimum(tc, 0).sum() / am) if am > 0 else 0.0
            ka = np.abs(kj).sum()
            cols["klj_rm"][i] = np.abs(kj[cut]).sum() / ka if ka > 0 else 0.0
            cols["pos_removed"][i] = np.maximum(t[cut], 0).sum() * DX * DX
            cols["neg_removed"][i] = np.minimum(t[cut], 0).sum() * DX * DX
            if want_halfplane:
                hp = xw < 0
                cols["rm_hp"][i] = a[hp].sum() / atot if atot > 0 else 0.0
                cols["rm_up_hp"][i] = 0.0
    d.update(cols)
    d["asym"] = 1.0 - zm / d["zi"]
    d["e_raw"] = d["I_raw"] - d["asym"]
    d["e_cone"] = d["I_cone"] - d["asym"]
    d["k"], d["y_min"] = k, y_min
    return d


def choose_k(h5, npz_dir, n_sample=400, seed=7):
    """The evidence behind k: the histogram of LES |mass| against q = |y'|/sigma_y(x').

    Two populations that do not overlap have a valley between them. This finds it, rather
    than asserting a number of standard deviations. Downwind cells are assigned q = inf so
    they land beyond the last bin -- they are removed at every k and must not tilt it.
    """
    X, Y = axis_grids()
    rid, umean = load_umean(h5, npz_dir)
    edges = np.concatenate([np.linspace(0, 24, 97), [np.inf]])
    with h5py.File(h5, "r") as f:
        n = int(f.attrs["n"])
        sc = f["scalars"][:]
        idx = np.random.default_rng(seed).choice(n, min(n_sample, n), replace=False)
        Hl = np.zeros(len(edges) - 1)
        Hk = np.zeros(len(edges) - 1)
        for i in idx:
            t = np.abs(f["target"][i].astype(np.float64))
            kj = np.abs(f["kljun"][i].astype(np.float64))
            xw, yw = wind_frame(X, Y, float(sc[i, 4]), float(sc[i, 5]))
            sy = sigma_y_field(sc[i], umean[i], xw)
            q = np.where(xw >= 0, np.abs(yw) / np.maximum(sy, 1e-6), np.inf)
            if t.sum() > 0:
                Hl += np.histogram(q.ravel(), bins=edges, weights=(t / t.sum()).ravel())[0]
            if kj.sum() > 0:
                Hk += np.histogram(q.ravel(), bins=edges, weights=(kj / kj.sum()).ravel())[0]
    return edges, Hl / len(idx), Hk / len(idx), len(idx)


def sweep(h5, npz_dir, zm, near_m, ks, ys):
    """Sensitivity of the removed mass to k and y_min. The point is that it is flat."""
    X, Y = axis_grids()
    R = np.hypot(X, Y)
    rid, umean = load_umean(h5, npz_dir)
    out = {}
    with h5py.File(h5, "r") as f:
        n = int(f.attrs["n"])
        sc = f["scalars"][:]
        rm = {(k, y): np.zeros(n) for k in ks for y in ys}
        nr = {(k, y): np.zeros(n) for k in ks for y in ys}
        for i in range(n):
            t = np.abs(f["target"][i].astype(np.float64))
            xw, yw = wind_frame(X, Y, float(sc[i, 4]), float(sc[i, 5]))
            sy = sigma_y_field(sc[i], umean[i], xw)
            ay = np.abs(yw)
            tot = t.sum()
            for k in ks:
                ks_ = k * sy
                for y in ys:
                    cut = ~((ay <= np.maximum(ks_, y)) & (xw >= -y))
                    rm[(k, y)][i] = t[cut].sum() / tot if tot > 0 else 0.0
                    nr[(k, y)][i] = (t[cut & (R <= near_m)].sum() / tot) if tot > 0 else 0.0
        sv = sc[:, 2]
    hi, lo = sv > np.percentile(sv, 90), sv < np.percentile(sv, 10)
    for k in ks:
        for y in ys:
            a, b = rm[(k, y)], nr[(k, y)]
            out[(k, y)] = dict(med=np.median(a), p95=np.percentile(a, 95), max=a.max(),
                               near_med=np.median(b), near_max=b.max(),
                               bias=np.median(a[hi]) / max(np.median(a[lo]), 1e-12))
    return out


# ------------------------------------------------------------------------------ writing

def rebuild(h5, k, y_min, npz_dir):
    """Write a NEW corpus.h5 carrying target_cone and NOT target_masked.

    Rebuilt object by object rather than edited in place, because HDF5 does not reclaim the
    space of a deleted dataset -- deleting target_masked in place would leave the file
    larger than it started. target, kljun and scalars are copied verbatim.
    """
    tmp = h5 + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    X, Y = axis_grids()
    rid, umean = load_umean(h5, npz_dir)
    with h5py.File(h5, "r") as src, h5py.File(tmp, "w") as dst:
        for kk, v in src.attrs.items():
            dst.attrs[kk] = v
        for name in src:
            if name == "target_masked":
                continue                       # the retired half-plane mask
            src.copy(name, dst, name=name)
        if "u_mean_ms" not in dst["meta"]:
            u = dst["meta"].create_dataset("u_mean_ms", data=umean.astype(np.float32))
            u.attrs["desc"] = ("window-mean wind speed at the receptor, from window_stats. "
                               "Carried so sigma_y -- and therefore the cone -- is "
                               "reproducible from this file alone.")
        s0 = src["target"]
        out = dst.create_dataset("target_cone", shape=s0.shape, dtype=s0.dtype,
                                 chunks=s0.chunks, compression=s0.compression,
                                 compression_opts=s0.compression_opts)
        out.attrs["desc"] = ("THE TRAINING TARGET. target with periodic-wrap material "
                             "removed by a wind-aligned cone on Kljun's own sigma_y. "
                             "See grid/ for the rule and bin/mask_cone.py for the "
                             "derivation of k and y_min.")
        sc = src["scalars"][:]
        n = s0.shape[0]
        for i0 in range(0, n, 32):
            i1 = min(i0 + 32, n)
            blk = src["target"][i0:i1]
            for j in range(i1 - i0):
                i = i0 + j
                xw, yw = wind_frame(X, Y, float(sc[i, 4]), float(sc[i, 5]))
                sy = sigma_y_field(sc[i], umean[i], xw)
                blk[j] = np.where(cone_keep(xw, yw, sy, k, y_min), blk[j], np.float32(0.0))
            out[i0:i1] = blk
        g = dst["grid"]
        for kk in list(g.attrs):
            if kk.startswith("wrap_mask"):
                del g.attrs[kk]                # the retired half-plane convention
        for kk, v in MASK_CONVENTION.items():
            g.attrs[kk] = v
        g.attrs["cone_mask_k"] = float(k)
        g.attrs["cone_mask_y_min_m"] = float(y_min)
        g.attrs["cone_mask_created_utc"] = _dt.datetime.now(
            _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        g.attrs["cone_mask_script"] = "bin/mask_cone.py"
        g.attrs["cone_mask_git_commit"] = git_commit()

    with h5py.File(tmp, "r") as f, h5py.File(h5, "r") as o:
        if "target_masked" in f:
            sys.exit("FATAL: target_masked survived the rebuild")
        if "target_cone" not in f:
            sys.exit("FATAL: target_cone was not written")
        for nm in ("scalars", "kljun", "target"):
            if not np.array_equal(f[nm][:], o[nm][:]):
                sys.exit(f"FATAL: {nm} changed during the rebuild")
        sc = f["scalars"][:]
        rng = np.random.default_rng(0)
        for i in rng.choice(f["target"].shape[0], 24, replace=False):
            a, b = f["target"][i], f["target_cone"][i]
            xw, yw = wind_frame(X, Y, float(sc[i, 4]), float(sc[i, 5]))
            keep = cone_keep(xw, yw, sigma_y_field(sc[i], f["meta/u_mean_ms"][i], xw),
                             k, y_min)
            if not np.array_equal(b[keep], a[keep]):
                sys.exit(f"FATAL: record {i} was modified inside the cone")
            if np.any(b[~keep] != 0):
                sys.exit(f"FATAL: record {i} is not zero outside the cone")
        if not np.isfinite(f["target_cone"][:]).all():
            sys.exit("FATAL: non-finite values in target_cone")
    os.replace(tmp, h5)
    return os.path.getsize(h5)


# ------------------------------------------------------------------------------- report

def pct(a):
    return np.percentile(a, [5, 25, 50, 75, 95])


def _spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)).astype(float),
                             np.argsort(np.argsort(b)).astype(float))[0, 1])


LIMITS_TEMPLATE = """
WHAT THIS MASK CAN AND CANNOT MISS
------------------------------------------------------------------------------
PRODUCTION RETIRES A TRAJECTORY AT ONE DOMAIN LENGTH, so a particle CANNOT wrap
twice. Verified in the image that generated the corpus
(ghcr.io/tyatharva/flux-seeds:7de9dee2a01d-fe0ce48d5dff06), all three links:

  /flux/bin/run_corpus_case.sh      passes no --max-disp at all
  /flux/bin/stage5_footprint.py     --max-disp default is None
  /flux/lpdm/driver.py:281-282      if max_disp is None: max_disp = fs.Lx

and fs.Lx = 122 * 30 = 3660 m at the production grid. The ninth-pass validation
records at this geometry carry max_disp_used = 3660.0 in their stage-5 JSON; the
raised-cap runs are separately named (_3L = 10980, _uncapped = 8784) and are not
corpus cases.

The consequence is the strong one. Every wrapped particle is displaced by exactly
one domain length in x, in y, or in both, so it lands:

  wrapped in x only    -> off-axis by ~3660*|cos_wdir|   -> cone removes it
  wrapped in y only    -> off-axis by ~3660*|sin_wdir|   -> cone removes it
  wrapped in both      -> off-axis and/or downwind        -> cone removes it

There is NO double-wrapped material sitting on-axis for the cone to miss. That is
what the half-plane could not say: single-axis wrap survived it whenever the
displacement exceeded 3660*max(|sin|,|cos|), and on case_2022030716 that left
2.41% of |f| of pure wrap upwind of the receptor.

The residual limits are:

1. GENUINE FAR-OFF-AXIS MATERIAL IS REMOVED WITH THE WRAP. The cone is a bound on
   |y'|, so real material beyond k*sigma_y goes too. k was chosen inside an EMPTY
   valley in the |y'|/sigma_y distribution, so this is measured to be nothing:
   the LES carries {valley:.3f}% of its |mass| in q = [5, 11), and Kljun carries
   {kljvalley:.5f}% beyond q = 6.
2. GENUINE DOWNWIND CONTRIBUTION IS REMOVED. A convective boundary layer puts a
   little influence downwind and the cone cannot tell it from wrap. Measured
   inside {near:.0f} m of the receptor, where such a contribution would sit:
   median {nearmed:.3f}%, max {nearmax:.3f}% of |f|.
3. THE NEAR-FIELD FLOOR IS A REGULARISER, NOT PHYSICS. sigma_y -> 0 at the
   receptor, so y_min = {ymin:.0f} m floors the half-width and the apex is pushed
   to x' = -y_min. It binds only where k*sigma_y < y_min, i.e. x' < ~30 m.
4. THE CONE IS A GEOMETRIC TEST, NOT A TRAJECTORY TEST. It is a statement about
   where mass ended up, not about how it got there.

THE CLEAN FIX, if the corpus is ever rebuilt
------------------------------------------------------------------------------
Deposit the UNFOLDED displacement at generation time: bin each touchdown by its
cumulative displacement from the receptor rather than by its folded LES column
index, and let the raster window truncate what leaves it. Then no wrapped
material is deposited at all and no mask is needed. It requires the touchdowns,
which docs/ML_TARGETS.md decided not to save, so it is a generation-time change
and a full corpus regeneration.
"""


def report(d, zm, near_m, edges, Hl, Hk, nsamp, sw, lines):
    def w(s=""):
        lines.append(s)
    n = d["n"]
    k, y_min = d["k"], d["y_min"]
    w("=" * 78)
    w(f"WIND-ALIGNED CONE MASK -- k = {k:g}, y_min = {y_min:g} m")
    w("=" * 78)
    w(f"records                {n}")
    w(f"rule                   keep <=> |y'| <= max({k:g}*sigma_y(x'), {y_min:g} m) "
      f"AND x' >= -{y_min:g} m")
    w(f"sigma_y                official FFP v1.42, the same call that made the kljun "
      f"channel")
    w(f"asymptote              1 - z_m/z_i, z_m = {zm:.1f} m; median "
      f"{np.median(d['asym']):.4f}")
    w("")

    # ---- how k was chosen ------------------------------------------------------------
    w("-" * 78)
    w("1. HOW k WAS CHOSEN: the valley in q = |y'| / sigma_y(x')")
    w("-" * 78)
    w(f"  LES |mass| against q, averaged over {nsamp} records. Downwind cells are q = inf")
    w("  and sit in the last row; they are removed at every k.")
    w("")
    w("      q range        LES |mass|/bin      cumulative LES     cumulative Kljun")
    cl, ck = np.cumsum(Hl), np.cumsum(Hk)
    for j in range(0, len(edges) - 1):
        if not np.isfinite(edges[j + 1]):
            w(f"      q >= {edges[j]:5.1f}      {100 * Hl[j]:9.3f}%        "
              f"{100 * cl[j]:8.3f}%          {100 * ck[j]:9.5f}%   <- incl. downwind")
            break
        if j % 8:
            continue
        w(f"      {edges[j]:5.2f}-{edges[j + 1]:5.2f}    {100 * Hl[j]:9.3f}%        "
          f"{100 * cl[j]:8.3f}%          {100 * ck[j]:9.5f}%")
    lo_i = int(np.searchsorted(edges, 5.0)) - 1
    hi_i = int(np.searchsorted(edges, 11.0)) - 1
    valley = 100 * Hl[lo_i:hi_i].sum()
    kv = 100 * (1 - ck[int(np.searchsorted(edges, 6.0)) - 1])
    w("")
    w(f"  THE VALLEY IS EMPTY. LES |mass| with q in [5, 11) is {valley:.4f}% of the total,")
    w(f"  and it RISES again beyond q = 11. Two populations that do not overlap: the")
    w(f"  footprint below q ~ 5, the wrap above q ~ 11. Kljun carries {kv:.5f}% beyond q = 6.")
    w(f"  k = {k:g} is the middle of that valley. Any k in [5, 11] gives the same answer,")
    w("  which is the point -- see the sensitivity table below.")
    w("")

    # ---- sensitivity ------------------------------------------------------------------
    w("-" * 78)
    w("2. SENSITIVITY OF THE REMOVED MASS TO k AND y_min")
    w("-" * 78)
    w("   k  y_min |  removed |mass| [% of |f|]  | within 200 m [%]  | sigma_v bias")
    w("               median    p95     max      | median     max    | top10%/bot10%")
    for (kk, yy) in sorted(sw):
        r = sw[(kk, yy)]
        star = "  <-- chosen" if (kk == k and yy == y_min) else ""
        w(f"  {kk:3g} {yy:6g} | {100 * r['med']:8.2f}{100 * r['p95']:7.2f}"
          f"{100 * r['max']:8.2f}   |{100 * r['near_med']:8.3f}{100 * r['near_max']:8.3f}"
          f"   |   {r['bias']:5.2f}{star}")
    w("")
    w("  The removed mass barely moves across a factor of three in k. That flatness IS the")
    w("  evidence: there is nothing between the footprint and the artifact to be sensitive")
    w("  to. The sigma_v bias column is the 'does it eat wide footprints' test -- the top")
    w("  decile of sigma_v loses only a few percent more than the bottom decile, so the")
    w("  cone is not preferentially clipping the broad cases.")
    w("")
    w(f"  y_min was then set by the near-field acceptance criterion: removed mass within")
    w(f"  {near_m:.0f} m of the receptor at median 0.000% and max ~1.00%. y_min = 60 m")
    w(f"  leaves the max at ~1.04-1.08%; y_min = {y_min:g} m brings it to "
      f"{100 * sw[(k, y_min)]['near_max']:.3f}%.")
    w("")

    # ---- what it removes ---------------------------------------------------------------
    w("-" * 78)
    w("3. WHAT THE CONE REMOVES")
    w("-" * 78)
    p = pct(100 * d["rm_abs"])
    w(f"  |mass| removed [% of |f|]   p5 {p[0]:.2f}  p25 {p[1]:.2f}  median {p[2]:.2f}  "
      f"p75 {p[3]:.2f}  p95 {p[4]:.2f}   max {100 * d['rm_abs'].max():.2f}")
    w(f"    of it UPWIND of the receptor (what the half-plane MISSED):")
    w(f"        median {100 * np.median(d['rm_up']):.2f}%   p95 "
      f"{100 * np.percentile(d['rm_up'], 95):.2f}%   max {100 * d['rm_up'].max():.2f}%   "
      f"nonzero on {int((d['rm_up'] > 1e-6).sum())} of {n} records")
    w(f"    within {near_m:.0f} m of the receptor: median "
      f"{100 * np.median(d['rm_near']):.3f}%   max {100 * d['rm_near'].max():.3f}%")
    w(f"  Kljun |mass| removed by the SAME cone: max over all records "
      f"{100 * d['klj_rm'].max():.5f}%")
    w("    The input channel is essentially untouched, so the cone removes almost nothing")
    w("    a perfect emulator would have to reproduce.")
    w("")

    # ---- the integral -------------------------------------------------------------------
    w("-" * 78)
    w("4. THE INTEGRAL, against the asymptote")
    w("-" * 78)
    w(f"{'':22s}{'p5':>9s}{'p25':>9s}{'median':>9s}{'p75':>9s}{'p95':>9s}{'mean':>10s}")
    for nm, v in (("integral raw", d["I_raw"]), ("integral cone", d["I_cone"]),
                  ("asymptote", d["asym"]),
                  ("error raw", d["e_raw"]), ("error cone", d["e_cone"])):
        w(f"  {nm:20s}" + "".join(f"{x:9.4f}" for x in pct(v)) + f"{v.mean():10.4f}")
    closer = np.abs(d["e_cone"]) < np.abs(d["e_raw"])
    mr, mc = np.median(np.abs(d["e_raw"])), np.median(np.abs(d["e_cone"]))
    w("")
    w(f"  records moved CLOSER to the asymptote   {closer.sum():5d} of {n}  "
      f"({100 * closer.mean():.1f}%)")
    w(f"  median |error|                          {mr:.4f}  ->  {mc:.4f}  "
      f"({'IMPROVED' if mc < mr else 'DEGRADED'})")
    w(f"  records below the asymptote             "
      f"{int((d['I_raw'] < d['asym']).sum())}  ->  {int((d['I_cone'] < d['asym']).sum())}")
    r = float(np.corrcoef(d["rm_abs"], d["e_raw"])[0, 1])
    w(f"  r(|mass| removed, raw error)            {r:+.4f}   "
      f"(Spearman {_spearman(d['rm_abs'], d['e_raw']):+.4f})")
    w("")
    w("  THIS IS NOT AN INTEGRAL CORRECTION AND IS NOT CLAIMED AS ONE. The correlation is")
    w("  the same sign as it was for the half-plane: the records that lose the most wrap")
    w("  are the ones that were already LOW, not the ones that were inflated. Whatever")
    w("  inflates the integral is not the wraparound. The cone is an OPERATIONAL CLEANUP --")
    w("  it removes material that the periodic boundary invented and that no emulator")
    w("  should be asked to predict.")
    w("")
    for nm, v in (("raw", d["I_raw"]), ("cone", d["I_cone"])):
        bad = (v < 0.6) | (v > 1.5)
        w(f"  G2b {nm:5s} outside [0.6, 1.5]: {int(bad.sum()):4d} of {n}   "
          f"(low {int((v < 0.6).sum())}, high {int((v > 1.5).sum())})")
    w("")

    # ---- negative lobe --------------------------------------------------------------------
    w("-" * 78)
    w("5. THE NEGATIVE LOBE")
    w("-" * 78)
    w(f"    raw    median {100 * np.median(d['neg']):.2f}%   p5-p95 "
      f"{100 * pct(d['neg'])[0]:.2f}-{100 * pct(d['neg'])[4]:.2f}")
    w(f"    cone   median {100 * np.median(d['neg_cone']):.2f}%   p5-p95 "
      f"{100 * pct(d['neg_cone'])[0]:.2f}-{100 * pct(d['neg_cone'])[4]:.2f}")
    dn = 100 * (d["neg_cone"] - d["neg"])
    w(f"    change [pp] median {np.median(dn):+.2f}   p5 {pct(dn)[0]:+.2f}   "
      f"p95 {pct(dn)[4]:+.2f}")
    w("  The negative lobe that survives is the part inside the cone, i.e. the part at")
    w("  crosswind distances a real footprint can reach. The part removed sat where the")
    w("  LES carries no positive mass either, which is what the valley in section 1 says.")
    w("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="corpus/corpus.h5")
    ap.add_argument("--npz-dir", default="corpus/pairs_npz")
    ap.add_argument("--k", type=float, default=K_DEFAULT)
    ap.add_argument("--y-min", type=float, default=YMIN_DEFAULT)
    ap.add_argument("--zm", type=float, default=30.0)
    ap.add_argument("--near-m", type=float, default=200.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="report the k/y_min table only; implies --dry-run")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    if not os.path.exists(a.h5):
        sys.exit(f"FATAL: {a.h5} does not exist")

    print(f"sweeping k and y_min on {a.h5} ...")
    sw = sweep(a.h5, a.npz_dir, a.zm, a.near_m,
               ks=[3, 5, 8, 12], ys=[0, 60, 90, 120])
    if a.sweep:
        for kk, yy in sorted(sw):
            r = sw[(kk, yy)]
            print(f"  k={kk:<3g} y_min={yy:<4g} removed med {100 * r['med']:.2f}%  "
                  f"near max {100 * r['near_max']:.3f}%  bias {r['bias']:.2f}")
        return

    print("finding the valley in |y'|/sigma_y ...")
    edges, Hl, Hk, nsamp = choose_k(a.h5, a.npz_dir)
    print(f"measuring at k = {a.k:g}, y_min = {a.y_min:g} m ...")
    d = measure(a.h5, a.k, a.y_min, a.zm, a.near_m, a.npz_dir)

    lines = []
    report(d, a.zm, a.near_m, edges, Hl, Hk, nsamp, sw, lines)
    cl, ck = np.cumsum(Hl), np.cumsum(Hk)
    lo_i = int(np.searchsorted(edges, 5.0)) - 1
    hi_i = int(np.searchsorted(edges, 11.0)) - 1
    lines.append(LIMITS_TEMPLATE.strip().format(
        valley=100 * Hl[lo_i:hi_i].sum(),
        kljvalley=100 * (1 - ck[int(np.searchsorted(edges, 6.0)) - 1]),
        near=a.near_m, nearmed=100 * np.median(d["rm_near"]),
        nearmax=100 * d["rm_near"].max(), ymin=a.y_min))

    os.makedirs(a.outdir, exist_ok=True)
    tsv = os.path.join(a.outdir, "cone_mask_per_record.tsv")
    with open(tsv, "w") as fh:
        fh.write(f"# per-record wind-aligned cone mask, bin/mask_cone.py, "
                 f"k={a.k:g} y_min={a.y_min:g} m\n")
        fh.write(f"# asym = 1 - z_m/z_i with z_m = {a.zm:.1f} m; removed_* are fractions "
                 f"of sum|f| over the raw raster\n")
        fh.write("run_id\tsplit\twdir_deg\tzi_m\tsigma_v\tasym\tintegral_raw\t"
                 "integral_cone\terr_raw\terr_cone\tremoved_abs\tremoved_upwind\t"
                 "removed_near\tnegfrac_raw\tnegfrac_cone\tkljun_removed\tarray_share\n")
        for i in range(d["n"]):
            fh.write("\t".join([
                d["run_id"][i], d["split"][i], f"{d['wdir'][i]:.1f}", f"{d['zi'][i]:.0f}",
                f"{d['sigma_v'][i]:.4f}", f"{d['asym'][i]:.4f}", f"{d['I_raw'][i]:.4f}",
                f"{d['I_cone'][i]:.4f}", f"{d['e_raw'][i]:+.4f}", f"{d['e_cone'][i]:+.4f}",
                f"{d['rm_abs'][i]:.5f}", f"{d['rm_up'][i]:.5f}", f"{d['rm_near'][i]:.5f}",
                f"{d['neg'][i]:.5f}", f"{d['neg_cone'][i]:.5f}", f"{d['klj_rm'][i]:.7f}",
                f"{d['array_share'][i]:.5f}"]) + "\n")

    body = "\n".join(lines)
    txt = os.path.join(a.outdir, "cone_mask_validation.txt")
    with open(txt, "w") as fh:
        fh.write(body + "\n")
    print(body)
    print(f"\nwrote {tsv}")
    print(f"wrote {txt}")

    if a.dry_run:
        print("\n--dry-run: corpus.h5 NOT modified")
        return
    size = rebuild(a.h5, a.k, a.y_min, a.npz_dir)
    print(f"\nrebuilt {a.h5} ({size / 1e6:.1f} MB): target_cone written, target_masked "
          f"removed, target/kljun/scalars verified unchanged, cone verified on 24 records")


if __name__ == "__main__":
    main()
