"""Footprint metrics on the 128^2 raster, by the PRODUCTION functions:

  peak_x        bin/fig_corpus_pairs.crosswind_integrated (conservative binning on the wind
                axis) -> bin/stage5_footprint.fy_metrics (argmax of a 5-cell running mean),
                the estimator behind meta/peak_x_m.
  centroid,     lpdm.footprint.FootprintGrid.metrics_map on a grid built from the raster's
  area80, peak  own cell edges (signed first moment; 80% area of the raw field).
  overlap80     lpdm.footprint.source_area_overlap(max(a,0), max(b,0)), Jaccard, exactly as
                every production caller passes it.
  array_share   the raster form of lpdm/driver.py:555 cover_share (signed, unrenormalised).
  integral      sum(f) * 900 m^2; also the departure from 1 - z_m/z_i.

Nothing here reimplements a production formula; the same function is applied to the LES
target, to Kljun and to the emulator, so a comparison between the three is a comparison of
fields, never of estimators.
"""
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lpdm.footprint import FootprintGrid, source_area_overlap        # noqa: E402
import fig_corpus_pairs as FCP                                       # noqa: E402
import stage5_footprint as S5                                        # noqa: E402
from ml import data as D                                             # noqa: E402

CELL_AREA = D.DX * D.DX
METRIC_KEYS = ("peak_x", "centroid", "overlap80", "array_share", "integral")


def _grid():
    xe = (np.arange(D.N + 1) - D.IJ_RECEPTOR - 0.5) * D.DX
    return FootprintGrid.from_edges(xe, xe)


_G = None


def map_metrics(f):
    """FootprintGrid.metrics_map on a raster in m^-2: assign f*area as the accumulated flux
    with one particle, so normalised() returns f itself."""
    global _G
    if _G is None:
        _G = _grid()
    _G.flux = np.asarray(f, dtype=np.float64) * _G.area
    _G.n_particles = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _G.metrics_map()


def wind_axis_metrics(f, wdir_deg):
    s, fy = FCP.crosswind_integrated(np.asarray(f, dtype=np.float64), float(wdir_deg))
    return S5.fy_metrics(s, fy, D.DX)


def record_metrics(f, wdir_deg, array_mask, asymptote):
    """All single-field quantities for one raster."""
    m = map_metrics(f)
    w = wind_axis_metrics(f, wdir_deg)
    f = np.asarray(f, dtype=np.float64)
    integ = float(f.sum() * CELL_AREA)
    tot = f.sum()
    share = float((f * array_mask).sum() / tot) if tot != 0 else np.nan
    # metrics_map returns a shorter dict (no peak_value / area50_ha) when the field's
    # total is <= 0 -- a degenerate prediction. Carry NaN rather than raise, so one bad
    # record scores as NaN and the run is still summarised.
    g = lambda k: float(m.get(k, np.nan))
    return dict(peak_x=w["peak_x"], x80=w["x80"], mean_x=w["mean_x"],
                centroid_e=g("centroid_e"), centroid_n=g("centroid_n"),
                centroid_dist=g("centroid_dist"), area80_ha=g("area80_ha"),
                peak_e=g("peak_e"), peak_n=g("peak_n"), peak_value=g("peak_value"),
                array_share=share, integral=integ, integral_asym_err=integ - asymptote,
                neg_frac=FCP.negative_fraction(f), degenerate=bool(m.get("degenerate", False)))


def pair_errors(f, ref, wdir_deg, array_mask, asymptote, mf=None, mr=None):
    """Errors of raster f against reference raster ref (the LES target)."""
    mf = record_metrics(f, wdir_deg, array_mask, asymptote) if mf is None else mf
    mr = record_metrics(ref, wdir_deg, array_mask, asymptote) if mr is None else mr
    ov = source_area_overlap(np.maximum(np.asarray(f, np.float64), 0),
                             np.maximum(np.asarray(ref, np.float64), 0), 0.80)
    return dict(
        peak_x=abs(mf["peak_x"] - mr["peak_x"]),
        centroid=float(np.hypot(mf["centroid_e"] - mr["centroid_e"],
                                mf["centroid_n"] - mr["centroid_n"])),
        overlap80=ov,
        array_share=abs(mf["array_share"] - mr["array_share"]) * 100.0,     # pp
        array_share_signed=(mf["array_share"] - mr["array_share"]) * 100.0,
        integral=abs(mf["integral"] - mr["integral"]),
        integral_signed=mf["integral"] - mr["integral"],
        integral_asym=abs(mf["integral_asym_err"]),
        x80=abs(mf["x80"] - mr["x80"]),
        area80_ratio=(mf["area80_ha"] / mr["area80_ha"]) if mr["area80_ha"] > 0 else np.nan,
        peak_xy=float(np.hypot(mf["peak_e"] - mr["peak_e"], mf["peak_n"] - mr["peak_n"])),
    )


def score_fields(fields, les, wdir_deg, array_mask, asymptote):
    """fields: dict name -> (n,128,128). Returns dict name -> dict key -> (n,) arrays, plus
    'les' -> record metrics of the reference."""
    n = les.shape[0]
    ref = [record_metrics(les[i], wdir_deg[i], array_mask, asymptote[i]) for i in range(n)]
    out = {"les": {k: np.array([r[k] for r in ref]) for k in ref[0]}}
    for name, arr in fields.items():
        rows = []
        for i in range(n):
            mf = record_metrics(arr[i], wdir_deg[i], array_mask, asymptote[i])
            e = pair_errors(arr[i], les[i], wdir_deg[i], array_mask, asymptote[i], mf, ref[i])
            e.update({"abs_" + k: v for k, v in mf.items()})
            rows.append(e)
        out[name] = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    return out


def error_of(scores, key):
    """The 'smaller is better' error for a metric key; overlap becomes 1 - Jaccard."""
    if key == "overlap80":
        return 1.0 - scores["overlap80"]
    return scores[key]


def composite(scores_model, scores_ref, mask=None, keys=METRIC_KEYS):
    """Geometric mean over metrics of median|err_model| / median|err_ref|; < 1 beats ref."""
    ratios = {}
    for k in keys:
        a = error_of(scores_model, k)
        b = error_of(scores_ref, k)
        if mask is not None:
            a, b = a[mask], b[mask]
        ma, mb = np.nanmedian(a), np.nanmedian(b)
        ratios[k] = float(ma / mb) if mb > 0 else (1.0 if ma == 0 else np.inf)
    vals = np.array([v for v in ratios.values() if np.isfinite(v) and v > 0])
    return float(np.exp(np.mean(np.log(vals)))) if len(vals) else np.nan, ratios


def summarise(scores, mask=None):
    """Median / mean of each error for one field, optionally on a subset."""
    out = {}
    for k in METRIC_KEYS + ("x80", "peak_xy", "area80_ratio", "integral_asym"):
        v = scores[k] if mask is None else scores[k][mask]
        out[k] = dict(median=float(np.nanmedian(v)), mean=float(np.nanmean(v)),
                      n=int(np.isfinite(v).sum()))
    return out
