"""Connected-component filter for footprint rasters. A footprint is one region upwind of
the receptor; isolated blobs are artifacts that per-cell metrics cannot see because they
sit at 1e-8. Two rules, neither with an absolute threshold:

  rule A  the 99.9% rule: support = the smallest set of cells carrying 99.9% of |mass|;
          8-connected components of it, kept in descending |mass| until 99.9% of the
          ORIGINAL |mass| is accounted for; everything else zeroed.
  rule B  the LES-connectivity level: the smallest level tau (fraction of max|f|) at which
          {|f| >= tau max|f|} is a single component, measured on the LES targets; the val
          median tau* is applied to every field, keeping the peak's component.

Applied identically to Kljun, the FNO, the CFM mean and every CFM sample. Mass removed is
reported per record; the LES target itself is scored unfiltered.
"""
import numpy as np
from scipy import ndimage

EIGHT = np.ones((3, 3), dtype=int)


def components(support):
    lab, n = ndimage.label(support, structure=EIGHT)
    return lab, n


def filter_mass(f, frac=0.999):
    """Rule A. Returns (filtered, info)."""
    f = np.asarray(f, dtype=np.float64)
    a = np.abs(f)
    tot = a.sum()
    info = dict(mass_removed_frac=0.0, tail_frac=0.0, n_components=0, n_kept=0,
                peak_kept=True, total_abs=float(tot))
    if tot <= 0:
        return f.copy(), info
    flat = a.ravel()
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order])
    k = int(np.searchsorted(cum, frac * tot)) + 1
    support = np.zeros(a.size, bool)
    support[order[:k]] = True
    support = support.reshape(a.shape)
    lab, n = components(support)
    masses = ndimage.sum(a, lab, index=np.arange(1, n + 1))
    idx = np.argsort(masses)[::-1]
    keep_ids, acc = [], 0.0
    for j in idx:
        keep_ids.append(j + 1)
        acc += masses[j]
        if acc >= frac * tot:
            break
    keep = np.isin(lab, keep_ids)
    out = np.where(keep, f, 0.0)
    pk = np.unravel_index(int(np.argmax(a)), a.shape)
    info.update(mass_removed_frac=float(1 - np.abs(out).sum() / tot),
                tail_frac=float(1 - cum[k - 1] / tot), n_components=int(n),
                n_kept=len(keep_ids), peak_kept=bool(keep[pk]))
    return out, info


def connectivity_level(f, levels=None):
    """Rule B measurement: the smallest tau (fraction of max|f|) at which the superlevel
    set is a single 8-connected component. Scans tau downward from 1; returns the last
    single-component level (1.0 if it never splits, nan if the field is empty)."""
    a = np.abs(np.asarray(f, dtype=np.float64))
    mx = a.max()
    if mx <= 0:
        return np.nan
    if levels is None:
        levels = np.logspace(0, -7, 71)
    last = 1.0
    for tau in levels:
        _, n = components(a >= tau * mx)
        if n > 1:
            break
        last = float(tau)
    return last


def filter_level(f, tau):
    """Rule B application: keep the peak's component of {|f| >= tau max|f|}."""
    f = np.asarray(f, dtype=np.float64)
    a = np.abs(f)
    tot = a.sum()
    if tot <= 0:
        return f.copy(), dict(mass_removed_frac=0.0, n_components=0)
    sup = a >= tau * a.max()
    lab, n = components(sup)
    pk = np.unravel_index(int(np.argmax(a)), a.shape)
    out = np.where(lab == lab[pk], f, 0.0)
    return out, dict(mass_removed_frac=float(1 - np.abs(out).sum() / tot), n_components=int(n))


def filter_stack(fields, rule="A", tau=None, frac=0.999):
    """fields (n,H,W) -> (filtered (n,H,W), per-record info dict of arrays)."""
    outs, infos = [], []
    for f in fields:
        if rule == "A":
            o, i = filter_mass(f, frac)
        elif rule == "B":
            o, i = filter_level(f, tau)
        else:
            raise ValueError(rule)
        outs.append(o.astype(np.float32))
        infos.append(i)
    keys = infos[0].keys()
    return np.stack(outs), {k: np.array([i[k] for i in infos]) for k in keys}
