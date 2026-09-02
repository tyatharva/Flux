"""The tail speckle: is it signal or sampling noise, and the per-record source-area
threshold that removes it from the LES TARGETS before training.

  threshold_sa(f, frac=0.99): zero every cell below the record's own 99% source-area level
  (the level such that cells with f >= level carry 99% of the positive sum). Every negative
  cell goes with it. No absolute number anywhere; the same rule applies to Kljun.

  --pair    the coherence measurement on the two windows of case_2023111718 (n = 1 pair):
            are the tail cells in the same places in both realisations? Jaccard of the tail
            supports against a shell-conditioned null, residual correlation of the values,
            sign agreement, with the body as the positive control; the realisation floor
            before and after thresholding.
  --corpus  mass removed by the rule on the train/val LES targets and on val Kljun.

The test split is never read (ml.data refuses it; nothing here asks).
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml import data as D                  # noqa: E402

PAIR_DIR = os.path.join(REPO, "validation_pairs_30m")
OUT_DIR = os.path.join(REPO, "results", "ml_cfm", "tail")
LEVELS = (0.90, 0.95, 0.99, 0.999)


# ------------------------------------------------------------------ the rule

def source_area_level(f, frac=0.99):
    """Level such that {f >= level} carries `frac` of the positive sum (0 if no positive mass)."""
    v = np.sort(np.maximum(np.asarray(f, np.float64), 0).ravel())[::-1]
    tot = v.sum()
    if tot <= 0:
        return 0.0
    cum = np.cumsum(v) / tot
    return float(v[min(int(np.searchsorted(cum, frac)), len(v) - 1)])


def threshold_sa(f, frac=0.99):
    """Zero every cell with f < level(frac). Returns (f', info)."""
    f = np.asarray(f, np.float64)
    lev = source_area_level(f, frac)
    keep = f >= lev if lev > 0 else np.zeros_like(f, bool)
    g = np.where(keep, f, 0.0)
    a = np.abs(f).sum()
    pos = np.maximum(f, 0).sum()
    neg = -np.minimum(f, 0).sum()
    info = dict(level=lev, level_over_peak=float(lev / f.max()) if f.max() > 0 else np.nan,
                mass_removed_frac=float((a - np.abs(g).sum()) / a) if a > 0 else 0.0,
                pos_mass_removed_frac=float((pos - g.sum()) / pos) if pos > 0 else 0.0,
                neg_mass_frac=float(neg / a) if a > 0 else 0.0,
                n_nonzero=int((f != 0).sum()), n_kept=int(keep.sum()))
    return g.astype(np.float32), info


def threshold_stack(fields, frac=0.99):
    out = np.empty_like(fields, dtype=np.float32)
    infos = []
    for i in range(fields.shape[0]):
        out[i], inf = threshold_sa(fields[i], frac)
        infos.append(inf)
    return out, {k: np.array([i[k] for i in infos], float) for k in infos[0]}


# ------------------------------------------------------------------ the pair

def _load_pair():
    import mask_cone as mc
    ws = []
    for w in (0, 1):
        z = np.load(os.path.join(PAIR_DIR, f"case_2023111718_w{w}.npz"), allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        ws.append(dict(target=z["target"].astype(np.float64), kljun=z["kljun"].astype(np.float64),
                       scalars=z["scalars"].astype(np.float64), meta=meta))
    g = D.read_grid_attrs()
    sc = ws[0]["scalars"]
    keep = D._cone_one((sc, float(ws[0]["meta"]["u_mean_ms"]), float(g["cone_mask_k"]),
                        float(g["cone_mask_y_min_m"]), float(g["cone_mask_x_min_m"])))
    valid = np.zeros((D.N, D.N), bool)
    valid[D.PAD:D.PAD + D.NG, D.PAD:D.PAD + D.NG] = True
    keep &= valid
    X, Y = mc.axis_grids()
    xw, yw = mc.wind_frame(X, Y, float(sc[4]), float(sc[5]))
    for w in ws:
        w["target"] = w["target"] * keep          # the corpus target is cone-cropped
        w["kljun"] = w["kljun"] * keep
    return ws, keep, xw, np.abs(yw)


def _bins(keep, xw, ayw, nx=10, ny=5):
    """Shell index per cone cell: nx quantile bins of x' times ny quantile bins of |y'|."""
    x, y = xw[keep], ayw[keep]
    xe = np.quantile(x, np.linspace(0, 1, nx + 1)[1:-1])
    ye = np.quantile(y, np.linspace(0, 1, ny + 1)[1:-1])
    return np.searchsorted(xe, x) * ny + np.searchsorted(ye, y)


def _jaccard(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def _null_jaccard(a, b, shell, rng, n_perm=200):
    """Expected Jaccard when each window's band is placed independently with the SAME
    per-shell occupancy: analytic expectation and a permutation band (cells shuffled
    within shells)."""
    ids = np.unique(shell)
    exp_over = 0.0
    for s in ids:
        m = shell == s
        n = m.sum()
        exp_over += n * (a[m].mean()) * (b[m].mean())
    j_exp = exp_over / (a.sum() + b.sum() - exp_over)
    js = []
    for _ in range(n_perm):
        bp = b.copy()
        for s in ids:
            m = np.where(shell == s)[0]
            bp[m] = b[m][rng.permutation(len(m))]
        js.append(_jaccard(a, bp))
    return float(j_exp), float(np.mean(js)), float(np.percentile(js, 2.5)), float(np.percentile(js, 97.5))


def _residual_corr(v0, v1, shell, m):
    """Pearson/Spearman of the two windows on cells m after removing each shell's mean."""
    from scipy.stats import pearsonr, spearmanr
    r0, r1 = v0[m].copy(), v1[m].copy()
    sh = shell[m]
    for s in np.unique(sh):
        k = sh == s
        r0[k] -= r0[k].mean()
        r1[k] -= r1[k].mean()
    if len(r0) < 10 or r0.std() == 0 or r1.std() == 0:
        return np.nan, np.nan, int(m.sum())
    return float(pearsonr(r0, r1)[0]), float(spearmanr(r0, r1)[0]), int(m.sum())


def coherence(ws, keep, xw, ayw, levels=LEVELS, seed=0):
    rng = np.random.default_rng(seed)
    shell = _bins(keep, xw, ayw)
    v0, v1 = ws[0]["target"][keep], ws[1]["target"][keep]
    out = {}
    for frac in levels:
        l0, l1 = source_area_level(ws[0]["target"], frac), source_area_level(ws[1]["target"], frac)
        tail0, tail1 = (v0 != 0) & (v0 < l0), (v1 != 0) & (v1 < l1)
        body0, body1 = v0 >= l0, v1 >= l1
        rec = {}
        for name, a, b in (("tail", tail0, tail1), ("body", body0, body1)):
            j = _jaccard(a, b)
            j_exp, j_perm, lo, hi = _null_jaccard(a, b, shell, rng)
            union = a | b
            rp, rs, n = _residual_corr(v0, v1, shell, union)
            rpi, rsi, ni = _residual_corr(v0, v1, shell, a & b)
            neg0 = (v0 < 0) & union
            rec[name] = dict(
                n0=int(a.sum()), n1=int(b.sum()), n_union=int(union.sum()),
                jaccard=j, jaccard_null_expected=j_exp, jaccard_null_perm_mean=j_perm,
                jaccard_null_perm_ci95=[lo, hi], jaccard_over_null=float(j / j_exp) if j_exp > 0 else np.nan,
                resid_pearson=rp, resid_spearman=rs,
                resid_pearson_intersection=rpi, resid_spearman_intersection=rsi, n_intersection=ni,
                mass_frac_w0=float(np.abs(v0[a]).sum() / np.abs(v0).sum()),
                mass_frac_w1=float(np.abs(v1[b]).sum() / np.abs(v1).sum()),
                sign_agree_neg=(float((v1[neg0] < 0).mean()) if neg0.sum() else np.nan),
                sign_agree_null=(float((v1[union] < 0).mean()) if union.sum() else np.nan),
                n_neg_w0=int(neg0.sum()))
        rec["level_w0"], rec["level_w1"] = l0, l1
        rec["level_over_peak"] = [float(l0 / ws[0]["target"].max()), float(l1 / ws[1]["target"].max())]
        out[f"{frac:g}"] = rec
    return out


def verdict(c, key="0.99"):
    t = c[key]["tail"]
    j, r = t["jaccard_over_null"], t["resid_pearson"]
    # coherence is POSITIVE correlation; a negative residual r on the union is the
    # structural artifact of cells that are tail in one window and body in the other
    if np.isfinite(j) and j <= 1.5 and r < 0.1:
        return "noise"
    if (np.isfinite(j) and j >= 3.0) or r >= 0.3:
        return "coherent"
    return "mixed"


def pair_floors(ws, arr, frac=0.99):
    from ml import metrics as M
    wd = float(ws[0]["meta"]["wdir_deg"])
    asym = 1.0 - D.Z_RECEPTOR / float(ws[0]["scalars"][0])
    keys = ("rel_l2", "shape_l1_2d", "overlap80", "array_share", "centroid", "integral", "peak_x", "shape_1d")
    raw = M.pair_errors(ws[1]["target"], ws[0]["target"], wd, arr, asym)
    t0, i0 = threshold_sa(ws[0]["target"], frac)
    t1, i1 = threshold_sa(ws[1]["target"], frac)
    thr = M.pair_errors(t1.astype(np.float64), t0.astype(np.float64), wd, arr, asym)
    kr = [M.pair_errors(ws[w]["kljun"], ws[w]["target"], wd, arr, asym) for w in (0, 1)]
    kt = []
    for w, tw in ((0, t0), (1, t1)):
        kk, _ = threshold_sa(ws[w]["kljun"], frac)
        kt.append(M.pair_errors(kk.astype(np.float64), tw.astype(np.float64), wd, arr, asym))
    pick = lambda e: {k: float(e[k]) for k in keys}
    return dict(w1_vs_w0_raw=pick(raw), w1_vs_w0_thresholded=pick(thr),
                kljun_vs_w_raw=[pick(e) for e in kr], kljun_thr_vs_w_thr=[pick(e) for e in kt],
                threshold_info=dict(w0=i0, w1=i1))


def pair_figure(ws, keep, c, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fig_corpus_pairs as FCP
    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    l = [source_area_level(w["target"], 0.99) for w in ws]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    lognorm, _, _ = FCP.pair_norms(ws[0]["kljun"], np.maximum(ws[0]["target"], ws[1]["target"]))
    for w in (0, 1):
        FCP.raster(ax[0][w], ws[w]["target"], lognorm, "magma", ext, mask_below=lognorm.vmin)
        ax[0][w].contour(xc, xc, ws[w]["target"], levels=[l[w]], colors="w", linewidths=0.7)
        ax[0][w].set_title(f"window {w}: LES target (cone), white = its 99% source-area level", fontsize=8)
        tail = (ws[w]["target"] != 0) & (ws[w]["target"] < l[w])
        img = np.where(tail, np.sign(ws[w]["target"]), np.nan)
        ax[1][w].imshow(img, origin="lower", extent=ext, cmap="bwr", vmin=-1, vmax=1)
        ax[1][w].set_title(f"window {w}: tail band (below the level): red +, blue -", fontsize=8)
    t0 = (ws[0]["target"] != 0) & (ws[0]["target"] < l[0])
    t1 = (ws[1]["target"] != 0) & (ws[1]["target"] < l[1])
    both = np.full(t0.shape, np.nan)
    both[t0 & ~t1] = 0; both[~t0 & t1] = 1; both[t0 & t1] = 2
    ax[0][2].imshow(both, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=2)
    tt = c["0.99"]["tail"]
    ax[0][2].set_title(f"tail bands: w0 only / w1 only / both (yellow)\nJaccard {tt['jaccard']:.3f} vs null "
                       f"{tt['jaccard_null_expected']:.3f} (x{tt['jaccard_over_null']:.2f}); resid r {tt['resid_pearson']:.2f}", fontsize=8)
    levs = list(c.keys())
    ax[1][2].plot(levs, [c[k]["tail"]["jaccard_over_null"] for k in levs], "o-", label="tail J / null")
    ax[1][2].plot(levs, [c[k]["body"]["jaccard_over_null"] for k in levs], "s-", label="body J / null (control)")
    ax[1][2].plot(levs, [c[k]["tail"]["resid_pearson"] for k in levs], "o--", label="tail residual r")
    ax[1][2].plot(levs, [c[k]["body"]["resid_pearson"] for k in levs], "s--", label="body residual r")
    ax[1][2].axhline(1, color="k", lw=0.6); ax[1][2].set_xlabel("source-area fraction defining the level")
    ax[1][2].legend(fontsize=7, frameon=False)
    for a in ax.ravel()[:5]:
        a.set_xlim(-1200, 1200); a.set_ylim(-1200, 1200); a.tick_params(labelsize=6)
    fig.suptitle("case_2023111718, two 30-min windows of one LES run: are the tail cells in the same places?", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=110); plt.close(fig)


def run_pair(outdir, frac=0.99):
    os.makedirs(outdir, exist_ok=True)
    ws, keep, xw, ayw = _load_pair()
    st = D.load_statics()
    arr = st["array"] > 0.5
    c = coherence(ws, keep, xw, ayw)
    v = verdict(c)
    floors = pair_floors(ws, arr, frac)
    res = dict(case="case_2023111718", split="train", n_pairs=1, wdir_deg=ws[0]["meta"]["wdir_deg"],
               cone_cells=int(keep.sum()), coherence=c, verdict_rule=(
                   "noise if tail J/null <= 1.5 and resid r < 0.1 (signed); coherent if J/null >= 3 or r >= 0.3; else mixed"),
               verdict=v, floors=floors)
    with open(os.path.join(outdir, "coherence.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    L = ["# Tail coherence on the two-window pair (case_2023111718, wdir 335 deg, a TRAIN record; n = 1 pair)", "",
         f"Cone cells {keep.sum()}. Tail band = cone cells with f != 0 below the window's own source-area level; "
         "body = cells at or above it. Null Jaccard: each window's band placed independently with the same "
         "occupancy per shell (10 x' quantile bins x 5 |y'| bins); residual r: after removing shell means.", "",
         "| level | part | n w0 | n w1 | |mass| frac w0 | Jaccard | null (expected / perm 95%) | J / null | resid Pearson (union) | resid Spearman (union) | resid Pearson (both-tail cells, n) | neg sign agree (null) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, rec in c.items():
        for part in ("tail", "body"):
            t = rec[part]
            L.append(f"| {k} (lev/peak {rec['level_over_peak'][0]:.3g}) | {part} | {t['n0']} | {t['n1']} | {100*t['mass_frac_w0']:.2f}% | "
                     f"{t['jaccard']:.3f} | {t['jaccard_null_expected']:.3f} / [{t['jaccard_null_perm_ci95'][0]:.3f}, {t['jaccard_null_perm_ci95'][1]:.3f}] | "
                     f"{t['jaccard_over_null']:.2f} | {t['resid_pearson']:.3f} | {t['resid_spearman']:.3f} | {t['resid_pearson_intersection']:.3f} ({t['n_intersection']}) | "
                     f"{t['sign_agree_neg']:.2f} ({t['sign_agree_null']:.2f}) |")
    L += ["", f"**Verdict at the 99% level: {v.upper()}** ({res['verdict_rule']}).", "",
          "## The realisation floor before and after the 99% source-area threshold (w1 vs w0)", "",
          "| quantity | raw | thresholded | Kljun vs w0/w1 raw | Kljun_thr vs w_thr |", "|---|---|---|---|---|"]
    f = floors
    for k in f["w1_vs_w0_raw"]:
        L.append(f"| {k} | {f['w1_vs_w0_raw'][k]:.4g} | {f['w1_vs_w0_thresholded'][k]:.4g} | "
                 f"{f['kljun_vs_w_raw'][0][k]:.4g} / {f['kljun_vs_w_raw'][1][k]:.4g} | "
                 f"{f['kljun_thr_vs_w_thr'][0][k]:.4g} / {f['kljun_thr_vs_w_thr'][1][k]:.4g} |")
    ti = f["threshold_info"]
    L += ["", f"Threshold removed {100*ti['w0']['mass_removed_frac']:.2f}% / {100*ti['w1']['mass_removed_frac']:.2f}% of |mass| "
          f"(negative mass {100*ti['w0']['neg_mass_frac']:.2f}% / {100*ti['w1']['neg_mass_frac']:.2f}%), keeping "
          f"{ti['w0']['n_kept']} / {ti['w1']['n_kept']} of {ti['w0']['n_nonzero']} / {ti['w1']['n_nonzero']} non-zero cells; "
          f"level / peak {ti['w0']['level_over_peak']:.3g} / {ti['w1']['level_over_peak']:.3g}.", ""]
    with open(os.path.join(outdir, "coherence.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    pair_figure(ws, keep, c, os.path.join(outdir, "tail_pair.png"))
    print("\n".join(L))
    return res


def run_corpus(outdir, frac=0.99):
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for name in ("train", "val"):
        sp = D.load_split(name)
        _, inf = threshold_stack(sp.target, frac)
        oc = sp.octant.astype(str)
        rec = dict(n=sp.n, frac=frac)
        for k in ("mass_removed_frac", "pos_mass_removed_frac", "neg_mass_frac", "level_over_peak", "n_kept", "n_nonzero"):
            rec[k] = dict(median=float(np.median(inf[k])), p25=float(np.percentile(inf[k], 25)),
                          p75=float(np.percentile(inf[k], 75)), max=float(inf[k].max()))
        rec["by_octant_mass_removed_median"] = {o: float(np.median(inf["mass_removed_frac"][oc == o]))
                                                for o in D.OCTANTS if (oc == o).any()}
        if name == "val":
            _, ik = threshold_stack(sp.kljun, frac)
            rec["kljun"] = {k: dict(median=float(np.median(ik[k])), p75=float(np.percentile(ik[k], 75)))
                            for k in ("mass_removed_frac", "level_over_peak", "n_kept")}
        out[name] = rec
    with open(os.path.join(outdir, "threshold_corpus.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", action="store_true")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--frac", type=float, default=0.99)
    ap.add_argument("--outdir", default=OUT_DIR)
    a = ap.parse_args(argv)
    if a.pair:
        run_pair(a.outdir, a.frac)
    if a.corpus:
        run_corpus(a.outdir, a.frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
