"""Sweep the source-area cut (none, 99.9 ... 99.0%) on the CFM mean at S = 800 and at the
fitted S = 70, for the five-seed pool and for the best single seed; metrics vs Kljun and
figures of the cuts on the five reference cases. Val only; the test split is never read.

    python -m ml_cfm.cut_sweep
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml import data as D                      # noqa: E402
from ml import metrics as M                   # noqa: E402
from ml import evaluate as E                  # noqa: E402
from ml_cfm import tailthresh as TT           # noqa: E402

OUT = os.path.join(REPO, "results", "ml_cfm", "calib", "samples")
FIG = os.path.join(REPO, "results", "ml_cfm", "calib", "final")
GROUPS = ("all", "north_N_NE_NW", "array_in_view_gt5pct")
FRACS = (None, 0.999, 0.997, 0.995, 0.993, 0.991, 0.99)
TAU = 1.19


def cut(fields, frac):
    if frac is None:
        return fields, np.zeros(fields.shape[0]), np.zeros(fields.shape[0])
    out, info = TT.threshold_stack(fields, frac)
    area = ((fields != 0).sum((1, 2)) - (out != 0).sum((1, 2))) / np.maximum((fields != 0).sum((1, 2)), 1)
    return out, info["mass_removed_frac"], area


def cases(split):
    oc = split.octant.astype(str)
    integ = split.meta["integral"]
    north = np.where(oc == "N")[0]
    rows = [int(north[np.argmax(split.meta["array_share"][north])])]
    for o in ("N", "NE", "NW", "W"):
        idx = np.where((oc == o) & ~np.isin(np.arange(split.n), rows))[0]
        rows.append(int(idx[np.argmin(np.abs(integ[idx] - np.median(integ[idx])))]))
    return rows


def figure(split, st, arr, means, rows, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import fig_corpus_pairs as FCP
    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    surf = dict(water=st["water"] > 0.5, array=arr)
    share = lambda f: 100 * D.raster_array_share(f, arr)
    les = split.target[rows]
    vmax = float(max(les.max(), means[None][rows].max()))
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    cols = [(("no cut" if f is None else f"{100*f:.1f}% source-area cut"), means[f][rows]) for f in FRACS] + [("LES target", les)]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.4 * len(cols), 3.5 * len(rows)), squeeze=False)
    for r, i in enumerate(rows):
        wd = float(split.wdir_deg[i])
        for c, (name, fld) in enumerate(cols):
            ax = axes[r][c]
            im = FCP.raster(ax, fld[r], norm, "magma", ext, mask_below=norm.vmin)
            FCP.draw_frame(ax, surf, fg="w")
            FCP.draw_wind(ax, wd)
            ax.set_title(f"{name}\narray share {share(fld[r]):.1f}%", fontsize=7)
            ax.tick_params(labelsize=5.5)
            if c == 0:
                ax.set_ylabel(f"{split.meta['run_id'][i]} {split.octant[i]} {wd:.0f} deg", fontsize=7)
    fig.subplots_adjust(left=0.035, right=0.94, top=0.94, bottom=0.03, wspace=0.1, hspace=0.25)
    cax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=6)
    fig.suptitle(title + "; one log scale, floor 1e-4 x peak", fontsize=9)
    fig.savefig(out, dpi=95)
    plt.close(fig)


def main():
    split = D.load_split("val")
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    tr = D.load_split("train")
    groups = {g: m for g, m in E.breakouts(split, np.isin(split.seed_key, list(set(tr.seed_key)))).items() if g in GROUPS}
    del tr
    sc_k = M.score_fields({"k": split.kljun}, split.target, split.wdir_deg, arr, split.asymptote)["k"]
    seeds = sorted(glob.glob(os.path.join(REPO, "results", "ml_cfm", "final", "seed?")))
    rng = np.random.default_rng(0)
    T_seed = {}
    for sd in seeds:
        name = os.path.basename(sd)
        with np.load(os.path.join(sd, "samples_val.npz")) as z:
            T0, s_out = z["samples_T"].astype(np.float32), z["s_out"].astype(np.float32)
        T1 = [np.load(p)["samples_T"].astype(np.float32) for p in sorted(glob.glob(os.path.join(sd, "samples_val_extra*.npz")))]
        T_seed[name] = np.concatenate([T0] + T1)
    phys = lambda T: (s_out[None, :, None, None] * np.sinh(np.clip(T, -20, 20)) * valid[None, None]).astype(np.float32)
    temp = lambda T: (lambda m: m + TAU * (T - m))(T.mean(0, keepdims=True))

    # per-seed composite at full S -> the best seed (a val selection; reported as such)
    per_seed_comp = {}
    for name, T in T_seed.items():
        s = M.score_fields({"m": phys(temp(T)).mean(0)}, split.target, split.wdir_deg, arr, split.asymptote)["m"]
        per_seed_comp[name] = {g: M.composite(s, sc_k, groups[g])[0] for g in GROUPS}
    best = min(per_seed_comp, key=lambda k: per_seed_comp[k]["all"])
    S_per = min(t.shape[0] for t in T_seed.values())

    variants = {}
    sub = lambda T, k: T[rng.choice(T.shape[0], k, replace=False)]
    variants["pool5_S800"] = phys(temp(np.concatenate([T_seed[n][:S_per] for n in T_seed]))).mean(0)
    variants["pool5_S70"] = phys(temp(np.concatenate([sub(T_seed[n], 14) for n in T_seed]))).mean(0)
    variants[f"{best}_S160"] = phys(temp(T_seed[best])).mean(0)
    variants[f"{best}_S70"] = phys(temp(sub(T_seed[best], 70))).mean(0)

    res, means_all = {}, {}
    for vname, mean in variants.items():
        means_all[vname] = {}
        fields, info = {}, {}
        for f in FRACS:
            c, rem, area = cut(mean, f)
            key = "none" if f is None else f"{100*f:.1f}"
            fields[key] = c
            info[key] = (float(np.median(rem)), float(np.median(area)))
            means_all[vname][f] = c
        sc = M.score_fields(fields, split.target, split.wdir_deg, arr, split.asymptote)
        res[vname] = {key: dict(mass_removed=info[key][0], cells_removed=info[key][1],
                                **{g: M.composite(sc[key], sc_k, groups[g])[0] for g in GROUPS},
                                array_share_pp={g: float(np.nanmedian(sc[key]["array_share"][groups[g]])) for g in GROUPS})
                      for key in fields}
    with open(os.path.join(OUT, "cut_sweep.json"), "w") as fh:
        json.dump(dict(per_seed=per_seed_comp, best_seed=best, results=res), fh, indent=1)

    L = ["# Source-area cut sweep on val: five-seed pool and the best single seed, at S = 800/160 and S = 70", "",
         "Per-seed composite vs Kljun (all / north / in view) at 160 samples each, tau 1.19: " +
         "; ".join(f"{k} {v['all']:.3f} / {v['north_N_NE_NW']:.3f} / {v['array_in_view_gt5pct']:.3f}" for k, v in per_seed_comp.items()) +
         f". Best single seed on val: **{best}** (a selection made on val).", ""]
    for vname, r in res.items():
        L += [f"## {vname}", "", "| cut | |mass| removed | cells removed | composite all | north | in view | array share pp all / north / in view |", "|---|---|---|---|---|---|---|"]
        for key, v in r.items():
            L.append(f"| {key} | {100*v['mass_removed']:.2f}% | {100*v['cells_removed']:.0f}% | {v['all']:.3f} | {v['north_N_NE_NW']:.3f} | "
                     f"{v['array_in_view_gt5pct']:.3f} | {v['array_share_pp']['all']:.3f} / {v['array_share_pp']['north_N_NE_NW']:.3f} / {v['array_share_pp']['array_in_view_gt5pct']:.3f} |")
        L.append("")
    with open(os.path.join(OUT, "cut_sweep.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    rows = cases(split)
    for vname in variants:
        figure(split, st, arr, means_all[vname], rows, f"CFM + tau {TAU}, {vname}: the source-area cuts",
               os.path.join(FIG, f"cuts_{vname}.png"))
        print("wrote", os.path.join(FIG, f"cuts_{vname}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())


def grid(case_ids=(0, 3), out=os.path.join(FIG, "cuts_grid.png")):
    """One figure: rows = the four variants of the sweep table, columns = cuts + LES, for the
    cases picked from cases() by index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import fig_corpus_pairs as FCP
    split = D.load_split("val")
    st = D.load_statics()
    arr = st["array"] > 0.5
    valid = split.valid_mask.astype(np.float32)
    rows = [cases(split)[k] for k in case_ids]
    seeds = sorted(glob.glob(os.path.join(REPO, "results", "ml_cfm", "final", "seed?")))
    rng = np.random.default_rng(0)
    T_seed = {}
    for sd in seeds:
        with np.load(os.path.join(sd, "samples_val.npz")) as z:
            T0, s_out = z["samples_T"][:, rows].astype(np.float32), z["s_out"].astype(np.float32)[rows]
        T1 = [np.load(p)["samples_T"][:, rows].astype(np.float32) for p in sorted(glob.glob(os.path.join(sd, "samples_val_extra*.npz")))]
        T_seed[os.path.basename(sd)] = np.concatenate([T0] + T1)
    phys = lambda T: (s_out[None, :, None, None] * np.sinh(np.clip(T, -20, 20)) * valid[None, None]).astype(np.float32)
    temp = lambda T: (lambda m: m + TAU * (T - m))(T.mean(0, keepdims=True))
    sub = lambda T, k: T[rng.choice(T.shape[0], k, replace=False)]
    with open(os.path.join(OUT, "cut_sweep.json")) as fh:
        best = json.load(fh)["best_seed"]
    S_per = min(t.shape[0] for t in T_seed.values())
    variants = {"pool5_S800": phys(temp(np.concatenate([T_seed[n][:S_per] for n in T_seed]))).mean(0),
                "pool5_S70": phys(temp(np.concatenate([sub(T_seed[n], 14) for n in T_seed]))).mean(0),
                f"{best}_S160": phys(temp(T_seed[best])).mean(0),
                f"{best}_S70": phys(temp(sub(T_seed[best], 70))).mean(0)}
    xc, xe = FCP.axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    surf = dict(water=st["water"] > 0.5, array=arr)
    share = lambda f: 100 * D.raster_array_share(f, arr)
    les = split.target[rows]
    vmax = float(max(les.max(), max(v.max() for v in variants.values())))
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    ncol = len(FRACS) + 1
    fig, axes = plt.subplots(len(rows) * len(variants), ncol, figsize=(3.2 * ncol, 3.3 * len(rows) * len(variants)), squeeze=False)
    r = 0
    for ci, i in enumerate(rows):
        wd = float(split.wdir_deg[i])
        for vname, mean in variants.items():
            for c, f in enumerate(FRACS):
                fld = mean[ci] if f is None else TT.threshold_sa(mean[ci], f)[0]
                ax = axes[r][c]
                im = FCP.raster(ax, fld, norm, "magma", ext, mask_below=norm.vmin)
                FCP.draw_frame(ax, surf, fg="w"); FCP.draw_wind(ax, wd)
                ax.set_title(("no cut" if f is None else f"{100*f:.1f}% cut") + f"  share {share(fld):.1f}%", fontsize=7)
                ax.tick_params(labelsize=5)
                if c == 0:
                    ax.set_ylabel(f"{split.meta['run_id'][i]} {split.octant[i]}\n{vname}", fontsize=7.5)
            ax = axes[r][ncol - 1]
            FCP.raster(ax, les[ci], norm, "magma", ext, mask_below=norm.vmin)
            FCP.draw_frame(ax, surf, fg="w"); FCP.draw_wind(ax, wd)
            ax.set_title(f"LES target  share {share(les[ci]):.1f}%", fontsize=7); ax.tick_params(labelsize=5)
            r += 1
    fig.subplots_adjust(left=0.04, right=0.94, top=0.96, bottom=0.02, wspace=0.1, hspace=0.3)
    cax = fig.add_axes([0.95, 0.2, 0.01, 0.6])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=6)
    fig.suptitle(f"The cut-sweep table as images: rows = variant (CFM + tau {TAU}), columns = cut; one log scale, floor 1e-4 x peak", fontsize=9)
    fig.savefig(out, dpi=95)
    print("wrote", out)
