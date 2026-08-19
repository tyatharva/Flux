#!/usr/bin/env python3
"""Figures for the 30 m second-pass results. Reads only saved arrays -- no LES, no LPDM.

Everything plotted is RENORMALISED: each field is divided by its own integral over the
plotted grid, so the comparison is of SHAPE. The integral that was divided out -- the
"captured fraction" -- is printed on every panel, because it is a result in its own right
(the LES flat case captures 0.80, the westerly 1.45; see STAGE2-6_RESULTS_V2.md).

usage: make_figures.py [--outdir figures]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lpdm import kljun as _kljun

RES = 60.0
NX_LES, NY_LES, DX_LES = 146, 50, 30.0
I_TOWER, J_TOWER = 110, 25          # bin/prep_stage6.py --itower default and the LPDM's
                                    # round(0.75*146); they agree, so there is no offset.

CASES = [
    dict(tag="flat",     npz="results/fv_stage5.npz", js="results/fv_stage5.json",
         title="Flat, uniform, neutral", cover=None, wind=None, nwin=721),
    dict(tag="westerly", npz="results/fv_t6w.npz",    js="results/fv_t6w.json",
         title="Real surface, wind from 270$\\degree$ (westerly)",
         cover="runs/fv_t6w_adj", wind=270.0, nwin=361),
    dict(tag="easterly", npz="results/fv_t6e.npz",    js="results/fv_t6e.json",
         title="Real surface, wind from 90$\\degree$ (easterly)",
         cover="runs/fv_t6e_adj", wind=90.0, nwin=361),
]


# --------------------------------------------------------------------------- helpers

def normalise(f, res=RES):
    """Return (renormalised field, captured fraction). Captured is the integral removed."""
    cap = float(f.sum()) * res * res
    return (f / cap if cap != 0 else f * 0.0), cap


def source_area_levels(f, fracs=(0.5, 0.8)):
    """Contour level enclosing each cumulative fraction of a POSITIVE field."""
    v = np.sort(np.maximum(f, 0).ravel())[::-1]
    tot = v.sum()
    if tot <= 0:
        return [np.nan] * len(fracs)
    cum = np.cumsum(v) / tot
    return [float(v[min(np.searchsorted(cum, q), len(v) - 1)]) for q in fracs]


def crosswind_integrated(f, res=RES):
    """f_y(x) in m^-1 from a 2-D field in m^-2."""
    return f.sum(axis=0) * res


def peak_and_centroid(f, xc, res=RES):
    fy = crosswind_integrated(f, res)
    tot = fy.sum()
    if tot <= 0:
        return np.nan, np.nan
    return float(xc[int(np.argmax(fy))]), float((fy * xc).sum() / tot)


def cover_in_footprint_frame(cover_dir, ang, ntile=1):
    """Rotate the LES surface masks into the receptor-relative footprint frame.

    driver.compute_footprint maps a touchdown to  X = -(dx ca + dy sa),  Y = -dx sa + dy ca
    with ang = atan2(V, U) at the receptor. Apply the same map to cell centres. The domain
    is periodic and trajectories wrap up to the displacement cap, so tile +/- ntile periods
    -- otherwise the rotated tile is a parallelogram that leaves gaps in the plotted window.
    """
    z0 = np.load(os.path.join(cover_dir, "z0m.npy"))
    water = np.load(os.path.join(cover_dir, "water.npy")) > 0.5
    array = z0 > 0.1
    topo = np.load(os.path.join(cover_dir, "topo.npy"))
    ny, nx = water.shape
    ii = np.arange(-ntile * nx, (ntile + 1) * nx)
    jj = np.arange(-ntile * ny, (ntile + 1) * ny)
    I, J = np.meshgrid(ii, jj)
    dxm = (I - I_TOWER) * DX_LES
    dym = (J - J_TOWER) * DX_LES
    ca, sa = np.cos(ang), np.sin(ang)
    X = -(dxm * ca + dym * sa)
    Y = -dxm * sa + dym * ca
    tile = lambda a: a[J % ny, I % nx]
    out = dict(Xw=X, Yw=Y, water_w=tile(water).astype(float),
               array_w=tile(array).astype(float))
    I0, J0 = np.meshgrid(np.arange(nx), np.arange(ny))
    dx0 = (I0 - I_TOWER) * DX_LES
    dy0 = (J0 - J_TOWER) * DX_LES
    out.update(X=-(dx0 * ca + dy0 * sa), Y=-dx0 * sa + dy0 * ca,
               water=water.astype(float), array=array.astype(float), topo=topo)
    return out


def draw_cover(ax, cv, terrain=False):
    """Periodic images faint and dashed; the primary tile solid on top of them.

    The images are not decoration -- backward trajectories genuinely wrap into them, up to
    the one-domain-length displacement cap -- so they are drawn rather than hidden, but
    they must not be mistaken for a second lake.
    """
    if cv is None:
        return
    for key, col, lw in (("water", "#00b8ff", 1.0), ("array", "#39ff14", 1.3)):
        ax.contour(cv["Xw"], cv["Yw"], cv[key + "_w"], levels=[0.5], colors=col,
                   linewidths=lw * 0.7, linestyles=":", alpha=0.45)
    if terrain:
        ax.contour(cv["X"], cv["Y"], cv["topo"], levels=7, colors="k",
                   linewidths=0.4, alpha=0.30)
    ax.contourf(cv["X"], cv["Y"], cv["water"], levels=[0.5, 1.5], colors=["#00b8ff"],
                alpha=0.16)
    ax.contourf(cv["X"], cv["Y"], cv["array"], levels=[0.5, 1.5], colors=["#39ff14"],
                alpha=0.30)
    ax.contour(cv["X"], cv["Y"], cv["water"], levels=[0.5], colors="#00b8ff",
               linewidths=1.2)
    ax.contour(cv["X"], cv["Y"], cv["array"], levels=[0.5], colors="#39ff14",
               linewidths=1.6)


def draw_tower(ax, colour="w"):
    ax.plot(0, 0, marker="*", ms=15, mfc=colour, mec="k", mew=0.8, ls="none", zorder=6)


def wind_annotation(ax, case, ang_deg):
    """The frame is wind-aligned: upwind is +x. Say so, and say what it is geographically."""
    if case["wind"] is None:
        txt = ("frame aligned to the mean wind\n"
               "flat uniform — no geographic direction\n"
               f"residual yaw {ang_deg:+.1f}° from LES $+x$")
    else:
        txt = (f"wind FROM {case['wind']:.0f}° (geographic)\n"
               "frame aligned to the mean wind\n"
               f"residual yaw {ang_deg:+.1f}° from LES $+x$")
    ax.annotate("", xy=(0.055, 0.90), xytext=(0.30, 0.90), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color="w"))
    ax.text(0.055, 0.855, "mean wind", transform=ax.transAxes, color="w",
            fontsize=7.5, va="top")
    ax.text(0.985, 0.965, txt, transform=ax.transAxes, color="w", fontsize=7.5,
            ha="right", va="top",
            bbox=dict(fc="k", ec="none", alpha=0.45, boxstyle="round,pad=0.3"))


def cover_shares(tag):
    """Footprint-weighted land-cover shares, parsed from the saved stage-5 report.

    These come from the touchdowns in LES INDEX space, not from the 60 m footprint grid --
    rotating and binning a 30 m patch would blur exactly the number that must not blur.
    """
    f = {"westerly": "results/fv_t6w.txt", "easterly": "results/fv_t6e.txt"}.get(tag)
    if not f or not os.path.exists(f):
        return None
    out, on = [], False
    for line in open(f):
        if "land-cover share" in line:
            on = True; continue
        if on:
            if "(domain area share" not in line:
                break
            name = line.split()[0] if not line.strip().startswith("solar") else "array"
            fp = float(line.split("%")[0].split()[-1])
            area = float(line.split("area share")[1].split("%")[0])
            out.append((name, fp, area))
    if not out:
        return None
    txt = "footprint-weighted share (touchdowns)\n"
    for nm, fp, ar in out:
        if nm == "grass":
            continue
        txt += f"{nm:<12s}{fp:6.2f}%  of {ar:5.2f}% area  = {fp / ar:5.2f}x\n"
    return txt.rstrip()


def kljun_input_string(st, zm):
    p4 = st["u_mean"] / st["ustar"] * 0.4
    z0_imp = zm * np.exp(-p4)
    L = st["L"]
    Ls = "$\\infty$ (neutral)" if not np.isfinite(L) else f"{L:.0f} m"
    return (f"Kljun 2015 FFP inputs:  $z_m$={zm:.1f} m,  $h$={st['h']:.0f} m,  "
            f"$u_*$={st['ustar']:.3f} m s$^{{-1}}$,  $\\sigma_v$={st['sigma_v']:.3f} m s$^{{-1}}$,  "
            f"$\\bar u$={st['u_mean']:.2f} m s$^{{-1}}$,  $L$={Ls}\n"
            f"$\\Pi_4$ taken from the measured $\\bar u/u_*$ rather than $\\ln(z_m/z_0)$; "
            f"implied $z_0$ = {z0_imp:.4f} m  (prescribed grass $z_0$ = 0.03 m)")


# --------------------------------------------------------------------------- figure 1-3

def case_figure(case, outdir):
    z = np.load(case["npz"])
    d = json.load(open(case["js"]))
    st = d["stats"]
    zm = d["zm"]
    xc, yc = z["xc"], z["yc"]
    ang = np.arctan2(st["V"], st["U"])

    les, cap_les = normalise(z["les"])
    klj, cap_klj = normalise(z["kljun"])
    cv = cover_in_footprint_frame(case["cover"], ang) if case["cover"] else None

    ext = [z["xe"][0], z["xe"][-1], z["ye"][0], z["ye"][-1]]
    vmax = max(les.max(), klj.max())
    norm = LogNorm(vmin=vmax / 1e5, vmax=vmax)

    fig = plt.figure(figsize=(16.4, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26,
                          left=0.055, right=0.955, top=0.825, bottom=0.070)

    panels = [(les, "LES + backward LPDM", cap_les, gs[0, 0]),
              (klj, "Kljun et al. (2015)", cap_klj, gs[0, 1])]
    for F, name, cap, cell in panels:
        ax = fig.add_subplot(cell)
        im = ax.imshow(np.ma.masked_less_equal(F, 0), origin="lower", extent=ext,
                       aspect="equal", cmap="magma", norm=norm)
        Fs = gaussian_filter(np.maximum(F, 0), 1.0)
        l50, l80 = source_area_levels(Fs)
        ax.contour(xc, yc, Fs, levels=[l80], colors="w", linewidths=1.0,
                   linestyles="--")
        ax.contour(xc, yc, Fs, levels=[l50], colors="w", linewidths=1.4)
        draw_cover(ax, cv)
        draw_tower(ax)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        pk, ct = peak_and_centroid(F, xc)
        if name.startswith("LES"):
            cs = cover_shares(case["tag"])
            if cs:
                ax.text(0.015, 0.035, cs, transform=ax.transAxes, color="w",
                        fontsize=8, family="monospace", va="bottom",
                        bbox=dict(fc="k", ec="none", alpha=0.55,
                                  boxstyle="round,pad=0.35"))
        ax.set_title(f"{name}\ncaptured fraction {cap:.3f}   |   "
                     f"peak {pk:.0f} m, centroid {ct:.0f} m", fontsize=10)
        ax.set_xlabel("upwind distance (m)")
        ax.set_ylabel("crosswind distance (m)")
        wind_annotation(ax, case, np.degrees(ang))
        cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.031)
        cb.set_label("$f$ (m$^{-2}$), renormalised", fontsize=8)

    # difference -----------------------------------------------------------------
    axd = fig.add_subplot(gs[1, 0])
    diff = les - klj
    m = float(np.abs(diff).max())
    imd = axd.imshow(diff, origin="lower", extent=ext, aspect="equal", cmap="RdBu_r",
                     norm=SymLogNorm(linthresh=vmax / 1e3, vmin=-m, vmax=m, base=10))
    draw_cover(axd, cv, terrain=True)
    draw_tower(axd, colour="yellow")
    axd.set_xlim(ext[0], ext[1]); axd.set_ylim(ext[2], ext[3])
    if cv is not None:
        axd.plot([], [], color="#39ff14", lw=1.6, label="solar array")
        axd.plot([], [], color="#00b8ff", lw=1.2, label="open water")
        axd.plot([], [], color="k", lw=0.5, alpha=0.4, label="terrain")
        axd.plot([], [], color="0.5", lw=0.8, ls=":", label="periodic images")
        axd.legend(frameon=True, fontsize=7.5, loc="lower left", framealpha=0.75)
    axd.set_title("LES $-$ Kljun, both renormalised  (red: LES puts more weight here)",
                  fontsize=10)
    axd.set_xlabel("upwind distance (m)"); axd.set_ylabel("crosswind distance (m)")
    cbd = fig.colorbar(imd, ax=axd, pad=0.012, fraction=0.031)
    cbd.set_label("$\\Delta f$ (m$^{-2}$)", fontsize=8)

    # crosswind-integrated -------------------------------------------------------
    axc = fig.add_subplot(gs[1, 1])
    fyl, fyk = crosswind_integrated(les), crosswind_integrated(klj)
    sm = np.convolve(fyl, np.ones(5) / 5.0, mode="same")
    axc.plot(xc, fyl, lw=1.0, color="#1f77b4", alpha=0.45)
    axc.plot(xc, sm, lw=2.2, color="#1f77b4", label="LES + LPDM (thin: raw)")
    axc.plot(xc, fyk, lw=2.2, ls="--", color="#d62728", label="Kljun 2015")
    pl, kl_ = xc[np.argmax(fyl)], xc[np.argmax(fyk)]
    axc.axvline(pl, color="#1f77b4", lw=0.9, ls=":")
    axc.axvline(kl_, color="#d62728", lw=0.9, ls=":")
    axc.text(pl + 40, 0.90 * max(fyl.max(), fyk.max()), f"LES peak {pl:.0f} m",
             color="#1f77b4", fontsize=9)
    axc.text(kl_ + 40, 0.98 * max(fyl.max(), fyk.max()), f"Kljun peak {kl_:.0f} m",
             color="#d62728", fontsize=9)
    axc.set_ylim(min(0.0, 1.15 * fyl.min()), 1.10 * max(fyl.max(), fyk.max()))
    axc.set_xlim(-100, 2600)
    axc.set_xlabel("upwind distance (m)")
    axc.set_ylabel("$f_y$ (m$^{-1}$), renormalised")
    axc.set_title(f"Crosswind-integrated footprint   "
                  f"($\\Delta$peak {pl - kl_:+.0f} m)", fontsize=10)
    axc.grid(alpha=0.25); axc.legend(frameon=False)

    fig.suptitle(f"{case['title']}   —   30 m grid, $dz_{{sfc}}$ = 8.56 m, "
                 f"{case['nwin']} dumps at 5 s\n" + kljun_input_string(st, zm),
                 fontsize=11)
    p = os.path.join(outdir, f"footprint_{case['tag']}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, dict(cap_les=cap_les, cap_klj=cap_klj, peak_les=pl, peak_klj=kl_)


# --------------------------------------------------------------------------- figure 4

def crosswind_figure(outdir):
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.0))
    for c, case in enumerate(CASES):
        z = np.load(case["npz"]); d = json.load(open(case["js"]))
        xc = z["xc"]
        les, cap_l = normalise(z["les"]); klj, cap_k = normalise(z["kljun"])
        fyl, fyk = crosswind_integrated(les), crosswind_integrated(klj)
        pl, kl_ = xc[np.argmax(fyl)], xc[np.argmax(fyk)]

        st = d["stats"]
        xmax_an = _kljun.peak_distance(d["zm"], st["h"], st["ustar"],
                                       umean=st["u_mean"], L=st["L"])
        sm = np.convolve(fyl, np.ones(5) / 5.0, mode="same")
        a = ax[0, c]
        a.plot(xc, fyl, lw=1.0, color="#1f77b4", alpha=0.45)
        a.plot(xc, sm, lw=2.2, color="#1f77b4", label="LES + LPDM (thin: raw)")
        a.plot(xc, fyk, lw=2.2, ls="--", color="#d62728", label="Kljun 2015")
        a.axvline(xmax_an, color="#d62728", lw=0.8, ls="-.", alpha=0.7)
        a.axvline(pl, color="#1f77b4", lw=0.9, ls=":")
        a.axvline(kl_, color="#d62728", lw=0.9, ls=":")
        a.set_xlim(-50, 2500); a.grid(alpha=0.25)
        a.set_title(f"{case['title']}\npeak: LES {pl:.0f} m vs Kljun {kl_:.0f} m "
                    f"({(pl - kl_) / kl_ * 100:+.0f}%)   "
                    f"[Kljun analytic $x_{{max}}$ = {xmax_an:.0f} m]", fontsize=9.5)
        a.set_xlabel("upwind distance (m)"); a.set_ylabel("$f_y$ (m$^{-1}$)")
        a.text(0.97, 0.55, f"captured\nLES {cap_l:.3f}\nKljun {cap_k:.3f}",
               transform=a.transAxes, ha="right", va="top", fontsize=8.5,
               bbox=dict(fc="w", ec="0.7", boxstyle="round,pad=0.3"))
        if c == 0:
            a.legend(frameon=False)

        b = ax[1, c]
        pos = xc > 0
        b.loglog(xc[pos], np.maximum(fyl[pos], 1e-12), lw=2.0, color="#1f77b4")
        b.loglog(xc[pos], np.maximum(fyk[pos], 1e-12), lw=2.0, ls="--", color="#d62728")
        b.set_ylim(1e-7, 3e-3); b.set_xlim(60, 5000); b.grid(alpha=0.25, which="both")
        b.set_xlabel("upwind distance (m)"); b.set_ylabel("$f_y$ (m$^{-1}$)")
        b.axvline(4380, color="0.4", lw=1.0, ls="-.")
        b.text(3100, 4e-6, "wrap cap\n(one domain length)", fontsize=7,
               color="0.4", ha="right", rotation=90, va="bottom")
        b.set_title("same curves, log-log — the tail\n"
                    "(LES dropouts are bins whose signed $w'$ weights sum to $\\leq$ 0)",
                    fontsize=9)

    fig.suptitle("Crosswind-integrated flux footprint, LES+LPDM vs Kljun et al. (2015)   "
                 "—   all fields renormalised to unit integral over the plotted grid\n"
                 "This is where the peak discrepancy lives: at 88% sub-grid $\\sigma_w^2$ "
                 "the near-field is set by the Langevin closure, which decorrelates faster "
                 "than real eddies and pushes the peak downwind.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(outdir, "crosswind_integrated_all.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# --------------------------------------------------------------------------- figure 5

def ensemble_figure(outdir, ndraw=400, seed=0):
    """Convergence against an INDEPENDENT held-out reference.

    A bootstrap band drawn from all M sub-windows collapses to zero width at n = M by
    construction -- only one subset of size M exists -- which reads as convergence and is
    not. So: shuffle, hold out 9 sub-windows as the reference, average the first n of the
    remainder, and report |difference|. Both sides then stay independent at every n.
    """
    z = np.load("results/fv_ensemble.npz")
    F, xc, yc = z["F"], z["xc"], z["yc"]
    M = F.shape[0]
    half = M // 2
    rng = np.random.default_rng(seed)

    ns = np.arange(1, half + 1)
    dpk = np.zeros((len(ns), 2)); dct = np.zeros((len(ns), 2))
    for a, n in enumerate(ns):
        dp, dc = [], []
        for _ in range(ndraw):
            perm = rng.permutation(M)
            fr, _c = normalise(F[perm[half:]].mean(axis=0))     # held-out reference
            fs_, _c = normalise(F[perm[:n]].mean(axis=0))       # sample of size n
            pr, cr = peak_and_centroid(fr, xc)
            ps, cs = peak_and_centroid(fs_, xc)
            dp.append(abs(ps - pr)); dc.append(abs(cs - cr))
        dpk[a] = np.percentile(dp, [50, 90]); dct[a] = np.percentile(dc, [50, 90])

    shown = [1, 3, 9, 18]
    fig = plt.figure(figsize=(17.0, 8.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.45, wspace=0.34,
                          left=0.068, right=0.975, top=0.835, bottom=0.078)
    ext = [xc[0] - 30, xc[-1] + 30, yc[0] - 30, yc[-1] + 30]
    fields = [normalise(F[:n].mean(axis=0))[0] for n in shown]
    vmax = max(f.max() for f in fields)
    norm = LogNorm(vmin=vmax / 1e4, vmax=vmax)
    for a, (n, f) in enumerate(zip(shown, fields)):
        ax = fig.add_subplot(gs[0, a])
        im = ax.imshow(np.ma.masked_less_equal(f, 0), origin="lower", extent=ext,
                       aspect="equal", cmap="magma", norm=norm)
        fsm = gaussian_filter(np.maximum(f, 0), 1.0)
        l50, l80 = source_area_levels(fsm)
        ax.contour(xc, yc, fsm, levels=[l80], colors="w", linewidths=0.9,
                   linestyles="--")
        ax.contour(xc, yc, fsm, levels=[l50], colors="w", linewidths=1.2)
        pkv, ctv = peak_and_centroid(f, xc)
        ax.axvline(pkv, color="#39ff14", lw=1.3)
        ax.axvline(ctv, color="#00b8ff", lw=1.3, ls="--")
        draw_tower(ax)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(f"first $n$ = {n} sub-window{'s' if n > 1 else ''}  "
                     f"({n * 2.5:.1f} min)\npeak {pkv:.0f} m (green), "
                     f"centroid {ctv:.0f} m (blue)", fontsize=9.5)
        ax.set_xlabel("upwind (m)")
        if a == 0:
            ax.set_ylabel("crosswind (m)")
        if a == 3:
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
            cb.set_label("$f$ (m$^{-2}$)", fontsize=8)

    # spread of f_y at each n -- several random draws, so the scatter is visible ------
    axa = fig.add_subplot(gs[1, 0:2])
    ref, _ = normalise(F.mean(axis=0))
    cols = {1: "#1f77b4", 3: "#ff7f0e", 9: "#2ca02c"}
    for n, col in cols.items():
        for k in range(8):
            idx = rng.choice(M, size=n, replace=False)
            fk, _c = normalise(F[idx].mean(axis=0))
            axa.plot(xc, crosswind_integrated(fk), lw=1.0, color=col, alpha=0.35,
                     label=f"$n$ = {n}" if k == 0 else None)
    axa.plot(xc, crosswind_integrated(ref), lw=2.6, color="k",
             label=f"all {M} (45 min)")
    axa.set_xlim(-50, 3000); axa.grid(alpha=0.25)
    axa.set_xlabel("upwind distance (m)"); axa.set_ylabel("$f_y$ (m$^{-1}$)")
    axa.set_title("Scatter of the crosswind-integrated footprint: 8 random subsets at "
                  "each $n$\n(a single 150 s window can go NEGATIVE — individual "
                  "touchdowns carry signed $w'$)", fontsize=9.5)
    axa.legend(frameon=False, ncol=4)

    for cell, q, lab, tgt, tlab, col in (
            (gs[1, 2], dpk, "|$\\Delta$ peak| (m)", 60.0, "one grid cell (60 m)", "#2ca02c"),
            (gs[1, 3], dct, "|$\\Delta$ centroid| (m)", 100.0, "100 m", "#1f77b4")):
        ax_ = fig.add_subplot(cell)
        ax_.plot(ns, q[:, 0], color="k", lw=1.8, marker="o", ms=4, label="median")
        ax_.plot(ns, q[:, 1], color=col, lw=2.0, marker="s", ms=4, label="p90")
        ax_.axhline(tgt, color="r", lw=1.2, ls="--", label=tlab)
        ax_.set_xlabel("$n$ sub-windows averaged  (2.5 min each)")
        ax_.set_ylabel(lab); ax_.grid(alpha=0.25); ax_.set_xticks(ns)
        ax_.legend(frameon=False, fontsize=8.5)
        ax_.set_title(lab.split("|")[1].strip() + " vs a held-out 9-window reference",
                      fontsize=9.5)

    n_pk = next((int(n) for n, v in zip(ns, dpk[:, 1]) if v <= 60.0), None)
    n_ct = next((int(n) for n, v in zip(ns, dct[:, 1]) if v <= 100.0), None)
    fig.suptitle(
        "Ensemble convergence — flat/neutral case, one integration split into "
        f"{M} independent sub-windows of 150 s (lag-1 autocorrelation +0.19 peak / "
        "$-$0.10 centroid, below $2/\\sqrt{18}$ = 0.47)\n"
        "Footprints are averaged, then the metric is recomputed — never the other way "
        f"round. Right-hand panels: {ndraw} draws, each compared with an INDEPENDENT "
        "held-out 9-sub-window reference.\n"
        f"CORPUS PARAMETER — peak within one cell at p90: $n$ = "
        f"{n_pk if n_pk else '>9'} ({(n_pk or 9) * 2.5:.1f} min).   "
        f"Centroid within 100 m at p90: $n$ = {n_ct if n_ct else '>9'}"
        f"{'' if n_ct else ' (>22.5 min — still improving)'}.", fontsize=10.5)
    p = os.path.join(outdir, "ensemble_convergence.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p, dpk, dct, ns


def surface_effect_figure(outdir):
    """LES(terrain) - LES(flat): the surface signal with the closure error divided out.

    The LES-minus-Kljun panels are dominated by the sub-grid closure pushing the peak
    downwind, which has nothing to do with the surface. Differencing two LES runs on the
    same grid, same closure and the same spun-up state cancels that and leaves the thing
    Stage 6 actually gates on.
    """
    zf = np.load(CASES[0]["npz"])
    flat, _c = normalise(zf["les"])
    xc, yc = zf["xc"], zf["yc"]
    ext = [zf["xe"][0], zf["xe"][-1], zf["ye"][0], zf["ye"][-1]]

    fig, ax = plt.subplots(2, 2, figsize=(15.0, 8.4))
    for c, case in enumerate(CASES[1:]):
        z = np.load(case["npz"]); d = json.load(open(case["js"]))
        ang = np.arctan2(d["stats"]["V"], d["stats"]["U"])
        ter, cap = normalise(z["les"])
        cv = cover_in_footprint_frame(case["cover"], ang)
        diff = ter - flat
        m = float(np.abs(diff).max())
        a = ax[0, c]
        im = a.imshow(diff, origin="lower", extent=ext, aspect="equal", cmap="RdBu_r",
                      norm=SymLogNorm(linthresh=m / 1e3, vmin=-m, vmax=m, base=10))
        draw_cover(a, cv)
        draw_tower(a, colour="yellow")
        a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
        a.set_xlabel("upwind distance (m)"); a.set_ylabel("crosswind distance (m)")
        a.set_title(f"{case['title']}\nreal surface $-$ flat, both renormalised  "
                    f"(red: real surface puts more weight here)", fontsize=9.5)
        cb = fig.colorbar(im, ax=a, pad=0.030, fraction=0.031)
        cb.set_label("$\\Delta f$ (m$^{-2}$)", fontsize=8, labelpad=2)
        cb.ax.tick_params(labelsize=7)
        cs = cover_shares(case["tag"])
        if cs:
            a.text(0.015, 0.035, cs, transform=a.transAxes, fontsize=8,
                   family="monospace", va="bottom",
                   bbox=dict(fc="w", ec="0.6", alpha=0.85, boxstyle="round,pad=0.35"))

        b = ax[1, c]
        fyt, fyf = crosswind_integrated(ter), crosswind_integrated(flat)
        b.plot(xc, fyf, lw=2.0, color="0.35", label="flat uniform")
        b.plot(xc, fyt, lw=2.2, color="#1f77b4", label=case["title"].split(",")[-1].strip())
        if case["tag"] == "westerly":
            b.axvspan(135, 255, color="#39ff14", alpha=0.30, label="solar array")
        else:
            b.axvspan(840, 3300, color="#00b8ff", alpha=0.18, label="open water")
            b.axvspan(-255, -135, color="#39ff14", alpha=0.30, label="solar array")
        b.set_xlim(-300, 3600); b.grid(alpha=0.25)
        b.set_xlabel("upwind distance (m)"); b.set_ylabel("$f_y$ (m$^{-1}$)")
        b.set_title("crosswind-integrated, against the flat control", fontsize=9.5)
        b.legend(frameon=False, fontsize=8.5)

    fig.suptitle("The surface signal, with the closure error differenced out — "
                 "LES(real surface) $-$ LES(flat), same grid, same closure, "
                 "same spun-up state\n"
                 "The LES$-$Kljun panels are dominated by the sub-grid closure pushing the "
                 "peak downwind, which has nothing to do with the surface; this comparison "
                 "cancels it.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(outdir, "surface_effect.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    paths = []
    for case in CASES:
        p, info = case_figure(case, a.outdir)
        print(f"  {p}   captured: LES {info['cap_les']:.3f}  Kljun {info['cap_klj']:.3f}"
              f"   peak {info['peak_les']:.0f} vs {info['peak_klj']:.0f} m")
        paths.append(p)
    p = crosswind_figure(a.outdir); print(f"  {p}"); paths.append(p)
    p = surface_effect_figure(a.outdir); print(f"  {p}"); paths.append(p)
    p, dpk, dct, ns = ensemble_figure(a.outdir); print(f"  {p}"); paths.append(p)
    print("\n   n  sampling   |d peak| med   p90   |d centroid| med      p90")
    for i, n in enumerate(ns):
        print(f"  {n:2d}  {n*2.5:5.1f} min   {dpk[i,0]:10.0f} {dpk[i,1]:5.0f}"
              f"   {dct[i,0]:14.0f} {dct[i,1]:8.0f}")
    print("\nWROTE:"); [print("  " + q) for q in paths]


if __name__ == "__main__":
    sys.exit(main())
