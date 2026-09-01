#!/usr/bin/env python3
"""Figures of the (input, target) pairs in corpus/corpus.h5 -- the ML dataset itself.

These are DATASET figures, not physics figures. Nothing here runs the LES or the LPDM;
everything is read out of corpus.h5, which is what the emulator will be trained on. If a
pair is wrong -- wrong orientation, wrong scale, the array in the wrong place, the zero pad
not zero -- it is wrong here, in the file the loader opens.

Every raster panel is in the SAME frame the corpus stores: north-up map, 30 m cells,
receptor at the centre of cell (64, 64), 122 real cells zero-padded to 128. The frame is
NOT wind-aligned, so the footprint swings around the receptor with the wind and the solar
array and the lake stay put. That is the point of the frame and it is the first thing to
check by eye.

Panels are NOT renormalised (unlike bin/make_figures.py). The absolute scale is an input to
the loss, so it is what is plotted; the integral is printed on every target panel instead.

usage: fig_corpus_pairs.py [--h5 corpus/corpus.h5] [--outdir figures]
                           [--surface data/grid30_raised] [--dpi 130]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

import h5py

# ------------------------------------------------------------------ frame constants
# All of these are ASSERTED against the file's own grid attributes in load_corpus(); they
# are written down rather than derived so a mismatch is an error instead of a silent
# re-scaling of every axis in every figure.
NPAD = 128
NRAW = 122
PAD = 3
DX = 30.0
IJ_RECEPTOR = 64             # padded index of the receptor cell, both axes

# The solar array, from bin/prep_surface.py:ARRAY_DX_W/E, ARRAY_DY_S/N. A rectangle in
# EPSG:3071 metres relative to the surveyed tower -- THE TOWER IS INSIDE IT -- so in this
# frame it is the same rectangle in every one of the 1366 records. If it moves between
# panels, the frame is wrong.
ARRAY_XY = (-60.0, 60.0, -100.0, 250.0)

SCALAR_UNITS = {"h": "m", "ustar": "m s$^{-1}$", "sigma_v": "m s$^{-1}$", "L": "m",
                "sin_wdir": "", "cos_wdir": ""}
SPLIT_COLOUR = {"train": "#4c72b0", "val": "#dd8452", "test": "#55a868"}
OCTANTS = [("N", 0.0), ("NE", 45.0), ("E", 90.0), ("SE", 135.0),
           ("S", 180.0), ("SW", 225.0), ("W", 270.0), ("NW", 315.0)]

CAPTION = (
    "North-up map frame, 30 m cells, receptor (star) at the origin, 122 real cells inside "
    "the dotted zero-pad boundary.  green: solar array (fixed -- the tower is inside it)   "
    "cyan: Lake Kegonsa   arrow: mean flow   white: 50% and 80% source-area contours   "
    "dashed cyan: region where the signed target is negative.\n"
    "The INPUT and TARGET panels of a row share ONE logarithmic colour scale spanning four "
    "decades below the larger of the two peaks; cells below that floor are left blank "
    "rather than painted the darkest colour.  Nothing is renormalised.\n"
    "The faint speckled lobe on the DOWNWIND side of the target is not a second footprint: "
    "touchdowns are binned by LES column index and folded modulo the periodic domain, so "
    "trajectories running more than one domain length upwind reappear through the seam.  "
    "Displacement is capped at one domain length, which is what bounds it.")


# ------------------------------------------------------------------------ loading

def _s(a):
    """h5py object column -> list[str]."""
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def load_corpus(path):
    with h5py.File(path, "r") as f:
        g = f["grid"].attrs
        # Rule 1: validate the artifact, not the constant you hoped it had.
        for name, want, got in (("n", NRAW, int(g["n"])),
                                ("pad", PAD, int(g["pad"])),
                                ("n_padded", NPAD, int(g["n_padded"])),
                                ("dx_m", DX, float(g["dx_m"]))):
            if want != got:
                sys.exit(f"FATAL: {path} grid/{name} is {got}, this script assumes {want}")
        d = dict(kljun=f["kljun"][:], target=f["target"][:], scalars=f["scalars"][:])
        m = f["meta"]
        d["scalar_names"] = _s(m["scalar_names"][:])
        for k in ("run_id", "datetime", "split", "gate_state"):
            d[k] = np.array(_s(m[k][:]))
        for k in ("wdir_deg", "array_share", "integral", "peak_x_m", "zi_achieved_m",
                  "centroid_dist_m", "inv_L"):
            d[k] = m[k][:]
        d["valid_mask"] = m["valid_mask"][:]
        d["norm_mean"] = f["norm/scalars_mean"][:]
        d["norm_std"] = f["norm/scalars_std"][:]
        d["n"] = int(f.attrs["n"])
    if not np.isfinite(d["target"]).all() or not np.isfinite(d["kljun"]).all():
        sys.exit("FATAL: non-finite values in the rasters")   # rule 9
    return d


def axes_m():
    """Cell centres and edges of the PADDED raster, metres from the receptor."""
    xc = (np.arange(NPAD) - IJ_RECEPTOR) * DX
    xe = (np.arange(NPAD + 1) - IJ_RECEPTOR - 0.5) * DX
    return xc, xe


def load_surface(surface_dir):
    """The production surface masks, padded into the corpus frame. None if absent."""
    try:
        water = np.load(os.path.join(surface_dir, "water.npy"))
        array = np.load(os.path.join(surface_dir, "array.npy"))
    except OSError:
        return None
    if water.shape != (NRAW, NRAW):
        print(f"  note: {surface_dir} is {water.shape}, not {(NRAW, NRAW)}; "
              "surface overlay skipped")
        return None
    pad = lambda a: np.pad(a.astype(float), PAD)
    return dict(water=pad(water > 0.5), array=pad(array > 0.5))


# ---------------------------------------------------------------------- diagnostics

def crosswind_integrated(f, wdir_deg, ds=DX, smax=1830.0):
    """f_y(s) in m^-1 against UPWIND distance s, by binning cells on the wind axis.

    Binning cell centres by their upwind coordinate is exactly conservative (every cell's
    m^-2 x area lands in exactly one bin) and needs no resampling of the raster, which is
    the whole reason the corpus stores a north-up frame rather than a wind-aligned one.
    """
    xc, _ = axes_m()
    X, Y = np.meshgrid(xc, xc)
    a = np.radians(wdir_deg)                 # meteorological: the direction wind comes FROM
    s = X * np.sin(a) + Y * np.cos(a)        # +s is upwind
    edges = np.arange(-smax, smax + ds, ds)
    tot, _ = np.histogram(s.ravel(), bins=edges, weights=(f * DX * DX).ravel())
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, tot / ds                 # m^-1


def source_area_levels(f, fracs=(0.5, 0.8)):
    """Contour level enclosing each cumulative fraction of the POSITIVE part of f."""
    v = np.sort(np.maximum(f, 0).ravel())[::-1]
    tot = v.sum()
    if tot <= 0:
        return [np.nan] * len(fracs)
    cum = np.cumsum(v) / tot
    return [float(v[min(np.searchsorted(cum, q), len(v) - 1)]) for q in fracs]


def array_upwind_span(wdir_deg):
    """The solar array's extent projected onto the wind axis, for THIS direction."""
    x0, x1, y0, y1 = ARRAY_XY
    a = np.radians(wdir_deg)
    s = [x * np.sin(a) + y * np.cos(a) for x in (x0, x1) for y in (y0, y1)]
    return float(min(s)), float(max(s))


def kljun_peak_distance(d):
    """Upwind distance of the INPUT's crosswind-integrated peak, per record.

    corpus_monitor.py takes Kljun's peak_x from the stage-5 report; corpus.h5 does not
    carry it, so it is re-derived here on the same wind axis the LES peak was measured on.
    The two agree to the 30 m cell -- the count printed at the end is checked against
    corpus/FLAGGED.tsv, which is the record of what the gate actually said.
    """
    out = np.empty(d["n"], dtype=float)
    for i in range(d["n"]):
        s, fk = crosswind_integrated(d["kljun"][i], float(d["wdir_deg"][i]))
        out[i] = s[int(np.argmax(fk))]
    return out


def negative_fraction(f):
    """|negative lobe| / |f|, the quantity PROJECT_BRIEF.md quotes as 5.8-11.1%."""
    a = np.abs(f).sum()
    return float(-np.minimum(f, 0).sum() / a) if a > 0 else 0.0


# ------------------------------------------------------------------------- drawing

def draw_frame(ax, surf, show_pad=True, fg="w"):
    """Receptor, array, lake and the zero-pad boundary. Identical on every raster panel."""
    xc, _ = axes_m()
    if surf is not None:
        ax.contour(xc, xc, surf["water"], levels=[0.5], colors="#00b8ff", linewidths=0.9,
                   alpha=0.85)
    x0, x1, y0, y1 = ARRAY_XY
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#39ff14", lw=1.3,
                           zorder=6))
    if show_pad:
        e = (NRAW / 2) * DX                  # 1830 m: the last real cell edge
        ax.add_patch(Rectangle((-e, -e), 2 * e, 2 * e, fill=False, ec=fg, lw=0.6,
                               ls=":", alpha=0.55, zorder=5))
    ax.plot(0, 0, marker="*", ms=9, mfc=fg, mec="k" if fg == "w" else "w", mew=0.6,
            ls="none", zorder=7)


def draw_negative(ax, tgt, vmax, frac=1e-3):
    """Outline where the SIGNED target is meaningfully negative.

    The raw mask is single-cell LPDM shot noise scattered over the whole domain and
    contouring it directly buries the panel in confetti. Smoothing before the 0.5 crossing
    keeps only regions several cells across, which is what the 5.8-11.1% negative lobe
    actually is; the printed percentage is computed from the UNSMOOTHED field.
    """
    xc, _ = axes_m()
    neg = gaussian_filter((tgt < -vmax * frac).astype(float), 1.5)
    if neg.max() > 0.5:
        ax.contour(xc, xc, neg, levels=[0.5], colors="#00e5ff", linewidths=0.8,
                   linestyles="--", alpha=0.95)


def draw_wind(ax, wdir_deg, colour="w"):
    """Arrow along the MEAN FLOW; the source area must lie on the other side of it."""
    a = np.radians(wdir_deg)
    ux, uy = -np.sin(a), -np.cos(a)          # FROM wdir -> flow points the other way
    ax.annotate("", xy=(0.5 + 0.30 * ux, 0.5 + 0.30 * uy),
                xytext=(0.5 - 0.10 * ux, 0.5 - 0.10 * uy), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=colour, alpha=0.85))


def pair_norms(klj, tgt, floor=1e-4):
    """One LogNorm shared by the input and the target, and a SymLogNorm for the residual.

    Sharing the norm is the entire point: a pair whose panels are scaled independently
    always 'looks fine'. vmax is the larger of the two peaks, so the target being 4x
    Kljun's peak is visible as saturation rather than hidden by a rescale.
    """
    vmax = float(max(klj.max(), tgt.max()))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    lin = vmax * floor
    return LogNorm(vmin=lin, vmax=vmax), SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax,
                                                    base=10), vmax


def raster(ax, f, norm, cmap, ext, mask_below=None):
    """mask_below masks everything under the colour floor, so the panel shows the plotted
    dynamic range and nothing else. Without it a LogNorm paints every sub-floor cell in the
    colormap's darkest colour, which reads as a large solid feature -- Kljun's field outside
    its own validity radius came out as a black disc covering half the domain."""
    F = np.ma.masked_less_equal(f, mask_below) if mask_below is not None else f
    return ax.imshow(F, origin="lower", extent=ext, aspect="equal", cmap=cmap, norm=norm,
                     interpolation="nearest")


def stamp(ax, txt, loc="upper left", size=6.4, colour="w"):
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.98, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, txt, transform=ax.transAxes, color=colour, fontsize=size, ha=ha, va=va,
            family="monospace",
            bbox=dict(fc="k", ec="none", alpha=0.62, boxstyle="round,pad=0.25"))


def scalar_line(d, i):
    s = d["scalars"][i]
    nm = d["scalar_names"]
    v = dict(zip(nm, s))
    return (f"h={v['h']:.0f} m  u*={v['ustar']:.3f}  sig_v={v['sigma_v']:.3f}\n"
            f"L={v['L']:+.0f} m  wdir={d['wdir_deg'][i]:.0f}deg\n"
            f"sin={v['sin_wdir']:+.3f} cos={v['cos_wdir']:+.3f}")


# ------------------------------------------------------------------- figure: gallery

def gallery(d, surf, idx, title, outpath, dpi):
    """One row per record: the INPUT raster, the TARGET, and target - input.

    Three panels because two are not enough to see agreement: the residual is where a
    systematic orientation or scale error shows up as a dipole rather than as noise.
    """
    xc, xe = axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    nrow = len(idx)
    fig, axes = plt.subplots(nrow, 3, figsize=(11.4, 3.55 * nrow), squeeze=False)
    H = 3.55 * nrow                                    # figure height, inches
    fig.subplots_adjust(left=0.055, right=0.935, top=1 - 0.42 / H, bottom=0.85 / H,
                        hspace=0.15, wspace=0.17)

    for r, i in enumerate(idx):
        klj, tgt = d["kljun"][i], d["target"][i]
        lognorm, symnorm, vmax = pair_norms(klj, tgt)
        wd = float(d["wdir_deg"][i])

        for c, (F, name, norm, cmap, floor) in enumerate((
                (klj, "INPUT  Kljun FFP v1.42", lognorm, "magma", lognorm.vmin),
                (tgt, "TARGET  LES + backward LPDM", lognorm, "magma", lognorm.vmin),
                (tgt - klj, "TARGET - INPUT", symnorm, "RdBu_r", None))):
            ax = axes[r][c]
            im = raster(ax, F, norm, cmap, ext, mask_below=floor)
            if c == 1:
                draw_negative(ax, tgt, vmax)
            if c < 2:
                l50, l80 = source_area_levels(F)
                if np.isfinite(l50):
                    ax.contour(xc, xc, F, levels=[l80], colors="w", linewidths=0.7,
                               linestyles="--", alpha=0.8)
                    ax.contour(xc, xc, F, levels=[l50], colors="w", linewidths=1.0,
                               alpha=0.9)
            draw_frame(ax, surf, fg="w" if c < 2 else "k")
            draw_wind(ax, wd, colour="w" if c < 2 else "k")
            ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(xe[0], xe[-1])
            ax.tick_params(labelsize=6.5)
            if c:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("north  [m]", fontsize=7.5)
            if r == nrow - 1:
                ax.set_xlabel("east  [m]", fontsize=7.5)
            else:
                ax.set_xticklabels([])
            if r == 0:
                ax.set_title(name, fontsize=8.5, pad=4)

            # ONE colourbar for the shared input/target scale, one for the residual --
            # a bar per panel implies three scales when there are two.
            if c:
                cax = ax.inset_axes([1.020, 0.0, 0.030, 1.0])
                cb = fig.colorbar(im, cax=cax)
                cb.ax.tick_params(labelsize=5.6)
                cb.set_label("m$^{-2}$", fontsize=5.8)

        stamp(axes[r][0], scalar_line(d, i))
        stamp(axes[r][1],
              f"integral {d['integral'][i]:.3f}\n"
              f"array    {100 * d['array_share'][i]:.2f}%\n"
              f"neg lobe {100 * negative_fraction(d['target'][i]):.1f}%\n"
              f"peak_x   {d['peak_x_m'][i]:.0f} m", loc="upper right")
        stamp(axes[r][2],
              f"{d['run_id'][i]}  [{d['split'][i]}]\n{d['datetime'][i]}",
              loc="lower left", colour="k")
        axes[r][2].texts[-1].set_bbox(dict(fc="w", ec="none", alpha=0.75,
                                           boxstyle="round,pad=0.25"))

    fig.suptitle(title, fontsize=11.5, y=1 - 0.10 / H)
    fig.text(0.055, 0.12 / H, CAPTION, fontsize=7.0, va="bottom", linespacing=1.5)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="w")
    plt.close(fig)
    print(f"  wrote {outpath}  ({nrow} records)")


# ------------------------------------------------------------------ figure: anatomy

def anatomy(d, surf, i, outpath, dpi):
    """One pair, taken apart: rasters, the crosswind-integrated profile, the six scalars."""
    xc, xe = axes_m()
    ext = [xe[0], xe[-1], xe[0], xe[-1]]
    klj, tgt = d["kljun"][i], d["target"][i]
    lognorm, symnorm, vmax = pair_norms(klj, tgt)
    wd = float(d["wdir_deg"][i])

    fig = plt.figure(figsize=(13.6, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.32, 1.0], hspace=0.30, wspace=0.20,
                          left=0.055, right=0.965, top=0.885, bottom=0.075)

    for c, (F, name, norm, cmap, floor) in enumerate((
            (klj, "INPUT: Kljun et al. (2015) FFP v1.42", lognorm, "magma", lognorm.vmin),
            (tgt, "TARGET: LES + backward LPDM (signed)", lognorm, "magma", lognorm.vmin),
            (tgt - klj, "TARGET - INPUT (what the FNO predicts)", symnorm, "RdBu_r",
             None))):
        ax = fig.add_subplot(gs[0, c])
        im = raster(ax, F, norm, cmap, ext, mask_below=floor)
        if c == 1:
            draw_negative(ax, tgt, vmax)
        if c < 2:
            l50, l80 = source_area_levels(F)
            ax.contour(xc, xc, F, levels=[l80], colors="w", linewidths=0.8, linestyles="--")
            ax.contour(xc, xc, F, levels=[l50], colors="w", linewidths=1.2)
        draw_frame(ax, surf, fg="w" if c < 2 else "k")
        draw_wind(ax, wd, colour="w" if c < 2 else "k")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("east  [m]", fontsize=8)
        if c == 0:
            ax.set_ylabel("north  [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6.5)

    # --- crosswind-integrated profile on the wind axis
    ax = fig.add_subplot(gs[1, 0])
    s, fk = crosswind_integrated(klj, wd)
    _, ft = crosswind_integrated(tgt, wd)
    ax.plot(s, fk, color="#c44e52", lw=1.4, label="input (Kljun)")
    ax.plot(s, ft, color="#4c72b0", lw=1.4, label="target (LES)")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    a_s = array_upwind_span(wd)
    ax.axvspan(a_s[0], a_s[1], color="#39ff14", alpha=0.20, lw=0,
               label=f"solar array, {a_s[0]:+.0f} to {a_s[1]:+.0f} m on this wind axis")
    ax.set_xlim(-300, 1830)
    ax.set_xlabel("upwind distance $s$  [m]", fontsize=8)
    ax.set_ylabel("$f_y(s)$  [m$^{-1}$]", fontsize=8)
    ax.set_title("crosswind-integrated", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.4, frameon=False)

    xk_i = s[int(np.argmax(fk))]

    # --- cumulative source area
    ax = fig.add_subplot(gs[1, 1])
    for f, col, nm in ((fk, "#c44e52", "input"), (ft, "#4c72b0", "target")):
        p = np.maximum(f, 0)
        cum = np.cumsum(p) / max(p.sum(), 1e-30)
        ax.plot(s, cum, color=col, lw=1.4, label=nm)
        for q, ls in ((0.5, ":"), (0.8, "--")):
            k = int(np.searchsorted(cum, q))
            if k < len(s):
                ax.plot([s[k], s[k]], [0, q], color=col, lw=0.8, ls=ls)
    for q in (0.5, 0.8):
        ax.axhline(q, color="k", lw=0.5, ls=":")
    ax.set_xlim(0, 1830); ax.set_ylim(0, 1.02)
    ax.set_xlabel("upwind distance $s$  [m]", fontsize=8)
    ax.set_ylabel("cumulative fraction of $f_y>0$", fontsize=8)
    ax.set_title("source-area distance", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.4, frameon=False, loc="lower right")

    # --- the six scalars, normalised and raw
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    z = (d["scalars"][i] - d["norm_mean"]) / d["norm_std"]
    rows = [f"{'input scalar':<10s}{'raw':>12s}{'z (train norm)':>16s}"]
    rows.append("-" * 38)
    for k, nm in enumerate(d["scalar_names"]):
        u = SCALAR_UNITS[nm]
        raw = d["scalars"][i][k]
        rows.append(f"{nm:<10s}{raw:>12.4g}{z[k]:>16.2f}   {u}")
    rows += ["", f"{'wdir':<10s}{d['wdir_deg'][i]:>12.1f}   deg (FROM)",
             f"{'z_i achieved':<14s}{d['zi_achieved_m'][i]:>8.0f}   m",
             "", "target diagnostics", "-" * 38,
             f"{'integral':<16s}{d['integral'][i]:>10.4f}   (asymptote 1 - z_m/z_i)",
             f"{'array share':<16s}{100*d['array_share'][i]:>10.2f} %",
             f"{'negative lobe':<16s}{100*negative_fraction(tgt):>10.2f} % of |f|",
             f"{'peak_x':<16s}{d['peak_x_m'][i]:>10.0f}   m",
             f"{'centroid':<16s}{d['centroid_dist_m'][i]:>10.0f}   m",
             f"{'peak ratio':<16s}{tgt.max()/max(klj.max(),1e-30):>10.2f}   target/input",
             "", f"{'gate_state':<16s}{d['gate_state'][i]:>10s}  (seed stationarity)",
             f"{'G2b/G3b':<16s}{gate_flags(d, i, xk_i):>10s}  (FLAGGED.tsv, not a filter)"]
    ax.text(0.0, 1.0, "\n".join(rows), transform=ax.transAxes, va="top", ha="left",
            fontsize=7.4, family="monospace")

    fig.suptitle(f"{d['run_id'][i]}   {d['datetime'][i]}   split={d['split'][i]}   "
                 f"wind FROM {wd:.0f}$\\degree$", fontsize=12)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="w")
    plt.close(fig)
    print(f"  wrote {outpath}  ({d['run_id'][i]})")


# ------------------------------------------------------------------- figure: inputs

def inputs_figure(d, outpath, dpi):
    """The input space the emulator has to cover, split by split."""
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.6))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.895, bottom=0.075, hspace=0.55,
                        wspace=0.28)
    sp = d["split"]
    order = ["train", "val", "test"]

    panels = [("h", "$h$  [m]", None), ("ustar", "$u_*$  [m s$^{-1}$]", None),
              ("sigma_v", "$\\sigma_v$  [m s$^{-1}$]", None),
              ("inv_L", "$1/L$  [m$^{-1}$]", "inv_L")]
    for k, (nm, lab, src) in enumerate(panels):
        ax = axes[0][k]
        v = d["inv_L"] if src else d["scalars"][:, d["scalar_names"].index(nm)]
        lo, hi = np.percentile(v, [0.5, 99.5])
        bins = np.linspace(lo, hi, 42)
        for s in order:
            ax.hist(v[sp == s], bins=bins, histtype="step", lw=1.4, density=True,
                    color=SPLIT_COLOUR[s], label=f"{s} (n={int((sp == s).sum())})")
        ax.set_xlabel(lab, fontsize=8.5)
        ax.set_ylabel("density", fontsize=8.5)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=6.6, frameon=False)

    # wind rose -- sin/cos_wdir are two of the six inputs, so this IS an input panel
    axes[1][0].remove()
    ax = fig.add_subplot(2, 4, 5, projection="polar")
    edges = np.radians(np.arange(-11.25, 360, 22.5))
    wd = np.radians(d["wdir_deg"] % 360.0)
    wdw = np.where(wd > edges[-1], wd - 2 * np.pi, wd)
    cnt, _ = np.histogram(wdw, bins=edges)
    ax.bar(0.5 * (edges[:-1] + edges[1:]), cnt, width=np.radians(21.0), color="#4c72b0",
           alpha=0.85, ec="w", lw=0.5)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_yticks(np.linspace(0, cnt.max(), 4)[1:])
    ax.set_title("wind direction FROM  (corpus rose)", fontsize=9, pad=16)
    ax.tick_params(labelsize=6.0)

    # array share -- the reason the rose matters
    ax = axes[1][1]
    a = 100 * d["array_share"]
    ax.hist(a, bins=np.linspace(0, max(a.max(), 1.0), 60), color="#39ff14", ec="k", lw=0.3)
    ax.axvline(5.0, color="r", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("array share of the footprint  [%]", fontsize=8.5)
    ax.set_ylabel("records", fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.text(0.97, 0.95, f"median {np.median(a):.2f}%\n>5%: {int((a > 5).sum())} of {d['n']}"
                        f"  ({100 * (a > 5).mean():.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            family="monospace")

    # array share against direction -- the skew, in one panel
    ax = axes[1][2]
    for s in order:
        m = sp == s
        ax.scatter(d["wdir_deg"][m] % 360, 100 * d["array_share"][m], s=5, alpha=0.55,
                   color=SPLIT_COLOUR[s], lw=0, label=s)
    ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("wind FROM  [deg]", fontsize=8.5)
    ax.set_ylabel("array share  [%]", fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.6, frameon=False, markerscale=2)
    ax.set_title("the signal lives in northerly flow", fontsize=8.5)

    # h against 1/L, coloured by split -- joint coverage, not just marginals
    ax = axes[1][3]
    hh = d["scalars"][:, d["scalar_names"].index("h")]
    for s in order:
        m = sp == s
        ax.scatter(d["inv_L"][m], hh[m], s=5, alpha=0.55, color=SPLIT_COLOUR[s], lw=0)
    ax.set_xlabel("$1/L$  [m$^{-1}$]", fontsize=8.5)
    ax.set_ylabel("$h$  [m]", fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.set_title("joint coverage of the two stability inputs", fontsize=8.5)

    fig.suptitle(f"corpus inputs -- the six scalars the emulator sees  "
                 f"(n = {d['n']}; normalisation from the train split alone)", fontsize=12)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="w")
    plt.close(fig)
    print(f"  wrote {outpath}")


# ------------------------------------------------------------------- figure: sanity

def sanity_figure(d, outpath, dpi):
    """Pair-level checks that do not need the LES: is every target the right SHAPE?"""
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.2))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.885, bottom=0.08, hspace=0.34,
                        wspace=0.28)
    sp, order = d["split"], ["train", "val", "test"]

    # integral, against the G2b window corpus_monitor.py already defines
    ax = axes[0][0]
    v = d["integral"]
    ax.hist(v, bins=np.linspace(min(0.4, v.min()), max(1.7, v.max()), 60), color="#8172b3",
            ec="k", lw=0.3)
    for b in (0.6, 1.5):
        ax.axvline(b, color="r", lw=1.0, ls="--")
    ax.axvline(1 - 30.0 / np.median(d["zi_achieved_m"]), color="k", lw=1.0)
    ax.set_xlabel("footprint integral", fontsize=8.5)
    ax.set_ylabel("records", fontsize=8.5)
    ax.set_title("G2b window [0.6, 1.5] (red);  $1-z_m/z_i$ asymptote (black)",
                 fontsize=8.2)
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.95, f"outside: {int(((v < 0.6) | (v > 1.5)).sum())} of {d['n']}",
            transform=ax.transAxes, va="top", fontsize=7, family="monospace")

    # G3b: LES peak DISTANCE / Kljun peak distance. corpus_monitor.py:149 compares
    # les["peak_x"] against klj["peak_x"] -- both are locations, not amplitudes. The
    # amplitude ratio is a different number and is NOT this gate; it is reported below the
    # histogram as a diagnostic and nothing thresholds it.
    ax = axes[0][1]
    xk = kljun_peak_distance(d)
    r = d["peak_x_m"] / np.maximum(xk, 1e-30)
    fin = np.isfinite(r) & (xk > 0)
    ax.hist(r[fin], bins=np.logspace(np.log10(max(r[fin].min(), 1e-2)),
                                     np.log10(r[fin].max()), 60),
            color="#937860", ec="k", lw=0.3)
    ax.set_xscale("log")
    for b in (0.4, 2.5):
        ax.axvline(b, color="r", lw=1.0, ls="--")
    ax.set_xlabel("LES peak distance / Kljun peak distance", fontsize=8.5)
    ax.set_ylabel("records", fontsize=8.5)
    ax.set_title("G3b window [0.4, 2.5] (red)", fontsize=8.2)
    ax.tick_params(labelsize=7)
    amp = (d["target"].reshape(d["n"], -1).max(axis=1)
           / np.maximum(d["kljun"].reshape(d["n"], -1).max(axis=1), 1e-30))
    ax.text(0.02, 0.95,
            f"outside: {int(((r < 0.4) | (r > 2.5)).sum())} of {d['n']}\n"
            f"peak AMPLITUDE ratio (not a gate):\n"
            f"  median {np.median(amp):.2f}, p5-p95 "
            f"{np.percentile(amp, 5):.2f}-{np.percentile(amp, 95):.2f}",
            transform=ax.transAxes, va="top", fontsize=6.6, family="monospace")

    # negative lobe
    ax = axes[0][2]
    nf = 100 * np.array([negative_fraction(t) for t in d["target"]])
    ax.hist(nf, bins=60, color="#00b8ff", ec="k", lw=0.3)
    ax.axvspan(5.8, 11.1, color="k", alpha=0.12, lw=0)
    ax.set_xlabel("negative lobe  [% of $|f|$]", fontsize=8.5)
    ax.set_ylabel("records", fontsize=8.5)
    ax.set_title("signed target; grey = the 5.8-11.1% PROJECT_BRIEF.md quotes", fontsize=8.2)
    ax.tick_params(labelsize=7)

    # the zero pad, checked but not PLOTTED -- a bar chart of two exact zeros is two
    # invisible bars, which says nothing. The number goes in the subtitle instead.
    pad = np.ones((NPAD, NPAD), bool)
    pad[PAD:PAD + NRAW, PAD:PAD + NRAW] = False
    worst = max(float(np.abs(d["target"][:, pad]).max()),
                float(np.abs(d["kljun"][:, pad]).max()))

    # mean input and mean target, side by side and on ONE scale: the corpus-wide check
    # that the frame is not silently rotating between the two channels.
    xc, xe = axes_m()
    mk, mt = d["kljun"].mean(axis=0), d["target"].mean(axis=0)
    mnorm = LogNorm(vmin=max(mk.max(), mt.max()) * 1e-4, vmax=max(mk.max(), mt.max()))
    for ax, F, nm in ((axes[1][0], mk, "mean of all 1366 INPUTS"),
                      (axes[1][2], mt, "mean of all 1366 TARGETS")):
        im = ax.imshow(np.ma.masked_less_equal(F, mnorm.vmin), origin="lower",
                       extent=[xe[0], xe[-1], xe[0], xe[-1]], cmap="magma", norm=mnorm,
                       interpolation="nearest")
        draw_frame(ax, None)
        ax.set_title(f"{nm} -- should be the wind rose", fontsize=8.2)
        ax.set_xlabel("east  [m]", fontsize=8.5)
        ax.set_ylabel("north  [m]", fontsize=8.5)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6.5)

    # peak distance: target against input, the orientation check in one scatter
    ax = axes[1][1]
    # a few metres of jitter, because both axes are quantised to the 30 m cell and without
    # it a thousand records stack into thirty dots
    rng = np.random.default_rng(0)
    bad = (r < 0.4) | (r > 2.5)
    for sname in order:
        m = (sp == sname) & ~bad
        ax.scatter(xk[m] + rng.normal(0, 3.5, m.sum()),
                   d["peak_x_m"][m] + rng.normal(0, 3.5, m.sum()),
                   s=5, alpha=0.45, color=SPLIT_COLOUR[sname], lw=0, label=sname)
    ax.scatter(xk[bad], d["peak_x_m"][bad], s=14, facecolors="none", edgecolors="r",
               lw=0.5, alpha=0.7, label=f"outside G3b (n={int(bad.sum())})")
    # Both peaks land on a 30 m grid and 95% of the corpus sits inside 400 m, so the full
    # range is 95% empty space. Clip the view and SAY how many records it hides.
    hi = float(max(np.percentile(xk, 99.5), np.percentile(d["peak_x_m"], 99.5)))
    lim = [0, hi * 1.15]
    ax.plot(lim, lim, "k-", lw=0.8)
    for f_ in (0.4, 2.5):
        ax.plot(lim, [f_ * v for v in lim], "r", lw=0.7, ls=":")
    n_out = int(((xk > lim[1]) | (d["peak_x_m"] > lim[1])).sum())
    if n_out:
        ax.text(0.98, 0.04, f"{n_out} record(s) beyond the axes", transform=ax.transAxes,
                ha="right", fontsize=6.4, color="0.35")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("input peak upwind distance  [m]", fontsize=8.5)
    ax.set_ylabel("target peak distance  [m]", fontsize=8.5)
    ax.set_title("both peaks on the SAME wind axis; both quantised to the 30 m cell",
                 fontsize=8.2)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.4, frameon=False, markerscale=1.6, loc="lower right")

    fig.suptitle("pair sanity -- every panel is computed from corpus.h5 alone\n"
                 f"zero pad: max |value| over all {d['n']} records and both channels = "
                 f"{worst:.3e}"
                 f"{'  (exactly zero)' if worst == 0 else '  -- NOT ZERO, investigate'};  "
                 "all values finite",
                 fontsize=11)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="w")
    plt.close(fig)
    print(f"  wrote {outpath}")
    return dict(pad_max=worst,
                n_g2b=int(((d["integral"] < 0.6) | (d["integral"] > 1.5)).sum()),
                n_g3b=int(((r < 0.4) | (r > 2.5)).sum()),
                neg_median=float(np.median(nf)))


# ---------------------------------------------------------------------- selection

def gate_flags(d, i, kljun_peak_x):
    """The two per-case gates corpus_monitor.py defines, for ONE record.

    Recomputed rather than read from FLAGGED.tsv so the figure is self-contained; the
    counts printed by main() are checked against that file, which is the record.
    """
    f = []
    if not (0.6 <= d["integral"][i] <= 1.5):
        f.append("G2b")
    if kljun_peak_x > 0:
        r = d["peak_x_m"][i] / kljun_peak_x
        if not (0.4 <= r <= 2.5):
            f.append("G3b")
    return ",".join(f) if f else "-"


def flagged_counts(path):
    """What the gate ACTUALLY said, from corpus/FLAGGED.tsv. None if the file is absent.

    Rule 7: assert on the artifact. The G2b/G3b counts this script re-derives from the
    rasters are only trustworthy if they reproduce the file the pipeline wrote, so they
    are printed side by side and a disagreement is called out rather than averaged over.
    """
    if not os.path.exists(path):
        return None
    g2 = g3 = tot = 0
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        col = line.rstrip("\n").split("\t")
        if len(col) < 3:
            continue
        tot += 1
        g2 += "G2b" in col[2]
        g3 += "G3b" in col[2]
    return dict(total=tot, g2b=g2, g3b=g3)


def pick_by_octant(d):
    """One record per 45 deg sector, the one with the largest array share in it.

    Largest-share rather than random because the array is the site-specific signal: if the
    frame or the rotation is wrong, the sector with the array in view is where it shows.
    """
    out = []
    for name, c in OCTANTS:
        dd = (d["wdir_deg"] - c + 180.0) % 360.0 - 180.0
        m = np.abs(dd) <= 22.5
        if not m.any():
            continue
        idx = np.where(m)[0]
        out.append(int(idx[np.argmax(d["array_share"][idx])]))
    return out


def pick_random(d, split, k, seed):
    idx = np.where(d["split"] == split)[0]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(idx, size=min(k, len(idx)), replace=False).tolist())


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="corpus/corpus.h5")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--surface", default="data/grid30_raised",
                    help="production surface dir, for the lake outline; optional")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--seed", type=int, default=20260901,
                    help="seed for the random per-split galleries; fixed so the figures "
                         "are reproducible from the file alone")
    a = ap.parse_args()

    if not os.path.exists(a.h5):
        sys.exit(f"FATAL: {a.h5} does not exist")
    os.makedirs(a.outdir, exist_ok=True)
    print(f"reading {a.h5}")
    d = load_corpus(a.h5)
    print(f"  {d['n']} records, splits " +
          ", ".join(f"{s}={int((d['split'] == s).sum())}"
                    for s in ("train", "val", "test")))
    surf = load_surface(a.surface)
    print(f"  surface overlay: {'on' if surf else 'OFF'}  ({a.surface})")

    o = lambda n: os.path.join(a.outdir, n)

    # the pair with the most array in view: the one case where the site signal is largest
    i_best = int(np.argmax(d["array_share"]))
    anatomy(d, surf, i_best, o("pair_anatomy_array.png"), a.dpi)
    # ... and a typical one: median array share, so it is not a hand-picked success
    i_typ = int(np.argsort(d["array_share"])[d["n"] // 2])
    anatomy(d, surf, i_typ, o("pair_anatomy_typical.png"), a.dpi)

    gallery(d, surf, pick_by_octant(d),
            "input / target pairs, one per 45$\\degree$ wind sector "
            "(the record with the most array in view in each)",
            o("pairs_by_direction.png"), a.dpi)

    top = np.argsort(d["array_share"])[::-1][:6]
    gallery(d, surf, sorted(int(i) for i in top),
            "the six records with the most solar array in the footprint "
            "-- the site-specific signal the emulator exists to learn",
            o("pairs_array_signal.png"), a.dpi)

    for s in ("train", "val", "test"):
        gallery(d, surf, pick_random(d, s, 6, a.seed),
                f"six random {s} records (seed {a.seed}) -- unselected, so this is what "
                f"the split actually looks like",
                o(f"pairs_random_{s}.png"), a.dpi)

    inputs_figure(d, o("corpus_inputs.png"), a.dpi)
    st = sanity_figure(d, o("pairs_sanity.png"), a.dpi)

    print("\nsummary")
    print(f"  zero pad max |value|      {st['pad_max']:.3e}  "
          f"({'exactly zero' if st['pad_max'] == 0 else 'NOT ZERO -- investigate'})")
    fl = flagged_counts(os.path.join(os.path.dirname(a.h5) or ".", "FLAGGED.tsv"))
    ref = (lambda k: f"   (FLAGGED.tsv: {fl[k]})") if fl else (lambda k: "")
    print(f"  outside G2b [0.6, 1.5]    {st['n_g2b']} of {d['n']}{ref('g2b')}")
    print(f"  outside G3b [0.4, 2.5]    {st['n_g3b']} of {d['n']}{ref('g3b')}")
    if fl:
        for nm, mine, theirs in (("G2b", st["n_g2b"], fl["g2b"]),
                                 ("G3b", st["n_g3b"], fl["g3b"])):
            if mine != theirs:
                print(f"    NOTE: {nm} re-derived from the rasters is {mine}, the "
                      f"pipeline recorded {theirs}. The figure plots the re-derived "
                      f"value; FLAGGED.tsv is the record.")
    print(f"  median negative lobe      {st['neg_median']:.2f}% of |f|")
    print(f"  most array in view        {d['run_id'][i_best]}  "
          f"{100 * d['array_share'][i_best]:.1f}%")


if __name__ == "__main__":
    main()
