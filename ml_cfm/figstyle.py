"""Shared drawing for the report figures: a footprint panel on the turbo colour map in the
example_plot.py style (filled contours of log10 f, letter top-left, gridline labels inside the
frame, no titles), the metric table panel, and the crosswind panel."""
import numpy as np

from ml import data as D                      # noqa: E402

COL = dict(les="#1f4e79", kljun="#c0392b", fno="#2e8b57", cfm="#7b3fa0")
MODEL_COL = dict(Kljun="#c0392b", FNO="#2e8b57", CFM="#7b3fa0")
DECADES = 4.0
N_LEVELS = 33
# (key, label, format, higher-is-better)
TABLE_ROWS = (("peak_x", "peak error [m]", "{:.0f}", False), ("centroid", "centroid error [m]", "{:.0f}", False),
              ("integral", "integral error", "{:.2f}", False), ("overlap80", "overlap80", "{:.2f}", True),
              ("rel_l2", "rel. L2", "{:.2f}", False), ("sw1_m", "sliced W1 [m]", "{:.0f}", False),
              ("js_dist", "JS distance", "{:.2f}", False), ("ms_ssim", "MS-SSIM", "{:.2f}", True))


def levels(vmax):
    top = np.log10(vmax)
    return np.linspace(top - DECADES, top, N_LEVELS)


def footprint_panel(ax, f, vmax, statics, wdir_deg, letter=None, half=1830.0, cmap="turbo", labels="inside",
                    contour=True):
    """Filled log10 contours of f (m^-2) on the turbo map; returns the mappable."""
    import fig_corpus_pairs as FCP
    from matplotlib.patches import Rectangle
    xc = D.cell_coords_m() if hasattr(D, "cell_coords_m") else (np.arange(D.N) - D.IJ_RECEPTOR) * D.DX
    lv = levels(vmax)
    ff = np.asarray(f, np.float64)
    lf = np.ma.masked_less_equal(np.log10(np.maximum(ff, 10 ** lv[0])), lv[0])
    ax.set_facecolor("#ececec")
    if contour:
        m = ax.contourf(xc, xc, lf, levels=lv, cmap=cmap, extend="neither", zorder=2)
    else:
        m = ax.pcolormesh(xc, xc, np.ma.masked_less_equal(lf, lv[0]), cmap=cmap, vmin=lv[0], vmax=lv[-1], shading="nearest", zorder=2)
    water = statics["water"] > 0.5
    ax.contour(xc, xc, water, levels=[0.5], colors="#5fd0ff", linewidths=1.0, zorder=3)
    x0, x1, y0, y1 = D.ARRAY_XY
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#39ff14", lw=1.5, zorder=6))
    ax.plot(0, 0, marker="*", ms=10, mfc="w", mec="k", mew=0.7, zorder=7)
    FCP.draw_wind(ax, wdir_deg, colour="k")
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal", adjustable="box")
    ticks = [-1000, 0, 1000]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t/1000:+.0f} km" if t else "0" for t in ticks]); ax.set_yticklabels([f"{t/1000:+.0f} km" if t else "0" for t in ticks])
    ax.grid(color="w", alpha=0.45, lw=0.5, zorder=4)
    if labels == "inside":
        ax.tick_params(axis="x", direction="in", labeltop=True, labelbottom=False, top=True, pad=-14, labelsize=8.5, length=3)
        ax.tick_params(axis="y", direction="in", labelleft=True, pad=-30, labelsize=8.5, length=3)
        for lab in ax.get_yticklabels():
            lab.set_ha("left")
    else:
        ax.tick_params(labelsize=8.5, length=3)
    if letter:
        ax.text(0.03, 0.965, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left", zorder=8)
    for s in ax.spines.values():
        s.set_linewidth(0.9)
    return m


def table_panel(ax, values, letter=None, fontsize=11):
    """values: dict key -> (kljun, fno, cfm). Best of three shaded."""
    ax.set_box_aspect(1); ax.axis("off")
    cells, colours = [], []
    for key, label, fmt, hi in TABLE_ROWS:
        v = np.asarray(values[key], float)
        best = int(np.nanargmax(v) if hi else np.nanargmin(v))
        cells.append([label] + [fmt.format(x) for x in v])
        colours.append(["#f7f7f7"] + ["#cfe8cf" if j == best else "white" for j in range(3)])
    tb = ax.table(cellText=cells, colLabels=["", "Kljun", "FNO", "CFM"], cellColours=colours,
                  loc="center", cellLoc="center", colWidths=[0.44, 0.185, 0.185, 0.185], bbox=[0.0, 0.0, 1.0, 1.0])
    tb.auto_set_font_size(False); tb.set_fontsize(fontsize)
    for (ri, ci), cell in tb.get_celld().items():
        cell.set_edgecolor("#bbbbbb"); cell.set_linewidth(0.6)
        if ci == 0 and ri > 0:
            cell.get_text().set_ha("left"); cell.PAD = 0.04
        if ri == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#e3e3e3")
    if letter:
        ax.text(0.03, 0.965, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left", zorder=8,
                bbox=dict(fc="white", ec="none", pad=1))
    return tb


def crosswind_panel(ax, fields, wdir_deg, letter=None, legend=False, xlabel=True, ylabel=True, bands=None):
    """fields: list of (key, field, label) drawn in order; bands: (s, p5, p25, p50, p75, p95) in 1e-3 m^-1."""
    import fig_corpus_pairs as FCP
    if bands is not None:
        s_, p5, p25, p50, p75, p95 = bands
        ax.fill_between(s_, p5, p95, color=COL["cfm"], alpha=0.15, lw=0, label="CFM: 90% of samples")
        ax.fill_between(s_, p25, p75, color=COL["cfm"], alpha=0.30, lw=0, label="CFM: 50% of samples")
    for key, fld, lab in fields:
        s_, fy = FCP.crosswind_integrated(np.asarray(fld, np.float64), float(wdir_deg))
        ax.plot(s_, fy * 1e3, color=COL[key], lw=2.0 if key == "les" else 1.5, label=lab, zorder=5 if key == "les" else 4)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(-50, 1500); ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.grid(alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=8.5, length=3)
    ax.set_box_aspect(1)
    if xlabel:
        ax.set_xlabel("upwind distance from the tower [m]", fontsize=9.5)
    if ylabel:
        ax.set_ylabel(r"crosswind-integrated footprint $f_y$  [10$^{-3}$ m$^{-1}$]", fontsize=9.5)
    if legend:
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    if letter:
        ax.text(0.03, 0.965, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left", zorder=8)
