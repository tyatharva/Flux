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


_TOPO = None


def model_terrain():
    """USGS 3DEP terrain sampled at every cell centre of the 128 padded LES frame, metres ASL
    (the model surface is the same DEM tapered at the wrap seams; the untapered one is drawn).
    Cached at data/raw/dem_les_grid.npy."""
    global _TOPO
    if _TOPO is None:
        import os
        cache = os.path.join(D.REPO, "data", "raw", "dem_les_grid.npy")
        if os.path.exists(cache):
            _TOPO = np.load(cache)
            return _TOPO
        import rasterio
        from pyproj import Transformer
        g = os.path.join(D.REPO, "data", "grid30_raised")
        meta = np.load(os.path.join(g, "meta.npy"), allow_pickle=True).item()
        x0, y0, dx = float(meta["x0"]), float(meta["y0"]), float(meta["dx"])
        ii = np.arange(D.N) - D.PAD
        XX, YY = np.meshgrid(x0 + ii * dx, y0 + ii * dx)
        lon, lat = Transformer.from_crs("EPSG:3071", "EPSG:4269", always_xy=True).transform(XX, YY)
        with rasterio.open(os.path.join(D.REPO, "data", "raw", "output_USGS10m.tif")) as dem:
            from rasterio.transform import rowcol
            r, c = rowcol(dem.transform, lon.ravel(), lat.ravel())
            r, c = np.asarray(r, int), np.asarray(c, int)
            Z = dem.read(1)
            t = Z[np.clip(r, 0, Z.shape[0] - 1), np.clip(c, 0, Z.shape[1] - 1)].astype(float).reshape(D.N, D.N)
            t[t == dem.nodata] = np.nan
        np.save(cache, t)
        _TOPO = t
    return _TOPO


_IMG = None


def imagery_on_grid(npx=1024, zoom=16):
    """Esri World Imagery resampled onto the 128-cell LES frame (tower-relative metres, north
    up), npx x npx RGB. Cached at data/raw/esri_les_grid.npy."""
    global _IMG
    if _IMG is None:
        import os
        cache = os.path.join(D.REPO, "data", "raw", f"esri_les_grid_{npx}_z{zoom}.npy")
        if os.path.exists(cache):
            _IMG = np.load(cache)
            return _IMG
        from pyproj import Transformer
        from ml_cfm import fig_domain as FD
        g = os.path.join(D.REPO, "data", "grid30_raised")
        meta = np.load(os.path.join(g, "meta.npy"), allow_pickle=True).item()
        tx, ty = float(meta["tower_x"]), float(meta["tower_y"])
        half = D.N * D.DX / 2
        u = (np.arange(npx) + 0.5) / npx * 2 * half - half
        XX, YY = np.meshgrid(tx + u, ty + u)
        mx, my = Transformer.from_crs("EPSG:3071", "EPSG:3857", always_xy=True).transform(XX, YY)
        img, ext = FD.mosaic(mx.min() - 100, mx.max() + 100, my.min() - 100, my.max() + 100, zoom)
        h, w = img.shape[:2]
        col = np.clip(((mx - ext[0]) / (ext[1] - ext[0]) * w).astype(int), 0, w - 1)
        row = np.clip(((ext[3] - my) / (ext[3] - ext[2]) * h).astype(int), 0, h - 1)
        _IMG = img[row, col]
        np.save(cache, _IMG)
    return _IMG


def background(ax, half, alpha=0.5):
    """The imagery under a footprint panel, washed towards white by (1 - alpha)."""
    img = imagery_on_grid()
    ax.imshow(img, extent=[-half, half, -half, half], origin="lower", interpolation="bilinear", alpha=alpha, zorder=1)


def footprint_panel(ax, f, vmax, statics, wdir_deg, letter=None, half=1830.0, cmap="turbo", terrain=True, alpha=1.0, arrow=False):
    """The footprint as cells (no interpolation) on the turbo map over translucent Esri imagery,
    example_plot.py style: letter top-left, no ticks. Returns the mappable."""
    import fig_corpus_pairs as FCP
    from matplotlib.patches import Rectangle
    xc = (np.arange(D.N) - D.IJ_RECEPTOR) * D.DX
    xe = np.concatenate([xc - D.DX / 2, [xc[-1] + D.DX / 2]])
    lv = levels(vmax)
    ff = np.asarray(f, np.float64)
    lf = np.ma.masked_less_equal(np.log10(np.maximum(ff, 10 ** lv[0])), lv[0])
    ax.set_facecolor("white")
    if terrain:
        background(ax, D.N * D.DX / 2)
    m = ax.pcolormesh(xe, xe, lf, cmap=cmap, vmin=lv[0], vmax=lv[-1], shading="flat", alpha=alpha, zorder=3)
    x0, x1, y0, y1 = D.ARRAY_XY
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ff00ff", lw=1.6, zorder=6))
    ax.plot(0, 0, marker="*", ms=11, mfc="w", mec="k", mew=0.8, zorder=7)
    if arrow:
        FCP.draw_wind(ax, wdir_deg, colour="k")
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    if letter:
        ax.text(0.03, 0.965, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left", zorder=8)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    return m


def table_panel(ax, values, letter=None, fontsize=11):
    """values: dict key -> (kljun, fno, cfm). Best of three shaded."""
    ax.axis("off")
    cells, colours = [], []
    for key, label, fmt, hi in TABLE_ROWS:
        v = np.asarray(values[key], float)
        txt = [fmt.format(x) for x in v]
        best = fmt.format(np.nanmax(v) if hi else np.nanmin(v))     # ties at the printed precision all shade
        cells.append([label] + txt)
        colours.append(["#f7f7f7"] + ["#cfe8cf" if t == best else "white" for t in txt])
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
        ax.text(0.03, 0.965, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left", zorder=8,
                bbox=dict(fc="white", ec="none", pad=1))
    return tb


def crosswind_panel(ax, fields, wdir_deg, letter=None, legend=False, xlabel=True, ylabel=True, bands=None):
    """fields: list of (key, field, label) drawn in order; bands: (s, p5, p25, p50, p75, p95) in 1e-3 m^-1."""
    import fig_corpus_pairs as FCP
    if bands is not None:
        s_, p5, p25, p50, p75, p95 = bands
        ax.fill_between(s_, p5, p95, color=COL["cfm"], alpha=0.22, lw=0, label="CFM: 90% of samples")
    for key, fld, lab in fields:
        s_, fy = FCP.crosswind_integrated(np.asarray(fld, np.float64), float(wdir_deg))
        ax.plot(s_, fy * 1e3, color=COL[key], lw=2.0 if key == "les" else 1.5, label=lab, zorder=5 if key == "les" else 4)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(-50, 1500); ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.grid(alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=9, length=3)
    if xlabel:
        ax.set_xlabel("upwind distance from the tower [m]", fontsize=9.5)
    if ylabel:
        ax.set_ylabel(r"crosswind-integrated footprint $f_y$  [10$^{-3}$ m$^{-1}$]", fontsize=9.5)
    if legend:
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    if letter:
        ax.text(0.03, 0.965, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left", zorder=8)
