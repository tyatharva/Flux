"""Corpus I/O for the emulator: the split loader with its test guard, the file's own
normalisation, the static surface fields on the padded grid, and the cone mask rebuilt
from the file's own rule.

THE TEST SPLIT IS OFF LIMITS. `load_split("test")` raises TestSplitForbidden unless
`allow_test=True` is passed explicitly. No module under ml/ passes it; the only place the
flag can be set is the `--allow-test` option of ml/evaluate.py, which exists so the test
evaluation can be run deliberately, by hand, once. Every call is appended to
results/ml/loader_audit.jsonl, so "the test split was never read" is a statement about an
artifact and not about intent (docs/reference/standing-rules.md standing rule 7).

Rows are read by FANCY-INDEXING the requested split's rows only. The file interleaves the
splits (train rows span the whole file, val and test sit in the middle), so a slice would
not do, and reading everything and discarding would put test rows in memory.

Everything here is numpy + h5py; torch enters in ml/features.py and ml/model.py.
"""
import datetime as _dt
import json
import os
import sys
import time

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lpdm.corpus import TEST_YEAR, split_of  # noqa: E402

H5_DEFAULT = os.path.join(REPO, "corpus", "corpus_cone.h5")
STATIC_DIR = os.path.join(REPO, "data", "grid30_raised")
RESULTS_ML = os.path.join(REPO, "results", "ml")
AUDIT_PATH = os.path.join(RESULTS_ML, "loader_audit.jsonl")
CACHE_DIR = os.path.join(RESULTS_ML, "cache")

# The frame, from corpus/README.md and bin/consolidate_corpus.py:73. Validated against the
# file's own grid/ attributes at load (standing rule 1: the artifact, not the constant).
N, PAD, NG, DX = 128, 3, 122, 30.0
IJ_RECEPTOR = 64
Z_RECEPTOR = 28.5            # grid/receptor_z_m: the aerodynamic height the record carries
SCALAR_NAMES = ("h", "ustar", "sigma_v", "L", "sin_wdir", "cos_wdir")
SPLITS = ("train", "val", "test")

# Octant convention of bin/fig_corpus_pairs.py:59 / bin/phaseA_geometry.py:34.
OCTANTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# Solar array rectangle, metres from the tower (bin/prep_surface.py:42-43). Used only to
# cross-check the padded array.npy, never to build the mask.
ARRAY_XY = (-60.0, 60.0, -100.0, 250.0)


class TestSplitForbidden(RuntimeError):
    """Raised when the test split is requested without the explicit flag."""


class CorpusMismatch(RuntimeError):
    """The file's own attributes disagree with what this loader assumes."""


def _s(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def _audit(event):
    os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
    event = dict(utc=_dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), **event)
    with open(AUDIT_PATH, "a") as fh:
        fh.write(json.dumps(event) + "\n")


def _check_grid(f, path):
    g = f["grid"].attrs
    want = dict(n=NG, pad=PAD, n_padded=N, dx_m=DX, receptor_z_m=Z_RECEPTOR)
    for k, v in want.items():
        got = g[k]
        if abs(float(got) - float(v)) > 1e-9:
            raise CorpusMismatch(f"{path}: grid/{k} is {got}, this loader assumes {v}")
    m = f["meta"].attrs
    if int(m["pad_cells"]) != PAD or int(m["grid_n_padded"]) != N:
        raise CorpusMismatch(f"{path}: meta pad attrs disagree with {PAD}/{N}")


class Split:
    """One split of the corpus, fully in memory. Arrays are float32, rasters (n,128,128)."""

    def __init__(self, name, rows, scalars, kljun, target, meta, valid_mask, path):
        self.name = name
        self.rows = rows
        self.scalars = scalars
        self.kljun = kljun
        self.target = target
        self.meta = meta
        self.valid_mask = valid_mask
        self.path = path
        self.n = int(len(rows))

    def __len__(self):
        return self.n

    @property
    def wdir_deg(self):
        return self.meta["wdir_deg"]

    @property
    def octant(self):
        return octant_of(self.meta["wdir_deg"])

    @property
    def zL(self):
        """z_m / L at the aerodynamic receptor height, finite everywhere (meta/inv_L)."""
        return Z_RECEPTOR * self.meta["inv_L"]

    @property
    def asymptote(self):
        """1 - z_m/z_i (Steinfeld 2008). h (scalar 0) == meta/zi_achieved_m exactly."""
        return 1.0 - Z_RECEPTOR / self.scalars[:, 0]

    @property
    def seed_key(self):
        return np.array([f"{j}:{int(r)}" for j, r in zip(self.meta["seed_job"],
                                                        self.meta["seed_rot"])])


def load_split(split, h5_path=H5_DEFAULT, *, allow_test=False, audit=True):
    """Read one split. Refuses `test` unless allow_test=True is passed explicitly."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; one of {SPLITS}")
    if split == "test" and not allow_test:
        if audit:
            _audit(dict(file=os.path.relpath(h5_path, REPO), split=split, n=0,
                        allow_test=False, refused=True))
        raise TestSplitForbidden(
            "the test split is off limits during development. It is read only when "
            "allow_test=True is passed explicitly (ml/evaluate.py --allow-test), which "
            "nothing under ml/ does.")
    t0 = time.time()
    with h5py.File(h5_path, "r") as f:
        _check_grid(f, h5_path)
        m = f["meta"]
        order = [str(x.decode() if isinstance(x, bytes) else x) for x in m.attrs["split_order"]]
        si = m["split_index"][:]
        want = order.index(split)
        rows = np.where(si == want)[0]
        if len(rows) == 0:
            raise CorpusMismatch(f"{h5_path}: no rows with split_index {want} ({split})")
        # Fancy indexing with a sorted, increasing index: h5py reads only these rows.
        scalars = f["scalars"][rows].astype(np.float32)
        kljun = f["kljun"][rows].astype(np.float32)
        target = f["target"][rows].astype(np.float32)
        meta = {}
        for k in ("datetime", "parent_case", "run_id", "split", "gate_state", "seed_job"):
            meta[k] = _s(m[k][rows])
        for k in ("wdir_deg", "inv_L", "zi_achieved_m", "u_mean_ms", "array_share",
                  "integral", "peak_x_m", "centroid_dist_m"):
            meta[k] = m[k][rows].astype(np.float32)
        meta["seed_rot"] = m["seed_rot"][rows].astype(np.int8)
        meta["split_index"] = si[rows]
        valid_mask = m["valid_mask"][:].astype(bool)
        names = tuple(_s(m["scalar_names"][:]))
    if names != SCALAR_NAMES:
        raise CorpusMismatch(f"{h5_path}: scalar_names {names} != {SCALAR_NAMES}")

    # === THE SPLIT, CHECKED AGAINST THE PRODUCTION FUNCTION ON EVERY ROW ================
    if not np.all(meta["split"] == split):
        raise CorpusMismatch(f"{h5_path}: meta/split disagrees with split_index for {split}")
    for stamp in meta["datetime"]:
        when = _dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")).replace(tzinfo=None)
        derived = split_of(when.date())
        if derived != split:
            raise CorpusMismatch(f"{h5_path}: record {stamp} is {derived!r} under "
                                 f"lpdm.corpus.split_of but was requested as {split!r}")
        if split != "test" and when.year == TEST_YEAR:
            raise CorpusMismatch(f"{h5_path}: a {TEST_YEAR} record reached the {split} split")
    if not (np.isfinite(scalars).all() and np.isfinite(kljun).all()
            and np.isfinite(target).all()):
        raise CorpusMismatch(f"{h5_path}: non-finite values in {split}")   # rule 9
    border = np.concatenate([target[:, :PAD].ravel(), target[:, -PAD:].ravel(),
                             target[:, :, :PAD].ravel(), target[:, :, -PAD:].ravel()])
    if np.any(border != 0.0):
        raise CorpusMismatch(f"{h5_path}: non-zero cells in the {PAD}-cell pad of {split}")
    if int(valid_mask.sum()) != NG * NG or not valid_mask[PAD:PAD + NG, PAD:PAD + NG].all():
        raise CorpusMismatch(f"{h5_path}: meta/valid_mask is not the {NG}^2 interior")

    if audit:
        _audit(dict(file=os.path.relpath(h5_path, REPO), split=split, n=int(len(rows)),
                    rows_min=int(rows.min()), rows_max=int(rows.max()),
                    allow_test=bool(allow_test), seconds=round(time.time() - t0, 2),
                    caller=os.path.basename(sys.argv[0]) if sys.argv else ""))
    return Split(split, rows, scalars, kljun, target, meta, valid_mask, h5_path)


def read_norm(h5_path=H5_DEFAULT):
    """The file's own train-only normalisation. Read, never recomputed."""
    with h5py.File(h5_path, "r") as f:
        g = f["norm"]
        a = g.attrs
        out = dict(scalars_mean=g["scalars_mean"][:].astype(np.float32),
                   scalars_std=g["scalars_std"][:].astype(np.float32),
                   kljun_scale=float(a["kljun_scale"]),
                   target_scale=float(a["target_scale"]),
                   computed_on=str(a["computed_on"]), n_train=int(a["n_train"]),
                   raster_transform=str(a["raster_transform"]))
    if out["computed_on"] != "train split only":
        raise CorpusMismatch(f"{h5_path}: norm/computed_on is {out['computed_on']!r}")
    if not out["raster_transform"].startswith("y = arcsinh(x / s)"):
        raise CorpusMismatch(f"{h5_path}: unexpected norm/raster_transform "
                             f"{out['raster_transform']!r}")
    if out["scalars_mean"].shape != (6,) or out["scalars_std"].shape != (6,):
        raise CorpusMismatch("norm/scalars_* are not (6,)")
    if not (np.isfinite(out["scalars_std"]).all() and (out["scalars_std"] > 0).all()):
        raise CorpusMismatch("norm/scalars_std is not finite and positive")
    return out


def read_grid_attrs(h5_path=H5_DEFAULT):
    with h5py.File(h5_path, "r") as f:
        g = dict(f["grid"].attrs)
        variant = f.attrs.get("variant", "raw")
    g = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in g.items()}
    g["variant"] = variant.decode() if isinstance(variant, bytes) else str(variant)
    return g


# ------------------------------------------------------------------ geometry helpers

def axes_m():
    """Cell centres of the padded raster, metres from the receptor (east or north)."""
    return (np.arange(N) - IJ_RECEPTOR) * DX


def meshgrid_m():
    xc = axes_m()
    return np.meshgrid(xc, xc)          # X east, Y north, both (128,128), [j, i]


def octant_of(wdir_deg):
    """Eight sectors of the FROM-direction, 45 deg wide, centred on N=0, NE=45, ..."""
    w = np.asarray(wdir_deg, dtype=float)
    idx = np.round(w / 45.0).astype(int) % 8
    return np.array(OCTANTS, dtype=object)[idx]


# ------------------------------------------------------------------ static surface

STATIC_FIELDS = ("array", "water", "lcclass", "topo", "z0m", "htFlux", "dmap")
LC_CLASSES = (10, 30, 40, 50, 60, 80, 90)     # ESA WorldCover codes present on this grid


def load_statics(static_dir=STATIC_DIR):
    """The production surface, padded into the corpus frame exactly as the rasters were
    (bin/fig_corpus_pairs.py:145-157 and bin/make_pair.py:140-149: np.pad(a, 3), zero fill).

    Verifies the grid the fields sit on against the corpus frame: meta.npy says the tower is
    cell (61, 61) of a 122^2 30 m grid, which pads to (64, 64) and must lie inside the array.
    """
    meta = np.load(os.path.join(static_dir, "meta.npy"), allow_pickle=True).item()
    want = dict(nx=NG, ny=NG, dx=DX, itower=NG // 2, jtower=NG // 2)
    for k, v in want.items():
        if abs(float(meta[k]) - float(v)) > 1e-9:
            raise CorpusMismatch(f"{static_dir}/meta.npy: {k} is {meta[k]}, corpus frame "
                                 f"needs {v}")
    out = {}
    for name in STATIC_FIELDS:
        a = np.load(os.path.join(static_dir, name + ".npy"))
        if a.shape != (NG, NG):
            raise CorpusMismatch(f"{static_dir}/{name}.npy is {a.shape}, not ({NG}, {NG})")
        out[name] = np.pad(a.astype(np.float64), PAD, mode="constant", constant_values=0.0)
    if int(meta["itower"]) + PAD != IJ_RECEPTOR or int(meta["jtower"]) + PAD != IJ_RECEPTOR:
        raise CorpusMismatch("tower cell does not pad to the receptor cell")
    arr = out["array"] > 0.5
    if not arr[IJ_RECEPTOR, IJ_RECEPTOR]:
        raise CorpusMismatch("the receptor cell (64, 64) is not inside the padded array")
    # The rectangle bin/prep_surface.py:327-334 built -- cell CENTRES at x0 + (i+0.5)*dx in
    # map metres against the SURVEYED tower point, which is not on a cell centre -- re-derived
    # from meta.npy with the same formula, must be the same 44 cells. This is the check that
    # the static mask is on the corpus's cells and not merely near them.
    xc = float(meta["x0"]) + (np.arange(NG) + 0.5) * DX - float(meta["tower_x"])
    yc = float(meta["y0"]) + (np.arange(NG) + 0.5) * DX - float(meta["tower_y"])
    XX, YY = np.meshgrid(xc, yc)
    x0, x1, y0, y1 = ARRAY_XY
    rect = np.pad((XX >= x0) & (XX <= x1) & (YY >= y0) & (YY <= y1), PAD)
    if not np.array_equal(rect, arr):
        raise CorpusMismatch(f"array.npy ({int(arr.sum())} cells) != the rectangle rebuilt "
                             f"from meta.npy ({int(rect.sum())} cells) on the padded grid")
    out["meta"] = meta
    return out


# ------------------------------------------------------------------ the cone, rebuilt

def cone_masks(split, grid_attrs=None, cache=True, workers=None, verbose=True):
    """Rebuild bin/mask_cone.py's keep-mask for every record of a split, from the file's
    own scalars, meta/u_mean_ms and grid/cone_mask_* parameters. Cached per split under
    results/ml/cache because sigma_y_field evaluates the official FFP at nx=2000 per record.
    """
    import mask_cone as mc                      # bin/mask_cone.py, importable by design
    if grid_attrs is None:
        grid_attrs = read_grid_attrs(split.path)
    k = float(grid_attrs["cone_mask_k"])
    y_min = float(grid_attrs["cone_mask_y_min_m"])
    x_min = float(grid_attrs["cone_mask_x_min_m"])
    if mc.NPAD != N or mc.IJ_RECEPTOR != IJ_RECEPTOR or abs(mc.DX - DX) > 1e-9 \
            or abs(mc.Z_RECEPTOR - Z_RECEPTOR) > 1e-9:
        raise CorpusMismatch("bin/mask_cone.py constants disagree with the loader frame")
    key = f"cone_{split.name}_k{k:g}_y{y_min:g}_x{x_min:g}_n{split.n}.npz"
    path = os.path.join(CACHE_DIR, key)
    ids = split.meta["run_id"]
    if cache and os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:
            if np.array_equal(z["run_id"], ids):
                return z["keep"].astype(bool)
    if verbose:
        print(f"  building {split.n} cone masks for {split.name} (official FFP per record)")
    args = [(split.scalars[i], float(split.meta["u_mean_ms"][i]), k, y_min, x_min)
            for i in range(split.n)]
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) // 2))
    if workers > 1:
        from multiprocessing import get_context
        with get_context("fork").Pool(workers) as pool:
            keeps = pool.map(_cone_one, args, chunksize=8)
    else:
        keeps = [_cone_one(a) for a in args]
    keep = np.stack(keeps).astype(bool)
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, keep=keep, run_id=ids)
    return keep


def _cone_one(a):
    import mask_cone as mc
    sc, umean, k, y_min, x_min = a
    X, Y = mc.axis_grids()
    xw, yw = mc.wind_frame(X, Y, float(sc[4]), float(sc[5]))
    sy = mc.sigma_y_field(np.asarray(sc, dtype=float), umean, xw)
    return mc.cone_keep(xw, yw, sy, k, y_min, x_min)


# ------------------------------------------------------------------ raster array share

def raster_array_share(f, array_mask):
    """Signed share of the raster's total on the array cells -- lpdm/driver.py:555's
    cover_share definition applied to the raster instead of to touchdowns."""
    f = np.asarray(f)
    tot = f.sum(axis=(-2, -1))
    on = (f * array_mask).sum(axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot != 0, on / tot, np.nan)


def audit_lines():
    if not os.path.exists(AUDIT_PATH):
        return []
    with open(AUDIT_PATH) as fh:
        return [json.loads(l) for l in fh if l.strip()]
