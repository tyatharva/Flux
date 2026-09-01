"""From a loaded Split to model tensors: the asinh transforms, the scalar encoding, the
distance and static channels, and per-record loss weights. Numpy only; torch enters in
ml/model.py.

The transform is the file's own: y = asinh(x / s), signed, no clipping (norm/ attrs). Two
scales exist -- `kljun_scale` for the INPUT channel and `target_scale` for the TARGET -- and
the residual head needs the base to live in the target's space, so:

    x_in     = asinh(kljun / s_in)          the input channel
    base_T   = asinh(kljun / s_out)         what the residual is added to
    target_T = asinh(target / s_out)        what the sum is scored against
    f_pred   = s_out * sinh(base_T + r)     the prediction in m^-2

A zero residual therefore reproduces Kljun EXACTLY (bin/test_ml_model.py asserts it).
`norm_mode="record"` replaces the two global scales by each record's own Kljun peak, which
is available at inference because Kljun is an input.
"""
from dataclasses import dataclass, asdict

import numpy as np

from ml import data as D


@dataclass
class FeatureSpec:
    norm_mode: str = "global"      # global | record
    knee: float = 1.0              # multiplies the scale: <1 more log-like, >1 more linear
    stab: str = "zL"               # zL: 28.5*inv_L unscaled | L: raw L z-scored (file stats)
    dist: str = "lin_exp"          # none | lin_exp | lin_exp_xy
    exp_scale_m: float = 300.0
    statics: str = "B"             # none | B | C | B_rot90 (control: same maps, wrong site)
    weight: str = "none"           # none | north3 (x3 on N/NE/NW records, mean 1)

    def to_dict(self):
        return asdict(self)


def fwd(x, s):
    return np.arcsinh(np.asarray(x, dtype=np.float64) / np.asarray(s, dtype=np.float64))


def inv(y, s):
    return np.asarray(s, dtype=np.float64) * np.sinh(np.asarray(y, dtype=np.float64))


def raster_scales(split, norm, spec):
    """(s_in, s_out), each (n,) float64."""
    n = split.n
    if spec.norm_mode == "global":
        s_in = np.full(n, spec.knee * norm["kljun_scale"])
        s_out = np.full(n, spec.knee * norm["target_scale"])
    elif spec.norm_mode == "record":
        peak = np.abs(split.kljun * split.valid_mask[None]).reshape(n, -1).max(axis=1)
        if not (peak > 0).all():
            raise ValueError("a record's Kljun raster has no positive peak")
        s_in = spec.knee * peak.astype(np.float64)
        s_out = s_in.copy()
    else:
        raise ValueError(f"norm_mode {spec.norm_mode!r}")
    return s_in, s_out


def encode_scalars(split, norm, spec):
    """(n, 6) float32: z-scored with the file's train-only statistics, except the
    stability column, which is either raw L z-scored (file stats) or z_m/L unscaled."""
    sc = split.scalars.astype(np.float64)
    out = (sc - norm["scalars_mean"]) / norm["scalars_std"]
    if spec.stab == "zL":
        out[:, 3] = split.zL
    elif spec.stab != "L":
        raise ValueError(f"stab {spec.stab!r}")
    return out.astype(np.float32)


def dist_channels(spec):
    X, Y = D.meshgrid_m()
    d = np.hypot(X, Y)
    half = (D.NG / 2) * D.DX                       # 1830 m, the last real cell edge
    if spec.dist == "none":
        return np.zeros((0, D.N, D.N), np.float32), []
    ch = [d / half, np.exp(-d / spec.exp_scale_m)]
    names = ["dist_lin", "dist_exp"]
    if spec.dist == "lin_exp_xy":
        ch += [X / half, Y / half]
        names += ["x_east", "y_north"]
    elif spec.dist != "lin_exp":
        raise ValueError(f"dist {spec.dist!r}")
    return np.stack(ch).astype(np.float32), names


def static_channels(statics, spec, valid):
    if spec.statics == "none":
        return np.zeros((0, D.N, D.N), np.float32), []
    if spec.statics == "B_rot90":
        # THE CONTROL for the statics ablation. The statics are constant across records, so
        # any val gain from them can only be a bias term; rotating every map by 90 deg keeps
        # the statistics and the channel count and destroys the geography. If the rotated
        # maps buy the same gain, the gain was capacity, not the site.
        rot = {k: np.rot90(v_) for k, v_ in statics.items() if k != "meta"}
        ch, names = static_channels(rot, FeatureSpec(statics="B"), valid)
        return ch, [n + "_rot90" for n in names]
    v = valid.astype(np.float64)
    z0 = np.where(valid, np.log10(np.maximum(statics["z0m"], 1e-6)), 0.0)
    ch = [statics["topo"] / 20.0, np.where(valid, (z0 + 2.0) / 2.0, 0.0),
          (statics["array"] > 0.5) * v, (statics["water"] > 0.5) * v]
    names = ["topo", "log_z0", "array", "water"]
    if spec.statics == "C":
        lc = statics["lcclass"]
        for c in D.LC_CLASSES:
            ch.append((lc == c) * v)
            names.append(f"lc{c}")
        ch.append(statics["htFlux"] / 0.1)
        names.append("htFlux")
    elif spec.statics != "B":
        raise ValueError(f"statics {spec.statics!r}")
    return np.stack(ch).astype(np.float32), names


def record_weights(split, spec):
    w = np.ones(split.n, np.float32)
    if spec.weight == "north3":
        north = np.isin(split.octant.astype(str), ("N", "NE", "NW"))
        w[north] = 3.0
        w /= w.mean()
    elif spec.weight != "none":
        raise ValueError(f"weight {spec.weight!r}")
    return w


class Features:
    """Everything the trainer needs for one split, as float32 numpy arrays."""

    def __init__(self, split, statics, norm, spec):
        self.split = split
        self.spec = spec
        self.s_in, self.s_out = raster_scales(split, norm, spec)
        self.x_in = fwd(split.kljun, self.s_in[:, None, None]).astype(np.float32)[:, None]
        self.base_T = fwd(split.kljun, self.s_out[:, None, None]).astype(np.float32)
        self.target_T = fwd(split.target, self.s_out[:, None, None]).astype(np.float32)
        self.scal = encode_scalars(split, norm, spec)
        dch, dn = dist_channels(spec)
        sch, sn = static_channels(statics, spec, split.valid_mask)
        self.const = np.concatenate([dch, sch], axis=0) if (len(dn) + len(sn)) else \
            np.zeros((0, D.N, D.N), np.float32)
        self.channel_names = ["kljun_T"] + dn + sn
        self.valid = split.valid_mask.astype(bool)
        self.weights = record_weights(split, spec)
        self.n_channels = 1 + self.const.shape[0]

    def to_physical(self, pred_T, idx=None):
        """(m,128,128) transformed prediction for records idx (all if None) -> m^-2,
        zero on the pad."""
        s = self.s_out if idx is None else self.s_out[np.asarray(idx)]
        f = inv(pred_T, s[:, None, None])
        return (f * self.valid[None]).astype(np.float32)
