"""Load a checkpoint and prepare a split for sampling. Shared by the solver study, the final
driver and the evaluator. The split is whatever the caller passes; nothing here can read
the test split without ml.data's explicit flag, which nothing in ml_cfm passes.
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml import data as D                      # noqa: E402
from ml import features as F                  # noqa: E402
from ml_cfm import flow as FL                 # noqa: E402
from ml_cfm.model import build_model          # noqa: E402
from ml_cfm.train import CfmConfig            # noqa: E402


def load_checkpoint(path, dev=None):
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CfmConfig(**ck["config"])
    model = build_model(cfg, ck["c_in"])
    model.load_state_dict(ck["state_dict"])
    return cfg, model.to(dev).eval(), ck


class Prepared:
    """Tensors of one split on the device for a given config: x_in, scal, base (cropped to
    the cone when gated), mask, s_out; and the Features object for to_physical."""

    def __init__(self, cfg, split, statics, norm, dev):
        self.split = split
        self.fx = F.Features(split, statics, norm, cfg.feature_spec())
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        self.valid = self.fx.valid
        if cfg.gate == "cone":
            keep = D.cone_masks(split, verbose=False) & self.valid
            self.mask = t(keep.astype(np.float32))
        else:
            self.mask = t(self.valid.astype(np.float32))
        base = t(self.fx.base_T)
        self.T = dict(x_in=t(self.fx.x_in), scal=t(self.fx.scal),
                      base=base * self.mask if cfg.gate == "cone" else base,
                      s_out=t(self.fx.s_out.astype(np.float32)))
        self.const = t(self.fx.const)
        self.dev = dev

    def samples(self, model, cfg, S, steps, solver, seed, idx=None):
        """(S, n, 128, 128) asinh-space samples and the wall seconds."""
        idx = torch.arange(self.split.n, device=self.dev) if idx is None else idx
        gs = torch.Generator(device=self.dev).manual_seed(seed)
        return FL.draw_samples(model, cfg.param, self.T, self.const, idx, self.mask,
                               cfg.sigma, S, steps, solver, gs)

    def physical(self, T_field, idx=None):
        """asinh-space (m,128,128) or (S,m,128,128) -> m^-2."""
        a = np.asarray(T_field)
        if a.ndim == 4:
            return np.stack([self.fx.to_physical(x, idx) for x in a])
        return self.fx.to_physical(a, idx)
