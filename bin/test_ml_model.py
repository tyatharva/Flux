#!/usr/bin/env python3
"""Gate for ml/model.py, ml/losses.py and ml/metrics.py.

ASSERTED:
  1. At initialisation the residual is exactly zero, so inverse(base + model(x)) reproduces
     the Kljun raster to 1e-5 relative -- "predicting zero reproduces Kljun exactly".
  2. FiLM produces one (gamma, beta) pair per channel per block and starts at (1, 0).
  3. Perturbing the pad cells of the target leaves the masked loss unchanged: the pad
     never enters the loss.
  4. With a non-zero projection every parameter receives a gradient (the FFT backward and
     the FiLM path both work).
  5. Twenty optimiser steps on 32 val records reduce the loss (the whole graph trains).
  6. The metric wrappers: LES scored against itself gives overlap 1 and zero errors; the
     production peak estimator applied to the raster agrees with meta/peak_x_m (a
     touchdown-binned, raw-field number) to a median of <= 30 m (one cell).

MEASURED AND PRINTED: parameter count of the baseline, forward+backward time per batch of
16, and Kljun's own val errors against the LES -- the number the emulator has to beat.

usage: /home/atyagi/miniforge3/envs/LESNet/bin/python bin/test_ml_model.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import data as D                    # noqa: E402
from ml import features as F                # noqa: E402
from ml import losses as L                  # noqa: E402
from ml import metrics as M                 # noqa: E402
from ml.model import FNO2d                  # noqa: E402

fails = []


def check(ok, msg):
    print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        fails.append(msg)


class Cfg:
    width, modes, depth, local, film_hidden, dropout = 32, 16, 4, "conv1x1", 64, 0.0


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    va = D.load_split("val")
    st = D.load_statics()
    norm = D.read_norm()
    fx = F.Features(va, st, norm, F.FeatureSpec())
    torch.manual_seed(0)
    model = FNO2d(fx.n_channels, Cfg.width, Cfg.modes, Cfg.depth, Cfg.local,
                  Cfg.film_hidden, Cfg.dropout).to(dev)
    print(f"  baseline FNO: {model.n_params():,} parameters, {fx.n_channels} input channels "
          f"{fx.channel_names}")

    const = torch.from_numpy(fx.const).to(dev)
    valid = torch.from_numpy(fx.valid).to(dev)
    idx = np.arange(8)
    x_in = torch.from_numpy(fx.x_in[idx]).to(dev)
    scal = torch.from_numpy(fx.scal[idx]).to(dev)
    base = torch.from_numpy(fx.base_T[idx]).to(dev)
    tgt = torch.from_numpy(fx.target_T[idx]).to(dev)

    # 1. zero residual == Kljun
    with torch.no_grad():
        r = model(x_in, const, scal)
    check(r.shape == (8, 128, 128) and float(r.abs().max()) == 0.0,
          f"residual at init is exactly zero (max |r| = {float(r.abs().max()):g})")
    back = fx.to_physical((base + r).cpu().numpy(), idx)
    rel = np.abs(back - va.kljun[idx]).max() / np.abs(va.kljun[idx]).max()
    check(rel < 1e-5, f"inverse(base + 0) reproduces kljun: max rel err {rel:.2e}")

    # 2. FiLM
    gb = model.film(scal).view(8, Cfg.depth, 2, Cfg.width)
    check(gb.shape == (8, 4, 2, 32) and float(gb.abs().max()) == 0.0,
          "FiLM gives (B, depth, 2, width) and starts at gamma=1, beta=0")

    # 3. the pad never enters the loss
    loss0 = L.masked_mse(base, tgt, valid)
    tgt_p = tgt.clone()
    tgt_p[:, :3, :] = 7.0
    tgt_p[:, :, -3:] = -7.0
    loss1 = L.masked_mse(base, tgt_p, valid)
    check(float(abs(loss0 - loss1)) == 0.0,
          f"masked loss ignores the pad: {float(loss0):.6f} vs {float(loss1):.6f}")

    # 4. gradients everywhere (with a non-zero projection)
    with torch.no_grad():
        model.proj2.weight.normal_(0, 1e-2)
        model.film[-1].weight.normal_(0, 1e-2)
    X, Y = D.meshgrid_m()
    Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(dev)
    s_out = torch.from_numpy(fx.s_out[idx].astype(np.float32)).to(dev)
    asym = torch.from_numpy(va.asymptote[idx].astype(np.float32)).to(dev)
    model.zero_grad()
    pred = base + model(x_in, const, scal)
    loss = (L.masked_mse(pred, tgt, valid) + 0.1 * L.peak_term(pred, tgt, valid, Xt, Yt)
            + 0.1 * L.integral_term(pred, s_out, valid, asym))
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    check(not missing, f"every parameter has a non-zero gradient (missing: {missing})")

    # 5. twenty steps reduce the loss
    idx2 = np.arange(32)
    xb = torch.from_numpy(fx.x_in[idx2]).to(dev)
    sb = torch.from_numpy(fx.scal[idx2]).to(dev)
    bb = torch.from_numpy(fx.base_T[idx2]).to(dev)
    tb = torch.from_numpy(fx.target_T[idx2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = None
    torch.cuda.synchronize() if dev.type == "cuda" else None
    tt = time.time()
    for it in range(20):
        opt.zero_grad()
        l = L.masked_mse(bb + model(xb, const, sb), tb, valid)
        l.backward()
        opt.step()
        first = float(l) if first is None else first
    torch.cuda.synchronize() if dev.type == "cuda" else None
    per_batch16 = (time.time() - tt) / 20 / 2
    check(float(l) < first, f"loss falls over 20 steps: {first:.5f} -> {float(l):.5f} "
                            f"({per_batch16*1e3:.1f} ms per fwd+bwd of 16 at width 32)")
    if dev.type == "cuda":
        print(f"  peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # 6. metrics
    arr = st["array"] > 0.5
    e = M.pair_errors(va.target[0], va.target[0], va.wdir_deg[0], arr, va.asymptote[0])
    check(e["overlap80"] == 1.0 and e["peak_x"] == 0 and e["centroid"] == 0
          and e["array_share"] == 0 and e["integral"] == 0, "LES vs itself: overlap 1, errors 0")
    tm = time.time()
    sc = M.score_fields({"kljun": va.kljun}, va.target, va.wdir_deg, arr, va.asymptote)
    dt_m = time.time() - tm
    dpk = np.abs(sc["les"]["peak_x"] - va.meta["peak_x_m"])
    check(np.median(dpk) <= 30.0,
          f"raster peak_x vs meta/peak_x_m: median |diff| {np.median(dpk):.0f} m, "
          f"within one cell {100*np.mean(dpk <= 30):.0f}% (cone vs raw field, both 5-cell)")
    k = M.summarise(sc["kljun"])
    print(f"  Kljun vs LES on val ({va.n} records, {dt_m:.1f} s to score two fields):")
    for key in M.METRIC_KEYS:
        print(f"    {key:12s} median {k[key]['median']:9.3f}  mean {k[key]['mean']:9.3f}")
    north = np.isin(va.octant.astype(str), ("N", "NE", "NW"))
    kn = M.summarise(sc["kljun"], north)
    print(f"  Kljun vs LES, N/NE/NW only ({int(north.sum())} records): "
          + "  ".join(f"{key} {kn[key]['median']:.3f}" for key in M.METRIC_KEYS))

    print(f"test_ml_model: {'FAIL' if fails else 'PASS'} ({len(fails)} failures, "
          f"{time.time()-t0:.0f} s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
