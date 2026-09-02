#!/usr/bin/env python3
"""Gate for the prior-anchored CFM (ml_cfm/): the test-split guard, the interpolant
identities, the zero-init sample, the x->v conversion, the parameter band, the loss mask,
the connected-component filter invariants, and that ml/ and results/ml/final/ are untouched.

    python bin/test_cfm.py            (LESNet env; needs the corpus and a GPU or CPU torch)
"""
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ml import data as D                                   # noqa: E402
from ml_cfm import ccfilter as CC                          # noqa: E402


def main():
    import torch
    from ml_cfm import flow as FL
    from ml_cfm.model import UNetFiLM
    from ml_cfm.train import CfmConfig, fm_loss
    from ml_cfm.infer import Prepared
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    # 1. the test split is refused
    try:
        D.load_split("test")
        check("test split refused", False)
    except D.TestSplitForbidden:
        check("test split refused", True)
    src = subprocess.check_output(["grep", "-rnI", "allow_test", "ml_cfm/"],
                                  cwd=REPO, text=True).strip().splitlines()
    bad = [l for l in src if "allow_test=True" in l or "allow_test = True" in l]
    check("allow_test never set under ml_cfm/", not bad, f"({len(src)} mentions)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    va = D.load_split("val")
    st = D.load_statics()
    norm = D.read_norm()
    cfg = CfmConfig()
    prep = Prepared(cfg, va, st, norm, dev)
    idx = torch.arange(8, device=dev)
    xp = prep.T["base"][idx]
    xl = torch.from_numpy(prep.fx.target_T[:8]).to(dev)
    eps = 0.1 * torch.randn(xp.shape, device=dev) * prep.mask[idx]

    # 2. interpolant end points
    z0, v = FL.interpolate(xp, xl, eps, torch.zeros(8, device=dev))
    z1, _ = FL.interpolate(xp, xl, eps, torch.ones(8, device=dev))
    check("z_0 = x_prior + eps", torch.allclose(z0, xp + eps, atol=1e-6))
    check("z_1 = x_les", torch.allclose(z1, xl, atol=1e-6))
    check("v = d - eps", torch.allclose(v, xl - xp - eps, atol=1e-6))

    # 3. zero-init model: sample == prior + eps; sigma = 0 reproduces Kljun (inside the cone)
    model = UNetFiLM(1 + prep.fx.n_channels).to(dev).eval()
    vf = FL.velocity_fn(model, "velocity", prep.T["x_in"][idx], prep.const,
                        prep.T["scal"][idx], prep.mask[idx])
    s = FL.sample(vf, xp + eps, 8)
    check("zero-init sample = prior + eps", torch.allclose(s, xp + eps, atol=1e-6))
    f = prep.physical(FL.sample(vf, xp, 4).cpu().numpy(), np.arange(8))
    kl = va.kljun[:8] * prep.mask[idx].cpu().numpy()
    rel = np.abs(f - kl).max() / kl.max()
    check("sigma = 0 sample reproduces Kljun (cone-cropped)", rel < 1e-5, f"max rel {rel:.1e}")
    check("param count in [2e6, 4e6]", 2e6 <= model.n_params() <= 4e6, f"{model.n_params()/1e6:.2f} M")

    # 4. x -> v conversion is the algebraic identity when x_hat = x_les
    t = torch.rand(8, device=dev) * 0.9
    zt, vt = FL.interpolate(xp, xl, eps, t)
    check("x->v conversion", torch.allclose(FL.x_to_v(xl, zt, t), vt, atol=1e-4, rtol=1e-4))

    # 5. the loss ignores the target outside the mask
    T = dict(prep.T)
    T["tgt"] = xl.clone()
    T["base"] = xp.clone()
    T["x_in"] = prep.T["x_in"][idx]
    T["scal"] = prep.T["scal"][idx]
    i8 = torch.arange(8, device=dev)
    l0 = float(fm_loss(model, cfg, T, prep.const, i8, prep.mask[idx], t, eps))
    T2 = dict(T)
    T2["tgt"] = xl + 5.0 * (1 - prep.mask[idx])
    l1 = float(fm_loss(model, cfg, T2, prep.const, i8, prep.mask[idx], t, eps))
    check("loss blind to the target outside the mask", abs(l0 - l1) < 1e-9, f"{l0:.3e} vs {l1:.3e}")

    # 6. the filter
    tg = va.target[:16].astype(np.float64)
    out, info = CC.filter_stack(tg, "A")
    check("filter A: kept cells unchanged", all(np.all((o == 0) | (o == g)) for o, g in zip(out, tg)))
    rem = 1 - np.abs(out).sum(axis=(1, 2)) / np.abs(tg).sum(axis=(1, 2))
    check("filter A: mass removed accounted exactly", np.allclose(rem, info["mass_removed_frac"], atol=1e-6),
          f"median removed {100*np.median(rem):.3f}%")
    z, zi = CC.filter_mass(np.zeros((128, 128)))
    check("filter A: empty field passes", z.sum() == 0 and zi["mass_removed_frac"] == 0)
    g = np.exp(-((np.arange(128)[:, None] - 64) ** 2 + (np.arange(128)[None] - 64) ** 2) / 200.0)
    o, i = CC.filter_mass(g)
    check("filter A: one-component field keeps 99.9%+", i["n_kept"] == 1 and i["mass_removed_frac"] < 1.1e-3,
          f"removed {i['mass_removed_frac']:.2e}")
    tau = CC.connectivity_level(g)
    check("filter B: gaussian is single-connected down to the floor", tau <= 1e-6, f"tau {tau:.1e}")

    # 7. ml/ and the FNO's final results are untouched
    diff = subprocess.check_output(["git", "diff", "--stat", "--", "ml/", "results/ml/final/"],
                                   cwd=REPO, text=True).strip()
    check("git diff on ml/ and results/ml/final/ is empty", diff == "", diff[:80])

    print("test_cfm:", "FAIL " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
