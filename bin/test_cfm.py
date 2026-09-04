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

    # 8. CRPS (ml_cfm/crps.py): identities
    from ml_cfm import crps as CR
    from ml_cfm import tailthresh as TT
    gcpu = torch.Generator().manual_seed(0)
    xs = torch.randn(7, 5, 6, generator=gcpu, dtype=torch.float64)
    y = torch.randn(5, 6, generator=gcpu, dtype=torch.float64)
    check("CRPS sorted form == pairwise form", torch.allclose(CR.crps_sorted(xs, y), CR.crps_pairwise(xs, y), atol=1e-12))
    pm = y.unsqueeze(0).expand(3, -1, -1) + 0.3
    check("CRPS of a point mass = |x - y|", torch.allclose(CR.crps_sorted(pm, y), torch.full_like(y, 0.3), atol=1e-12))
    check("CRPS of samples equal to y = 0", float(CR.crps_sorted(y.unsqueeze(0).expand(4, -1, -1), y).abs().max()) < 1e-12)
    # the fair estimator is unbiased in S: the mean over many S=2 draws matches the S=2000 value
    big = torch.randn(2000, 4000, generator=gcpu, dtype=torch.float64)
    yb = torch.zeros(4000, dtype=torch.float64)
    c_big = float(CR.crps_sorted(big, yb).mean())
    c_two = float(CR.crps_sorted(big[:2], yb).mean())
    check("fair CRPS at S=2 agrees with S=2000 (unbiased)", abs(c_two - c_big) < 0.03, f"{c_two:.4f} vs {c_big:.4f}")
    cfg_c = CfmConfig(loss="crps", crps_S=2, crps_steps=2)
    gd = torch.Generator(device=dev).manual_seed(1)
    valid_t = torch.from_numpy(prep.valid.astype(np.float32)).to(dev)
    arr_t = torch.from_numpy((st["array"] > 0.5).astype(np.float32)).to(dev)
    T["s_out"] = prep.T["s_out"][idx]
    lc, terms = CR.crps_field_loss(model, cfg_c, T, prep.const, i8, prep.mask[idx], gd, valid_t, arr_t)
    lc.backward()
    gnorm = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
    check("CRPS loss through the sampler is finite and has a gradient", np.isfinite(float(lc)) and gnorm > 0,
          f"loss {float(lc):.3e} grad {gnorm:.2e}")
    model.zero_grad(set_to_none=True)
    d = CfmConfig()
    check("default config takes the unchanged fm path", d.loss == "fm" and d.select == "ref"
          and d.target_thresh == "none" and d.init_from == "")

    # 9. the source-area threshold
    tg = va.target[:32].astype(np.float64)
    thr, ti = TT.threshold_stack(tg, 0.99)
    check("threshold: kept cells unchanged", all(np.all((o == 0) | (o == g)) for o, g in zip(thr, tg)))
    rem = 1 - np.abs(thr).sum(axis=(1, 2)) / np.abs(tg).sum(axis=(1, 2))
    check("threshold: mass removed accounted exactly", np.allclose(rem, ti["mass_removed_frac"], atol=1e-6),
          f"median removed {100*np.median(rem):.2f}%")
    ok = True
    for g, o, lev in zip(tg, thr, ti["level"]):
        sa = g >= lev                      # the 99% source area by construction
        ok &= np.array_equal(o != 0, sa & (g != 0)) and (o >= 0).all()
        kept = g[sa].sum() / np.maximum(g, 0).sum()
        ok &= kept >= 0.99 - 1e-9
    check("threshold: support == the 99% source area, no negatives, >= 99% of positive mass kept", ok)
    check("threshold: gaussian keeps 99% and removes < 1.1%", TT.threshold_sa(g_ := np.exp(-((np.arange(128)[:, None] - 64) ** 2
          + (np.arange(128)[None] - 64) ** 2) / 200.0))[1]["mass_removed_frac"] < 0.011)

    # 10. the reporting field metrics: identities on synthetic fields
    from ml_cfm import report_metrics as RM
    v = RM._interior()
    xc = (np.arange(128) - 64) * 30.0
    X, Y = np.meshgrid(xc, xc)
    g0 = np.exp(-((X - 300) ** 2 + Y ** 2) / (2 * 150.0 ** 2)) * v; g0 *= 1e-5 / g0.max()
    g1 = np.exp(-((X - 600) ** 2 + Y ** 2) / (2 * 150.0 ** 2)) * v; g1 *= 1e-5 / g1.max()
    same = RM.field_metrics(g0, g0, v, X, Y)
    check("field metrics: identical fields give log-MSE 0, W1 0, MS-SSIM 1, KL = the eps-smoothing bias (< 0.01 nats)",
          same["log_mse"] == 0 and same["sw1_m"] == 0 and 0 <= same["kl_nats"] < 0.01 and abs(same["ms_ssim"] - 1) < 1e-9,
          f"KL bias {same['kl_nats']:.4f}")
    w1 = RM.sliced_w1(g1, g0, v, X, Y)
    check("field metrics: sliced W1 of a 300 m shift is 300 * 2/pi (mean |cos| over directions)", abs(w1 - 300 * 2 / np.pi) < 6, f"{w1:.1f} m")
    check("field metrics: KL(P||Q) >= 0 and asymmetric on distinct fields",
          RM.kl_nats(g0, g1, v) > 0 and RM.kl_nats(g0, g1, v) != RM.kl_nats(g1, g0, v))
    check("field metrics: log-MSE is scale-blind on the log grid, W1 is amplitude-blind",
          abs(RM.log_mse(2 * g0, 2 * g0, v)) == 0 and abs(RM.sliced_w1(3 * g1, g0, v, X, Y) - w1) < 1e-9)
    check("report_metrics refuses the test split", subprocess.run([sys.executable, "-m", "ml_cfm.report_metrics", "--split", "test"],
          cwd=REPO, capture_output=True).returncode != 0)

    # 7. ml/ and the FNO's final results are untouched
    frozen = ["ml/", "results/ml/final/", "results/ml_cfm/final/", "results/ml_cfm/phase1/", "results/ml_cfm/eval/"]
    diff = subprocess.check_output(["git", "diff", "--stat", "--"] + frozen, cwd=REPO, text=True).strip()
    check("git diff on ml/, results/ml/final/ and the first CFM run is empty", diff == "", diff[:80])

    print("test_cfm:", "FAIL " + ", ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
