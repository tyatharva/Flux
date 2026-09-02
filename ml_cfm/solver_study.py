"""On one checkpoint: the sample mean's val_mse_ref and production metrics against Kljun
for several (solver, steps) settings, S samples each, with the wall time per record.

    python -m ml_cfm.solver_study --ckpt results/ml_cfm/phase1/smoke60/best.pt --S 8
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml import data as D                      # noqa: E402
from ml import losses as L                    # noqa: E402
from ml import metrics as M                   # noqa: E402
from ml_cfm.infer import load_checkpoint, Prepared    # noqa: E402

SETTINGS = [("euler", 4), ("euler", 8), ("euler", 16), ("euler", 32), ("heun", 8), ("heun", 16)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev = torch.device("cuda")
    cfg, model, ck = load_checkpoint(a.ckpt, dev)
    va = D.load_split("val")
    st = D.load_statics()
    norm = D.read_norm()
    prep = Prepared(cfg, va, st, norm, dev)
    valid = torch.from_numpy(prep.valid).to(dev)
    s_ref = float(norm["target_scale"])
    tgt_ref = torch.from_numpy(np.arcsinh(va.target / s_ref).astype(np.float32)).to(dev)
    arr = st["array"] > 0.5
    rows = []
    for solver, steps in SETTINGS:
        samp, secs = prep.samples(model, cfg, a.S, steps, solver, seed=777)
        mean_T = samp.mean(0)
        f_phys = prep.T["s_out"][:, None, None] * torch.sinh(mean_T.clamp(-20, 20)) * valid
        v_ref = float(L.masked_mse(torch.asinh(f_phys / s_ref), tgt_ref, valid))
        f_va = prep.physical(mean_T.cpu().numpy())
        sc = M.score_fields({"cfm": f_va, "kljun": va.kljun}, va.target, va.wdir_deg, arr,
                            va.asymptote)
        comp, ratios = M.composite(sc["cfm"], sc["kljun"])
        med = {k: float(np.nanmedian(sc["cfm"][k])) for k in M.METRIC_KEYS + M.SHAPE_KEYS}
        rows.append(dict(solver=solver, steps=steps, S=a.S, val_mse_ref=v_ref, composite=comp,
                         ratios=ratios, medians=med, seconds=round(secs, 2),
                         ms_per_record_per_sample=round(1000 * secs / (va.n * a.S), 3)))
        print(f"{solver:6s} {steps:3d}: ref {v_ref:.6f} composite {comp:.3f} "
              f"{rows[-1]['ms_per_record_per_sample']:.2f} ms/rec/sample  {med}")
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(a.ckpt)), "solver_study.json")
    with open(out, "w") as fh:
        json.dump(dict(ckpt=a.ckpt, rows=rows), fh, indent=1)
    md = out.replace(".json", ".md")
    with open(md, "w") as fh:
        fh.write(f"# Solver study on `{os.path.relpath(a.ckpt, REPO)}`, S = {a.S}\n\n"
                 "| solver | steps | val_mse_ref | composite | peak_x | centroid | overlap80 | "
                 "array_share | integral | shape_l1_2d | ms / record / sample |\n"
                 "|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            m = r["medians"]
            fh.write(f"| {r['solver']} | {r['steps']} | {r['val_mse_ref']:.6f} | "
                     f"{r['composite']:.3f} | {m['peak_x']:.1f} | {m['centroid']:.1f} | "
                     f"{m['overlap80']:.3f} | {m['array_share']:.3f} | {m['integral']:.4f} | "
                     f"{m['shape_l1_2d']:.3f} | {r['ms_per_record_per_sample']:.2f} |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
