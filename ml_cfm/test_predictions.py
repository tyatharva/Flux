"""Write the test-split artifacts the frozen recipe needs: the FNO prediction of each seed and
the CFM samples of each seed (samples_per_seed of the recipe, Euler 16, the sampler as trained).
THE USER RUNS THIS, ONCE. Like every script under ml_cfm/, it passes the --allow-test flag through and never sets it itself.

    python -m ml_cfm.test_predictions --allow-test

Outputs under results/ml_cfm/test/: fno_seedN_pred_test.npz, seedN_samples_test.npz. Refuses to
overwrite an existing file, so the test numbers cannot drift between runs.
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "bin")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml import data as D                      # noqa: E402
from ml_cfm import final_recipe as FR         # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="sampler RNG seed")
    a = ap.parse_args(argv)
    if a.split == "test" and not a.allow_test:
        raise SystemExit("refusing the test split without --allow-test")
    import torch
    from ml import evaluate as E
    from ml_cfm import infer as I
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = D.load_split(a.split, allow_test=a.allow_test)
    st = D.load_statics()
    norm = D.read_norm()
    valid = split.valid_mask.astype(np.float32)
    os.makedirs(FR.TEST_DIR, exist_ok=True)
    R = FR.RECIPE
    for sd in R["fno_seeds"]:
        out = FR.fno_path(sd, a.split)
        if os.path.exists(out):
            print("exists, not overwriting:", out); continue
        f, n = E.predict_split([os.path.join(REPO, "results", "ml", "final", sd, "best.pt")], split, st, norm, dev)
        assert n == 1 and np.isfinite(f).all()
        np.savez_compressed(out, fno=f.astype(np.float32), run_id=split.meta["run_id"])
        print("wrote", out)
    for sd in R["cfm_seeds"]:
        out = FR.sample_path(sd, a.split)
        if os.path.exists(out):
            print("exists, not overwriting:", out); continue
        cfg, model, _ = I.load_checkpoint(os.path.join(REPO, "results", "ml_cfm", "final", sd, "best.pt"), dev)
        prep = I.Prepared(cfg, split, st, norm, dev)
        t0 = time.time()
        sT, secs = prep.samples(model, cfg, R["samples_per_seed"], a.steps, "euler", a.seed)
        T = sT.cpu().numpy().astype(np.float32)
        assert np.isfinite(T).all() and T.shape[0] == R["samples_per_seed"]
        np.savez_compressed(out, samples_T=T.astype(np.float16), run_id=split.meta["run_id"],
                            s_out=prep.fx.s_out.astype(np.float32), seconds=secs, steps=a.steps, seed=a.seed)
        print("wrote", out, f"{time.time() - t0:.0f} s")
        del model, prep, sT
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
