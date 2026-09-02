"""Run a {name: overrides} JSON of ml_cfm.train configurations, K at a time, with the GPU
monitor and the z-score summary of ml/phase1.py (imported, not copied; only the launched
module differs).

    python -m ml_cfm.campaign --runs runs.json -K 4 --outdir results/ml_cfm/phase1 --tag p1
    python -m ml_cfm.campaign --summarise --outdir ... --baseline-prefix v_s0.1_seed
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml import phase1 as P                    # noqa: E402

PY = sys.executable
BASE = dict(epochs=300, save_ckpt=True, save_samples=8)


def run_one(name, overrides, outdir, base):
    out = os.path.join(outdir, name)
    if os.path.exists(os.path.join(out, "run.json")):
        return name, "cached", 0.0
    cfg = dict(base)
    cfg.update(overrides)
    args = [PY, "-u", "-m", "ml_cfm.train", "--out", out] + [f"--set={k}={v}" for k, v in cfg.items()]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(out, "log.txt"), "w") as log:
        rc = subprocess.call(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    ok = rc == 0 and os.path.exists(os.path.join(out, "run.json"))
    return name, ("ok" if ok else f"FAILED rc={rc}"), time.time() - t0


P.run_one = run_one          # the campaign loop is ml.phase1's; only the launcher differs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=None)
    ap.add_argument("-K", type=int, default=4)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--tag", default="custom")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--baseline-prefix", default="v_s0.1_seed")
    a = ap.parse_args(argv)
    if a.summarise:
        P.summarise(a.outdir, a.baseline_prefix)
        return 0
    base = dict(BASE)
    if a.base:
        base.update(json.loads(a.base))
    with open(a.runs) as fh:
        runs = json.load(fh)
    P.campaign(runs, a.outdir, a.K, base, a.tag)
    P.summarise(a.outdir, a.baseline_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
