"""After Phase 2: retrain the best configuration with several seeds, evaluate every seed and
the seed ensemble on val, and write results/ml/final/final.json.

    python -m ml.final --study fno_v2 [--trial N] [--seeds 5] [-K 2]

The best trial is read from the study's database (or --trial overrides it). Each seed is a
full-length run (150 epochs, patience 25) with a checkpoint and val predictions. The
ensemble is the mean of the seeds' physical-space predictions. Selection between "best
single seed" and "ensemble" is made on the val composite -- val is the selection split;
the test split is never read.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
PY = sys.executable
OUT = os.path.join(REPO, "results", "ml", "final")
EVAL = os.path.join(REPO, "results", "ml", "eval")

FIXED = dict(head="residual", statics="none", dist="lin_exp_xy", norm_mode="global",
             knee=1.0, stab="zL", lam_peak=0.0, lam_int=0.0, weight="none")


def best_params(study, storage, trial=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    s = optuna.load_study(study_name=study, storage=storage)
    if trial is not None:
        t = [t for t in s.trials if t.number == trial][0]
    else:
        t = s.best_trial
    return t.number, dict(t.params), t.value, dict(t.user_attrs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--study", default="fno_v2")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("-K", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--override", default=None,
                    help="JSON overrides applied on top of the trial's params (the haze "
                         "round's winner: gate, lam_l1, knee)")
    a = ap.parse_args(argv)
    storage = a.storage or "sqlite:///" + os.path.join(REPO, "results", "ml", "phase2",
                                                        f"optuna_{a.study}.db")
    number, params, value, attrs = best_params(a.study, storage, a.trial)
    print(f"best trial #{number}: value {value:.6g} params {params}")
    os.makedirs(OUT, exist_ok=True)
    cfg = dict(FIXED)
    cfg.update(params)
    if a.override:
        cfg.update(json.loads(a.override))
    runs = {f"seed{s}": dict(cfg, seed=s) for s in range(a.seeds)}
    with open(os.path.join(OUT, "runs.json"), "w") as fh:
        json.dump(runs, fh, indent=1)
    base = json.dumps(dict(epochs=a.epochs, patience=a.patience, eval_every=25,
                           save_ckpt=True, save_pred=True))
    subprocess.check_call([PY, "-u", "-m", "ml.phase1", "--runs", os.path.join(OUT, "runs.json"),
                           "-K", str(a.K), "--outdir", OUT, "--base", base, "--tag", "final"],
                          cwd=REPO)
    # per-seed and ensemble evaluation on val
    ckpts = sorted(glob.glob(os.path.join(OUT, "seed*", "best.pt")))
    per_seed = {}
    for c in ckpts:
        name = os.path.basename(os.path.dirname(c))
        subprocess.check_call([PY, "-m", "ml.evaluate", "--ckpt", c, "--tag", f"final_{name}",
                               "--no-figures"], cwd=REPO, stdout=subprocess.DEVNULL)
        with open(os.path.join(EVAL, f"final_{name}", "eval.json")) as fh:
            e = json.load(fh)
        per_seed[name] = dict(composite=e["composite"]["fno"]["all"],
                              composite_cone=e["composite"]["fno_cone"]["all"],
                              composite_north=e["composite"]["fno"]["north_N_NE_NW"])
    subprocess.check_call([PY, "-m", "ml.evaluate"] + sum([["--ckpt", c] for c in ckpts], [])
                          + ["--tag", "final_ensemble"], cwd=REPO, stdout=subprocess.DEVNULL)
    with open(os.path.join(EVAL, "final_ensemble", "eval.json")) as fh:
        ens = json.load(fh)
    best_seed = min(per_seed, key=lambda k: per_seed[k]["composite"])
    summary = dict(study=a.study, best_trial=number, params=params, objective=value,
                   override=json.loads(a.override) if a.override else {}, config=cfg,
                   trial_attrs=attrs, seeds=per_seed, best_seed=best_seed,
                   ensemble=dict(composite=ens["composite"]["fno"]["all"],
                                 composite_cone=ens["composite"]["fno_cone"]["all"],
                                 composite_north=ens["composite"]["fno"]["north_N_NE_NW"],
                                 n_members=len(ckpts)),
                   selected=("ensemble" if ens["composite"]["fno"]["all"]
                             <= per_seed[best_seed]["composite"] else best_seed),
                   train_val_gap={os.path.basename(os.path.dirname(p)):
                                  json.load(open(p))["gap"] for p in
                                  sorted(glob.glob(os.path.join(OUT, "seed*", "run.json")))})
    with open(os.path.join(OUT, "final.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
