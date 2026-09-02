"""Phase 2: Optuna (TPE, median pruning) on a SQLite study, K worker processes, resumable.

    python -m ml.phase2_optuna --study fno_v1 -K 4 --n-trials 120 --timeout 14400 \\
        --fixed '{"statics": "none", "head": "residual"}' --space '{"modes": [8, 24, 4]}'
    python -m ml.phase2_optuna --study fno_v1 --summarise

Re-running the driver with the same --study resumes: completed trials are in the database,
running ones are retried once (heartbeat + RetryFailedTrialCallback). The objective is
`val_mse_ref` -- the masked asinh-MSE re-evaluated in the FILE's global transform space --
so trials with different transforms are comparable; `composite` (the metric ratio against
Kljun) is recorded as a user attribute on every trial and can be made the objective with
--objective composite.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
PY = sys.executable
OUT_DEFAULT = os.path.join(REPO, "results", "ml", "phase2")

# name -> (kind, spec). int: [lo, hi, step]; float: [lo, hi, "log"|"lin"]; cat: [choices]
DEFAULT_SPACE = {
    "modes": ("int", [8, 32, 4]),
    "width": ("cat", [16, 24, 32, 48, 64]),
    "depth": ("int", [2, 6, 1]),
    "lr": ("float", [1e-4, 5e-3, "log"]),
    "wd": ("float", [1e-6, 1e-1, "log"]),
    "batch": ("cat", [8, 16, 32]),
    "film_hidden": ("cat", [32, 64, 128]),
    "dropout": ("float", [0.0, 0.3, "lin"]),
    "knee": ("float", [0.3, 3.0, "log"]),
}
BASE = dict(epochs=150, patience=25, eval_every=50, save_ckpt=False)


def suggest(trial, space):
    out = {}
    for k, (kind, spec) in space.items():
        if kind == "int":
            out[k] = trial.suggest_int(k, spec[0], spec[1], step=spec[2] if len(spec) > 2 else 1)
        elif kind == "float":
            out[k] = trial.suggest_float(k, spec[0], spec[1], log=(len(spec) > 2 and spec[2] == "log"))
        elif kind == "cat":
            out[k] = trial.suggest_categorical(k, spec)
        else:
            raise ValueError(kind)
    return out


def make_sampler(seed):
    import optuna
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=12)


def make_pruner():
    import optuna
    # 20 warm-up epochs of 150 before a trial can be pruned; compared every 5 epochs.
    return optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=20, interval_steps=5)


def make_storage(url):
    import optuna
    from optuna.storages import RetryFailedTrialCallback
    return optuna.storages.RDBStorage(
        url=url, engine_kwargs={"connect_args": {"timeout": 120}},
        heartbeat_interval=60, grace_period=240,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=1))


def worker(a):
    import optuna
    import torch
    from ml import train as T
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = make_storage(a.storage)
    # THE PRUNER LIVES IN THE PROCESS THAT CALLS optimize(), NOT IN THE DATABASE. A worker
    # that load_study()s without one gets MedianPruner() with ZERO warm-up and pruned every
    # trial at epoch 0 once five had completed. The sampler seed is per worker so three
    # workers do not propose the same point at the same time.
    study = optuna.load_study(study_name=a.study, storage=storage,
                              sampler=make_sampler(a.seed + 1000 * (a.worker_id + 1)),
                              pruner=make_pruner())
    fixed = json.loads(a.fixed) if a.fixed else {}
    space = dict(DEFAULT_SPACE)
    for k, v in (json.loads(a.space) if a.space else {}).items():
        space[k] = tuple(v) if v is not None else None
    space = {k: v for k, v in space.items() if v is not None and k not in fixed}
    outdir = os.path.join(a.outdir, "trials")
    data = None
    base = dict(BASE)
    base.update(json.loads(a.base) if a.base else {})

    def objective(trial):
        nonlocal data
        params = suggest(trial, space)
        cfg_kw = dict(base)
        cfg_kw.update(fixed)
        cfg_kw.update(params)
        cfg_kw["seed"] = int(cfg_kw.get("seed", 0)) + trial.number
        cfg_kw["out"] = os.path.join(outdir, f"t{trial.number:04d}")
        cfg_kw["name"] = f"t{trial.number:04d}"
        cfg = T.TrainConfig(**cfg_kw)
        if data is None:
            data = T._Data(cfg, torch.device("cuda"))
        t0 = time.time()
        r = T.train_one(cfg, data=data, trial=trial, log=lambda s: None)
        for k in ("composite", "composite_north", "val_loss", "val_mse_ref", "best_epoch",
                  "epochs_run", "n_params", "wall_s", "peak_vram_gb"):
            trial.set_user_attr(k, r[k])
        trial.set_user_attr("gap_ratio", r["gap"]["loss_ratio"])
        for k, v in r["composite_ratios"].items():
            trial.set_user_attr("r_" + k, v)
        for k, v in r["composite_north_ratios"].items():
            trial.set_user_attr("rn_" + k, v)
        if a.objective == "composite":
            return r["composite"]
        return r["val_mse_ref"]

    study.optimize(objective, n_trials=a.n_trials, timeout=a.timeout,
                   catch=(Exception,), gc_after_trial=True)
    return 0


def summarise(a):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=a.study, storage=make_storage(a.storage))
    ts = study.get_trials(deepcopy=False)
    S = optuna.trial.TrialState
    counts = {s.name: sum(1 for t in ts if t.state == s) for s in
              (S.COMPLETE, S.PRUNED, S.FAIL, S.RUNNING, S.WAITING)}
    done = [t for t in ts if t.state == S.COMPLETE]
    rows = []
    for t in ts:
        rows.append(dict(number=t.number, state=t.state.name, value=t.value,
                         **{"p_" + k: v for k, v in t.params.items()},
                         **{k: v for k, v in t.user_attrs.items()},
                         seconds=((t.datetime_complete - t.datetime_start).total_seconds()
                                  if t.datetime_complete and t.datetime_start else None)))
    os.makedirs(a.outdir, exist_ok=True)
    import csv
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("number", "state", "value"), k))
    with open(os.path.join(a.outdir, "trials.tsv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["number"]):
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    out = dict(study=a.study, storage=a.storage, objective=a.objective, counts=counts,
               n_trials=len(ts))
    lines = [f"# Phase 2: Optuna study `{a.study}`", "",
             f"objective `{a.objective}`; trials: " + ", ".join(f"{k} {v}" for k, v in counts.items()),
             ""]
    if done:
        best = study.best_trial
        out["best"] = dict(number=best.number, value=best.value, params=best.params,
                           user_attrs=best.user_attrs)
        lines += [f"## Best trial: #{best.number}, value {best.value:.6g}", "",
                  "params: " + json.dumps(best.params), "",
                  "attrs: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in best.user_attrs.items()}), ""]
        try:
            imp = optuna.importance.get_param_importances(study)
            out["importance"] = imp
            lines += ["## Parameter importance (fANOVA)", "", "| param | importance |", "|---|---|"]
            lines += [f"| {k} | {v:.3f} |" for k, v in imp.items()]
        except Exception as e:                       # noqa: BLE001
            lines += [f"(importance unavailable: {e})"]
        top = sorted(done, key=lambda t: t.value)[:10]
        lines += ["", "## Top 10 by objective", "",
                  "| # | value | composite | north | gap | best_ep | params |", "|---|---|---|---|---|---|---|"]
        for t in top:
            u = t.user_attrs
            lines.append(f"| {t.number} | {t.value:.6g} | {u.get('composite', float('nan')):.3f} | "
                         f"{u.get('composite_north', float('nan')):.3f} | "
                         f"{u.get('gap_ratio', float('nan')):.2f} | {u.get('best_epoch')} | "
                         f"{json.dumps(t.params)} |")
        vals = [t.value for t in done]
        out["values"] = dict(best=float(min(vals)), median=float(np.median(vals)), n=len(vals))
    with open(os.path.join(a.outdir, "study_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    with open(os.path.join(a.outdir, "study_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--study", default="fno_v1")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    ap.add_argument("-K", type=int, default=4)
    ap.add_argument("--n-trials", type=int, default=120, help="total across workers")
    ap.add_argument("--timeout", type=float, default=4 * 3600, help="seconds, per worker")
    ap.add_argument("--fixed", default=None, help="JSON overrides removed from the space")
    ap.add_argument("--space", default=None, help="JSON {name: [kind spec] or null}")
    ap.add_argument("--base", default=None, help="JSON overrides of BASE (epochs, patience)")
    ap.add_argument("--objective", default="val_mse_ref", choices=("val_mse_ref", "composite"))
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--enqueue", default=None,
                    help="JSON list of param dicts to run first (e.g. a prior study's best)")
    a = ap.parse_args(argv)
    if a.storage is None:
        os.makedirs(a.outdir, exist_ok=True)
        a.storage = "sqlite:///" + os.path.join(a.outdir, f"optuna_{a.study}.db")
    if a.worker:
        return worker(a)
    if a.summarise:
        summarise(a)
        return 0
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = make_storage(a.storage)
    study = optuna.create_study(
        study_name=a.study, storage=storage, direction="minimize", load_if_exists=True,
        sampler=make_sampler(a.seed), pruner=make_pruner())
    if a.enqueue:
        with open(a.enqueue) as fh:
            for params in json.load(fh):
                study.enqueue_trial(params, skip_if_exists=True)
    study.set_user_attr("fixed", a.fixed or "")
    study.set_user_attr("space", a.space or "")
    study.set_user_attr("base", a.base or "")
    n_done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    remaining = max(0, a.n_trials - n_done)
    print(f"study {a.study}: {len(study.trials)} trials in db ({n_done} finished); "
          f"{remaining} to go, K={a.K}, timeout {a.timeout:.0f} s")
    if remaining == 0:
        summarise(a)
        return 0
    per = math.ceil(remaining / a.K)
    procs = []
    t0 = time.time()
    for k in range(a.K):
        args = [PY, "-m", "ml.phase2_optuna", "--worker", "--study", a.study, "--storage",
                a.storage, "--outdir", a.outdir, "--n-trials", str(per),
                "--timeout", str(a.timeout), "--objective", a.objective,
                "--worker-id", str(k), "--seed", str(a.seed)]
        for flag, val in (("--fixed", a.fixed), ("--space", a.space), ("--base", a.base)):
            if val:
                args += [flag, val]
        log = open(os.path.join(a.outdir, f"worker{k}.log"), "a")
        procs.append(subprocess.Popen(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(3)          # stagger the study loads
    rcs = [p.wait() for p in procs]
    print(f"workers done in {time.time()-t0:.0f} s, return codes {rcs}")
    summarise(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
