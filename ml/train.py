"""Train one configuration. Writes <out>/run.json (config, history, best epoch, the
train/val gap, val metrics against Kljun) and optionally best.pt and pred_val.npz.

    python -m ml.train --out results/ml/runs/b0 --set width=32 --set modes=16 ...

Early stopping is on the VAL masked asinh-MSE in the run's own transformed space. Two
further numbers are recorded so runs with different transforms can be compared:
`val_mse_ref`, the same masked MSE re-evaluated in the FILE's global asinh space from the
physical prediction, and `composite`, the geometric mean over the five production metrics
of median|err_FNO| / median|err_Kljun| on val (< 1 beats Kljun). The test split is never
loaded: ml.data.load_split refuses it.
"""
import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml import data as D                          # noqa: E402
from ml import features as F                      # noqa: E402
from ml import losses as L                        # noqa: E402
from ml import metrics as M                       # noqa: E402
from ml.model import build_model                  # noqa: E402


@dataclass
class TrainConfig:
    # features (ml/features.py)
    norm_mode: str = "global"
    knee: float = 1.0
    stab: str = "zL"
    dist: str = "lin_exp"
    exp_scale_m: float = 300.0
    statics: str = "B"
    weight: str = "none"
    # model
    head: str = "residual"          # residual | direct
    width: int = 32
    modes: int = 16
    depth: int = 4
    local: str = "conv1x1"          # none | conv1x1 | conv3x3
    film_hidden: int = 64
    dropout: float = 0.0
    # loss
    lam_peak: float = 0.0
    lam_int: float = 0.0
    int_ref: str = "target"         # target | asymptote
    peak_tau: float = 10.0
    # optimisation
    lr: float = 1e-3
    wd: float = 1e-4
    batch: int = 16
    epochs: int = 80
    warmup: int = 5
    patience: int = 20
    min_epochs: int = 10
    seed: int = 0
    eval_every: int = 10
    # io
    h5: str = D.H5_DEFAULT
    out: str = ""
    name: str = ""
    save_ckpt: bool = True
    save_pred: bool = False

    def feature_spec(self):
        return F.FeatureSpec(norm_mode=self.norm_mode, knee=self.knee, stab=self.stab,
                             dist=self.dist, exp_scale_m=self.exp_scale_m,
                             statics=self.statics, weight=self.weight)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _coerce(field, value):
    t = field.type
    if t is bool or t == "bool":
        return str(value).lower() in ("1", "true", "yes")
    if t is int or t == "int":
        return int(value)
    if t is float or t == "float":
        return float(value)
    return str(value)


def config_from_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="JSON file of TrainConfig fields")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    fields = {f.name: f for f in dataclasses.fields(TrainConfig)}
    kw = {}
    if a.config:
        with open(a.config) as fh:
            for k, v in json.load(fh).items():
                if k in fields:
                    kw[k] = _coerce(fields[k], v)
    for s in a.set:
        k, v = s.split("=", 1)
        if k not in fields:
            sys.exit(f"unknown config key {k!r}")
        kw[k] = _coerce(fields[k], v)
    kw["out"] = a.out
    kw.setdefault("name", os.path.basename(a.out.rstrip("/")))
    return TrainConfig(**kw)


class _Data:
    """Both splits as GPU tensors. Built once per process; shared across trials."""

    def __init__(self, cfg, dev):
        self.tr = D.load_split("train", cfg.h5)
        self.va = D.load_split("val", cfg.h5)
        self.st = D.load_statics()
        self.norm = D.read_norm(cfg.h5)
        self.dev = dev
        self.array = self.st["array"] > 0.5
        self._spec_key = None

    def features(self, cfg):
        spec = cfg.feature_spec()
        key = json.dumps(spec.to_dict(), sort_keys=True)
        if key != self._spec_key:
            self.fx_tr = F.Features(self.tr, self.st, self.norm, spec)
            self.fx_va = F.Features(self.va, self.st, self.norm, spec)
            self._spec_key = key
        return self.fx_tr, self.fx_va


def _tensors(fx, dev):
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    return dict(x_in=t(fx.x_in), scal=t(fx.scal), base=t(fx.base_T), tgt=t(fx.target_T),
                w=t(fx.weights), s_out=t(fx.s_out.astype(np.float32)),
                asym=t(fx.split.asymptote.astype(np.float32)),
                I_tgt=t((fx.split.target.sum(axis=(1, 2)) * M.CELL_AREA).astype(np.float32)))


def _predict_T(model, T, const, idx, head):
    r = model(T["x_in"][idx], const, T["scal"][idx])
    return T["base"][idx] + r if head == "residual" else r


def train_one(cfg, data=None, trial=None, log=print):
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    if data is None:
        data = _Data(cfg, dev)
    fx_tr, fx_va = data.features(cfg)
    Ttr, Tva = _tensors(fx_tr, dev), _tensors(fx_va, dev)
    const = torch.from_numpy(fx_tr.const).to(dev)
    valid = torch.from_numpy(fx_tr.valid).to(dev)
    X, Y = D.meshgrid_m()
    Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(dev)
    # the file's global space, for the cross-run comparable val_mse_ref
    s_ref = float(data.norm["target_scale"])
    tgt_ref = torch.from_numpy(np.arcsinh(data.va.target / s_ref).astype(np.float32)).to(dev)

    model = build_model(cfg, fx_tr.n_channels).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_tr = fx_tr.split.n
    steps_per_epoch = math.ceil(n_tr / cfg.batch)
    total_steps = cfg.epochs * steps_per_epoch
    warm = cfg.warmup * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def loss_fn(pred, idx, T, weighted):
        mse = L.masked_mse(pred, T["tgt"][idx], valid, T["w"][idx] if weighted else None)
        tot = mse
        # THE SCALES DIFFER BY THREE ORDERS OF MAGNITUDE. The masked MSE is a mean over
        # 14,884 mostly-zero cells and converges near 1e-4; the peak term (metres / 300 m)
        # and the integral term (dimensionless) are O(0.1). AUX_SCALE puts lam = 1 at
        # "comparable to the MSE" so that lam is a weight and not a switch.
        if cfg.lam_peak > 0:
            tot = tot + cfg.lam_peak * L.AUX_SCALE * L.peak_term(
                pred, T["tgt"][idx], valid, Xt, Yt, cfg.peak_tau)
        if cfg.lam_int > 0:
            ref = T["I_tgt"][idx] if cfg.int_ref == "target" else T["asym"][idx]
            tot = tot + cfg.lam_int * L.AUX_SCALE * L.integral_term(
                pred, T["s_out"][idx], valid, ref)
        return tot, mse

    @torch.no_grad()
    def val_pass():
        model.eval()
        n = fx_va.split.n
        preds = []
        for i in range(0, n, 64):
            idx = torch.arange(i, min(n, i + 64), device=dev)
            preds.append(_predict_T(model, Tva, const, idx, cfg.head))
        pred = torch.cat(preds)
        v_loss = float(L.masked_mse(pred, Tva["tgt"], valid))
        f_phys = Tva["s_out"][:, None, None] * torch.sinh(pred.clamp(-20, 20)) * valid
        v_ref = float(L.masked_mse(torch.asinh(f_phys / s_ref), tgt_ref, valid))
        model.train()
        return v_loss, v_ref, pred

    hist = []
    best = dict(epoch=-1, val_loss=float("inf"))
    best_state = None
    bad = 0
    step = 0
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_tr, generator=g).to(dev)
        tl, tm = 0.0, 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * cfg.batch:(b + 1) * cfg.batch]
            pred = _predict_T(model, Ttr, const, idx, cfg.head)
            tot, mse = loss_fn(pred, idx, Ttr, weighted=True)
            opt.zero_grad(set_to_none=True)
            tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            tl += float(tot) * len(idx)
            tm += float(mse) * len(idx)
        v_loss, v_ref, _ = val_pass()
        rec = dict(epoch=epoch, train_loss=tl / n_tr, train_mse=tm / n_tr, val_loss=v_loss,
                   val_mse_ref=v_ref, lr=opt.param_groups[0]["lr"],
                   t=round(time.time() - t_start, 1))
        hist.append(rec)
        if not np.isfinite(v_loss):
            log(f"  epoch {epoch}: non-finite val loss; stopping")
            break
        if v_loss < best["val_loss"]:
            best = dict(epoch=epoch, val_loss=v_loss, val_mse_ref=v_ref,
                        train_mse=tm / n_tr, train_loss=tl / n_tr)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            log(f"  epoch {epoch:3d} train {tm/n_tr:.5f} val {v_loss:.5f} "
                f"(ref {v_ref:.5f}) best {best['val_loss']:.5f}@{best['epoch']} "
                f"lr {rec['lr']:.2e} {rec['t']:.0f}s")
        if trial is not None:
            # val_mse_ref is in the FILE's global asinh space, so trials with different
            # transforms (knee, norm_mode) are pruned against a comparable number.
            trial.report(v_ref, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if bad >= cfg.patience and epoch + 1 >= cfg.min_epochs:
            log(f"  early stop at epoch {epoch}: no val improvement for {bad} epochs")
            break

    if best_state is None:
        raise RuntimeError("no finite epoch")
    model.load_state_dict(best_state)

    # ---- the evaluation at the best epoch --------------------------------------------
    v_loss, v_ref, pred_va = val_pass()
    f_va = fx_va.to_physical(pred_va.cpu().numpy())
    va = data.va
    sc = M.score_fields({"fno": f_va, "kljun": va.kljun}, va.target, va.wdir_deg,
                        data.array, va.asymptote)
    north = np.isin(va.octant.astype(str), ("N", "NE", "NW"))
    comp, ratios = M.composite(sc["fno"], sc["kljun"])
    comp_n, ratios_n = M.composite(sc["fno"], sc["kljun"], north)
    # the train side of the gap, same estimator
    with torch.no_grad():
        preds = []
        n = n_tr
        for i in range(0, n, 64):
            idx = torch.arange(i, min(n, i + 64), device=dev)
            preds.append(_predict_T(model, Ttr, const, idx, cfg.head))
        pred_tr = torch.cat(preds)
        tr_loss = float(L.masked_mse(pred_tr, Ttr["tgt"], valid))
    f_tr = fx_tr.to_physical(pred_tr.cpu().numpy())
    tr = data.tr
    sc_tr = M.score_fields({"fno": f_tr, "kljun": tr.kljun}, tr.target, tr.wdir_deg,
                           data.array, tr.asymptote)
    comp_tr, _ = M.composite(sc_tr["fno"], sc_tr["kljun"])

    out = dict(
        name=cfg.name, config=dataclasses.asdict(cfg), n_params=model.n_params(),
        channels=fx_tr.channel_names, n_train=n_tr, n_val=va.n,
        best_epoch=best["epoch"], epochs_run=len(hist), stopped_early=len(hist) < cfg.epochs,
        val_loss=v_loss, val_mse_ref=v_ref, train_loss_at_best=tr_loss,
        gap=dict(loss=v_loss - tr_loss, loss_ratio=v_loss / max(tr_loss, 1e-12),
                 composite_train=comp_tr, composite_val=comp),
        composite=comp, composite_ratios=ratios, composite_north=comp_n,
        composite_north_ratios=ratios_n,
        val_metrics=dict(fno=M.summarise(sc["fno"]), kljun=M.summarise(sc["kljun"]),
                         fno_north=M.summarise(sc["fno"], north),
                         kljun_north=M.summarise(sc["kljun"], north)),
        train_metrics=dict(fno=M.summarise(sc_tr["fno"]), kljun=M.summarise(sc_tr["kljun"])),
        history=hist, wall_s=round(time.time() - t_start, 1),
        peak_vram_gb=(torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0),
        device=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        torch=torch.__version__, git=_git_commit(), seed=cfg.seed,
    )
    if cfg.out:
        os.makedirs(cfg.out, exist_ok=True)
        with open(os.path.join(cfg.out, "run.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        if cfg.save_ckpt:
            torch.save(dict(state_dict=best_state, config=dataclasses.asdict(cfg),
                            channels=fx_tr.channel_names, norm=data.norm,
                            n_channels=fx_tr.n_channels), os.path.join(cfg.out, "best.pt"))
        if cfg.save_pred:
            np.savez_compressed(os.path.join(cfg.out, "pred_val.npz"), fno=f_va,
                                run_id=va.meta["run_id"])
    log(f"  done {cfg.name}: best epoch {best['epoch']} val {v_loss:.5f} ref {v_ref:.5f} "
        f"composite {comp:.3f} (north {comp_n:.3f}) gap x{out['gap']['loss_ratio']:.2f} "
        f"{out['wall_s']:.0f}s")
    return out


def main(argv=None):
    cfg = config_from_args(argv)
    train_one(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
