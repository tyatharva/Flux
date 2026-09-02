"""Train one prior-anchored CFM configuration. Writes <out>/run.json, best.pt and, on
request, samples_val.npz. Mirrors ml/train.py's outputs so the same summary code reads both.

    python -m ml_cfm.train --out results/ml_cfm/phase1/v_s0.1 --set param=velocity ...

Selection is on `val_mse_ref`: the masked MSE, in the file's asinh space, of the ODE sample
MEAN (S samples, fixed noise seed) against the val target -- the same number the FNO runs
were selected on. The flow-matching loss on fixed (t, eps) draws is recorded per epoch as a
low-variance history. The test split is never loaded: ml.data.load_split refuses it.
"""
import argparse
import copy
import dataclasses
import json
import math
import os
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
from ml.train import _git_commit, _coerce         # noqa: E402
from ml_cfm import flow as FL                     # noqa: E402
from ml_cfm.model import build_model              # noqa: E402


@dataclass
class CfmConfig:
    # features: the FNO's final choices (ml/features.py)
    norm_mode: str = "global"
    knee: float = 1.0
    stab: str = "zL"
    dist: str = "lin_exp_xy"
    exp_scale_m: float = 300.0
    statics: str = "none"
    weight: str = "none"
    # flow
    param: str = "velocity"        # velocity | x
    sigma: float = 0.1             # noise sd in asinh space
    gate: str = "cone"             # cone | none: noise, prior and samples confined to the cone
    t_weight: str = "uniform"      # uniform | logit_normal
    lam_l1: float = 0.0
    # model
    widths: str = "32,64,128,192"
    film_hidden: int = 128
    dropout: float = 0.0
    ema: float = 0.999             # 0 = off
    # sampling used for selection during training
    S_sel: int = 4
    steps_sel: int = 16
    solver: str = "euler"
    # optimisation
    lr: float = 1e-3
    wd: float = 1e-4
    batch: int = 16
    epochs: int = 300
    warmup: int = 5
    patience: int = 10             # in evaluations (every eval_every epochs)
    eval_every: int = 5
    min_epochs: int = 20
    seed: int = 0
    # io
    h5: str = D.H5_DEFAULT
    out: str = ""
    name: str = ""
    save_ckpt: bool = True
    save_samples: int = 0          # S samples of val to write at the end (0 = none)
    steps_final: int = 16

    def feature_spec(self):
        return F.FeatureSpec(norm_mode=self.norm_mode, knee=self.knee, stab=self.stab,
                             dist=self.dist, exp_scale_m=self.exp_scale_m,
                             statics=self.statics, weight=self.weight)


def config_from_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    fields = {f.name: f for f in dataclasses.fields(CfmConfig)}
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
    return CfmConfig(**kw)


class _Data:
    def __init__(self, cfg, dev):
        self.tr = D.load_split("train", cfg.h5)
        self.va = D.load_split("val", cfg.h5)
        self.st = D.load_statics()
        self.norm = D.read_norm(cfg.h5)
        self.dev = dev
        self.array = self.st["array"] > 0.5
        self._spec_key = None

    def cones(self):
        if not hasattr(self, "_cones"):
            self._cones = (D.cone_masks(self.tr, verbose=False),
                           D.cone_masks(self.va, verbose=False))
        return self._cones

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
                w=t(fx.weights), s_out=t(fx.s_out.astype(np.float32)))


def masks_for(cfg, data, valid, dev):
    """(H,W) valid mask, or (n,H,W) cone-and-valid masks per split when gated."""
    if cfg.gate == "cone":
        ctr, cva = data.cones()
        return (torch.from_numpy((ctr & valid.cpu().numpy()).astype(np.float32)).to(dev),
                torch.from_numpy((cva & valid.cpu().numpy()).astype(np.float32)).to(dev))
    if cfg.gate != "none":
        raise ValueError(f"gate {cfg.gate!r}")
    v = valid.to(torch.float32)
    return v, v


def draw_t(n, cfg, gen, dev):
    u = torch.rand(n, generator=gen, device=dev)
    if cfg.t_weight == "logit_normal":
        z = torch.randn(n, generator=gen, device=dev)
        return torch.sigmoid(z)
    if cfg.t_weight != "uniform":
        raise ValueError(f"t_weight {cfg.t_weight!r}")
    return u


def fm_loss(model, cfg, T, const, idx, mask, t, eps, weights=None):
    """The flow-matching loss on given (t, eps) draws for records idx."""
    xp, xl = T["base"][idx], T["tgt"][idx]
    m = mask[idx] if mask.dim() == 3 else mask
    eps = eps * m
    z, v_tgt = FL.interpolate(xp, xl, eps, t)
    B = len(idx)
    x = torch.cat([z[:, None], T["x_in"][idx], const.unsqueeze(0).expand(B, -1, -1, -1)], dim=1)
    out = model(x, T["scal"][idx], t)
    target = v_tgt if cfg.param == "velocity" else xl
    if cfg.param not in ("velocity", "x"):
        raise ValueError(f"param {cfg.param!r}")
    # masked MSE per record over the cells the noise lives on (the pad never enters)
    per = (((out - target) ** 2) * m).sum(dim=(-2, -1)) / m.sum(dim=(-2, -1)).clamp(min=1)
    if cfg.lam_l1 > 0:
        per = per + cfg.lam_l1 * (((out - target).abs()) * m).sum(dim=(-2, -1)) \
            / m.sum(dim=(-2, -1)).clamp(min=1)
    if weights is not None:
        per = per * weights[idx]
    return per.mean()


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
    mtr, mva = masks_for(cfg, data, valid, dev)
    if cfg.gate == "cone":
        # the prior is cropped to the cone like the FNO's gate: outside it z_t = 0, v = 0,
        # and the sample is exactly 0 where the target is 0 by construction
        Ttr["base"] = Ttr["base"] * mtr
        Tva["base"] = Tva["base"] * mva
    s_ref = float(data.norm["target_scale"])
    tgt_ref = torch.from_numpy(np.arcsinh(data.va.target / s_ref).astype(np.float32)).to(dev)
    c_in = 1 + fx_tr.n_channels                     # z_t + kljun_T + const channels

    model = build_model(cfg, c_in).to(dev)
    ema = copy.deepcopy(model).eval() if cfg.ema > 0 else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    n_tr, n_va = fx_tr.split.n, fx_va.split.n
    steps_per_epoch = math.ceil(n_tr / cfg.batch)
    total_steps = cfg.epochs * steps_per_epoch
    warm = cfg.warmup * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    g = torch.Generator(device=dev).manual_seed(cfg.seed)
    g_cpu = torch.Generator(device="cpu").manual_seed(cfg.seed)
    # fixed val draws for the low-variance FM-loss history
    g_fix = torch.Generator(device=dev).manual_seed(12345)
    t_fix = torch.rand(n_va, generator=g_fix, device=dev)
    eps_fix = cfg.sigma * torch.randn((n_va, D.N, D.N), generator=g_fix, device=dev)

    def eval_model():
        return ema if ema is not None else model

    @torch.no_grad()
    def val_pass(S=cfg.S_sel, steps=cfg.steps_sel, solver=cfg.solver, seed=777):
        m = eval_model()
        m.eval()
        fl = 0.0
        for i in range(0, n_va, 64):
            idx = torch.arange(i, min(n_va, i + 64), device=dev)
            fl += float(fm_loss(m, cfg, Tva, const, idx, mva, t_fix[idx], eps_fix[idx])) * len(idx)
        gs = torch.Generator(device=dev).manual_seed(seed)
        idx = torch.arange(n_va, device=dev)
        samp, secs = FL.draw_samples(m, cfg.param, Tva, const, idx, mva, cfg.sigma, S, steps,
                                     solver, gs)
        mean_T = samp.mean(0)
        f_phys = Tva["s_out"][:, None, None] * torch.sinh(mean_T.clamp(-20, 20)) * valid
        v_ref = float(L.masked_mse(torch.asinh(f_phys / s_ref), tgt_ref, valid))
        model.train()
        return fl / n_va, v_ref, mean_T, samp, secs

    hist = []
    best = dict(epoch=-1, val_mse_ref=float("inf"))
    best_state = None
    bad = 0
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_tr, generator=g_cpu).to(dev)
        tl = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * cfg.batch:(b + 1) * cfg.batch]
            t = draw_t(len(idx), cfg, g, dev)
            eps = cfg.sigma * torch.randn((len(idx), D.N, D.N), generator=g, device=dev)
            loss = fm_loss(model, cfg, Ttr, const, idx, mtr, t, eps, Ttr["w"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(cfg.ema).add_(pm.detach(), alpha=1 - cfg.ema)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            tl += float(loss) * len(idx)
        rec = dict(epoch=epoch, train_loss=tl / n_tr, lr=opt.param_groups[0]["lr"],
                   t=round(time.time() - t_start, 1))
        do_eval = (epoch % cfg.eval_every == 0) or epoch == cfg.epochs - 1
        if do_eval:
            v_fm, v_ref, _, _, secs = val_pass()
            rec.update(val_fm=v_fm, val_mse_ref=v_ref, sample_s=round(secs, 2))
            if not np.isfinite(v_ref):
                log(f"  epoch {epoch}: non-finite val; stopping")
                hist.append(rec)
                break
            if v_ref < best["val_mse_ref"]:
                best = dict(epoch=epoch, val_mse_ref=v_ref, val_fm=v_fm, train_loss=tl / n_tr)
                best_state = {k: v.detach().clone() for k, v in eval_model().state_dict().items()}
                bad = 0
            else:
                bad += 1
            log(f"  epoch {epoch:3d} train {tl/n_tr:.5f} val_fm {v_fm:.5f} ref {v_ref:.6f} "
                f"best {best['val_mse_ref']:.6f}@{best['epoch']} lr {rec['lr']:.2e} "
                f"{rec['t']:.0f}s")
            if trial is not None:
                trial.report(v_ref, epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
        hist.append(rec)
        if do_eval and bad >= cfg.patience and epoch + 1 >= cfg.min_epochs:
            log(f"  early stop at epoch {epoch}: no val improvement for {bad} evaluations")
            break

    if best_state is None:
        raise RuntimeError("no finite evaluation")
    final = eval_model()
    final.load_state_dict(best_state)
    final.eval()

    # ---- the evaluation at the best epoch: sample mean vs Kljun on val and train --------
    v_fm, v_ref, mean_va, samp_va, secs = val_pass(S=max(cfg.S_sel, cfg.save_samples or 0),
                                                   steps=cfg.steps_final)
    f_va = fx_va.to_physical(mean_va.cpu().numpy())
    va = data.va
    sc = M.score_fields({"cfm": f_va, "kljun": va.kljun}, va.target, va.wdir_deg,
                        data.array, va.asymptote)
    north = np.isin(va.octant.astype(str), ("N", "NE", "NW"))
    comp, ratios = M.composite(sc["cfm"], sc["kljun"])
    comp_n, ratios_n = M.composite(sc["cfm"], sc["kljun"], north)
    with torch.no_grad():
        gs = torch.Generator(device=dev).manual_seed(778)
        idx = torch.arange(n_tr, device=dev)
        samp_tr, _ = FL.draw_samples(final, cfg.param, Ttr, const, idx, mtr, cfg.sigma,
                                     cfg.S_sel, cfg.steps_final, cfg.solver, gs)
        mean_tr = samp_tr.mean(0)
        s_ref_t = torch.from_numpy(np.arcsinh(data.tr.target / s_ref).astype(np.float32)).to(dev)
        f_tr_phys = Ttr["s_out"][:, None, None] * torch.sinh(mean_tr.clamp(-20, 20)) * valid
        tr_ref = float(L.masked_mse(torch.asinh(f_tr_phys / s_ref), s_ref_t, valid))
    f_tr = fx_tr.to_physical(mean_tr.cpu().numpy())
    tr = data.tr
    sc_tr = M.score_fields({"cfm": f_tr, "kljun": tr.kljun}, tr.target, tr.wdir_deg,
                           data.array, tr.asymptote)
    comp_tr, _ = M.composite(sc_tr["cfm"], sc_tr["kljun"])

    out = dict(
        name=cfg.name, config=dataclasses.asdict(cfg), n_params=final.n_params(),
        channels=["z_t"] + fx_tr.channel_names, n_train=n_tr, n_val=va.n,
        best_epoch=best["epoch"], epochs_run=len(hist), stopped_early=len(hist) < cfg.epochs,
        val_loss=v_ref, val_mse_ref=v_ref, val_fm_loss=v_fm, train_mse_ref_at_best=tr_ref,
        gap=dict(loss=v_ref - tr_ref, loss_ratio=v_ref / max(tr_ref, 1e-12),
                 composite_train=comp_tr, composite_val=comp),
        composite=comp, composite_ratios=ratios, composite_north=comp_n,
        composite_north_ratios=ratios_n,
        val_metrics=dict(cfm=M.summarise(sc["cfm"]), kljun=M.summarise(sc["kljun"]),
                         cfm_north=M.summarise(sc["cfm"], north),
                         kljun_north=M.summarise(sc["kljun"], north)),
        train_metrics=dict(cfm=M.summarise(sc_tr["cfm"]), kljun=M.summarise(sc_tr["kljun"])),
        sampling=dict(S=int(samp_va.shape[0]), steps=cfg.steps_final, solver=cfg.solver,
                      seconds_total=round(secs, 2),
                      ms_per_record_per_sample=round(1000 * secs / (n_va * samp_va.shape[0]), 3)),
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
                            channels=out["channels"], norm=data.norm, c_in=c_in),
                       os.path.join(cfg.out, "best.pt"))
        if cfg.save_samples:
            np.savez_compressed(os.path.join(cfg.out, "samples_val.npz"),
                                samples_T=samp_va[:cfg.save_samples].cpu().numpy().astype(np.float16),
                                mean=f_va, run_id=va.meta["run_id"], s_out=fx_va.s_out)
    log(f"  done {cfg.name}: best epoch {best['epoch']} ref {v_ref:.6f} composite {comp:.3f} "
        f"(north {comp_n:.3f}) gap x{out['gap']['loss_ratio']:.2f} {out['wall_s']:.0f}s")
    return out


def main(argv=None):
    cfg = config_from_args(argv)
    train_one(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
