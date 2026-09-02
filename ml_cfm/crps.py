"""CRPS for the CFM: the fair sample estimator (sorted O(S log S) form) as a training loss
through the model's own sampler, and the calibration table used by ml_cfm/calibrate.py.

    CRPS(x_1..x_S; y) = mean_s |x_s - y|  -  (1 / (2 S (S-1))) sum_{s != s'} |x_s - x_s'|

The second term is evaluated from the sorted samples: sum_{s<s'} (x_(s') - x_(s)) =
sum_i (2i - S - 1) x_(i). `alpha` mixes the fair (1/(S(S-1))) and the biased (1/S^2)
weights of the spread term (alpha = 1 is fair; Lang et al. 2024 use 0.95).
"""
import numpy as np
import torch


def crps_sorted(x, y, alpha=1.0):
    """x (S, ...) samples, y (...) observation -> CRPS (...). Differentiable."""
    S = x.shape[0]
    if S < 2:
        raise ValueError("CRPS needs at least two samples")
    term1 = (x - y.unsqueeze(0)).abs().mean(0)
    xs, _ = torch.sort(x, dim=0)
    i = torch.arange(1, S + 1, device=x.device, dtype=x.dtype)
    coef = (2 * i - S - 1).view(-1, *([1] * (x.dim() - 1)))
    pair_sum = (coef * xs).sum(0)                       # sum over s<s' of |x_s - x_s'|
    w = alpha / (S * (S - 1)) + (1 - alpha) / (S * S)
    return term1 - w * pair_sum


def crps_pairwise(x, y):
    """The O(S^2) definition, for the gate."""
    S = x.shape[0]
    term1 = (x - y.unsqueeze(0)).abs().mean(0)
    d = (x.unsqueeze(0) - x.unsqueeze(1)).abs().sum((0, 1))
    return term1 - d / (2 * S * (S - 1))


def crps_np(x, y, alpha=1.0):
    """numpy front end: x (S, n), y (n) -> (n)."""
    return crps_sorted(torch.from_numpy(np.asarray(x, np.float64)),
                       torch.from_numpy(np.asarray(y, np.float64)), alpha).numpy()


def sample_with_grad(model, param, x_in, const, scal, mask, z0, steps):
    """Euler integration of the learned velocity with gradients, one checkpoint per step.
    z0 (B,H,W); the closure is ml_cfm.flow.velocity_fn."""
    from torch.utils.checkpoint import checkpoint
    from ml_cfm import flow as FL
    v = FL.velocity_fn(model, param, x_in, const, scal, mask)
    z = z0
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((z.shape[0],), k * dt, device=z.device, dtype=z.dtype)
        z = z + dt * checkpoint(v, z, t, use_reentrant=False)
    return z


def crps_field_loss(model, cfg, T, const, idx, mask, gen, valid=None, arr=None, weights=None):
    """Pixelwise fair CRPS in asinh space over the mask cells, S = cfg.crps_S samples drawn
    through cfg.crps_steps Euler steps. Optional scalar CRPS on the physical array share
    (cfg.lam_share). Returns (loss, dict of the two terms)."""
    xp, xl = T["base"][idx], T["tgt"][idx]
    B, H, W = xp.shape
    S = cfg.crps_S
    m = mask[idx] if mask.dim() == 3 else mask.unsqueeze(0).expand(B, -1, -1)
    eps = cfg.sigma * torch.randn((S, B, H, W), generator=gen, device=xp.device) * m.unsqueeze(0)
    z0 = (xp.unsqueeze(0) + eps).reshape(S * B, H, W)
    rep = lambda a: a.unsqueeze(0).expand(S, *a.shape).reshape(S * B, *a.shape[1:])
    x = sample_with_grad(model, cfg.param, rep(T["x_in"][idx]), const, rep(T["scal"][idx]),
                         rep(m), z0, cfg.crps_steps).view(S, B, H, W)
    cmap = crps_sorted(x, xl, cfg.alpha_fair)                      # (B,H,W)
    per = (cmap * m).sum((-2, -1)) / m.sum((-2, -1)).clamp(min=1)
    terms = dict(crps_field=float(per.mean()))
    if cfg.lam_share > 0:
        s_out = T["s_out"][idx].view(1, B, 1, 1)
        f = s_out * torch.sinh(x.clamp(-20, 20)) * valid
        fl = T["s_out"][idx].view(B, 1, 1) * torch.sinh(xl.clamp(-20, 20)) * valid
        share = (f * arr).sum((-2, -1)) / f.sum((-2, -1)).clamp(min=1e-12)          # (S,B)
        share_l = (fl * arr).sum((-2, -1)) / fl.sum((-2, -1)).clamp(min=1e-12)      # (B,)
        cs = crps_sorted(share, share_l, cfg.alpha_fair)
        per = per + cfg.lam_share * cs
        terms["crps_share"] = float(cs.mean())
    if weights is not None:
        per = per * weights[idx]
    return per.mean(), terms


@torch.no_grad()
def crps_field_eval(samples_T, target_T, mask, chunk=16):
    """samples_T (S,n,H,W), target_T (n,H,W), mask (n,H,W) or (H,W) -> per-record masked
    mean CRPS (n,), on whatever device the inputs are on."""
    n = target_T.shape[0]
    out = torch.empty(n, dtype=torch.float64, device=target_T.device)
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        m = mask[i:j] if mask.dim() == 3 else mask.unsqueeze(0).expand(j - i, -1, -1)
        c = crps_sorted(samples_T[:, i:j].to(torch.float32), target_T[i:j], 1.0)
        out[i:j] = ((c * m).sum((-2, -1)) / m.sum((-2, -1)).clamp(min=1)).double()
    return out


# ------------------------------------------------------------------ the calibration table

def spread_skill(s, l):
    """s (S,n) samples, l (n) observations -> (spread, skill, ratio). spread = rms of the
    per-record sample sd corrected by sqrt((S+1)/S); skill = rmse of the sample mean."""
    S = s.shape[0]
    sd = s.std(0, ddof=1)
    spread = float(np.sqrt(np.mean(sd ** 2) * (S + 1) / S))
    skill = float(np.sqrt(np.mean((l - s.mean(0)) ** 2)))
    return spread, skill, spread / skill if skill > 0 else np.nan


def calibration_table(sm, les, mask=None, keys=("array_share", "integral", "peak_x", "centroid_dist")):
    """ml_cfm.evaluate.calibration plus CRPS and the spread-skill ratio, per key."""
    from ml_cfm import evaluate as E2
    base = E2.calibration(sm, les, mask)
    for k in keys:
        s = sm[k]
        l = les[k]
        if k == "array_share":
            s, l = s * 100.0, l * 100.0
        if mask is not None:
            s, l = s[:, mask], l[mask]
        ok = np.isfinite(l) & np.isfinite(s).all(0)
        s, l = s[:, ok], l[ok]
        c = crps_np(s, l)
        sp, sk, r = spread_skill(s, l)
        base[k].update(crps_mean=float(c.mean()), crps_median=float(np.median(c)),
                       mae_of_mean=float(np.mean(np.abs(l - s.mean(0)))),
                       spread=sp, skill_rmse=sk, spread_skill=r)
    return base
