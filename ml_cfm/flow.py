"""The prior-anchored flow: interpolant, targets, the x -> v conversion and the ODE samplers.

    z_t = x_prior + t d + (1 - t) eps,   d = x_les - x_prior,   v = dz/dt = d - eps

Everything is (B, H, W) in the asinh target space; `mask` (H,W) or (B,H,W) confines the noise
(the pad is always excluded; the cone too when the gate is on).
"""
import torch


def interpolate(x_prior, x_les, eps, t):
    """z_t and its velocity target, t (B,)."""
    tt = t[:, None, None]
    d = x_les - x_prior
    z = x_prior + tt * d + (1 - tt) * eps
    return z, d - eps


def x_to_v(x_hat, z_t, t, t_clip=0.02):
    """Velocity implied by an x_les prediction: v = (x_hat - z_t) / (1 - t)."""
    return (x_hat - z_t) / (1 - t).clamp(min=t_clip)[:, None, None]


def velocity_fn(model, param, x_in, const, scal, mask):
    """Closure v(z, t) for the samplers. `const` (C,H,W) is broadcast over the batch."""
    def v(z, t):
        B = z.shape[0]
        x = torch.cat([z[:, None], x_in, const.unsqueeze(0).expand(B, -1, -1, -1)], dim=1)
        out = model(x, scal, t)
        if param == "x":
            out = x_to_v(out, z, t)
        return out * mask
    return v


@torch.no_grad()
def sample(v, z0, steps, solver="euler"):
    """Integrate dz/dt = v(z, t) from t = 0 to 1 with `steps` fixed steps."""
    z = z0
    B = z.shape[0]
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((B,), k * dt, device=z.device)
        if solver == "euler":
            z = z + dt * v(z, t)
        elif solver == "heun":
            k1 = v(z, t)
            if k == steps - 1:
                z = z + dt * k1
            else:
                k2 = v(z + dt * k1, t + dt)
                z = z + 0.5 * dt * (k1 + k2)
        elif solver == "rk4":
            k1 = v(z, t)
            k2 = v(z + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = v(z + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = v(z + dt * k3, t + dt)
            z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"solver {solver!r}")
    return z


@torch.no_grad()
def draw_samples(model, param, T, const, idx, mask, sigma, S, steps, solver, gen,
                 chunk=64):
    """S samples for the records idx: (S, m, H, W) in asinh space, plus wall seconds.
    The noise is drawn from `gen` (a CUDA generator), so a fixed seed gives fixed draws."""
    import time
    t0 = time.time()
    out = []
    n = len(idx)
    for s in range(S):
        rows = []
        for i in range(0, n, chunk):
            ii = idx[i:i + chunk]
            xp = T["base"][ii]
            m = mask[ii] if mask.dim() == 3 else mask
            eps = sigma * torch.randn(xp.shape, device=xp.device, generator=gen) * m
            v = velocity_fn(model, param, T["x_in"][ii], const, T["scal"][ii], m)
            rows.append(sample(v, xp + eps, steps, solver))
        out.append(torch.cat(rows))
    return torch.stack(out), time.time() - t0
