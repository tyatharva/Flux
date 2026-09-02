"""A small U-Net (2-4 M parameters) with FiLM conditioning on the six scalars and the flow
time t. Input (B, c_in, 128, 128) -> output (B, 128, 128). The last conv is zero-initialised,
so the untrained model predicts zero velocity and a sample is exactly x_prior + eps.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fnn


def t_embedding(t, dim=64):
    """Sinusoidal embedding of t in [0, 1], (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000.0 * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Block(nn.Module):
    """conv3x3 -> GroupNorm -> FiLM -> GELU, twice; residual 1x1 when widths differ."""

    def __init__(self, cin, cout, cond_dim, dropout=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.film = nn.Linear(cond_dim, 4 * cout)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)          # gamma = 1, beta = 0 at init
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cout = cout

    def forward(self, x, cond):
        g1, b1, g2, b2 = self.film(cond).view(-1, 4, self.cout, 1, 1).unbind(1)
        h = Fnn.gelu((1 + g1) * self.n1(self.c1(x)) + b1)
        h = self.drop(h)
        h = Fnn.gelu((1 + g2) * self.n2(self.c2(h)) + b2)
        return h + self.skip(x)


class UNetFiLM(nn.Module):
    def __init__(self, c_in, widths=(32, 64, 128, 192), film_hidden=128, dropout=0.0,
                 n_scalars=6, t_dim=64):
        super().__init__()
        self.widths = list(widths)
        self.cond = nn.Sequential(nn.Linear(n_scalars + t_dim, film_hidden), nn.GELU(),
                                  nn.Linear(film_hidden, film_hidden), nn.GELU())
        self.t_dim = t_dim
        cd = film_hidden
        self.inp = nn.Conv2d(c_in, widths[0], 3, padding=1)
        self.down = nn.ModuleList()
        prev = widths[0]
        for w in widths:
            self.down.append(Block(prev, w, cd, dropout))
            prev = w
        self.mid = Block(prev, prev, cd, dropout)
        self.up = nn.ModuleList()
        ws = list(widths)[::-1]
        for i, w in enumerate(ws):
            nxt = ws[i + 1] if i + 1 < len(ws) else ws[-1]
            self.up.append(Block(w + w, nxt, cd, dropout))
            prev = nxt
        self.out = nn.Conv2d(prev, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, scal, t):
        cond = self.cond(torch.cat([scal, t_embedding(t, self.t_dim)], dim=1))
        h = self.inp(x)
        skips = []
        for i, blk in enumerate(self.down):
            h = blk(h, cond)
            skips.append(h)
            if i < len(self.down) - 1:
                h = Fnn.avg_pool2d(h, 2)
        h = self.mid(h, cond)
        for i, blk in enumerate(self.up):
            s = skips[-1 - i]
            if h.shape[-1] != s.shape[-1]:
                h = Fnn.interpolate(h, scale_factor=2, mode="nearest")
            h = blk(torch.cat([h, s], dim=1), cond)
        return self.out(h)[:, 0]

    def n_params(self):
        return int(sum(p.numel() for p in self.parameters()))


def build_model(cfg, c_in):
    widths = tuple(int(w) for w in str(cfg.widths).split(","))
    return UNetFiLM(c_in, widths=widths, film_hidden=cfg.film_hidden, dropout=cfg.dropout)
