"""The FNO. Written directly on torch.fft (no neuraloperator dependency): a spectral
convolution on the low-frequency corner blocks of rfft2, an optional parallel local path,
FiLM conditioning on the six scalars, and a projection whose last layer is ZERO-INITIALISED
so the residual head starts at exactly Kljun.

Input:  x_in  (B, 1, 128, 128)   asinh(kljun / s_in)
        const (C, 128, 128)      distance and static channels, identical for every record
        scal  (B, 6)             encoded scalars (FiLM only; never a spatial plane)
Output: r     (B, 128, 128)      the residual in transformed space; pred_T = base_T + r
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fnn


class SpectralConv2d(nn.Module):
    """Multiply the lowest `modes` Fourier coefficients (both signed frequencies on the
    first axis, non-negative on the rfft axis) by learned complex weights."""

    def __init__(self, cin, cout, modes1, modes2):
        super().__init__()
        self.cin, self.cout, self.m1, self.m2 = cin, cout, modes1, modes2
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, modes1, modes2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, modes1, modes2, 2))

    @staticmethod
    def _mul(a, w):
        return torch.einsum("bixy,ioxy->boxy", a, torch.view_as_complex(w))

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x.float(), norm="ortho")
        out = torch.zeros(B, self.cout, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = self._mul(xf[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = self._mul(xf[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, c_in, width=32, modes=16, depth=4, local="conv1x1",
                 film_hidden=64, dropout=0.0, n_scalars=6, proj_hidden=128):
        super().__init__()
        self.width, self.depth, self.local = width, depth, local
        self.lift = nn.Conv2d(c_in, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes)
                                   for _ in range(depth)])
        if local == "none":
            self.loc = None
        elif local == "conv1x1":
            self.loc = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        elif local == "conv3x3":
            self.loc = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1)
                                      for _ in range(depth)])
        else:
            raise ValueError(f"local {local!r}")
        # FiLM: one (gamma, beta) pair per channel per block, from the six scalars.
        self.film = nn.Sequential(nn.Linear(n_scalars, film_hidden), nn.GELU(),
                                  nn.Linear(film_hidden, 2 * width * depth))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)          # gamma = 1, beta = 0 at init
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else None
        self.proj1 = nn.Conv2d(width, proj_hidden, 1)
        self.proj2 = nn.Conv2d(proj_hidden, 1, 1)
        nn.init.zeros_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)             # residual is exactly 0 at init

    def forward(self, x_in, const, scal):
        B = x_in.shape[0]
        if const.shape[0]:
            x = torch.cat([x_in, const.unsqueeze(0).expand(B, -1, -1, -1)], dim=1)
        else:
            x = x_in
        x = self.lift(x)
        gb = self.film(scal).view(B, self.depth, 2, self.width)
        for k in range(self.depth):
            y = self.spec[k](x)
            if self.loc is not None:
                y = y + self.loc[k](x)
            gamma = 1.0 + gb[:, k, 0][:, :, None, None]
            beta = gb[:, k, 1][:, :, None, None]
            y = gamma * y + beta
            if k < self.depth - 1:
                y = Fnn.gelu(y)
                if self.drop is not None:
                    y = self.drop(y)
            x = y
        r = self.proj2(Fnn.gelu(self.proj1(x)))
        return r[:, 0]

    def n_params(self):
        return int(sum(p.numel() for p in self.parameters()))


def build_model(cfg, c_in):
    """cfg: any object with width, modes, depth, local, film_hidden, dropout attributes."""
    return FNO2d(c_in, width=cfg.width, modes=cfg.modes, depth=cfg.depth, local=cfg.local,
                 film_hidden=cfg.film_hidden, dropout=cfg.dropout)
