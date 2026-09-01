"""Loss terms, all in torch and all masked to the 122^2 interior.

    total = masked_mse + lam_peak * peak_term + lam_int * integral_term

masked_mse     : per-record mean squared error over valid cells in asinh space, then the
                 (optionally weighted) mean over the batch. The pad never enters.
peak_term      : distance in metres between the soft-argmax locations of the predicted and
                 the target transformed fields, / 300 m. Soft-argmax with a temperature is
                 the differentiable stand-in for the 5-cell-smoothed argmax the production
                 peak metric uses.
integral_term  : |sum(f_pred) * 900 - I_ref| with f_pred in m^-2 and I_ref either the
                 target's own integral or the Steinfeld asymptote 1 - z_m/z_i.
"""
import torch

# lam = 1 on an auxiliary term means "worth about one converged MSE" (~1e-4 against O(0.1)
# aux values); see ml/train.py loss_fn. A weight, not a switch.
AUX_SCALE = 1e-3


def masked_mse(pred_T, target_T, valid, weights=None):
    """pred_T, target_T (B,H,W); valid (H,W) bool; weights (B,) or None."""
    v = valid.to(pred_T.dtype)
    per = (((pred_T - target_T) ** 2) * v).sum(dim=(-2, -1)) / v.sum()
    if weights is not None:
        per = per * weights
    return per.mean()


def soft_peak_xy(field_T, valid, X, Y, tau):
    """Expected (east, north) under softmax(tau * field) over valid cells. (B,), (B,)."""
    logits = (field_T * tau).masked_fill(~valid, -1e9).flatten(1)
    p = torch.softmax(logits, dim=-1)
    return (p * X.flatten()).sum(-1), (p * Y.flatten()).sum(-1)


def peak_term(pred_T, target_T, valid, X, Y, tau=10.0, scale_m=300.0):
    px, py = soft_peak_xy(pred_T, valid, X, Y, tau)
    tx, ty = soft_peak_xy(target_T.detach(), valid, X, Y, tau)
    d = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2 + 1e-6)
    return (d / scale_m).mean()


def physical(pred_T, s_out, valid, cell_area=900.0):
    """Transformed prediction -> m^-2 field on the interior, zero elsewhere."""
    f = s_out[:, None, None] * torch.sinh(pred_T.clamp(-20.0, 20.0))
    return f * valid.to(f.dtype)


def integral_term(pred_T, s_out, valid, ref, cell_area=900.0):
    f = physical(pred_T, s_out, valid)
    integ = f.sum(dim=(-2, -1)) * cell_area
    return (integ - ref).abs().mean()
