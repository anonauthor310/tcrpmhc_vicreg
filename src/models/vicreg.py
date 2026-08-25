"""VICReg loss matching the manuscript.

Variance uses the *population* (biased) estimator ``1/B`` via
``var(dim=0, unbiased=False)``. Covariance uses the *sample* estimator
``1/(B-1)``. The covariance penalty is the squared off-diagonal mass
within each view, averaged by latent width ``d``.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F


def vicreg_variance(u: torch.Tensor, gamma: float = 1.0, eps_var: float = 1e-4) -> torch.Tensor:
    u = u - u.mean(dim=0, keepdim=True)
    std = torch.sqrt(u.var(dim=0, unbiased=False) + eps_var)
    return F.relu(gamma - std).mean()


def vicreg_covariance(u: torch.Tensor) -> torch.Tensor:
    B, d = u.shape
    if B <= 1:
        return torch.tensor(0.0, device=u.device, dtype=u.dtype)
    u = u - u.mean(dim=0, keepdim=True)
    cov = (u.T @ u) / (B - 1)
    cov_off = cov - torch.diag_embed(torch.diag(cov))
    return (cov_off ** 2).sum() / d


def plain_vicreg_loss(
    zT: torch.Tensor,
    zPH: torch.Tensor,
    alpha: float,
    beta: float,
    delta: float,
    gamma_var: float = 1.0,
    eps_var: float = 1e-4,
    return_parts: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    L_inv = F.mse_loss(zT, zPH)
    L_var = vicreg_variance(zT, gamma_var, eps_var) + vicreg_variance(zPH, gamma_var, eps_var)
    L_cov = vicreg_covariance(zT) + vicreg_covariance(zPH)
    loss = alpha * L_inv + beta * L_var + delta * L_cov
    if not return_parts:
        return loss
    return loss, {
        "L_total": float(loss.detach().cpu()),
        "L_inv": float(L_inv.detach().cpu()),
        "L_var": float(L_var.detach().cpu()),
        "L_cov": float(L_cov.detach().cpu()),
        "weighted_inv": float((alpha * L_inv).detach().cpu()),
        "weighted_var": float((beta * L_var).detach().cpu()),
        "weighted_cov": float((delta * L_cov).detach().cpu()),
        "zT_std": float(zT.std(unbiased=False).detach().cpu()),
        "zPH_std": float(zPH.std(unbiased=False).detach().cpu()),
    }
