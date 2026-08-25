"""Paper scoring rule and score-decomposition variants.

The training/evaluation score is negative mean squared error:

    s(T, P) = - (1/d) ||z_T - z_P||_2^2

which is ``-(z_T - z_P).pow(2).mean(dim=-1)`` in PyTorch. Higher is more
binder-like. Cosine is never used for model selection.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def score_from_vectors(zT: torch.Tensor, zPH: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mse_distance = (zT - zPH).pow(2).mean(dim=-1)
    return -mse_distance, mse_distance


def decompose_scores(zT: np.ndarray, zPH: np.ndarray) -> Dict[str, np.ndarray]:
    """Score variants used in the occurrence-matched diagnostic.

    On an occurrence-matched test set the TCR-norm score has AUROC 0.5
    because the TCR occurrence marginal is identical for positives and
    negatives.
    """
    zT = np.asarray(zT, dtype=np.float64)
    zPH = np.asarray(zPH, dtype=np.float64)
    diff = zT - zPH
    mse = np.mean(diff ** 2, axis=-1)
    tcr_norm = np.linalg.norm(zT, axis=-1)
    pmhc_norm = np.linalg.norm(zPH, axis=-1)
    dot = np.sum(zT * zPH, axis=-1)
    denom = np.clip(tcr_norm * pmhc_norm, 1e-12, None)
    cosine = dot / denom
    return {
        "full_score": -mse,
        "tcr_norm_score": -tcr_norm,
        "dot_product_score": dot,
        "cosine_score": cosine,
        "mse_distance": mse,
    }
