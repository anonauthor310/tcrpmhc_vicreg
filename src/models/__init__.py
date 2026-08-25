"""Projection heads, VICReg loss and the scoring rule.

The per-family training loops in ``src/trainers/`` import the loss and score
from here rather than keeping their own copies.
"""

from src.models.projection_heads import LowRankProjectionHead, PMHCProjectionHead
from src.models.scoring import score_from_vectors
from src.models.vicreg import plain_vicreg_loss, vicreg_covariance, vicreg_variance

__all__ = [
    "LowRankProjectionHead",
    "PMHCProjectionHead",
    "plain_vicreg_loss",
    "score_from_vectors",
    "vicreg_covariance",
    "vicreg_variance",
]
