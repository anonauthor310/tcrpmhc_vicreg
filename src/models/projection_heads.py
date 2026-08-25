"""Low-rank projection heads used by all three VICReg families.

TCR view: one head of width ``d``.
pMHC view: peptide and MHC (HLA sequence) heads concatenated with
``d_P = round(R_PH * d)`` and ``d_H = d - d_P``. For the paper operating
point ``d=256`` and ``R_PH=0.7`` this is 179 / 77.

Each head is a low-rank bilinear map ``A_c^T X B_c`` flattened through
``H_c``, then a two-layer expander. The expander output is the VICReg
latent ``z``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LowRankProjectionHead(nn.Module):
    def __init__(self, D: int, rL: int, rD: int, d: int, L_max: int, dropout: float = 0.1):
        super().__init__()
        self.D = D
        self.rL = rL
        self.rD = rD
        self.d = d
        self.L_max = L_max
        self.B_c = nn.Parameter(torch.empty(D, rD))
        self.A_c = nn.Parameter(torch.empty(L_max, rL))
        self.H_c = nn.Parameter(torch.empty(rL * rD, d))
        nn.init.xavier_uniform_(self.B_c)
        nn.init.xavier_uniform_(self.A_c)
        nn.init.xavier_uniform_(self.H_c)
        self.expander = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    def project_pre_expander(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L_pad, D_in = emb.shape
        if D_in != self.D:
            raise ValueError(f"Embedding dimension mismatch: got {D_in}, expected {self.D}")
        if L_pad > self.L_max:
            raise ValueError(f"Sequence length {L_pad} exceeds L_max {self.L_max}")
        L_true = mask.sum(dim=1)
        z_list = []
        for b in range(B):
            Lb = int(L_true[b].item())
            if Lb == 0:
                z_list.append(torch.zeros(self.d, device=emb.device, dtype=emb.dtype))
                continue
            Xb = emb[b, :Lb, :] * mask[b, :Lb].unsqueeze(-1).float()
            Yb = Xb @ self.B_c
            Ub = self.A_c[:Lb, :].T @ Yb
            z_list.append(Ub.reshape(-1) @ self.H_c)
        return torch.stack(z_list, dim=0)

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.expander(self.project_pre_expander(emb, mask))


class PMHCProjectionHead(nn.Module):
    """Peptide and MHC heads concatenated to width ``d``."""

    def __init__(
        self,
        D: int,
        rL: int,
        rD: int,
        d: int,
        L_P_max: int,
        L_H_max: int,
        R_PH: float,
        dropout: float,
    ):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        if d_P <= 0 or d_H <= 0:
            raise ValueError(f"Invalid R_PH={R_PH}; produced d_P={d_P}, d_H={d_H}")
        self.d_P = d_P
        self.d_H = d_H
        self.pep_encoder = LowRankProjectionHead(D, rL, rD, d_P, L_P_max, dropout)
        self.hla_encoder = LowRankProjectionHead(D, rL, rD, d_H, L_H_max, dropout)

    def forward(
        self,
        emb_P: torch.Tensor,
        mask_P: torch.Tensor,
        emb_H: torch.Tensor,
        mask_H: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [self.pep_encoder(emb_P, mask_P), self.hla_encoder(emb_H, mask_H)],
            dim=-1,
        )
