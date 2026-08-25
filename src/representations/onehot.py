"""One-hot amino-acid encoding used by the one-hot VICReg family."""

from __future__ import annotations

from typing import Tuple

import torch

AA20 = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {aa: i for i, aa in enumerate(AA20)}
VOCAB["X"] = len(VOCAB)
VOCAB["SEP"] = len(VOCAB)
VOCAB_SIZE = len(VOCAB)
UNK_IDX = VOCAB["X"]


def clean_seq(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    if s in {"NAN", "NONE", "<UNK>", "UNK", "UNKNOWN"}:
        return ""
    for ch in [" ", "-", ":", "|", ";", ","]:
        s = s.replace(ch, "")
    return s


def onehot_encode(seq: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(max_len, VOCAB_SIZE, dtype=torch.float32)
    m = torch.zeros(max_len, dtype=torch.bool)
    seq = clean_seq(seq)
    n = min(len(seq), max_len)
    for i, aa in enumerate(seq[:n]):
        x[i, VOCAB.get(aa, UNK_IDX)] = 1.0
        m[i] = True
    return x, m
