"""Novelty-regime definitions shared by split construction and evaluation.

A validation/test row is labelled by comparing its **source positive** against
the training positives. For an occurrence-matched decoy the source positive is
the row named by ``source_pair_id``, not the decoy's own donor TCR: the decoy
borrows a TCR from elsewhere in the same split, so labelling the decoy by its
own TCR would misreport novelty.

Regimes (evaluated on the MHC sequence, the peptide sequence, and the
concatenated alpha+beta TCR sequence):

    unseen_HLA         MHC sequence not in train (checked first)
    completely_unseen  TCR and peptide both absent from train
    unseen_TCR         TCR absent, peptide present
    unseen_peptide     TCR present, peptide absent
    both_seen          both present

``experiments/novelty_regimes.py`` uses these definitions to compute the
per-regime AUROC rows of Table 1.
"""

from __future__ import annotations

from typing import Iterable, Set

import pandas as pd

REGIMES = (
    "unseen_HLA",
    "completely_unseen",
    "unseen_TCR",
    "unseen_peptide",
    "both_seen",
)


def clean_seq(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return "".join(ch for ch in str(value).strip().upper() if ch.isalpha())


def assign_regime(
    tcr: str,
    peptide: str,
    mhc: str,
    train_tcr: Set[str],
    train_peptide: Set[str],
    train_mhc: Set[str],
) -> str:
    seen_tcr = tcr in train_tcr
    seen_peptide = peptide in train_peptide
    seen_mhc = mhc in train_mhc
    if not seen_mhc:
        return "unseen_HLA"
    if not seen_tcr and not seen_peptide:
        return "completely_unseen"
    if not seen_tcr:
        return "unseen_TCR"
    if not seen_peptide:
        return "unseen_peptide"
    return "both_seen"


def train_entity_sets(train: pd.DataFrame):
    return (
        set(train["TCR_full"].map(clean_seq)),
        set(train["Peptide"].map(clean_seq)),
        set(train["HLA_sequence"].map(clean_seq)),
    )


def label_split(split: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Add a ``novelty_regime`` column, resolving decoys via ``source_pair_id``."""
    train_tcr, train_peptide, train_mhc = train_entity_sets(train)
    out = split.copy()
    by_id = out.set_index("pair_id", drop=False)
    regimes = []
    for _, row in out.iterrows():
        source = row
        if int(row.get("binding_flag", 1)) == 0:
            sid = row.get("source_pair_id")
            if pd.notna(sid) and sid in by_id.index:
                candidate = by_id.loc[sid]
                source = candidate.iloc[0] if isinstance(candidate, pd.DataFrame) else candidate
        regimes.append(
            assign_regime(
                clean_seq(source["TCR_full"]),
                clean_seq(source["Peptide"]),
                clean_seq(source["HLA_sequence"]),
                train_tcr,
                train_peptide,
                train_mhc,
            )
        )
    out["novelty_regime"] = regimes
    return out
