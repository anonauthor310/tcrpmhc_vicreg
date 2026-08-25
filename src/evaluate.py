"""Central evaluation metrics.

Implements global AUROC, peptide-weighted AUROC, peptide-macro AUROC,
and McClish partial AUROC at FPR 0.1. The score is negative MSE as in
``src.models.scoring``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, auc, roc_auc_score, roc_curve

from src.models.scoring import score_from_vectors  # noqa: F401  (re-exported)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def safe_partial_auc_raw(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("max_fpr must be in (0, 1]")
    if max_fpr not in fpr:
        stop = np.searchsorted(fpr, max_fpr, side="right")
        fpr_ext = np.concatenate([fpr[:stop], [max_fpr]])
        tpr_ext = np.concatenate([tpr[:stop], [np.interp(max_fpr, fpr, tpr)]])
    else:
        keep = fpr <= max_fpr
        fpr_ext = fpr[keep]
        tpr_ext = tpr[keep]
    return float(auc(fpr_ext, tpr_ext))


def safe_partial_auc_mcclish(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    """sklearn's ``max_fpr`` AUROC is the McClish standardised partial AUC."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))


def per_peptide_table(
    labels: np.ndarray,
    scores: np.ndarray,
    peptides: np.ndarray,
    max_fpr: float = 0.1,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    df = pd.DataFrame(
        {
            "label": labels.astype(int),
            "score": scores.astype(float),
            "peptide": peptides.astype(str),
        }
    )
    for pep, grp in df.groupby("peptide", sort=True):
        y = grp["label"].to_numpy()
        s = grp["score"].to_numpy()
        valid = len(np.unique(y)) == 2
        rows.append(
            {
                "peptide": pep,
                "n": int(len(grp)),
                "n_pos": int(y.sum()),
                "n_neg": int((y == 0).sum()),
                "auroc": float(roc_auc_score(y, s)) if valid else float("nan"),
                "auc0.1_raw": safe_partial_auc_raw(y, s, max_fpr) if valid else float("nan"),
                "auc0.1_mcclish": safe_partial_auc_mcclish(y, s, max_fpr) if valid else float("nan"),
                "valid": bool(valid),
            }
        )
    table = pd.DataFrame(rows).sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True)
    valid = table[table["valid"]].copy()
    if len(valid) == 0:
        summary = {
            "peptide_macro_auroc": float("nan"),
            "peptide_weighted_auroc": float("nan"),
            "peptide_macro_auc0.1_mcclish": float("nan"),
            "peptide_weighted_auc0.1_mcclish": float("nan"),
            "n_peptides_total": int(len(table)),
            "n_peptides_valid": 0,
        }
    else:
        summary = {
            "peptide_macro_auroc": float(valid["auroc"].mean()),
            "peptide_weighted_auroc": float(np.average(valid["auroc"], weights=valid["n"])),
            "peptide_macro_auc0.1_mcclish": float(valid["auc0.1_mcclish"].mean()),
            "peptide_weighted_auc0.1_mcclish": float(np.average(valid["auc0.1_mcclish"], weights=valid["n"])),
            "n_peptides_total": int(len(table)),
            "n_peptides_valid": int(len(valid)),
        }
    return table, summary


def metrics_for_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    peptides: np.ndarray,
    max_fpr: float = 0.1,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    pep_table, pep_summary = per_peptide_table(labels, scores, peptides, max_fpr)
    metrics = {
        "n_examples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "global_auroc": safe_auroc(labels, scores),
        "auprc": safe_auprc(labels, scores),
        "global_auc0.1_raw": safe_partial_auc_raw(labels, scores, max_fpr),
        "global_auc0.1_mcclish": safe_partial_auc_mcclish(labels, scores, max_fpr),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        **pep_summary,
    }
    return metrics, pep_table
