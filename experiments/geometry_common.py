"""Shared loading helpers for Figure 2b and 2d.

Only utilities used by both same-peptide recovery and multi-cognate retrieval:
model labels, sequence cleaning, metadata attachment, and latent loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
OUT_DIR = REPO / "results/paper_analysis/refined_geometry"

MODEL_ORDER = [
    "onehot_composition",
    "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool",
    "onehot_vicreg",
    "raw_esmc_vicreg",
    "finetuned_esmc_vicreg",
]
MODEL_LABELS = {
    "onehot_composition": "One-hot",
    "pretrained_esmc_meanpool": "Raw ESMC",
    "finetuned_esmc_meanpool": "LoRA ESMC",
    "onehot_vicreg": "One-hot+VICReg",
    "raw_esmc_vicreg": "Raw ESMC+VICReg",
    "finetuned_esmc_vicreg": "LoRA ESMC+VICReg",
}
RUNS = [
    "onehot_vicreg_complete",
    "esm_vicreg_raw_complete",
    "esm_vicreg_finetuned_complete",
]
DETERMINISTIC_MODELS = {
    "onehot_composition",
    "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool",
}
SEEDS = [31, 37, 43, 49, 55]


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_seq(s) -> str:
    if pd.isna(s):
        return ""
    return "".join(ch for ch in str(s).strip().upper() if ch.isalpha())


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def attach_meta(split: str, pair_ids: Sequence[str]) -> pd.DataFrame:
    path = {
        "test": REPO / "data/processed/test.csv.gz",
        "val": REPO / "data/processed/val.csv.gz",
    }[split]
    meta = pd.read_csv(path, usecols=["pair_id", "TCR_full", "Peptide", "HLA_sequence", "binding_flag"])
    meta["pair_id"] = meta["pair_id"].astype(str)
    meta = meta.set_index("pair_id").reindex(list(pair_ids)).reset_index()
    meta["tcr"] = meta["TCR_full"].map(clean_seq)
    meta["pep"] = meta["Peptide"].map(clean_seq)
    meta["hla"] = meta["HLA_sequence"].map(clean_seq)
    meta["pmhc"] = meta["pep"] + "|" + meta["hla"]
    return meta


def model_arrays_from_run(run_name: str, npz: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    rn = run_name.lower()
    if "zT_vicreg" in npz and "zPH_vicreg" in npz:
        out["onehot_vicreg"] = (np.asarray(npz["zT_vicreg"], float), np.asarray(npz["zPH_vicreg"], float))
    if "T_composition" in npz and "PH_composition" in npz:
        out["onehot_composition"] = (np.asarray(npz["T_composition"], float), np.asarray(npz["PH_composition"], float))
    if "zT_esm_vicreg" in npz and "zPH_esm_vicreg" in npz:
        key = "raw_esmc_vicreg" if "raw" in rn else "finetuned_esmc_vicreg"
        out[key] = (np.asarray(npz["zT_esm_vicreg"], float), np.asarray(npz["zPH_esm_vicreg"], float))
    if "T_pretrained_meanpool" in npz and "PH_pretrained_meanpool" in npz:
        out["pretrained_esmc_meanpool"] = (
            np.asarray(npz["T_pretrained_meanpool"], float),
            np.asarray(npz["PH_pretrained_meanpool"], float),
        )
    if "T_finetuned_meanpool" in npz and "PH_finetuned_meanpool" in npz:
        out["finetuned_esmc_meanpool"] = (
            np.asarray(npz["T_finetuned_meanpool"], float),
            np.asarray(npz["PH_finetuned_meanpool"], float),
        )
    return out


def load_seed_split(split: str, seed: int) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    models: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    meta = None
    pair_ids = None
    for run_name in RUNS:
        p = OUTPUTS / run_name / f"seed_{seed}" / f"{split}_latents.npz"
        if not p.exists():
            print(f"MISSING {p}", flush=True)
            continue
        npz = dict(np.load(p, allow_pickle=True))
        if pair_ids is None:
            pair_ids = np.asarray(npz["pair_id"]).astype(str)
            labels = np.asarray(npz["label"]).astype(int)
            peptides = np.asarray(npz["peptide"]).astype(str)
            meta = attach_meta(split, pair_ids)
            meta["label"] = labels
            meta["peptide_latent"] = peptides
        else:
            pid = np.asarray(npz["pair_id"]).astype(str)
            if len(pid) != len(pair_ids) or not np.all(pid == pair_ids):
                raise RuntimeError(f"Row-order mismatch in {p}")
        models.update(model_arrays_from_run(run_name, npz))
    if meta is None:
        raise FileNotFoundError(f"No latents found for {split} seed {seed}")
    return meta, models
