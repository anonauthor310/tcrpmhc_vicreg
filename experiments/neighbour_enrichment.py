#!/usr/bin/env python3
"""Figure 2c: same-peptide k-nearest-neighbour enrichment.

Positive interactions only, grouped by peptide sequence irrespective of MHC.
The query set is the full positive split (no training-seed subsample or cap).

For each query row:
  * exclude the query itself and every row with the same exact TCR sequence;
  * take the k nearest remaining TCRs by Euclidean distance, k in {5, 10, 20};
  * observed purity = fraction of those k neighbours sharing the query peptide;
  * expected purity = prevalence of the query peptide in that query's candidate
    pool after identical-TCR exclusion;
  * query enrichment = observed / expected.

Aggregation is peptide-balanced over peptides with at least 5 distinct TCR
sequences (the same eligibility as Figure 2b): mean query enrichment within
each eligible peptide, then an equal mean across those peptides. k-NN itself
uses the full positive split after identical-TCR exclusion; low-count peptides
are omitted only from the reported average so that a 2-TCR peptide with
expected purity ~1/N cannot dominate. Training-seed variation therefore comes
from the learned spaces, not from a changing biological evaluation set.

    PYTHONPATH=. python experiments/neighbour_enrichment.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry_common import DETERMINISTIC_MODELS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
OUT_CSV = REPO / "results/paper_analysis/crossreactivity/knn_peptide_purity.csv"
SPLIT_CSV = {
    "test": REPO / "data/processed/test.csv.gz",
    "val": REPO / "data/processed/val.csv.gz",
    "immrep_test": REPO / "data/processed/immrep_test.csv.gz",
}

RUNS = [
    "onehot_vicreg_complete",
    "esm_vicreg_raw_complete",
    "esm_vicreg_finetuned_complete",
]
MODEL_ORDER = [
    "onehot_composition",
    "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool",
    "onehot_vicreg",
    "raw_esmc_vicreg",
    "finetuned_esmc_vicreg",
]
SEEDS = [31, 37, 43, 49, 55]
KNN_K = [5, 10, 20]


def clean_seq(s) -> str:
    if pd.isna(s):
        return ""
    return "".join(ch for ch in str(s).strip().upper() if ch.isalpha())


def is_raw_run(run_name: str) -> bool:
    s = run_name.lower()
    return ("raw" in s or "pretrained" in s) and "finetuned" not in s


def is_finetuned_run(run_name: str) -> bool:
    s = run_name.lower()
    return "finetuned" in s or "lora" in s or "adapted" in s


def sort_model_df(df: pd.DataFrame) -> pd.DataFrame:
    if "model_name" not in df.columns:
        return df
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    tmp = df.copy()
    tmp["_model_order"] = tmp["model_name"].map(order).fillna(999).astype(int)
    sort_cols = [c for c in ["split", "_model_order", "model_name", "k"] if c in tmp.columns]
    tmp = tmp.sort_values(sort_cols).drop(columns=["_model_order"])
    return tmp


def latent_arrays_from_npz(run_name: str, npz: Dict[str, np.ndarray], skip_duplicate_baselines_from_raw_runs: bool = True) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    rn = run_name.lower()

    if "zT_vicreg" in npz:
        arrays["onehot_vicreg__T"] = np.asarray(npz["zT_vicreg"])
    if "T_composition" in npz:
        arrays["onehot_composition__T"] = np.asarray(npz["T_composition"])

    if "zT_esm_vicreg" in npz:
        model = "raw_esmc_vicreg" if is_raw_run(rn) else "finetuned_esmc_vicreg" if is_finetuned_run(rn) else f"{run_name}__esm_vicreg"
        arrays[f"{model}__T"] = np.asarray(npz["zT_esm_vicreg"])

    if not (skip_duplicate_baselines_from_raw_runs and is_raw_run(rn)):
        if "T_finetuned_meanpool" in npz:
            arrays["finetuned_esmc_meanpool__T"] = np.asarray(npz["T_finetuned_meanpool"])
        if "T_pretrained_meanpool" in npz:
            arrays["pretrained_esmc_meanpool__T"] = np.asarray(npz["T_pretrained_meanpool"])
    return arrays


def load_representations_for_seed_split(seed: int, split: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    meta: Optional[pd.DataFrame] = None
    arrays: Dict[str, np.ndarray] = {}
    for rn in RUNS:
        p = OUTPUTS / rn / f"seed_{seed}" / f"{split}_latents.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as data:
            npz = {k: data[k] for k in data.files}
        if meta is None:
            meta = pd.DataFrame({
                "pair_id": np.asarray(npz["pair_id"]).astype(str),
                "label": np.asarray(npz["label"]).astype(int),
            })
            src = pd.read_csv(
                SPLIT_CSV[split],
                usecols=["pair_id", "TCR_full", "Peptide"],
            )
            src["pair_id"] = src["pair_id"].astype(str)
            src["tcr"] = src["TCR_full"].map(clean_seq)
            src["peptide"] = src["Peptide"].map(clean_seq)
            meta = meta.merge(src[["pair_id", "tcr", "peptide"]], on="pair_id", how="left")
        else:
            pid = np.asarray(npz["pair_id"]).astype(str)
            if len(pid) != len(meta) or not np.all(pid == meta["pair_id"].to_numpy(str)):
                warnings.warn(f"Latent row order mismatch for {p}; skipping this run.")
                continue
        arrays.update(latent_arrays_from_npz(rn, npz, skip_duplicate_baselines_from_raw_runs=True))
    if meta is None:
        return pd.DataFrame(), {}
    return meta, arrays


def peptide_balanced_knn_enrichment(
    Z: np.ndarray,
    peptides: np.ndarray,
    tcrs: np.ndarray,
    ks: Sequence[int],
    min_unique_tcrs: int = 5,
) -> List[Dict]:
    n = Z.shape[0]
    if n < 3:
        return []
    D = euclidean_distances(Z)
    np.fill_diagonal(D, np.inf)
    unique_tcr_count = {
        pep: len(set(tcrs[peptides == pep].tolist())) for pep in np.unique(peptides)
    }
    rows = []
    for k in ks:
        pep_means = []
        n_queries = 0
        obs_acc = []
        exp_acc = []
        for pep in np.unique(peptides):
            if unique_tcr_count[pep] < min_unique_tcrs:
                continue
            pep_idx = np.where(peptides == pep)[0]
            q_enrich = []
            for i in pep_idx:
                cand = tcrs != tcrs[i]
                cand_idx = np.where(cand)[0]
                if len(cand_idx) < k:
                    continue
                n_same = int(np.sum(peptides[cand_idx] == pep))
                expected = n_same / float(len(cand_idx))
                if expected <= 0:
                    continue
                order = cand_idx[np.argsort(D[i, cand_idx], kind="mergesort")]
                neigh = order[:k]
                observed = float(np.mean(peptides[neigh] == pep))
                q_enrich.append(observed / expected)
                obs_acc.append(observed)
                exp_acc.append(expected)
            if q_enrich:
                pep_means.append(float(np.mean(q_enrich)))
                n_queries += len(q_enrich)
        if not pep_means:
            rows.append({
                "k": int(k),
                "purity_enrichment": float("nan"),
                "purity_mean": float("nan"),
                "random_purity_expected": float("nan"),
                "n_peptides": 0,
                "n_queries": 0,
            })
            continue
        rows.append({
            "k": int(k),
            "purity_enrichment": float(np.mean(pep_means)),
            "purity_mean": float(np.mean(obs_acc)) if obs_acc else float("nan"),
            "random_purity_expected": float(np.mean(exp_acc)) if exp_acc else float("nan"),
            "n_peptides": int(len(pep_means)),
            "n_queries": int(n_queries),
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--knn-k", nargs="+", type=int, default=list(KNN_K))
    p.add_argument("--out", type=Path, default=OUT_CSV)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    knn_rows = []
    seen_deterministic = {}
    for seed in args.seeds:
        for split in args.splits:
            meta, arrays = load_representations_for_seed_split(seed, split)
            if meta.empty or not arrays:
                continue
            pos = meta["label"].astype(int).to_numpy() == 1
            idx = np.where(pos)[0]
            peps = meta.loc[idx, "peptide"].to_numpy(str)
            tcrs = meta.loc[idx, "tcr"].to_numpy(str)
            for key, Z_all in arrays.items():
                if not key.endswith("__T"):
                    continue
                model = key[: -len("__T")]
                Z = np.asarray(Z_all[idx], dtype=float)
                for row in peptide_balanced_knn_enrichment(Z, peps, tcrs, args.knn_k):
                    rec = {"seed": seed, "split": split, "model_name": model, **row}
                    det_key = (split, model, int(row["k"]))
                    if model in DETERMINISTIC_MODELS:
                        prev = seen_deterministic.get(det_key)
                        if prev is not None:
                            if not np.isclose(prev, rec["purity_enrichment"], equal_nan=True):
                                raise RuntimeError(
                                    f"Deterministic {model} k={row['k']} enrichment changed "
                                    f"across training seeds (seed {seed} vs first evaluation)"
                                )
                            continue
                        seen_deterministic[det_key] = rec["purity_enrichment"]
                    knn_rows.append(rec)
    knn_df = sort_model_df(pd.DataFrame(knn_rows)) if knn_rows else pd.DataFrame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    knn_df.to_csv(args.out, index=False)
    print(knn_df.to_string(index=False))
    print(f"\nwrote {args.out} ({len(knn_df)} rows)")


if __name__ == "__main__":
    main()
