#!/usr/bin/env python3
"""Figure 2c: same-peptide k-nearest-neighbour enrichment.

Positive interactions only, grouped by peptide sequence. Peptides with fewer
than 5 positive TCR rows are dropped; groups larger than 25 rows are randomly
capped at 25 using a NumPy RNG seeded by the model-training seed. On that
sampled pool, Euclidean k-NN is computed for k in {5, 10, 20}, excluding the
query row itself (not every identical TCR sequence).

Observed purity is the same-peptide fraction of those k neighbours. Expected
purity for query i is (freq(peptide_i) - 1) / (n_sampled - 1). Reported
enrichment is the pooled mean observed purity divided by the pooled mean
expected purity. This is not a peptide-macro average of per-query ratios.

    PYTHONPATH=. python experiments/neighbour_enrichment.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
OUT_CSV = REPO / "results/paper_analysis/crossreactivity/knn_peptide_purity.csv"

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
                "peptide": np.asarray(npz["peptide"]).astype(str),
                "label": np.asarray(npz["label"]).astype(int),
            })
        else:
            pid = np.asarray(npz["pair_id"]).astype(str)
            if len(pid) != len(meta) or not np.all(pid == meta["pair_id"].to_numpy(str)):
                warnings.warn(f"Latent row order mismatch for {p}; skipping this run.")
                continue
        arrays.update(latent_arrays_from_npz(rn, npz, skip_duplicate_baselines_from_raw_runs=True))
    if meta is None:
        return pd.DataFrame(), {}
    return meta, arrays


def frequency_capped_positive_indices(
    meta: pd.DataFrame,
    min_group_size: int,
    max_per_peptide: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = meta[meta["label"].astype(int) == 1].copy()
    idxs: List[int] = []
    for _pep, grp in pos.groupby("peptide"):
        if len(grp) < min_group_size:
            continue
        indices = grp.index.to_numpy()
        if len(indices) > max_per_peptide:
            indices = rng.choice(indices, size=max_per_peptide, replace=False)
        idxs.extend(indices.tolist())
    return np.array(sorted(idxs), dtype=int)


def knn_pooled_enrichment(Z: np.ndarray, peptides: np.ndarray, ks: Sequence[int]) -> List[Dict]:
    n = Z.shape[0]
    if n < 3:
        return []
    max_k = min(max(ks), n - 1)
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    nbrs.fit(Z)
    _, indices = nbrs.kneighbors(Z)
    indices = indices[:, 1:]  # exclude the query row itself
    unique, counts = np.unique(peptides, return_counts=True)
    freq = dict(zip(unique, counts))
    rows = []
    for k in ks:
        if k > max_k:
            continue
        purities = []
        random_purities = []
        for i in range(n):
            neigh_peps = peptides[indices[i, :k]]
            purities.append(float(np.mean(neigh_peps == peptides[i])))
            random_purities.append(float((freq.get(peptides[i], 1) - 1) / max(n - 1, 1)))
        obs = float(np.mean(purities))
        rnd = float(np.mean(random_purities))
        rows.append({
            "k": int(k),
            "purity_mean": obs,
            "random_purity_expected": rnd,
            "purity_enrichment": float(obs / rnd) if rnd > 0 else np.nan,
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--knn-k", nargs="+", type=int, default=list(KNN_K))
    p.add_argument("--min-group-size", type=int, default=5)
    p.add_argument("--max-tcrs-per-peptide", type=int, default=25)
    p.add_argument("--out", type=Path, default=OUT_CSV)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    knn_rows = []
    for seed in args.seeds:
        for split in args.splits:
            meta, arrays = load_representations_for_seed_split(seed, split)
            if meta.empty or not arrays:
                continue
            idx = frequency_capped_positive_indices(
                meta, args.min_group_size, args.max_tcrs_per_peptide, seed
            )
            if len(idx) < 3:
                continue
            peps = meta.loc[idx, "peptide"].to_numpy(str)
            for key, Z_all in arrays.items():
                if not key.endswith("__T"):
                    continue
                model = key[: -len("__T")]
                Z = np.asarray(Z_all[idx], dtype=float)
                for row in knn_pooled_enrichment(Z, peps, args.knn_k):
                    knn_rows.append({"seed": seed, "split": split, "model_name": model, **row})
    knn_df = sort_model_df(pd.DataFrame(knn_rows)) if knn_rows else pd.DataFrame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    knn_df.to_csv(args.out, index=False)
    print(knn_df.to_string(index=False))
    print(f"\nwrote {args.out} ({len(knn_df)} rows)")


if __name__ == "__main__":
    main()
