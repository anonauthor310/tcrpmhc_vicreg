#!/usr/bin/env python3
"""Figure 2b: same-peptide recovery AUROC.

Positive internal-test interactions only. Eligibility and pair sampling are
over distinct TCR sequences (exact cleaned alpha+beta string), not raw table
rows. A peptide is eligible if it has at least 5 distinct TCRs and
C(n, 2) >= 10. Exactly 10 within-peptide pairs are drawn per eligible peptide
(35 peptides, 350 same-peptide pairs). Different-peptide pairs (350) sample
two distinct eligible peptides uniformly, then one distinct TCR from each.

Pair indices are drawn once with --eval-seed (independent of training seeds)
and reused for every representation. Every sampled pair is asserted to have
two different TCR sequences. Score is negative Euclidean TCR-TCR distance;
smaller distance => higher score. One pooled AUROC.

    PYTHONPATH=. python experiments/same_peptide_recovery.py --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry_common import (  # noqa: E402
    DETERMINISTIC_MODELS,
    OUT_DIR,
    SEEDS,
    load_seed_split,
    mkdir,
    safe_auroc,
)


def _unique_tcr_indices(peps: np.ndarray, tcrs: np.ndarray) -> Dict[str, np.ndarray]:
    """First occurrence of each distinct TCR within each peptide, sorted peptides."""
    pep_to_idx: Dict[str, list] = {}
    pep_seen: Dict[str, set] = {}
    for i, (pep, tcr) in enumerate(zip(peps, tcrs)):
        if pep not in pep_to_idx:
            pep_to_idx[pep] = []
            pep_seen[pep] = set()
        if tcr in pep_seen[pep]:
            continue
        pep_seen[pep].add(tcr)
        pep_to_idx[pep].append(i)
    return {p: np.asarray(pep_to_idx[p], dtype=int) for p in sorted(pep_to_idx)}


def sample_peptide_balanced_pair_indices(
    peps: np.ndarray,
    tcrs: np.ndarray,
    seed: int,
    pairs_per_peptide: int,
    min_tcrs: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Fixed within pairs per peptide over distinct TCRs; between via peptide then TCR."""
    rng = np.random.default_rng(seed)
    pep_to_idx = _unique_tcr_indices(peps, tcrs)
    eligible = [
        p
        for p, idx in pep_to_idx.items()
        if len(idx) >= min_tcrs and len(idx) * (len(idx) - 1) // 2 >= pairs_per_peptide
    ]
    within = []
    for p in eligible:
        idx = pep_to_idx[p]
        pairs = [(a, b) for a in range(len(idx)) for b in range(a + 1, len(idx))]
        chosen = rng.choice(len(pairs), size=pairs_per_peptide, replace=False)
        for c in chosen:
            a, b = pairs[int(c)]
            within.append((int(idx[a]), int(idx[b])))
    within = np.asarray(within, dtype=int).reshape(-1, 2)

    between = []
    if len(eligible) < 2 or len(within) == 0:
        return within, np.zeros((0, 2), dtype=int), {"n_peptides_eligible": len(eligible)}

    target = len(within)
    attempts = 0
    while len(between) < target and attempts < target * 80:
        attempts += 1
        p1, p2 = rng.choice(eligible, size=2, replace=False)
        i = int(rng.choice(pep_to_idx[p1]))
        j = int(rng.choice(pep_to_idx[p2]))
        between.append((i, j))

    info = {
        "n_peptides_eligible": len(eligible),
        "pairs_per_peptide": pairs_per_peptide,
        "n_within": int(len(within)),
        "n_between": int(len(between)),
        "n_attempts_between": attempts,
    }
    return within, np.asarray(between, dtype=int).reshape(-1, 2), info


def assert_distinct_tcr_pairs(tcrs: np.ndarray, pairs: np.ndarray, label: str) -> None:
    if len(pairs) == 0:
        return
    same = tcrs[pairs[:, 0]] == tcrs[pairs[:, 1]]
    n_same = int(np.sum(same))
    if n_same:
        raise RuntimeError(f"{label}: {n_same} sampled pairs have identical TCR sequences")


def pair_l2(Z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.asarray([], dtype=float)
    return np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1)


def lift_pair_indices(subset_idx: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Map (n, 2) indices from a subset array back to full-row indices."""
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.column_stack([subset_idx[pairs[:, 0]], subset_idx[pairs[:, 1]]])


def auroc_from_pairs(within: np.ndarray, between: np.ndarray) -> float:
    if len(within) == 0 or len(between) == 0:
        return float("nan")
    labels = np.concatenate([np.ones(len(within)), np.zeros(len(between))])
    scores = -np.concatenate([within, between])  # smaller distance => higher score
    return safe_auroc(labels, scores)


def eval_pair_table(
    meta: pd.DataFrame,
    pairs: np.ndarray,
    same_peptide: int,
) -> pd.DataFrame:
    pair_id = meta["pair_id"].astype(str).to_numpy()
    pep = meta["pep"].astype(str).to_numpy()
    tcr = meta["tcr"].astype(str).to_numpy()
    rows = []
    for a, b in pairs:
        rows.append(
            {
                "pair_id_a": pair_id[a],
                "pair_id_b": pair_id[b],
                "tcr_a": tcr[a],
                "tcr_b": tcr[b],
                "pep_a": pep[a],
                "pep_b": pep[b],
                "same_peptide": int(same_peptide),
            }
        )
    return pd.DataFrame(rows)


def run_same_peptide_recovery(
    split: str,
    seeds: Sequence[int],
    min_group: int,
    pairs_per_peptide: int,
    eval_seed: int,
    pair_out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Score every representation on one fixed peptide-balanced pair set."""
    meta0, _ = load_seed_split(split, seeds[0])
    meta0 = meta0.reset_index(drop=True)
    pair_ids0 = meta0["pair_id"].astype(str).to_numpy()

    pos_idx = np.where(meta0["label"].astype(int).to_numpy() == 1)[0]
    pos_peps = meta0.loc[pos_idx, "pep"].to_numpy(str)
    pos_tcrs = meta0.loc[pos_idx, "tcr"].to_numpy(str)

    pep_rows = pd.DataFrame({"pep": pos_peps, "tcr": pos_tcrs})
    pep_counts = (
        pep_rows.groupby("pep")
        .agg(n_rows=("tcr", "size"), n_unique_tcr=("tcr", "nunique"))
        .reset_index()
    )
    dup_peps = pep_counts[pep_counts["n_rows"] != pep_counts["n_unique_tcr"]]
    n_eligible_unique = int(
        (
            (pep_counts["n_unique_tcr"] >= min_group)
            & (pep_counts["n_unique_tcr"] * (pep_counts["n_unique_tcr"] - 1) // 2 >= pairs_per_peptide)
        ).sum()
    )
    print(
        f"TCR-sequence audit ({split} positives): {len(pep_counts)} peptides, "
        f"{int((pep_counts['n_rows'] != pep_counts['n_unique_tcr']).sum())} with "
        f"within-peptide duplicate TCRs, {n_eligible_unique} peptides eligible by unique TCR.",
        flush=True,
    )
    if len(dup_peps):
        print(dup_peps.to_string(index=False), flush=True)

    within_pos, between_pos, info = sample_peptide_balanced_pair_indices(
        pos_peps,
        pos_tcrs,
        eval_seed,
        pairs_per_peptide=pairs_per_peptide,
        min_tcrs=min_group,
    )
    assert_distinct_tcr_pairs(pos_tcrs, within_pos, "same-peptide pairs")
    assert_distinct_tcr_pairs(pos_tcrs, between_pos, "different-peptide pairs")
    within_b = lift_pair_indices(pos_idx, within_pos)
    between_b = lift_pair_indices(pos_idx, between_pos)
    tcrs_full = meta0["tcr"].to_numpy(str)
    assert_distinct_tcr_pairs(tcrs_full, within_b, "lifted same-peptide pairs")

    if pair_out_path is not None:
        pairs_df = pd.concat(
            [
                eval_pair_table(meta0, within_b, 1),
                eval_pair_table(meta0, between_b, 0),
            ],
            ignore_index=True,
        )
        pairs_df.insert(0, "eval_seed", eval_seed)
        ident = int((pairs_df["tcr_a"] == pairs_df["tcr_b"]).sum())
        if ident:
            raise RuntimeError(f"eval pair table contains {ident} identical-TCR pairs")
        pairs_df.to_csv(pair_out_path, index=False)
        print(f"Wrote {pair_out_path} ({len(pairs_df)} pairs)", flush=True)

    rows = []
    seen_deterministic: Dict[str, float] = {}
    for seed in seeds:
        meta, models = load_seed_split(split, seed)
        meta = meta.reset_index(drop=True)
        pids = meta["pair_id"].astype(str).to_numpy()
        if len(pids) != len(pair_ids0) or not np.all(pids == pair_ids0):
            raise RuntimeError(f"pair_id order mismatch for {split} seed {seed}")

        for model, (zT, _zPH) in models.items():
            auc_b = auroc_from_pairs(pair_l2(zT, within_b), pair_l2(zT, between_b))
            is_det = model in DETERMINISTIC_MODELS
            if is_det:
                prev = seen_deterministic.get(model)
                if prev is not None:
                    if not np.isclose(prev, auc_b):
                        raise RuntimeError(
                            f"Deterministic {model} AUROC changed across training seeds "
                            f"(seed {seed} vs first evaluation)"
                        )
                    continue
                seen_deterministic[model] = auc_b
                seed_out = eval_seed
            else:
                seed_out = seed

            rows.append(
                {
                    "split": split,
                    "seed": seed_out,
                    "eval_seed": eval_seed,
                    "training_seed": np.nan if is_det else seed,
                    "model_name": model,
                    "protocol": "peptide_balanced",
                    "auroc": auc_b,
                    "n_within": int(len(within_b)),
                    "n_between": int(len(between_b)),
                    "n_peptides": int(info.get("n_peptides_eligible", 0)),
                    "n_tcrs": int(len(pos_idx)),
                    "pairs_per_peptide": pairs_per_peptide,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--min-group-size", type=int, default=5)
    p.add_argument("--pairs-per-peptide", type=int, default=10)
    p.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="Pair-sampling seed, independent of model-training seeds.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = mkdir(OUT_DIR)

    auroc_df = run_same_peptide_recovery(
        args.split,
        args.seeds,
        min_group=args.min_group_size,
        pairs_per_peptide=args.pairs_per_peptide,
        eval_seed=args.eval_seed,
        pair_out_path=out_dir / f"{args.split}_peptide_balanced_eval_pairs.csv",
    )
    auroc_df.to_csv(out_dir / f"{args.split}_same_peptide_auroc_protocols_by_seed.csv", index=False)
    summary = (
        auroc_df.groupby(["protocol", "model_name"])["auroc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(out_dir / f"{args.split}_same_peptide_auroc_protocols_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"\nTables: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
