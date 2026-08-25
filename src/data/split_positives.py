#!/usr/bin/env python3
"""Category-aware train/val/test split of complete-chain positives.

Ported from the workshop split protocol:
  unseen HLA (rare alleles, backbone HLA stay in train),
  completely unseen, unseen TCR, unseen peptide.
Cross-reactive peptides in PEPTIDES_TO_KEEP are not placed in unseen-peptide
regimes. Train is then stripped of every unseen TCR / peptide / HLA entity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, write_json

PEPTIDES_TO_KEEP = [
    "KLGGALQAK", "GILGFVFTL", "AVFDRKSDAK", "RAKFKQLL", "SPRWYFYYL",
    "YLQPRTFLL", "TTDPSFLGRY", "GLCTLVAML", "RVRAYTYSK", "IVTDFSVIK",
    "LLWNGPMAV", "LLLDRLNQL", "NLVPMVATV", "LLAGIGTVPI", "RLRAEAQVK",
    "ELAGIGILTV", "YVLDHLIVV", "LTDEMIAQY", "CINGVCWTV", "TPRVTGGGAM",
    "VMATRRNVL", "KTFPPTEPK", "QYIKWPWYI", "DATYQRTRALVR", "NQKLIANQF",
    "FLRGRAYGL", "CTELKLSDY", "ATDALMTGF", "RPPIFIRRL", "NYNYLYRLF",
    "FLYALALLL", "VMTTVLATL", "CLGGLLTMV", "KSKRTPMGF", "RPHERNGFTVL",
    "MEVTPSGTWL", "FTSDYYQLY", "RPIIRPATL", "ALAGIGILTV", "LLYDANYFL",
    "HPVTKYIM", "RLPGVLPRA", "RFPLTFGWCF", "VYFLQSINF", "PTDNYITTY",
    "ALWEIQQVV", "QAKWRLQTL", "RTATKQYNV", "LLFGYPVYV",
]

BACKBONE_HLAS = {
    "HLA-A*02:01",
    "HLA-A*01:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*08:01",
    "HLA-A*03:01",
    "HLA-A*11:01",
}


def make_unseen_tcr_peptide_categories(
    df_in: pd.DataFrame,
    peptides_to_keep: Iterable[str],
    rng: np.random.Generator,
    pct_tcr_cat1: float = 0.02,
    pct_tcr_cat2: float = 0.05,
    pct_peptide_cat3: float = 0.01,
    category_col: str = "category",
):
    df = df_in.copy()
    unique_tcrs = df["TCR_full"].unique()
    if len(unique_tcrs) == 0:
        empty = df.iloc[0:0].copy()
        return empty, empty, empty, empty

    n_cat1_tcrs = max(1, int(len(unique_tcrs) * pct_tcr_cat1))
    selected_tcrs_cat1 = set(rng.choice(unique_tcrs, size=n_cat1_tcrs, replace=False))
    tcr_pairs = df[df["TCR_full"].isin(selected_tcrs_cat1)]
    tcr_pairs = tcr_pairs[~tcr_pairs["Peptide"].isin(peptides_to_keep)]
    selected_peptides_cat1 = set(tcr_pairs["Peptide"].unique())
    selected_tcrs_cat1 = set(tcr_pairs["TCR_full"].unique())
    cat1_df = df[
        df["TCR_full"].isin(selected_tcrs_cat1) & df["Peptide"].isin(selected_peptides_cat1)
    ].copy()
    cat1_df[category_col] = "completely_unseen"

    train_candidate = df[
        ~(df["TCR_full"].isin(selected_tcrs_cat1) | df["Peptide"].isin(selected_peptides_cat1))
    ].copy()

    remaining_unique_tcrs = train_candidate["TCR_full"].unique()
    if len(remaining_unique_tcrs) > 0:
        n_cat2_tcrs = max(1, int(len(remaining_unique_tcrs) * pct_tcr_cat2))
        selected_tcrs_cat2 = set(rng.choice(remaining_unique_tcrs, size=n_cat2_tcrs, replace=False))
    else:
        selected_tcrs_cat2 = set()
    cat2_df = train_candidate[train_candidate["TCR_full"].isin(selected_tcrs_cat2)].copy()
    cat2_df[category_col] = "unseen_TCR"
    train_candidate = train_candidate.drop(cat2_df.index)

    remaining_unique_peptides = train_candidate["Peptide"].unique()
    if len(remaining_unique_peptides) > 0:
        n_cat3_peptides = max(1, int(len(remaining_unique_peptides) * pct_peptide_cat3))
        selected_peptides_cat3 = set(
            rng.choice(remaining_unique_peptides, size=n_cat3_peptides, replace=False)
        ) - set(peptides_to_keep)
    else:
        selected_peptides_cat3 = set()
    cat3_df = train_candidate[train_candidate["Peptide"].isin(selected_peptides_cat3)].copy()
    cat3_df[category_col] = "unseen_peptide"
    remaining_df = train_candidate.drop(cat3_df.index)
    return cat1_df, cat2_df, cat3_df, remaining_df


def choose_unseen_hlas(
    table: pd.DataFrame,
    rng: np.random.Generator,
    min_peptides: int,
    n_choose: int,
    exclude: Set[str],
) -> Set[str]:
    candidates = set(
        table.loc[
            (table["Peptide"] >= min_peptides) & (~table["HLA"].isin(exclude)),
            "HLA",
        ]
    )
    if not candidates:
        return set()
    n = min(n_choose, len(candidates))
    return set(rng.choice(list(candidates), size=n, replace=False))


def split_tcr_dataset(
    df: pd.DataFrame,
    rng: np.random.Generator,
    peptides_to_keep=PEPTIDES_TO_KEEP,
    backbone_hlas=BACKBONE_HLAS,
    n_test_unseen_hla: int = 2,
    n_val_unseen_hla: int = 2,
    min_peptides_unseen_hla_test: int = 10,
    min_peptides_unseen_hla_val: int = 10,
):
    df = df.copy()
    df["_split_key"] = (
        df["TCR_full"].astype(str) + "||" + df["Peptide"].astype(str) + "||" + df["HLA"].astype(str)
    )
    hla_table = (
        df.groupby("HLA")
        .agg(TCR_full=("TCR_full", "nunique"), Peptide=("Peptide", "nunique"))
        .reset_index()
        .sort_values("Peptide", ascending=False)
    )
    test_unseen_hlas = choose_unseen_hlas(
        hla_table, rng, min_peptides_unseen_hla_test, n_test_unseen_hla, set(backbone_hlas)
    )
    cat0_test = df[df["HLA"].isin(test_unseen_hlas)].copy()
    cat0_test["novelty_category"] = "unseen_HLA"
    pool = df[~df["HLA"].isin(test_unseen_hlas)].copy()
    cat1_test, cat2_test, cat3_test, _ = make_unseen_tcr_peptide_categories(
        pool, peptides_to_keep, rng, category_col="novelty_category"
    )
    test_df = pd.concat([cat0_test, cat1_test, cat2_test, cat3_test]).drop_duplicates()
    test_keys = set(test_df["_split_key"])
    dev_pool = df[~df["_split_key"].isin(test_keys)].copy()

    test_hlas = set(test_df["HLA"])
    hla_table_dev = (
        dev_pool.groupby("HLA")
        .agg(TCR_full=("TCR_full", "nunique"), Peptide=("Peptide", "nunique"))
        .reset_index()
    )
    val_unseen_hlas = choose_unseen_hlas(
        hla_table_dev,
        rng,
        min_peptides_unseen_hla_val,
        n_val_unseen_hla,
        set(backbone_hlas) | test_hlas,
    )
    cat0_val = dev_pool[dev_pool["HLA"].isin(val_unseen_hlas)].copy()
    cat0_val["novelty_category"] = "unseen_HLA"
    pool_val = dev_pool[~dev_pool["HLA"].isin(val_unseen_hlas)].copy()
    cat1_val, cat2_val, cat3_val, _ = make_unseen_tcr_peptide_categories(
        pool_val, peptides_to_keep, rng, category_col="novelty_category"
    )
    val_df = pd.concat([cat0_val, cat1_val, cat2_val, cat3_val]).drop_duplicates()
    val_keys = set(val_df["_split_key"])

    forbidden_tcrs = (
        set(cat1_test["TCR_full"])
        | set(cat2_test["TCR_full"])
        | set(cat1_val["TCR_full"])
        | set(cat2_val["TCR_full"])
    )
    forbidden_peps = (
        set(cat1_test["Peptide"])
        | set(cat3_test["Peptide"])
        | set(cat1_val["Peptide"])
        | set(cat3_val["Peptide"])
    )
    forbidden_hlas = test_unseen_hlas | val_unseen_hlas
    train_df = df[
        (~df["_split_key"].isin(test_keys))
        & (~df["_split_key"].isin(val_keys))
        & (~df["TCR_full"].isin(forbidden_tcrs))
        & (~df["Peptide"].isin(forbidden_peps))
        & (~df["HLA"].isin(forbidden_hlas))
    ].copy()
    train_df["novelty_category"] = "train"

    for frame in (train_df, val_df, test_df):
        if "_split_key" in frame.columns:
            frame.drop(columns=["_split_key"], inplace=True)

    meta = {
        "test_unseen_hlas": sorted(test_unseen_hlas),
        "val_unseen_hlas": sorted(val_unseen_hlas),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }
    return train_df, val_df, test_df, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-csv", type=Path, default=PROCESSED / "03_positives_complete_dedup.csv")
    p.add_argument("--out-dir", type=Path, default=PROCESSED)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.in_csv)
    rng = np.random.default_rng(args.seed)
    train_df, val_df, test_df, meta = split_tcr_dataset(df, rng)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.out_dir / "04_train_positives.csv", index=False)
    val_df.to_csv(args.out_dir / "04_val_positives.csv", index=False)
    test_df.to_csv(args.out_dir / "04_test_positives.csv", index=False)
    write_json(args.out_dir / "04_split_audit.json", {"seed": args.seed, **meta})
    print(json.dumps({"seed": args.seed, **meta}, indent=2))


if __name__ == "__main__":
    main()
