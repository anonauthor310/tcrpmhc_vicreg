#!/usr/bin/env python3
"""Occurrence-matched evaluation decoys.

For every validation/test positive:
  * the target pMHC (peptide + MHC sequence) is kept fixed;
  * positive TCR occurrences are reassigned one-to-one;
  * same-peptide donors are excluded;
  * recorded positive triples are excluded.

TCR and pMHC occurrence marginals therefore match before any later
completeness filtering. Train stays positives-only.
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, read_csv, write_csv, write_json

Row = Dict[str, str]
PairKey = Tuple[str, str, str]
PMHCKey = Tuple[str, str]


def full_pair(row: Row) -> PairKey:
    return row["TCR_full"], row["Peptide"], row["HLA_sequence"]


def pmhc(row: Row) -> PMHCKey:
    return row["Peptide"], row["HLA_sequence"]


def proposed_pair(donor: Row, target: Row) -> PairKey:
    return donor["TCR_full"], target["Peptide"], target["HLA_sequence"]


def ensure_pair_ids(rows: List[Row], prefix: str) -> None:
    for i, row in enumerate(rows):
        row["pair_id"] = f"{prefix}_{i:06d}"
        row["binding_flag"] = "1"


def match_occurrences(
    positives: List[Row],
    forbidden: Set[PairKey],
    seed: int,
    max_attempts: int = 20,
) -> Dict[int, int]:
    rng = random.Random(seed)
    n = len(positives)
    targets = list(range(n))
    donors = list(range(n))

    def valid(target_idx: int, donor_idx: int, extra_forbidden: collections.Counter) -> bool:
        target, donor = positives[target_idx], positives[donor_idx]
        if donor["Peptide"] == target["Peptide"]:
            return False
        key = proposed_pair(donor, target)
        if key in forbidden or extra_forbidden[key] > 0:
            return False
        return True

    for _ in range(max_attempts):
        t_order = targets[:]
        d_order = donors[:]
        rng.shuffle(t_order)
        rng.shuffle(d_order)
        rows_i, cols_j = [], []
        for i, target_idx in enumerate(t_order):
            for j, donor_idx in enumerate(d_order):
                if valid(target_idx, donor_idx, collections.Counter()):
                    rows_i.append(i)
                    cols_j.append(j)
        adjacency = csr_matrix(
            (np.ones(len(rows_i), dtype=np.int8), (rows_i, cols_j)),
            shape=(n, n),
        )
        match = maximum_bipartite_matching(adjacency, perm_type="column")
        if len(match) != n or np.any(match < 0):
            continue
        assignment = {t_order[i]: d_order[int(match[i])] for i in range(n)}
        if len(set(assignment.values())) != n:
            continue

        key_counts: collections.Counter = collections.Counter()
        for ti, di in assignment.items():
            key_counts[proposed_pair(positives[di], positives[ti])] += 1

        def duplicate_targets() -> List[int]:
            seen: collections.Counter = collections.Counter()
            bad = []
            for ti in assignment:
                key = proposed_pair(positives[assignment[ti]], positives[ti])
                seen[key] += 1
                if seen[key] > 1:
                    bad.append(ti)
            return bad

        def can_swap(i: int, j: int) -> bool:
            if i == j:
                return False
            di, dj = assignment[i], assignment[j]
            if not valid(i, dj, collections.Counter()) or not valid(j, di, collections.Counter()):
                return False
            old_i = proposed_pair(positives[di], positives[i])
            old_j = proposed_pair(positives[dj], positives[j])
            new_i = proposed_pair(positives[dj], positives[i])
            new_j = proposed_pair(positives[di], positives[j])
            if new_i == new_j:
                return False
            rem_i = key_counts[new_i] - int(new_i == old_i) - int(new_i == old_j)
            rem_j = key_counts[new_j] - int(new_j == old_i) - int(new_j == old_j)
            return rem_i == 0 and rem_j == 0

        def do_swap(i: int, j: int) -> None:
            old_i = proposed_pair(positives[assignment[i]], positives[i])
            old_j = proposed_pair(positives[assignment[j]], positives[j])
            key_counts[old_i] -= 1
            key_counts[old_j] -= 1
            assignment[i], assignment[j] = assignment[j], assignment[i]
            key_counts[proposed_pair(positives[assignment[i]], positives[i])] += 1
            key_counts[proposed_pair(positives[assignment[j]], positives[j])] += 1

        ok = True
        for _round in range(20):
            bad = duplicate_targets()
            if not bad:
                break
            rng.shuffle(bad)
            for target_idx in bad:
                others = list(assignment.keys())
                rng.shuffle(others)
                repaired = False
                for other in others[: min(len(others), 5000)]:
                    if can_swap(target_idx, other):
                        do_swap(target_idx, other)
                        repaired = True
                        break
                if not repaired:
                    ok = False
                    break
            if not ok:
                break
        if ok and not duplicate_targets():
            return assignment

    raise RuntimeError("Could not find a valid perfect occurrence matching")


def build_decoys(
    split: str,
    positives: List[Row],
    forbidden: Set[PairKey],
    seed: int,
) -> Tuple[List[Row], dict]:
    assignment = match_occurrences(positives, forbidden, seed)
    decoys: List[Row] = []
    for target_idx, target in enumerate(positives):
        donor = positives[assignment[target_idx]]
        decoys.append(
            {
                "pair_id": f"{split}_occurrence_matched_decoy_{target_idx:06d}",
                "binding_flag": "0",
                "Peptide": target["Peptide"],
                "HLA": target.get("HLA", ""),
                "HLA_sequence": target["HLA_sequence"],
                "TCRa": donor.get("TCRa", ""),
                "TCRb": donor.get("TCRb", ""),
                "TCR_full": donor["TCR_full"],
                "decoy_type": "occurrence_matched_same_pmhc_donor_tcr",
                "source_pair_id": target["pair_id"],
                "donor_pair_id": donor["pair_id"],
                "donor_peptide": donor["Peptide"],
                "novelty_category": target.get("novelty_category", ""),
                "pep_len": target.get("pep_len", str(len(target["Peptide"]))),
                "hla_len": target.get("hla_len", str(len(target["HLA_sequence"]))),
                "tcra_len": donor.get("tcra_len", str(len(donor.get("TCRa", "")))),
                "tcrb_len": donor.get("tcrb_len", str(len(donor.get("TCRb", "")))),
            }
        )
    keys = [full_pair(r) for r in decoys]
    audit = {
        "split": split,
        "n_positives": len(positives),
        "n_decoys": len(decoys),
        "tcr_marginal_exact": collections.Counter(r["TCR_full"] for r in positives)
        == collections.Counter(r["TCR_full"] for r in decoys),
        "pmhc_marginal_exact": collections.Counter(pmhc(r) for r in positives)
        == collections.Counter(pmhc(r) for r in decoys),
        "known_positive_collisions": sum(k in forbidden for k in keys),
        "duplicate_negative_pairs": len(keys) - len(set(keys)),
        "same_donor_target_peptide": sum(r["donor_peptide"] == r["Peptide"] for r in decoys),
    }
    return decoys, audit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=PROCESSED / "04_train_positives.csv")
    p.add_argument("--val", type=Path, default=PROCESSED / "04_val_positives.csv")
    p.add_argument("--test", type=Path, default=PROCESSED / "04_test_positives.csv")
    p.add_argument("--out-dir", type=Path, default=PROCESSED)
    p.add_argument("--val-seed", type=int, default=31)
    p.add_argument("--test-seed", type=int, default=37)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train = read_csv(args.train)
    val = read_csv(args.val)
    test = read_csv(args.test)
    ensure_pair_ids(train, "train_pos")
    ensure_pair_ids(val, "val_pos")
    ensure_pair_ids(test, "test_pos")

    known_pos = {full_pair(r) for rows in (train, val, test) for r in rows}
    val_decoys, val_audit = build_decoys("val", val, known_pos, args.val_seed)
    val_keys = {full_pair(r) for r in val + val_decoys}
    test_decoys, test_audit = build_decoys("test", test, known_pos | val_keys, args.test_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "05_train_positives.csv", train)
    write_csv(args.out_dir / "05_val_positives_and_decoys.csv", val + val_decoys)
    write_csv(args.out_dir / "05_test_positives_and_decoys.csv", test + test_decoys)
    write_json(
        args.out_dir / "05_decoy_audit.json",
        {"val": val_audit, "test": test_audit, "seeds": {"val": args.val_seed, "test": args.test_seed}},
    )
    print("val", val_audit)
    print("test", test_audit)


if __name__ == "__main__":
    main()
