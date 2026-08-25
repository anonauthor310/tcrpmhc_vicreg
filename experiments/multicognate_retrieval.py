#!/usr/bin/env python3
"""Figure 2d: multi-cognate TCR -> peptide-MHC retrieval.

Queries are the 77 unique TCRs with at least two recorded cognate pMHCs.
The gallery is the 345 unique peptide-MHC complexes in the internal test
split. Rank by TCR-pMHC Euclidean distance in latent space. The reported
metric is Recall@10. Analytic random ranking is 10/345 = 2.9%.
95% CIs use 2,000 bootstrap resamples over query TCRs.

TCR identifiers use SHA-256 (deterministic), not Python's salted hash().

    PYTHONPATH=. python experiments/multicognate_retrieval.py --split test
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry_common import (  # noqa: E402
    OUT_DIR,
    SEEDS,
    load_seed_split,
    mkdir,
)


def tcr_key(tcr: str) -> str:
    return hashlib.sha256(str(tcr).encode("utf-8")).hexdigest()[:16]


def zt_consistency_audit(meta: pd.DataFrame, models: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    pos = meta[meta["label"].astype(int) == 1]
    rows = []
    for model, (zT, _) in models.items():
        max_diff = 0.0
        n_multi = 0
        for tcr, grp in pos.groupby("tcr"):
            if not tcr or len(grp) < 2:
                continue
            n_multi += 1
            vecs = zT[grp.index.to_numpy()]
            for a in range(len(vecs)):
                for b in range(a + 1, len(vecs)):
                    max_diff = max(max_diff, float(np.linalg.norm(vecs[a] - vecs[b])))
        rows.append(
            {
                "model_name": model,
                "n_tcrs_with_multiple_rows": n_multi,
                "max_l2_within_tcr": max_diff,
                "approx_zero": bool(max_diff < 1e-5),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(vals, size=len(vals), replace=True).mean()) for _ in range(n_boot)]
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return mean, lo, hi


def multicognate_retrieval(
    split: str,
    seeds: Sequence[int],
    n_boot: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    per_tcr_rows = []
    summary_rows = []
    ks = [5, 10, 20]

    for seed in seeds:
        meta, models = load_seed_split(split, seed)
        meta = meta.reset_index(drop=True)
        audit = zt_consistency_audit(meta, models)
        audit["split"] = split
        audit["seed"] = seed
        audit_rows.append(audit)

        pos = meta[meta["label"].astype(int) == 1].copy()
        pmhc_to_idx = {
            pmhc: grp.index.to_numpy()
            for pmhc, grp in pos.groupby("pmhc")
            if pmhc and not pmhc.startswith("|")
        }
        gallery_pmhcs = sorted(pmhc_to_idx.keys())
        tcr_to_pmhc = pos.groupby("tcr")["pmhc"].apply(
            lambda s: set(x for x in s if x and not str(x).startswith("|"))
        )
        multi = {t: set(p) for t, p in tcr_to_pmhc.items() if t and len(p) >= 2}

        for model, (zT, zPH) in models.items():
            gallery = np.asarray([zPH[pmhc_to_idx[p]].mean(axis=0) for p in gallery_pmhcs], float)
            n_gal = len(gallery_pmhcs)
            pmhc_to_gal = {p: j for j, p in enumerate(gallery_pmhcs)}

            query_metrics = []
            for tcr, cognates in multi.items():
                relevant = [p for p in cognates if p in pmhc_to_gal]
                if not relevant:
                    continue
                q_rows = pos.index[pos["tcr"] == tcr].to_numpy()
                q = zT[q_rows[0]]
                if len(q_rows) > 1:
                    q = zT[q_rows].mean(axis=0)

                dists = np.linalg.norm(gallery - q[None, :], axis=1)
                order = np.argsort(dists)
                ranks = {gallery_pmhcs[j]: r + 1 for r, j in enumerate(order)}
                rel_ranks = sorted(ranks[p] for p in relevant)

                row = {
                    "split": split,
                    "seed": seed,
                    "model_name": model,
                    "tcr_id": tcr_key(tcr),
                    "n_cognates": len(relevant),
                    "n_gallery": n_gal,
                    "best_rank": int(rel_ranks[0]),
                }
                for k in ks:
                    row[f"recall@{k}"] = float(np.mean([1.0 if ranks[p] <= k else 0.0 for p in relevant]))
                    row[f"random_recall@{k}"] = float(k / n_gal) if n_gal else float("nan")
                per_tcr_rows.append(row)
                query_metrics.append(row)

            if not query_metrics:
                continue
            qdf = pd.DataFrame(query_metrics)
            summary = {
                "split": split,
                "seed": seed,
                "model_name": model,
                "n_multi_tcrs": len(multi),
                "n_queries": len(qdf),
                "n_gallery_pmhc": n_gal,
            }
            for metric in ["recall@5", "recall@10", "recall@20",
                           "random_recall@5", "random_recall@10", "random_recall@20"]:
                mean, lo, hi = bootstrap_ci(qdf[metric].to_numpy(float), n_boot=n_boot, seed=seed + 17)
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_ci_lo"] = lo
                summary[f"{metric}_ci_hi"] = hi
            summary_rows.append(summary)

    audit_df = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    per_tcr_df = pd.DataFrame(per_tcr_rows)
    summary_df = pd.DataFrame(summary_rows)
    return audit_df, per_tcr_df, summary_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = mkdir(OUT_DIR)
    audit_df, per_tcr_df, summary_df = multicognate_retrieval(
        args.split,
        args.seeds,
        n_boot=args.n_bootstrap,
    )
    audit_df.to_csv(out_dir / f"{args.split}_zt_within_tcr_audit.csv", index=False)
    per_tcr_df.to_csv(
        out_dir / f"{args.split}_multicognate_retrieval_per_tcr_hardened.csv", index=False
    )
    summary_df.to_csv(
        out_dir / f"{args.split}_multicognate_retrieval_summary_hardened.csv", index=False
    )
    cols = [
        "model_name",
        "seed",
        "n_queries",
        "n_gallery_pmhc",
        "recall@10_mean",
        "recall@10_ci_lo",
        "recall@10_ci_hi",
        "random_recall@10_mean",
    ]
    print(summary_df[cols].sort_values(["model_name", "seed"]).to_string(index=False))
    n_gal = int(summary_df["n_gallery_pmhc"].iloc[0]) if len(summary_df) else 345
    print(f"\nanalytic random Recall@10 = 10/{n_gal} = {10 / n_gal:.4f}")
    print(f"Tables: {out_dir}")


if __name__ == "__main__":
    main()
