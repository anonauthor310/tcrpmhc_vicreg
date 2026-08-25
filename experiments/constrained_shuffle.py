#!/usr/bin/env python3
"""Constrained pairing-shuffle control (Table 4).

Jointly reassigns peptide+MHC (the pMHC unit) across the 24,456 training TCR
rows, preserves the exact pMHC multiset, and forbids original cognates and
other recorded training-positive triples. One independent reassignment is
drawn per seed. Validation, test and IMMREP stay unshuffled.

All five shuffled runs select epoch 1. ``best_epoch`` is written into the
result CSV.

    PYTHONPATH=. python experiments/constrained_shuffle.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import RESULTS  # noqa: E402

SEEDS = [31, 37, 43, 49, 55]
PAPER = REPO / "configs" / "paper.yaml"
ONEHOT = REPO / "configs" / "onehot.yaml"
RUN = "onehot_vicreg_shuffle_pmhc"


def run_seed(seed: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(PAPER),
        "--config",
        str(ONEHOT),
        "--seed",
        str(seed),
        "--run-tag",
        RUN,
        "--checkpoint-root",
        str(REPO / "outputs" / "checkpoints" / RUN),
        "--output-root",
        str(REPO / "outputs" / RUN),
        "--figure-root",
        str(REPO / "figures" / RUN),
        "--shuffle-train-pmhc",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def collect() -> pd.DataFrame:
    rows = []
    root = REPO / "outputs" / RUN
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        for split, models in summary.get("metrics", {}).items():
            for model_name, metrics in models.items():
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "model": model_name,
                        "best_epoch": summary.get("best_epoch"),
                        "global_auroc": metrics.get("global_auroc"),
                        "peptide_weighted_auroc": metrics.get("peptide_weighted_auroc"),
                        "peptide_macro_auroc": metrics.get("peptide_macro_auroc"),
                        "auprc": metrics.get("auprc"),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--collect-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.collect_only:
        for seed in args.seeds:
            run_seed(seed)
    df = collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "shuffle_control.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)
    shuffled = df[(df["split"] == "test") & (df["model"] == "onehot_vicreg")]
    if len(shuffled):
        print(
            "test onehot_vicreg best_epoch values:",
            sorted(shuffled["best_epoch"].unique().tolist()),
            flush=True,
        )


if __name__ == "__main__":
    main()
