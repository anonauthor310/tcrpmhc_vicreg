#!/usr/bin/env python3
"""Render Table 1 from checked-in result artefacts. No GPU, no retraining.

Reads ``results/main_results.csv`` (internal test + IMMREP, per model) and
``results/novelty_regimes.csv`` (novelty-regime rows), and prints the table as
markdown plus writes ``results/table1.csv``.

    PYTHONPATH=. python scripts/make_table1.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import RESULTS  # noqa: E402

MODEL_ORDER = [
    "onehot_composition",
    "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool",
    "onehot_vicreg",
    "raw_esmc_vicreg",
    "finetuned_esmc_vicreg",
]
LABELS = {
    "onehot_composition": "Input: one-hot",
    "pretrained_esmc_meanpool": "Input: raw ESMC",
    "finetuned_esmc_meanpool": "Input: LoRA ESMC",
    "onehot_vicreg": "One-hot + VICReg",
    "raw_esmc_vicreg": "Raw ESMC + VICReg",
    "finetuned_esmc_vicreg": "LoRA ESMC + VICReg",
}
# Novelty-regime CSV uses the trainer-side model keys.
NOVELTY_ALIAS = {
    "onehot_composition": "onehot_composition",
    "pretrained_esmc_meanpool": "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool": "finetuned_esmc_meanpool",
    "onehot_vicreg": "onehot_vicreg",
    "raw_esmc_vicreg": "esm_vicreg_raw",
    "finetuned_esmc_vicreg": "esm_vicreg_finetuned",
}
REGIMES = [
    ("unseen_TCR", "Unseen TCR"),
    ("unseen_HLA", "Unseen MHC"),
    ("completely_unseen", "Unseen TCR + peptide"),
]


def fmt(mean: float, std: float | None, deterministic: bool) -> str:
    if pd.isna(mean):
        return "--"
    if deterministic or std is None or pd.isna(std) or round(float(std), 2) < 0.01:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def main_rows(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, metrics in [
        ("test", [("global_auroc", "Internal global AUROC"),
                  ("peptide_weighted_auroc", "Internal peptide-weighted AUROC")]),
        ("immrep_test", [("peptide_macro_auroc", "IMMREP peptide-macro AUROC"),
                         ("global_auc0.1_mcclish", "IMMREP McClish pAUC@0.1")]),
    ]:
        for col, label in metrics:
            row = {"metric": label}
            for model in MODEL_ORDER:
                sub = main[(main["split"] == split) & (main["model_name"] == model)]
                if sub.empty:
                    row[LABELS[model]] = "--"
                    continue
                rec = sub.iloc[0]
                row[LABELS[model]] = fmt(
                    rec.get(f"{col}_mean"),
                    rec.get(f"{col}_std"),
                    bool(rec.get("deterministic_baseline", False)),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def novelty_rows(nov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    nov = nov[(nov["split"] == "test") & (nov["metric"] == "global_auroc")]
    for regime, label in REGIMES:
        sub_regime = nov[nov["category"] == regime]
        if sub_regime.empty:
            continue
        n_pairs = int(sub_regime["n_pairs"].iloc[0])
        row = {"metric": f"{label} (n={n_pairs}) global AUROC"}
        for model in MODEL_ORDER:
            sub = sub_regime[sub_regime["model"] == NOVELTY_ALIAS[model]]
            if sub.empty:
                row[LABELS[model]] = "--"
                continue
            rec = sub.iloc[0]
            deterministic = model in {
                "onehot_composition",
                "pretrained_esmc_meanpool",
                "finetuned_esmc_meanpool",
            }
            row[LABELS[model]] = fmt(rec["mean"], rec.get("std"), deterministic)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main", type=Path, default=RESULTS / "main_results.csv")
    p.add_argument("--novelty", type=Path, default=RESULTS / "novelty_regimes.csv")
    p.add_argument("--out", type=Path, default=RESULTS / "table1.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    main_df = pd.read_csv(args.main)
    parts = [main_rows(main_df)]
    if args.novelty.exists():
        parts.append(novelty_rows(pd.read_csv(args.novelty)))
    table = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {args.out}")
    print("Values are mean ± SD over five seeds (31/37/43/49/55).")
    print("Fixed input representations are deterministic and have no seed SD.")


if __name__ == "__main__":
    main()
