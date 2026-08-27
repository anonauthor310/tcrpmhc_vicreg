#!/usr/bin/env python3
"""Assemble Table 6 from committed stagewise CSVs. No GPU, no retraining.

Table 6 reports TCR geometry at input / pre-expander / final latent, comparing
IMMREP with the internal test set:

  * MSE ratio: IMMREP / test median pairwise MSE (unnormalised)
  * cosine Δ: IMMREP − test median pairwise cosine distance
  * norm ratio: IMMREP / test mean latent norm
  * covariance ratio: IMMREP / test covariance-trace
  * Test δ / IMMREP δ: Cliff's δ on PN vs PP pairs (cosine class separation)

Sources (written by the original stagewise diagnostic; committed here):

  results/paper_analysis/immrep_transfer_stage_diagnostic/
    localisation_table.csv                  cosine Δ and Cliff's δ
    stagewise_unnormalised_mse_summary.csv  MSE / norm / covariance ratios

``results/stagewise_transfer.csv`` is the geometry-summary dump and does not
contain these columns.

    PYTHONPATH=. python scripts/make_table6.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import RESULTS  # noqa: E402

STAGE_DIR = RESULTS / "paper_analysis" / "immrep_transfer_stage_diagnostic"
MODEL_ORDER = ["onehot_vicreg", "raw_esmc_vicreg", "lora_esmc_vicreg"]
STAGE_ORDER = ["input", "pre_expander", "final_latent"]
STAGE_LABELS = {
    "input": "Input",
    "pre_expander": "Pre-exp.",
    "final_latent": "Final",
}
LABELS = {
    "onehot_vicreg": "One-hot + VICReg",
    "raw_esmc_vicreg": "Raw ESMC + VICReg",
    "lora_esmc_vicreg": "LoRA ESMC + VICReg",
}


def mse_ratio(mse: pd.DataFrame, model: str, stage: str, metric: str) -> float:
    sub = mse[
        (mse["model"] == model)
        & (mse["stage"] == stage)
        & (mse["metric"] == metric)
        & (mse["side"] == "tcr")
        & mse["immrep_div_test"].notna()
    ]
    if sub.empty:
        return float("nan")
    return float(sub["immrep_div_test"].iloc[0])


def loc_value(loc: pd.DataFrame, model: str, stage: str, metric: str, column: str) -> float:
    sub = loc[
        (loc["model"] == model)
        & (loc["stage"] == stage)
        & (loc["side"] == "tcr")
        & (loc["metric"] == metric)
    ]
    if sub.empty:
        return float("nan")
    return float(sub[column].iloc[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--localisation", type=Path, default=STAGE_DIR / "localisation_table.csv")
    p.add_argument(
        "--mse-summary",
        type=Path,
        default=STAGE_DIR / "stagewise_unnormalised_mse_summary.csv",
    )
    p.add_argument("--out", type=Path, default=RESULTS / "table6.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loc = pd.read_csv(args.localisation)
    mse = pd.read_csv(args.mse_summary)
    rows = []
    for model in MODEL_ORDER:
        for stage in STAGE_ORDER:
            rows.append(
                {
                    "model": model,
                    "model_label": LABELS[model],
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "mse_ratio": mse_ratio(mse, model, stage, "median_pairwise_mse"),
                    "cosine_delta": loc_value(
                        loc, model, stage, "median_pairwise_cosine", "immrep_minus_test"
                    ),
                    "norm_ratio": mse_ratio(mse, model, stage, "mean_latent_norm"),
                    "covariance_ratio": mse_ratio(mse, model, stage, "covariance_trace"),
                    "test_cliffs_delta": loc_value(
                        loc, model, stage, "cliffs_delta_pn_vs_pp", "test"
                    ),
                    "immrep_cliffs_delta": loc_value(
                        loc, model, stage, "cliffs_delta_pn_vs_pp", "immrep"
                    ),
                }
            )
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    show = table.copy()
    for col in [
        "mse_ratio",
        "cosine_delta",
        "norm_ratio",
        "covariance_ratio",
        "test_cliffs_delta",
        "immrep_cliffs_delta",
    ]:
        show[col] = show[col].map(lambda v: f"{v:.3f}")
    print(
        show[
            [
                "model_label",
                "stage_label",
                "mse_ratio",
                "cosine_delta",
                "norm_ratio",
                "covariance_ratio",
                "test_cliffs_delta",
                "immrep_cliffs_delta",
            ]
        ].to_string(index=False)
    )
    print(f"\nwrote {args.out}")
    print("MSE / norm / covariance ratios: IMMREP divided by internal test.")
    print("cosine Δ: IMMREP minus internal test. Cliff's δ is PN vs PP.")


if __name__ == "__main__":
    main()
