#!/usr/bin/env python3
"""Run the five-seed paper experiment for one or all model families.

    PYTHONPATH=. python experiments/run_five_seed.py
    PYTHONPATH=. python experiments/run_five_seed.py --family onehot

Writes per-seed trainer outputs under ``outputs/`` and a compact table at
``results/main_results.csv``, which is what Table 1 is built from. Rerunning
overwrites that file.
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
from src.train import FAMILY_TO_MODULE  # noqa: E402

FAMILIES = ["onehot", "raw_esmc", "lora_esmc"]
SEEDS = [31, 37, 43, 49, 55]
PAPER = REPO / "configs" / "paper.yaml"
FAMILY_YAML = {
    "onehot": REPO / "configs" / "onehot.yaml",
    "raw_esmc": REPO / "configs" / "raw_esmc.yaml",
    "lora_esmc": REPO / "configs" / "lora_esmc.yaml",
}
RUN_DIRS = {
    "onehot": "onehot_vicreg_complete",
    "raw_esmc": "esm_vicreg_raw_complete",
    "lora_esmc": "esm_vicreg_finetuned_complete",
}
VICREG_MODEL = {
    "onehot": "onehot_vicreg",
    "raw_esmc": "esm_vicreg",
    "lora_esmc": "esm_vicreg",
}
INPUT_MODEL = {
    "onehot": "onehot_composition",
    "raw_esmc": "pretrained_esmc_meanpool",
    "lora_esmc": "finetuned_esmc_meanpool",
}


def run_one(family: str, seed: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(PAPER),
        "--config",
        str(FAMILY_YAML[family]),
        "--seed",
        str(seed),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def collect() -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        out_root = REPO / "outputs" / RUN_DIRS[family]
        for seed in SEEDS:
            summary_path = out_root / f"seed_{seed}" / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            for split, models in summary.get("metrics", {}).items():
                for model_name, metrics in models.items():
                    rows.append(
                        {
                            "family": family,
                            "run_name": RUN_DIRS[family],
                            "model_name": model_name,
                            "split": split,
                            "seed": seed,
                            "best_epoch": summary.get("best_epoch"),
                            "best_selection_metric": summary.get("best_selection_metric"),
                            "best_selection_value": summary.get("best_selection_value"),
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=FAMILIES, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    families = [args.family] if args.family else FAMILIES
    if not args.collect_only:
        for family in families:
            for seed in args.seeds:
                summary = REPO / "outputs" / RUN_DIRS[family] / f"seed_{seed}" / "summary.json"
                if args.skip_existing and summary.exists():
                    print(f"skip existing {family} seed {seed}", flush=True)
                    continue
                run_one(family, seed)
    df = collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    long_path = RESULTS / "main_results_long.csv"
    df.to_csv(long_path, index=False)
    print(f"wrote {long_path}  n={len(df)}", flush=True)


if __name__ == "__main__":
    main()
