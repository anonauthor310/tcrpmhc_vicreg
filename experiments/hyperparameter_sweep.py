#!/usr/bin/env python3
"""Single-seed 3×3 hyperparameter sweep.

Grid: alpha=beta in {1, 25, 50}, d in {64, 128, 256}, delta=1, seed=31.
Selection used *validation peptide-weighted AUROC only*. Test and IMMREP
numbers in the raw CSV are diagnostic and were not used for selection.

    PYTHONPATH=. python experiments/hyperparameter_sweep.py
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

FAMILIES = ["onehot", "raw_esmc", "lora_esmc"]
ALPHAS = [1, 25, 50]
DS = [64, 128, 256]
DELTA = 1.0
SEED = 31
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


def cell_tag(alpha: int, d: int) -> str:
    return f"a{alpha}_d{d}"


def run_cell(family: str, alpha: int, d: int) -> None:
    tag = cell_tag(alpha, d)
    dest = f"hpo/{RUN_DIRS[family]}/{tag}"
    extra = [
        "--alpha",
        str(alpha),
        "--beta",
        str(alpha),
        "--delta",
        str(DELTA),
        "--d",
        str(d),
        "--seed",
        str(SEED),
        "--run-tag",
        f"hpo_{RUN_DIRS[family]}_{tag}",
        "--checkpoint-root",
        str(REPO / "outputs" / "checkpoints" / dest),
        "--output-root",
        str(REPO / "outputs" / dest),
        "--figure-root",
        str(REPO / "figures" / dest),
    ]
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(PAPER),
        "--config",
        str(FAMILY_YAML[family]),
        *extra,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def collect() -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for alpha in ALPHAS:
            for d in DS:
                tag = cell_tag(alpha, d)
                path = REPO / "outputs" / "hpo" / RUN_DIRS[family] / tag / f"seed_{SEED}" / "summary.json"
                if not path.exists():
                    continue
                summary = json.loads(path.read_text())
                cfg = summary.get("config", {})
                row = {
                    "model": {
                        "onehot": "onehot_vicreg",
                        "raw_esmc": "raw_esmc_vicreg",
                        "lora_esmc": "lora_esmc_vicreg",
                    }[family],
                    "alpha": float(cfg.get("alpha", alpha)),
                    "beta": float(cfg.get("beta", alpha)),
                    "delta": float(cfg.get("delta", DELTA)),
                    "d": int(cfg.get("d", d)),
                    "seed": int(summary.get("seed", SEED)),
                    "best_epoch": summary.get("best_epoch"),
                    "cell": tag,
                }
                metrics = summary.get("metrics", {})
                for split, models in metrics.items():
                    vicreg_key = "onehot_vicreg" if family == "onehot" else "esm_vicreg"
                    m = models.get(vicreg_key, {})
                    for metric_name, value in m.items():
                        row[f"{split}_{metric_name}"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=FAMILIES, default=None)
    p.add_argument("--collect-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    families = [args.family] if args.family else FAMILIES
    if not args.collect_only:
        for family in families:
            for alpha in ALPHAS:
                for d in DS:
                    run_cell(family, alpha, d)
    df = collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "hyperparameter_sweep.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  n={len(df)}", flush=True)
    print(
        "Selection used validation peptide-weighted AUROC only; "
        "test/IMMREP columns are diagnostic.",
        flush=True,
    )


if __name__ == "__main__":
    main()
