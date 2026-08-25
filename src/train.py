"""YAML-driven training entry point.

    PYTHONPATH=. python -m src.train --config configs/paper.yaml --config configs/onehot.yaml --seed 31

The family config selects the trainer. Shared hyperparameters live in
``configs/paper.yaml`` rather than in argparse defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.paths import REPO_ROOT

FAMILY_TO_MODULE = {
    "onehot": "src.trainers.onehot",
    "raw_esmc": "src.trainers.raw_esmc",
    "lora_esmc": "src.trainers.lora_esmc",
}

KEY_TO_FLAG = {
    "run_tag": "--run-tag",
    "checkpoint_root": "--checkpoint-root",
    "output_root": "--output-root",
    "figure_root": "--figure-root",
    "train_csv": "--train-csv",
    "val_csv": "--val-csv",
    "test_csv": "--test-csv",
    "immrep_csv": "--immrep-csv",
    "missing_chain_policy": "--missing-chain-policy",
    "seed": "--seed",
    "batch_size": "--batch-size",
    "epochs": "--epochs",
    "patience": "--patience",
    "min_epochs": "--min-epochs",
    "rL": "--rL",
    "rD": "--rD",
    "d": "--d",
    "R_PH": "--R-PH",
    "dropout": "--dropout",
    "lr": "--lr",
    "weight_decay": "--weight-decay",
    "alpha": "--alpha",
    "beta": "--beta",
    "delta": "--delta",
    "pretrained_embed_root": "--pretrained-embed-root",
    "pretrained_immrep_shard_dir": "--pretrained-immrep-shard-dir",
    "finetuned_embed_root": "--finetuned-embed-root",
    "finetuned_immrep_shard_dir": "--finetuned-immrep-shard-dir",
}

BOOL_FLAGS = {
    "save_latents": "--save-latents",
    "overwrite": "--overwrite",
    "shuffle_train_pmhc": "--shuffle-train-pmhc",
}


def load_merged_config(paths: List[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path in paths:
        with path.open() as handle:
            payload = yaml.safe_load(handle) or {}
        merged.update(payload)
    return merged


def resolve_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((REPO_ROOT / candidate).resolve())


def config_to_argv(cfg: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for key, flag in KEY_TO_FLAG.items():
        if key not in cfg or cfg[key] is None:
            continue
        value = cfg[key]
        if key.endswith("_csv") or key.endswith("_root") or key.endswith("_dir"):
            value = resolve_path(value)
        argv.extend([flag, str(value)])
    for key, flag in BOOL_FLAGS.items():
        if bool(cfg.get(key)):
            argv.append(flag)
    return argv


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        action="append",
        type=Path,
        required=True,
        help="YAML file. Later files override earlier ones. Pass paper.yaml then a family yaml.",
    )
    p.add_argument("--seed", type=int, default=None, help="Override seed from YAML.")
    p.add_argument("--family", choices=sorted(FAMILY_TO_MODULE), default=None)
    p.add_argument("--shuffle-train-pmhc", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved trainer command and exit.",
    )
    args, unknown = p.parse_known_args(argv)
    args.unknown = unknown
    return args


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_merged_config(args.config)
    if args.shuffle_train_pmhc:
        cfg["shuffle_train_pmhc"] = True
    if args.family:
        cfg["family"] = args.family
    if args.seed is not None:
        cfg["seed"] = args.seed
    family = cfg.get("family")
    if family not in FAMILY_TO_MODULE:
        raise SystemExit(f"Config must set family to one of {sorted(FAMILY_TO_MODULE)}; got {family!r}")
    trainer_argv = config_to_argv(cfg) + list(getattr(args, "unknown", []))
    if args.dry_run:
        print(FAMILY_TO_MODULE[family], " ".join(trainer_argv))
        return
    module = __import__(FAMILY_TO_MODULE[family], fromlist=["main"])
    sys.argv = [FAMILY_TO_MODULE[family], *trainer_argv]
    module.main()


if __name__ == "__main__":
    main()
