"""Repository-relative paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
CONFIGS = REPO_ROOT / "configs"
OUTPUTS = REPO_ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
EMBEDDINGS = REPO_ROOT / "data" / "embeddings"
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

TRAIN_CSV = DATA_PROCESSED / "train.csv.gz"
VAL_CSV = DATA_PROCESSED / "val.csv.gz"
TEST_CSV = DATA_PROCESSED / "test.csv.gz"
IMMREP_CSV = DATA_PROCESSED / "immrep_test.csv.gz"
