#!/usr/bin/env python3
"""Shared helpers for the data-construction scripts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed"

REQUIRED_POSITIVE_COLS = ("Peptide", "HLA", "TCRa", "TCRb")


def clean_seq(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text in {"NAN", "NONE", "<UNK>", "UNK", "UNKNOWN", "X"}:
        return ""
    return "".join(ch for ch in text if ch.isalpha())


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot write empty CSV without fieldnames: {path}")
    names = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(rows: List[Dict[str, str]], required: Iterable[str], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{name} missing columns {missing}; have {list(rows[0].keys())}")


def molecular_key(row: Dict[str, str]) -> tuple[str, str, str]:
    return row["TCR_full"], row["Peptide"], row.get("HLA_sequence", row.get("HLA", ""))
