#!/usr/bin/env python3
"""Concatenate IEDB and VDJdb positive tables into one positives file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, REQUIRED_POSITIVE_COLS, PROCESSED, require_columns, write_csv, write_json


def load_source(path: Path, source: str) -> list[dict]:
    from common import read_csv

    rows = read_csv(path)
    require_columns(rows, REQUIRED_POSITIVE_COLS, path.name)
    out = []
    for row in rows:
        item = dict(row)
        item["source_db"] = source
        item["binding_flag"] = "1"
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iedb", type=Path, default=RAW / "iedb_positives.csv")
    p.add_argument("--vdjdb", type=Path, default=RAW / "vdjdb_positives.csv")
    p.add_argument("--out", type=Path, default=PROCESSED / "01_positives_concat.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.iedb.exists() or not args.vdjdb.exists():
        raise FileNotFoundError(
            "Place IEDB and VDJdb positive CSVs in data/raw/ "
            f"(expected {args.iedb} and {args.vdjdb})."
        )
    iedb = load_source(args.iedb, "iedb")
    vdjdb = load_source(args.vdjdb, "vdjdb")
    rows = iedb + vdjdb
    fieldnames = list(dict.fromkeys([*iedb[0].keys(), *vdjdb[0].keys()]))
    write_csv(args.out, rows, fieldnames)
    write_json(
        args.out.with_suffix(".audit.json"),
        {
            "iedb_rows": len(iedb),
            "vdjdb_rows": len(vdjdb),
            "concat_rows": len(rows),
            "iedb_path": str(args.iedb),
            "vdjdb_path": str(args.vdjdb),
        },
    )
    print(f"Wrote {args.out}  n={len(rows)}  (iedb={len(iedb)}, vdjdb={len(vdjdb)})")


if __name__ == "__main__":
    main()
