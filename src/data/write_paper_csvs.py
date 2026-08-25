#!/usr/bin/env python3
"""Write final split CSVs and SHA256 checksums."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, read_csv, sha256_file, write_csv, write_json


FINAL_NAMES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=PROCESSED / "05_train_positives.csv")
    p.add_argument("--val", type=Path, default=PROCESSED / "05_val_positives_and_decoys.csv")
    p.add_argument("--test", type=Path, default=PROCESSED / "05_test_positives_and_decoys.csv")
    p.add_argument("--out-dir", type=Path, default=PROCESSED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
    }
    checksum_lines = []
    summary = {}
    for split, src in mapping.items():
        rows = read_csv(src)
        dest = args.out_dir / FINAL_NAMES[split]
        write_csv(dest, rows)
        digest = sha256_file(dest)
        checksum_lines.append(f"{digest}  {FINAL_NAMES[split]}")
        n_pos = sum(r.get("binding_flag", "1") == "1" for r in rows)
        n_neg = len(rows) - n_pos
        summary[split] = {
            "path": str(dest),
            "n_rows": len(rows),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "sha256": digest,
        }
        print(f"{split}: {dest}  rows={len(rows)}  pos={n_pos}  neg={n_neg}")
        print(f"  sha256 {digest}")

    checksum_path = args.out_dir / "CHECKSUMS.sha256"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    write_json(args.out_dir / "06_final_audit.json", summary)
    print(f"Wrote {checksum_path}")


if __name__ == "__main__":
    main()
