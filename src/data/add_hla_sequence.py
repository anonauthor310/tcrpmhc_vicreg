#!/usr/bin/env python3
"""Map HLA allele names to IMGT protein sequences.

The FASTA is not shipped. Download hla_prot.fasta from IPD-IMGT/HLA
(ANHIG/IMGTHLA Latest) and pass --hla-fasta. See the repo README.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, RAW, clean_seq, read_csv, require_columns, write_csv, write_json

ALLELE_IN_HEADER = re.compile(r"([A-Z]+\d*\*[0-9:]+[A-Z0-9]*)")


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = None
    chunks: List[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            records.append((header, "".join(chunks)))
    return records


def is_null_allele(header: str) -> bool:
    # IMGT null alleles end in N (e.g. A*01:01:01:02N).
    match = ALLELE_IN_HEADER.search(header.replace("HLA-", ""))
    if match is None:
        return " N " in f" {header} "
    return match.group(1).rstrip().endswith("N")


def two_field_allele(allele: str) -> str:
    text = allele.strip().upper().replace("HLA-", "")
    text = text.split()[0].split(",")[0]
    parts = text.split("*")
    if len(parts) != 2:
        return text
    fields = parts[1].split(":")
    if len(fields) >= 2:
        return f"{parts[0]}*{fields[0]}:{fields[1]}"
    return text


def build_first_hit_index(records: List[Tuple[str, str]]) -> Dict[str, str]:
    """First non-null FASTA record whose header contains the two-field allele."""
    index: Dict[str, str] = {}
    for header, seq in records:
        if is_null_allele(header):
            continue
        match = ALLELE_IN_HEADER.search(header.replace("HLA-", ""))
        if match is None:
            continue
        key = two_field_allele(match.group(1))
        index.setdefault(key, clean_seq(seq))
    return index


def lookup_sequence(allele: str, index: Dict[str, str], records: List[Tuple[str, str]]) -> Optional[str]:
    key = two_field_allele(allele)
    if key in index:
        return index[key]
    needle = key
    for header, seq in records:
        if is_null_allele(header):
            continue
        if needle in header.replace("HLA-", ""):
            return clean_seq(seq)
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--positives", type=Path, default=PROCESSED / "01_positives_concat.csv")
    p.add_argument("--hla-fasta", type=Path, default=RAW / "hla_prot.fasta")
    p.add_argument("--out", type=Path, default=PROCESSED / "02_positives_with_hla.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hla_fasta.exists():
        raise FileNotFoundError(
            f"IMGT protein FASTA not found: {args.hla_fasta}\n"
            "Download hla_prot.fasta from "
            "https://github.com/ANHIG/IMGTHLA/blob/Latest/fasta/hla_prot.fasta\n"
            "and save it as data/raw/hla_prot.fasta (this file cannot be redistributed)."
        )
    rows = read_csv(args.positives)
    require_columns(rows, ["Peptide", "HLA"], args.positives.name)
    records = parse_fasta(args.hla_fasta)
    index = build_first_hit_index(records)

    n_hit = n_miss = 0
    missing: Dict[str, int] = {}
    for row in rows:
        seq = lookup_sequence(row["HLA"], index, records)
        if seq:
            row["HLA_sequence"] = seq
            n_hit += 1
        else:
            row["HLA_sequence"] = ""
            n_miss += 1
            missing[row["HLA"]] = missing.get(row["HLA"], 0) + 1

    write_csv(args.out, rows)
    write_json(
        args.out.with_suffix(".audit.json"),
        {
            "fasta": str(args.hla_fasta),
            "n_fasta_records": len(records),
            "n_indexed_two_field_alleles": len(index),
            "n_rows": len(rows),
            "n_mapped": n_hit,
            "n_unmapped": n_miss,
            "unmapped_alleles": missing,
        },
    )
    print(f"Wrote {args.out}  mapped={n_hit}  unmapped={n_miss}")
    if missing:
        print("Unmapped alleles:")
        for allele, count in sorted(missing.items(), key=lambda x: -x[1]):
            print(f"  {allele}: {count}")


if __name__ == "__main__":
    main()
