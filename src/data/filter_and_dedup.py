#!/usr/bin/env python3
"""Keep complete αβ pairs with peptide + HLA sequence; drop duplicate molecules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, clean_seq, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-csv", type=Path, default=PROCESSED / "02_positives_with_hla.csv")
    p.add_argument("--out", type=Path, default=PROCESSED / "03_positives_complete_dedup.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.in_csv)
    kept = []
    seen = set()
    n_incomplete = n_dup = 0
    for row in rows:
        peptide = clean_seq(row.get("Peptide", ""))
        hla_seq = clean_seq(row.get("HLA_sequence", ""))
        tcra = clean_seq(row.get("TCRa", ""))
        tcrb = clean_seq(row.get("TCRb", ""))
        if not (peptide and hla_seq and tcra and tcrb):
            n_incomplete += 1
            continue
        tcr_full = tcra + tcrb
        key = (tcr_full, peptide, hla_seq)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        out = dict(row)
        out.update(
            {
                "Peptide": peptide,
                "HLA_sequence": hla_seq,
                "TCRa": tcra,
                "TCRb": tcrb,
                "TCR_full": tcr_full,
                "binding_flag": "1",
                "pep_len": str(len(peptide)),
                "hla_len": str(len(hla_seq)),
                "tcra_len": str(len(tcra)),
                "tcrb_len": str(len(tcrb)),
            }
        )
        kept.append(out)

    write_csv(args.out, kept)
    write_json(
        args.out.with_suffix(".audit.json"),
        {
            "input_rows": len(rows),
            "kept": len(kept),
            "dropped_incomplete": n_incomplete,
            "dropped_duplicate_molecule": n_dup,
        },
    )
    print(
        f"Wrote {args.out}  kept={len(kept)}  "
        f"incomplete={n_incomplete}  duplicate={n_dup}"
    )


if __name__ == "__main__":
    main()
