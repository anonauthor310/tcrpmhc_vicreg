#!/usr/bin/env python3
"""Run the whole data-construction chain in order.

    PYTHONPATH=. python src/data/prepare_data.py --hla-fasta data/raw/hla_prot.fasta

Stages (each is also runnable on its own):

    ingest_positives     concatenate IEDB + VDJdb positives
    add_hla_sequence     attach the MHC protein sequence per two-field allele
    filter_and_dedup     complete alpha+beta, peptide and MHC required;
                         drop duplicate (TCR, peptide, MHC) triples
    split_positives      seed-42 split + novelty-regime assignment; the 49
                         high-frequency peptides are protected from
                         unseen-peptide regimes
    build_eval_decoys    occurrence-matched validation/test decoys
    write_paper_csvs     final CSVs + SHA256 checksums

Important: IEDB and VDJdb are living databases, so running this against a newer
snapshot will not reproduce the paper's row counts or pair_id strings. The
frozen paper splits in ``data/processed/*.csv.gz`` are the artefact that
recovers 24,456 / 1,672 + 1,670 / 1,900 + 1,900 exactly. See data/README.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

STAGES = [
    "ingest_positives.py",
    "add_hla_sequence.py",
    "filter_and_dedup.py",
    "split_positives.py",
    "build_eval_decoys.py",
    "write_paper_csvs.py",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hla-fasta",
        type=Path,
        default=REPO / "data" / "raw" / "hla_prot.fasta",
        help="IPD-IMGT/HLA hla_prot.fasta (release 3.60.0 for the paper). Not redistributed.",
    )
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-decoy-seed", type=int, default=31)
    p.add_argument("--test-decoy-seed", type=int, default=37)
    p.add_argument("--start-at", choices=STAGES, default=STAGES[0])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hla_fasta.exists():
        raise SystemExit(
            f"Missing {args.hla_fasta}. Download hla_prot.fasta from IPD-IMGT/HLA "
            "(see data/README.md); it is CC BY-ND and is not redistributed here."
        )
    extra = {
        "add_hla_sequence.py": ["--hla-fasta", str(args.hla_fasta)],
        "split_positives.py": ["--seed", str(args.split_seed)],
        "build_eval_decoys.py": [
            "--val-seed",
            str(args.val_decoy_seed),
            "--test-seed",
            str(args.test_decoy_seed),
        ],
    }
    for stage in STAGES[STAGES.index(args.start_at) :]:
        cmd = [sys.executable, str(HERE / stage), *extra.get(stage, [])]
        print("==", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
    print("prepare_data: done", flush=True)


if __name__ == "__main__":
    main()
