# Data acquisition and preparation

Original third-party database exports and model weights are not included in
this directory. data/processed/ contains the frozen processed splits used for
the paper, including source-derived sequence fields required for reproduction.
These remain subject to the terms described in ../THIRD_PARTY.md.

## Generator scripts and frozen splits

The scripts in `src/data/` document the protocol: how positives are ingested,
how MHC sequences are attached, how duplicate triples are removed, how the
seed-42 split and the novelty regimes are assigned, and how the
occurrence-matched decoys are built. Run them to apply the method to a current
database snapshot.

The CSVs in `data/processed/` are the frozen splits, meaning the exact rows the
reported numbers were computed from, with SHA256 checksums in
`CHECKSUMS.sha256` and counts in `SPLIT_COUNTS.json`.

Both are here because IEDB and VDJdb are living databases. Re-running the
generator against a newer snapshot gives different row counts and different
`pair_id` strings, since new binders have been deposited since we built the
splits. The frozen files are what recover 24,456 train positives, 1,672
validation positives with 1,670 matched negatives, and 1,900 / 1,900 internal
test, and they are what `configs/paper.yaml` points at by default.

## Frozen split counts

| File | Rows | Positives | Negatives |
|---|---|---|---|
| `processed/train.csv.gz` | 24,456 | 24,456 | 0 |
| `processed/val.csv.gz` | 3,342 | 1,672 | 1,670 |
| `processed/test.csv.gz` | 3,800 | 1,900 | 1,900 |
| `processed/immrep_test.csv.gz` | 10,000 | 1,000 | 9,000 |

Validation has 1,670 rather than 1,672 negatives: occurrence matching produces
one decoy per positive, and two decoys are then removed by the complete-chain
filter (both α and β chain required, plus non-empty peptide and MHC sequence).
Occurrence marginals are matched *before* that filter, as described in
`src/data/build_eval_decoys.py`.

Verify integrity with:

```bash
cd data/processed && sha256sum -c CHECKSUMS.sha256
```

Columns: `pair_id`, `Peptide`, `HLA_sequence` (the MHC protein sequence),
`TCR_full` (concatenated α+β), `binding_flag`, and the length fields
`pep_len`/`tcra_len`/`tcrb_len`/`hla_len`. Evaluation splits additionally carry
`decoy_type`, `source_pair_id` (the positive a decoy was derived from),
`donor_pair_id` and `donor_peptide`. Novelty regimes are recomputed from the
train split by `experiments/novelty_regimes.py`, so they do not need to be
stored.

## Required downloads

### 1. IPD-IMGT/HLA protein FASTA (release 3.60.0)

Not redistributable: IPD-IMGT/HLA data are CC BY-ND
([licence](https://www.ebi.ac.uk/ipd/imgt/hla/licence/)).

```bash
mkdir -p data/raw
curl -L -o data/raw/hla_prot.fasta \
  https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/hla_prot.fasta
```

`Latest` tracks the newest release. To pin the release used for the paper,
check out the `3600` tag of `ANHIG/IMGTHLA` (or download that release's
`hla_prot.fasta`) and confirm `release_version.txt` reads 3.60.0. Per-locus
files such as `A_prot.fasta` are not a substitute unless you concatenate every
classical class I locus.

`src/data/add_hla_sequence.py` maps each two-field allele name (e.g.
`HLA-A*02:01`) to the first non-null FASTA record whose header contains that
allele. The manuscript calls these MHC sequences.

### 2. IEDB and VDJdb positives

Export paired-chain TCR–epitope records to `data/raw/iedb_positives.csv` and
`data/raw/vdjdb_positives.csv`. Both must contain the columns `Peptide`, `HLA`,
`TCRa`, `TCRb` (extra columns are ignored). `HLA` holds an allele name such as
`HLA-A*02:01`; `TCRa`/`TCRb` hold the α and β chain sequences.

- IEDB: <https://www.iedb.org/> (TCR search, paired-chain receptor export)
- VDJdb: <https://vdjdb.cdr3.net/> or <https://github.com/antigenomics/vdjdb-db>

### 3. IMMREP25 test benchmark

Download the IMMREP 2025 benchmark test set and convert it to the same column
convention as the processed splits (`pair_id`, `Peptide`, `HLA_sequence`,
`TCR_full`, `binding_flag`, length fields). The frozen copy used for the paper
is `processed/immrep_test.csv.gz`. IMMREP is never used for training,
validation or model selection.

### 4. ESMC-300M weights

Required only for the two ESMC input families. Weights are fetched by the `esm`
package on first use:

```python
from esm.models.esmc import ESMC
model = ESMC.from_pretrained("esmc_300m")
```

Accept the model terms at download time. LoRA adapters for the LoRA family are
trained locally; see `src/representations/export_lora_esmc.py`.

## Running the generator

```bash
# after placing the three raw inputs under data/raw/
PYTHONPATH=. python src/data/ingest_positives.py
PYTHONPATH=. python src/data/add_hla_sequence.py --hla-fasta data/raw/hla_prot.fasta
PYTHONPATH=. python src/data/filter_and_dedup.py
PYTHONPATH=. python src/data/split_positives.py --seed 42
PYTHONPATH=. python src/data/build_eval_decoys.py
PYTHONPATH=. python src/data/write_paper_csvs.py
```

Stage by stage:

| Script | What it does |
|---|---|
| `ingest_positives.py` | concatenate IEDB and VDJdb positives, tag `source_db` |
| `add_hla_sequence.py` | attach the MHC protein sequence per two-field allele |
| `filter_and_dedup.py` | require complete α+β, peptide and MHC sequence; drop duplicate (TCR, peptide, MHC) triples |
| `split_positives.py` | seed-42 split with novelty-regime assignment; protects the 49 high-frequency peptides from unseen-peptide regimes; strips unseen entities from train |
| `build_eval_decoys.py` | occurrence-matched validation/test decoys (train stays positives-only) |
| `write_paper_csvs.py` | final CSVs plus SHA256 checksums |

## Embedding shards

ESMC shards are large and are not committed. Export them locally:

```bash
PYTHONPATH=. python -m src.representations.export_raw_esmc
PYTHONPATH=. python -m src.representations.export_lora_esmc
```

They are written under `data/embeddings/` (gitignored), matching the paths in
`configs/raw_esmc.yaml` and `configs/lora_esmc.yaml`.
