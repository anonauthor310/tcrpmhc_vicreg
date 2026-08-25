# Non-contrastive TCR–pMHC representation learning

Anonymous code release for the accompanying NeurIPS workshop submission.

This repository contains the data-processing, VICReg training, evaluation and
analysis code used in the paper. The three input families are one-hot, frozen
ESMC-300M and LoRA-adapted ESMC-300M.

The paper experiments use positive-only VICReg training, occurrence-matched
internal evaluation and the IMMREP25 external benchmark.

| Family | Input | Batch size |
|---|---|---|
| One-hot | 22-symbol amino-acid one-hot | 64 |
| Raw ESMC | frozen `esmc_300m` per-residue embeddings | 8 |
| LoRA ESMC | LoRA-adapted `esmc_300m` per-residue embeddings | 8 |

Training uses the 24,456 observed binding triples and no negatives. The loss is
VICReg on the TCR and pMHC views (`src/models/vicreg.py`). The evaluation score
is negative mean squared error, `s(T, P) = -(1/d) * ||z_T - z_P||_2^2`.
Validation/test decoys keep each target pMHC fixed and reallocate positive TCR
occurrences one-to-one (`src/data/build_eval_decoys.py`).

## Selected configuration

Shared operating point, chosen on validation peptide-weighted AUROC. Encoded in
`configs/paper.yaml`; each family file adds batch size and input paths.

| Setting | Value |
|---|---|
| Latent width `d` | 256 |
| pMHC split (peptide / MHC) | 179 / 77 (`R_PH = 0.7`) |
| Low-rank ranks | `r_L = 8`, `r_D = 16` |
| VICReg weights (inv, var, cov) | 25, 25, 1 |
| Optimiser | AdamW, lr 3e-4, weight decay 0.01 |
| Epochs | max 30, min 10, patience 10 |
| Seeds | 31, 37, 43, 49, 55 |
| Batch size | 64 (one-hot), 8 (ESMC) |
| Selection metric | validation peptide-weighted AUROC |

## Quick reproduction

```bash
conda env create -f environment.yml
conda activate tcrpmhc-vicreg

# One-hot needs no embedding export.
PYTHONPATH=. python experiments/run_five_seed.py --family onehot

# Raw ESMC: export frozen esmc_300m shards, then train.
PYTHONPATH=. python -m src.representations.export_raw_esmc
PYTHONPATH=. python experiments/run_five_seed.py --family raw_esmc

# LoRA ESMC: train/export LoRA adapters first (not the raw shards above).
# See data/README.md and src/representations/export_lora_esmc.py.
PYTHONPATH=. python -m src.representations.export_lora_esmc
PYTHONPATH=. python experiments/run_five_seed.py --family lora_esmc

# All three families, five seeds each (15 runs):
PYTHONPATH=. python experiments/run_five_seed.py
```

A single run:

```bash
PYTHONPATH=. python -m src.train \
  --config configs/paper.yaml --config configs/onehot.yaml --seed 31
```

`--dry-run` prints the resolved trainer invocation without training.

Five-seed means on the occurrence-matched internal test set (seeds 31/37/43/49/55):

| Model | Internal global AUROC | IMMREP peptide-macro AUROC |
|---|---|---|
| One-hot + VICReg | ≈ 0.72 | ≈ 0.50 |
| Raw ESMC + VICReg | ≈ 0.79 | ≈ 0.51 |
| LoRA ESMC + VICReg | ≈ 0.79 | ≈ 0.51 |
| Fixed input representations (no VICReg) | ≈ 0.50 | ≈ 0.50 |

Checked-in values are in `results/main_results.csv`.
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) maps each table and figure to a command.

## Environment

Runs used a single NVIDIA RTX 4070 Ti SUPER (16 GB): Python 3.10.17, PyTorch
2.9.0+cu128, CUDA 12.8, NumPy 1.26.4, pandas 2.3.3, SciPy 1.13.1,
scikit-learn 1.7.2, matplotlib 3.9.1, `esm` 3.2.1.post1, `peft` 0.17.1.

`environment.yml` builds a conda environment; `requirements.txt` lists direct
dependencies; `requirements-lock.txt` pins the set used for the reported numbers.

## Data

Original third-party database exports and model weights are not redistributed.
The frozen processed splits contain source-derived sequence fields required to
reproduce the reported analyses and remain subject to the terms of the
underlying resources; see THIRD_PARTY.md. `data/README.md` gives releases and
download steps for IEDB/VDJdb positives, IMMREP25, ESMC-300M weights, and
IPD-IMGT/HLA 3.60.0 `hla_prot.fasta`.

MHC sequences come from IPD-IMGT/HLA (records named as HLA alleles). The
manuscript says "MHC"; code columns such as `HLA_sequence` mean the same
sequence.

Frozen splits under `data/processed/` recover the paper counts:

| Split | Rows | Positives | Negatives |
|---|---|---|---|
| train | 24,456 | 24,456 | 0 |
| validation | 3,342 | 1,672 | 1,670 |
| internal test | 3,800 | 1,900 | 1,900 |
| IMMREP25 test | 10,000 | 1,000 | 9,000 |

`src/data/` documents the protocol. See `data/README.md` for why both the
generator and the frozen CSVs are included.

## Layout

```text
configs/              paper.yaml + one file per input family
src/models/           projection_heads.py, vicreg.py, scoring.py
src/train.py          config-driven training entry point
src/evaluate.py       AUROC / peptide-weighted AUROC / McClish pAUC
src/data/             splits, occurrence-matched decoys
src/trainers/         per-family training loops
experiments/          five-seed run, sweep, controls, Figure 2, diagnostics
results/              CSV outputs behind the paper tables
scripts/make_table1.py
figures/make_figure2.py
```

## Licences

MIT ([LICENSE](LICENSE)) covers this repository's code, not third-party data or
model weights. See [THIRD_PARTY.md](THIRD_PARTY.md).
