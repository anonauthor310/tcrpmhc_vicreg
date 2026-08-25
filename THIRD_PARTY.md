# Third-party resources

The MIT licence in `LICENSE` covers this repository's code only. None of the
resources below are redistributed here; each must be obtained from its own
source under its own terms. Versions/releases listed are the ones used for the
reported results.

## ESMC-300M (protein language model)

- Resource: `esmc_300m` checkpoint, loaded through the `esm` Python package
  (`ESMC.from_pretrained("esmc_300m")`).
- Version used: `esm==3.2.1.post1`.
- Source: <https://github.com/evolutionaryscale/esm>
- Terms: EvolutionaryScale Cambrian Open licence / non-commercial community
  terms as published with the model. Check the current licence and the model
  card before use; weights are gated and must be accepted at download time.
- Use here: frozen per-residue embeddings (raw family) and LoRA-adapted
  per-residue embeddings (LoRA family). LoRA adapters are trained locally with
  `peft` and are not redistributed.

## IPD-IMGT/HLA (MHC protein sequences)

- Resource: `hla_prot.fasta`, combined HLA protein sequences.
- Release used: **IPD-IMGT/HLA 3.60.0**.
- Source: <https://github.com/ANHIG/IMGTHLA> (`fasta/hla_prot.fasta`; the release
  string is in `release_version.txt` in the same repository). Project home:
  <https://www.ebi.ac.uk/ipd/imgt/hla/>
- Terms: Creative Commons Attribution-NoDerivs (CC BY-ND). See
  <https://www.ebi.ac.uk/ipd/imgt/hla/licence/>. The NoDerivs term is why this
  repository ships **no** FASTA file and no derived alignment; you download it
  yourself.
- Use here: mapping each two-field allele name (e.g. `HLA-A*02:01`) to a protein
  sequence. The manuscript refers to these as MHC sequences.

## IEDB (Immune Epitope Database)

- Resource: curated TCR–epitope binding records used as positive interactions.
- Source: <https://www.iedb.org/>
- Terms: IEDB is free to use with attribution; see
  <https://www.iedb.org/citing_iedb_v3> and the site terms of use. Please cite
  IEDB if you rebuild the dataset.
- Use here: positive TCR–pMHC triples (paired α/β chains, peptide, allele).

## VDJdb

- Resource: curated TCR–epitope specificity records used as positive
  interactions.
- Source: <https://vdjdb.cdr3.net/> and <https://github.com/antigenomics/vdjdb-db>
- Terms: CC BY 4.0 as stated by the VDJdb project; cite the VDJdb paper.
- Use here: positive TCR–pMHC triples, concatenated with the IEDB-derived table.

## IMMREP25 benchmark

- Resource: IMMREP 2025 TCR-specificity prediction benchmark test set
  (1,000 positives and 9,000 negatives over 20 peptides).
- Source: IMMREP workshop benchmark release; see
  <https://github.com/viragbioinfo/IMMREP_2023_TCRSpecificity> for the workshop
  series and follow the current-year release linked from it.
- Terms: as published by the benchmark organisers. Used unmodified as an
  external held-out evaluation set.
- Use here: external transfer evaluation only. No IMMREP data is used for
  training, validation or model selection.

## Python dependencies

Direct dependencies and their versions are listed in `requirements.txt` and
pinned in `requirements-lock.txt`. Each retains its own upstream licence
(PyTorch: BSD-3-Clause; NumPy, pandas, SciPy, scikit-learn, matplotlib: BSD-3;
`peft`, `transformers`: Apache-2.0).
