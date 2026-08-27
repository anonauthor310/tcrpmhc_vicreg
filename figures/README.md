# Figures

| File | Paper item | Produced by |
|---|---|---|
| `test_geometry_multipanel.png` | Figure 2 | `PYTHONPATH=. python figures/make_figure2.py --split test` |

The stagewise diagnostic figure is a PDF written by
`experiments/stagewise_transfer.py` to
`results/paper_analysis/immrep_transfer_stage_diagnostic/stagewise_diagnostic_figure.pdf`
(and the same name under `figures/paper_analysis/immrep_transfer_stage_diagnostic/`
when that script is rerun). It is not `figures/stagewise_diagnostic.png`. Table 6
is assembled from committed CSVs by `PYTHONPATH=. python scripts/make_table6.py`.

Figure 1 is a hand-drawn schematic of the architecture and is not part of this
repository.

`make_figure2.py` reads saved result CSVs only, so it needs no GPU and no
retraining. Panel (a) is a schematic drawn in matplotlib inside that script;
panels (b), (c) and (d) come from `results/paper_analysis/refined_geometry/` and
`results/paper_analysis/crossreactivity/`.
