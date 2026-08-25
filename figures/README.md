# Figures

| File | Paper item | Produced by |
|---|---|---|
| `test_geometry_multipanel.png` | Figure 2 | `PYTHONPATH=. python figures/make_figure2.py --split test` |
| `stagewise_diagnostic.png` | Stagewise transfer diagnostic | `PYTHONPATH=. python experiments/stagewise_transfer.py` |

Figure 1 is a hand-drawn schematic of the architecture and is not part of this
repository.

`make_figure2.py` reads saved result CSVs only, so it needs no GPU and no
retraining. Panel (a) is a schematic drawn in matplotlib inside that script;
panels (b), (c) and (d) come from `results/paper_analysis/refined_geometry/` and
`results/paper_analysis/crossreactivity/`.
