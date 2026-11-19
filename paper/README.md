# Guide to the Paper Artifacts

The [NeurIPS 2026 paper and appendices](https://arxiv.org/abs/2611.21486v1) present the learning framework, the generalization experiments, and the proofs. Use this guide to trace a plotted value back to the run that produced the measurement or a theorem back to the controller and checker that support the claim.

## Find a Figure or Proof

The [figure selections](figures.yaml) identify the final runs, training histories, and extraction cases behind the figures. The [analysis catalog](analyses.yaml) gives the notebook index, source checkpoint, extracted machine, input domain, proof, checker, and saved report for each of the eight certificates. In a catalog entry, `source_run` points to the trained neural controller and `path` points to the extraction case. Proof paths start at the case directory.

Each notebook below addresses a different question. For the headline result, start with the front-page summary. For variation across settings, open the corresponding arithmetic or universal sweep.

| Result | Notebook |
|---|---|
| Arithmetic DFST sweeps | [DFST arithmetic](../notebooks/dfst_arithmetic.ipynb) |
| Arithmetic Softmax sweeps | [Softmax arithmetic](../notebooks/dfst_softmax_arithmetic.ipynb) |
| Six DFST runs for the universal learning task | [DFST universal](../notebooks/dfst_universal.ipynb) |
| Nine DFST_Softmax runs for the universal learning task | [DFST Softmax universal](../notebooks/dfst_softmax_universal.ipynb) |
| Selected arithmetic results and results for the universal learning task | [Front-page summary](../notebooks/front_page_summary.ipynb) |
| Loss and 100k-step ULG/RLG at a 1M-step training budget | [Training dynamics](../notebooks/training_dynamics.ipynb) |
| Metric extraction cases | [Metric extraction](../notebooks/metric_FST_extraction.ipynb) |
| Annealing cases | [Annealing](../notebooks/anneal_dfst_softmax_to_fst.ipynb) |

## Reproduce the Figures

After installing the [notebook dependencies](../README.md#install), open a notebook and run the cells in order. To execute all eight paper workflows on temporary copies, run the following command from the repository root.

```sh
python scripts/check_notebooks.py --paper --execute --timeout 180
```

The figure notebooks read released evaluations, including the `at_cap` flags that identify successful ceiling probes. Each training-dynamics curve draws on a loss log and ten evaluations spaced 100,000 updates apart. To compare another history, change `RUNS` in the notebook and supply the full one-million-update history. Set a selection to `None` to omit a curve.

The extraction notebooks display a saved case by default. Choose `CASE_INDEX` to inspect a different controller. When recomputing a case, enable `SAVE_INTERMEDIATES` to retain diagnostic artifacts alongside final outputs. The [analysis guide](../analyses/README.md) describes the files and proof domains. In particular, the universal metric-extraction proof covers nonempty inputs, and the universal annealing proof also covers the empty input.

For maintainers adding a result, the [release tools](../scripts/README.md#import-manifests) import training outputs and extraction cases from explicit private manifests. Released checkpoints retain model, optimizer, and random-generator state and carry a format version for loading. The export removes tracking identities and records checksums in the evaluation and analysis files.

## Figure Outputs

Browse the [rendered figures and numerical tables](../analysis_outputs/paper_figures/) to inspect a result without starting Jupyter. The [notebook output guide](../notebooks/README.md#figure-outputs) lists the destination for each figure family and explains how to choose a different output directory.
