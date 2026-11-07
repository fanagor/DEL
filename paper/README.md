# Paper Artifacts

[figures.yaml](figures.yaml) selects final-evaluation runs, all eight extracted-controller cases and the two 1M-step training-dynamics comparisons. [analyses.yaml](analyses.yaml) catalogs the five metric extractions and three annealing cases, with notebook indices, released source runs, proof domains, proof documents, checkers and saved reports. A separate `training_dynamics` section catalogs saved loss histories, checkpoint evaluations and measurement settings. Figure selections remain in `figures.yaml`.

All case paths point directly to curated directories under `analyses/metric_extraction/` or `analyses/annealing/`. The `path` field identifies the published artifacts; `source_run` identifies the released neural checkpoint used for extraction. Proof paths resolve relative to the case directory. Proof domains concern extracted discrete controllers, not the continuous neural models. Universal metric extraction covers nonempty inputs; universal annealing also covers empty input.

Run the paper workflows from the repository root:

```sh
python scripts/check_notebooks.py --paper --execute --timeout 180
```

The command uses released files and saved histories without a tracking-service connection. Temporary copies and output directories keep published artifacts unchanged. Extraction and annealing notebooks display the selected saved case by default.

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

The final-results notebooks require evaluations for every selected run. Evaluation ceilings retain the saved `at_cap` flags.

Use the main paper for the central comparisons, extraction mechanism, and precise claims. Include the supporting proof statements and arguments in the appendix or supplement. The repository contains the configurations, machine-readable tables, plots, and executable checkers needed to inspect those claims.

The training-dynamics notebook loads 1M-step add2 and mult2 selections from
`training_dynamics` in [figures.yaml](figures.yaml). Override `RUNS` in the notebook
for a local comparison. The default selections include DFST, RNN and GRU
histories for each task under `analyses/training_dynamics/<model>/<task>/`;
leave `None` to omit a model. Each supplied run requires complete loss history
and ten 100k-step ULG/RLG evaluations. These figures use trajectory evaluations,
while the sweep figures use final evaluations from `runs/`.

Maintainer import tools require explicit manifests outside the public repository.
Use `scripts/export_paper_runs.py --manifest /private/path/selection.yaml` for
training outputs, or `scripts/curate_analyses.py --manifest /private/path/imports.yaml`
for new extraction outputs. Existing published cases require no import step.

Public checkpoints omit tracking identities and disable tracking by default.
Sanitization preserves model tensors, optimizer state and random-generator state.
Evaluation files and analysis configurations record the sanitized checkpoint
checksums. Checkpoint `format_version` fields remain necessary for model loading;
the two paper manifests need no schema-version field.

The method notebooks use `CASE_INDEX` to display one saved case. Compact outputs
contain configuration, final machine maps, plots, diagnostic data and checks.
Set `SAVE_INTERMEDIATES = True` during recomputation to save debugging artifacts
under `intermediates/`. Annealing probability snapshots support the viewer in
both storage modes. See [the analysis layout](../analyses/README.md) for the folder contents.


## Figure Outputs

Figure notebooks save PDFs, SVGs, PNGs and supporting tables under
`analysis_outputs/paper_figures/`:

| Folder | Contents |
|---|---|
| `DFST/<task>/` | Generalization plots for add2, add10, mult2, mult10 or universal; universal run and metric tables |
| `DFST/arithmetic/` | Combined arithmetic plots and the arithmetic metric table |
| `DFST_Softmax/<task>/` | Generalization plots for add2, mult2 or universal; universal run and metric tables |
| `DFST_Softmax/arithmetic/` | Arithmetic metric table |
| `front_page_summary/` | Summary figure and selected-run table |
| `training_dynamics/` | Per-task ULG and RLG figures, curve tables and settings for add2 and mult2 |

Set `DEL_ANALYSIS_DIR` to replace the figure root. Every figure notebook appends
the model/task or figure-family subfolders shown above. Relative roots resolve
from the repository root. Repeated exports replace matching filenames.
The controller walkthrough displays examples and animations inline.
Extraction and annealing use separate case directories under `analysis_outputs/`;
for those two notebooks, `DEL_ANALYSIS_DIR` specifies the complete case directory.
