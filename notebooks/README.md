# Notebooks

Install the package with `pip install -e '.[analysis,dev]'` from the repository root and use that Python environment in Jupyter.

[controller_walkthrough.ipynb](controller_walkthrough.ipynb) introduces the implemented experts with a state-count catalog, single-input PTO measurements and animations. The default walkthrough requires no model weights and complements the eight paper notebooks. Check the animation switches before running those cells; an enabled animation serves the viewer until interrupted.

The directory contains nine notebooks: the controller walkthrough above and these eight paper workflows:

| Notebook | Purpose |
|---|---|
| [dfst_arithmetic.ipynb](dfst_arithmetic.ipynb) | Arithmetic warmup and training-budget sweeps |
| [dfst_softmax_arithmetic.ipynb](dfst_softmax_arithmetic.ipynb) | Arithmetic hidden-dimension and dataset-size sweeps |
| [dfst_universal.ipynb](dfst_universal.ipynb) | Six DFST runs for the universal learning task across dataset sizes and training budgets |
| [dfst_softmax_universal.ipynb](dfst_softmax_universal.ipynb) | Nine DFST_Softmax runs for the universal learning task across hidden dimensions and dataset sizes |
| [front_page_summary.ipynb](front_page_summary.ipynb) | Selected arithmetic results and results for the universal learning task in the front-page plot |
| [training_dynamics.ipynb](training_dynamics.ipynb) | Averaged loss and 100k-step ULG/RLG for DFST, RNN and GRU on add2 and mult2 at 1M steps |
| [metric_FST_extraction.ipynb](metric_FST_extraction.ipynb) | Saved metric-extraction cases and a repeatable procedure |
| [anneal_dfst_softmax_to_fst.ipynb](anneal_dfst_softmax_to_fst.ipynb) | Saved annealing cases, probability-heatmap sliders for add2/mult2/universal, and a repeatable procedure |

The arithmetic, universal and front-page notebooks load final `evaluation.json` files from `runs/`. Each selected result requires a final evaluation. The two method notebooks display the selected curated `analyses/` case by default; set `RECOMPUTE = True` and choose `CASE_INDEX` to repeat a procedure from the recorded settings.

Metric-extraction `CASE_INDEX` follows **0 = add2, 1 = mult2, 2 = add10, 3 = mult10, 4 = universal**. The universal case uses DFST, dataset size 75 and 500k training steps. Annealing uses a separate index sequence.

Both method notebooks expose `CHECKPOINT_PATH` in Section 1. Set `RECOMPUTE = True`, choose a published recipe with `CASE_INDEX`, then replace `CHECKPOINT_PATH` with a released or intermediate checkpoint file. The loader reads the saved step and configuration. Review the method settings and final PTO-test settings in the same cell. Relative paths resolve from the repository root.

The loading cell only loads and displays the checkpoint. Continue through the remaining cells for the full analysis: metric extraction keeps the neural weights fixed; annealing continues training before FST extraction. Both conclude with sampled PTO checks of the extracted controller. Section 1 explains the supported models, and the final section lists the output files. Restart the kernel after changing checkpoints to refresh dependent results.

New outputs go to `analysis_outputs/` at the repository root. The Figure Outputs section lists the export folders and `DEL_ANALYSIS_DIR` behavior. `DEL_REPO_ROOT` selects the checkout when a notebook runs outside the checkout.

`training_dynamics.ipynb` exports four 1 × 2 figures, one per task and metric, using 1M-step histories. The default selections contain DFST, RNN and GRU. Edit paths in `RUNS`; leave `None` to omit a model. Supplied paths require complete histories and ten ULG/RLG evaluations. Legends and exported tables contain only the selected models. Single-input PTO measurement appears in `controller_walkthrough.ipynb`.

Validate the eight paper notebooks and execute temporary copies:

```sh
python scripts/check_notebooks.py --paper --execute --timeout 180
```

The command checks the eight paper notebooks and needs the released runs and analysis cases. Saved notebooks and published figures remain unchanged. To validate every notebook's structure, omit `--paper --execute`.


For metric extraction, `TEACHING_SET = CASE_CONFIG.get("teaching_set", False)`
controls optional teaching inputs. Set `True` to append the complete teaching
core for the loaded checkpoint's expert after the standard or custom sample.
`MAX_LENGTH` bounds random inputs; teaching inputs retain the prescribed lengths.
New experiment metadata records the random count, teaching count and total count.
When restoring a recipe, use `CASE_CONFIG.get("random_samples", CASE_CONFIG["samples"])`
for `N_TRAJECTORIES` to avoid counting teaching inputs twice.

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
