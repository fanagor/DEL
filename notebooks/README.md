# Exploring the Notebooks

To see how a controller acts on a single input, open the [controller walkthrough](controller_walkthrough.ipynb). You can inspect an expert’s state count, measure the length of a policy-trajectory observation, and follow the head across the grid. The walkthrough runs with built-in experts by default. If you enable a local animation, interrupt the viewer cell when you want to continue.

For the paper experiments, install DEL with `pip install -e '.[analysis,dev]'` from the repository root and select the same Python environment in Jupyter. The eight notebooks below read the released runs and analyses.

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

## Rebuild a Figure

The arithmetic, universal, and front-page notebooks read each selected run’s final `evaluation.json`. Start with the default selections to reconstruct the published plots. The training-dynamics notebook loads one million updates of loss history and ten checkpoint evaluations for each binary task and controller family. The notebook exports four two-panel figures pairing loss with ULG or RLG. Edit `RUNS` to choose your own histories, or set a model entry to `None` to omit the corresponding curve.

## Inspect or Repeat an Extraction

Both method notebooks open a saved case with `RECOMPUTE = False`. For metric extraction, indices 0 through 4 select binary addition, binary multiplication, decimal addition, decimal multiplication, and universal simulation, respectively. The universal case uses a DFST trained on 75 demonstrations for 500,000 updates. Annealing has a separate index for each of three cases. The [case catalog](../paper/analyses.yaml) records both sets of indices.

To analyze another checkpoint, set `RECOMPUTE = True`, choose a starting recipe with `CASE_INDEX`, and set `CHECKPOINT_PATH` in Section 1 to a released or intermediate checkpoint. Review the method parameters and final PTO checks in the same cell. The loader displays the saved step and configuration. Run the remaining cells to extract a controller and compare sampled executions with the expert. Metric extraction keeps the weights fixed. Annealing continues training before the discrete readout. After changing checkpoints, restart the kernel and run the cells again.

For metric extraction, enable `TEACHING_SET` to append the complete teaching core after the random or custom sample. `MAX_LENGTH` bounds the random inputs. Teaching examples retain the prescribed lengths. When restoring a recipe, use `CASE_CONFIG.get("random_samples", CASE_CONFIG["samples"])` for `N_TRAJECTORIES` and `CASE_CONFIG.get("teaching_set", False)` for the teaching-set switch. These settings count the random sample and the teaching core separately.

The default output includes the configuration, final controller maps, plots, measurements, and checks. To save additional diagnostic files under `intermediates/`, enable `SAVE_INTERMEDIATES`. Both output modes support the annealing probability slider. The [analysis guide](../analyses/README.md) explains how to inspect a saved case.

## Check the Notebooks

Run the following command from the repository root to validate and execute temporary copies of the eight paper workflows. Omit both `--paper` and `--execute` to validate the structure and code of all nine notebooks, including the walkthrough.

```sh
python scripts/check_notebooks.py --paper --execute --timeout 180
```

## Figure Outputs

By default, the figure notebooks save PDFs, SVGs, PNGs, and supporting tables under
`analysis_outputs/paper_figures/`. The subdirectories group results by model and figure family.

| Folder | Contents |
|---|---|
| `DFST/<task>/` | Generalization plots for add2, add10, mult2, mult10 or universal, plus universal run and metric tables |
| `DFST/arithmetic/` | Combined arithmetic plots and the arithmetic metric table |
| `DFST_Softmax/<task>/` | Generalization plots for add2, mult2 or universal, plus universal run and metric tables |
| `DFST_Softmax/arithmetic/` | Arithmetic metric table |
| `front_page_summary/` | Summary figure and selected-run table |
| `training_dynamics/` | Per-task ULG and RLG figures, curve tables and settings for add2 and mult2 |

Set `DEL_REPO_ROOT` when running notebooks outside the checkout. Set `DEL_ANALYSIS_DIR` to replace the figure root. Every figure notebook appends
the model/task or figure-family subfolders shown above. Relative roots resolve
from the repository root. Repeated exports replace matching filenames.
The controller walkthrough displays examples and animations inline.
Extraction and annealing use separate case directories under `analysis_outputs/`.
For those two notebooks, `DEL_ANALYSIS_DIR` specifies the complete case directory.
