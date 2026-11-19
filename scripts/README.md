# Maintainer Tools

Use these tools to check a notebook, regenerate the headline figure, or prepare artifacts for release. For an existing paper figure, the [notebook guide](../notebooks/README.md) takes you directly to the published inputs. Run the commands below from the repository root after installing `.[dev,analysis]`.

| Script | Purpose | Inputs and outputs |
|---|---|---|
| `check_notebooks.py` | Validate notebook structure and Python code and optionally execute temporary copies | `--execute` runs all nine notebooks. Adding `--paper` selects the eight paper workflows. Temporary output directories protect published figures. |
| `plot_front_page.py` | Generate the summary figure from explicit run selections | Supply `--add2`, `--add10`, `--mult2`, `--mult10` and `--universal-tp`. The default prefix uses `analysis_outputs/paper_figures/front_page_summary/front_page`. `--evaluate-missing` explicitly enables checkpoint evaluation. |
| `export_grid_viewer.py` | Export expert programs and inference weights for the web viewer | Pass the destination `del/grid/data` directory. The exporter reads released checkpoints, applies Softmax normalization where appropriate, and writes float32 weights plus a JSON catalog. Add `--parity <file.json>` to export reference executions from the Python viewer. |
| `export_paper_runs.py` | Export selected final configurations, checkpoints and evaluations | Requires a private `--manifest`. Writes the selected `runs/` directories and refuses conflicting final weights. |
| `curate_analyses.py` | Import compact extraction or annealing cases | Requires a private `--manifest`. Verifies the released source checkpoint and copies machines, figures, measurements, checks and supplied proofs. |

## Import Manifests

A private manifest tells the importer which source experiments correspond to the public selections. Keep tracking identifiers and source locations in that manifest.
Export manifests contain a `runs` list. Each entry supplies `source`, `path`,
`model`, `task`, `expert_digest`, `wandb_project`, `wandb_id`, and `expected`.
The `expected` mapping specifies `steps` and any selected `warmup_steps`,
`hidden_size`, `data_size` or `teaching_set` constraints.

Analysis manifests contain an `analyses` list. Each entry supplies `source`,
`path`, `source_run`, `method` and `title`. Paths resolve from the repository
root. `source_run` points to a released run containing
`final_model_checkpoint.pt`. The source case records the matching checksum.

Use `--help` for each command's options. Review imported cases and update
`paper/analyses.yaml` and `paper/figures.yaml` when adding published results.

## Check the GitHub Pages Viewer

From the DEL root, point `viewer` at the Hugo source directory containing the explorer’s `engine.mjs`. These commands regenerate the browser weights and compare the JavaScript simulation with the Python viewer on the released controllers. Node.js 20 or later runs the comparison.

```sh
viewer=../my-website/hugo-site/static/del/grid
python scripts/export_grid_viewer.py "$viewer/data" --parity /tmp/del-grid-parity.json
node tests/check_grid_export.mjs "$viewer" /tmp/del-grid-parity.json
```
