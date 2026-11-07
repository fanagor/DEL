# Maintainer Tools

Run these commands from the repository root after installing `.[dev,analysis]`.
The paper notebooks read released artifacts without invoking the import tools.

| Script | Purpose | Inputs and outputs |
|---|---|---|
| `plot_readme_overview.py` | Generate the binary-addition overview | Replays `101+11` with the expert and saved extracted controller, checks PTO agreement, and writes SVG and PNG files under `docs/assets/`. Run with `python scripts/plot_readme_overview.py`. |
| `check_notebooks.py` | Validate notebook structure and Python code; optionally execute temporary copies | `--execute` runs all nine notebooks; `--paper` selects the eight paper workflows. Temporary output directories protect published figures. |
| `plot_front_page.py` | Generate the summary figure from explicit run selections | Supply `--add2`, `--add10`, `--mult2`, `--mult10` and `--universal-tp`. The default prefix uses `analysis_outputs/paper_figures/front_page_summary/front_page`. `--evaluate-missing` explicitly enables checkpoint evaluation. |
| `export_paper_runs.py` | Export selected final configurations, checkpoints and evaluations | Requires a private `--manifest`; writes the selected `runs/` directories and refuses conflicting final weights. |
| `curate_analyses.py` | Import compact extraction or annealing cases | Requires a private `--manifest`; verifies the released source checkpoint and copies machines, figures, measurements, checks and supplied proofs. |

## Import Manifests

Keep tracking identifiers and source locations outside the public repository.
Export manifests contain a `runs` list. Each entry supplies `source`, `path`,
`model`, `task`, `expert_digest`, `wandb_project`, `wandb_id`, and `expected`.
The `expected` mapping specifies `steps` and any selected `warmup_steps`,
`hidden_size`, `data_size` or `teaching_set` constraints.

Analysis manifests contain an `analyses` list. Each entry supplies `source`,
`path`, `source_run`, `method` and `title`. Paths resolve from the repository
root. `source_run` points to a released run containing
`final_model_checkpoint.pt`; the source case records the matching checksum.

Use `--help` for each command's options. Review imported cases and update
`paper/analyses.yaml` and `paper/figures.yaml` when adding published results.
