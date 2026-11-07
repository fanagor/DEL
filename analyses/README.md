# Analysis Artifacts

[paper/analyses.yaml](../paper/analyses.yaml) catalogs controller extractions and training histories in separate sections. Extracted-controller paths group cases by method and source run:

```text
analyses/<method>/<model>/<task>/<run_id>/
    README.md
    config.yaml
    machines/       # Extracted and expert JSON, readable CSV, PDF, state maps
    figures/        # Main plots in PDF and PNG
    data/           # Plot measurements, extraction inputs or probability snapshots
    checks/         # Sampled PTO results, counterexamples and coverage checks
    checkpoints/    # Final annealed checkpoint, for annealing only
    proof/          # Self-contained proof, checker, dependencies and saved report
```

`config.yaml` records the source run, checkpoint checksum and method settings.
Metric extraction saves sampled inputs in `data/inputs.json`, stability measurements
in `data/graph_stability.csv` and threshold attempts in `data/epsilon_attempts.json`.
Annealing saves `data/sharpness.csv`, `data/training.jsonl`, probability tensors in
`data/probabilities/` and `checkpoints/annealed_model_checkpoint.pt`.

`machines/extracted.json` preserves the final controller structure. The matching
CSV uses readable symbols and movements; the PDF presents the transition table.
`machines/expert.*` provides the same formats for the expert. State numbering
matches across all formats. `machines/state_map.json` records minimization;
`machines/expert_state_map.json` records expert numbering.

Each `proof/` directory contains an independent mathematical argument, checker,
reference implementations, certificate dependencies and saved report. Keep this
bundle intact: fingerprints bind the checker to the accompanying files.

The notebooks display only the selected `CASE_INDEX`. Saved views require no
trajectory generation or training. `SAVE_INTERMEDIATES = False` produces the
compact layout. Set `True` to additionally save trajectories, prefix graphs,
intermediate checkpoints and probability figures under `intermediates/`.
Published cases contain the compact artifacts and proof bundles.

| Method | Source run | Case |
|---|---|---|
| Metric extraction | DFST add2, warmup 50,000, steps 250,000 | [Binary addition](metric_extraction/DFST/add2/warmup_50000_steps_250000/README.md) |
| Metric extraction | DFST mult2, warmup 25,000, steps 500,000 | [Binary multiplication](metric_extraction/DFST/mult2/warmup_25000_steps_500000/README.md) |
| Metric extraction | DFST add10, warmup 50,000, steps 4,000,000 | [Decimal addition](metric_extraction/DFST/add10/warmup_50000_steps_4000000/README.md) |
| Metric extraction | DFST mult10, warmup 0, steps 4,000,000 | [Decimal multiplication](metric_extraction/DFST/mult10/warmup_0_steps_4000000/README.md) |
| Metric extraction | DFST universal, data 75, steps 500,000 | [Universal Teleporting Agent](metric_extraction/DFST/universal/data_75_steps_500000/README.md) |
| Annealing | Softmax add2, hidden 22, data 75 | [Binary addition](annealing/DFST_Softmax/add2/hidden_22_data_75/README.md) |
| Annealing | Softmax mult2, hidden 64, data 100 | [Binary multiplication](annealing/DFST_Softmax/mult2/hidden_64_data_100/README.md) |
| Annealing | Softmax universal, hidden 30, data 150 | [Universal Teleporting Agent](annealing/DFST_Softmax/universal/hidden_30_data_150/README.md) |

Metric-extraction notebook indices follow **0 = add2, 1 = mult2, 2 = add10, 3 = mult10, 4 = universal**. The universal proof covers correctly encoded binary Turing machines with nonempty binary inputs.

These cases illustrate selected extraction regimes. Each theorem states the valid input domain and concerns the extracted discrete controller. The checker establishes finite local obligations and runs supplementary executions. The mathematical argument supplies protocol coverage and the unbounded claim. Finite tests and graph-stability plots do not establish that claim for the continuous neural model.

Use [metric extraction](../notebooks/metric_FST_extraction.ipynb) or [annealing](../notebooks/anneal_dfst_softmax_to_fst.ipynb) to inspect the saved cases. Set `RECOMPUTE = True` to repeat a selected procedure. New results go to `analysis_outputs/`.

[paper/analyses.yaml](../paper/analyses.yaml) catalogs the curated cases directly. To import a new case, supply a separate import manifest to `python scripts/curate_analyses.py --manifest /private/path/imports.yaml`. Published cases require no access to the source extraction directories.

## Proof Conventions

All eight equivalence proofs use the same notation and document structure:

- Write `\mathcal E` for the expert and `\mathcal C` for the exported controller.
- Use phase index `i`, expert state `g_i`, and controller-state set `Q_i`. Write a permitted edge as `i --x--> j`; the local certificate compares transitions on read symbol `x` and requires the controller successor to belong to `Q_j`.
- Write `(q′, w, M)` for a transition, `lambda` for blank, and `PTO` for the sequence of read/write/motion triples. Count the final stopping action once. For universal simulation, use `kappa` for the ternary code width and include hidden rune symbols in the configuration relation.
- Present the controllers, input domain, execution convention, PTO definition, and theorem first. Follow with expert invariants, protocol coverage, the simulation certificate, the equivalence argument, and reproducibility information. Keep task-specific refinements and domain restrictions explicit.
- Avoid third-person personal and possessive pronouns. Avoid standalone forms of “to be”; auxiliary uses in passive or progressive constructions remain permitted. Prefer active voice, descriptive verbs, concise prose, and American English. Use an en dash rather than an em dash.
- Keep each LaTeX source standalone. Match the common preamble, typography, headings, and terminology across the eight documents. Use section and theorem references instead of fixed page numbers.

Every proof bundle contains `verify_equivalence.py` and `fingerprints.json`.
The checker declares an exact required-artifact list and verifies every listed
SHA-256 checksum before checking the finite certificate. Required artifacts cover
the controller exports, expert tables and reference implementation, protocol or
certificate files, supplementary replay inputs, and the checker itself.
Transition-table PDFs accompany and identify rendered controller tables where
provided. The manifest excludes manuscript LaTeX/PDF files, generated verification
reports, training configurations and experiment metadata.

`--quick` checks artifact integrity and all finite proof obligations. The default
command also performs the case-specific replay checks. Mult10 stores the
extraction replay instances in `replay_inputs.json`; the checksum covers those
inputs without depending on run names or training metadata. Preserve each case's
stated input domain and proof scope when interpreting the results.

Compile each `proof/equivalence_proof.tex` twice with `pdflatex` to resolve cross-references, then inspect the rendered PDF. Run the adjacent `verify_equivalence.py` after revisions. Preserve the controller exports, reference snapshots, finite protocols, and saved verification reports when editing presentation alone; the checkers pin these dependencies by fingerprint.

## Training Dynamics

Each model/task folder contains one selected training history. The configuration
records the seed, training budget, warmup and other settings.
Each history folder supports plotting independently of `runs/`. The notebooks
read the local configuration, loss log and saved evaluations; no released
checkpoint or cross-directory run link participates in loading a history.

`training_dynamics/<model>/<task>/` contains the histories read by
[training_dynamics.ipynb](../notebooks/training_dynamics.ipynb):

```text
config.yaml
metrics.jsonl
experiment.json
evaluations/
    step_0000100000.json
    step_0000200000.json
    ...
```

Each training-history `experiment.json` contains the same four fields:

- `evaluation`: evaluation settings for the recorded checkpoints.
- `evaluation_every`: checkpoint interval in training steps.
- `evaluation_steps`: scheduled checkpoint steps; each completed history contains
  the corresponding files under `evaluations/`.
- `metrics_sha256`: SHA-256 checksum of the adjacent `metrics.jsonl` loss history.

The checksum detects changes to the exact log bytes, including formatting.
Run folders identify the model, task and selected training regime. Each evaluation
contains `evaluation_config`, `protocol`, `random_samples`, `scan`, `seed`,
`step` and `tasks`. Checkpoint-source paths and fingerprints do not form part
of the training-history evaluation schema. Proof bundles store any supplementary replay inputs separately from training metadata.

Keep the configuration, loss history, experiment settings and checkpoint
evaluations for reproduction. Temporary evaluation progress, locks and sync logs
do not form part of the curated histories. The training-dynamics notebook accepts
additional RNN and GRU histories through explicit run selections.
An omitted model contributes no curve. Figure exports go to
`analysis_outputs/paper_figures/training_dynamics/`.

[training_dynamics.ipynb](../notebooks/training_dynamics.ipynb)
reads the DFST add2 history above together with `DFST/mult2`,
`RNN/add2`, `RNN/mult2`, `GRU/add2` and
`GRU/mult2`. Each run contains ten checkpoint evaluations.
The run configurations use DEL's supported fields and select `hard_cases: all`
for new evaluations. Saved evaluation files record the evaluation settings in
`evaluation_config` and each task's `hard_cases`; changing a run configuration
does not change saved scores.

The five runs listed above contain RLG measurements searched up to the evaluation
cap. DEL's evaluator bounds the RLG search by the measured ULG. All recorded hard-case settings select all ordered repeated-digit pairs.
The saved length-search results retain the measured scores and tested lengths.

Use each run's `config.yaml` to train with the recorded training settings, or
choose a representative recipe from [configs/](../configs/README.md).
