# Analysis Artifacts

The analyses answer two questions about a trained controller. First, can we recover a discrete program and prove that the program matches the expert? Second, how does length generalization develop during training? Start with an extraction case below to inspect a recovered program, or jump to [Training Dynamics](#training-dynamics) to follow a controller across checkpoints.

## Choose an Extraction Case

The table links each proof to the experiment that supplied the neural weights. Metric extraction uses fixed DFST weights and hidden-state trajectories. Annealing continues training a Softmax DFST before reading off discrete transitions. The [analysis catalog](../paper/analyses.yaml) records the source runs, case indices, input domains, proof files, and checker commands.

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

For a first example, try binary addition. The case guide takes you from the extracted transition table to the proof for operands of arbitrary finite length. The universal cases illustrate why the input domain matters. Metric extraction certifies the interpreter on nonempty binary inputs. The annealed controller’s proof also covers the empty input.

## Inside a Case

Each case keeps the neural checkpoint reference, the extracted machine, and the evidence supporting the proof together. The following layout shows where to find each artifact.

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

The configuration records the source checkpoint checksum and the method settings. In a metric-extraction case, `data/inputs.json` records the sampled inputs, `data/epsilon_attempts.json` records the thresholds tried, and `data/graph_stability.csv` tracks the graph as more trajectories enter the sample. For annealing, `data/training.jsonl` and `data/sharpness.csv` track optimization and concentration. The probability snapshots drive the notebook slider. The final annealed weights live in `checkpoints/annealed_model_checkpoint.pt`.

Use the JSON machine for computation, the CSV to look up a transition, and the PDF to read the table in full. All three formats share state numbers. The accompanying state maps record minimization and expert numbering. To display a case, select `CASE_INDEX` in the [metric-extraction notebook](../notebooks/metric_FST_extraction.ipynb) or the [annealing notebook](../notebooks/anneal_dfst_softmax_to_fst.ipynb). Set `RECOMPUTE = True` to repeat the procedure in `analysis_outputs/`. Enable `SAVE_INTERMEDIATES` to keep trajectories, prefix graphs, intermediate checkpoints, and diagnostic plots where applicable.

## Proof Conventions

Each proof begins with the expert, the extracted controller, and the valid inputs. The argument relates an expert state and grid configuration to a phase of the algorithm, then shows that every permitted transition preserves this relation. The finite certificate checks transitions. The phase invariants and induction establish agreement throughout executions of arbitrary length. The resulting theorem concerns the discrete controller. The [paper](https://arxiv.org/abs/2611.21486v1) gives the assumptions for extending the theorem to neural execution.

The proof bundles use common notation so readers can compare cases.

| Symbol or Convention | Meaning |
|---|---|
| $\mathcal E$, $\mathcal C$ | Expert and exported controller |
| $i$, $g_i$, $Q_i$ | Phase index, expert state, and related set of controller states |
| $i \xrightarrow{x} j$ | Permitted phase transition on read symbol $x$. The certificate compares actions and checks membership of the successor in $Q_j$. |
| $(q',w,M)$ | Successor state, written symbol, and movement |
| $\lambda$, PTO | Blank symbol and policy-trajectory observation. Each trajectory counts the final stopping action once. |
| $\kappa$ | Ternary code width in the universal interpreter. The configuration relation also tracks hidden rune symbols. |

Every `proof/` bundle includes a standalone LaTeX argument, `verify_equivalence.py`, a fingerprint manifest, and a saved report. Run the checker with `--quick` for file integrity and all finite proof obligations. Run the default command for supplementary executions as well. Decimal multiplication stores the replay instances in `replay_inputs.json`. Keep the bundled reference code, exported machines, certificate inputs, and checker together so the SHA-256 fingerprints continue to identify the same proof inputs.

To edit a proof, retain the domain and execution convention, use section and theorem references, and compile the standalone LaTeX file twice. The fingerprint manifest covers the computational dependencies and excludes the manuscript, generated reports, and training metadata. Follow the [repository writing conventions](../AGENTS.md) for definitions, narration, and headings. Maintainers importing a new case can follow the [import guide](../scripts/README.md#import-manifests).

## Training Dynamics

The six histories compare DFST, RNN, and GRU on binary addition and multiplication. Each directory under `training_dynamics/<model>/<task>/` contains one complete million-update history. The [training-dynamics notebook](../notebooks/training_dynamics.ipynb) reads the files directly without loading weights or contacting a tracking service.

| File | What to Inspect |
|---|---|
| `config.yaml` | Dataset, seed, controller dimensions, optimizer, warmup, and training budget |
| `metrics.jsonl` | Loss measurements over the course of training |
| `experiment.json` | Evaluation settings (`evaluation`), checkpoint interval (`evaluation_every`), step list (`evaluation_steps`), and the loss-log checksum (`metrics_sha256`) |
| `evaluations/step_*.json` | Ten checkpoint measurements at 100,000-update intervals, including ULG and RLG, tested lengths, sampling settings, scan, seed, and protocol |

RLG checks all four ordered binary repdigit pairs at each candidate length. To add a local history to a comparison, select the directory in the notebook and supply the configuration, complete loss log, experiment settings, and checkpoint evaluations. The notebook exports figures and tables to `analysis_outputs/paper_figures/training_dynamics/`. For training from scratch, use the saved run configuration or choose a [representative recipe](../configs/README.md).
