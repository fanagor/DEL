# Paper Runs

The release contains 69 final runs spanning 36 arithmetic DFSTs, six universal DFSTs, and 27 Softmax DFSTs. To inspect a point on a plot, look up the run in the [figure selections](../paper/figures.yaml), then open the corresponding directory below.

## Inside a Run

Each directory follows the same layout. The YAML records the training settings, the checkpoint stores the learned parameters, and the JSON records the final evaluation.

```text
runs/<model>/<task>/<run_id>/
    config.yaml
    final_model_checkpoint.pt
    evaluation.json
```

For the DFST, RNN, and GRU training-dynamics comparison, open the [six binary-arithmetic histories](../analyses/README.md#training-dynamics). Each model/task directory contains the resolved configuration, loss log, and ten checkpoint evaluations.

The release removes tracking identities and disables tracking in the checkpoint configuration. Model parameters, optimizer state and random-generator state retain the values from the selected final training step. `evaluation.json` contains only the final-step result and records the sanitized checkpoint’s SHA-256 checksum. The plotting notebooks verify this checksum before reading a result.

`config.yaml` specifies resolved hidden dimensions and initial-state settings. Several null settings carry specific meanings. For example, `embed_dim: null` selects fixed one-hot embeddings, `data.max_steps: null` removes the execution limit, and `initialization.checkpoint: null` selects ordinary initialization. A numeric embedding dimension instead adds trainable embedding weights.

## Retrain or Load a Controller

To repeat a selected training run, pass the saved configuration to the trainer. This example retrains binary addition and selects an available device.

```sh
del train --config runs/DFST/add2/warmup_25000_steps_250000/config.yaml run.device=auto run.wandb.enabled=false run.name=add2-reproduction
```

The nine universal Softmax configurations explicitly select the scaling-A program used during training with `data.scaling_a_terminal_step: false`. Loading the original checkpoint recovers this setting from the expert fingerprint. See the [configuration guide](../configs/README.md#reproduce-a-selected-run).

To explore the learned controller directly, load the released checkpoint on the CPU. The loader returns the neural model, the matching expert, the configuration, and the saved checkpoint data.

```python
from differentiable_exact_learning.training import load_run
model, expert, config, saved = load_run(
    "runs/DFST/add2/warmup_25000_steps_250000", device="cpu"
)
```

For released arithmetic runs, the loader uses the full repdigit cross-pairs recorded in the checksum-matched final evaluation. Keep `evaluation.json` beside the checkpoint to recover that measurement protocol. For standalone weights, pass `evaluation.hard_cases=all` explicitly when evaluating. Re-evaluation writes `final_model_checkpoint.evaluation.json` by default.

Maintainers adding another final run can follow the [release-tool guide](../scripts/README.md#import-manifests) to validate the source experiment and export a checkpoint with public metadata.
